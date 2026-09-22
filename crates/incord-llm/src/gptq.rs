//! GPTQ: Hessian-aware rounding (roadmap Milestone 4).
//!
//! Plain rounding treats every weight on its own. GPTQ quantizes one input
//! row of `W: [k, n]` at a time and pushes that row's rounding error onto the
//! rows not yet quantized, weighted by how the layer's inputs correlate
//! (`H = Xᵀ·X` over calibration activations). The error that reaches the
//! layer output — not the weight error — is what gets minimised.
//!
//! Our layout helps: quant groups run along the output dimension `n`, inside
//! one input row, so every row is quantized as a unit with its own group
//! scales chosen after earlier rows' errors have been folded in. Rows can be
//! processed in any order, so act-order (largest `H` diagonal first) is free.
//!
//! Inverse-Hessian factor: GPTQ needs the upper Cholesky factor `U` of `H⁻¹`
//! (`H⁻¹ = Uᵀ·U`). With `J` the reversal permutation and `J·H·J = L·Lᵀ`,
//! `U = J·L⁻¹·J`, which costs one Cholesky and one triangular inverse.

use faer::linalg::matmul::matmul;
use faer::linalg::triangular_inverse::invert_lower_triangular;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};

/// Running `Σ xᵀx` over calibration rows for one layer input.
pub struct Hessian {
    pub k: usize,
    sum: Vec<f32>,
    pub samples: usize,
}

impl Hessian {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sum: vec![0.0; k * k],
            samples: 0,
        }
    }

    /// Accumulate `rows` input vectors (`x` is row-major `[rows, k]`).
    pub fn add(&mut self, x: &[f32], rows: usize) {
        assert_eq!(x.len(), rows * self.k);
        let xm = MatRef::from_row_major_slice(x, rows, self.k);
        let hm = MatMut::from_row_major_slice_mut(&mut self.sum, self.k, self.k);
        matmul(hm, Accum::Add, xm.transpose(), xm, 1.0f32, Par::rayon(0));
        self.samples += rows;
    }

    /// Diagonal entry `i` (for tests and diagnostics).
    pub fn diag(&self, i: usize) -> f32 {
        self.sum[i * self.k + i]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GptqConfig {
    /// Dampening added to the diagonal, as a fraction of its mean.
    pub damp: f64,
    /// Rows per lazy-update block.
    pub block: usize,
    /// Quantize rows with the largest Hessian diagonal first.
    pub act_order: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            damp: 0.01,
            block: 128,
            act_order: true,
        }
    }
}

/// Quantizes one input row (`n` values) of a weight matrix.
pub trait RowQuantizer: Sync {
    type Code: Send;
    /// Quantize `row`, write its dequantized values to `deq`, return the
    /// stored representation.
    fn quantize(&self, row: &[f32], deq: &mut [f32]) -> Self::Code;
}

/// Result of [`gptq`]: one code per input row, in the original row order.
pub struct GptqResult<C> {
    pub rows: Vec<C>,
    /// Damping fraction finally used (raised if the Hessian was not
    /// positive definite at the requested value).
    pub damp_used: f64,
}

/// `U` with `H⁻¹ = Uᵀ·U` (upper triangular), for a positive definite `H`.
/// Returns `None` if the Cholesky factorisation fails.
fn inverse_upper_cholesky(h: &Mat<f64>) -> Option<Mat<f64>> {
    let k = h.nrows();
    let rev = Mat::<f64>::from_fn(k, k, |i, j| h[(k - 1 - i, k - 1 - j)]);
    let llt = rev.llt(Side::Lower).ok()?;
    let mut linv = Mat::<f64>::zeros(k, k);
    invert_lower_triangular(linv.as_mut(), llt.L(), Par::rayon(0));
    Some(Mat::<f64>::from_fn(k, k, |i, j| linv[(k - 1 - i, k - 1 - j)]))
}

/// Run GPTQ on `w` (row-major `[k, n]`, input rows) with Hessian `h`.
pub fn gptq<Q: RowQuantizer>(
    w: &[f32],
    k: usize,
    n: usize,
    h: &Hessian,
    cfg: GptqConfig,
    q: &Q,
) -> Result<GptqResult<Q::Code>, String> {
    assert_eq!(w.len(), k * n);
    assert_eq!(h.k, k, "Hessian size {} != weight rows {k}", h.k);

    // Hessian in f64, dead inputs (never active) pinned so they stay
    // invertible; their weight rows cannot matter and are zeroed.
    let scale = if h.samples > 0 { 2.0 / h.samples as f64 } else { 1.0 };
    let mut hm = Mat::<f64>::from_fn(k, k, |i, j| h.sum[i * k + j] as f64 * scale);
    let mut w = w.to_vec();
    for i in 0..k {
        if hm[(i, i)] == 0.0 {
            hm[(i, i)] = 1.0;
            w[i * n..(i + 1) * n].fill(0.0);
        }
    }

    // Processing order.
    let mut perm: Vec<usize> = (0..k).collect();
    if cfg.act_order {
        perm.sort_by(|&a, &b| hm[(b, b)].total_cmp(&hm[(a, a)]).then(a.cmp(&b)));
    }
    let hp = Mat::<f64>::from_fn(k, k, |i, j| hm[(perm[i], perm[j])]);
    let mut wp = vec![0.0f32; k * n];
    for (dst, &src) in wp.chunks_mut(n).zip(&perm) {
        dst.copy_from_slice(&w[src * n..(src + 1) * n]);
    }

    // Dampen; raise the dampening if H is numerically not positive definite.
    let mean_diag = (0..k).map(|i| hp[(i, i)]).sum::<f64>() / k as f64;
    let mut damp = cfg.damp;
    let u = loop {
        let hd = Mat::<f64>::from_fn(k, k, |i, j| {
            hp[(i, j)] + if i == j { damp * mean_diag } else { 0.0 }
        });
        if let Some(u) = inverse_upper_cholesky(&hd) {
            break u;
        }
        if damp >= 1.0 {
            return Err("Hessian is not positive definite even with damp 1.0".into());
        }
        damp *= 10.0;
    };
    let uf = Mat::<f32>::from_fn(k, k, |i, j| u[(i, j)] as f32);

    let mut codes: Vec<Option<Q::Code>> = (0..k).map(|_| None).collect();
    let mut deq = vec![0.0f32; n];
    let block = cfg.block.max(1);
    for i1 in (0..k).step_by(block) {
        let i2 = (i1 + block).min(k);
        let count = i2 - i1;
        let mut err = vec![0.0f32; count * n];
        {
            let wb = &mut wp[i1 * n..i2 * n];
            for r in 0..count {
                let i = i1 + r;
                let (done, rest) = wb.split_at_mut((r + 1) * n);
                let row = &done[r * n..];
                codes[perm[i]] = Some(q.quantize(row, &mut deq));
                let d = uf[(i, i)];
                let e = &mut err[r * n..(r + 1) * n];
                for c in 0..n {
                    e[c] = (row[c] - deq[c]) / d;
                }
                // Fold the error into the remaining rows of this block.
                for (r2, later) in rest.chunks_mut(n).enumerate() {
                    let f = uf[(i, i + 1 + r2)];
                    for c in 0..n {
                        later[c] -= e[c] * f;
                    }
                }
            }
        }
        // Lazy update of every row after the block: W[i2..] -= U[blk, i2..]ᵀ · Err.
        if i2 < k {
            let dst = MatMut::from_row_major_slice_mut(&mut wp[i2 * n..], k - i2, n);
            let lhs = uf.as_ref().submatrix(i1, i2, count, k - i2).transpose();
            let rhs = MatRef::from_row_major_slice(&err, count, n);
            matmul(dst, Accum::Add, lhs, rhs, -1.0f32, Par::rayon(0));
        }
    }

    Ok(GptqResult {
        rows: codes.into_iter().map(|c| c.expect("every row quantized")).collect(),
        damp_used: damp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    /// Correlated calibration inputs (a few strong shared directions), the
    /// situation where GPTQ beats plain rounding.
    fn inputs(rows: usize, k: usize) -> Vec<f32> {
        let basis = data(4 * k, 99);
        let coef = data(rows * 4, 98);
        let noise = data(rows * k, 97);
        (0..rows * k)
            .map(|idx| {
                let (r, i) = (idx / k, idx % k);
                let mut v = 0.05 * noise[idx];
                for b in 0..4 {
                    v += coef[r * 4 + b] * basis[b * k + i] * 3.0;
                }
                v
            })
            .collect()
    }

    /// Round-to-nearest on a fixed grid, so GPTQ's effect is isolated.
    struct Grid(f32);
    impl RowQuantizer for Grid {
        type Code = Vec<f32>;
        fn quantize(&self, row: &[f32], deq: &mut [f32]) -> Vec<f32> {
            for (d, &x) in deq.iter_mut().zip(row) {
                *d = (x / self.0).round() * self.0;
            }
            deq.to_vec()
        }
    }

    fn out_err(x: &[f32], rows: usize, k: usize, n: usize, w: &[f32], q: &[f32]) -> f64 {
        let mut s = 0.0;
        for r in 0..rows {
            for j in 0..n {
                let d: f64 = (0..k).map(|i| x[r * k + i] as f64 * (w[i * n + j] - q[i * n + j]) as f64).sum();
                s += d * d;
            }
        }
        s
    }

    #[test]
    fn inverse_upper_cholesky_inverts() {
        let k = 24;
        let x = inputs(200, k);
        let mut h = Hessian::new(k);
        h.add(&x, 200);
        let hm = Mat::<f64>::from_fn(k, k, |i, j| h.sum[i * k + j] as f64 + if i == j { 1e-3 } else { 0.0 });
        let u = inverse_upper_cholesky(&hm).expect("positive definite");
        for i in 0..k {
            for j in 0..i {
                assert_eq!(u[(i, j)], 0.0, "U must be upper triangular");
            }
        }
        // H · (Uᵀ·U) = I
        let hinv = u.transpose() * &u;
        let prod = &hm * &hinv;
        for i in 0..k {
            for j in 0..k {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod[(i, j)] - want).abs() < 1e-6, "({i},{j}) = {}", prod[(i, j)]);
            }
        }
    }

    #[test]
    fn hessian_accumulates_xtx() {
        let (rows, k) = (7, 5);
        let x = data(rows * k, 3);
        let mut h = Hessian::new(k);
        h.add(&x[..3 * k], 3);
        h.add(&x[3 * k..], 4);
        for i in 0..k {
            for j in 0..k {
                let want: f32 = (0..rows).map(|r| x[r * k + i] * x[r * k + j]).sum();
                assert!((h.sum[i * k + j] - want).abs() < 1e-4);
            }
        }
        assert_eq!(h.samples, 7);
    }

    /// GPTQ must cut the layer-output error well below plain rounding on the
    /// same grid, and return one code per row in the original order.
    #[test]
    fn gptq_beats_round_to_nearest_on_output_error() {
        let (rows, k, n) = (256, 48, 20);
        let x = inputs(rows, k);
        let w = data(k * n, 5);
        let mut h = Hessian::new(k);
        h.add(&x, rows);
        let grid = Grid(0.25);

        let rtn: Vec<f32> = w.iter().map(|v| (v / 0.25).round() * 0.25).collect();
        let res = gptq(&w, k, n, &h, GptqConfig { block: 16, ..Default::default() }, &grid).unwrap();
        assert_eq!(res.rows.len(), k);
        let q: Vec<f32> = res.rows.concat();
        for v in &q {
            assert!(((v / 0.25).round() * 0.25 - v).abs() < 1e-6, "off grid: {v}");
        }
        let (e_rtn, e_gptq) = (out_err(&x, rows, k, n, &w, &rtn), out_err(&x, rows, k, n, &w, &q));
        assert!(e_gptq < 0.5 * e_rtn, "gptq {e_gptq} vs rtn {e_rtn}");
    }

    /// Block size and act-order change the path, not the validity.
    #[test]
    fn gptq_variants_all_beat_rtn() {
        let (rows, k, n) = (256, 40, 12);
        let x = inputs(rows, k);
        let w = data(k * n, 6);
        let mut h = Hessian::new(k);
        h.add(&x, rows);
        let rtn: Vec<f32> = w.iter().map(|v| (v / 0.25).round() * 0.25).collect();
        let e_rtn = out_err(&x, rows, k, n, &w, &rtn);
        for (block, act_order) in [(1, false), (7, true), (128, true), (128, false)] {
            let cfg = GptqConfig { block, act_order, ..Default::default() };
            let q = gptq(&w, k, n, &h, cfg, &Grid(0.25)).unwrap().rows.concat();
            let e = out_err(&x, rows, k, n, &w, &q);
            assert!(e < e_rtn, "block {block} act_order {act_order}: {e} vs {e_rtn}");
        }
    }

    #[test]
    fn dead_inputs_get_zero_rows() {
        let (rows, k, n) = (64, 8, 4);
        let mut x = inputs(rows, k);
        for r in 0..rows {
            x[r * k + 3] = 0.0; // input 3 never fires
        }
        let w = data(k * n, 7);
        let mut h = Hessian::new(k);
        h.add(&x, rows);
        let q = gptq(&w, k, n, &h, GptqConfig::default(), &Grid(0.25)).unwrap().rows;
        assert!(q[3].iter().all(|&v| v == 0.0));
    }
}
