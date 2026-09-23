//! E8-lattice vector quantization (roadmap Milestone 8).
//!
//! Scalar 2-bit rounding puts every weight on a 4-level grid. Quantizing 8
//! weights together onto points of the E8 lattice packs the codewords far
//! more efficiently (E8 is the densest packing in 8 dimensions), so at the
//! same number of bits the reconstruction error is lower — the idea behind
//! QuIP#. After a Hadamard rotation the weights are close to Gaussian, so a
//! single fixed codebook serves every layer and every model.
//!
//! Codebook: every E8 point with squared norm ≤ [`MAX_NORM2`], i.e. the
//! shells 0, 2, 4, 6, 8, 10 — 56,881 points, addressed by a 16-bit code, so
//! 8 weights cost 16 bits (2 bits/weight, before the per-group scale).
//! E8 = D8 ∪ (D8 + ½), D8 = integer vectors with an even coordinate sum.
//! Points are stored doubled (all coordinates are then integers in -6..=6).
//!
//! [`encode`] returns the exact nearest codeword: the nearest lattice point
//! (Conway & Sloane) when it lies in the ball, otherwise a norm-pruned search
//! over the codebook (rare: only outlier vectors land outside the ball).
//! [`encode_fast`] replaces that search by a radial-shrink heuristic whose
//! error is an upper bound of the exact one; it is used to rank candidate
//! scales, where speed matters and a small overestimate is harmless.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Vector length.
pub const DIM: usize = 8;
/// Largest squared norm (in lattice units) kept in the codebook.
pub const MAX_NORM2: i32 = 10;
/// Codebook size: E8 points with norm² ≤ 10.
pub const CODEBOOK_LEN: usize = 56_881;

/// The codebook: doubled coordinates, sorted by (norm², coordinates).
pub struct Codebook {
    /// `CODEBOOK_LEN` points, each `DIM` doubled coordinates.
    pub points: Vec<[i8; DIM]>,
    index: HashMap<[i8; DIM], u16>,
}

impl Codebook {
    fn build() -> Self {
        let mut points: Vec<[i8; DIM]> = Vec::with_capacity(CODEBOOK_LEN);
        // Doubled coordinates: integer points are even in -6..=6, half-integer
        // points odd in -5..=5. |p|² ≤ 10 ⇔ Σ c² ≤ 40 in doubled units.
        let mut c = [0i8; DIM];
        fn rec(pos: usize, c: &mut [i8; DIM], sumsq: i32, parity_odd: bool, out: &mut Vec<[i8; DIM]>) {
            if sumsq > 4 * MAX_NORM2 {
                return;
            }
            if pos == DIM {
                // D8 (even doubled coords): Σ(c/2) must be even ⇔ Σc ≡ 0 mod 4.
                // D8 + ½ (odd doubled coords): Σ(c/2 - ½) even ⇔ Σc ≡ 0 mod 4 as well.
                let s: i32 = c.iter().map(|&v| v as i32).sum();
                if s.rem_euclid(4) == 0 {
                    out.push(*c);
                }
                return;
            }
            let (lo, hi) = if parity_odd { (-5, 5) } else { (-6, 6) };
            let mut v = lo;
            while v <= hi {
                c[pos] = v as i8;
                rec(pos + 1, c, sumsq + v * v, parity_odd, out);
                v += 2;
            }
        }
        rec(0, &mut c, 0, false, &mut points);
        rec(0, &mut c, 0, true, &mut points);
        points.sort_by_key(|p| (p.iter().map(|&v| v as i32 * v as i32).sum::<i32>(), *p));
        assert_eq!(points.len(), CODEBOOK_LEN, "E8 ball enumeration");
        let index = points.iter().enumerate().map(|(i, p)| (*p, i as u16)).collect();
        Self { points, index }
    }

    /// Code of a doubled-coordinate point, if it is in the codebook.
    pub fn code_of(&self, p: &[i8; DIM]) -> Option<u16> {
        self.index.get(p).copied()
    }
}

/// The shared codebook (built once, ~0.5 MB).
pub fn codebook() -> &'static Codebook {
    static CB: OnceLock<Codebook> = OnceLock::new();
    CB.get_or_init(Codebook::build)
}

/// Nearest point of D8 (or of D8 shifted by `shift`) to `x`, in lattice
/// units: round every coordinate; if the coordinate sum is odd, re-round
/// the coordinate with the largest rounding error the other way.
fn nearest_d8(x: &[f32; DIM], shift: f32) -> [f32; DIM] {
    let mut p = [0f32; DIM];
    let mut worst = 0usize;
    let mut worst_err = -1.0f32;
    let mut sum = 0i64;
    for i in 0..DIM {
        let y = x[i] - shift;
        let r = y.round();
        p[i] = r;
        sum += r as i64;
        let e = (y - r).abs();
        if e > worst_err {
            worst_err = e;
            worst = i;
        }
    }
    if sum.rem_euclid(2) != 0 {
        let y = x[worst] - shift;
        p[worst] += if y > p[worst] { 1.0 } else { -1.0 };
    }
    for v in p.iter_mut() {
        *v += shift;
    }
    p
}

fn dist2(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Nearest E8 point to `x` (lattice units).
pub fn nearest_e8(x: &[f32; DIM]) -> [f32; DIM] {
    let a = nearest_d8(x, 0.0);
    let b = nearest_d8(x, 0.5);
    if dist2(x, &a) <= dist2(x, &b) { a } else { b }
}

fn norm2(p: &[f32; DIM]) -> f32 {
    p.iter().map(|v| v * v).sum()
}

fn doubled(p: &[f32; DIM]) -> [i8; DIM] {
    let mut c = [0i8; DIM];
    for (d, v) in c.iter_mut().zip(p) {
        *d = (v * 2.0).round() as i8;
    }
    c
}

/// Nearest lattice point if it is a codeword.
fn encode_in_ball(x: &[f32; DIM]) -> Option<(u16, f32)> {
    let p = nearest_e8(x);
    if norm2(&p) <= MAX_NORM2 as f32 + 1e-4 {
        let code = codebook().code_of(&doubled(&p)).expect("in-ball E8 point is in the codebook");
        Some((code, dist2(x, &p)))
    } else {
        None
    }
}

/// Code of the codebook point nearest to `x` (lattice units), and its
/// squared distance. Exact.
pub fn encode(x: &[f32; DIM]) -> (u16, f32) {
    if let Some(hit) = encode_in_ball(x) {
        return hit;
    }
    // Nearest lattice point is outside the ball: search the codebook.
    // |x - p| ≥ | |x| - |p| |, so a codeword whose norm is too far from |x|
    // cannot beat the best found so far (codebook is sorted by norm).
    let cb = codebook();
    let nx = norm2(x).sqrt();
    let (mut best_code, mut best_d) = encode_fast(x);
    for (i, c) in cb.points.iter().enumerate().rev() {
        let np = (c.iter().map(|&v| v as i32 * v as i32).sum::<i32>() as f32 / 4.0).sqrt();
        let gap = nx - np;
        if gap > 0.0 && gap * gap >= best_d {
            break; // every remaining codeword is even shorter
        }
        let mut d = 0.0f32;
        for (xv, &cv) in x.iter().zip(c) {
            let e = xv - cv as f32 * 0.5;
            d += e * e;
        }
        if d < best_d {
            best_d = d;
            best_code = i as u16;
        }
    }
    (best_code, best_d)
}

/// Fast encode: exact inside the ball; outside it, the best in-ball point
/// found by shrinking the input radially. Its distance is ≥ the exact one.
pub fn encode_fast(x: &[f32; DIM]) -> (u16, f32) {
    let cb = codebook();
    if let Some(hit) = encode_in_ball(x) {
        return hit;
    }
    let mut best: Option<(u16, f32)> = None;
    let mut f = 0.97f32;
    for _ in 0..40 {
        let mut y = *x;
        y.iter_mut().for_each(|v| *v *= f);
        let q = nearest_e8(&y);
        if norm2(&q) <= MAX_NORM2 as f32 + 1e-4 {
            let d = dist2(x, &q);
            let code = cb.code_of(&doubled(&q)).expect("in-ball E8 point is in the codebook");
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((code, d));
            }
        }
        f *= 0.97;
    }
    best.unwrap_or_else(|| (cb.code_of(&[0; DIM]).expect("origin is a codeword"), norm2(x)))
}

/// Codebook point of `code`, in lattice units.
pub fn decode(code: u16) -> [f32; DIM] {
    let c = codebook().points[code as usize];
    let mut p = [0f32; DIM];
    for (d, v) in p.iter_mut().zip(c) {
        *d = v as f32 * 0.5;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_has_the_known_shell_sizes() {
        let cb = codebook();
        assert_eq!(cb.points.len(), CODEBOOK_LEN);
        let mut shells = std::collections::BTreeMap::new();
        for p in &cb.points {
            let n2: i32 = p.iter().map(|&v| v as i32 * v as i32).sum::<i32>() / 4;
            *shells.entry(n2).or_insert(0) += 1;
        }
        // Theta series of E8: 1 + 240q + 2160q² + 6720q³ + 17520q⁴ + 30240q⁵.
        let want: Vec<(i32, i32)> =
            vec![(0, 1), (2, 240), (4, 2160), (6, 6720), (8, 17520), (10, 30240)];
        assert_eq!(shells.into_iter().collect::<Vec<_>>(), want);
    }

    #[test]
    fn every_codeword_decodes_and_reencodes_to_itself() {
        for code in (0..CODEBOOK_LEN as u16).step_by(37) {
            let p = decode(code);
            let (back, d) = encode(&p);
            assert_eq!(back, code);
            assert!(d < 1e-9);
        }
    }

    /// `encode` is exact for inputs inside and outside the ball (checked by
    /// brute force); `encode_fast` never reports less error than exact.
    #[test]
    fn encode_matches_brute_force() {
        let mut s = 0x1234_5678_9ABC_DEF1u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 3.0 - 1.5) as f32
        };
        for _ in 0..200 {
            let x: [f32; DIM] = std::array::from_fn(|_| rnd());
            let (code, d) = encode(&x);
            let brute = (0..CODEBOOK_LEN as u16)
                .map(|c| dist2(&x, &decode(c)))
                .fold(f32::INFINITY, f32::min);
            assert!((d - brute).abs() < 1e-5, "encode {d} vs brute {brute} (code {code})");
            assert!((dist2(&x, &decode(code)) - d).abs() < 1e-5, "reported distance matches code");
            let (_, fast) = encode_fast(&x);
            assert!(fast >= brute - 1e-5, "fast {fast} below exact {brute}");
        }
    }

    /// Far-out inputs still map to a valid in-ball codeword.
    #[test]
    fn outliers_stay_in_the_codebook() {
        let x = [9.0f32, -7.5, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (code, _) = encode(&x);
        let p = decode(code);
        assert!(norm2(&p) <= MAX_NORM2 as f32 + 1e-6);
    }
}
