//! Randomized Hadamard rotation for quantization (roadmap Milestone 3).
//!
//! A linear layer computes `y = x · W` with `W: [k, n]`. For any orthogonal
//! `R`, `y = (x · R) · (Rᵀ · W)`. Storing `W' = Rᵀ · W` and feeding the layer
//! `x' = x · R` gives the same output, but `W'` has its outliers spread
//! across the whole input dimension, so it quantizes with far less error.
//!
//! `R = diag(s) · H / √b`: random ±1 signs `s`, then a Walsh–Hadamard
//! transform over blocks of `b` (the largest power of two dividing `k`, so
//! 2048 → one block, 6144 → three blocks of 2048). `H` is symmetric, so the
//! same transform [`forward`] is used for:
//!
//! - activations: `x' = forward(x)` (a row vector times `R`),
//! - weight columns: `W'[:, j] = forward(W[:, j])` (that is `Rᵀ · W`),
//! - output-head rows (stored `[vocab, hidden]`): `row' = forward(row)`.
//!
//! [`inverse`] undoes [`forward`]. The signs depend only on the length and a
//! seed. The seed is stored in the `.incord` header (format v4) and carried by
//! every `Weight::Rotated`, so a file always decodes with the seed it was
//! written with; v3 files predate the header field and used [`DEFAULT_SEED`].

/// Seed used by the converter unless told otherwise, and implied by v3 files.
pub const DEFAULT_SEED: u64 = 0x1C0B_D5A1_7E57_2026;

/// Identifier of this rotation scheme in the file header (sign flip +
/// blockwise normalised Walsh–Hadamard, block = largest power of two
/// dividing the length, signs from the xorshift below).
pub const SCHEME_BLOCK_HADAMARD: u8 = 1;

/// Hadamard block length for a vector of length `len`: the largest power of
/// two dividing it.
pub fn block_len(len: usize) -> usize {
    assert!(len > 0, "cannot rotate an empty vector");
    1 << len.trailing_zeros()
}

/// The ±1 sign of every coordinate for vectors of length `len`.
pub fn signs(len: usize, seed: u64) -> Vec<f32> {
    // Must stay bit-identical: v3 files decode with DEFAULT_SEED through this.
    // (A zero state only yields all-positive signs — still orthogonal.)
    let mut s = seed ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Unnormalised in-place fast Walsh–Hadamard transform of one block
/// (`x.len()` must be a power of two).
fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Blockwise normalised Hadamard transform in place.
fn hadamard_blocks(x: &mut [f32]) {
    let b = block_len(x.len());
    let norm = 1.0 / (b as f32).sqrt();
    for blk in x.chunks_mut(b) {
        fwht(blk);
        for v in blk.iter_mut() {
            *v *= norm;
        }
    }
}

/// `x ← x · R`: sign flip, then blockwise normalised Hadamard.
pub fn forward(x: &mut [f32], seed: u64) {
    let s = signs(x.len(), seed);
    for (v, si) in x.iter_mut().zip(s) {
        *v *= si;
    }
    hadamard_blocks(x);
}

/// Inverse of [`forward`]: blockwise normalised Hadamard, then sign flip.
pub fn inverse(x: &mut [f32], seed: u64) {
    hadamard_blocks(x);
    let s = signs(x.len(), seed);
    for (v, si) in x.iter_mut().zip(s) {
        *v *= si;
    }
}

/// `W ← Rᵀ · W` for a row-major `[k, n]` matrix: [`forward`] applied down
/// every column. Works on whole rows (butterflies between rows), so it is
/// cache-friendly and vectorises over `n`.
pub fn rotate_columns(w: &mut [f32], k: usize, n: usize, seed: u64) {
    assert_eq!(w.len(), k * n);
    let s = signs(k, seed);
    for (row, &si) in w.chunks_mut(n).zip(&s) {
        if si < 0.0 {
            row.iter_mut().for_each(|v| *v = -*v);
        }
    }
    let b = block_len(k);
    let norm = 1.0 / (b as f32).sqrt();
    for blk in w.chunks_mut(b * n) {
        let mut h = 1;
        while h < b {
            for i in (0..b).step_by(2 * h) {
                for j in i..i + h {
                    let (top, bottom) = blk.split_at_mut((j + h) * n);
                    let (a, c) = (&mut top[j * n..(j + 1) * n], &mut bottom[..n]);
                    for (x, y) in a.iter_mut().zip(c.iter_mut()) {
                        let (p, q) = (*x, *y);
                        *x = p + q;
                        *y = p - q;
                    }
                }
            }
            h *= 2;
        }
        blk.iter_mut().for_each(|v| *v *= norm);
    }
}

/// [`forward`] applied to every row of a row-major `[rows, len]` matrix
/// (used for the output head, stored `[vocab, hidden]`).
pub fn rotate_rows(w: &mut [f32], len: usize, seed: u64) {
    assert!(w.len().is_multiple_of(len));
    let s = signs(len, seed);
    let b = block_len(len);
    let norm = 1.0 / (b as f32).sqrt();
    for row in w.chunks_mut(len) {
        for (v, si) in row.iter_mut().zip(&s) {
            *v *= si;
        }
        for blk in row.chunks_mut(b) {
            fwht(blk);
            blk.iter_mut().for_each(|v| *v *= norm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "i={i}: {x} vs {y}");
        }
    }

    #[test]
    fn block_len_matches_model_dims() {
        assert_eq!(block_len(2048), 2048);
        assert_eq!(block_len(6144), 2048);
        assert_eq!(block_len(256), 256);
    }

    #[test]
    fn inverse_undoes_forward() {
        for len in [8, 96, 2048, 6144] {
            let x = data(len, 1);
            let mut y = x.clone();
            forward(&mut y, DEFAULT_SEED);
            inverse(&mut y, DEFAULT_SEED);
            close(&y, &x, 1e-5);
        }
    }

    /// Any seed gives a valid (invertible) rotation, and different seeds
    /// give different sign patterns.
    #[test]
    fn seeds_are_independent_and_invertible() {
        assert_ne!(signs(2048, DEFAULT_SEED), signs(2048, 7));
        for seed in [0, 7, u64::MAX] {
            let x = data(2048, 9);
            let mut y = x.clone();
            forward(&mut y, seed);
            inverse(&mut y, seed);
            close(&y, &x, 1e-5);
        }
    }

    #[test]
    fn forward_preserves_norm() {
        let x = data(6144, 2);
        let mut y = x.clone();
        forward(&mut y, DEFAULT_SEED);
        let n = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
        assert!((n(&x) - n(&y)).abs() < 1e-3 * n(&x));
    }

    /// The whole point: (x·R)·(Rᵀ·W) == x·W.
    #[test]
    fn rotated_layer_gives_same_output() {
        let (k, n) = (96, 40);
        let w = data(k * n, 3);
        let x = data(k, 4);
        let matvec = |x: &[f32], w: &[f32]| -> Vec<f32> {
            (0..n).map(|j| (0..k).map(|i| x[i] * w[i * n + j]).sum()).collect()
        };
        let expect = matvec(&x, &w);
        let mut wr = w.clone();
        rotate_columns(&mut wr, k, n, DEFAULT_SEED);
        let mut xr = x.clone();
        forward(&mut xr, DEFAULT_SEED);
        close(&matvec(&xr, &wr), &expect, 1e-4);
    }

    /// Output head: logits = x · rowᵀ is preserved with rotated rows.
    #[test]
    fn rotated_head_gives_same_logits() {
        let (vocab, hidden) = (30, 64);
        let w = data(vocab * hidden, 5);
        let x = data(hidden, 6);
        let logits = |x: &[f32], w: &[f32]| -> Vec<f32> {
            w.chunks(hidden).map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
        };
        let expect = logits(&x, &w);
        let mut wr = w.clone();
        rotate_rows(&mut wr, hidden, DEFAULT_SEED);
        let mut xr = x.clone();
        forward(&mut xr, DEFAULT_SEED);
        close(&logits(&xr, &wr), &expect, 1e-4);
    }

    /// rotate_columns must equal forward() applied to each column.
    #[test]
    fn column_rotation_matches_vector_transform() {
        let (k, n) = (6144 / 64, 3); // 96 rows: 3 blocks of 32
        let w = data(k * n, 7);
        let mut wr = w.clone();
        rotate_columns(&mut wr, k, n, DEFAULT_SEED);
        for j in 0..n {
            let mut col: Vec<f32> = (0..k).map(|i| w[i * n + j]).collect();
            forward(&mut col, DEFAULT_SEED);
            let got: Vec<f32> = (0..k).map(|i| wr[i * n + j]).collect();
            close(&got, &col, 1e-5);
        }
    }

    /// Rotation spreads an outlier: max |value| drops sharply.
    #[test]
    fn rotation_spreads_outliers() {
        let mut x = vec![0.01f32; 2048];
        x[17] = 50.0;
        let before = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        forward(&mut x, DEFAULT_SEED);
        let after = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(after < before / 20.0, "max {before} -> {after}");
    }
}
