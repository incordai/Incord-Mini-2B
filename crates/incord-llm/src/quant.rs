//! Weight quantizers used by the converter (src/convert.rs).
//!
//! Both formats are symmetric, groups of [`GROUP`] values along the output
//! dimension of a `[k, n]` row-major matrix, one f16 scale per group:
//!
//! - Q4: value = scale * (q - 8), q in 0..=15, two values per byte
//!   (low nibble first). Read by `gemv::Q4Weight` kernels.
//! - Q8: value = scale * q, q in -128..=127. Read by `gemv::Q8Weight` kernels.
//!
//! [`Rounding::Rtn`] is the original round-to-nearest with scale
//! `absmax / qmax`. [`Rounding::Mse`] searches the scale that minimises the
//! group's squared error, evaluated with the f16-rounded scale exactly as it
//! is stored. The RTN result is always one of the candidates, so MSE rounding
//! is never worse than RTN on any group. The stored layout is identical, so
//! no kernel changes.

use half::f16;
use rayon::prelude::*;

/// Values per quantization group.
pub const GROUP: usize = 32;

/// How a group's scale is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Round-to-nearest with scale = absmax / qmax (original behaviour).
    Rtn,
    /// Squared-error-minimising scale search (never worse than `Rtn`).
    Mse,
}

impl std::str::FromStr for Rounding {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "rtn" => Ok(Rounding::Rtn),
            "mse" => Ok(Rounding::Mse),
            _ => Err(format!("unknown rounding '{s}' (expected rtn or mse)")),
        }
    }
}

/// Integer range and scale search grid of one format.
struct Format {
    qmin: i32,
    qmax: i32,
    /// Scale candidates are absmax / d for d in [d_lo, d_hi] step d_step.
    d_lo: f32,
    d_hi: f32,
    d_step: f32,
}

const Q4: Format = Format {
    qmin: -8,
    qmax: 7,
    d_lo: 6.0,
    d_hi: 9.0,
    d_step: 1.0 / 16.0,
};

const Q8: Format = Format {
    qmin: -128,
    qmax: 127,
    d_lo: 126.0,
    d_hi: 129.0,
    d_step: 0.25,
};

const Q3: Format = Format {
    qmin: -4,
    qmax: 3,
    d_lo: 2.5,
    d_hi: 5.0,
    d_step: 1.0 / 16.0,
};

const Q2: Format = Format {
    qmin: -2,
    qmax: 1,
    d_lo: 0.75,
    d_hi: 2.75,
    d_step: 1.0 / 16.0,
};

/// Sub-4-bit formats (roadmap Milestone 6). Groups of [`GROUP`] along `n`,
/// one f16 scale per group, value = scale * q:
///
/// - Q3: q in -4..=3. Per group 12 bytes: 8 bytes of low 2 bits (four
///   values per byte, value `i` at bits `2*(i%4)` of byte `i/4`) then 4 bytes
///   of high bits (value `i` at bit `i%8` of byte `8 + i/8`). 3.5 bits/weight.
/// - Q2: q in -2..=1. Per group 8 bytes, four values per byte as above.
///   2.5 bits/weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowBits {
    Q3,
    Q2,
}

impl LowBits {
    /// Packed bytes per group of [`GROUP`] values.
    pub fn group_bytes(self) -> usize {
        match self {
            LowBits::Q3 => 12,
            LowBits::Q2 => 8,
        }
    }

    fn format(self) -> &'static Format {
        match self {
            LowBits::Q3 => &Q3,
            LowBits::Q2 => &Q2,
        }
    }

    fn offset(self) -> i32 {
        -self.format().qmin
    }
}

/// Pack one group's ints (`q` in the format's range) into `dst`
/// (`bits.group_bytes()` bytes).
fn pack_low(bits: LowBits, q: &[i32], dst: &mut [u8]) {
    dst.fill(0);
    let off = bits.offset();
    for (i, &qi) in q.iter().enumerate() {
        let u = (qi + off) as u8;
        dst[i / 4] |= (u & 0b11) << (2 * (i % 4));
        if bits == LowBits::Q3 {
            dst[8 + i / 8] |= ((u >> 2) & 1) << (i % 8);
        }
    }
}

/// Dequantize one packed group (`GROUP` values) with scale `s` into `dst`.
pub fn unpack_low(bits: LowBits, src: &[u8], s: f32, dst: &mut [f32]) {
    let off = bits.offset();
    for (i, d) in dst.iter_mut().enumerate() {
        let mut u = ((src[i / 4] >> (2 * (i % 4))) & 0b11) as i32;
        if bits == LowBits::Q3 {
            u |= (((src[8 + i / 8] >> (i % 8)) & 1) as i32) << 2;
        }
        *d = (u - off) as f32 * s;
    }
}

/// Quantize one row (`row.len()` a multiple of [`GROUP`]) to Q3 / Q2.
pub fn quantize_low_row(
    row: &[f32],
    bits: LowBits,
    rounding: Rounding,
    prow: &mut [u8],
    srow: &mut [f16],
) {
    let gb = bits.group_bytes();
    let mut q = [0i32; GROUP];
    for (g, (grp, s)) in row.chunks(GROUP).zip(srow.iter_mut()).enumerate() {
        *s = quantize_group(grp, bits.format(), rounding, &mut q);
        pack_low(bits, &q, &mut prow[g * gb..(g + 1) * gb]);
    }
}

/// Quantize a `[k, n]` matrix to Q3 / Q2. Returns (packed bytes, scales).
pub fn quantize_low(
    data: &[f32],
    k: usize,
    n: usize,
    bits: LowBits,
    rounding: Rounding,
) -> (Vec<u8>, Vec<f16>) {
    check_shape(data, k, n);
    let gpr = n / GROUP;
    let row_bytes = gpr * bits.group_bytes();
    let mut packed = vec![0u8; k * row_bytes];
    let mut scales = vec![f16::from_f32(0.0); k * gpr];
    packed
        .par_chunks_mut(row_bytes)
        .zip(scales.par_chunks_mut(gpr))
        .zip(data.par_chunks(n))
        .for_each(|((prow, srow), row)| quantize_low_row(row, bits, rounding, prow, srow));
    (packed, scales)
}

/// Dequantize Q3 / Q2 back to f32 (reference for tests and tooling).
pub fn dequantize_low(packed: &[u8], scales: &[f16], bits: LowBits, k: usize, n: usize) -> Vec<f32> {
    let gb = bits.group_bytes();
    let mut out = vec![0.0f32; k * n];
    for (g, dst) in out.chunks_mut(GROUP).enumerate() {
        unpack_low(bits, &packed[g * gb..(g + 1) * gb], scales[g].to_f32(), dst);
    }
    out
}

/// Quantized ints of `group` for a stored (f16) scale `s`, and the squared
/// reconstruction error.
fn quantize_with(group: &[f32], s: f32, inv: f32, f: &Format, q: &mut [i32]) -> f64 {
    let mut sse = 0.0f64;
    for (x, qi) in group.iter().zip(q.iter_mut()) {
        *qi = ((x * inv).round() as i32).clamp(f.qmin, f.qmax);
        let e = (*x - s * *qi as f32) as f64;
        sse += e * e;
    }
    sse
}

/// Choose the scale for one group and write its ints into `q`.
/// Returns the stored scale.
fn quantize_group(group: &[f32], f: &Format, rounding: Rounding, q: &mut [i32]) -> f16 {
    let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax == 0.0 {
        q.fill(0);
        return f16::from_f32(0.0);
    }

    // RTN exactly as the original converter did: ints from the f32 scale,
    // the f16-rounded scale stored.
    let rtn = absmax / f.qmax as f32;
    let rtn_stored = f16::from_f32(rtn);
    let mut best_sse = quantize_with(group, rtn_stored.to_f32(), 1.0 / rtn, f, q);
    let mut best = rtn_stored;
    if rounding == Rounding::Rtn {
        return best;
    }

    let mut cand = [0i32; GROUP];
    let cand = &mut cand[..group.len()];
    let mut try_scale = |s: f16, best: &mut f16, best_sse: &mut f64, q: &mut [i32]| {
        let sv = s.to_f32();
        if sv <= 0.0 || !sv.is_finite() {
            return;
        }
        let sse = quantize_with(group, sv, 1.0 / sv, f, cand);
        if sse < *best_sse {
            *best_sse = sse;
            *best = s;
            q.copy_from_slice(cand);
        }
    };

    let steps = ((f.d_hi - f.d_lo) / f.d_step).round() as usize;
    for t in 0..=steps {
        let d = f.d_lo + t as f32 * f.d_step;
        try_scale(f16::from_f32(absmax / d), &mut best, &mut best_sse, q);
    }

    // Least-squares refit of the scale for the chosen ints, then re-round.
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, &qi) in group.iter().zip(q.iter()) {
        num += *x as f64 * qi as f64;
        den += (qi * qi) as f64;
    }
    if den > 0.0 {
        try_scale(f16::from_f32((num / den) as f32), &mut best, &mut best_sse, q);
    }
    best
}

fn check_shape(data: &[f32], k: usize, n: usize) {
    assert_eq!(data.len(), k * n, "data length {} != k*n = {}", data.len(), k * n);
    assert!(n.is_multiple_of(GROUP), "n={n} not divisible by group size {GROUP}");
}

/// Quantize one row (`row.len()` a multiple of [`GROUP`]) to Q4, writing
/// `row.len() / 2` packed bytes and `row.len() / GROUP` scales.
pub fn quantize_q4_row(row: &[f32], rounding: Rounding, prow: &mut [u8], srow: &mut [f16]) {
    let mut q = [0i32; GROUP];
    for (g, (grp, s)) in row.chunks(GROUP).zip(srow.iter_mut()).enumerate() {
        *s = quantize_group(grp, &Q4, rounding, &mut q);
        let dst = &mut prow[g * GROUP / 2..(g + 1) * GROUP / 2];
        for (j, b) in dst.iter_mut().enumerate() {
            let lo = (q[2 * j] + 8) as u8;
            let hi = (q[2 * j + 1] + 8) as u8;
            *b = lo | (hi << 4);
        }
    }
}

/// Quantize one row (`row.len()` a multiple of [`GROUP`]) to Q8.
pub fn quantize_q8_row(row: &[f32], rounding: Rounding, orow: &mut [i8], srow: &mut [f16]) {
    let mut q = [0i32; GROUP];
    for (g, (grp, s)) in row.chunks(GROUP).zip(srow.iter_mut()).enumerate() {
        *s = quantize_group(grp, &Q8, rounding, &mut q);
        for (o, &qi) in orow[g * GROUP..(g + 1) * GROUP].iter_mut().zip(&q) {
            *o = qi as i8;
        }
    }
}

/// Quantize a `[k, n]` matrix to Q4. Returns (packed bytes, scales).
pub fn quantize_q4(data: &[f32], k: usize, n: usize, rounding: Rounding) -> (Vec<u8>, Vec<f16>) {
    check_shape(data, k, n);
    let gpr = n / GROUP;
    let mut packed = vec![0u8; k * n / 2];
    let mut scales = vec![f16::from_f32(0.0); k * gpr];
    packed
        .par_chunks_mut(n / 2)
        .zip(scales.par_chunks_mut(gpr))
        .zip(data.par_chunks(n))
        .for_each(|((prow, srow), row)| quantize_q4_row(row, rounding, prow, srow));
    (packed, scales)
}

/// Quantize a `[k, n]` matrix to Q8. Returns (ints, scales).
pub fn quantize_q8(data: &[f32], k: usize, n: usize, rounding: Rounding) -> (Vec<i8>, Vec<f16>) {
    check_shape(data, k, n);
    let gpr = n / GROUP;
    let mut out = vec![0i8; k * n];
    let mut scales = vec![f16::from_f32(0.0); k * gpr];
    out.par_chunks_mut(n)
        .zip(scales.par_chunks_mut(gpr))
        .zip(data.par_chunks(n))
        .for_each(|((orow, srow), row)| quantize_q8_row(row, rounding, orow, srow));
    (out, scales)
}

/// Dequantize Q4 back to f32 (reference for tests and tooling).
pub fn dequantize_q4(packed: &[u8], scales: &[f16], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * n];
    for (i, v) in out.iter_mut().enumerate() {
        let b = packed[i / 2];
        let q = if i % 2 == 0 { b & 0x0F } else { b >> 4 } as i32 - 8;
        *v = scales[i / GROUP].to_f32() * q as f32;
    }
    out
}

/// Dequantize Q8 back to f32 (reference for tests and tooling).
pub fn dequantize_q8(data: &[i8], scales: &[f16], k: usize, n: usize) -> Vec<f32> {
    (0..k * n)
        .map(|i| scales[i / GROUP].to_f32() * data[i] as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemv::{Q4Weight, Q8Weight, Weight, gemv};

    /// Deterministic pseudo-random values with occasional outliers — the
    /// shape real weight rows have.
    fn sample(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..len)
            .map(|i| {
                // Box-Muller normal, scaled; every 97th value is an outlier.
                let (u1, u2) = (next().max(1e-12), next());
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let base = 0.02 * z as f32;
                if i % 97 == 0 { base * 12.0 } else { base }
            })
            .collect()
    }

    fn sse(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
    }

    #[test]
    fn q4_mse_is_never_worse_than_rtn_on_any_group() {
        let (k, n) = (64, 256);
        let w = sample(k * n, 1);
        let (pr, sr) = quantize_q4(&w, k, n, Rounding::Rtn);
        let (pm, sm) = quantize_q4(&w, k, n, Rounding::Mse);
        let dr = dequantize_q4(&pr, &sr, k, n);
        let dm = dequantize_q4(&pm, &sm, k, n);
        let mut strictly_better = 0;
        for g in 0..k * n / GROUP {
            let r = g * GROUP..(g + 1) * GROUP;
            let (er, em) = (sse(&w[r.clone()], &dr[r.clone()]), sse(&w[r.clone()], &dm[r]));
            assert!(em <= er + 1e-15, "group {g}: mse {em} > rtn {er}");
            if em < er {
                strictly_better += 1;
            }
        }
        // The search must actually help on typical data, not just tie.
        assert!(strictly_better * 2 > k * n / GROUP, "only {strictly_better} groups improved");
        let (tr, tm) = (sse(&w, &dr), sse(&w, &dm));
        assert!(tm < 0.9 * tr, "total error {tm} not clearly below rtn {tr}");
    }

    #[test]
    fn q4_rtn_matches_original_converter_bytes() {
        // The original converter: scale = absmax/7, ints from the f32 scale.
        let (k, n) = (8, 64);
        let w = sample(k * n, 2);
        let (packed, scales) = quantize_q4(&w, k, n, Rounding::Rtn);
        let mut exp_p = Vec::new();
        let mut exp_s = Vec::new();
        for grp in w.chunks(GROUP) {
            let absmax = grp.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 7.0 } else { 0.0 };
            exp_s.push(f16::from_f32(scale));
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for j in (0..GROUP).step_by(2) {
                let q0 = ((grp[j] * inv).round() as i32).clamp(-8, 7) + 8;
                let q1 = ((grp[j + 1] * inv).round() as i32).clamp(-8, 7) + 8;
                exp_p.push((q0 as u8) | ((q1 as u8) << 4));
            }
        }
        assert_eq!(packed, exp_p);
        assert_eq!(scales, exp_s);
    }

    #[test]
    fn q8_mse_is_never_worse_than_rtn() {
        let (k, n) = (32, 128);
        let w = sample(k * n, 3);
        let (dr, sr) = quantize_q8(&w, k, n, Rounding::Rtn);
        let (dm, sm) = quantize_q8(&w, k, n, Rounding::Mse);
        let er = dequantize_q8(&dr, &sr, k, n);
        let em = dequantize_q8(&dm, &sm, k, n);
        for g in 0..k * n / GROUP {
            let r = g * GROUP..(g + 1) * GROUP;
            assert!(sse(&w[r.clone()], &em[r.clone()]) <= sse(&w[r.clone()], &er[r]) + 1e-15);
        }
    }

    /// Every level of both low-bit formats survives pack -> unpack.
    #[test]
    fn low_bit_pack_roundtrips_every_level() {
        for bits in [LowBits::Q3, LowBits::Q2] {
            let f = bits.format();
            let q: Vec<i32> = (0..GROUP as i32).map(|i| f.qmin + i % (f.qmax - f.qmin + 1)).collect();
            let mut packed = vec![0u8; bits.group_bytes()];
            pack_low(bits, &q, &mut packed);
            let mut out = vec![0.0f32; GROUP];
            unpack_low(bits, &packed, 1.0, &mut out);
            let back: Vec<i32> = out.iter().map(|&v| v as i32).collect();
            assert_eq!(back, q, "{bits:?}");
        }
    }

    #[test]
    fn low_bit_mse_is_never_worse_than_rtn() {
        let (k, n) = (32, 128);
        let w = sample(k * n, 6);
        for bits in [LowBits::Q3, LowBits::Q2] {
            let (pr, sr) = quantize_low(&w, k, n, bits, Rounding::Rtn);
            let (pm, sm) = quantize_low(&w, k, n, bits, Rounding::Mse);
            let er = dequantize_low(&pr, &sr, bits, k, n);
            let em = dequantize_low(&pm, &sm, bits, k, n);
            for g in 0..k * n / GROUP {
                let r = g * GROUP..(g + 1) * GROUP;
                assert!(
                    sse(&w[r.clone()], &em[r.clone()]) <= sse(&w[r.clone()], &er[r]) + 1e-15,
                    "{bits:?} group {g}"
                );
            }
        }
    }

    /// Fewer bits, more error: Q2 > Q3 > Q4 on the same data.
    #[test]
    fn error_grows_as_bits_shrink() {
        let (k, n) = (32, 128);
        let w = sample(k * n, 7);
        let (p4, s4) = quantize_q4(&w, k, n, Rounding::Mse);
        let e4 = sse(&w, &dequantize_q4(&p4, &s4, k, n));
        let (p3, s3) = quantize_low(&w, k, n, LowBits::Q3, Rounding::Mse);
        let e3 = sse(&w, &dequantize_low(&p3, &s3, LowBits::Q3, k, n));
        let (p2, s2) = quantize_low(&w, k, n, LowBits::Q2, Rounding::Mse);
        let e2 = sse(&w, &dequantize_low(&p2, &s2, LowBits::Q2, k, n));
        assert!(e4 < e3 && e3 < e2, "q4 {e4} q3 {e3} q2 {e2}");
    }

    #[test]
    fn zero_group_quantizes_to_zero() {
        let w = vec![0.0f32; GROUP];
        let (p, s) = quantize_q4(&w, 1, GROUP, Rounding::Mse);
        assert!(dequantize_q4(&p, &s, 1, GROUP).iter().all(|&v| v == 0.0));
    }

    /// The kernels must read the new quantizer's output exactly as the
    /// dequantized matrix: same layout, no kernel change needed.
    #[test]
    fn kernels_agree_with_dequantized_matrix() {
        let (k, n) = (64, 96);
        let w = sample(k * n, 4);
        let x: Vec<f32> = sample(k, 5).iter().map(|v| v * 50.0).collect();
        let dense = |d: &[f32]| -> Vec<f32> {
            (0..n)
                .map(|j| (0..k).map(|i| x[i] * d[i * n + j]).sum())
                .collect()
        };

        let (packed, scales) = quantize_q4(&w, k, n, Rounding::Mse);
        let expect = dense(&dequantize_q4(&packed, &scales, k, n));
        let got = gemv(&x, &Weight::Q4(Q4Weight { packed, scales, k, n }));
        for (a, b) in got.iter().zip(&expect) {
            assert!((a - b).abs() <= 1e-3 * (1.0 + b.abs()), "q4 {a} vs {b}");
        }

        let (data, scales) = quantize_q8(&w, k, n, Rounding::Mse);
        let expect = dense(&dequantize_q8(&data, &scales, k, n));
        let got = gemv(&x, &Weight::Q8(Q8Weight { data, scales, k, n }));
        for (a, b) in got.iter().zip(&expect) {
            assert!((a - b).abs() <= 1e-3 * (1.0 + b.abs()), "q8 {a} vs {b}");
        }
    }
}
