//! Fast GEMV/GEMM for Incord-Mini-2B hybrid architecture (Gated DeltaNet + Full Attention).
//!
//! Supports two weight formats:
//! - **F16**: Half precision. Better quality.
//! - **Q4**: 4-bit symmetric quantization. Faster, lower memory.
//!
//! Q4 uses per-group (32 values) symmetric quantization:
//!   scale = absmax / 7, q = round(val/scale) + 8, packed 2 per byte.
//!   Dequant: val = (q - 8) * scale

use crate::config::{LayerType, IncordConfig};
use half::f16;
use rayon::prelude::*;

// ============================================================
// Weight format types
// ============================================================

const Q4_GROUP_SIZE: usize = 32;

/// A weight matrix stored as f16 with shape metadata.
#[derive(Clone)]
pub struct F16Weight {
    pub data: Vec<f16>,
    pub k: usize, // rows (input dim)
    pub n: usize, // cols (output dim)
}

/// A weight matrix stored in 4-bit symmetric quantization.
#[derive(Clone)]
pub struct Q4Weight {
    pub packed: Vec<u8>,
    pub scales: Vec<f16>,
    pub k: usize,
    pub n: usize,
}

pub const Q8_GROUP_SIZE: usize = 32;

/// A weight matrix stored in 8-bit symmetric quantization.
#[derive(Clone)]
pub struct Q8Weight {
    pub data: Vec<i8>,
    pub scales: Vec<f16>,
    pub k: usize,
    pub n: usize,
}

/// Polymorphic weight — either f16, Q4, or Q8.
#[derive(Clone)]
pub enum Weight {
    F16(F16Weight),
    Q4(Q4Weight),
    Q8(Q8Weight),
    /// The inner weight is stored in Hadamard-rotated input space
    /// (see `crate::rotate`): a matmul weight `[k, n]` holds `Rᵀ·W`, a head /
    /// embedding `[vocab, hidden]` holds every row times `R`. Every compute
    /// path rotates the activation with `rotate::forward` before using the
    /// inner weight (embedding lookup applies `rotate::inverse` after), so
    /// results equal the unrotated weight's.
    Rotated(Box<Weight>),
}

impl Weight {
    pub fn n(&self) -> usize {
        match self {
            Weight::F16(w) => w.n,
            Weight::Q4(w) => w.n,
            Weight::Q8(w) => w.n,
            Weight::Rotated(w) => w.n(),
        }
    }
    pub fn k(&self) -> usize {
        match self {
            Weight::F16(w) => w.k,
            Weight::Q4(w) => w.k,
            Weight::Q8(w) => w.k,
            Weight::Rotated(w) => w.k(),
        }
    }
    pub fn memory_bytes(&self) -> usize {
        match self {
            Weight::F16(w) => w.data.len() * 2,
            Weight::Q4(w) => w.packed.len() + w.scales.len() * 2,
            Weight::Q8(w) => w.data.len() + w.scales.len() * 2,
            Weight::Rotated(w) => w.memory_bytes(),
        }
    }

    /// Storage format id of the underlying data (0 = F16, 1 = Q4, 2 = Q8).
    pub fn format_id(&self) -> u8 {
        match self {
            Weight::F16(_) => 0,
            Weight::Q4(_) => 1,
            Weight::Q8(_) => 2,
            Weight::Rotated(w) => w.format_id(),
        }
    }

    pub fn is_rotated(&self) -> bool {
        matches!(self, Weight::Rotated(_))
    }

    /// Write row `i`, columns `c0 .. c0 + dst.len()`, as f32. For quantized
    /// formats `c0` and `dst.len()` must be multiples of the 32-value group
    /// (quant groups run along `n`; element `(i, c)` is value `i*n + c`).
    pub fn row_segment_f32(&self, i: usize, c0: usize, dst: &mut [f32]) {
        match self {
            Weight::F16(w) => {
                use half::slice::HalfFloatSliceExt;
                let o = i * w.n + c0;
                w.data[o..o + dst.len()].convert_to_f32_slice(dst);
            }
            Weight::Q4(w) => {
                let first = (i * w.n + c0) / Q4_GROUP_SIZE;
                for (gd, g) in dst.chunks_mut(Q4_GROUP_SIZE).zip(first..) {
                    let s = w.scales[g].to_f32();
                    let bytes = &w.packed[g * Q4_GROUP_SIZE / 2..(g + 1) * Q4_GROUP_SIZE / 2];
                    for (pair, &b) in gd.chunks_mut(2).zip(bytes) {
                        pair[0] = ((b & 0x0F) as i32 - 8) as f32 * s;
                        pair[1] = ((b >> 4) as i32 - 8) as f32 * s;
                    }
                }
            }
            Weight::Q8(w) => {
                let first = (i * w.n + c0) / Q8_GROUP_SIZE;
                for (gd, g) in dst.chunks_mut(Q8_GROUP_SIZE).zip(first..) {
                    let s = w.scales[g].to_f32();
                    let q = &w.data[g * Q8_GROUP_SIZE..(g + 1) * Q8_GROUP_SIZE];
                    for (d, &v) in gd.iter_mut().zip(q) {
                        *d = v as f32 * s;
                    }
                }
            }
            // Raw stored (rotated-space) values; callers rotate the input.
            Weight::Rotated(w) => w.row_segment_f32(i, c0, dst),
        }
    }
}

/// `x` rotated into a rotated weight's input space.
fn rotated(x: &[f32]) -> Vec<f32> {
    let mut v = x.to_vec();
    crate::rotate::forward(&mut v);
    v
}

/// Every `len`-long row of `x` rotated into rotated-weight input space.
fn rotated_rows(x: &[f32], len: usize) -> Vec<f32> {
    let mut v = x.to_vec();
    v.par_chunks_mut(len).for_each(crate::rotate::forward);
    v
}

// ============================================================
// Per-layer weight structures (hybrid)
// ============================================================

/// Weights for a Gated DeltaNet layer.
pub struct DeltaNetLayerWeights {
    pub in_proj_qkv: Weight,        // [qkv_dim, hidden]
    pub in_proj_a: Weight,          // [num_v_heads, hidden]
    pub in_proj_b: Weight,          // [num_v_heads, hidden]
    pub in_proj_z: Weight,          // [v_dim, hidden]
    pub out_proj: Weight,           // [hidden, v_dim]
    pub conv1d_weight: Vec<f32>,    // [qkv_dim, kernel_size] flattened
    pub a_log: Vec<f32>,            // [num_v_heads]
    pub dt_bias: Vec<f32>,          // [num_v_heads]
    pub attn_norm_weight: Vec<f16>, // [deltanet_head_dim] RMSNorm (per-head)
    // MLP
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

/// Weights for a full attention (GQA) layer.
pub struct FullAttnLayerWeights {
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Vec<f16>, // [attn_head_dim]
    pub k_norm: Vec<f16>, // [attn_head_dim]
    // MLP
    pub gate_proj: Weight,
    pub up_proj: Weight,
    pub down_proj: Weight,
    pub input_norm: Vec<f16>,
    pub post_attn_norm: Vec<f16>,
}

/// Per-layer weights — one of the two layer types.
pub enum HybridLayerWeights {
    DeltaNet(DeltaNetLayerWeights),
    FullAttn(FullAttnLayerWeights),
}

impl HybridLayerWeights {
    /// Every matmul weight of the layer.
    pub fn matmul_weights(&self) -> Vec<&Weight> {
        match self {
            HybridLayerWeights::DeltaNet(d) => vec![
                &d.in_proj_qkv,
                &d.in_proj_a,
                &d.in_proj_b,
                &d.in_proj_z,
                &d.out_proj,
                &d.gate_proj,
                &d.up_proj,
                &d.down_proj,
            ],
            HybridLayerWeights::FullAttn(f) => vec![
                &f.q_proj,
                &f.k_proj,
                &f.v_proj,
                &f.o_proj,
                &f.gate_proj,
                &f.up_proj,
                &f.down_proj,
            ],
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            HybridLayerWeights::DeltaNet(d) => {
                d.in_proj_qkv.memory_bytes()
                    + d.in_proj_a.memory_bytes()
                    + d.in_proj_b.memory_bytes()
                    + d.in_proj_z.memory_bytes()
                    + d.out_proj.memory_bytes()
                    + d.conv1d_weight.len() * 4
                    + (d.a_log.len() + d.dt_bias.len()) * 4
                    + d.attn_norm_weight.len() * 2
                    + d.gate_proj.memory_bytes()
                    + d.up_proj.memory_bytes()
                    + d.down_proj.memory_bytes()
                    + (d.input_norm.len() + d.post_attn_norm.len()) * 2
            }
            HybridLayerWeights::FullAttn(f) => {
                f.q_proj.memory_bytes()
                    + f.k_proj.memory_bytes()
                    + f.v_proj.memory_bytes()
                    + f.o_proj.memory_bytes()
                    + (f.q_norm.len() + f.k_norm.len()) * 2
                    + f.gate_proj.memory_bytes()
                    + f.up_proj.memory_bytes()
                    + f.down_proj.memory_bytes()
                    + (f.input_norm.len() + f.post_attn_norm.len()) * 2
            }
        }
    }
}

// ============================================================
// DeltaNet state cache
// ============================================================

/// Per-layer DeltaNet recurrent state.
pub struct DeltaNetState {
    /// State matrices: [num_v_heads, head_k_dim, head_v_dim] flattened.
    /// Each V head has its own [128, 128] state matrix.
    pub s: Vec<f32>,
    /// Convolution buffer: last (kernel_size-1) QKV values.
    /// Layout: [qkv_dim, kernel_size-1] flattened.
    pub conv_buf: Vec<f32>,
    /// How many tokens have been fed (for conv warmup).
    pub conv_pos: usize,
}

impl DeltaNetState {
    pub fn new(config: &IncordConfig) -> Self {
        let num_v = config.num_v_heads; // 32
        let k_hd = config.deltanet_head_dim; // 128
        let v_hd = config.deltanet_head_dim; // 128 (same dim)
        let qkv_dim = config.deltanet_qkv_dim();
        let kernel_size = config.conv_kernel_size;
        Self {
            s: vec![0.0; num_v * k_hd * v_hd],
            conv_buf: vec![0.0; qkv_dim * (kernel_size - 1)],
            conv_pos: 0,
        }
    }
}

// ============================================================
// Hybrid cache (DeltaNet states + KV caches)
// ============================================================

/// KV cache entry for a full attention layer.
pub struct KvCacheEntry {
    pub k: Vec<f16>, // [seq_len, kv_heads, head_dim] flat — stored as f16 (50% vs f32)
    pub v: Vec<f16>,
    pub seq_len: usize,
}

/// Hybrid cache for the full model.
pub struct HybridCache {
    /// One entry per layer (32 total).
    pub entries: Vec<CacheEntry>,
}

pub enum CacheEntry {
    DeltaNet(DeltaNetState),
    KvCache(KvCacheEntry),
}

impl HybridCache {
    pub fn new(config: &IncordConfig) -> Self {
        let mut entries = Vec::with_capacity(config.num_layers);
        for lt in &config.layer_types {
            match lt {
                LayerType::DeltaNet => {
                    entries.push(CacheEntry::DeltaNet(DeltaNetState::new(config)))
                }
                LayerType::FullAttn => entries.push(CacheEntry::KvCache(KvCacheEntry {
                    k: Vec::with_capacity(config.num_kv_heads * 512 * config.attn_head_dim),
                    v: Vec::with_capacity(config.num_kv_heads * 512 * config.attn_head_dim),
                    seq_len: 0,
                })),
            }
        }
        Self { entries }
    }

    /// Clone this cache (deep copy of all states).
    pub fn snapshot(&self) -> Self {
        let entries = self
            .entries
            .iter()
            .map(|e| match e {
                CacheEntry::DeltaNet(ds) => CacheEntry::DeltaNet(DeltaNetState {
                    s: ds.s.clone(),
                    conv_buf: ds.conv_buf.clone(),
                    conv_pos: ds.conv_pos,
                }),
                CacheEntry::KvCache(kv) => CacheEntry::KvCache(KvCacheEntry {
                    k: kv.k.clone(),
                    v: kv.v.clone(),
                    seq_len: kv.seq_len,
                }),
            })
            .collect();
        Self { entries }
    }

    /// Restore from a snapshot (deep copy from src).
    pub fn restore_from(&mut self, src: &HybridCache) {
        for (dst, s) in self.entries.iter_mut().zip(src.entries.iter()) {
            match (dst, s) {
                (CacheEntry::DeltaNet(d), CacheEntry::DeltaNet(s)) => {
                    d.s.copy_from_slice(&s.s);
                    d.conv_buf.copy_from_slice(&s.conv_buf);
                    d.conv_pos = s.conv_pos;
                }
                (CacheEntry::KvCache(d), CacheEntry::KvCache(s)) => {
                    d.k.clear();
                    d.k.extend_from_slice(&s.k);
                    d.v.clear();
                    d.v.extend_from_slice(&s.v);
                    d.seq_len = s.seq_len;
                }
                _ => panic!("Cache type mismatch during restore"),
            }
        }
    }
}

/// Cached system prompt state for fast re-prefill.
pub struct SystemPromptCache {
    pub tokens: Vec<u32>,
    pub cache_snapshot: HybridCache,
}

// ============================================================
// Model weights
// ============================================================

/// All model weights for Incord inference.
pub struct ModelWeights {
    pub layers: Vec<HybridLayerWeights>,
    pub embed: Weight,
    pub lm_head: Weight,
    pub vocab: usize,
    pub hidden: usize,
    pub final_norm: Vec<f16>,
    pub rope_cos: Vec<f32>, // [max_pos, rope_dim/2] (32 entries per position)
    pub rope_sin: Vec<f32>,
    pub config: IncordConfig,
    pub format_name: &'static str,
    pub vision: Option<crate::vision::VisionEncoder>,
}

impl ModelWeights {
    pub fn memory_bytes(&self) -> usize {
        let mut total = self.embed.memory_bytes();
        if !self.config.tie_word_embeddings {
            total += self.lm_head.memory_bytes();
        }
        total += self.final_norm.len() * 2;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        for l in &self.layers {
            total += l.memory_bytes();
        }
        if let Some(v) = &self.vision {
            total += v.memory_bytes();
        }
        total
    }

    pub fn format_id(&self) -> u8 {
        self.embed.format_id()
    }

    /// True when any weight is stored in rotated input space.
    pub fn has_rotated(&self) -> bool {
        self.embed.is_rotated()
            || self.lm_head.is_rotated()
            || self
                .layers
                .iter()
                .any(|l| l.matmul_weights().iter().any(|w| w.is_rotated()))
    }
}

// ============================================================
// GEMV dispatch (single-token decode)
// ============================================================

/// GEMV: output = input @ weight.
#[inline]
pub fn gemv(input: &[f32], weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemv_f16(input, w),
        Weight::Q4(w) => gemv_q4(input, w),
        Weight::Q8(w) => gemv_q8(input, w),
        Weight::Rotated(w) => gemv(&rotated(input), w),
    }
}

/// GEMM: [seq_len, k] @ [k, n] -> [seq_len, n].
#[inline]
pub fn gemm(x: &[f32], seq_len: usize, weight: &Weight) -> Vec<f32> {
    if let Weight::Rotated(w) = weight {
        return gemm(&rotated_rows(x, w.k()), seq_len, w);
    }
    if seq_len >= GEMM_TILED_MIN_TOKENS {
        return gemm_tiled(x, seq_len, weight);
    }
    match weight {
        Weight::F16(w) => gemm_f16(x, seq_len, w),
        Weight::Q4(w) => gemm_q4(x, seq_len, w),
        Weight::Q8(w) => gemm_q8(x, seq_len, w),
        Weight::Rotated(_) => unreachable!("unwrapped above"),
    }
}

/// Fused gate+up GEMV: silu(gate(x)) * up(x) in one pass.
#[inline]
pub(crate) fn fused_gate_up_gemv(input: &[f32], gate_w: &Weight, up_w: &Weight) -> Vec<f32> {
    match (gate_w, up_w) {
        (Weight::Q4(gw), Weight::Q4(uw)) => fused_gate_up_q4(input, gw, uw),
        // Same input, same rotation: rotate once, keep the fused kernel.
        (Weight::Rotated(gw), Weight::Rotated(uw)) => fused_gate_up_gemv(&rotated(input), gw, uw),
        _ => {
            let gate = gemv(input, gate_w);
            let up = gemv(input, up_w);
            let n = gate.len();
            let mut out = vec![0.0f32; n];
            for j in 0..n {
                let g = gate[j];
                out[j] = (g / (1.0 + (-g).exp())) * up[j];
            }
            out
        }
    }
}

/// Embedding lookup.
#[inline]
pub fn embed_lookup(weight: &Weight, token_id: usize, hidden: usize) -> Vec<f32> {
    match weight {
        Weight::F16(w) => {
            let start = token_id * hidden;
            w.data[start..start + hidden]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        }
        Weight::Q4(w) => embed_lookup_q4(w, token_id),
        Weight::Q8(w) => embed_lookup_q8(w, token_id),
        Weight::Rotated(w) => {
            let mut row = embed_lookup(w, token_id, hidden);
            crate::rotate::inverse(&mut row);
            row
        }
    }
}

// ============================================================
// F16 compute kernels
// ============================================================

#[inline]
fn gemv_f16(input: &[f32], weight: &F16Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    let w = &weight.data;

    // AVX-512 fast path
    if crate::simd::has_avx512() {
        if k * n < 4_000_000 {
            return unsafe { crate::simd::gemv_f16_avx512(input, w, k, n) };
        }
        // Threaded: split across k, each thread uses AVX-512
        let num_threads = rayon::current_num_threads();
        let chunk_k = k.div_ceil(num_threads);
        let partials: Vec<Vec<f32>> = (0..num_threads)
            .into_par_iter()
            .filter_map(|t| {
                let k_start = t * chunk_k;
                let k_end = ((t + 1) * chunk_k).min(k);
                if k_start >= k {
                    return None;
                }
                Some(unsafe {
                    crate::simd::gemv_f16_avx512(
                        &input[k_start..k_end],
                        &w[k_start * n..],
                        k_end - k_start,
                        n,
                    )
                })
            })
            .collect();
        let mut output = vec![0.0f32; n];
        for p in &partials {
            for j in 0..n {
                output[j] += p[j];
            }
        }
        return output;
    }

    // AVX2 + FMA + F16C fallback path — same shape as AVX-512 above.
    if crate::simd::has_avx2() && crate::simd::has_fma() && crate::simd::has_f16c() {
        if k * n < 4_000_000 {
            return unsafe { crate::simd::gemv_f16_avx2(input, w, k, n) };
        }
        let num_threads = rayon::current_num_threads();
        let chunk_k = k.div_ceil(num_threads);
        let partials: Vec<Vec<f32>> = (0..num_threads)
            .into_par_iter()
            .filter_map(|t| {
                let k_start = t * chunk_k;
                let k_end = ((t + 1) * chunk_k).min(k);
                if k_start >= k {
                    return None;
                }
                Some(unsafe {
                    crate::simd::gemv_f16_avx2(
                        &input[k_start..k_end],
                        &w[k_start * n..],
                        k_end - k_start,
                        n,
                    )
                })
            })
            .collect();
        let mut output = vec![0.0f32; n];
        for p in &partials {
            for j in 0..n {
                output[j] += p[j];
            }
        }
        return output;
    }

    if k * n < 4_000_000 {
        let mut output = vec![0.0f32; n];
        for (ki, &val) in input.iter().enumerate().take(k) {
            let row = ki * n;
            for j in 0..n {
                output[j] += val * w[row + j].to_f32();
            }
        }
        return output;
    }
    let num_threads = rayon::current_num_threads();
    let chunk_k = k.div_ceil(num_threads);
    let partials: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k {
                return None;
            }
            let mut out = vec![0.0f32; n];
            for (i, &val) in input[k_start..k_end].iter().enumerate() {
                let ki = k_start + i;
                let row = ki * n;
                for j in 0..n {
                    out[j] += val * w[row + j].to_f32();
                }
            }
            Some(out)
        })
        .collect();
    let mut output = vec![0.0f32; n];
    for p in &partials {
        for j in 0..n {
            output[j] += p[j];
        }
    }
    output
}

// ============================================================
// Token-tiled GEMM (prefill)
// ============================================================

/// Prompts at least this long use [`gemm_tiled`]; shorter batches (e.g.
/// multi-slot decode) keep the per-token path, which parallelises better
/// when there are few tokens.
const GEMM_TILED_MIN_TOKENS: usize = 16;
/// Tokens per tile: each weight row segment is dequantized once per tile.
const GEMM_TILE_TOKENS: usize = 32;
/// Output columns per tile (multiple of the 32-value quant group).
const GEMM_TILE_COLS: usize = 128;

/// `out[t, j] = Σ_i x[t, i] · W[i, j]` for `x: [seq_len, k]`, `W: [k, n]`.
///
/// The work is split into (token tile × column tile) blocks; within a block
/// each weight row segment is dequantized once and applied to every token of
/// the tile, so the weights are streamed `seq_len / GEMM_TILE_TOKENS` times
/// instead of once per token.
fn gemm_tiled(x: &[f32], seq_len: usize, weight: &Weight) -> Vec<f32> {
    let (k, n) = (weight.k(), weight.n());
    let t_tiles = seq_len.div_ceil(GEMM_TILE_TOKENS);
    let c_tiles = n.div_ceil(GEMM_TILE_COLS);
    let blocks: Vec<(usize, usize, usize, Vec<f32>)> = (0..t_tiles * c_tiles)
        .into_par_iter()
        .map(|b| {
            let t0 = (b / c_tiles) * GEMM_TILE_TOKENS;
            let c0 = (b % c_tiles) * GEMM_TILE_COLS;
            let nt = (seq_len - t0).min(GEMM_TILE_TOKENS);
            let nc = (n - c0).min(GEMM_TILE_COLS);
            let mut acc = vec![0.0f32; nt * nc];
            let mut wrow = vec![0.0f32; nc];
            for i in 0..k {
                weight.row_segment_f32(i, c0, &mut wrow);
                for t in 0..nt {
                    let xv = x[(t0 + t) * k + i];
                    for (a, &w) in acc[t * nc..(t + 1) * nc].iter_mut().zip(&wrow) {
                        *a += xv * w;
                    }
                }
            }
            (t0, c0, nc, acc)
        })
        .collect();
    let mut out = vec![0.0f32; seq_len * n];
    for (t0, c0, nc, acc) in blocks {
        for (t, row) in acc.chunks(nc).enumerate() {
            let o = (t0 + t) * n + c0;
            out[o..o + nc].copy_from_slice(row);
        }
    }
    out
}

#[inline]
fn gemm_f16(x: &[f32], seq_len: usize, weight: &F16Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    let w = &weight.data;
    let mut output = vec![0.0f32; seq_len * n];
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let x_row = &x[t * k..(t + 1) * k];
            for (ki, &val) in x_row.iter().enumerate().take(k) {
                let w_start = ki * n;
                for j in 0..n {
                    out_row[j] += val * w[w_start + j].to_f32();
                }
            }
        });
    output
}

// ============================================================
// Q4 compute kernels
// ============================================================

#[inline]
fn gemv_q4_inner(
    input: &[f32],
    packed: &[u8],
    scales: &[f16],
    _k: usize,
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    debug_assert!(
        n.is_multiple_of(Q4_GROUP_SIZE),
        "Q4 GEMV: n={n} not divisible by {Q4_GROUP_SIZE}"
    );

    // AVX-512 fast path
    if crate::simd::has_avx512() {
        return unsafe { crate::simd::gemv_q4_avx512(input, packed, scales, n, k_start, k_end) };
    }
    // AVX2 + FMA fallback
    if crate::simd::has_avx2() && crate::simd::has_fma() {
        return unsafe { crate::simd::gemv_q4_avx2(input, packed, scales, n, k_start, k_end) };
    }
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;
    let mut output = vec![0.0f32; n];
    for (i, &val) in input[k_start..k_end].iter().enumerate() {
        let ki = k_start + i;
        if val == 0.0 {
            continue;
        }
        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;
        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 {
                continue;
            }
            let lut = [
                s * -8.0,
                s * -7.0,
                s * -6.0,
                s * -5.0,
                s * -4.0,
                s * -3.0,
                s * -2.0,
                -s,
                0.0,
                s,
                s * 2.0,
                s * 3.0,
                s * 4.0,
                s * 5.0,
                s * 6.0,
                s * 7.0,
            ];
            let po = pack_base + g * packed_per_group;
            let oo = g * Q4_GROUP_SIZE;
            for j in 0..packed_per_group {
                let byte = packed[po + j];
                output[oo + j * 2] += lut[(byte & 0x0F) as usize];
                output[oo + j * 2 + 1] += lut[(byte >> 4) as usize];
            }
        }
    }
    output
}

#[inline]
fn gemv_q4(input: &[f32], weight: &Q4Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    if k * n < 4_000_000 {
        return gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, 0, k);
    }
    let num_threads = rayon::current_num_threads();
    let chunk_k = k.div_ceil(num_threads);
    let partials: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k {
                return None;
            }
            Some(gemv_q4_inner(
                input,
                &weight.packed,
                &weight.scales,
                k,
                n,
                k_start,
                k_end,
            ))
        })
        .collect();
    let mut output = vec![0.0f32; n];
    for p in &partials {
        for j in 0..n {
            output[j] += p[j];
        }
    }
    output
}

fn fused_gate_up_q4(input: &[f32], gate_w: &Q4Weight, up_w: &Q4Weight) -> Vec<f32> {
    let (k, n) = (gate_w.k, gate_w.n);
    let num_threads = rayon::current_num_threads();
    let chunk_k = k.div_ceil(num_threads);
    let use_avx512 = crate::simd::has_avx512();
    let use_avx2 = !use_avx512 && crate::simd::has_avx2() && crate::simd::has_fma();
    let partials: Vec<(Vec<f32>, Vec<f32>)> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k {
                return None;
            }

            if use_avx512 {
                return Some(unsafe {
                    crate::simd::fused_gate_up_q4_avx512(
                        input,
                        &gate_w.packed,
                        &gate_w.scales,
                        &up_w.packed,
                        &up_w.scales,
                        n,
                        k_start,
                        k_end,
                    )
                });
            }

            if use_avx2 {
                return Some(unsafe {
                    crate::simd::fused_gate_up_q4_avx2(
                        input,
                        &gate_w.packed,
                        &gate_w.scales,
                        &up_w.packed,
                        &up_w.scales,
                        n,
                        k_start,
                        k_end,
                    )
                });
            }

            // Scalar fallback
            let groups_per_row = n / Q4_GROUP_SIZE;
            let packed_per_group = Q4_GROUP_SIZE / 2;
            let mut gate_out = vec![0.0f32; n];
            let mut up_out = vec![0.0f32; n];
            for (i, &val) in input[k_start..k_end].iter().enumerate() {
                let ki = k_start + i;
                if val == 0.0 {
                    continue;
                }
                let sb = ki * groups_per_row;
                let pb = ki * groups_per_row * packed_per_group;
                for g in 0..groups_per_row {
                    let gs = gate_w.scales[sb + g].to_f32() * val;
                    let us = up_w.scales[sb + g].to_f32() * val;
                    if gs == 0.0 && us == 0.0 {
                        continue;
                    }
                    let g_lut = [
                        gs * -8.0,
                        gs * -7.0,
                        gs * -6.0,
                        gs * -5.0,
                        gs * -4.0,
                        gs * -3.0,
                        gs * -2.0,
                        -gs,
                        0.0,
                        gs,
                        gs * 2.0,
                        gs * 3.0,
                        gs * 4.0,
                        gs * 5.0,
                        gs * 6.0,
                        gs * 7.0,
                    ];
                    let u_lut = [
                        us * -8.0,
                        us * -7.0,
                        us * -6.0,
                        us * -5.0,
                        us * -4.0,
                        us * -3.0,
                        us * -2.0,
                        -us,
                        0.0,
                        us,
                        us * 2.0,
                        us * 3.0,
                        us * 4.0,
                        us * 5.0,
                        us * 6.0,
                        us * 7.0,
                    ];
                    let po = pb + g * packed_per_group;
                    let oo = g * Q4_GROUP_SIZE;
                    for j in 0..packed_per_group {
                        let gb = gate_w.packed[po + j];
                        let ub = up_w.packed[po + j];
                        gate_out[oo + j * 2] += g_lut[(gb & 0x0F) as usize];
                        gate_out[oo + j * 2 + 1] += g_lut[(gb >> 4) as usize];
                        up_out[oo + j * 2] += u_lut[(ub & 0x0F) as usize];
                        up_out[oo + j * 2 + 1] += u_lut[(ub >> 4) as usize];
                    }
                }
            }
            Some((gate_out, up_out))
        })
        .collect();
    let mut gf = vec![0.0f32; n];
    let mut uf = vec![0.0f32; n];
    for (gp, up) in &partials {
        for j in 0..n {
            gf[j] += gp[j];
            uf[j] += up[j];
        }
    }
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let g = gf[j];
        out[j] = (g / (1.0 + (-g).exp())) * uf[j];
    }
    out
}

#[inline]
fn gemm_q4(x: &[f32], seq_len: usize, weight: &Q4Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    // Always parallelize across rows for GEMM (seq_len > 1)
    if seq_len <= 1 {
        let row = gemv_q4_inner(&x[..k], &weight.packed, &weight.scales, k, n, 0, k);
        return row;
    }
    let mut output = vec![0.0f32; seq_len * n];
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let row = gemv_q4_inner(
                &x[t * k..(t + 1) * k],
                &weight.packed,
                &weight.scales,
                k,
                n,
                0,
                k,
            );
            out_row.copy_from_slice(&row);
        });
    output
}

fn embed_lookup_q4(weight: &Q4Weight, token_id: usize) -> Vec<f32> {
    let n = weight.n;
    debug_assert!(
        n.is_multiple_of(Q4_GROUP_SIZE),
        "Q4 embed: n={n} not divisible by {Q4_GROUP_SIZE}"
    );
    let gpr = n / Q4_GROUP_SIZE;
    let ppg = Q4_GROUP_SIZE / 2;
    let sb = token_id * gpr;
    let pb = token_id * gpr * ppg;
    let mut output = vec![0.0f32; n];
    for g in 0..gpr {
        let scale = weight.scales[sb + g].to_f32();
        let po = pb + g * ppg;
        let oo = g * Q4_GROUP_SIZE;
        for j in 0..ppg {
            let byte = weight.packed[po + j];
            output[oo + j * 2] = scale * ((byte & 0x0F) as i32 - 8) as f32;
            output[oo + j * 2 + 1] = scale * (((byte >> 4) & 0x0F) as i32 - 8) as f32;
        }
    }
    output
}

// ============================================================
// Q8 compute kernels
// ============================================================

#[inline]
fn gemv_q8_inner(
    input: &[f32],
    data: &[i8],
    scales: &[f16],
    _k: usize,
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    debug_assert!(
        n.is_multiple_of(Q8_GROUP_SIZE),
        "Q8 GEMV: n={n} not divisible by {Q8_GROUP_SIZE}"
    );
    let groups_per_row = n / Q8_GROUP_SIZE;
    let mut output = vec![0.0f32; n];
    for (i, &val) in input[k_start..k_end].iter().enumerate() {
        let ki = k_start + i;
        if val == 0.0 {
            continue;
        }
        let scale_base = ki * groups_per_row;
        let data_base = ki * n;
        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 {
                continue;
            }
            let row_offset = data_base + g * Q8_GROUP_SIZE;
            let out_offset = g * Q8_GROUP_SIZE;
            for j in 0..Q8_GROUP_SIZE {
                output[out_offset + j] += (data[row_offset + j] as f32) * s;
            }
        }
    }
    output
}

#[inline]
fn gemv_q8(input: &[f32], weight: &Q8Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    if k * n < 4_000_000 {
        return gemv_q8_inner(input, &weight.data, &weight.scales, k, n, 0, k);
    }
    let num_threads = rayon::current_num_threads();
    let chunk_k = k.div_ceil(num_threads);
    let partials: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k {
                return None;
            }
            Some(gemv_q8_inner(
                input,
                &weight.data,
                &weight.scales,
                k,
                n,
                k_start,
                k_end,
            ))
        })
        .collect();
    let mut output = vec![0.0f32; n];
    for p in &partials {
        for j in 0..n {
            output[j] += p[j];
        }
    }
    output
}

#[inline]
fn gemm_q8(x: &[f32], seq_len: usize, weight: &Q8Weight) -> Vec<f32> {
    let (k, n) = (weight.k, weight.n);
    if seq_len <= 1 {
        return gemv_q8_inner(&x[..k], &weight.data, &weight.scales, k, n, 0, k);
    }
    let mut output = vec![0.0f32; seq_len * n];
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let row = gemv_q8_inner(
                &x[t * k..(t + 1) * k],
                &weight.data,
                &weight.scales,
                k,
                n,
                0,
                k,
            );
            out_row.copy_from_slice(&row);
        });
    output
}

fn embed_lookup_q8(weight: &Q8Weight, token_id: usize) -> Vec<f32> {
    let n = weight.n;
    debug_assert!(
        n.is_multiple_of(Q8_GROUP_SIZE),
        "Q8 embed: n={n} not divisible by {Q8_GROUP_SIZE}"
    );
    let gpr = n / Q8_GROUP_SIZE;
    let sb = token_id * gpr;
    let db = token_id * n;
    let mut output = vec![0.0f32; n];
    for g in 0..gpr {
        let scale = weight.scales[sb + g].to_f32();
        let po = db + g * Q8_GROUP_SIZE;
        let oo = g * Q8_GROUP_SIZE;
        for j in 0..Q8_GROUP_SIZE {
            output[oo + j] = scale * (weight.data[po + j] as f32);
        }
    }
    output
}

/// Quantize f32 data to Q8 symmetric (group_size=32).
/// Returns (Vec<i8>, Vec<f16>).
pub fn quantize_q8(data: &[f32], k: usize, n: usize) -> (Vec<i8>, Vec<f16>) {
    let group_size = Q8_GROUP_SIZE;
    assert!(
        n.is_multiple_of(group_size),
        "n={n} not divisible by group_size={group_size}"
    );
    let groups_per_row = n / group_size;

    let mut q8_data = Vec::with_capacity(k * n);
    let mut scales = Vec::with_capacity(k * groups_per_row);

    for ki in 0..k {
        for g in 0..groups_per_row {
            let offset = ki * n + g * group_size;
            let group = &data[offset..offset + group_size];

            let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if absmax > 0.0 { absmax / 127.0 } else { 0.0 };
            scales.push(f16::from_f32(scale));

            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for &val in group {
                let q = ((val * inv_scale).round() as i32).clamp(-128, 127) as i8;
                q8_data.push(q);
            }
        }
    }

    (q8_data, scales)
}

// ============================================================
// Shared compute kernels
// ============================================================

/// RmsNorm with f16 gamma. eps = 1e-6.
#[inline]
pub fn rms_norm(x: &[f32], gamma: &[f16]) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / n as f32 + 1e-6).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = x[i] * inv_rms * gamma[i].to_f32();
    }
    out
}

/// Per-head RMS norm for QK norms in full attention.
/// Input: [num_heads * head_dim], applies separate RMS norm per head using shared gamma[head_dim].
#[inline]
fn per_head_rms_norm(x: &mut [f32], num_heads: usize, head_dim: usize, gamma: &[f16]) {
    for h in 0..num_heads {
        let base = h * head_dim;
        let head = &x[base..base + head_dim];
        let sum_sq: f32 = head.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
        for d in 0..head_dim {
            x[base + d] = x[base + d] * inv_rms * gamma[d].to_f32();
        }
    }
}

/// Apply partial RoPE (only first rotary_dim dimensions of each head).
/// Incord-Mini-2B uses split-half pairing: pair (i, i + half) for i in 0..half, where half = rotary_dim/2.
/// cos/sin tables have `half` entries per position (since each pair shares the same frequency).
/// rotary_dim = total dims rotated per head (e.g. 64 = 256 * 0.25).
#[inline]
fn apply_partial_rope(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    rotary_dim: usize, // total dims rotated per head (64)
    position: usize,
) {
    let half = rotary_dim / 2; // 32: number of pairs
    let cos_offset = position * half; // table has `half` entries per position
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let x1 = data[base + i]; // dim i
            let x2 = data[base + half + i]; // dim i + half
            let c = cos_table[cos_offset + i];
            let s = sin_table[cos_offset + i];
            // Split-half RoPE: both elements of pair use same frequency
            data[base + i] = x1 * c - x2 * s;
            data[base + half + i] = x2 * c + x1 * s;
        }
    }
}

/// SiLU activation: x * sigmoid(x).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// L2 normalize a vector in-place.
#[inline]
fn l2_normalize(x: &mut [f32]) {
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

/// In-place softmax.
#[inline]
fn softmax_raw(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

/// Parallel lm_head.
/// Logits for several positions at once: `rows` is `[n_rows, hidden]`, the
/// head weight is `[vocab, hidden]` (same layout `lm_head_parallel` reads).
/// Returns `[n_rows, vocab]`. Each vocab row is dequantized once and dotted
/// with every position, instead of once per position.
pub fn lm_head_batch(rows: &[f32], n_rows: usize, weight: &Weight) -> Vec<f32> {
    const VOCAB_CHUNK: usize = 256;
    let (vocab, hidden) = (weight.k(), weight.n());
    assert_eq!(rows.len(), n_rows * hidden);
    if let Weight::Rotated(w) = weight {
        return lm_head_batch(&rotated_rows(rows, hidden), n_rows, w);
    }
    let chunks: Vec<Vec<f32>> = (0..vocab.div_ceil(VOCAB_CHUNK))
        .into_par_iter()
        .map(|c| {
            let v0 = c * VOCAB_CHUNK;
            let nv = (vocab - v0).min(VOCAB_CHUNK);
            let mut out = vec![0.0f32; n_rows * nv];
            let mut w = vec![0.0f32; hidden];
            for vi in 0..nv {
                weight.row_segment_f32(v0 + vi, 0, &mut w);
                for t in 0..n_rows {
                    out[t * nv + vi] = dot_lanes(&w, &rows[t * hidden..(t + 1) * hidden]);
                }
            }
            out
        })
        .collect();
    let mut logits = vec![0.0f32; n_rows * vocab];
    for (c, out) in chunks.iter().enumerate() {
        let v0 = c * VOCAB_CHUNK;
        let nv = out.len() / n_rows.max(1);
        for t in 0..n_rows {
            logits[t * vocab + v0..t * vocab + v0 + nv].copy_from_slice(&out[t * nv..(t + 1) * nv]);
        }
    }
    logits
}

/// Dot product with 16 independent accumulators so the compiler vectorizes
/// it (a single running sum forces sequential float adds).
fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    const L: usize = 16;
    let mut acc = [0.0f32; L];
    let (ca, cb) = (a.chunks_exact(L), b.chunks_exact(L));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (xa, xb) in ca.zip(cb) {
        for l in 0..L {
            acc[l] += xa[l] * xb[l];
        }
    }
    acc.iter().sum::<f32>() + tail
}

pub fn lm_head_parallel(input: &[f32], embed: &Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    if let Weight::Rotated(w) = embed {
        return lm_head_parallel(&rotated(input), w, vocab, hidden);
    }
    match embed {
        Weight::F16(w) => {
            let mut output = vec![0.0f32; vocab];
            output
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * 256;
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let r = start + i;
                        let row = &w.data[r * hidden..(r + 1) * hidden];
                        let mut sum = 0.0f32;
                        for j in 0..hidden {
                            sum += input[j] * row[j].to_f32();
                        }
                        *out = sum;
                    }
                });
            output
        }
        Weight::Q4(w) => {
            let gpr = hidden / Q4_GROUP_SIZE;
            let ppg = Q4_GROUP_SIZE / 2;
            let ppr = gpr * ppg;
            let mut output = vec![0.0f32; vocab];
            output
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * 256;
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let v = start + i;
                        let mut dot = 0.0f32;
                        let sb = v * gpr;
                        let pb = v * ppr;
                        for g in 0..gpr {
                            let scale = w.scales[sb + g].to_f32();
                            if scale == 0.0 {
                                continue;
                            }
                            let po = pb + g * ppg;
                            let io = g * Q4_GROUP_SIZE;
                            let lut = [
                                scale * -8.0,
                                scale * -7.0,
                                scale * -6.0,
                                scale * -5.0,
                                scale * -4.0,
                                scale * -3.0,
                                scale * -2.0,
                                -scale,
                                0.0,
                                scale,
                                scale * 2.0,
                                scale * 3.0,
                                scale * 4.0,
                                scale * 5.0,
                                scale * 6.0,
                                scale * 7.0,
                            ];
                            for j in 0..ppg {
                                let byte = w.packed[po + j];
                                dot += input[io + j * 2] * lut[(byte & 0x0F) as usize];
                                dot += input[io + j * 2 + 1] * lut[(byte >> 4) as usize];
                            }
                        }
                        *out = dot;
                    }
                });
            output
        }
        Weight::Q8(w) => {
            let gpr = hidden / Q8_GROUP_SIZE;
            let mut output = vec![0.0f32; vocab];
            output
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let start = ci * 256;
                    for (i, out) in chunk.iter_mut().enumerate() {
                        let v = start + i;
                        let mut dot = 0.0f32;
                        let sb = v * gpr;
                        let db = v * hidden;
                        for g in 0..gpr {
                            let scale = w.scales[sb + g].to_f32();
                            if scale == 0.0 {
                                continue;
                            }
                            let po = db + g * Q8_GROUP_SIZE;
                            let io = g * Q8_GROUP_SIZE;
                            let mut g_dot = 0.0f32;
                            for j in 0..Q8_GROUP_SIZE {
                                g_dot += input[io + j] * (w.data[po + j] as f32);
                            }
                            dot += g_dot * scale;
                        }
                        *out = dot;
                    }
                });
            output
        }
        Weight::Rotated(_) => unreachable!("unwrapped above"),
    }
}

// ============================================================
// DeltaNet single-step forward
// ============================================================

/// Run one step of the DeltaNet inner loop given pre-computed projections.
/// Mirrors the per-token inner block of forward_deltanet_decode.
///
/// `qkv_t`: slice of in_proj_qkv output for this token [qkv_dim]
/// `a_t`:   slice of in_proj_a output for this token [num_v_heads]
/// `b_t`:   slice of in_proj_b output for this token [num_v_heads]
/// `z_t`:   slice of in_proj_z output for this token [v_dim]
///
/// Returns the gated attention output `gated` of length [num_v_heads * head_dim],
/// ready to be passed to out_proj.
pub(crate) fn deltanet_step(
    qkv_t: &[f32],
    a_t: &[f32],
    b_t: &[f32],
    z_t: &[f32],
    lw: &DeltaNetLayerWeights,
    state: &mut DeltaNetState,
    config: &IncordConfig,
) -> Vec<f32> {
    let num_qk = config.num_qk_heads; // 16
    let num_v = config.num_v_heads; // 32
    let hd = config.deltanet_head_dim; // 128
    let q_dim = config.deltanet_q_dim(); // 2048
    let k_dim = config.deltanet_k_dim(); // 2048
    let v_dim = config.deltanet_v_dim(); // 4096
    let qkv_dim = config.deltanet_qkv_dim(); // 8192
    let ks = config.conv_kernel_size; // 4
    let buf_len = ks - 1; // 3
    let v_per_qk = num_v / num_qk; // 2 — GVA ratio
    let scale = 1.0 / (hd as f32).sqrt(); // 1/sqrt(128)

    // Conv1d (kernel=4, depthwise across qkv_dim channels)
    // Apply conv FIRST using old buffer + current qkv, THEN update buffer.
    let conv_w = &lw.conv1d_weight;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    for ch in 0..qkv_dim {
        let mut sum = 0.0f32;
        for t in 0..buf_len {
            sum += state.conv_buf[ch * buf_len + t] * conv_w[ch * ks + t];
        }
        sum += qkv_t[ch] * conv_w[ch * ks + buf_len];
        qkv_conv[ch] = sum;
    }

    // Update buffer for next step
    {
        let buf = &mut state.conv_buf;
        for ch in 0..qkv_dim {
            for t in 0..buf_len - 1 {
                buf[ch * buf_len + t] = buf[ch * buf_len + t + 1];
            }
            buf[ch * buf_len + buf_len - 1] = qkv_t[ch];
        }
    }
    state.conv_pos += 1;

    // SiLU on ALL channels (Q, K, V)
    for val in qkv_conv.iter_mut() {
        *val = silu(*val);
    }

    // Split QKV
    let mut q = qkv_conv[..q_dim].to_vec(); // [2048]
    let mut k = qkv_conv[q_dim..q_dim + k_dim].to_vec(); // [2048]
    let v = &qkv_conv[q_dim + k_dim..]; // [4096]

    // L2 normalize Q and K per head, then scale Q by 1/sqrt(head_dim)
    for h in 0..num_qk {
        l2_normalize(&mut q[h * hd..(h + 1) * hd]);
        l2_normalize(&mut k[h * hd..(h + 1) * hd]);
        for j in 0..hd {
            q[h * hd + j] *= scale;
        }
    }

    // Compute alpha and beta per V head from pre-computed a_t, b_t
    let mut alpha = vec![0.0f32; num_v];
    let mut beta = vec![0.0f32; num_v];
    for vh in 0..num_v {
        let a_exp = lw.a_log[vh].exp();
        let sp = softplus(a_t[vh] + lw.dt_bias[vh]);
        alpha[vh] = (-a_exp * sp).exp();
        beta[vh] = 1.0 / (1.0 + (-b_t[vh]).exp());
    }

    // GVA expansion: repeat Q and K from 16 to 32 heads
    let mut q_exp = vec![0.0f32; num_v * hd]; // [32, 128]
    let mut k_exp = vec![0.0f32; num_v * hd]; // [32, 128]
    for vh in 0..num_v {
        let qk_h = vh / v_per_qk;
        q_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&q[qk_h * hd..(qk_h + 1) * hd]);
        k_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&k[qk_h * hd..(qk_h + 1) * hd]);
    }

    // State update per V head (delta rule + gating)
    let mut y = vec![0.0f32; v_dim]; // [4096]
    let use_avx512 = crate::simd::has_avx512();

    for vh in 0..num_v {
        let al = alpha[vh];
        let be = beta[vh];
        let q_h = &q_exp[vh * hd..(vh + 1) * hd];
        let k_h = &k_exp[vh * hd..(vh + 1) * hd];
        let v_h = &v[vh * hd..(vh + 1) * hd];
        let s_base = vh * hd * hd;
        let s = &mut state.s[s_base..s_base + hd * hd];
        let y_base = vh * hd;

        if use_avx512 {
            unsafe {
                crate::simd::deltanet_head_update_avx512(
                    s,
                    q_h,
                    k_h,
                    v_h,
                    al,
                    be,
                    hd,
                    &mut y[y_base..y_base + hd],
                );
            }
            continue;
        }

        // Scalar fallback: decay state
        for val in s.iter_mut() {
            *val *= al;
        }

        // Retrieve from decayed state: pred = S @ k_h → [hd]
        let mut pred = vec![0.0f32; hd];
        for r in 0..hd {
            let mut dot = 0.0f32;
            for c in 0..hd {
                dot += s[r * hd + c] * k_h[c];
            }
            pred[r] = dot;
        }

        // Delta update: S += beta * (v_h - pred) @ k_h^T
        for r in 0..hd {
            let delta = be * (v_h[r] - pred[r]);
            for c in 0..hd {
                s[r * hd + c] += delta * k_h[c];
            }
        }

        // Output: y[vh*hd..(vh+1)*hd] = S @ q_h
        for r in 0..hd {
            let mut dot = 0.0f32;
            for c in 0..hd {
                dot += s[r * hd + c] * q_h[c];
            }
            y[y_base + r] = dot;
        }
    }

    // Output gating: per_head_rms_norm(y) * silu(z_t)
    let norm_gamma: Vec<f32> = lw.attn_norm_weight.iter().map(|v| v.to_f32()).collect();
    let mut y_normed = vec![0.0f32; v_dim];
    for h in 0..num_v {
        let base = h * hd;
        let head_slice = &y[base..base + hd];
        let sum_sq: f32 = head_slice.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
        for j in 0..hd {
            y_normed[base + j] = head_slice[j] * inv_rms * norm_gamma[j];
        }
    }

    let mut gated = vec![0.0f32; v_dim];
    for j in 0..v_dim {
        gated[j] = y_normed[j] * silu(z_t[j]);
    }

    gated
}

/// Batched fused gate+up: [N, hidden] -> [N, intermediate], applying SiLU(gate) * up per-row.
pub(crate) fn fused_gate_up_gemm(x: &[f32], n: usize, gate_w: &Weight, up_w: &Weight) -> Vec<f32> {
    let intermediate = gate_w.n();
    let hidden = gate_w.k();
    let mut out = vec![0.0f32; n * intermediate];
    out.par_chunks_mut(intermediate)
        .zip(x.par_chunks(hidden))
        .for_each(|(out_row, x_row)| {
            let row = fused_gate_up_gemv(x_row, gate_w, up_w);
            out_row.copy_from_slice(&row);
        });
    out
}

/// Process one token through a Gated DeltaNet layer.
/// Returns residual-added output [hidden].
pub(crate) fn forward_deltanet_decode(
    x: &[f32],
    lw: &DeltaNetLayerWeights,
    state: &mut DeltaNetState,
    config: &IncordConfig,
) -> Vec<f32> {
    let hidden = config.hidden_size;

    // 1. Pre-attention RmsNorm
    let x_norm = rms_norm(x, &lw.input_norm);

    // 2. QKV projection
    let qkv = gemv(&x_norm, &lw.in_proj_qkv); // [qkv_dim]

    // 3-9. Inner DeltaNet step: conv1d + state update + z gating
    let a_proj = gemv(&x_norm, &lw.in_proj_a); // [num_v]
    let b_proj = gemv(&x_norm, &lw.in_proj_b); // [num_v]
    let z_proj = gemv(&x_norm, &lw.in_proj_z); // [v_dim]
    let gated = deltanet_step(&qkv, &a_proj, &b_proj, &z_proj, lw, state, config);

    // 10. Output projection
    let attn_out = gemv(&gated, &lw.out_proj);

    // Residual
    let mut out = vec![0.0f32; hidden];
    for j in 0..hidden {
        out[j] = x[j] + attn_out[j];
    }

    // MLP
    let x_norm2 = rms_norm(&out, &lw.post_attn_norm);
    let intermediate = fused_gate_up_gemv(&x_norm2, &lw.gate_proj, &lw.up_proj);
    let mlp_out = gemv(&intermediate, &lw.down_proj);
    for j in 0..hidden {
        out[j] += mlp_out[j];
    }

    out
}

// ============================================================
// Full attention single-step forward
// ============================================================

/// Dot product between an f16 KV-cache slice and an f32 query vector.
/// Dequantizes each f16 element on the fly; no extra allocation.
#[inline]
fn dot_f16_f32(k: &[f16], q: &[f32]) -> f32 {
    k.iter().zip(q).map(|(&kv, &qv)| kv.to_f32() * qv).sum()
}

/// Run one token's attention given pre-projected q (with gate interleaved), k, v.
///
/// `q` layout: `[num_attn_heads * 2 * attn_head_dim]` — per head: `[Q:head_dim, gate:head_dim]`
/// `k` layout: `[num_kv_heads * attn_head_dim]`
/// `v` layout: `[num_kv_heads * attn_head_dim]`
///
/// Applies QK norm + RoPE at `rope_pos`, appends k/v to the cache (f16-quantizing),
/// computes GQA attention, applies output gating, and writes the gated attention
/// output into `out_row` (length `num_attn_heads * attn_head_dim`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_step_into(
    q_gate_full: &[f32], // [num_attn_heads * 2 * attn_head_dim] — Q + gate interleaved per head
    k_in: &[f32],        // [num_kv_heads * attn_head_dim]
    v_in: &[f32],        // [num_kv_heads * attn_head_dim]
    kv: &mut KvCacheEntry,
    rope_pos: usize,
    rope_cos: &[f32],
    rope_sin: &[f32],
    lw: &FullAttnLayerWeights,
    config: &IncordConfig,
    out_row: &mut [f32], // [num_attn_heads * attn_head_dim]
) {
    let num_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = num_heads * head_dim;
    let num_kv_groups = config.num_kv_groups();
    let rotary_dim = config.rope_dim();

    // Step 4: Split Q and gate per-head from the interleaved projection output if gated
    let is_gated = q_gate_full.len() == 2 * q_dim;
    let (mut q, gate_vec) = if is_gated {
        let mut q = vec![0.0f32; q_dim];
        let mut gate_vec = vec![0.0f32; q_dim];
        for h in 0..num_heads {
            let src = h * 2 * head_dim;
            let dst = h * head_dim;
            q[dst..dst + head_dim].copy_from_slice(&q_gate_full[src..src + head_dim]);
            gate_vec[dst..dst + head_dim]
                .copy_from_slice(&q_gate_full[src + head_dim..src + 2 * head_dim]);
        }
        (q, Some(gate_vec))
    } else {
        (q_gate_full.to_vec(), None)
    };

    let mut k_new = k_in.to_vec();

    // QK norm (only if weights present)
    if !lw.q_norm.is_empty() {
        per_head_rms_norm(&mut q, num_heads, head_dim, &lw.q_norm);
    }
    if !lw.k_norm.is_empty() {
        per_head_rms_norm(&mut k_new, num_kv_heads, head_dim, &lw.k_norm);
    }

    // Step 5: Partial RoPE
    apply_partial_rope(
        &mut q, num_heads, head_dim, rope_cos, rope_sin, rotary_dim, rope_pos,
    );
    apply_partial_rope(
        &mut k_new,
        num_kv_heads,
        head_dim,
        rope_cos,
        rope_sin,
        rotary_dim,
        rope_pos,
    );

    // Step 6: Append to KV cache (quantize f32 → f16 on write)
    kv.k.extend(k_new.iter().map(|&v| f16::from_f32(v)));
    kv.v.extend(v_in.iter().map(|&v| f16::from_f32(v)));
    kv.seq_len = rope_pos + 1;
    let kv_seq_len = kv.seq_len;

    // Step 7: GQA attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    let mut attn_output: Vec<f32> = if kv_seq_len >= 64 {
        let head_results: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / num_kv_groups;
                let q_off = h * head_dim;
                let q_vec = &q[q_off..q_off + head_dim];
                let mut scores = vec![0.0f32; kv_seq_len];
                for (s, score) in scores.iter_mut().enumerate().take(kv_seq_len) {
                    let k_off = s * kv_stride + kv_h * head_dim;
                    *score = dot_f16_f32(&kv.k[k_off..k_off + head_dim], q_vec) * scale;
                }
                softmax_raw(&mut scores);
                let mut head_out = vec![0.0f32; head_dim];
                for (s, &sc) in scores.iter().enumerate().take(kv_seq_len) {
                    let v_off = s * kv_stride + kv_h * head_dim;
                    for (d, ho) in head_out.iter_mut().enumerate().take(head_dim) {
                        *ho += sc * kv.v[v_off + d].to_f32();
                    }
                }
                head_out
            })
            .collect();
        head_results.into_iter().flatten().collect()
    } else {
        let mut attn_out = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / num_kv_groups;
            let q_off = h * head_dim;
            let q_vec = &q[q_off..q_off + head_dim];
            let mut scores = vec![0.0f32; kv_seq_len];
            for (s, score) in scores.iter_mut().enumerate().take(kv_seq_len) {
                let k_off = s * kv_stride + kv_h * head_dim;
                *score = dot_f16_f32(&kv.k[k_off..k_off + head_dim], q_vec) * scale;
            }
            softmax_raw(&mut scores);
            let out_off = h * head_dim;
            for (s, &sc) in scores.iter().enumerate().take(kv_seq_len) {
                let v_off = s * kv_stride + kv_h * head_dim;
                for d in 0..head_dim {
                    attn_out[out_off + d] += sc * kv.v[v_off + d].to_f32();
                }
            }
        }
        attn_out
    };

    // Step 8: Apply output gate: attn_output *= sigmoid(gate)
    if let Some(ref gate_vec) = gate_vec {
        for j in 0..q_dim {
            attn_output[j] *= 1.0 / (1.0 + (-gate_vec[j]).exp());
        }
    }

    out_row.copy_from_slice(&attn_output);
}

/// Process one token through a full attention (GQA) layer.
/// q_proj outputs [2 * q_dim] when attn_output_gate=true: first half Q, second half gate.
pub(crate) fn forward_attn_decode(
    x: &[f32],
    lw: &FullAttnLayerWeights,
    kv: &mut KvCacheEntry,
    config: &IncordConfig,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Vec<f32> {
    let hidden = config.hidden_size;
    let num_heads = config.num_attn_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = num_heads * head_dim;
    let offset = kv.seq_len;

    // Step 1: Pre-attention RmsNorm
    let x_norm = rms_norm(x, &lw.input_norm);

    // Steps 2-3: Q projection (interleaved Q+gate), K, V projections
    let q_gate_full = gemv(&x_norm, &lw.q_proj);
    let k_new = gemv(&x_norm, &lw.k_proj);
    let v_new = gemv(&x_norm, &lw.v_proj);

    // Steps 4-8: QK norm, RoPE, KV append, attention, output gating
    let mut attn_out = vec![0.0f32; q_dim];
    attn_step_into(
        &q_gate_full,
        &k_new,
        &v_new,
        kv,
        offset,
        rope_cos,
        rope_sin,
        lw,
        config,
        &mut attn_out,
    );

    // Step 9: O projection + residual
    let o_out = gemv(&attn_out, &lw.o_proj);
    let mut out = vec![0.0f32; hidden];
    for j in 0..hidden {
        out[j] = x[j] + o_out[j];
    }

    // Steps 10-13: Post-attn RMSNorm + fused gate+up MLP + down projection + residual
    let x_norm2 = rms_norm(&out, &lw.post_attn_norm);
    let intermediate = fused_gate_up_gemv(&x_norm2, &lw.gate_proj, &lw.up_proj);
    let mlp_out = gemv(&intermediate, &lw.down_proj);
    for j in 0..hidden {
        out[j] += mlp_out[j];
    }

    out
}

// ============================================================
// Hybrid forward decode (single token)
// ============================================================

/// Process one token through all layers (updates cache) but skip lm_head.
/// Use when the next token is already known (e.g. forced THINK_END).
pub fn forward_decode_no_logits(weights: &ModelWeights, token_id: usize, cache: &mut HybridCache) {
    let hidden = weights.hidden;
    let config = &weights.config;
    let mut x = embed_lookup(&weights.embed, token_id, hidden);
    for i in 0..config.num_layers {
        x = match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                forward_deltanet_decode(&x, lw, state, config)
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                forward_attn_decode(&x, lw, kv, config, &weights.rope_cos, &weights.rope_sin)
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        };
    }
    // Skip final_norm + lm_head — only cache updates matter
}

/// Process one token through the full hybrid model. Returns logits.
pub fn forward_decode(
    weights: &ModelWeights,
    token_id: usize,
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let mut x = embed_lookup(&weights.embed, token_id, hidden);
    for i in 0..config.num_layers {
        x = match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                forward_deltanet_decode(&x, lw, state, config)
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                forward_attn_decode(&x, lw, kv, config, &weights.rope_cos, &weights.rope_sin)
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        };
    }
    x = rms_norm(&x, &weights.final_norm);
    lm_head_parallel(&x, &weights.lm_head, weights.vocab, hidden)
}

// ============================================================
// Batched DeltaNet prefill (GEMM projections + sequential state)
// ============================================================

/// Prefill a DeltaNet layer with batched GEMM projections.
/// The state update is inherently sequential, but all matrix multiplications
/// are batched into GEMM ops that parallelize across tokens.
fn prefill_deltanet_layer(
    x: &mut [f32],
    seq_len: usize,
    lw: &DeltaNetLayerWeights,
    state: &mut DeltaNetState,
    config: &IncordConfig,
) {
    let hidden = config.hidden_size;
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = num_v * hd;
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let buf_len = ks - 1;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    // ── Step 1: RMSNorm all tokens ──
    let mut x_norm = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = rms_norm(&x[t * hidden..(t + 1) * hidden], &lw.input_norm);
        x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    // ── Step 2: Batch ALL input projections as GEMM ──
    let qkv_all = gemm(&x_norm, seq_len, &lw.in_proj_qkv);
    let a_all = gemm(&x_norm, seq_len, &lw.in_proj_a);
    let b_all = gemm(&x_norm, seq_len, &lw.in_proj_b);
    let z_all = gemm(&x_norm, seq_len, &lw.in_proj_z);

    // ── Step 3: Sequential loop — conv1d + state update only ──
    let mut y_all = vec![0.0f32; seq_len * v_dim];
    let conv_w = &lw.conv1d_weight;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    let mut q_exp = vec![0.0f32; num_v * hd];
    let mut k_exp = vec![0.0f32; num_v * hd];
    let mut pred = vec![0.0f32; hd];

    let a_exp_table: Vec<f32> = lw.a_log.iter().map(|v| v.exp()).collect();

    for t in 0..seq_len {
        let qkv_t = &qkv_all[t * qkv_dim..(t + 1) * qkv_dim];

        // Conv1d (causal, depthwise)
        for ch in 0..qkv_dim {
            let mut sum = 0.0f32;
            for k in 0..buf_len {
                sum += state.conv_buf[ch * buf_len + k] * conv_w[ch * ks + k];
            }
            sum += qkv_t[ch] * conv_w[ch * ks + buf_len];
            qkv_conv[ch] = sum;
        }
        for (ch, &qkv_val) in qkv_t.iter().enumerate().take(qkv_dim) {
            for k in 0..buf_len - 1 {
                state.conv_buf[ch * buf_len + k] = state.conv_buf[ch * buf_len + k + 1];
            }
            state.conv_buf[ch * buf_len + buf_len - 1] = qkv_val;
        }
        state.conv_pos += 1;

        for val in qkv_conv.iter_mut() {
            *val = silu(*val);
        }

        // L2 normalize Q and K per head in-place, scale Q
        for h in 0..num_qk {
            l2_normalize(&mut qkv_conv[h * hd..(h + 1) * hd]);
            l2_normalize(&mut qkv_conv[q_dim + h * hd..q_dim + (h + 1) * hd]);
            for j in 0..hd {
                qkv_conv[h * hd + j] *= scale;
            }
        }

        let a_t = &a_all[t * num_v..(t + 1) * num_v];
        let b_t = &b_all[t * num_v..(t + 1) * num_v];

        // GVA expansion
        for vh in 0..num_v {
            let qk_h = vh / v_per_qk;
            q_exp[vh * hd..(vh + 1) * hd].copy_from_slice(&qkv_conv[qk_h * hd..(qk_h + 1) * hd]);
            k_exp[vh * hd..(vh + 1) * hd]
                .copy_from_slice(&qkv_conv[q_dim + qk_h * hd..q_dim + (qk_h + 1) * hd]);
        }

        // State update per V head (delta rule)
        let v_slice = &qkv_conv[q_dim + k_dim..];
        for vh in 0..num_v {
            let al = (-a_exp_table[vh] * softplus(a_t[vh] + lw.dt_bias[vh])).exp();
            let be = 1.0 / (1.0 + (-b_t[vh]).exp());

            let q_h = &q_exp[vh * hd..(vh + 1) * hd];
            let k_h = &k_exp[vh * hd..(vh + 1) * hd];
            let v_h = &v_slice[vh * hd..(vh + 1) * hd];
            let s_base = vh * hd * hd;
            let s = &mut state.s[s_base..s_base + hd * hd];

            for val in s.iter_mut() {
                *val *= al;
            }

            for r in 0..hd {
                let mut dot = 0.0f32;
                let row = &s[r * hd..(r + 1) * hd];
                for c in 0..hd {
                    dot += row[c] * k_h[c];
                }
                pred[r] = dot;
            }

            for r in 0..hd {
                let delta = be * (v_h[r] - pred[r]);
                let row = &mut s[r * hd..(r + 1) * hd];
                for c in 0..hd {
                    row[c] += delta * k_h[c];
                }
            }

            let y_base = t * v_dim + vh * hd;
            for r in 0..hd {
                let mut dot = 0.0f32;
                let row = &s[r * hd..(r + 1) * hd];
                for c in 0..hd {
                    dot += row[c] * q_h[c];
                }
                y_all[y_base + r] = dot;
            }
        }
    }

    // ── Step 4: Per-head norm + z gate (batched) ──
    let norm_gamma: Vec<f32> = lw.attn_norm_weight.iter().map(|v| v.to_f32()).collect();
    for t in 0..seq_len {
        let z_t = &z_all[t * v_dim..(t + 1) * v_dim];
        let y_t = &mut y_all[t * v_dim..(t + 1) * v_dim];
        for h in 0..num_v {
            let base = h * hd;
            let sum_sq: f32 = y_t[base..base + hd].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
            for j in 0..hd {
                y_t[base + j] = y_t[base + j] * inv_rms * norm_gamma[j];
            }
        }
        for j in 0..v_dim {
            y_t[j] *= silu(z_t[j]);
        }
    }

    // ── Step 5: Output projection as GEMM + residual ──
    let attn_out = gemm(&y_all, seq_len, &lw.out_proj);
    for j in 0..seq_len * hidden {
        x[j] += attn_out[j];
    }

    // ── Step 6: MLP as GEMM ──
    let mut x_norm2 = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = rms_norm(&x[t * hidden..(t + 1) * hidden], &lw.post_attn_norm);
        x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }
    let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
    let up = gemm(&x_norm2, seq_len, &lw.up_proj);
    let inter_size = lw.gate_proj.n();
    let mut intermediate = vec![0.0f32; seq_len * inter_size];
    for j in 0..seq_len * inter_size {
        let g = gate[j];
        intermediate[j] = (g / (1.0 + (-g).exp())) * up[j];
    }
    let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);
    for j in 0..seq_len * hidden {
        x[j] += mlp_out[j];
    }
}

// ============================================================
// Hybrid prefill (full prompt)
// ============================================================

/// Prefill all tokens, return logits for last token.
pub fn prefill(weights: &ModelWeights, token_ids: &[u32], cache: &mut HybridCache) -> Vec<f32> {
    let hidden = weights.hidden;
    let seq_len = token_ids.len();
    let x = prefill_layers(weights, token_ids, cache);

    // Final norm (last token only)
    let last = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm(last, &weights.final_norm);
    lm_head_parallel(&normed, &weights.lm_head, weights.vocab, hidden)
}

/// Prefill returning the final-normed hidden state of EVERY position,
/// `[seq_len, hidden]`. Feed a row to `lm_head_parallel` for that position's
/// logits. Used by the quality evaluator, which scores every position.
pub fn prefill_hidden_all(
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let mut x = prefill_layers(weights, token_ids, cache);
    for row in x.chunks_mut(hidden) {
        let normed = rms_norm(row, &weights.final_norm);
        row.copy_from_slice(&normed);
    }
    x
}

/// Embedding + every layer over the whole prompt. Returns the residual stream
/// `[seq_len, hidden]` before the final norm.
fn prefill_layers(weights: &ModelWeights, token_ids: &[u32], cache: &mut HybridCache) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let seq_len = token_ids.len();

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row = embed_lookup(&weights.embed, tid as usize, hidden);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    let num_layers = config.num_layers;
    for i in 0..num_layers {
        eprint!("\r  Prefill: layer {}/{num_layers}  ", i + 1);
        match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                prefill_deltanet_layer(&mut x, seq_len, lw, state, config);
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                prefill_attn_layer(
                    &mut x,
                    seq_len,
                    lw,
                    kv,
                    config,
                    &weights.rope_cos,
                    &weights.rope_sin,
                );
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        }
    }
    eprint!("\r                              \r");
    x
}

/// Prefill with vision embeddings injected at IMAGE_PAD positions.
/// `vision_embeds` is [num_vision_tokens, hidden] from the vision encoder.
/// The token_ids should contain IMAGE_PAD tokens at the right positions.
pub fn prefill_with_vision(
    weights: &ModelWeights,
    token_ids: &[u32],
    vision_embeds: &[f32],
    num_vision_tokens: usize,
    cache: &mut HybridCache,
) -> Vec<f32> {
    let hidden = weights.hidden;
    let config = &weights.config;
    let seq_len = token_ids.len();

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    let mut vision_idx = 0usize;
    for (t, &tid) in token_ids.iter().enumerate() {
        if (tid == crate::tokenizer::IMAGE_PAD || tid == crate::tokenizer::VIDEO_PAD)
            && vision_idx < num_vision_tokens
        {
            let src = &vision_embeds[vision_idx * hidden..(vision_idx + 1) * hidden];
            x[t * hidden..(t + 1) * hidden].copy_from_slice(src);
            vision_idx += 1;
        } else {
            let row = embed_lookup(&weights.embed, tid as usize, hidden);
            x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
        }
    }
    if vision_idx != num_vision_tokens {
        eprintln!(
            "WARNING: Expected {num_vision_tokens} vision tokens but found {vision_idx} vision pad positions"
        );
    }

    let num_layers = config.num_layers;
    for i in 0..num_layers {
        eprint!("\r  Prefill: layer {}/{num_layers}  ", i + 1);
        match (&weights.layers[i], &mut cache.entries[i]) {
            (HybridLayerWeights::DeltaNet(lw), CacheEntry::DeltaNet(state)) => {
                prefill_deltanet_layer(&mut x, seq_len, lw, state, config);
            }
            (HybridLayerWeights::FullAttn(lw), CacheEntry::KvCache(kv)) => {
                prefill_attn_layer(
                    &mut x,
                    seq_len,
                    lw,
                    kv,
                    config,
                    &weights.rope_cos,
                    &weights.rope_sin,
                );
            }
            _ => panic!("Layer type mismatch at layer {i}"),
        }
    }
    eprint!("\r                              \r");

    // Final norm (last token only)
    let last = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm(last, &weights.final_norm);
    lm_head_parallel(&normed, &weights.lm_head, weights.vocab, hidden)
}

/// Prefill a full attention layer with all tokens at once (efficient GEMM).
/// q_proj outputs [2 * q_dim] when attn_output_gate=true.
fn prefill_attn_layer(
    x: &mut [f32],
    seq_len: usize,
    lw: &FullAttnLayerWeights,
    kv: &mut KvCacheEntry,
    config: &IncordConfig,
    rope_cos: &[f32],
    rope_sin: &[f32],
) {
    let hidden = config.hidden_size;
    let num_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.attn_head_dim;
    let q_dim = num_heads * head_dim; // 4096
    let num_kv_groups = config.num_kv_groups();
    let rotary_dim = config.rope_dim(); // 64: total dims rotated per head

    // Pre-attention RmsNorm per token
    let mut x_norm = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &x[t * hidden..(t + 1) * hidden];
        let normed = rms_norm(row, &lw.input_norm);
        x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
    }

    // Q projection: output is [seq_len, q_dim] (standard LLaMA) or [seq_len, 2*q_dim] (gated)
    let q_gate_all = gemm(&x_norm, seq_len, &lw.q_proj);
    let is_gated = lw.q_proj.n() == 2 * q_dim;

    let (mut q_all, gate_all) = if is_gated {
        let q_gate_stride = 2 * q_dim;
        let mut q_all = vec![0.0f32; seq_len * q_dim];
        let mut gate_all = vec![0.0f32; seq_len * q_dim];
        for t in 0..seq_len {
            let src_base = t * q_gate_stride;
            let q_base = t * q_dim;
            for h in 0..num_heads {
                let src = src_base + h * 2 * head_dim;
                let dst = q_base + h * head_dim;
                q_all[dst..dst + head_dim].copy_from_slice(&q_gate_all[src..src + head_dim]);
                gate_all[dst..dst + head_dim]
                    .copy_from_slice(&q_gate_all[src + head_dim..src + 2 * head_dim]);
            }
        }
        (q_all, Some(gate_all))
    } else {
        (q_gate_all, None)
    };

    // K, V GEMM
    let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
    let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

    // Per-head QK norm (only if present)
    if !lw.q_norm.is_empty() {
        for t in 0..seq_len {
            let q_start = t * q_dim;
            per_head_rms_norm(
                &mut q_all[q_start..q_start + q_dim],
                num_heads,
                head_dim,
                &lw.q_norm,
            );
        }
    }
    if !lw.k_norm.is_empty() {
        for t in 0..seq_len {
            let k_start = t * num_kv_heads * head_dim;
            per_head_rms_norm(
                &mut k_all[k_start..k_start + num_kv_heads * head_dim],
                num_kv_heads,
                head_dim,
                &lw.k_norm,
            );
        }
    }

    // RoPE per token
    for t in 0..seq_len {
        let q_start = t * q_dim;
        apply_partial_rope(
            &mut q_all[q_start..q_start + q_dim],
            num_heads,
            head_dim,
            rope_cos,
            rope_sin,
            rotary_dim,
            t,
        );
        let k_start = t * num_kv_heads * head_dim;
        apply_partial_rope(
            &mut k_all[k_start..k_start + num_kv_heads * head_dim],
            num_kv_heads,
            head_dim,
            rope_cos,
            rope_sin,
            rotary_dim,
            t,
        );
    }

    // Store in KV cache (quantize f32 → f16 on write)
    kv.k = k_all.iter().map(|&v| f16::from_f32(v)).collect();
    kv.v = v_all.iter().map(|&v| f16::from_f32(v)).collect();
    kv.seq_len = seq_len;

    // Causal attention — parallel across heads
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    let head_results: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            let kv_h = h / num_kv_groups;
            let mut head_out = vec![0.0f32; seq_len * head_dim];
            for t1 in 0..seq_len {
                let attend_len = t1 + 1;
                let q_off = t1 * q_dim + h * head_dim;
                let q_vec = &q_all[q_off..q_off + head_dim];
                let mut scores = vec![0.0f32; attend_len];
                for (t2, score) in scores.iter_mut().enumerate().take(attend_len) {
                    let k_off = t2 * kv_stride + kv_h * head_dim;
                    *score = dot_f16_f32(&kv.k[k_off..k_off + head_dim], q_vec) * scale;
                }
                softmax_raw(&mut scores);
                let out_base = t1 * head_dim;
                for (t2, &sc) in scores.iter().enumerate().take(attend_len) {
                    let v_off = t2 * kv_stride + kv_h * head_dim;
                    for d in 0..head_dim {
                        head_out[out_base + d] += sc * kv.v[v_off + d].to_f32();
                    }
                }
            }
            head_out
        })
        .collect();

    // Interleave head results
    let mut attn_output = vec![0.0f32; seq_len * q_dim];
    for (h, hr) in head_results.iter().enumerate() {
        for t in 0..seq_len {
            let src = &hr[t * head_dim..(t + 1) * head_dim];
            let dst = t * q_dim + h * head_dim;
            attn_output[dst..dst + head_dim].copy_from_slice(src);
        }
    }

    // Apply output gate: attn_output *= sigmoid(gate) per token
    if let Some(ref gate_all) = gate_all {
        for j in 0..seq_len * q_dim {
            attn_output[j] *= 1.0 / (1.0 + (-gate_all[j]).exp());
        }
    }

    // O projection + residual
    let o_out = gemm(&attn_output, seq_len, &lw.o_proj);
    for j in 0..seq_len * hidden {
        x[j] += o_out[j];
    }

    // Pre-MLP RmsNorm
    let mut x_norm2 = vec![0.0f32; seq_len * hidden];
    for t in 0..seq_len {
        let row = &x[t * hidden..(t + 1) * hidden];
        let normed = rms_norm(row, &lw.post_attn_norm);
        x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
    }

    // MLP
    let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
    let up = gemm(&x_norm2, seq_len, &lw.up_proj);
    let inter_size = lw.gate_proj.n();
    let mut intermediate = vec![0.0f32; seq_len * inter_size];
    for j in 0..seq_len * inter_size {
        let g = gate[j];
        intermediate[j] = (g / (1.0 + (-g).exp())) * up[j];
    }
    let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);

    for j in 0..seq_len * hidden {
        x[j] += mlp_out[j];
    }
}

// ============================================================
// Mock weights (tiny stub for tests)
// ============================================================

#[cfg(any(test, feature = "mock-weights"))]
pub mod mock {
    //! Tiny stub model weights for tests. Tiny dimensions, deterministic outputs.
    //! Allocates zero-filled F16 weights at every projection.

    use super::*;
    use crate::config::{LayerType, IncordConfig};
    use half::f16;

    /// Build a tiny config with the same fields as IncordConfig::default_4b
    /// but with small dimensions (suitable for unit-test speed).
    pub fn tiny_config() -> IncordConfig {
        let layer_types = vec![
            LayerType::DeltaNet,
            LayerType::DeltaNet,
            LayerType::DeltaNet,
            LayerType::FullAttn,
        ];

        IncordConfig {
            // 248_070 covers all real tokenizer IDs (max special token = 248_069).
            // The embedding table is vocab * hidden = 248_070 * 64 * 2B ≈ 31 MB —
            // acceptable for integration tests that exercise the real tokenizer.
            vocab_size: 248_070,
            hidden_size: 64,
            num_layers: 4,
            num_attn_heads: 2,
            num_kv_heads: 1,
            attn_head_dim: 32,
            num_qk_heads: 2,
            num_v_heads: 4,
            deltanet_head_dim: 16,
            conv_kernel_size: 4,
            intermediate_size: 128,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            eos_token_id: 0,
            bos_token_id: 0,
            max_position_embeddings: 512,
            tie_word_embeddings: true,
            layer_types,
            has_vision: false,
            vision_hidden: 64,
            vision_layers: 2,
            vision_heads: 2,
            vision_ffn: 128,
            patch_size: 4,
            spatial_merge_size: 2,
            num_position_embeddings: 512,
        }
    }

    /// Helper: zero-filled F16 weight of shape [k, n].
    fn f16_zeros(k: usize, n: usize) -> Weight {
        Weight::F16(F16Weight {
            data: vec![f16::from_f32(0.0); k * n],
            k,
            n,
        })
    }

    /// Build a tiny ModelWeights with all-zero F16 weights at every projection.
    /// Vision is None because tiny_config().has_vision = false.
    pub fn tiny_weights() -> ModelWeights {
        let config = tiny_config();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size;
        let qkv_dim = config.deltanet_qkv_dim();
        let num_v_heads = config.num_v_heads;
        let v_dim = num_v_heads * config.deltanet_head_dim;
        let kernel = config.conv_kernel_size;

        let mut layers = Vec::with_capacity(config.num_layers);
        for lt in &config.layer_types {
            match lt {
                LayerType::DeltaNet => {
                    layers.push(HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                        in_proj_qkv: f16_zeros(hidden, qkv_dim),
                        in_proj_a: f16_zeros(hidden, num_v_heads),
                        in_proj_b: f16_zeros(hidden, num_v_heads),
                        in_proj_z: f16_zeros(hidden, v_dim),
                        out_proj: f16_zeros(v_dim, hidden),
                        conv1d_weight: vec![0.0f32; qkv_dim * kernel],
                        a_log: vec![0.0f32; num_v_heads],
                        dt_bias: vec![0.0f32; num_v_heads],
                        attn_norm_weight: vec![f16::from_f32(1.0); config.deltanet_head_dim],
                        gate_proj: f16_zeros(hidden, intermediate),
                        up_proj: f16_zeros(hidden, intermediate),
                        down_proj: f16_zeros(intermediate, hidden),
                        input_norm: vec![f16::from_f32(1.0); hidden],
                        post_attn_norm: vec![f16::from_f32(1.0); hidden],
                    }));
                }
                LayerType::FullAttn => {
                    // q_proj outputs interleaved [Q, gate] per head: 2 x q_dim total
                    let q_dim = config.num_attn_heads * config.attn_head_dim;
                    let kv_dim = config.num_kv_heads * config.attn_head_dim;
                    layers.push(HybridLayerWeights::FullAttn(FullAttnLayerWeights {
                        q_proj: f16_zeros(hidden, q_dim * 2),
                        k_proj: f16_zeros(hidden, kv_dim),
                        v_proj: f16_zeros(hidden, kv_dim),
                        o_proj: f16_zeros(q_dim, hidden),
                        q_norm: vec![f16::from_f32(1.0); config.attn_head_dim],
                        k_norm: vec![f16::from_f32(1.0); config.attn_head_dim],
                        gate_proj: f16_zeros(hidden, intermediate),
                        up_proj: f16_zeros(hidden, intermediate),
                        down_proj: f16_zeros(intermediate, hidden),
                        input_norm: vec![f16::from_f32(1.0); hidden],
                        post_attn_norm: vec![f16::from_f32(1.0); hidden],
                    }));
                }
            }
        }

        // Embedding: vocab x hidden — embed_lookup uses token_id * hidden as row offset.
        let embed = f16_zeros(vocab, hidden);

        // RoPE tables sized for max_position_embeddings x rope_dim/2 entries per position.
        // 512 is enough to cover real chat prompts (system prompt + user message) in tests.
        let max_pos: usize = 512;
        let rope_dim = config.rope_dim();
        let rope_cos = vec![1.0f32; max_pos * (rope_dim / 2)];
        let rope_sin = vec![0.0f32; max_pos * (rope_dim / 2)];

        ModelWeights {
            layers,
            embed: embed.clone(),
            lm_head: embed,
            vocab,
            hidden,
            final_norm: vec![f16::from_f32(1.0); hidden],
            rope_cos,
            rope_sin,
            config,
            format_name: "F16-mock",
            vision: None,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tiny_weights_does_not_panic_on_forward_decode() {
            let w = tiny_weights();
            let mut cache = HybridCache::new(&w.config);
            // Valid token_id (vocab is small, use 1).
            let logits = forward_decode(&w, 1, &mut cache);
            assert_eq!(logits.len(), w.vocab);
        }

        #[test]
        fn tiny_weights_dimensions() {
            let w = tiny_weights();
            assert_eq!(w.hidden, 64);
            assert_eq!(w.vocab, 248_070);
            assert_eq!(w.layers.len(), 4);
            // 3 DeltaNet + 1 FullAttn
            let n_delta = w
                .layers
                .iter()
                .filter(|l| matches!(l, HybridLayerWeights::DeltaNet(_)))
                .count();
            let n_full = w
                .layers
                .iter()
                .filter(|l| matches!(l, HybridLayerWeights::FullAttn(_)))
                .count();
            assert_eq!(n_delta, 3);
            assert_eq!(n_full, 1);
        }

        #[test]
        fn test_gemv_q8_accuracy() {
            let k = 64;
            let n = 64;
            let mut data = vec![0.0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    data[i * n + j] = ((i * 7 + j * 13) % 100) as f32 / 50.0 - 1.0;
                }
            }
            let input: Vec<f32> = (0..k).map(|i| (i as f32) / (k as f32)).collect();

            // Expected f32 GEMV
            let mut expected = vec![0.0f32; n];
            for i in 0..k {
                for j in 0..n {
                    expected[j] += input[i] * data[i * n + j];
                }
            }

            // Quantize to Q8
            let (q8_data, scales) = quantize_q8(&data, k, n);
            let weight = Q8Weight {
                data: q8_data,
                scales,
                k,
                n,
            };

            let out = gemv_q8(&input, &weight);
            assert_eq!(out.len(), n);

            // Verify accuracy: Q8 has 127 quantization levels per group, max group difference < 0.05
            for j in 0..n {
                let diff = (out[j] - expected[j]).abs();
                assert!(diff < 0.05, "Mismatch at j={j}: out={}, expected={}, diff={diff}", out[j], expected[j]);
            }
        }

        #[test]
        fn test_embed_lookup_q8_accuracy() {
            let vocab = 4;
            let hidden = 64;
            let mut data = vec![0.0f32; vocab * hidden];
            for i in 0..vocab {
                for j in 0..hidden {
                    data[i * hidden + j] = (i as f32 + 1.0) * ((j as f32) * 0.1).sin();
                }
            }

            let (q8_data, scales) = quantize_q8(&data, vocab, hidden);
            let weight = Q8Weight {
                data: q8_data,
                scales,
                k: vocab,
                n: hidden,
            };

            for token_id in 0..vocab {
                let lookup = embed_lookup_q8(&weight, token_id);
                assert_eq!(lookup.len(), hidden);
                for j in 0..hidden {
                    let expected = data[token_id * hidden + j];
                    let diff = (lookup[j] - expected).abs();
                    assert!(diff < 0.05, "Embed mismatch at token {token_id}, dim {j}: got {}, expected {}, diff={diff}", lookup[j], expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod gemm_tiled_tests {
    use super::*;
    use crate::quant::{Rounding, quantize_q4, quantize_q8};

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// Tiled GEMM must equal running the single-token GEMV on every row.
    fn check(weight: &Weight) {
        let (k, n) = (weight.k(), weight.n());
        let seq = 37; // > GEMM_TILED_MIN_TOKENS, not a multiple of the tile
        assert!(seq >= GEMM_TILED_MIN_TOKENS);
        let x = data(seq * k, 7);
        let tiled = gemm(&x, seq, weight);
        for t in 0..seq {
            let row = gemv(&x[t * k..(t + 1) * k], weight);
            for j in 0..n {
                let (a, b) = (tiled[t * n + j], row[j]);
                assert!((a - b).abs() <= 1e-4 * (1.0 + b.abs()), "t={t} j={j}: {a} vs {b}");
            }
        }
    }

    // n = 320: not a multiple of the 128-column tile.
    const K: usize = 96;
    const N: usize = 320;

    #[test]
    fn tiled_f16_matches_gemv() {
        let w = data(K * N, 1).into_iter().map(f16::from_f32).collect();
        check(&Weight::F16(F16Weight { data: w, k: K, n: N }));
    }

    #[test]
    fn tiled_q4_matches_gemv() {
        let (packed, scales) = quantize_q4(&data(K * N, 2), K, N, Rounding::Mse);
        check(&Weight::Q4(Q4Weight { packed, scales, k: K, n: N }));
    }

    #[test]
    fn tiled_q8_matches_gemv() {
        let (data, scales) = quantize_q8(&data(K * N, 3), K, N, Rounding::Mse);
        check(&Weight::Q8(Q8Weight { data, scales, k: K, n: N }));
    }

    /// Batched head must equal the per-position head. Vocab 600 is not a
    /// multiple of the 256-row chunk.
    fn check_head(weight: &Weight) {
        let (vocab, hidden) = (weight.k(), weight.n());
        let n_rows = 5;
        let rows = data(n_rows * hidden, 9);
        let batch = lm_head_batch(&rows, n_rows, weight);
        for t in 0..n_rows {
            let one = lm_head_parallel(&rows[t * hidden..(t + 1) * hidden], weight, vocab, hidden);
            for v in 0..vocab {
                let (a, b) = (batch[t * vocab + v], one[v]);
                assert!((a - b).abs() <= 1e-4 * (1.0 + b.abs()), "t={t} v={v}: {a} vs {b}");
            }
        }
    }

    const VOCAB: usize = 600;
    const HIDDEN: usize = 96;

    #[test]
    fn head_batch_f16_matches() {
        let w = data(VOCAB * HIDDEN, 4).into_iter().map(f16::from_f32).collect();
        check_head(&Weight::F16(F16Weight { data: w, k: VOCAB, n: HIDDEN }));
    }

    #[test]
    fn head_batch_q4_matches() {
        let (packed, scales) = quantize_q4(&data(VOCAB * HIDDEN, 5), VOCAB, HIDDEN, Rounding::Mse);
        check_head(&Weight::Q4(Q4Weight { packed, scales, k: VOCAB, n: HIDDEN }));
    }

    /// A Rotated weight must give the same result as the plain weight on
    /// every compute path. Both store different f16 values (the rotated
    /// matrix rounds differently), so each carries its own ~1e-3 f16 noise;
    /// a wrong rotation would be off by O(1). The bound is absolute because
    /// outputs near zero come from cancellation.
    #[test]
    fn rotated_weight_matches_plain_on_every_path() {
        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len());
            let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(worst <= 1e-2, "{what}: max |diff| {worst}");
        };
        // Matmul weights [k=64, n=96] (k has one 64-wide Hadamard block).
        let (k, n) = (64, 96);
        let raw = data(k * n, 11);
        let plain = Weight::F16(F16Weight { data: raw.iter().map(|&v| f16::from_f32(v)).collect(), k, n });
        let mut rot = raw.clone();
        crate::rotate::rotate_columns(&mut rot, k, n);
        let rotw = Weight::Rotated(Box::new(Weight::F16(F16Weight {
            data: rot.iter().map(|&v| f16::from_f32(v)).collect(),
            k,
            n,
        })));
        let x = data(k, 12);
        // Exact f64 result of the unquantized weights: rotated and plain
        // paths must be equally close to it (proves the gap is f16 noise).
        let exact: Vec<f32> = (0..n)
            .map(|j| (0..k).map(|i| x[i] as f64 * raw[i * n + j] as f64).sum::<f64>() as f32)
            .collect();
        let err = |v: &[f32]| v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let (e_rot, e_plain) = (err(&gemv(&x, &rotw)), err(&gemv(&x, &plain)));
        assert!(e_rot <= 5e-3 && e_plain <= 5e-3, "vs exact: rotated {e_rot}, plain {e_plain}");
        close(&gemv(&x, &rotw), &gemv(&x, &plain), "gemv");
        let xs = data(37 * k, 13);
        close(&gemm(&xs, 37, &rotw), &gemm(&xs, 37, &plain), "gemm tiled");
        close(&gemm(&xs[..3 * k], 3, &rotw), &gemm(&xs[..3 * k], 3, &plain), "gemm short");
        close(
            &fused_gate_up_gemv(&x, &rotw, &rotw),
            &fused_gate_up_gemv(&x, &plain, &plain),
            "fused gate/up",
        );

        // Head / embedding [vocab=40, hidden=64], rows rotated.
        let (vocab, hidden) = (40, 64);
        let raw = data(vocab * hidden, 14);
        let plain = Weight::F16(F16Weight { data: raw.iter().map(|&v| f16::from_f32(v)).collect(), k: vocab, n: hidden });
        let mut rot = raw.clone();
        crate::rotate::rotate_rows(&mut rot, hidden);
        let rotw = Weight::Rotated(Box::new(Weight::F16(F16Weight {
            data: rot.iter().map(|&v| f16::from_f32(v)).collect(),
            k: vocab,
            n: hidden,
        })));
        let h = data(hidden, 15);
        close(
            &lm_head_parallel(&h, &rotw, vocab, hidden),
            &lm_head_parallel(&h, &plain, vocab, hidden),
            "head",
        );
        let hs = data(3 * hidden, 16);
        close(&lm_head_batch(&hs, 3, &rotw), &lm_head_batch(&hs, 3, &plain), "head batch");
        for tok in [0, 7, 39] {
            close(&embed_lookup(&rotw, tok, hidden), &embed_lookup(&plain, tok, hidden), "embed");
        }
    }

    #[test]
    fn head_batch_q8_matches() {
        let (data, scales) = quantize_q8(&data(VOCAB * HIDDEN, 6), VOCAB, HIDDEN, Rounding::Mse);
        check_head(&Weight::Q8(Q8Weight { data, scales, k: VOCAB, n: HIDDEN }));
    }
}
