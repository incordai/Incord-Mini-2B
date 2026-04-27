//! Batched forward pass for QORA-4B hybrid architecture.
//!
//! Wraps existing kernels in `gemv.rs` so that one forward step processes
//! N concurrent slots together, reading the model weights only once per step.
//!
//! Invariant (bitwise / ULP tolerance for f16-cached layers):
//!   forward_decode_batch with N=1 must equal forward_decode.

use crate::gemv::HybridCache;

pub struct SlotInferenceState {
    pub cache: HybridCache,
    /// Sequence position used for RoPE lookup on the next token.
    pub rope_pos: usize,
    /// Last token emitted (input to the next forward step).
    pub last_token: u32,
}

impl std::fmt::Debug for SlotInferenceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlotInferenceState")
            .field("cache_layers", &self.cache.entries.len())
            .field("rope_pos", &self.rope_pos)
            .field("last_token", &self.last_token)
            .finish()
    }
}

impl SlotInferenceState {
    pub fn new(config: &crate::config::Qor4bConfig) -> Self {
        Self {
            cache: HybridCache::new(config),
            rope_pos: 0,
            last_token: 0,
        }
    }
}

use crate::gemv::{CacheEntry, HybridLayerWeights, ModelWeights};
#[cfg(test)]
use crate::gemv::forward_decode;
use rayon::prelude::*;

pub fn forward_decode_batch(
    weights: &ModelWeights,
    slots: &mut [&mut SlotInferenceState],
) -> Vec<Vec<f32>> {
    let n = slots.len();
    if n == 0 { return Vec::new(); }

    let config = &weights.config;
    let hidden = weights.hidden;

    // Gather per-slot inputs
    let tokens: Vec<u32> = slots.iter().map(|s| s.last_token).collect();
    let rope_positions: Vec<usize> = slots.iter().map(|s| s.rope_pos).collect();

    // Embed
    let mut x = batched_embed_lookup(&weights.embed, &tokens, hidden);

    // Per-layer
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                // Collect mut refs to each slot's DeltaNet state at this layer
                let mut states: Vec<&mut crate::gemv::DeltaNetState> = slots.iter_mut()
                    .map(|s| match &mut s.cache.entries[layer_idx] {
                        CacheEntry::DeltaNet(d) => d,
                        _ => panic!("layer/cache type mismatch at layer {layer_idx}"),
                    })
                    .collect();
                x = forward_deltanet_decode_batch(&x, n, lw, &mut states, config);
            }
            HybridLayerWeights::FullAttn(lw) => {
                let mut kvs: Vec<&mut KvCacheEntry> = slots.iter_mut()
                    .map(|s| match &mut s.cache.entries[layer_idx] {
                        CacheEntry::KvCache(k) => k,
                        _ => panic!("layer/cache type mismatch at layer {layer_idx}"),
                    })
                    .collect();
                x = forward_fullattn_decode_batch(
                    &x, n, lw, &mut kvs,
                    &rope_positions, &weights.rope_cos, &weights.rope_sin, config,
                );
            }
        }
    }

    // Bump rope positions for next decode
    for s in slots.iter_mut() { s.rope_pos += 1; }

    // Final norm + lm_head
    batched_lm_head(&x, n, &weights.final_norm, &weights.embed, weights.vocab, hidden)
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod tests {
    use super::*;
    use crate::gemv::mock::tiny_weights;

    #[test]
    fn n_1_batch_equals_single() {
        let w = tiny_weights();
        let mut state_a = SlotInferenceState::new(&w.config);
        let mut state_b = SlotInferenceState::new(&w.config);
        state_a.last_token = 5;
        state_b.last_token = 5;

        let single = forward_decode(&w, 5, &mut state_a.cache);
        let batched = forward_decode_batch(&w, &mut [&mut state_b]);

        assert_eq!(single.len(), batched[0].len());
        for (i, (a, b)) in single.iter().zip(batched[0].iter()).enumerate() {
            // f16 KV cache: per-multiply error ~1e-3, accumulated over head_dim=32 ~1e-2 worst case
            assert!((a - b).abs() < 5e-3, "logit {i} mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn n_8_batch_each_independent() {
        let w = tiny_weights();
        let mut singles: Vec<Vec<f32>> = Vec::new();
        for i in 0..8u32 {
            let mut s = SlotInferenceState::new(&w.config);
            s.last_token = i;
            let l = forward_decode(&w, i as usize, &mut s.cache);
            singles.push(l);
        }

        let mut batched_states: Vec<SlotInferenceState> = (0..8u32)
            .map(|i| {
                let mut s = SlotInferenceState::new(&w.config);
                s.last_token = i;
                s
            })
            .collect();

        let mut refs: Vec<&mut SlotInferenceState> = batched_states.iter_mut().collect();
        let batched = forward_decode_batch(&w, &mut refs);

        for (i, (a, b)) in singles.iter().zip(batched.iter()).enumerate() {
            for (j, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
                // f16 KV cache: per-multiply error ~1e-3, accumulated over head_dim=32 ~1e-2 worst case
                assert!((av - bv).abs() < 5e-3, "slot {i} logit {j}: {av} vs {bv}");
            }
        }
    }
}

use crate::gemv::Weight;

// ============================================================
// Batched DeltaNet decode
// ============================================================

use crate::gemv::{DeltaNetLayerWeights, DeltaNetState, deltanet_step, fused_gate_up_gemm, gemm, rms_norm};

/// Batched DeltaNet decode: process N slots' single-token inputs through one DeltaNet layer.
/// Returns [N, hidden] residual outputs (input + attn_out + mlp_out).
pub fn forward_deltanet_decode_batch(
    x_batch: &[f32],
    n: usize,
    lw: &DeltaNetLayerWeights,
    states: &mut [&mut DeltaNetState],
    config: &crate::config::Qor4bConfig,
) -> Vec<f32> {
    debug_assert_eq!(x_batch.len(), n * config.hidden_size);
    debug_assert_eq!(states.len(), n);

    let hidden = config.hidden_size;
    let num_v_heads = config.num_v_heads;
    let head_dim = config.deltanet_head_dim;
    let v_dim = num_v_heads * head_dim;
    let qkv_dim = config.deltanet_qkv_dim();
    let a_dim = num_v_heads; // in_proj_a output size
    let b_dim = num_v_heads; // in_proj_b output size

    // Step 1: per-slot RMSNorm
    let mut x_norm = vec![0.0f32; n * hidden];
    for i in 0..n {
        let r = rms_norm(&x_batch[i * hidden..(i + 1) * hidden], &lw.input_norm);
        x_norm[i * hidden..(i + 1) * hidden].copy_from_slice(&r);
    }

    // Step 2: batched projections (one GEMM each)
    let qkv_all = gemm(&x_norm, n, &lw.in_proj_qkv); // [N, qkv_dim]
    let a_all   = gemm(&x_norm, n, &lw.in_proj_a);   // [N, num_v_heads]
    let b_all   = gemm(&x_norm, n, &lw.in_proj_b);   // [N, num_v_heads]
    let z_all   = gemm(&x_norm, n, &lw.in_proj_z);   // [N, v_dim]

    // Step 3: per-slot state update via deltanet_step (parallel across slots)
    // Each slot's recurrent state is independent, so this is embarrassingly parallel.
    let mut y_all = vec![0.0f32; n * v_dim];
    // Use indexed parallel iteration: zip (index, out_chunk, state_mut)
    {
        use rayon::prelude::IndexedParallelIterator;
        (0..n)
            .into_par_iter()
            .zip(y_all.par_chunks_mut(v_dim))
            .zip(states.par_iter_mut())
            .for_each(|((i, out_slice), state)| {
                let qkv_t = &qkv_all[i * qkv_dim..(i + 1) * qkv_dim];
                let a_t   = &a_all[i * a_dim..(i + 1) * a_dim];
                let b_t   = &b_all[i * b_dim..(i + 1) * b_dim];
                let z_t   = &z_all[i * v_dim..(i + 1) * v_dim];
                let gated = deltanet_step(qkv_t, a_t, b_t, z_t, lw, state, config);
                out_slice.copy_from_slice(&gated);
            });
    }

    // Step 4: batched output projection
    let proj = gemm(&y_all, n, &lw.out_proj); // [N, hidden]

    // Step 5: first residual: x + attn_out, then post-attn RMSNorm (mirrors single path).
    // Single path: out = x + attn_out; x_norm2 = rms_norm(&out, post_attn_norm)
    let mut x_post = vec![0.0f32; n * hidden];
    for i in 0..n {
        let residual_start = i * hidden;
        let residual_end = residual_start + hidden;
        // Compute x + attn_out for this slot
        let mut residual = vec![0.0f32; hidden];
        for j in 0..hidden {
            residual[j] = x_batch[residual_start + j] + proj[residual_start + j];
        }
        let r = rms_norm(&residual, &lw.post_attn_norm);
        x_post[residual_start..residual_end].copy_from_slice(&r);
    }

    // Step 6: batched fused gate+up MLP
    let mlp_inter = fused_gate_up_gemm(&x_post, n, &lw.gate_proj, &lw.up_proj);

    // Step 7: batched down projection
    let mlp_out = gemm(&mlp_inter, n, &lw.down_proj);

    // Step 8: final residual: (x + attn_out) + mlp_out
    // The first residual (x + proj) is already captured per-slot above; we
    // reconstruct it here so the final output matches the single-slot path:
    //   out = x + attn_out + mlp_out
    let mut out = vec![0.0f32; n * hidden];
    for i in 0..n * hidden {
        out[i] = x_batch[i] + proj[i] + mlp_out[i];
    }
    out
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod deltanet_batch_tests {
    use super::*;
    use crate::config::LayerType;
    use crate::gemv::mock::tiny_weights;
    use crate::gemv::{DeltaNetState, HybridLayerWeights, embed_lookup, forward_deltanet_decode};

    #[test]
    fn deltanet_batch_n1_equals_single() {
        let w = tiny_weights();
        let layer_idx = w.config.layer_types.iter()
            .position(|l| matches!(l, LayerType::DeltaNet))
            .expect("tiny_config has DeltaNet layers");

        let token_emb = embed_lookup(&w.embed, 1, w.hidden);

        let lw = match &w.layers[layer_idx] {
            HybridLayerWeights::DeltaNet(d) => d,
            _ => unreachable!(),
        };

        // Single-slot reference
        let mut state_single = DeltaNetState::new(&w.config);
        let single_out = forward_deltanet_decode(&token_emb, lw, &mut state_single, &w.config);

        // Batched N=1
        let mut state_batched = DeltaNetState::new(&w.config);
        let batched = forward_deltanet_decode_batch(
            &token_emb, 1, lw, &mut [&mut state_batched], &w.config,
        );

        assert_eq!(single_out.len(), batched.len());
        for (a, b) in single_out.iter().zip(batched.iter()) {
            assert!((a - b).abs() < 5e-4, "deltanet diverged: {a} vs {b}");
        }
    }
}

// ============================================================
// Batched FullAttn decode
// ============================================================

use crate::gemv::{FullAttnLayerWeights, KvCacheEntry, attn_step_into};

/// Batched FullAttn decode: N slots' single-token inputs through one FullAttn layer.
/// Returns [N, hidden] residual outputs.
pub fn forward_fullattn_decode_batch(
    x_batch: &[f32],            // [N, hidden]
    n: usize,
    lw: &FullAttnLayerWeights,
    kvs: &mut [&mut KvCacheEntry],
    rope_positions: &[usize],
    rope_cos: &[f32],
    rope_sin: &[f32],
    config: &crate::config::Qor4bConfig,
) -> Vec<f32> {
    debug_assert_eq!(rope_positions.len(), n);
    debug_assert_eq!(kvs.len(), n);

    let hidden = config.hidden_size;
    let head_dim = config.attn_head_dim;
    let num_q = config.num_attn_heads;
    let num_kv = config.num_kv_heads;
    // Q projection includes interleaved gate: output dim is num_q * 2 * head_dim
    let q_dim_with_gate = num_q * 2 * head_dim;
    let kv_dim = num_kv * head_dim;
    let attn_out_dim = num_q * head_dim;

    // Step 1: per-slot RMSNorm
    let mut x_norm = vec![0.0f32; n * hidden];
    for i in 0..n {
        let r = rms_norm(&x_batch[i * hidden..(i + 1) * hidden], &lw.input_norm);
        x_norm[i * hidden..(i + 1) * hidden].copy_from_slice(&r);
    }

    // Step 2: batched Q/K/V projections (single GEMM each)
    let q_all = gemm(&x_norm, n, &lw.q_proj); // [N, q_dim_with_gate]
    let k_all = gemm(&x_norm, n, &lw.k_proj); // [N, kv_dim]
    let v_all = gemm(&x_norm, n, &lw.v_proj); // [N, kv_dim]

    // Step 3: per-slot attention via attn_step_into (parallelized across slots)
    // Each slot's KV cache is independent, so this is embarrassingly parallel.
    let mut attn_out = vec![0.0f32; n * attn_out_dim];
    {
        use rayon::prelude::IndexedParallelIterator;
        (0..n)
            .into_par_iter()
            .zip(attn_out.par_chunks_mut(attn_out_dim))
            .zip(kvs.par_iter_mut())
            .for_each(|((i, out_row), kv)| {
                let q = &q_all[i * q_dim_with_gate..(i + 1) * q_dim_with_gate];
                let k = &k_all[i * kv_dim..(i + 1) * kv_dim];
                let v = &v_all[i * kv_dim..(i + 1) * kv_dim];
                let pos = rope_positions[i];
                attn_step_into(q, k, v, *kv, pos, rope_cos, rope_sin, lw, config, out_row);
            });
    }

    // Step 4: batched output projection
    let proj = gemm(&attn_out, n, &lw.o_proj); // [N, hidden]

    // Step 5: first residual: x + o_proj, then post-attn RMSNorm (mirrors single path).
    // Single path: out = x + o_out; x_norm2 = rms_norm(&out, post_attn_norm)
    let mut x_post = vec![0.0f32; n * hidden];
    for i in 0..n {
        let residual_start = i * hidden;
        // Compute x + o_proj for this slot
        let mut residual = vec![0.0f32; hidden];
        for j in 0..hidden {
            residual[j] = x_batch[residual_start + j] + proj[residual_start + j];
        }
        let r = rms_norm(&residual, &lw.post_attn_norm);
        x_post[residual_start..residual_start + hidden].copy_from_slice(&r);
    }

    // Step 6: batched fused gate+up MLP + down projection
    let mlp_inter = fused_gate_up_gemm(&x_post, n, &lw.gate_proj, &lw.up_proj);
    let mlp_out = gemm(&mlp_inter, n, &lw.down_proj);

    // Step 7: final residual: (x + o_proj) + mlp_out = x + o_proj + mlp_out
    let mut out = vec![0.0f32; n * hidden];
    for i in 0..n * hidden {
        out[i] = x_batch[i] + proj[i] + mlp_out[i];
    }
    out
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod fullattn_batch_tests {
    use super::*;
    use crate::config::LayerType;
    use crate::gemv::mock::tiny_weights;
    use crate::gemv::{KvCacheEntry, HybridLayerWeights, embed_lookup, forward_attn_decode};

    #[test]
    fn fullattn_batch_n1_equals_single() {
        let w = tiny_weights();
        let layer_idx = w.config.layer_types.iter()
            .position(|l| matches!(l, LayerType::FullAttn))
            .expect("tiny_config has 1 FullAttn layer");

        let token_emb = embed_lookup(&w.embed, 2, w.hidden);
        let lw = match &w.layers[layer_idx] {
            HybridLayerWeights::FullAttn(f) => f,
            _ => unreachable!(),
        };

        // Single-slot reference
        let mut kv_a = KvCacheEntry { k: Vec::new(), v: Vec::new(), seq_len: 0 };
        let single = forward_attn_decode(&token_emb, lw, &mut kv_a, &w.config, &w.rope_cos, &w.rope_sin);

        // Batched N=1 with fresh KV cache
        let mut kv_b = KvCacheEntry { k: Vec::new(), v: Vec::new(), seq_len: 0 };
        let batched = forward_fullattn_decode_batch(
            &token_emb, 1, lw, &mut [&mut kv_b],
            &[0], &w.rope_cos, &w.rope_sin, &w.config,
        );

        assert_eq!(single.len(), batched.len());
        // Tolerance accounts for f16 KV (~1e-3 per multiply) and head_dim accumulation
        for (a, b) in single.iter().zip(batched.iter()) {
            assert!((a - b).abs() < 5e-3, "fullattn diverged: {a} vs {b}");
        }
    }
}

// ============================================================
// Batched lm_head + final norm
// ============================================================

/// Final RMSNorm + lm_head, batched across N slots. Returns N logits vectors.
pub fn batched_lm_head(
    x_batch: &[f32],   // [N, hidden]
    n: usize,
    final_norm: &[half::f16],
    embed: &Weight,
    vocab: usize,
    hidden: usize,
) -> Vec<Vec<f32>> {
    debug_assert_eq!(x_batch.len(), n * hidden);

    // Per-slot RMSNorm (cheap; not worth batching)
    let mut x_norm = vec![0.0f32; n * hidden];
    for i in 0..n {
        let r = crate::gemv::rms_norm(&x_batch[i * hidden..(i + 1) * hidden], final_norm);
        x_norm[i * hidden..(i + 1) * hidden].copy_from_slice(&r);
    }

    // Batched lm_head: N stacked hidden states × tied embedding -> [N, vocab].
    // The embed weight is stored as [vocab, hidden] (rows are vocab entries).
    // For lm_head, we need to compute hidden · embed[token]^T for each token; this is the
    // same operation as gemm where we use embed as the projection.
    //
    // The existing lm_head_parallel does this for one row. We have a few choices:
    // 1. Call gemm on the [N, hidden] matrix with embed as the weight if dimensions align.
    // 2. Loop over N rows calling lm_head_parallel for each, parallelized via rayon.
    //
    // Option 2 is safer because lm_head_parallel may have specific memory access patterns
    // (e.g., chunked writes to a parallel-accessible vector) that gemm doesn't replicate.
    //
    // We use option 2 here. The gain over single calls is parallelism across slots — gemm
    // already parallelizes over output rows so we don't lose much by going slot-by-slot.

    use rayon::prelude::*;
    let mut logits: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
    logits.par_iter_mut().enumerate().for_each(|(i, slot_logits)| {
        let row = &x_norm[i * hidden..(i + 1) * hidden];
        *slot_logits = crate::gemv::lm_head_parallel(row, embed, vocab, hidden);
    });
    logits
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod lm_head_batch_tests {
    use super::*;
    use crate::gemv::mock::tiny_weights;

    #[test]
    fn batched_lm_head_matches_single() {
        let w = tiny_weights();
        let n = 3;
        let mut x = vec![0.0f32; n * w.hidden];
        for i in 0..n * w.hidden {
            x[i] = 0.01 * i as f32;
        }

        // Single-slot reference
        let single: Vec<Vec<f32>> = (0..n).map(|i| {
            let row = &x[i * w.hidden..(i + 1) * w.hidden];
            let normed = crate::gemv::rms_norm(row, &w.final_norm);
            crate::gemv::lm_head_parallel(&normed, &w.embed, w.vocab, w.hidden)
        }).collect();

        let batched = batched_lm_head(&x, n, &w.final_norm, &w.embed, w.vocab, w.hidden);

        for i in 0..n {
            assert_eq!(single[i].len(), batched[i].len());
            for j in 0..w.vocab {
                assert!((single[i][j] - batched[i][j]).abs() < 1e-3,
                    "slot {i} vocab {j}: {} vs {}", single[i][j], batched[i][j]);
            }
        }
    }
}

/// Embed N tokens. Returns a [N, hidden] flattened vector (f32).
pub fn batched_embed_lookup(
    embed: &Weight,
    token_ids: &[u32],
    hidden: usize,
) -> Vec<f32> {
    let n = token_ids.len();
    let mut out = vec![0.0f32; n * hidden];

    // Parallelize the gathers — each token is an independent vocab row read.
    out.par_chunks_mut(hidden)
        .zip(token_ids.par_iter())
        .for_each(|(slot_out, &tid)| {
            let row = crate::gemv::embed_lookup(embed, tid as usize, hidden);
            slot_out.copy_from_slice(&row);
        });

    out
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod embed_tests {
    use super::*;
    use crate::gemv::mock::tiny_weights;

    #[test]
    fn batched_embed_matches_single() {
        let w = tiny_weights();
        let tokens = [3u32, 7, 11];
        let batched = batched_embed_lookup(&w.embed, &tokens, w.hidden);
        for (i, &tid) in tokens.iter().enumerate() {
            let single = crate::gemv::embed_lookup(&w.embed, tid as usize, w.hidden);
            let slice = &batched[i * w.hidden..(i + 1) * w.hidden];
            assert_eq!(slice, single.as_slice(), "embed mismatch at slot {i} token {tid}");
        }
    }

    #[test]
    fn batched_embed_empty_input() {
        let w = tiny_weights();
        let out = batched_embed_lookup(&w.embed, &[], w.hidden);
        assert!(out.is_empty());
    }
}
