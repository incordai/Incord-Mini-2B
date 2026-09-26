//! GPU inference for Incord on cortex-v2.
//!
//! Prefill and single-token decode run every layer on the GPU for the hybrid
//! DeltaNet + full-attention architecture. Each step mirrors the CPU path in
//! `gemv.rs` (the reference): plain-gamma RMSNorm, gated or plain attention
//! detected from q_proj's width, QK norms only when the model has them, and
//! split-half partial RoPE. The output head runs on the GPU too; only the
//! embedding row lookup stays on the CPU (`embed_lookup`).

use cortex::tensor::activation;
use cortex::tensor::{Device, Tensor, TensorData};
use std::io::Write;
use std::time::Instant;

use crate::config::IncordConfig;
use crate::gemv::{ModelWeights, embed_lookup};
use crate::generate::GenerateParams;
use crate::gpu_loader::{GpuDeltaNetLayer, GpuFullAttnLayer, GpuHybridLayer, GpuModel, load_model_gpu};
use crate::tokenizer::IncordTokenizer;

// ============================================================
// GPU cache
// ============================================================

pub struct GpuHybridCache {
    pub entries: Vec<GpuCacheEntry>,
}

pub enum GpuCacheEntry {
    DeltaNet(GpuDeltaNetState),
    KvCache(GpuKvCacheEntry),
}

pub struct GpuDeltaNetState {
    /// State matrices `[num_v_heads, head_dim, head_dim]`.
    pub s: Tensor<3>,
    /// Conv buffer `[qkv_dim, kernel_size - 1]`.
    pub conv_buf: Tensor<2>,
    pub conv_pos: usize,
}

pub struct GpuKvCacheEntry {
    pub k: Option<Tensor<3>>, // [kv_heads, seq_len, head_dim]
    pub v: Option<Tensor<3>>,
    pub seq_len: usize,
}

impl GpuHybridCache {
    pub fn new(config: &IncordConfig, device: &Device) -> Self {
        let entries = config
            .layer_types
            .iter()
            .map(|lt| match lt {
                crate::config::LayerType::DeltaNet => {
                    let hd = config.deltanet_head_dim;
                    GpuCacheEntry::DeltaNet(GpuDeltaNetState {
                        s: Tensor::zeros([config.num_v_heads, hd, hd], device),
                        conv_buf: Tensor::zeros([config.deltanet_qkv_dim(), config.conv_kernel_size - 1], device),
                        conv_pos: 0,
                    })
                }
                crate::config::LayerType::FullAttn => {
                    GpuCacheEntry::KvCache(GpuKvCacheEntry { k: None, v: None, seq_len: 0 })
                }
            })
            .collect();
        Self { entries }
    }
}

// ============================================================
// Helper ops
// ============================================================

/// Row-wise RMSNorm with plain gamma, eps 1e-6 (`gemv::rms_norm`).
fn rms_norm<const D: usize>(x: Tensor<D>, gamma: &Tensor<1>) -> Tensor<D> {
    let rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + 1e-6).sqrt();
    (x / rms) * gamma.clone().unsqueeze::<D>()
}

/// L2-normalize each row: x / (||x|| + 1e-12).
fn l2_normalize_rows(x: Tensor<2>) -> Tensor<2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt() + 1e-12;
    x / norm
}

/// Softplus: log(1 + exp(x)).
fn softplus_1d(x: Tensor<1>) -> Tensor<1> {
    (x.exp() + 1.0).log()
}

/// SwiGLU MLP with its pre-norm and residual.
fn mlp_block(out: Tensor<2>, norm: &Tensor<1>, gate: &Tensor<2>, up: &Tensor<2>, down: &Tensor<2>) -> Tensor<2> {
    let x_norm = rms_norm(out.clone(), norm);
    let g = x_norm.clone().matmul(gate.clone());
    let u = x_norm.matmul(up.clone());
    out + activation::silu(g).mul(u).matmul(down.clone())
}

/// Split q_proj output `[rows, width]` into Q `[rows, q_dim]` and, for gated
/// attention, the gate `[rows, q_dim]` (per head: `[Q(head_dim), gate(head_dim)]`).
fn split_q_gate(q_out: Tensor<2>, layer: &GpuFullAttnLayer, rows: usize, heads: usize, head_dim: usize) -> (Tensor<2>, Option<Tensor<2>>) {
    let q_dim = heads * head_dim;
    if !layer.gated {
        return (q_out, None);
    }
    let q4 = q_out.reshape([rows, heads, 2, head_dim]);
    let q = q4.clone().slice([0..rows, 0..heads, 0..1, 0..head_dim]).reshape([rows, q_dim]);
    let gate = q4.slice([0..rows, 0..heads, 1..2, 0..head_dim]).reshape([rows, q_dim]);
    (q, Some(gate))
}

// ============================================================
// DeltaNet: single token
// ============================================================

fn forward_deltanet_decode_gpu(x: Tensor<2>, layer: &GpuDeltaNetLayer, state: &mut GpuDeltaNetState, config: &IncordConfig) -> Tensor<2> {
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = config.deltanet_v_dim();
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    let x_norm = rms_norm(x.clone(), &layer.input_norm_gamma);
    let qkv = x_norm.clone().matmul(layer.in_proj_qkv.clone());

    // Depthwise causal conv over the last `ks` inputs of every channel.
    let conv_input = Tensor::cat(vec![state.conv_buf.clone(), qkv.reshape([qkv_dim, 1])], 1);
    let qkv_conv = (conv_input.clone() * layer.conv1d_weight.clone()).sum_dim(1).reshape([1, qkv_dim]);
    state.conv_buf = conv_input.slice([0..qkv_dim, 1..ks]);
    state.conv_pos += 1;

    let qkv_act = activation::silu(qkv_conv);
    let q = qkv_act.clone().slice([0..1, 0..q_dim]).reshape([num_qk, hd]);
    let k = qkv_act.clone().slice([0..1, q_dim..q_dim + k_dim]).reshape([num_qk, hd]);
    let v = qkv_act.slice([0..1, q_dim + k_dim..qkv_dim]).reshape([num_v, hd]);
    let q = l2_normalize_rows(q).mul_scalar(scale);
    let k = l2_normalize_rows(k);

    let a_proj = x_norm.clone().matmul(layer.in_proj_a.clone()).reshape([num_v]);
    let b_proj = x_norm.clone().matmul(layer.in_proj_b.clone()).reshape([num_v]);
    let alpha = (layer.a_log.clone().exp() * softplus_1d(a_proj + layer.dt_bias.clone())).neg().exp();
    let beta = activation::sigmoid(b_proj);

    // Each QK head serves `v_per_qk` V heads.
    let q_exp = q.unsqueeze_dim::<3>(1).repeat_dim(1, v_per_qk).reshape([num_v, hd]);
    let k_exp = k.unsqueeze_dim::<3>(1).repeat_dim(1, v_per_qk).reshape([num_v, hd]);

    // Delta rule: S *= alpha; S += beta (V - S k) kᵀ; y = S q.
    state.s = state.s.clone() * alpha.reshape([num_v, 1, 1]);
    let pred = state.s.clone().matmul(k_exp.clone().unsqueeze_dim::<3>(2)).reshape([num_v, hd]);
    let delta_v = (v - pred) * beta.reshape([num_v, 1]);
    state.s = state.s.clone() + delta_v.unsqueeze_dim::<3>(2).matmul(k_exp.unsqueeze_dim::<3>(1));
    let y = state.s.clone().matmul(q_exp.unsqueeze_dim::<3>(2)).reshape([num_v, hd]);

    let y_normed = rms_norm(y, &layer.attn_norm_weight);
    let z = activation::silu(x_norm.matmul(layer.in_proj_z.clone())).reshape([num_v, hd]);
    let out = x + (y_normed * z).reshape([1, v_dim]).matmul(layer.out_proj.clone());

    mlp_block(out, &layer.post_attn_norm_gamma, &layer.gate_proj, &layer.up_proj, &layer.down_proj)
}

// ============================================================
// Full attention: single token
// ============================================================

/// Split-half RoPE on `x: [rows, rotary_dim]` with one cos/sin row per x row.
fn rope_rows(x: Tensor<2>, cos: Tensor<2>, sin: Tensor<2>, half: usize) -> Tensor<2> {
    let rows = x.dims()[0];
    let first = x.clone().slice([0..rows, 0..half]);
    let second = x.slice([0..rows, half..2 * half]);
    let new_first = first.clone() * cos.clone() - second.clone() * sin.clone();
    let new_second = second * cos + first * sin;
    Tensor::cat(vec![new_first, new_second], 1)
}

/// Partial RoPE on every head of `x: [positions * heads, head_dim]` for the
/// consecutive positions `start..start + positions`.
fn apply_partial_rope(x: Tensor<2>, rope_cos: &Tensor<2>, rope_sin: &Tensor<2>, start: usize, positions: usize, heads: usize, head_dim: usize, rotary_dim: usize) -> Tensor<2> {
    if rotary_dim == 0 {
        return x;
    }
    let half = rotary_dim / 2;
    let rows = positions * heads;
    let expand = |t: &Tensor<2>| {
        t.clone()
            .slice([start..start + positions, 0..half])
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, heads)
            .reshape([rows, half])
    };
    let rotated = rope_rows(x.clone().slice([0..rows, 0..rotary_dim]), expand(rope_cos), expand(rope_sin), half);
    if rotary_dim == head_dim {
        return rotated;
    }
    Tensor::cat(vec![rotated, x.slice([0..rows, rotary_dim..head_dim])], 1)
}

/// Expand KV heads `[kv_heads, seq, hd]` to `[heads, seq, hd]` (GQA).
fn expand_kv(t: Tensor<3>, heads: usize, groups: usize) -> Tensor<3> {
    if groups == 1 {
        return t;
    }
    let [kvh, seq, hd] = t.dims();
    debug_assert_eq!(kvh * groups, heads);
    t.unsqueeze_dim::<4>(1).repeat_dim(1, groups).reshape([heads, seq, hd])
}

/// Attention over `seq` new positions starting at `start` (prefill when
/// start == 0, decode when seq == 1). Updates the KV cache.
fn attention_block(x: Tensor<2>, layer: &GpuFullAttnLayer, kv: &mut GpuKvCacheEntry, config: &IncordConfig, rope_cos: &Tensor<2>, rope_sin: &Tensor<2>) -> Tensor<2> {
    let seq = x.dims()[0];
    let heads = config.num_attn_heads;
    let kv_heads = config.num_kv_heads;
    let hd = config.attn_head_dim;
    let q_dim = heads * hd;
    let rotary_dim = config.rope_dim();
    let start = kv.seq_len;

    let x_norm = rms_norm(x.clone(), &layer.input_norm_gamma);
    let (q, gate) = split_q_gate(x_norm.clone().matmul(layer.q_proj.clone()), layer, seq, heads, hd);
    let k = x_norm.clone().matmul(layer.k_proj.clone());
    let v = x_norm.matmul(layer.v_proj.clone());

    let mut q = q.reshape([seq * heads, hd]);
    let mut k = k.reshape([seq * kv_heads, hd]);
    if let Some(g) = &layer.q_norm {
        q = rms_norm(q, g);
    }
    if let Some(g) = &layer.k_norm {
        k = rms_norm(k, g);
    }
    let q = apply_partial_rope(q, rope_cos, rope_sin, start, seq, heads, hd, rotary_dim);
    let k = apply_partial_rope(k, rope_cos, rope_sin, start, seq, kv_heads, hd, rotary_dim);

    // [seq, heads, hd] -> [heads, seq, hd]
    let q = q.reshape([seq, heads, hd]).swap_dims(0, 1);
    let k_new = k.reshape([seq, kv_heads, hd]).swap_dims(0, 1);
    let v_new = v.reshape([seq, kv_heads, hd]).swap_dims(0, 1);
    let (k_all, v_all) = match (kv.k.take(), kv.v.take()) {
        (Some(pk), Some(pv)) => (Tensor::cat(vec![pk, k_new], 1), Tensor::cat(vec![pv, v_new], 1)),
        _ => (k_new, v_new),
    };
    kv.seq_len = start + seq;
    let total = kv.seq_len;
    kv.k = Some(k_all.clone());
    kv.v = Some(v_all.clone());

    let groups = config.num_kv_groups();
    let k_exp = expand_kv(k_all, heads, groups);
    let v_exp = expand_kv(v_all, heads, groups);
    let mut scores = q.matmul(k_exp.swap_dims(1, 2)).mul_scalar(1.0 / (hd as f32).sqrt()); // [heads, seq, total]
    if seq > 1 {
        // New position i (absolute start + i) may not see absolute positions > start + i.
        let mask = Tensor::<2>::ones([seq, total], &x.device()).triu(start as i64 + 1).mul_scalar(-1e9f32);
        scores = scores + mask.unsqueeze_dim::<3>(0);
    }
    let context = activation::softmax(scores, 2).matmul(v_exp); // [heads, seq, hd]
    let mut attn = context.swap_dims(0, 1).reshape([seq, q_dim]);
    if let Some(g) = gate {
        attn = attn * activation::sigmoid(g);
    }
    let out = x + attn.matmul(layer.o_proj.clone());

    mlp_block(out, &layer.post_attn_norm_gamma, &layer.gate_proj, &layer.up_proj, &layer.down_proj)
}

// ============================================================
// DeltaNet: prefill (GPU projections, CPU recurrence)
// ============================================================

/// Batched DeltaNet prefill: projections on the GPU, the sequential conv +
/// state recurrence on the CPU, then gating, out_proj and MLP on the GPU.
fn prefill_deltanet_layer_gpu(x: Tensor<2>, layer: &GpuDeltaNetLayer, state: &mut GpuDeltaNetState, config: &IncordConfig) -> Tensor<2> {
    let device = x.device();
    let seq_len = x.dims()[0];
    let num_qk = config.num_qk_heads;
    let num_v = config.num_v_heads;
    let hd = config.deltanet_head_dim;
    let q_dim = config.deltanet_q_dim();
    let k_dim = config.deltanet_k_dim();
    let v_dim = config.deltanet_v_dim();
    let qkv_dim = config.deltanet_qkv_dim();
    let ks = config.conv_kernel_size;
    let buf_len = ks - 1;
    let v_per_qk = num_v / num_qk;
    let scale = 1.0 / (hd as f32).sqrt();

    let x_norm = rms_norm(x.clone(), &layer.input_norm_gamma);
    let to_vec1 = |t: Tensor<1>| t.into_data().to_vec::<f32>().unwrap();
    let to_vec2 = |t: Tensor<2>| t.into_data().to_vec::<f32>().unwrap();
    let qkv_cpu = to_vec2(x_norm.clone().matmul(layer.in_proj_qkv.clone()));
    let a_cpu = to_vec2(x_norm.clone().matmul(layer.in_proj_a.clone()));
    let b_cpu = to_vec2(x_norm.clone().matmul(layer.in_proj_b.clone()));
    let z_all = x_norm.matmul(layer.in_proj_z.clone());

    let conv_w = to_vec2(layer.conv1d_weight.clone());
    let a_exp_table: Vec<f32> = to_vec1(layer.a_log.clone()).iter().map(|v| v.exp()).collect();
    let dt_bias = to_vec1(layer.dt_bias.clone());
    let attn_norm_w = to_vec1(layer.attn_norm_weight.clone());
    let mut s: Vec<f32> = state.s.clone().into_data().to_vec::<f32>().unwrap();
    let mut conv_buf: Vec<f32> = to_vec2(state.conv_buf.clone());

    let mut output_y = vec![0.0f32; seq_len * v_dim];
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    let mut q_exp = vec![0.0f32; num_v * hd];
    let mut k_exp = vec![0.0f32; num_v * hd];
    let mut pred = vec![0.0f32; num_v * hd];

    for t in 0..seq_len {
        let qkv_t = &qkv_cpu[t * qkv_dim..(t + 1) * qkv_dim];
        let a_t = &a_cpu[t * num_v..(t + 1) * num_v];
        let b_t = &b_cpu[t * num_v..(t + 1) * num_v];

        for ch in 0..qkv_dim {
            let buf_off = ch * buf_len;
            let w_off = ch * ks;
            let mut sum = qkv_t[ch] * conv_w[w_off + buf_len];
            for j in 0..buf_len {
                sum += conv_buf[buf_off + j] * conv_w[w_off + j];
            }
            let v = sum;
            qkv_conv[ch] = v * (1.0 / (1.0 + (-v).exp()));
            conv_buf.copy_within(buf_off + 1..buf_off + buf_len, buf_off);
            conv_buf[buf_off + buf_len - 1] = qkv_t[ch];
        }

        let q = &qkv_conv[0..q_dim];
        let k = &qkv_conv[q_dim..q_dim + k_dim];
        let v_data = &qkv_conv[q_dim + k_dim..qkv_dim];
        for h in 0..num_qk {
            let o = h * hd;
            let q_sq: f32 = q[o..o + hd].iter().map(|x| x * x).sum();
            let k_sq: f32 = k[o..o + hd].iter().map(|x| x * x).sum();
            let q_inv = scale / (q_sq.sqrt() + 1e-12);
            let k_inv = 1.0 / (k_sq.sqrt() + 1e-12);
            for g in 0..v_per_qk {
                let e = (h * v_per_qk + g) * hd;
                for d in 0..hd {
                    q_exp[e + d] = q[o + d] * q_inv;
                    k_exp[e + d] = k[o + d] * k_inv;
                }
            }
        }

        for h in 0..num_v {
            let sp = (1.0 + (a_t[h] + dt_bias[h]).exp()).ln();
            let alpha = (-a_exp_table[h] * sp).exp();
            let beta = 1.0 / (1.0 + (-b_t[h]).exp());
            let so = h * hd * hd;
            let ho = h * hd;
            s[so..so + hd * hd].iter_mut().for_each(|x| *x *= alpha);
            for i in 0..hd {
                pred[ho + i] = (0..hd).map(|j| s[so + i * hd + j] * k_exp[ho + j]).sum();
            }
            for i in 0..hd {
                let delta = beta * (v_data[ho + i] - pred[ho + i]);
                for j in 0..hd {
                    s[so + i * hd + j] += delta * k_exp[ho + j];
                }
            }
            for i in 0..hd {
                pred[ho + i] = (0..hd).map(|j| s[so + i * hd + j] * q_exp[ho + j]).sum();
            }
        }

        let y_out = &mut output_y[t * v_dim..(t + 1) * v_dim];
        for h in 0..num_v {
            let ho = h * hd;
            let sum_sq: f32 = pred[ho..ho + hd].iter().map(|x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-6).sqrt();
            for d in 0..hd {
                y_out[ho + d] = pred[ho + d] * inv_rms * attn_norm_w[d];
            }
        }
    }

    let y_gpu: Tensor<2> = Tensor::from_data(TensorData::new(output_y, [seq_len, v_dim]), &device);
    state.s = Tensor::from_data(TensorData::new(s, [num_v, hd, hd]), &device);
    state.conv_buf = Tensor::from_data(TensorData::new(conv_buf, [qkv_dim, buf_len]), &device);
    state.conv_pos += seq_len;

    let out = x + (y_gpu * activation::silu(z_all)).matmul(layer.out_proj.clone());
    mlp_block(out, &layer.post_attn_norm_gamma, &layer.gate_proj, &layer.up_proj, &layer.down_proj)
}

// ============================================================
// Whole-model forward
// ============================================================

/// Final norm and output head on the GPU: `lm_head [vocab, hidden] · hᵀ`.
fn head_logits(model: &GpuModel, last: Tensor<2>) -> Vec<f32> {
    let h = rms_norm(last, &model.final_norm_gamma); // [1, hidden]
    let logits = model.lm_head.clone().matmul(h.swap_dims(0, 1)); // [vocab, 1]
    logits.into_data().to_vec::<f32>().unwrap()
}

fn embed_rows(model: &GpuModel, weights: &ModelWeights, tokens: &[u32]) -> Tensor<2> {
    let mut data = Vec::with_capacity(tokens.len() * model.hidden);
    for &t in tokens {
        data.extend(embed_lookup(&weights.embed, t as usize, model.hidden));
    }
    Tensor::from_data(TensorData::new(data, [tokens.len(), model.hidden]), &model.device)
}

fn run_layers(model: &GpuModel, mut x: Tensor<2>, cache: &mut GpuHybridCache, prefill: bool) -> Tensor<2> {
    let config = &model.config;
    for (i, (layer, entry)) in model.layers.iter().zip(cache.entries.iter_mut()).enumerate() {
        x = match (layer, entry) {
            (GpuHybridLayer::DeltaNet(dl), GpuCacheEntry::DeltaNet(ds)) => {
                if prefill {
                    prefill_deltanet_layer_gpu(x, dl, ds, config)
                } else {
                    forward_deltanet_decode_gpu(x, dl, ds, config)
                }
            }
            (GpuHybridLayer::FullAttn(fl), GpuCacheEntry::KvCache(kv)) => {
                attention_block(x, fl, kv, config, &model.rope_cos, &model.rope_sin)
            }
            _ => panic!("Layer/cache type mismatch at layer {i}"),
        };
    }
    x
}

/// Prefill the prompt; logits for its last position.
pub fn forward_prefill_gpu(model: &GpuModel, weights: &ModelWeights, tokens: &[u32], cache: &mut GpuHybridCache) -> Vec<f32> {
    let x = run_layers(model, embed_rows(model, weights, tokens), cache, true);
    let seq = tokens.len();
    head_logits(model, x.slice([seq - 1..seq, 0..model.hidden]))
}

/// One decode step; logits for the new token's position.
pub fn forward_decode_gpu(model: &GpuModel, weights: &ModelWeights, token: u32, cache: &mut GpuHybridCache) -> Vec<f32> {
    if !profiling() {
        let x = run_layers(model, embed_rows(model, weights, &[token]), cache, false);
        return head_logits(model, x);
    }
    // INCORD_GPU_PROFILE=1: where one decode step spends its time.
    let t0 = Instant::now();
    let x0 = embed_rows(model, weights, &[token]);
    let t_embed = t0.elapsed();
    let x = run_layers(model, x0, cache, false);
    let t_submit = t0.elapsed();
    let logits = head_logits(model, x);
    let t_all = t0.elapsed();
    eprintln!(
        "[profile] embed {:.1?} | layers submitted {:.1?} | head + GPU finish + readback {:.1?} | total {:.1?}",
        t_embed,
        t_submit - t_embed,
        t_all - t_submit,
        t_all
    );
    logits
}

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("INCORD_GPU_PROFILE").is_some_and(|v| v != "0"))
}

/// Reject models the GPU path cannot run yet.
fn check_supported(weights: &ModelWeights) -> Result<(), String> {
    if weights.has_rotated() {
        return Err("model has rotated weights, not supported on the GPU path yet".into());
    }
    if weights.has_low_bit() {
        return Err("model has Q3/Q2/E8 weights, not supported on the GPU path yet".into());
    }
    Ok(())
}

/// Logits after prefilling `prompt`, then after each token of `then`
/// (teacher-forced), all on the GPU. Used to check GPU against CPU.
pub fn logits_gpu(weights: &ModelWeights, device: &Device, prompt: &[u32], then: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    check_supported(weights)?;
    let model = load_model_gpu(weights, device)?;
    let mut cache = GpuHybridCache::new(&model.config, device);
    let mut out = vec![forward_prefill_gpu(&model, weights, prompt, &mut cache)];
    for &t in then {
        out.push(forward_decode_gpu(&model, weights, t, &mut cache));
    }
    Ok(out)
}

// ============================================================
// Sampling
// ============================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0
    }
}

fn sample_token_top_k(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let inv_temp = 1.0 / temperature;
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    let max_logit = indexed[0].1;
    let mut probs: Vec<(u32, f32)> = indexed.iter().map(|&(i, l)| (i, ((l - max_logit) * inv_temp).exp())).collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let candidates = &probs[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    let rand_val = rng.next_f32() * total;
    let mut accum = 0.0f32;
    for &(token_id, prob) in candidates {
        accum += prob;
        if accum >= rand_val {
            return token_id;
        }
    }
    candidates[0].0
}

fn apply_presence_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 0.0 || generated.is_empty() {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &tok in generated {
        if let Some(s) = seen.get_mut(tok as usize) {
            *s = true;
        }
    }
    for (idx, &was_seen) in seen.iter().enumerate() {
        if was_seen {
            logits[idx] -= penalty;
        }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    let window = generated.len().min(64);
    for &tok in &generated[generated.len() - window..] {
        if let Some(l) = logits.get_mut(tok as usize) {
            if *l > 0.0 {
                *l /= penalty;
            } else {
                *l *= penalty;
            }
        }
    }
}

// ============================================================
// Main entry point
// ============================================================

pub fn generate_gpu(weights: &ModelWeights, tokenizer: &IncordTokenizer, prompt: &str, params: &GenerateParams) -> Result<(), String> {
    check_supported(weights)?;
    let result = std::cell::RefCell::new(None);

    std::thread::scope(|s| {
        let builder = std::thread::Builder::new().name("gpu-llm".into()).stack_size(128 * 1024 * 1024);
        let handle = builder.spawn_scoped(s, || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| generate_gpu_inner(weights, tokenizer, prompt, params)))
        });
        *result.borrow_mut() = Some(match handle {
            Ok(h) => match h.join() {
                Ok(Ok(inner)) => inner,
                Ok(Err(panic_payload)) => {
                    let msg = panic_payload
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic_payload.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown error (likely out of VRAM)");
                    Err(format!("GPU panic: {msg}"))
                }
                Err(_) => Err("GPU thread panicked unexpectedly".to_string()),
            },
            Err(e) => Err(format!("Failed to spawn GPU thread: {e}")),
        });
    });

    result.into_inner().unwrap_or(Err("No result from GPU thread".into()))
}

/// The GPU to run on: CUDA device 0 in a CUDA build, otherwise the first
/// discrete GPU through wgpu. Not `Device::default()`: on Windows the default
/// wgpu adapter can be the software renderer, which runs on the CPU.
pub fn gpu_device() -> Device {
    #[cfg(feature = "gpu-cuda")]
    {
        Device::cuda(0)
    }
    #[cfg(all(feature = "gpu", not(feature = "gpu-cuda")))]
    {
        Device::wgpu(cortex::tensor::DeviceKind::DiscreteGpu(0))
    }
    // Apple silicon has no discrete GPU; its default adapter is the GPU.
    #[cfg(all(feature = "gpu-metal", not(any(feature = "gpu", feature = "gpu-cuda"))))]
    {
        Device::default()
    }
}

fn detect_free_vram_mb() -> Option<usize> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8(output.stdout).ok()?.lines().next()?.trim().parse().ok()
}

fn generate_gpu_inner(weights: &ModelWeights, tokenizer: &IncordTokenizer, prompt: &str, params: &GenerateParams) -> Result<(), String> {
    // Q4/Q8 stay quantized on the device; headroom for cache and activations.
    let needed_vram_mb = weights.memory_bytes() / (1024 * 1024) + 800;
    if let Some(free_mb) = detect_free_vram_mb() {
        eprintln!("VRAM: {free_mb} MB free, ~{needed_vram_mb} MB needed ({})", weights.format_name);
        if free_mb < needed_vram_mb {
            return Err(format!("{free_mb} MB free VRAM is not enough (~{needed_vram_mb} MB needed)"));
        }
    }

    let device = gpu_device();
    eprintln!("GPU device: {device:?}");
    let t0 = Instant::now();
    let model = load_model_gpu(weights, &device)?;
    eprintln!("Weights loaded to GPU in {:.1?}", t0.elapsed());

    let tokens = tokenizer.format_chat(prompt, params.think, params.max_new_tokens);
    eprintln!("Prompt tokens: {}", tokens.len());

    let mut cache = GpuHybridCache::new(&model.config, &device);
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut in_think_block = params.think;
    let mut think_content = String::new();
    let mut think_tokens = 0u32;
    let mut rng = Rng::new();

    let t0 = Instant::now();
    let mut logits = forward_prefill_gpu(&model, weights, &tokens, &mut cache);
    let prefill_time = t0.elapsed();
    eprintln!(
        "Prefill (GPU): {} tokens in {prefill_time:.1?} ({:.1} tok/s)",
        tokens.len(),
        tokens.len() as f64 / prefill_time.as_secs_f64()
    );

    if !in_think_block && params.temperature > 0.0 {
        apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
    }
    let mut next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);

    let mut decode_tokens = 0u32;
    let decode_start = Instant::now();

    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id
            || next_token_id == crate::tokenizer::IM_END
            || next_token_id == crate::tokenizer::ENDOFTEXT
        {
            eprintln!("\n[EOS after {step} tokens (id={next_token_id})]");
            break;
        }
        generated_tokens.push(next_token_id);

        if next_token_id == crate::tokenizer::THINK_START {
            in_think_block = true;
            if params.show_think {
                eprint!("<think>");
            }
        } else if next_token_id == crate::tokenizer::THINK_END {
            in_think_block = false;
            eprintln!("\n[thinking done: {think_tokens} tokens, {} chars]", think_content.len());
            if params.show_think {
                eprintln!("</think>");
            }
            think_content.clear();
            think_tokens = 0;
        } else {
            let token_text = tokenizer.decode(&[next_token_id]);
            if in_think_block {
                think_content.push_str(&token_text);
                think_tokens += 1;
                if params.show_think {
                    eprint!("{token_text}");
                } else if think_tokens % 50 == 0 {
                    eprint!("[thinking: {think_tokens} tokens...] ");
                }
            } else {
                print!("{token_text}");
                std::io::stdout().flush().ok();
            }
        }

        // Stop at a sentence end only within the last 10 tokens of the budget.
        if !in_think_block && step >= params.max_new_tokens.saturating_sub(10) {
            let piece = tokenizer.decode(&[next_token_id]);
            let trimmed = piece.trim_end();
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                eprintln!("\n[clean stop near token limit at step {step}]");
                break;
            }
        }
        if crate::generate::is_stuck_in_loop(&generated_tokens) {
            eprintln!("\n[loop detected at step {decode_tokens}, forcing EOS]");
            break;
        }

        let mut logits = forward_decode_gpu(&model, weights, next_token_id, &mut cache);

        if in_think_block {
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
            logits[params.eos_token_id as usize] = f32::NEG_INFINITY;
            logits[crate::tokenizer::IM_END as usize] = f32::NEG_INFINITY;
            logits[crate::tokenizer::ENDOFTEXT as usize] = f32::NEG_INFINITY;
        } else if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
        }

        if in_think_block && think_tokens >= params.think_budget as u32 {
            next_token_id = crate::tokenizer::THINK_END;
            eprintln!("\n[think budget reached ({} tokens), forcing </think>]", params.think_budget);
        } else {
            next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
        }
        decode_tokens += 1;
    }

    if in_think_block {
        eprintln!("\n[WARNING: thinking did not finish within {} tokens]", params.max_new_tokens);
    }
    println!();
    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!(
            "Decode (GPU): {decode_tokens} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens as f64 / decode_elapsed.as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! GPU vs CPU on small random models. The CPU path (`gemv.rs`) is the
    //! reference; the GPU must give the same next-token logits after the
    //! prompt and after every following token.

    use super::*;
    use crate::config::LayerType;
    use crate::gemv::{
        DeltaNetLayerWeights, F16Weight, FullAttnLayerWeights, HybridCache, HybridLayerWeights, Q4Weight,
        Q8Weight, Weight, forward_decode, prefill,
    };
    use crate::quant::{Rounding, quantize_q4, quantize_q8};
    use half::f16;

    #[derive(Clone, Copy, PartialEq)]
    enum Fmt {
        Q4,
        Q8,
    }

    struct Gen(u64);
    impl Gen {
        fn vals(&mut self, len: usize, amp: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * amp
                })
                .collect()
        }
        /// `[k, n]` weight; groups of 32 need n % 32 == 0, else F16.
        fn weight(&mut self, k: usize, n: usize, fmt: Fmt) -> Weight {
            let w = self.vals(k * n, 0.2);
            if n % 32 != 0 {
                return Weight::F16(F16Weight { data: w.iter().map(|&v| f16::from_f32(v)).collect(), k, n });
            }
            match fmt {
                Fmt::Q4 => {
                    let (p, s) = quantize_q4(&w, k, n, Rounding::Rtn);
                    Weight::Q4(Q4Weight { packed: p.into(), scales: s.into(), k, n })
                }
                Fmt::Q8 => {
                    let (d, s) = quantize_q8(&w, k, n, Rounding::Rtn);
                    Weight::Q8(Q8Weight { data: d.into(), scales: s.into(), k, n })
                }
            }
        }
        fn gamma(&mut self, len: usize) -> Vec<f16> {
            self.vals(len, 0.3).iter().map(|&v| f16::from_f32(1.0 + v)).collect()
        }
    }

    /// Small model: `qwen_like` = DeltaNet + gated attention with QK norms;
    /// otherwise full attention only, plain query, no QK norms (MiniCPM-like).
    fn tiny_model(qwen_like: bool, fmt: Fmt) -> ModelWeights {
        let mut c = crate::gemv::mock::tiny_config();
        c.vocab_size = 512;
        c.tie_word_embeddings = false;
        c.layer_types = if qwen_like {
            vec![LayerType::DeltaNet, LayerType::FullAttn, LayerType::DeltaNet, LayerType::FullAttn]
        } else {
            vec![LayerType::FullAttn; 3]
        };
        c.num_layers = c.layer_types.len();
        let (h, inter, q_dim, kv_dim) = (c.hidden_size, c.intermediate_size, c.attn_q_dim(), c.attn_kv_dim());
        let base: u64 = if qwen_like { 0x51 } else { 0x77 };
        let mut g = Gen((base ^ ((fmt as u64) << 8)) | 1);

        let layers = c
            .layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttn => HybridLayerWeights::FullAttn(FullAttnLayerWeights {
                    q_proj: g.weight(h, if qwen_like { 2 * q_dim } else { q_dim }, fmt),
                    k_proj: g.weight(h, kv_dim, fmt),
                    v_proj: g.weight(h, kv_dim, fmt),
                    o_proj: g.weight(q_dim, h, fmt),
                    q_norm: if qwen_like { g.gamma(c.attn_head_dim) } else { Vec::new() },
                    k_norm: if qwen_like { g.gamma(c.attn_head_dim) } else { Vec::new() },
                    gate_proj: g.weight(h, inter, fmt),
                    up_proj: g.weight(h, inter, fmt),
                    down_proj: g.weight(inter, h, fmt),
                    input_norm: g.gamma(h),
                    post_attn_norm: g.gamma(h),
                }),
                LayerType::DeltaNet => HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                    in_proj_qkv: g.weight(h, c.deltanet_qkv_dim(), fmt),
                    in_proj_a: g.weight(h, c.num_v_heads, fmt),
                    in_proj_b: g.weight(h, c.num_v_heads, fmt),
                    in_proj_z: g.weight(h, c.deltanet_v_dim(), fmt),
                    out_proj: g.weight(c.deltanet_v_dim(), h, fmt),
                    conv1d_weight: g.vals(c.deltanet_qkv_dim() * c.conv_kernel_size, 0.5),
                    a_log: g.vals(c.num_v_heads, 0.5),
                    dt_bias: g.vals(c.num_v_heads, 0.5),
                    attn_norm_weight: g.gamma(c.deltanet_head_dim),
                    gate_proj: g.weight(h, inter, fmt),
                    up_proj: g.weight(h, inter, fmt),
                    down_proj: g.weight(inter, h, fmt),
                    input_norm: g.gamma(h),
                    post_attn_norm: g.gamma(h),
                }),
            })
            .collect();

        let half = c.rope_dim() / 2;
        let max_pos = c.max_position_embeddings;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f64 / 10_000f64.powf(2.0 * i as f64 / c.rope_dim() as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        ModelWeights {
            layers,
            embed: g.weight(c.vocab_size, h, Fmt::Q8),
            lm_head: g.weight(c.vocab_size, h, fmt),
            vocab: c.vocab_size,
            hidden: h,
            final_norm: g.gamma(h),
            rope_cos: cos,
            rope_sin: sin,
            format_name: "test",
            vision: None,
            config: c,
        }
    }

    fn cpu_logits(w: &ModelWeights, prompt: &[u32], then: &[u32]) -> Vec<Vec<f32>> {
        let mut cache = HybridCache::new(&w.config);
        let mut out = vec![prefill(w, prompt, &mut cache)];
        for &t in then {
            out.push(forward_decode(w, t as usize, &mut cache));
        }
        out
    }

    fn argmax(v: &[f32]) -> usize {
        (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
    }

    fn check(qwen_like: bool, fmt: Fmt) {
        let w = tiny_model(qwen_like, fmt);
        let prompt = [3u32, 17, 42, 99, 7, 256, 11, 400];
        let then = [5u32, 77, 300, 12];
        let cpu = cpu_logits(&w, &prompt, &then);
        let gpu = logits_gpu(&w, &gpu_device(), &prompt, &then).expect("GPU forward");
        assert_eq!(cpu.len(), gpu.len());
        for (step, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            let scale = c.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = c.iter().zip(g).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("step {step}: max |cpu - gpu| = {diff:.2e} (logit scale {scale:.2})");
            // The CPU keeps its KV cache in f16; the GPU in f32.
            assert!(diff <= 1e-2 * scale, "step {step}: max |cpu - gpu| {diff} vs logit scale {scale}");
            assert_eq!(argmax(c), argmax(g), "step {step}: top token differs");
        }
    }

    #[test]
    fn gpu_matches_cpu_minicpm_like_q4() {
        check(false, Fmt::Q4);
    }

    #[test]
    fn gpu_matches_cpu_minicpm_like_q8() {
        check(false, Fmt::Q8);
    }

    #[test]
    fn gpu_matches_cpu_qwen_like_q4() {
        check(true, Fmt::Q4);
    }

    #[test]
    fn gpu_matches_cpu_qwen_like_q8() {
        check(true, Fmt::Q8);
    }
}
