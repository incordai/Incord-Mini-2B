//! GPU weight loading for Incord (cortex-v2).
//!
//! Converts CPU `ModelWeights` into Cortex GPU tensors. Q4 and Q8 weights stay
//! quantized on the device (block-32 symmetric, f16 scales) and are multiplied
//! by cortex's fused dequant-matmul; F16 weights are uploaded as f32.
//!
//! The output head is uploaded like any other weight. The embedding stays in
//! `ModelWeights` on the CPU: the forward looks up one row per token.

use cortex::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use cortex::tensor::{DType, Device, Tensor, TensorData};
use half::f16;

use crate::config::IncordConfig;
use crate::gemv::{DeltaNetLayerWeights, FullAttnLayerWeights, HybridLayerWeights, ModelWeights, Weight};

// ============================================================
// GPU model structures
// ============================================================

pub struct GpuModel {
    pub layers: Vec<GpuHybridLayer>,
    pub hidden: usize,
    pub vocab: usize,
    pub final_norm_gamma: Tensor<1>,
    /// Output head `[vocab, hidden]` in its stored format; logits are
    /// `lm_head · h`, so its quant groups run along the reduced dim.
    pub lm_head: Tensor<2>,
    /// RoPE tables `[max_pos, rope_dim / 2]`.
    pub rope_cos: Tensor<2>,
    pub rope_sin: Tensor<2>,
    pub config: IncordConfig,
    pub device: Device,
}

pub enum GpuHybridLayer {
    DeltaNet(GpuDeltaNetLayer),
    FullAttn(GpuFullAttnLayer),
}

pub struct GpuDeltaNetLayer {
    pub in_proj_qkv: Tensor<2>,
    pub in_proj_a: Tensor<2>,
    pub in_proj_b: Tensor<2>,
    pub in_proj_z: Tensor<2>,
    pub out_proj: Tensor<2>,
    pub conv1d_weight: Tensor<2>,    // [qkv_dim, kernel_size]
    pub a_log: Tensor<1>,            // [num_v_heads]
    pub dt_bias: Tensor<1>,          // [num_v_heads]
    pub attn_norm_weight: Tensor<1>, // [head_dim]
    pub gate_proj: Tensor<2>,
    pub up_proj: Tensor<2>,
    pub down_proj: Tensor<2>,
    pub input_norm_gamma: Tensor<1>,
    pub post_attn_norm_gamma: Tensor<1>,
}

pub struct GpuFullAttnLayer {
    pub q_proj: Tensor<2>,
    pub k_proj: Tensor<2>,
    pub v_proj: Tensor<2>,
    pub o_proj: Tensor<2>,
    /// Per-head QK norms; `None` when the model has none (e.g. Llama-style).
    pub q_norm: Option<Tensor<1>>,
    pub k_norm: Option<Tensor<1>>,
    /// q_proj also emits an output gate per head (`[Q, gate]` interleaved),
    /// detected from its width exactly as the CPU path does.
    pub gated: bool,
    pub gate_proj: Tensor<2>,
    pub up_proj: Tensor<2>,
    pub down_proj: Tensor<2>,
    pub input_norm_gamma: Tensor<1>,
    pub post_attn_norm_gamma: Tensor<1>,
}

// ============================================================
// Weight conversion
// ============================================================

/// incord's Q4/Q8 layout: groups of 32 along the output dim `n`, symmetric,
/// f16 scales. Cortex packs along the last dim (`PackedU32(0)`), low bits
/// first, which is the same element order.
fn block32_scheme(value: QuantValue) -> QuantScheme {
    QuantScheme {
        value,
        param: QuantParam::F16,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([32])),
        mode: QuantMode::Symmetric,
    }
}

/// Packed values padded to the 4-byte scale alignment, then the f16 scales.
fn quantized_bytes(mut values: Vec<u8>, scales: &[f16]) -> Vec<u8> {
    while values.len() % 4 != 0 {
        values.push(0);
    }
    values.reserve(scales.len() * 2);
    for s in scales {
        values.extend(s.to_le_bytes());
    }
    values
}

fn q4_weight_to_gpu(packed: &[u8], scales: &[f16], k: usize, n: usize, device: &Device) -> Tensor<2> {
    // incord stores each nibble as q + 8; cortex reads signed nibbles.
    let values = packed.iter().map(|&b| b ^ 0x88).collect();
    let data = TensorData::from_bytes_vec(
        quantized_bytes(values, scales),
        [k, n],
        DType::QFloat(block32_scheme(QuantValue::Q4S)),
    );
    Tensor::from_data(data, device)
}

fn q8_weight_to_gpu(q: &[i8], scales: &[f16], k: usize, n: usize, device: &Device) -> Tensor<2> {
    let values = q.iter().map(|&v| v as u8).collect();
    let data = TensorData::from_bytes_vec(
        quantized_bytes(values, scales),
        [k, n],
        DType::QFloat(block32_scheme(QuantValue::Q8S)),
    );
    Tensor::from_data(data, device)
}

fn f16_weight_to_gpu(data: &[f16], k: usize, n: usize, device: &Device) -> Tensor<2> {
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    Tensor::from_data(TensorData::new(f32_data, [k, n]), device)
}

fn weight_to_gpu(w: &Weight, device: &Device) -> Result<Tensor<2>, String> {
    match w {
        Weight::F16(fw) => Ok(f16_weight_to_gpu(&fw.data, fw.k, fw.n, device)),
        Weight::Q4(qw) => Ok(q4_weight_to_gpu(&qw.packed, &qw.scales, qw.k, qw.n, device)),
        Weight::Q8(qw) => Ok(q8_weight_to_gpu(&qw.data, &qw.scales, qw.k, qw.n, device)),
        Weight::LowBit(_) => Err("Q3/Q2/E8 weights are not supported on the GPU path yet".into()),
        Weight::Rotated(..) => Err("rotated weights are not supported on the GPU path yet".into()),
    }
}

fn norm_to_gpu(gamma: &[f16], device: &Device) -> Tensor<1> {
    let f32_data: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
    Tensor::from_data(TensorData::new(f32_data, [gamma.len()]), device)
}

fn f32_vec_to_gpu_1d(data: &[f32], device: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

fn f32_vec_to_gpu_2d(data: &[f32], rows: usize, cols: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

// ============================================================
// Model loading
// ============================================================

pub fn load_model_gpu(weights: &ModelWeights, device: &Device) -> Result<GpuModel, String> {
    let config = &weights.config;
    eprintln!("Loading {} layers to GPU ({})...", config.num_layers, weights.format_name);

    let mut layers = Vec::with_capacity(config.num_layers);
    for (i, lw) in weights.layers.iter().enumerate() {
        if i % 6 == 0 {
            eprintln!("  GPU layer {i}/{}...", config.num_layers);
        }
        layers.push(match lw {
            HybridLayerWeights::DeltaNet(d) => GpuHybridLayer::DeltaNet(load_deltanet_layer(d, config, device)?),
            HybridLayerWeights::FullAttn(f) => GpuHybridLayer::FullAttn(load_fullattn_layer(f, config, device)?),
        });
    }

    let final_norm_gamma = norm_to_gpu(&weights.final_norm, device);
    let rope_half = config.rope_dim() / 2;
    let max_pos = weights.rope_cos.len() / rope_half;
    let rope_cos = f32_vec_to_gpu_2d(&weights.rope_cos, max_pos, rope_half, device);
    let rope_sin = f32_vec_to_gpu_2d(&weights.rope_sin, max_pos, rope_half, device);

    Ok(GpuModel {
        layers,
        hidden: weights.hidden,
        vocab: weights.vocab,
        final_norm_gamma,
        lm_head: weight_to_gpu(&weights.lm_head, device)?,
        rope_cos,
        rope_sin,
        config: config.clone(),
        device: device.clone(),
    })
}

fn load_deltanet_layer(d: &DeltaNetLayerWeights, config: &IncordConfig, device: &Device) -> Result<GpuDeltaNetLayer, String> {
    let qkv_dim = config.deltanet_qkv_dim();
    let kernel = config.conv_kernel_size;
    Ok(GpuDeltaNetLayer {
        in_proj_qkv: weight_to_gpu(&d.in_proj_qkv, device)?,
        in_proj_a: weight_to_gpu(&d.in_proj_a, device)?,
        in_proj_b: weight_to_gpu(&d.in_proj_b, device)?,
        in_proj_z: weight_to_gpu(&d.in_proj_z, device)?,
        out_proj: weight_to_gpu(&d.out_proj, device)?,
        conv1d_weight: f32_vec_to_gpu_2d(&d.conv1d_weight, qkv_dim, kernel, device),
        a_log: f32_vec_to_gpu_1d(&d.a_log, device),
        dt_bias: f32_vec_to_gpu_1d(&d.dt_bias, device),
        attn_norm_weight: norm_to_gpu(&d.attn_norm_weight, device),
        gate_proj: weight_to_gpu(&d.gate_proj, device)?,
        up_proj: weight_to_gpu(&d.up_proj, device)?,
        down_proj: weight_to_gpu(&d.down_proj, device)?,
        input_norm_gamma: norm_to_gpu(&d.input_norm, device),
        post_attn_norm_gamma: norm_to_gpu(&d.post_attn_norm, device),
    })
}

fn load_fullattn_layer(f: &FullAttnLayerWeights, config: &IncordConfig, device: &Device) -> Result<GpuFullAttnLayer, String> {
    let optional_norm = |g: &[f16]| (!g.is_empty()).then(|| norm_to_gpu(g, device));
    Ok(GpuFullAttnLayer {
        q_proj: weight_to_gpu(&f.q_proj, device)?,
        k_proj: weight_to_gpu(&f.k_proj, device)?,
        v_proj: weight_to_gpu(&f.v_proj, device)?,
        o_proj: weight_to_gpu(&f.o_proj, device)?,
        q_norm: optional_norm(&f.q_norm),
        k_norm: optional_norm(&f.k_norm),
        gated: f.q_proj.n() == 2 * config.attn_q_dim(),
        gate_proj: weight_to_gpu(&f.gate_proj, device)?,
        up_proj: weight_to_gpu(&f.up_proj, device)?,
        down_proj: weight_to_gpu(&f.down_proj, device)?,
        input_norm_gamma: norm_to_gpu(&f.input_norm, device),
        post_attn_norm_gamma: norm_to_gpu(&f.post_attn_norm, device),
    })
}
