//! Save/load Incord-Mini-2B ModelWeights to compact binary format (.incord).
//!
//! File format:
//!   Header: magic "QOR4" + version(u32) + format(u8: 0=F16, 1=Q4)
//!   Config: all architecture params
//!   Per-layer: type byte + layer-specific weights
//!   Global: embedding + final norm + RoPE tables
//!   Vision: (optional) patch_embed, pos_embed, 24 blocks, merger

use half::f16;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::config::{LayerType, IncordConfig};
use crate::gemv::*;

const MAGIC: &[u8; 4] = b"QOR4";
const VERSION: u32 = 2; // v2: vision weights stored as f16 (was f32 in v1)
/// v3: a weight's format byte may carry [`ROTATED_FLAG`]; the sign seed is
/// implicitly `rotate::DEFAULT_SEED`. Read-only legacy: new files use v4.
pub const VERSION_ROTATED: u32 = 3;
/// v4: like v3, plus rotation metadata right after the header's format
/// byte: scheme id (u8, `rotate::SCHEME_BLOCK_HADAMARD`) and sign seed
/// (u64). Written only when the file holds rotated weights, so plain files
/// stay v2 and older loaders reject rotated files at the header.
pub const VERSION_ROTATION_META: u32 = 4;
/// Set on a weight's per-weight format byte when its data is stored in
/// Hadamard-rotated input space (`Weight::Rotated`, see `crate::rotate`).
pub const ROTATED_FLAG: u8 = 0x80;

// ============================================================
// I/O helpers
// ============================================================

fn write_u8(w: &mut impl Write, val: u8) -> io::Result<()> {
    w.write_all(&[val])
}
fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_f32(w: &mut impl Write, val: f32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_f64(w: &mut impl Write, val: f64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_f16_vec(w: &mut impl Write, data: &[f16]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_bytes(w: &mut impl Write, data: &[u8]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    w.write_all(data)
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_f16_vec(r: &mut impl Read) -> io::Result<Vec<f16>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(2) {
        out.push(f16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(len);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Write f32 data as f16 on disk (half the size, minimal quality loss).
fn write_f32_as_f16(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    write_f16_vec(w, &f16_data)
}

/// Read f16 data from disk and convert to f32 for computation.
fn read_f16_as_f32(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let f16_data = read_f16_vec(r)?;
    Ok(f16_data.iter().map(|v| v.to_f32()).collect())
}

fn read_bytes_vec(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ============================================================
// Weight I/O
// ============================================================

fn write_weight(w: &mut impl Write, weight: &Weight) -> io::Result<()> {
    write_u64(w, weight.k() as u64)?;
    write_u64(w, weight.n() as u64)?;
    let (fmt, weight) = match weight {
        Weight::Rotated(inner, _) => (inner.format_id() | ROTATED_FLAG, inner.as_ref()),
        other => (other.format_id(), other),
    };
    write_u8(w, fmt)?;
    match weight {
        Weight::F16(fw) => write_f16_vec(w, &fw.data)?,
        Weight::Q4(qw) => {
            write_bytes(w, &qw.packed)?;
            write_f16_vec(w, &qw.scales)?;
        }
        Weight::Q8(qw) => {
            let bytes = unsafe {
                std::slice::from_raw_parts(qw.data.as_ptr() as *const u8, qw.data.len())
            };
            write_bytes(w, bytes)?;
            write_f16_vec(w, &qw.scales)?;
        }
        Weight::LowBit(lw) => {
            write_bytes(w, &lw.packed)?;
            write_f16_vec(w, &lw.scales)?;
        }
        Weight::Rotated(..) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "nested rotated weight cannot be stored",
            ));
        }
    }
    Ok(())
}

/// `rot_seed` is the file's rotation seed (None for files without rotation
/// metadata, where a rotated weight is invalid).
fn read_weight(r: &mut impl Read, rot_seed: Option<u64>) -> io::Result<Weight> {
    let k = read_u64(r)? as usize;
    let n = read_u64(r)? as usize;
    let per_weight_fmt = read_u8(r)?;
    if per_weight_fmt & ROTATED_FLAG != 0 {
        let seed = rot_seed.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "rotated weight in a file without rotation metadata",
            )
        })?;
        let inner = read_weight_body(r, per_weight_fmt & !ROTATED_FLAG, k, n)?;
        return Ok(Weight::Rotated(Box::new(inner), seed));
    }
    read_weight_body(r, per_weight_fmt, k, n)
}

/// The single rotation seed shared by every rotated weight that gets
/// written, None if nothing is rotated. Mixed seeds cannot be described by
/// the one header field. A tied head is not written, so it is not counted.
fn model_rotation_seed(weights: &ModelWeights) -> io::Result<Option<u64>> {
    let head = (!weights.config.tie_word_embeddings).then_some(&weights.lm_head);
    let mut seeds = std::iter::once(&weights.embed)
        .chain(head)
        .chain(weights.layers.iter().flat_map(|l| l.matmul_weights()))
        .filter_map(Weight::rotation_seed);
    let Some(first) = seeds.next() else {
        return Ok(None);
    };
    if seeds.any(|s| s != first) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rotated weights use different seeds; one file holds one seed",
        ));
    }
    Ok(Some(first))
}

/// Rotation metadata that follows the header's format byte in v4 files.
pub fn write_rotation_meta(w: &mut impl Write, seed: u64) -> io::Result<()> {
    write_u8(w, crate::rotate::SCHEME_BLOCK_HADAMARD)?;
    write_u64(w, seed)
}

fn read_weight_body(r: &mut impl Read, per_weight_fmt: u8, k: usize, n: usize) -> io::Result<Weight> {
    match per_weight_fmt {
        0 => {
            let data = read_f16_vec(r)?;
            Ok(Weight::F16(F16Weight { data, k, n }))
        }
        1 => {
            let packed = read_bytes_vec(r)?;
            let scales = read_f16_vec(r)?;
            Ok(Weight::Q4(Q4Weight {
                packed,
                scales,
                k,
                n,
            }))
        }
        2 => {
            let raw_bytes = read_bytes_vec(r)?;
            let data: Vec<i8> = raw_bytes.into_iter().map(|b| b as i8).collect();
            let scales = read_f16_vec(r)?;
            Ok(Weight::Q8(Q8Weight {
                data,
                scales,
                k,
                n,
            }))
        }
        3 | 4 => {
            let bits = if per_weight_fmt == 3 {
                crate::quant::LowBits::Q3
            } else {
                crate::quant::LowBits::Q2
            };
            let packed = read_bytes_vec(r)?;
            let scales = read_f16_vec(r)?;
            let groups = k * n / crate::quant::GROUP;
            if scales.len() != groups || packed.len() != groups * bits.group_bytes() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{bits:?} weight [{k}, {n}] has inconsistent packed/scale sizes"),
                ));
            }
            Ok(Weight::LowBit(LowBitWeight { bits, packed, scales, k, n }))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown weight format {per_weight_fmt}"),
        )),
    }
}

// ============================================================
// Save
// ============================================================

pub fn save_model(weights: &ModelWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    let config = &weights.config;
    let format_id = weights.format_id();

    // Header
    w.write_all(MAGIC)?;
    let rot_seed = model_rotation_seed(weights)?;
    let version = if rot_seed.is_some() { VERSION_ROTATION_META } else { VERSION };
    write_u32(&mut w, version)?;
    write_u8(&mut w, format_id)?;
    if let Some(seed) = rot_seed {
        write_rotation_meta(&mut w, seed)?;
    }

    // Config
    write_u32(&mut w, config.vocab_size as u32)?;
    write_u32(&mut w, config.hidden_size as u32)?;
    write_u32(&mut w, config.num_layers as u32)?;
    write_u32(&mut w, config.num_attn_heads as u32)?;
    write_u32(&mut w, config.num_kv_heads as u32)?;
    write_u32(&mut w, config.attn_head_dim as u32)?;
    write_u32(&mut w, config.num_qk_heads as u32)?;
    write_u32(&mut w, config.num_v_heads as u32)?;
    write_u32(&mut w, config.deltanet_head_dim as u32)?;
    write_u32(&mut w, config.conv_kernel_size as u32)?;
    write_u32(&mut w, config.intermediate_size as u32)?;
    write_f64(&mut w, config.rope_theta)?;
    write_f32(&mut w, config.partial_rotary_factor)?;
    write_f64(&mut w, config.rms_norm_eps)?;
    write_u32(&mut w, config.eos_token_id)?;
    write_u8(&mut w, config.tie_word_embeddings as u8)?;
    write_u8(&mut w, config.has_vision as u8)?;

    // Layer types
    for lt in &config.layer_types {
        write_u8(&mut w, *lt as u8)?;
    }

    // Per-layer weights
    for (i, layer) in weights.layers.iter().enumerate() {
        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                write_weight(&mut w, &lw.in_proj_qkv)?;
                write_weight(&mut w, &lw.in_proj_a)?;
                write_weight(&mut w, &lw.in_proj_b)?;
                write_weight(&mut w, &lw.in_proj_z)?;
                write_weight(&mut w, &lw.out_proj)?;
                write_f32_vec(&mut w, &lw.conv1d_weight)?;
                write_f32_vec(&mut w, &lw.a_log)?;
                write_f32_vec(&mut w, &lw.dt_bias)?;
                write_f16_vec(&mut w, &lw.attn_norm_weight)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
            HybridLayerWeights::FullAttn(lw) => {
                write_weight(&mut w, &lw.q_proj)?;
                write_weight(&mut w, &lw.k_proj)?;
                write_weight(&mut w, &lw.v_proj)?;
                write_weight(&mut w, &lw.o_proj)?;
                write_f16_vec(&mut w, &lw.q_norm)?;
                write_f16_vec(&mut w, &lw.k_norm)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
        }
        if i % 6 == 0 {
            eprintln!("  Saved layer {i}/{}...", config.num_layers);
        }
    }

    // Global
    write_weight(&mut w, &weights.embed)?;
    if !config.tie_word_embeddings {
        write_weight(&mut w, &weights.lm_head)?;
    }
    write_f16_vec(&mut w, &weights.final_norm)?;
    write_f32_vec(&mut w, &weights.rope_cos)?;
    write_f32_vec(&mut w, &weights.rope_sin)?;

    // Vision weights (stored as f16 on disk, loaded as f32 for compute)
    if let Some(vision) = &weights.vision {
        eprintln!("  Saving vision encoder (f16)...");
        write_f32_as_f16(&mut w, &vision.patch_proj_weight)?;
        write_f32_as_f16(&mut w, &vision.patch_proj_bias)?;
        write_f32_as_f16(&mut w, &vision.pos_embed)?;
        for b in &vision.blocks {
            write_f32_as_f16(&mut w, &b.norm1_weight)?;
            write_f32_as_f16(&mut w, &b.norm1_bias)?;
            write_f32_as_f16(&mut w, &b.qkv_weight)?;
            write_f32_as_f16(&mut w, &b.qkv_bias)?;
            write_f32_as_f16(&mut w, &b.proj_weight)?;
            write_f32_as_f16(&mut w, &b.proj_bias)?;
            write_f32_as_f16(&mut w, &b.norm2_weight)?;
            write_f32_as_f16(&mut w, &b.norm2_bias)?;
            write_f32_as_f16(&mut w, &b.fc1_weight)?;
            write_f32_as_f16(&mut w, &b.fc1_bias)?;
            write_f32_as_f16(&mut w, &b.fc2_weight)?;
            write_f32_as_f16(&mut w, &b.fc2_bias)?;
        }
        write_f32_as_f16(&mut w, &vision.merger.norm_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.norm_bias)?;
        write_f32_as_f16(&mut w, &vision.merger.fc1_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.fc1_bias)?;
        write_f32_as_f16(&mut w, &vision.merger.fc2_weight)?;
        write_f32_as_f16(&mut w, &vision.merger.fc2_bias)?;
    }

    w.flush()?;
    Ok(())
}

// ============================================================
// Load
// ============================================================

pub fn load_model(path: &Path) -> io::Result<ModelWeights> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    // Header. Accept QOR4 (4B) and QR08 (0.8B). Same Qwen3 family, identical
    // layer-config layout after the magic — but the vision encoder dims
    // differ (24 blocks @ 1024-d for 4B vs 12 blocks @ 768-d for 0.8B).
    // We branch on the magic to pick the right vision-config defaults
    // before the optional vision section gets read.
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    let is_mini = match &magic {
        m if m == MAGIC => false,
        b"QR08" => true,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid magic {:?} (expected QOR4 or QR08)",
                    std::str::from_utf8(&magic).unwrap_or("?")
                ),
            ));
        }
    };
    let version = read_u32(&mut r)?;
    if !(1..=VERSION_ROTATION_META).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported version {version} (expected 1..={VERSION_ROTATION_META})"),
        ));
    }
    let format_id = read_u8(&mut r)?;
    let rot_seed = match version {
        VERSION_ROTATED => Some(crate::rotate::DEFAULT_SEED),
        VERSION_ROTATION_META => {
            let scheme = read_u8(&mut r)?;
            if scheme != crate::rotate::SCHEME_BLOCK_HADAMARD {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown rotation scheme {scheme}"),
                ));
            }
            Some(read_u64(&mut r)?)
        }
        _ => None,
    };

    // Config
    let vocab_size = read_u32(&mut r)? as usize;
    let hidden_size = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let num_attn_heads = read_u32(&mut r)? as usize;
    let num_kv_heads = read_u32(&mut r)? as usize;
    let attn_head_dim = read_u32(&mut r)? as usize;
    let num_qk_heads = read_u32(&mut r)? as usize;
    let num_v_heads = read_u32(&mut r)? as usize;
    let deltanet_head_dim = read_u32(&mut r)? as usize;
    let conv_kernel_size = read_u32(&mut r)? as usize;
    let intermediate_size = read_u32(&mut r)? as usize;
    let rope_theta = read_f64(&mut r)?;
    let partial_rotary_factor = read_f32(&mut r)?;
    let rms_norm_eps = read_f64(&mut r)?;
    let eos_token_id = read_u32(&mut r)?;
    let tie_word_embeddings = read_u8(&mut r)? != 0;
    let has_vision = read_u8(&mut r)? != 0;

    // Layer types
    let mut layer_types = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let lt = read_u8(&mut r)?;
        layer_types.push(match lt {
            0 => LayerType::FullAttn,
            1 => LayerType::DeltaNet,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown layer type {lt}"),
                ));
            }
        });
    }

    let config = IncordConfig {
        vocab_size,
        hidden_size,
        num_layers,
        num_attn_heads,
        num_kv_heads,
        attn_head_dim,
        num_qk_heads,
        num_v_heads,
        deltanet_head_dim,
        conv_kernel_size,
        intermediate_size,
        rope_theta,
        partial_rotary_factor,
        rms_norm_eps,
        eos_token_id,
        bos_token_id: 0,
        max_position_embeddings: 131_072,
        tie_word_embeddings,
        layer_types: layer_types.clone(),
        has_vision,
        // Vision defaults — pick the variant matching the magic.
        // 4B (QOR4): 1024-d, 24 blocks, 16 heads, 4096 ffn.
        // 0.8B (QR08): 768-d, 12 blocks, 12 heads, 3072 ffn.
        // Other dims (patch_size, merge, position embeds) are family-shared.
        vision_hidden: if is_mini { 768 } else { 1024 },
        vision_layers: if is_mini { 12 } else { 24 },
        vision_heads: if is_mini { 12 } else { 16 },
        vision_ffn: if is_mini { 3072 } else { 4096 },
        patch_size: 16,
        spatial_merge_size: 2,
        num_position_embeddings: 2304,
    };

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for (i, lt) in layer_types.iter().enumerate() {
        if i % 6 == 0 {
            eprintln!("  Loading layer {i}/{num_layers}...");
        }
        match lt {
            LayerType::DeltaNet => {
                layers.push(HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                    in_proj_qkv: read_weight(&mut r, rot_seed)?,
                    in_proj_a: read_weight(&mut r, rot_seed)?,
                    in_proj_b: read_weight(&mut r, rot_seed)?,
                    in_proj_z: read_weight(&mut r, rot_seed)?,
                    out_proj: read_weight(&mut r, rot_seed)?,
                    conv1d_weight: read_f32_vec(&mut r)?,
                    a_log: read_f32_vec(&mut r)?,
                    dt_bias: read_f32_vec(&mut r)?,
                    attn_norm_weight: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, rot_seed)?,
                    up_proj: read_weight(&mut r, rot_seed)?,
                    down_proj: read_weight(&mut r, rot_seed)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
            LayerType::FullAttn => {
                layers.push(HybridLayerWeights::FullAttn(FullAttnLayerWeights {
                    q_proj: read_weight(&mut r, rot_seed)?,
                    k_proj: read_weight(&mut r, rot_seed)?,
                    v_proj: read_weight(&mut r, rot_seed)?,
                    o_proj: read_weight(&mut r, rot_seed)?,
                    q_norm: read_f16_vec(&mut r)?,
                    k_norm: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, rot_seed)?,
                    up_proj: read_weight(&mut r, rot_seed)?,
                    down_proj: read_weight(&mut r, rot_seed)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
        }
    }

    // Global
    let embed = read_weight(&mut r, rot_seed)?;
    let lm_head = if !tie_word_embeddings {
        read_weight(&mut r, rot_seed)?
    } else {
        embed.clone()
    };
    let final_norm = read_f16_vec(&mut r)?;
    let rope_cos = read_f32_vec(&mut r)?;
    let rope_sin = read_f32_vec(&mut r)?;

    // Vision weights: v1=f32 on disk, v2=f16 on disk → both loaded as f32 for compute
    let vision = if has_vision {
        eprintln!("  Loading vision encoder...");
        use crate::vision::*;
        // Helper closure: read vision vec based on file version
        let read_vis = |r: &mut BufReader<File>| -> io::Result<Vec<f32>> {
            if version >= 2 {
                read_f16_as_f32(r)
            } else {
                read_f32_vec(r)
            }
        };
        let patch_proj_weight = read_vis(&mut r)?;
        let patch_proj_bias = read_vis(&mut r)?;
        let pos_embed = read_vis(&mut r)?;
        let mut blocks = Vec::with_capacity(config.vision_layers);
        for _ in 0..config.vision_layers {
            blocks.push(VisionBlock {
                norm1_weight: read_vis(&mut r)?,
                norm1_bias: read_vis(&mut r)?,
                qkv_weight: read_vis(&mut r)?,
                qkv_bias: read_vis(&mut r)?,
                proj_weight: read_vis(&mut r)?,
                proj_bias: read_vis(&mut r)?,
                norm2_weight: read_vis(&mut r)?,
                norm2_bias: read_vis(&mut r)?,
                fc1_weight: read_vis(&mut r)?,
                fc1_bias: read_vis(&mut r)?,
                fc2_weight: read_vis(&mut r)?,
                fc2_bias: read_vis(&mut r)?,
            });
        }
        let merger = Merger {
            norm_weight: read_vis(&mut r)?,
            norm_bias: read_vis(&mut r)?,
            fc1_weight: read_vis(&mut r)?,
            fc1_bias: read_vis(&mut r)?,
            fc2_weight: read_vis(&mut r)?,
            fc2_bias: read_vis(&mut r)?,
        };
        Some(VisionEncoder {
            patch_proj_weight,
            patch_proj_bias,
            pos_embed,
            blocks,
            merger,
        })
    } else {
        None
    };

    let format_name = match format_id {
        0 => "F16",
        1 => "Q4",
        2 => "Q8",
        3 => "Q3",
        4 => "Q2",
        _ => "unknown",
    };
    eprintln!(
        "  Loaded {format_name} model: {num_layers} layers, vocab={vocab_size}, hidden={hidden_size}"
    );

    Ok(ModelWeights {
        layers,
        embed,
        lm_head,
        vocab: vocab_size,
        hidden: hidden_size,
        final_norm,
        rope_cos,
        rope_sin,
        config,
        format_name,
        vision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(weight: &Weight, seed: Option<u64>) -> Weight {
        let mut buf = Vec::new();
        write_weight(&mut buf, weight).unwrap();
        read_weight(&mut buf.as_slice(), seed).unwrap()
    }

    fn q4(k: usize, n: usize) -> Weight {
        let (packed, scales) = crate::quant::quantize_q4(
            &(0..k * n).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect::<Vec<_>>(),
            k,
            n,
            crate::quant::Rounding::Mse,
        );
        Weight::Q4(Q4Weight { packed, scales, k, n })
    }

    #[test]
    fn rotated_weight_survives_save_and_load() {
        let w = Weight::Rotated(Box::new(q4(64, 32)), 7);
        let back = roundtrip(&w, Some(7));
        assert_eq!(back.rotation_seed(), Some(7));
        assert_eq!((back.k(), back.n(), back.format_id()), (64, 32, 1));
        match (&w, &back) {
            (Weight::Rotated(a, _), Weight::Rotated(b, _)) => match (a.as_ref(), b.as_ref()) {
                (Weight::Q4(a), Weight::Q4(b)) => {
                    assert_eq!(a.packed, b.packed);
                    assert_eq!(a.scales, b.scales);
                }
                _ => panic!("inner format changed"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn plain_weight_has_no_rotation_flag() {
        let w = Weight::F16(F16Weight { data: vec![f16::from_f32(1.0); 32], k: 1, n: 32 });
        let mut buf = Vec::new();
        write_weight(&mut buf, &w).unwrap();
        assert_eq!(buf[16], 0, "format byte must stay 0 for plain F16");
        assert!(!roundtrip(&w, None).is_rotated());
    }

    #[test]
    fn rotated_weight_without_metadata_is_rejected() {
        let mut buf = Vec::new();
        write_weight(&mut buf, &Weight::Rotated(Box::new(q4(32, 32)), 7)).unwrap();
        assert!(read_weight(&mut buf.as_slice(), None).is_err());
    }

    /// Whole-model round trip: the seed must come back from the v4 header,
    /// and a plain model must still be written as v2.
    #[test]
    fn model_rotation_seed_survives_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = crate::gemv::mock::tiny_weights();
        // Untied like the real model, so the head is written on its own.
        w.config.tie_word_embeddings = false;
        let plain_path = dir.path().join("plain.incord");
        save_model(&w, &plain_path).unwrap();
        let header = std::fs::read(&plain_path).unwrap();
        assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), VERSION);

        let seed = 0xDEAD_BEEF_u64;
        w.lm_head = Weight::Rotated(Box::new(w.lm_head.clone()), seed);
        let rot_path = dir.path().join("rot.incord");
        save_model(&w, &rot_path).unwrap();
        let header = std::fs::read(&rot_path).unwrap();
        assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), VERSION_ROTATION_META);
        let back = load_model(&rot_path).unwrap();
        assert_eq!(back.lm_head.rotation_seed(), Some(seed));
        assert!(!back.embed.is_rotated());
    }

    #[test]
    fn mixed_seeds_cannot_be_saved() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = crate::gemv::mock::tiny_weights();
        w.config.tie_word_embeddings = false;
        w.lm_head = Weight::Rotated(Box::new(w.lm_head.clone()), 1);
        w.embed = Weight::Rotated(Box::new(w.embed.clone()), 2);
        assert!(save_model(&w, &dir.path().join("x.incord")).is_err());
    }
}
