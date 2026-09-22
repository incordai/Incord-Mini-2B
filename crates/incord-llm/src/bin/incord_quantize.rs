//! incord-quantize — GPTQ quantization driven by calibration text
//! (roadmap Milestone 4).
//!
//! ```text
//! incord-quantize --model model_f16.incord --tokenizer tokenizer.json \
//!     --calib eval/calib --out model_q4_gptq.incord \
//!     [--format q4|q8|f16] [--plan SPEC] [--rotate] [--rounding mse|rtn] \
//!     [--seq-len 512] [--chunks-per-file 64] [--damp 0.01] [--block 128] [--no-act-order]
//! ```
//!
//! `--plan` overrides `--format` per tensor, e.g. `table=q8,down@0-3=q8`
//! (syntax in `incord::plan`).
//!
//! Sequential, layer by layer: calibration activations run through layer
//! `i` (full precision) while every projection input is folded into a
//! Hessian `H = XᵀX` per input site; the layer's weights are then quantized
//! with GPTQ, and the activations are pushed through the *quantized* layer
//! so layer `i + 1` is calibrated on what it will really receive. The
//! embedding is quantized first for the same reason. The output head and
//! embedding use plain (MSE) rounding. Input must be an unrotated F16 model.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use half::f16;
use incord::eval::corpus_chunks;
use incord::gemv::{
    F16Weight, HybridCache, HybridLayerWeights, LinearObserver, LinearSite, ModelWeights,
    NoObserver, Q4Weight, Q8Weight, Weight, embed_lookup, prefill_layer,
};
use incord::gptq::{GptqConfig, Hessian, RowQuantizer, gptq};
use incord::plan::{Class, Fmt, Plan};
use incord::rotate::DEFAULT_SEED;
use incord::quant::{
    GROUP, Rounding, dequantize_q4, dequantize_q8, quantize_q4, quantize_q4_row, quantize_q8,
    quantize_q8_row,
};

const USAGE: &str = "usage: incord-quantize --model <f16.incord> --tokenizer <tokenizer.json> \
--calib <dir|file> --out <out.incord> [--format q4|q8|f16] [--plan SPEC] [--rotate] \
[--rounding mse|rtn] [--seq-len 512] [--chunks-per-file 64] [--damp 0.01] [--block 128] \
[--no-act-order]";

struct Opts {
    model: String,
    tokenizer: String,
    calib: String,
    out: String,
    plan: Plan,
    plan_spec: String,
    rotate: bool,
    rounding: Rounding,
    seq_len: usize,
    per_file: usize,
    gptq: GptqConfig,
}

fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn req(args: &[String], key: &str) -> Result<String, String> {
    opt(args, key)
        .map(str::to_string)
        .ok_or_else(|| format!("missing {key}\n{USAGE}"))
}

fn parsed<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    match opt(args, key) {
        Some(v) => v.parse().map_err(|e| format!("{key} {v}: {e}")),
        None => Ok(default),
    }
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let format = parsed(args, "--format", Fmt::Q4)?;
    let plan_spec = opt(args, "--plan").unwrap_or("").to_string();
    Ok(Opts {
        model: req(args, "--model")?,
        tokenizer: req(args, "--tokenizer")?,
        calib: req(args, "--calib")?,
        out: req(args, "--out")?,
        plan: Plan::parse(format, &plan_spec).map_err(|e| format!("--plan: {e}"))?,
        plan_spec,
        rotate: args.iter().any(|a| a == "--rotate"),
        rounding: parsed(args, "--rounding", Rounding::Mse)?,
        seq_len: parsed(args, "--seq-len", 512usize)?,
        per_file: parsed(args, "--chunks-per-file", 64usize)?,
        gptq: GptqConfig {
            damp: parsed(args, "--damp", 0.01f64)?,
            block: parsed(args, "--block", 128usize)?,
            act_order: !args.iter().any(|a| a == "--no-act-order"),
        },
    })
}

/// Collects one Hessian per projection site of the current layer.
struct Collector {
    rotate: bool,
    h: HashMap<LinearSite, Hessian>,
}

impl LinearObserver for Collector {
    fn observe(&mut self, site: LinearSite, x: &[f32], rows: usize) {
        let k = x.len() / rows;
        let h = self.h.entry(site).or_insert_with(|| Hessian::new(k));
        if self.rotate {
            // Rotated weights see rotated inputs: accumulate H of x·R.
            let mut xr = x.to_vec();
            xr.chunks_mut(k).for_each(|row| incord::rotate::forward(row, DEFAULT_SEED));
            h.add(&xr, rows);
        } else {
            h.add(x, rows);
        }
    }
}

struct Q4Rows(Rounding);

impl RowQuantizer for Q4Rows {
    type Code = (Vec<u8>, Vec<f16>);
    fn quantize(&self, row: &[f32], deq: &mut [f32]) -> Self::Code {
        let n = row.len();
        let mut p = vec![0u8; n / 2];
        let mut s = vec![f16::from_f32(0.0); n / GROUP];
        quantize_q4_row(row, self.0, &mut p, &mut s);
        deq.copy_from_slice(&dequantize_q4(&p, &s, 1, n));
        (p, s)
    }
}

struct Q8Rows(Rounding);

impl RowQuantizer for Q8Rows {
    type Code = (Vec<i8>, Vec<f16>);
    fn quantize(&self, row: &[f32], deq: &mut [f32]) -> Self::Code {
        let n = row.len();
        let mut q = vec![0i8; n];
        let mut s = vec![f16::from_f32(0.0); n / GROUP];
        quantize_q8_row(row, self.0, &mut q, &mut s);
        deq.copy_from_slice(&dequantize_q8(&q, &s, 1, n));
        (q, s)
    }
}

/// f32 copy of an unrotated F16 weight.
fn f16_data(w: &Weight, what: &str) -> Result<(Vec<f32>, usize, usize), String> {
    match w {
        Weight::F16(f) => Ok((f.data.iter().map(|v| v.to_f32()).collect(), f.k, f.n)),
        _ => Err(format!("{what}: input model must be unrotated F16")),
    }
}

fn wrap(w: Weight, rotated: bool) -> Weight {
    if rotated { Weight::Rotated(Box::new(w), DEFAULT_SEED) } else { w }
}

/// GPTQ-quantize one matmul weight `[k, n]` to `fmt` with the Hessian of
/// its input.
fn quantize_matmul(w: &Weight, h: &Hessian, fmt: Fmt, o: &Opts, what: &str) -> Result<Weight, String> {
    let (mut data, k, n) = f16_data(w, what)?;
    if fmt == Fmt::F16 || !n.is_multiple_of(GROUP) {
        return Ok(w.clone()); // F16 by plan, or too narrow for groups (as in the converter)
    }
    if o.rotate {
        incord::rotate::rotate_columns(&mut data, k, n, DEFAULT_SEED);
    }
    let q = match fmt {
        Fmt::Q4 => {
            let res = gptq(&data, k, n, h, o.gptq, &Q4Rows(o.rounding))?;
            let (packed, scales): (Vec<Vec<u8>>, Vec<Vec<f16>>) = res.rows.into_iter().unzip();
            Weight::Q4(Q4Weight { packed: packed.concat(), scales: scales.concat(), k, n })
        }
        Fmt::Q8 => {
            let res = gptq(&data, k, n, h, o.gptq, &Q8Rows(o.rounding))?;
            let (data, scales): (Vec<Vec<i8>>, Vec<Vec<f16>>) = res.rows.into_iter().unzip();
            Weight::Q8(Q8Weight { data: data.concat(), scales: scales.concat(), k, n })
        }
        Fmt::F16 => unreachable!(),
    };
    Ok(wrap(q, o.rotate))
}

/// Plain-rounded `[vocab, hidden]` table (embedding or output head).
fn quantize_table(w: &Weight, fmt: Fmt, o: &Opts, what: &str) -> Result<Weight, String> {
    let (mut data, k, n) = f16_data(w, what)?;
    if o.rotate {
        incord::rotate::rotate_rows(&mut data, n, DEFAULT_SEED);
    }
    let q = match fmt {
        Fmt::F16 => Weight::F16(F16Weight {
            data: data.iter().map(|&v| f16::from_f32(v)).collect(),
            k,
            n,
        }),
        Fmt::Q4 => {
            let (packed, scales) = quantize_q4(&data, k, n, o.rounding);
            Weight::Q4(Q4Weight { packed, scales, k, n })
        }
        Fmt::Q8 => {
            let (data, scales) = quantize_q8(&data, k, n, o.rounding);
            Weight::Q8(Q8Weight { data, scales, k, n })
        }
    };
    Ok(wrap(q, o.rotate))
}

/// Every matmul weight of a layer with the site that feeds it.
fn layer_sites(layer: &mut HybridLayerWeights) -> Vec<(LinearSite, &'static str, &mut Weight)> {
    use LinearSite::*;
    match layer {
        HybridLayerWeights::FullAttn(f) => vec![
            (AttnIn, "q_proj", &mut f.q_proj),
            (AttnIn, "k_proj", &mut f.k_proj),
            (AttnIn, "v_proj", &mut f.v_proj),
            (AttnOut, "o_proj", &mut f.o_proj),
            (MlpIn, "gate_proj", &mut f.gate_proj),
            (MlpIn, "up_proj", &mut f.up_proj),
            (MlpDown, "down_proj", &mut f.down_proj),
        ],
        HybridLayerWeights::DeltaNet(d) => vec![
            (AttnIn, "in_proj_qkv", &mut d.in_proj_qkv),
            (AttnIn, "in_proj_a", &mut d.in_proj_a),
            (AttnIn, "in_proj_b", &mut d.in_proj_b),
            (AttnIn, "in_proj_z", &mut d.in_proj_z),
            (AttnOut, "out_proj", &mut d.out_proj),
            (MlpIn, "gate_proj", &mut d.gate_proj),
            (MlpIn, "up_proj", &mut d.up_proj),
            (MlpDown, "down_proj", &mut d.down_proj),
        ],
    }
}

fn run(o: &Opts) -> Result<(), String> {
    let t_all = Instant::now();
    let tokenizer = incord::tokenizer::IncordTokenizer::from_file(&o.tokenizer)
        .map_err(|e| format!("load tokenizer {}: {e}", o.tokenizer))?;
    let mut w: ModelWeights =
        incord::save::load_model(Path::new(&o.model)).map_err(|e| format!("load {}: {e}", o.model))?;
    if w.has_rotated() {
        return Err("input model must be unrotated F16".into());
    }
    let chunks = corpus_chunks(
        &tokenizer,
        Path::new(&o.calib),
        o.seq_len,
        o.per_file,
        w.config.bos_token_id,
    )?;
    let tokens: usize = chunks.iter().map(Vec::len).sum();
    eprintln!(
        "Calibration: {} chunks, {tokens} tokens | default {:?} plan '{}' rotate {} {:?}",
        chunks.len(),
        o.plan.default,
        o.plan_spec,
        o.rotate,
        o.gptq
    );

    // Embedding first, so calibration runs on the embedding the model ships.
    let embed_fmt = o.plan.format_for(Class::Embed, None);
    w.embed = quantize_table(&w.embed, embed_fmt, o, "embedding")?;
    let hidden = w.hidden;
    let mut xs: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| c.iter().flat_map(|&t| embed_lookup(&w.embed, t as usize, hidden)).collect())
        .collect();

    let num_layers = w.config.num_layers;
    for i in 0..num_layers {
        let t0 = Instant::now();
        let mut col = Collector { rotate: o.rotate, h: HashMap::new() };
        for (x, c) in xs.iter().zip(&chunks) {
            let mut xc = x.clone();
            let mut cache = HybridCache::new(&w.config);
            prefill_layer(&w, i, &mut xc, c.len(), &mut cache, &mut col);
        }
        let t_obs = t0.elapsed();

        for (site, name, weight) in layer_sites(&mut w.layers[i]) {
            let h = col.h.get(&site).ok_or_else(|| format!("layer {i}: no input seen for {site:?}"))?;
            let class = Class::from_proj_name(name).expect("layer_sites uses known names");
            let fmt = o.plan.format_for(class, Some(i));
            *weight = quantize_matmul(weight, h, fmt, o, &format!("layer {i} {name}"))?;
        }
        let t_q = t0.elapsed() - t_obs;

        // Advance the calibration stream through the quantized layer.
        for (x, c) in xs.iter_mut().zip(&chunks) {
            let mut cache = HybridCache::new(&w.config);
            prefill_layer(&w, i, x, c.len(), &mut cache, &mut NoObserver);
        }
        eprintln!(
            "Layer {}/{num_layers} done in {:.1?} (observe {:.1?}, gptq {:.1?}, advance {:.1?})",
            i + 1,
            t0.elapsed(),
            t_obs,
            t_q,
            t0.elapsed() - t_obs - t_q
        );
    }

    let head_fmt = o.plan.format_for(Class::Head, None);
    w.lm_head = quantize_table(&w.lm_head, head_fmt, o, "output head")?;
    incord::save::save_model(&w, Path::new(&o.out)).map_err(|e| format!("save {}: {e}", o.out))?;
    eprintln!("Wrote {} in {:.1?}", o.out, t_all.elapsed());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = parse(&args).and_then(|o| run(&o)) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
