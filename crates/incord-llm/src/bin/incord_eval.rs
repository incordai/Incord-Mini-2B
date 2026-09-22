//! incord-eval — measure quantization quality against a full-precision reference.
//!
//! Build a reference once from the F16 model, then score any candidate:
//!
//! ```text
//! incord-eval reference --model model_f16.incord --tokenizer tokenizer.json \
//!     --corpus eval/corpus --out eval-data/ref.bin [--seq-len 512] [--chunks-per-file 2]
//! incord-eval score --model model_q4.incord --ref eval-data/ref.bin [--json eval-data/results.jsonl]
//! ```
//!
//! Metrics and file format: see `incord::eval`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use incord::eval::{
    DEFAULT_TOP_K, RefFile, RefPosition, ScoreAccumulator, corpus_chunks, log_softmax,
};
use incord::gemv::{HybridCache, ModelWeights, lm_head_batch, prefill_hidden_all};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("reference") => cmd_reference(&args[2..]),
        Some("score") => cmd_score(&args[2..]),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

const USAGE: &str = "usage:
  incord-eval reference --model <f16.incord> --tokenizer <tokenizer.json> --corpus <dir|file> --out <ref.bin>
                        [--seq-len 512] [--chunks-per-file 2] [--top-k 256]
  incord-eval score --model <candidate.incord> --ref <ref.bin> [--json <results.jsonl>] [--label <name>]";

/// `--key value` lookup.
fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn req<'a>(args: &'a [String], key: &str) -> Result<&'a str, String> {
    opt(args, key).ok_or_else(|| format!("missing {key}\n{USAGE}"))
}

fn num(args: &[String], key: &str, default: usize) -> Result<usize, String> {
    match opt(args, key) {
        Some(v) => v.parse().map_err(|_| format!("{key}: not a number: {v}")),
        None => Ok(default),
    }
}

fn load_model(path: &str) -> Result<ModelWeights, String> {
    let t0 = Instant::now();
    let w = incord::save::load_model(Path::new(path)).map_err(|e| format!("load {path}: {e}"))?;
    eprintln!(
        "Loaded {path} ({} format, {} MB) in {:.1?}",
        w.format_name,
        w.memory_bytes() / (1024 * 1024),
        t0.elapsed()
    );
    Ok(w)
}

/// Full-vocabulary log-probabilities for every position of one chunk.
/// Calls `f(position, logprobs)` for positions 0..len-1 (each predicts the
/// next token), so only one vocabulary-sized buffer is alive at a time.
fn for_each_position(w: &ModelWeights, chunk: &[u32], mut f: impl FnMut(usize, &[f32])) {
    let t0 = Instant::now();
    let mut cache = HybridCache::new(&w.config);
    let hidden = prefill_hidden_all(w, chunk, &mut cache);
    let t_prefill = t0.elapsed();
    let mut t_head = std::time::Duration::ZERO;
    // Positions 0..len-1 predict the next token; the head runs on tiles of
    // HEAD_TILE positions at a time to bound memory (tile x vocab floats).
    const HEAD_TILE: usize = 32;
    let scored = chunk.len() - 1;
    for p0 in (0..scored).step_by(HEAD_TILE) {
        let cnt = (scored - p0).min(HEAD_TILE);
        let th = Instant::now();
        let mut logits = lm_head_batch(&hidden[p0 * w.hidden..(p0 + cnt) * w.hidden], cnt, &w.lm_head);
        t_head += th.elapsed();
        for (i, row) in logits.chunks_mut(w.vocab).enumerate() {
            log_softmax(row);
            f(p0 + i, row);
        }
    }
    eprintln!(
        "  chunk done in {:.1?}: prefill {:.1?}, output head {:.1?}, scoring {:.1?}",
        t0.elapsed(),
        t_prefill,
        t_head,
        t0.elapsed() - t_prefill - t_head
    );
}

fn cmd_reference(args: &[String]) -> Result<(), String> {
    let model = req(args, "--model")?;
    let tok_path = req(args, "--tokenizer")?;
    let corpus = req(args, "--corpus")?;
    let out = req(args, "--out")?;
    let seq_len = num(args, "--seq-len", 512)?;
    let per_file = num(args, "--chunks-per-file", 2)?;
    let top_k = num(args, "--top-k", DEFAULT_TOP_K)?;
    if seq_len < 2 {
        return Err("--seq-len must be at least 2".into());
    }

    let tokenizer = incord::tokenizer::IncordTokenizer::from_file(tok_path)
        .map_err(|e| format!("load tokenizer {tok_path}: {e}"))?;
    let w = load_model(model)?;
    let chunks =
        corpus_chunks(&tokenizer, Path::new(corpus), seq_len, per_file, w.config.bos_token_id)?;

    let t0 = Instant::now();
    let mut positions = Vec::new();
    for (ci, chunk) in chunks.iter().enumerate() {
        eprintln!("Reference chunk {}/{}", ci + 1, chunks.len());
        for_each_position(&w, chunk, |pos, lp| {
            positions.push(RefPosition::from_logprobs(lp, chunk[pos + 1], top_k));
        });
    }
    let file = RefFile {
        top_k,
        chunks,
        positions,
    };

    if let Some(dir) = Path::new(out).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let mut wr = BufWriter::new(File::create(out).map_err(|e| format!("create {out}: {e}"))?);
    file.write_to(&mut wr)
        .and_then(|_| wr.flush())
        .map_err(|e| format!("write {out}: {e}"))?;
    eprintln!(
        "Wrote {out}: {} chunks, {} positions, top-{top_k}, in {:.1?}",
        file.chunks.len(),
        file.positions.len(),
        t0.elapsed()
    );
    Ok(())
}

fn cmd_score(args: &[String]) -> Result<(), String> {
    let model = req(args, "--model")?;
    let ref_path = req(args, "--ref")?;
    let label = opt(args, "--label").unwrap_or(model);

    let reference = {
        let f = File::open(ref_path).map_err(|e| format!("open {ref_path}: {e}"))?;
        RefFile::read_from(&mut BufReader::new(f)).map_err(|e| format!("read {ref_path}: {e}"))?
    };
    let w = load_model(model)?;
    if let Some(&max_id) = reference.chunks.iter().flatten().max() {
        if max_id as usize >= w.vocab {
            return Err(format!("reference token {max_id} is outside model vocab {}", w.vocab));
        }
    }

    let t0 = Instant::now();
    let mut acc = ScoreAccumulator::default();
    let mut next = 0usize;
    for (ci, chunk) in reference.chunks.iter().enumerate() {
        eprintln!("Score chunk {}/{}", ci + 1, reference.chunks.len());
        for_each_position(&w, chunk, |pos, lp| {
            acc.add(&reference.positions[next], lp, chunk[pos + 1]);
            next += 1;
        });
    }
    let s = acc.finish();
    let size = std::fs::metadata(model).map(|m| m.len()).unwrap_or(0);

    println!("model        {label}");
    println!("file size    {:.3} GB", size as f64 / 1e9);
    println!("positions    {}", s.positions);
    println!("KL mean      {:.6}", s.kl_mean);
    println!("KL median    {:.6}", s.kl_median);
    println!("KL p99       {:.6}", s.kl_p99);
    println!("KL max       {:.6}", s.kl_max);
    println!("top-1 agree  {:.2}%", s.top1_agreement * 100.0);
    println!("PPL ref      {:.4}", s.ppl_ref);
    println!("PPL cand     {:.4}  (ratio {:.4})", s.ppl_cand, s.ppl_cand / s.ppl_ref);
    eprintln!("Scored in {:.1?}", t0.elapsed());

    if let Some(json) = opt(args, "--json") {
        let line = serde_json::json!({
            "label": label,
            "model": model,
            "file_bytes": size,
            "positions": s.positions,
            "kl_mean": s.kl_mean,
            "kl_median": s.kl_median,
            "kl_p99": s.kl_p99,
            "kl_max": s.kl_max,
            "top1_agreement": s.top1_agreement,
            "ppl_ref": s.ppl_ref,
            "ppl_cand": s.ppl_cand,
        });
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(json)
            .map_err(|e| format!("open {json}: {e}"))?;
        writeln!(f, "{line}").map_err(|e| format!("write {json}: {e}"))?;
    }
    Ok(())
}
