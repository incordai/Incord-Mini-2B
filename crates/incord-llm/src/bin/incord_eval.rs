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
//!
//! Task benchmarks (ARC, MMLU, HellaSwag, WinoGrande, GSM8K; see
//! `incord::tasks`) run per model and are then compared question by question:
//!
//! ```text
//! incord-eval tasks --model model_f16.incord --tokenizer tokenizer.json \
//!     --items eval-data/tasks-f16.jsonl [--tasks arc,mmlu] [--limit 200]
//! incord-eval compare --ref eval-data/tasks-f16.jsonl --cand eval-data/tasks-q4.jsonl
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use incord::eval::{
    DEFAULT_TOP_K, RefFile, RefPosition, ScoreAccumulator, corpus_chunks, log_softmax,
};
use incord::gemv::{
    HybridCache, ModelWeights, forward_decode, lm_head_batch, prefill, prefill_hidden_all,
};
use incord::tasks::{self, GenItem, Items, McItem, Task};
use incord::tokenizer::IncordTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("reference") => cmd_reference(&args[2..]),
        Some("score") => cmd_score(&args[2..]),
        Some("tasks") => cmd_tasks(&args[2..]),
        Some("compare") => cmd_compare(&args[2..]),
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
  incord-eval score --model <candidate.incord> --ref <ref.bin> [--json <results.jsonl>] [--label <name>]
  incord-eval tasks --model <m.incord> --tokenizer <tokenizer.json> --items <items.jsonl>
                    [--data eval-data/tasks] [--tasks arc,mmlu,hellaswag,winogrande,gsm8k]
                    [--limit N] [--max-new 384] [--json <summary.jsonl>] [--label <name>]
  incord-eval compare --ref <items.jsonl> --cand <items.jsonl>";

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

// ============================================================
// Task benchmarks
// ============================================================

/// Questions per task when `--limit` is not given: enough that one standard
/// error is about 2-3 points, small enough for a CPU run.
fn default_limit(task: Task) -> usize {
    match task {
        Task::Arc => 300,
        Task::Mmlu => 570, // 10 per subject on average
        Task::Hellaswag => 400,
        Task::Winogrande => 400,
        Task::Gsm8k => 100,
    }
}

/// Log-likelihood of each choice (per character when `per_char`).
fn score_mc(
    w: &ModelWeights,
    tok: &IncordTokenizer,
    item: &McItem,
    per_char: bool,
) -> Result<Vec<f64>, String> {
    let bos = w.config.bos_token_id;
    let encode_ctx = |s: &str| {
        let mut ids = vec![bos];
        ids.extend(tok.encode(s));
        ids
    };
    let conts: Vec<Vec<u32>> = item.continuations.iter().map(|c| tok.encode(c)).collect();
    if conts.iter().any(Vec::is_empty) {
        return Err(format!("{}: empty continuation", item.id));
    }

    // One shared context and single-token answers (lettered choices): one
    // prefill scores every choice from the last position's distribution.
    if item.contexts.len() == 1 && conts.iter().all(|c| c.len() == 1) {
        let mut cache = HybridCache::new(&w.config);
        let mut lp = prefill(w, &encode_ctx(item.context(0)), &mut cache);
        log_softmax(&mut lp);
        return Ok(conts.iter().map(|c| lp[c[0] as usize] as f64).collect());
    }

    let mut scores = Vec::with_capacity(item.choices());
    for (i, cont) in conts.iter().enumerate() {
        let mut seq = encode_ctx(item.context(i));
        let ctx_len = seq.len();
        seq.extend(cont);
        let mut cache = HybridCache::new(&w.config);
        let hidden = prefill_hidden_all(w, &seq, &mut cache);
        // Position ctx_len-1+j predicts continuation token j.
        let rows = &hidden[(ctx_len - 1) * w.hidden..(seq.len() - 1) * w.hidden];
        let mut logits = lm_head_batch(rows, cont.len(), &w.lm_head);
        let mut sum = 0.0f64;
        for (row, &t) in logits.chunks_mut(w.vocab).zip(cont) {
            log_softmax(row);
            sum += row[t as usize] as f64;
        }
        if per_char {
            sum /= item.continuations[i].trim_start().chars().count().max(1) as f64;
        }
        scores.push(sum);
    }
    Ok(scores)
}

/// Greedy answer to a chat prompt (thinking off).
fn generate_greedy(w: &ModelWeights, tok: &IncordTokenizer, prompt: &str, max_new: usize) -> String {
    let ids = tok.format_plain_chat(prompt);
    let mut cache = HybridCache::new(&w.config);
    let mut logits = prefill(w, &ids, &mut cache);
    let mut out: Vec<u32> = Vec::new();
    for _ in 0..max_new {
        let next = tasks::argmax(&logits.iter().map(|&v| v as f64).collect::<Vec<_>>()) as u32;
        if next == incord::tokenizer::EOS || next == incord::tokenizer::IM_END {
            break;
        }
        out.push(next);
        if incord::generate::is_stuck_in_loop(&out) {
            break;
        }
        logits = forward_decode(w, next as usize, &mut cache);
    }
    tok.decode(&out)
}

/// `(task, id)` pairs already present in an items file, so an interrupted
/// run resumes where it stopped.
fn read_items(path: &str) -> Result<Vec<serde_json::Value>, String> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("open {path}: {e}")),
    };
    BufReader::new(f)
        .lines()
        .filter(|l| l.as_ref().map_or(true, |l| !l.trim().is_empty()))
        .map(|l| {
            let l = l.map_err(|e| format!("read {path}: {e}"))?;
            serde_json::from_str(&l).map_err(|e| format!("{path}: bad line: {e}"))
        })
        .collect()
}

fn item_key(v: &serde_json::Value) -> (String, String) {
    (
        v["task"].as_str().unwrap_or_default().to_string(),
        v["id"].as_str().unwrap_or_default().to_string(),
    )
}

fn cmd_tasks(args: &[String]) -> Result<(), String> {
    let model = req(args, "--model")?;
    let tok_path = req(args, "--tokenizer")?;
    let items_path = req(args, "--items")?;
    let data = Path::new(opt(args, "--data").unwrap_or("eval-data/tasks"));
    let label = opt(args, "--label").unwrap_or(model);
    let max_new = num(args, "--max-new", 384)?;
    let limit = opt(args, "--limit")
        .map(|v| v.parse::<usize>().map_err(|_| format!("--limit: not a number: {v}")))
        .transpose()?;
    let task_list: Vec<Task> = match opt(args, "--tasks") {
        Some(s) => s.split(',').map(|t| Task::parse(t.trim())).collect::<Result<_, _>>()?,
        None => Task::ALL.to_vec(),
    };

    let tok = IncordTokenizer::from_file(tok_path).map_err(|e| format!("load tokenizer {tok_path}: {e}"))?;
    let w = load_model(model)?;
    let done: HashSet<(String, String)> = read_items(items_path)?.iter().map(item_key).collect();
    if let Some(dir) = Path::new(items_path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(items_path)
        .map_err(|e| format!("open {items_path}: {e}"))?;

    for task in task_list {
        let items = tasks::load(data, task)?.subsample(limit.unwrap_or_else(|| default_limit(task)));
        let total = items.len();
        let t0 = Instant::now();
        let mut ran = 0usize;
        let mut write = |v: serde_json::Value| -> Result<(), String> {
            writeln!(out, "{v}")
                .and_then(|_| out.flush())
                .map_err(|e| format!("write {items_path}: {e}"))
        };
        let pending = |id: &str| !done.contains(&(task.name().to_string(), id.to_string()));
        match items {
            Items::Mc(list) => {
                for (n, it) in list.iter().enumerate().filter(|(_, it)| pending(&it.id)) {
                    let scores = score_mc(&w, &tok, it, task.length_normalized())?;
                    let pred = tasks::argmax(&scores);
                    write(serde_json::json!({
                        "task": task.name(), "id": it.id, "pred": pred, "gold": it.gold,
                        "correct": pred == it.gold, "scores": scores,
                    }))?;
                    ran += 1;
                    eprint!("\r  {} {}/{total}  ", task.name(), n + 1);
                }
            }
            Items::Gen(list) => {
                for (n, it) in list.iter().enumerate().filter(|(_, it)| pending(&it.id)) {
                    let GenItem { id, question, answer } = it;
                    let text = generate_greedy(&w, &tok, &tasks::gsm8k_prompt(question), max_new);
                    let pred = tasks::extract_answer(&text);
                    write(serde_json::json!({
                        "task": task.name(), "id": id, "pred": pred, "gold": answer,
                        "correct": pred.is_some_and(|p| tasks::same_number(p, *answer)),
                        "text": text,
                    }))?;
                    ran += 1;
                    eprint!("\r  {} {}/{total}  ", task.name(), n + 1);
                }
            }
        }
        eprintln!("\r  {}: {ran} new of {total} in {:.1?}", task.name(), t0.elapsed());
    }

    // Summary over everything in the items file (earlier runs included).
    let all = read_items(items_path)?;
    let mut by_task: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for v in &all {
        let e = by_task.entry(item_key(v).0).or_default();
        e.0 += 1;
        e.1 += v["correct"].as_bool().unwrap_or(false) as usize;
    }
    println!("model  {label}");
    let mut summary = serde_json::Map::new();
    summary.insert("label".into(), label.into());
    summary.insert("model".into(), model.into());
    for (task, (n, correct)) in &by_task {
        let acc = *correct as f64 / *n as f64;
        let se = (acc * (1.0 - acc) / *n as f64).sqrt();
        println!("{task:<11} {:>6.2}% ± {:.2}  ({correct}/{n})", acc * 100.0, se * 100.0);
        summary.insert(task.clone(), serde_json::json!({"n": n, "correct": correct, "acc": acc}));
    }
    if let Some(json) = opt(args, "--json") {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(json)
            .map_err(|e| format!("open {json}: {e}"))?;
        writeln!(f, "{}", serde_json::Value::Object(summary))
            .map_err(|e| format!("write {json}: {e}"))?;
    }
    Ok(())
}

/// Question-by-question comparison of two `tasks` runs over the questions
/// both answered.
fn cmd_compare(args: &[String]) -> Result<(), String> {
    let ref_path = req(args, "--ref")?;
    let cand_path = req(args, "--cand")?;
    let reference: BTreeMap<_, _> = read_items(ref_path)?.into_iter().map(|v| (item_key(&v), v)).collect();

    #[derive(Default)]
    struct Row {
        n: usize,
        ref_ok: usize,
        cand_ok: usize,
        same: usize,
        lost: usize,
        gained: usize,
    }
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for c in read_items(cand_path)? {
        let key = item_key(&c);
        let Some(r) = reference.get(&key) else { continue };
        let (ro, co) = (r["correct"].as_bool() == Some(true), c["correct"].as_bool() == Some(true));
        let row = rows.entry(key.0).or_default();
        row.n += 1;
        row.ref_ok += ro as usize;
        row.cand_ok += co as usize;
        row.same += (r["pred"] == c["pred"]) as usize;
        row.lost += (ro && !co) as usize;
        row.gained += (!ro && co) as usize;
    }
    if rows.is_empty() {
        return Err(format!("{ref_path} and {cand_path} share no questions"));
    }
    println!("ref   {ref_path}\ncand  {cand_path}");
    println!("{:<11} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6}", "task", "n", "ref", "cand", "same", "lost", "gained");
    let pct = |a: usize, n: usize| a as f64 * 100.0 / n as f64;
    for (task, r) in &rows {
        println!(
            "{task:<11} {:>5} {:>7.2}% {:>7.2}% {:>7.2}% {:>6} {:>6}",
            r.n,
            pct(r.ref_ok, r.n),
            pct(r.cand_ok, r.n),
            pct(r.same, r.n),
            r.lost,
            r.gained
        );
    }
    Ok(())
}
