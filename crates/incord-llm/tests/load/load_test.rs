//! Load test binary for Incord API server.
//!
//! Usage:
//!   cargo run --release --bin load_test -- \
//!     --url http://api.example.com \
//!     --key incord_xxx \
//!     --concurrency 100 --duration 60 --max-tokens 200

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::json;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let mut concurrency = 100usize;
    let mut duration_secs = 60u64;
    let mut url = "http://127.0.0.1:8080".to_string();
    let mut key = String::new();
    let mut max_tokens = 200u32;
    let mut prompt = "Explain entropy briefly.".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--concurrency" => {
                concurrency = args[i + 1].parse().expect("--concurrency expects a number");
                i += 2;
            }
            "--duration" => {
                duration_secs = args[i + 1].parse().expect("--duration expects seconds");
                i += 2;
            }
            "--url" => {
                url = args[i + 1].clone();
                i += 2;
            }
            "--key" => {
                key = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().expect("--max-tokens expects a number");
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!(
                    "load_test --url <base> --key <incord_xxx> [--concurrency N] [--duration S] [--max-tokens N] [--prompt TEXT]"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown flag: {}", args[i]);
                i += 1;
            }
        }
    }

    if key.is_empty() {
        eprintln!("--key is required");
        std::process::exit(1);
    }

    println!(
        "Load test: {concurrency} concurrent, {duration_secs}s, max_tokens={max_tokens}\nURL: {url}"
    );

    let stop = Instant::now() + Duration::from_secs(duration_secs);
    let total = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(Mutex::new(Vec::<u64>::with_capacity(100_000)));

    let url = Arc::new(url);
    let key = Arc::new(key);
    let prompt = Arc::new(prompt);

    let mut handles = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let url = url.clone();
        let key = key.clone();
        let prompt = prompt.clone();
        let total = total.clone();
        let errors = errors.clone();
        let latencies = latencies.clone();
        handles.push(tokio::spawn(async move {
            let client = reqwest::Client::new();
            while Instant::now() < stop {
                let started = Instant::now();
                let res = client
                    .post(format!("{url}/v1/chat/completions"))
                    .bearer_auth(&*key)
                    .json(&json!({
                        "model": "incord-mini-2b",
                        "messages": [{"role":"user","content":&*prompt}],
                        "max_tokens": max_tokens,
                        "stream": false,
                        "temperature": 0.0,
                        "think": false,
                    }))
                    .timeout(Duration::from_secs(120))
                    .send()
                    .await;

                let elapsed = started.elapsed().as_millis() as u64;
                match res {
                    Ok(r) if r.status().is_success() => {
                        let _ = r.bytes().await;
                        total.fetch_add(1, Ordering::Relaxed);
                        latencies.lock().push(elapsed);
                    }
                    Ok(r) => {
                        eprintln!("non-success status: {}", r.status());
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }

    let mut lats = latencies.lock().clone();
    lats.sort_unstable();
    let percentile = |q: f64| -> u64 {
        if lats.is_empty() {
            return 0;
        }
        let idx = ((lats.len() as f64) * q).min(lats.len() as f64 - 1.0) as usize;
        lats[idx]
    };

    let total_count = total.load(Ordering::Relaxed);
    let error_count = errors.load(Ordering::Relaxed);
    let throughput = total_count as f64 / duration_secs as f64;

    println!();
    println!("=== Results ===");
    println!("requests:      {total_count}");
    println!("errors:        {error_count}");
    println!("throughput:    {throughput:.2} req/sec");
    println!("p50 (ms):      {}", percentile(0.50));
    println!("p95 (ms):      {}", percentile(0.95));
    println!("p99 (ms):      {}", percentile(0.99));
    println!("min (ms):      {}", lats.first().copied().unwrap_or(0));
    println!("max (ms):      {}", lats.last().copied().unwrap_or(0));
}
