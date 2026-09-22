//! Full-model check for the evaluator's all-position prefill.
//!
//! #[ignore] by default: needs the real model file. Run with:
//!
//!   INCORD_MODEL_PATH=model/model.incord cargo test --release \
//!     --test eval_full_model -- --ignored

use std::path::PathBuf;

#[test]
#[ignore]
fn prefill_hidden_all_last_row_matches_prefill() {
    use incord::gemv::{HybridCache, lm_head_parallel, prefill, prefill_hidden_all};

    let path: PathBuf = std::env::var("INCORD_MODEL_PATH")
        .unwrap_or_else(|_| "model/model.incord".into())
        .into();
    let w = incord::save::load_model(&path).expect("model file required for this --ignored test");

    // Arbitrary in-vocab tokens.
    let tokens: Vec<u32> = (0..24u32).map(|i| 1000 + i * 37).collect();

    let mut cache_a = HybridCache::new(&w.config);
    let expected = prefill(&w, &tokens, &mut cache_a);

    let mut cache_b = HybridCache::new(&w.config);
    let hidden = prefill_hidden_all(&w, &tokens, &mut cache_b);
    assert_eq!(hidden.len(), tokens.len() * w.hidden);
    let last = &hidden[(tokens.len() - 1) * w.hidden..];
    let got = lm_head_parallel(last, &w.lm_head, w.vocab, w.hidden);

    // Same arithmetic in the same order: must be bit-identical.
    assert_eq!(expected, got);
}
