//! Full-model invariant tests for the batched decode pipeline.
//!
//! These tests are #[ignore] by default because they require:
//!   - The real Incord-Mini-2B model file (~3.5 GB)
//!   - Several seconds of model load time
//!
//! Run with:
//!
//!   INCORD_MODEL_PATH=model/model.incord cargo test --release \
//!     --test gemv_batch_full_model -- --ignored

use std::path::PathBuf;

#[test]
#[ignore]
fn full_model_n1_invariant() {
    use incord::gemv::{HybridCache, forward_decode};
    use incord::gemv_batch::{SlotInferenceState, forward_decode_batch};

    let path: PathBuf = std::env::var("INCORD_MODEL_PATH")
        .or_else(|_| std::env::var("INCORD_MODEL_PATH"))
        .unwrap_or_else(|_| "model/model.incord".into())
        .into();

    eprintln!("loading model from {}...", path.display());
    let weights =
        incord::save::load_model(&path).expect("model file required for this --ignored test");
    eprintln!(
        "model loaded ({} format, {} layers, hidden={}, vocab={})",
        weights.format_name,
        weights.layers.len(),
        weights.hidden,
        weights.vocab
    );

    let token_id: u32 = 1;

    // Single-slot reference
    let mut cache_a = HybridCache::new(&weights.config);
    let single = forward_decode(&weights, token_id as usize, &mut cache_a);

    // Batched N=1
    let mut state = SlotInferenceState::new(&weights.config);
    state.last_token = token_id;
    let batched = forward_decode_batch(&weights, &mut [&mut state]);

    assert_eq!(single.len(), batched[0].len(), "logit count mismatch");

    let max_diff: f32 = single
        .iter()
        .zip(batched[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff: f32 = single
        .iter()
        .zip(batched[0].iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / single.len() as f32;

    eprintln!("max logit diff: {max_diff:.6e}");
    eprintln!("mean logit diff: {mean_diff:.6e}");

    // Top-1 must match
    let argmax_single = single
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let argmax_batched = batched[0]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    eprintln!(
        "argmax single: {} batched: {}",
        argmax_single, argmax_batched
    );

    assert_eq!(
        argmax_single, argmax_batched,
        "argmax token diverged: single={argmax_single} batched={argmax_batched}"
    );

    // Tolerance: well-conditioned real weights should keep max diff under 0.1
    // (very loose — main correctness signal is argmax match)
    assert!(
        max_diff < 0.1,
        "logits diverged too much: max_diff={max_diff}"
    );
}

#[test]
#[ignore]
fn full_model_n2_independent_decodes() {
    use incord::gemv::{HybridCache, forward_decode};
    use incord::gemv_batch::{SlotInferenceState, forward_decode_batch};

    let path: PathBuf = std::env::var("INCORD_MODEL_PATH")
        .or_else(|_| std::env::var("INCORD_MODEL_PATH"))
        .unwrap_or_else(|_| "model/model.incord".into())
        .into();

    let weights =
        incord::save::load_model(&path).expect("model file required for this --ignored test");

    // Reference: 2 sequential single-slot decodes with different tokens
    let mut cache_a = HybridCache::new(&weights.config);
    let single_a = forward_decode(&weights, 1, &mut cache_a);

    let mut cache_b = HybridCache::new(&weights.config);
    let single_b = forward_decode(&weights, 2, &mut cache_b);

    // Batched: same 2 tokens in one batch call
    let mut state_a = SlotInferenceState::new(&weights.config);
    state_a.last_token = 1;
    let mut state_b = SlotInferenceState::new(&weights.config);
    state_b.last_token = 2;
    let batched = forward_decode_batch(&weights, &mut [&mut state_a, &mut state_b]);

    // Both slots must match their single-call counterparts
    let max_a: f32 = single_a
        .iter()
        .zip(batched[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_b: f32 = single_b
        .iter()
        .zip(batched[1].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("slot 0 max_diff: {max_a:.6e}");
    eprintln!("slot 1 max_diff: {max_b:.6e}");

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(
        argmax(&single_a),
        argmax(&batched[0]),
        "slot 0 argmax diverged"
    );
    assert_eq!(
        argmax(&single_b),
        argmax(&batched[1]),
        "slot 1 argmax diverged"
    );

    assert!(max_a < 0.1 && max_b < 0.1, "logits diverged too much");
}
