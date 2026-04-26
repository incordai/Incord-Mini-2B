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

use crate::gemv::{forward_decode, ModelWeights};
use rayon::prelude::*;

pub fn forward_decode_batch(
    weights: &ModelWeights,
    slots: &mut [&mut SlotInferenceState],
) -> Vec<Vec<f32>> {
    // Phase 0: parallel-independent. Bitwise identical to per-slot calls.
    // Future phases batch the inner kernels; for now this establishes the contract.
    slots
        .par_iter_mut()
        .map(|s| {
            let logits = forward_decode(weights, s.last_token as usize, &mut s.cache);
            s.rope_pos += 1;
            logits
        })
        .collect()
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
            // Phase 0 is bitwise identical (no batching, just parallel calls)
            assert!((a - b).abs() < 1e-6, "logit {i} mismatch: {a} vs {b}");
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
                assert!((av - bv).abs() < 1e-6, "slot {i} logit {j}: {av} vs {bv}");
            }
        }
    }
}
