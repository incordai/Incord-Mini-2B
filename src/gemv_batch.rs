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
