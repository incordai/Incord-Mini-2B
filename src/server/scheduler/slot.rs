use crate::gemv_batch::SlotInferenceState;
use crate::server::scheduler::Event;
use crate::server::session::core::Session;
use tokio::sync::mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotPhase {
    Idle,
    Prefilling,
    Decoding,
    Done,
}

pub struct Slot {
    pub idx: usize,
    pub phase: SlotPhase,
    pub session: Option<Session>,
    pub inference: Option<SlotInferenceState>,
    pub sse_tx: Option<mpsc::Sender<Event>>,
    pub admitted_at: Option<std::time::Instant>,
    /// Pre-encoded vision embeddings carried from PendingRequest.
    /// Consumed (set to None) after prefill_with_vision absorbs them into the cache.
    pub vision_embeds: Option<(Vec<f32>, usize)>,
}

impl Slot {
    pub fn empty(idx: usize) -> Self {
        Self {
            idx,
            phase: SlotPhase::Idle,
            session: None,
            inference: None,
            sse_tx: None,
            admitted_at: None,
            vision_embeds: None,
        }
    }

    pub fn is_free(&self) -> bool {
        self.phase == SlotPhase::Idle
    }

    pub fn release(&mut self) {
        self.phase = SlotPhase::Idle;
        self.session = None;
        self.inference = None;
        self.sse_tx = None;
        self.admitted_at = None;
        self.vision_embeds = None;
    }
}
