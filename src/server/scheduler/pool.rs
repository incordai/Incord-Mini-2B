use crate::config::Qor4bConfig;
use crate::gemv_batch::SlotInferenceState;
use crate::server::scheduler::queue::PendingRequest;
use crate::server::scheduler::slot::{Slot, SlotPhase};
use std::sync::{Arc, Mutex};

pub struct SlotPool {
    pub slots: Mutex<Vec<Slot>>,
    pub size: usize,
}

impl SlotPool {
    pub fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            size,
            slots: Mutex::new((0..size).map(Slot::empty).collect()),
        })
    }

    /// Admit a pending request into the first free slot.
    /// Returns true if admitted, false if no free slot.
    pub fn admit(&self, req: PendingRequest, config: &Qor4bConfig) -> bool {
        let mut slots = self.slots.lock().unwrap();
        for slot in slots.iter_mut() {
            if slot.is_free() {
                slot.session = Some(req.session);
                slot.inference = Some(SlotInferenceState::new(config));
                slot.sse_tx = Some(req.sse_tx);
                slot.vision_embeds = req.vision_embeds;
                slot.phase = SlotPhase::Prefilling;
                slot.admitted_at = Some(std::time::Instant::now());
                return true;
            }
        }
        false
    }

    pub fn count_active(&self) -> usize {
        self.slots
            .lock()
            .unwrap()
            .iter()
            .filter(|s| !s.is_free())
            .count()
    }
}

#[cfg(test)]
#[cfg(feature = "mock-weights")]
mod tests {
    use super::*;
    use crate::gemv::HybridCache;
    use crate::gemv::mock::tiny_weights;
    use crate::server::scheduler::queue::PendingRequest;
    use crate::server::session::core::Session;
    use crate::server::session::sampler::Sampler;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    fn fake_session(w: &crate::gemv::ModelWeights) -> PendingRequest {
        let (tx, _rx) = mpsc::channel(64);
        PendingRequest {
            session: Session {
                id: Uuid::new_v4(),
                key_id: "k".into(),
                user_id: "u".into(),
                model_name: "qora-4b".into(),
                cache: HybridCache::new(&w.config),
                prompt_tokens: vec![1],
                generated_tokens: Vec::new(),
                max_new_tokens: 4,
                eos_token_id: 0,
                sampler: Sampler::new(0.0, 1.0, 1, 0.0, 1.0),
                started_at: 0,
            },
            sse_tx: tx,
            vision_embeds: None,
        }
    }

    #[test]
    fn admit_fills_slot_then_returns_false_when_full() {
        let w = tiny_weights();
        let p = SlotPool::new(2);
        assert!(p.admit(fake_session(&w), &w.config));
        assert!(p.admit(fake_session(&w), &w.config));
        assert!(!p.admit(fake_session(&w), &w.config));
        assert_eq!(p.count_active(), 2);
    }

    #[test]
    fn release_returns_slot_to_pool() {
        let w = tiny_weights();
        let p = SlotPool::new(1);
        assert!(p.admit(fake_session(&w), &w.config));
        assert_eq!(p.count_active(), 1);

        // Manually release the slot
        {
            let mut slots = p.slots.lock().unwrap();
            slots[0].phase = SlotPhase::Done;
            slots[0].release();
        }

        assert_eq!(p.count_active(), 0);
        assert!(p.admit(fake_session(&w), &w.config));
    }
}
