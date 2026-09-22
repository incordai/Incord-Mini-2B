use crate::config::IncordConfig;
use crate::gemv::{ModelWeights, prefill, prefill_with_vision};
use crate::gemv_batch::forward_decode_batch;
use crate::server::scheduler::pool::SlotPool;
use crate::server::scheduler::queue::Queue;
use crate::server::scheduler::slot::SlotPhase;
use crate::server::scheduler::{Event, FinishReason};
use crate::tokenizer::IncordTokenizer;
use std::sync::Arc;

pub struct Batcher {
    pub pool: Arc<SlotPool>,
    pub queue: Arc<Queue>,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<IncordTokenizer>,
}

impl Batcher {
    /// Run the continuous-batching loop forever.
    pub fn run(self) {
        let config = self.weights.config.clone();
        let idle_sleep = std::time::Duration::from_millis(1);
        loop {
            // Update slot / queue metrics at the top of every iteration.
            crate::server::metrics::ACTIVE_SLOTS.set(self.pool.count_active() as f64);
            crate::server::metrics::QUEUE_DEPTH.set(self.queue.len() as f64);

            // 1. Admit pending requests into free slots (non-blocking)
            self.admit_pending(&config);

            // 2. Run prefill for slots in Prefilling phase
            self.prefill_pending();

            // 3. Gather all slots in Decoding phase, run one batched decode step
            let active_count = self.batched_decode_step();

            // If no work to do, briefly idle. Sleep is short to keep latency low.
            if active_count == 0 && self.queue.is_empty() {
                std::thread::sleep(idle_sleep);
            }
        }
    }

    /// Drain pending requests into free slots.
    fn admit_pending(&self, config: &IncordConfig) {
        let free = self.pool.size - self.pool.count_active();
        for _ in 0..free {
            // Non-blocking pop from the queue.
            // We can't await here (sync context), so peek the inner Mutex directly.
            let req = {
                let mut q = self.queue.inner.lock().unwrap();
                q.pop_front()
            };
            match req {
                Some(r) => {
                    if !self.pool.admit(r, config) {
                        break;
                    }
                }
                None => break,
            }
        }
    }

    /// For each slot in Prefilling phase: run prefill, sample first token, send via SSE,
    /// transition to Decoding (or Done if EOS / max_tokens).
    fn prefill_pending(&self) {
        let prefilling: Vec<usize> = {
            let slots = self.pool.slots.lock().unwrap();
            slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    if s.phase == SlotPhase::Prefilling {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        };

        for idx in prefilling {
            // Prefill is bounded by the prompt length. Run sequentially per slot for v1.
            // (Multi-slot prefill batching is a future optimization.)
            let mut slots = self.pool.slots.lock().unwrap();
            let slot = &mut slots[idx];
            if slot.phase != SlotPhase::Prefilling {
                continue;
            }

            let prompt_tokens = slot.session.as_ref().unwrap().prompt_tokens.clone();
            let inference = slot.inference.as_mut().unwrap();
            let mut logits = if let Some((ref embeds, num_tokens)) = slot.vision_embeds {
                prefill_with_vision(
                    &self.weights,
                    &prompt_tokens,
                    embeds,
                    num_tokens,
                    &mut inference.cache,
                )
            } else {
                prefill(&self.weights, &prompt_tokens, &mut inference.cache)
            };
            // Free vision embeddings: already absorbed into the KV/state cache.
            slot.vision_embeds = None;
            inference.rope_pos = prompt_tokens.len();

            // Sample the first token
            let session = slot.session.as_mut().unwrap();
            let token = session
                .sampler
                .sample(&mut logits, &session.generated_tokens);
            session.generated_tokens.push(token);
            inference.last_token = token;

            // Decode token text + send
            let text = self.tokenizer.decode(&[token]);
            let send = slot
                .sse_tx
                .as_ref()
                .unwrap()
                .try_send(Event::Token(token, text));
            if send.is_err() {
                slot.phase = SlotPhase::Done;
                continue;
            }

            // Stop conditions
            let session = slot.session.as_ref().unwrap();
            if token == session.eos_token_id {
                let _ = slot
                    .sse_tx
                    .as_ref()
                    .unwrap()
                    .try_send(Event::Finish(FinishReason::Stop));
                slot.phase = SlotPhase::Done;
                continue;
            }
            if session.generated_tokens.len() >= session.max_new_tokens {
                let _ = slot
                    .sse_tx
                    .as_ref()
                    .unwrap()
                    .try_send(Event::Finish(FinishReason::Length));
                slot.phase = SlotPhase::Done;
                continue;
            }

            slot.phase = SlotPhase::Decoding;
        }

        self.reap_done();
    }

    /// Run one batched decode step over all Decoding slots. Returns the number of active slots processed.
    fn batched_decode_step(&self) -> usize {
        let active_indices: Vec<usize> = {
            let slots = self.pool.slots.lock().unwrap();
            slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    if s.phase == SlotPhase::Decoding {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        };

        if active_indices.is_empty() {
            return 0;
        }

        // Acquire mut borrows on all active slots' inference states.
        // Use scope to release the lock before sending SSE messages (which can block).
        let logits_per_slot: Vec<Vec<f32>> = {
            let mut slots = self.pool.slots.lock().unwrap();

            // Build a Vec of "moved out" inference states, run batched decode,
            // put them back. This avoids split_at_mut complexity for disjoint mutable
            // borrows and is safe because we hold the lock for the duration of the borrow.
            let mut sorted = active_indices.clone();
            sorted.sort_unstable();

            let mut taken: Vec<crate::gemv_batch::SlotInferenceState> = sorted
                .iter()
                .map(|&i| {
                    slots[i]
                        .inference
                        .take()
                        .expect("active slot has inference")
                })
                .collect();
            let mut state_refs: Vec<&mut crate::gemv_batch::SlotInferenceState> =
                taken.iter_mut().collect();

            let logits = forward_decode_batch(&self.weights, &mut state_refs);

            // Put inference states back
            for (idx, taken_state) in sorted.iter().zip(taken.into_iter()) {
                slots[*idx].inference = Some(taken_state);
            }

            logits
        };

        // Now sample, send tokens, advance state per slot — lock acquired briefly per slot.
        let mut sorted = active_indices.clone();
        sorted.sort_unstable();
        for (slot_idx, mut row) in sorted.iter().zip(logits_per_slot.into_iter()) {
            let mut slots = self.pool.slots.lock().unwrap();
            let slot = &mut slots[*slot_idx];
            // Defensive: slot may have transitioned to Done between locks (rare)
            if slot.phase != SlotPhase::Decoding {
                continue;
            }

            let session = slot.session.as_mut().unwrap();
            let inference = slot.inference.as_mut().unwrap();
            let token = session.sampler.sample(&mut row, &session.generated_tokens);
            session.generated_tokens.push(token);
            inference.last_token = token;

            let text = self.tokenizer.decode(&[token]);
            let send = slot
                .sse_tx
                .as_ref()
                .unwrap()
                .try_send(Event::Token(token, text));
            if send.is_err() {
                let _ = slot
                    .sse_tx
                    .as_ref()
                    .unwrap()
                    .try_send(Event::Finish(FinishReason::Cancelled));
                slot.phase = SlotPhase::Done;
                continue;
            }

            if token == session.eos_token_id
                || session.generated_tokens.len() >= session.max_new_tokens
            {
                let reason = if token == session.eos_token_id {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                };
                let _ = slot
                    .sse_tx
                    .as_ref()
                    .unwrap()
                    .try_send(Event::Finish(reason));
                slot.phase = SlotPhase::Done;
            }
        }

        self.reap_done();

        active_indices.len()
    }

    fn reap_done(&self) {
        let mut slots = self.pool.slots.lock().unwrap();
        for slot in slots.iter_mut() {
            if slot.phase == SlotPhase::Done {
                slot.release();
            }
        }
    }
}
