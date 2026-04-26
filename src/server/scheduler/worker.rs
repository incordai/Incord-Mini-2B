use std::sync::Arc;
use crate::gemv::{ModelWeights, prefill, forward_decode};
use crate::tokenizer::QoraTokenizer;
use crate::server::scheduler::queue::{PendingRequest, Queue};
use crate::server::scheduler::{Event, FinishReason};

pub struct Worker {
    pub id: usize,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<QoraTokenizer>,
    pub queue: Arc<Queue>,
}

impl Worker {
    /// Spawn this worker as a blocking thread that drives prefill+decode
    /// for one session at a time. Returns the JoinHandle.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            loop {
                let pending = rt.block_on(self.queue.pop());
                self.run_one(pending);
            }
        })
    }

    fn run_one(&self, mut req: PendingRequest) {
        let session = &mut req.session;
        let tx = req.sse_tx;

        // Prefill: produces logits for the next token.
        let mut logits = prefill(&self.weights, &session.prompt_tokens, &mut session.cache);

        // Decode loop
        for _step in 0..session.max_new_tokens {
            // Sample next token
            let token = session.sampler.sample(&mut logits, &session.generated_tokens);
            session.generated_tokens.push(token);

            // Decode token to text for SSE delta.
            // QoraTokenizer::decode(&[u32]) -> String (returns String directly, no Result).
            let text = self.tokenizer.decode(&[token]);

            // Send token event. If channel closed (client gone), stop.
            if tx.blocking_send(Event::Token(token, text)).is_err() {
                let _ = tx.blocking_send(Event::Finish(FinishReason::Cancelled));
                return;
            }

            // Stop on EOS.
            if token == session.eos_token_id {
                let _ = tx.blocking_send(Event::Finish(FinishReason::Stop));
                return;
            }

            // Compute next logits via existing forward_decode (still uses our HybridCache).
            logits = forward_decode(&self.weights, token as usize, &mut session.cache);
        }

        // Hit max_new_tokens.
        let _ = tx.blocking_send(Event::Finish(FinishReason::Length));
    }
}
