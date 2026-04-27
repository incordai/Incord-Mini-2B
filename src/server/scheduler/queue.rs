use crate::server::session::core::Session;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

pub struct PendingRequest {
    pub session: Session,
    pub sse_tx: tokio::sync::mpsc::Sender<crate::server::scheduler::Event>,
    /// Optional pre-encoded vision embeddings. When present, prefill uses
    /// prefill_with_vision instead of prefill. Layout: [num_vision_tokens, hidden].
    pub vision_embeds: Option<(Vec<f32>, usize)>, // (embeds, num_vision_tokens)
}

pub struct Queue {
    pub(crate) inner: Mutex<VecDeque<PendingRequest>>,
    notify: Notify,
    max_size: usize,
}

impl Queue {
    pub fn new(max_size: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            max_size,
        })
    }

    /// Try to push a request. Returns Err if queue full.
    pub fn push(&self, req: PendingRequest) -> Result<(), Box<PendingRequest>> {
        let mut q = self.inner.lock().unwrap();
        if q.len() >= self.max_size {
            return Err(Box::new(req));
        }
        q.push_back(req);
        self.notify.notify_one();
        Ok(())
    }

    /// Pop one request, awaiting until one arrives.
    pub async fn pop(self: &Arc<Self>) -> PendingRequest {
        loop {
            {
                let mut q = self.inner.lock().unwrap();
                if let Some(r) = q.pop_front() {
                    return r;
                }
            }
            self.notify.notified().await;
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queue_starts_empty() {
        let q = Queue::new(10);
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }
}
