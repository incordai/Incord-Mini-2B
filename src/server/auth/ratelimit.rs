use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct WindowState {
    requests: VecDeque<Instant>,
    tokens: VecDeque<(Instant, u32)>,
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<HashMap<String, WindowState>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn check_request(&self, key_id: &str, rpm: u32) -> Result<(), u64> {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);
        let mut map = self.inner.lock().unwrap();
        let state = map.entry(key_id.to_string()).or_insert_with(|| WindowState {
            requests: VecDeque::new(),
            tokens: VecDeque::new(),
        });
        while let Some(&t) = state.requests.front() {
            if t < cutoff { state.requests.pop_front(); } else { break; }
        }
        if state.requests.len() >= rpm as usize {
            let oldest = *state.requests.front().unwrap();
            let retry_in = (oldest + Duration::from_secs(60)).saturating_duration_since(now);
            return Err(retry_in.as_secs().max(1));
        }
        state.requests.push_back(now);
        Ok(())
    }

    pub fn record_tokens(&self, key_id: &str, tpm: u32, count: u32) -> Result<(), u64> {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(60);
        let mut map = self.inner.lock().unwrap();
        let state = map.entry(key_id.to_string()).or_insert_with(|| WindowState {
            requests: VecDeque::new(),
            tokens: VecDeque::new(),
        });
        while let Some(&(t, _)) = state.tokens.front() {
            if t < cutoff { state.tokens.pop_front(); } else { break; }
        }
        let total: u32 = state.tokens.iter().map(|(_, n)| *n).sum();
        if total.saturating_add(count) > tpm {
            let oldest = state.tokens.front().map(|(t, _)| *t).unwrap_or(now);
            let retry_in = (oldest + Duration::from_secs(60)).saturating_duration_since(now);
            return Err(retry_in.as_secs().max(1));
        }
        state.tokens.push_back((now, count));
        Ok(())
    }
}

impl Default for RateLimiter { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_under_rpm() {
        let l = RateLimiter::new();
        for _ in 0..10 { assert!(l.check_request("k1", 60).is_ok()); }
    }

    #[test]
    fn blocks_at_rpm_limit() {
        let l = RateLimiter::new();
        for _ in 0..3 { assert!(l.check_request("k1", 3).is_ok()); }
        assert!(l.check_request("k1", 3).is_err());
    }

    #[test]
    fn separate_keys_dont_interfere() {
        let l = RateLimiter::new();
        for _ in 0..3 { l.check_request("k1", 3).unwrap(); }
        assert!(l.check_request("k2", 3).is_ok());
    }

    #[test]
    fn token_budget_enforced() {
        let l = RateLimiter::new();
        l.record_tokens("k1", 1000, 600).unwrap();
        l.record_tokens("k1", 1000, 300).unwrap();
        assert!(l.record_tokens("k1", 1000, 200).is_err());
    }
}
