//! Concrete ChatEngine wrapping the existing QORA LLM inference.

use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::mpsc;

use qor4b::gemv::{forward_decode, prefill, prefill_with_vision, HybridCache, ModelWeights};
use qor4b::tokenizer::QoraTokenizer;

use crate::engine::chat::{ChatEngine, ChatEvent, ChatRequest, FinishReason};
use crate::engine::{ModelEngine, Task};

/// xorshift64 sampler — simplified inline sampler for v1 LlmEngine.
/// Mirrors the Sampler used by the v1 chat handler.
fn sample_token(
    logits: &mut [f32],
    generated: &[u32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
    repetition_penalty: f32,
    rng_state: &mut u64,
) -> u32 {
    if repetition_penalty > 1.0 {
        let window = generated.len().min(64);
        let recent = &generated[generated.len() - window..];
        for &t in recent {
            if (t as usize) < logits.len() {
                let l = &mut logits[t as usize];
                if *l > 0.0 { *l /= repetition_penalty; } else { *l *= repetition_penalty; }
            }
        }
    }
    if presence_penalty > 0.0 && !generated.is_empty() {
        let mut seen = vec![false; logits.len()];
        for &t in generated { if (t as usize) < seen.len() { seen[t as usize] = true; } }
        for (i, &s) in seen.iter().enumerate() { if s { logits[i] -= presence_penalty; } }
    }
    if temperature == 0.0 {
        let mut best = 0;
        let mut best_v = logits[0];
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v { best_v = v; best = i; }
        }
        return best as u32;
    }
    for v in logits.iter_mut() { *v /= temperature; }
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap());
    idx.truncate(top_k);
    let max = idx.iter().map(|&i| logits[i as usize]).fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = idx.iter().map(|&i| (logits[i as usize] - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let probs: Vec<f32> = exp.iter().map(|p| p / sum).collect();
    let mut cum = 0.0;
    let mut cutoff = idx.len();
    for i in 0..idx.len() {
        cum += probs[i];
        if cum >= top_p { cutoff = i + 1; break; }
    }
    let final_idx = &idx[..cutoff];
    let final_probs = &probs[..cutoff];
    let final_sum: f32 = final_probs.iter().sum();
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let r = ((*rng_state >> 40) as f32 / 16777216.0) * final_sum;
    let mut acc = 0.0;
    for (i, &p) in final_probs.iter().enumerate() {
        acc += p;
        if r <= acc { return final_idx[i]; }
    }
    final_idx[final_idx.len() - 1]
}

pub struct LlmEngine {
    pub id: String,
    pub weights: Arc<ModelWeights>,
    pub tokenizer: Arc<QoraTokenizer>,
    pub description: String,
}

impl LlmEngine {
    pub fn new(
        id: impl Into<String>,
        weights: Arc<ModelWeights>,
        tokenizer: Arc<QoraTokenizer>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            weights,
            tokenizer,
            description: description.into(),
        }
    }
}

impl ModelEngine for LlmEngine {
    fn id(&self) -> &str { &self.id }
    fn task(&self) -> Task { Task::Chat }
    fn description(&self) -> &str { &self.description }
}

#[async_trait]
impl ChatEngine for LlmEngine {
    fn supports_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    async fn encode_image(&self, bytes: Vec<u8>) -> anyhow::Result<(Vec<f32>, usize)> {
        if self.weights.vision.is_none() {
            anyhow::bail!("LlmEngine `{}` has no vision weights", self.id);
        }
        let weights = self.weights.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<f32>, usize)> {
            let vision = weights.vision.as_ref()
                .ok_or_else(|| anyhow::anyhow!("no vision encoder"))?;
            let img = image::load_from_memory(&bytes)?;
            let rgb = img.to_rgb8();
            let h = rgb.height() as usize;
            let w = rgb.width() as usize;
            let (pixels, h2, w2) = qor4b::vision::pixels_from_rgb_bytes(rgb.as_raw(), w, h);
            let (embeds, num_tokens) = vision.encode_image(&pixels, h2, w2);
            Ok((embeds, num_tokens))
        }).await?
    }

    async fn chat(&self, req: ChatRequest, sse_tx: mpsc::Sender<ChatEvent>) -> anyhow::Result<()> {
        let weights = self.weights.clone();
        let tokenizer = self.tokenizer.clone();

        // Run inference on a blocking thread (heavy CPU work).
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            // Build prompt from messages
            let user_message = render_user_message(&req.messages);
            let think = req.think.unwrap_or(true);
            let max_new_tokens = req.max_tokens.unwrap_or(1024) as usize;
            let prompt_tokens = tokenizer.format_chat(&user_message, think, max_new_tokens);

            let mut cache = HybridCache::new(&weights.config);
            let mut generated: Vec<u32> = Vec::new();

            // Prefill (with optional vision embeds)
            let mut logits = if let Some((ref embeds, num_tokens)) = req.vision_embeds {
                prefill_with_vision(&weights, &prompt_tokens, embeds, num_tokens, &mut cache)
            } else {
                prefill(&weights, &prompt_tokens, &mut cache)
            };

            let temperature = req.temperature.unwrap_or(0.7);
            let top_p = req.top_p.unwrap_or(0.95);
            let top_k = 20usize;
            let presence_penalty = 1.5f32;
            let repetition_penalty = 1.0f32;
            let eos_token_id = weights.config.eos_token_id;

            let mut rng_state: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64 | 1;

            for _step in 0..max_new_tokens {
                let token = sample_token(
                    &mut logits,
                    &generated,
                    temperature,
                    top_p,
                    top_k,
                    presence_penalty,
                    repetition_penalty,
                    &mut rng_state,
                );
                generated.push(token);
                let text = tokenizer.decode(&[token]);

                if sse_tx.blocking_send(ChatEvent::Token(token, text)).is_err() {
                    let _ = sse_tx.blocking_send(ChatEvent::Finish(FinishReason::Cancelled));
                    return Ok(());
                }

                if token == eos_token_id {
                    let _ = sse_tx.blocking_send(ChatEvent::Finish(FinishReason::Stop));
                    return Ok(());
                }

                logits = forward_decode(&weights, token as usize, &mut cache);
            }

            let _ = sse_tx.blocking_send(ChatEvent::Finish(FinishReason::Length));
            Ok(())
        }).await?
    }
}

fn render_user_message(messages: &[crate::engine::chat::ChatMessage]) -> String {
    let mut combined = String::new();
    for m in messages {
        if m.role == "user" {
            combined.push_str(&m.content);
        } else {
            combined.push_str(&format!("[{}] {}\n", m.role, m.content));
        }
    }
    combined
}
