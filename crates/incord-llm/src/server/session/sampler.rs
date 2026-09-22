use std::time::SystemTime;

pub struct Sampler {
    state: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
}

impl Sampler {
    pub fn new(
        temperature: f32,
        top_p: f32,
        top_k: usize,
        presence_penalty: f32,
        repetition_penalty: f32,
    ) -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            | 1;
        Self {
            state: seed,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            repetition_penalty,
        }
    }

    fn rand(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0
    }

    pub fn sample(&mut self, logits: &mut [f32], generated: &[u32]) -> u32 {
        // Apply repetition penalty
        if self.repetition_penalty > 1.0 {
            let window = generated.len().min(64);
            let recent = &generated[generated.len() - window..];
            for &t in recent {
                if (t as usize) < logits.len() {
                    let l = &mut logits[t as usize];
                    if *l > 0.0 {
                        *l /= self.repetition_penalty;
                    } else {
                        *l *= self.repetition_penalty;
                    }
                }
            }
        }
        // Apply presence penalty
        if self.presence_penalty > 0.0 && !generated.is_empty() {
            let mut seen = vec![false; logits.len()];
            for &t in generated {
                if (t as usize) < seen.len() {
                    seen[t as usize] = true;
                }
            }
            for (i, &s) in seen.iter().enumerate() {
                if s {
                    logits[i] -= self.presence_penalty;
                }
            }
        }
        // Greedy fast path
        if self.temperature == 0.0 {
            let mut best = 0;
            let mut best_v = logits[0];
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best as u32;
        }
        // Apply temperature
        for v in logits.iter_mut() {
            *v /= self.temperature;
        }
        // Top-k
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| logits[b as usize].partial_cmp(&logits[a as usize]).unwrap());
        idx.truncate(self.top_k);
        // Softmax over top-k
        let max = idx
            .iter()
            .map(|&i| logits[i as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = idx
            .iter()
            .map(|&i| (logits[i as usize] - max).exp())
            .collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|p| p / sum).collect();
        // Top-p (cumulative)
        let mut cum = 0.0;
        let mut cutoff = idx.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= self.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let final_idx = &idx[..cutoff];
        let final_probs = &probs[..cutoff];
        let final_sum: f32 = final_probs.iter().sum();
        // Sample
        let r = self.rand() * final_sum;
        let mut acc = 0.0;
        for (i, &p) in final_probs.iter().enumerate() {
            acc += p;
            if r <= acc {
                return final_idx[i];
            }
        }
        final_idx[final_idx.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(0.0, 1.0, 1, 0.0, 1.0);
        let mut logits = vec![1.0, 5.0, 2.0, 3.0];
        assert_eq!(s.sample(&mut logits, &[]), 1);
    }
}
