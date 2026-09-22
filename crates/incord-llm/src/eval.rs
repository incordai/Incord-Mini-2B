//! Quality evaluation: compare a candidate model's next-token distributions
//! against a saved full-precision reference.
//!
//! The reference run stores, per scored position, the log-probability of the
//! true next token and the top-K log-probabilities with their token ids. The
//! candidate run recomputes full log-probabilities and measures:
//!
//! - KL(ref ‖ cand), computed over the reference top-K plus one "tail" bucket
//!   holding the remaining probability mass. Merging outcomes can only lower
//!   KL, so this is a lower bound of the full-vocabulary KL; with K = 256 the
//!   tail mass is tiny and the bound is tight.
//! - top-1 agreement (same most-likely token as the reference),
//! - perplexity of both models on the true text.
//!
//! Used by the `incord-eval` binary (src/bin/incord_eval.rs); the file format
//! is written/read by [`RefFile`].

use std::io::{self, Read, Write};

/// Magic bytes of a reference file.
pub const REF_MAGIC: &[u8; 4] = b"IEVR";
/// Reference file format version.
pub const REF_VERSION: u32 = 1;
/// Default number of reference log-probabilities stored per position.
pub const DEFAULT_TOP_K: usize = 256;

/// Smallest probability used inside logarithms, so a tail mass that rounds
/// to zero never produces `ln(0)`.
const MIN_PROB: f64 = 1e-12;

/// In-place log-softmax. Computed in f64 so large vocabularies do not lose
/// precision in the normaliser.
pub fn log_softmax(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| ((v as f64) - max).exp()).sum();
    let lse = max + sum.ln();
    for v in logits.iter_mut() {
        *v = ((*v as f64) - lse) as f32;
    }
}

/// Ids and log-probabilities of the `k` most likely tokens, highest first.
/// Ties are broken by the lower token id so results are deterministic.
pub fn top_k(logprobs: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let k = k.min(logprobs.len());
    let mut idx: Vec<u32> = (0..logprobs.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| {
        logprobs[*b as usize]
            .total_cmp(&logprobs[*a as usize])
            .then(a.cmp(b))
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp);
    let lps = idx.iter().map(|&i| logprobs[i as usize]).collect();
    (idx, lps)
}

/// One scored position of the reference model.
#[derive(Clone, Debug, PartialEq)]
pub struct RefPosition {
    /// Log-probability the reference assigns to the true next token.
    pub target_logprob: f32,
    /// Top-K token ids, most likely first.
    pub top_ids: Vec<u32>,
    /// Log-probabilities matching `top_ids`.
    pub top_logprobs: Vec<f32>,
}

impl RefPosition {
    /// Build from full reference log-probabilities.
    pub fn from_logprobs(logprobs: &[f32], target: u32, k: usize) -> Self {
        let (top_ids, top_logprobs) = top_k(logprobs, k);
        Self {
            target_logprob: logprobs[target as usize],
            top_ids,
            top_logprobs,
        }
    }
}

/// KL(ref ‖ cand) over the reference top-K plus a tail bucket.
/// `cand_logprobs` are the candidate's full-vocabulary log-probabilities.
pub fn kl_topk(r: &RefPosition, cand_logprobs: &[f32]) -> f64 {
    let mut kl = 0.0f64;
    let mut p_top = 0.0f64;
    let mut q_top = 0.0f64;
    for (&id, &lp) in r.top_ids.iter().zip(&r.top_logprobs) {
        let lp = lp as f64;
        let lq = cand_logprobs[id as usize] as f64;
        let p = lp.exp();
        kl += p * (lp - lq);
        p_top += p;
        q_top += lq.exp();
    }
    let p_tail = (1.0 - p_top).max(0.0);
    if p_tail > 0.0 {
        let q_tail = (1.0 - q_top).max(MIN_PROB);
        kl += p_tail * (p_tail.max(MIN_PROB).ln() - q_tail.ln());
    }
    // Rounding can push a near-zero KL slightly negative.
    kl.max(0.0)
}

/// Index of the largest value (lowest index wins ties).
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// A saved reference run: the exact token chunks that were scored plus the
/// reference distribution at every scored position.
#[derive(Clone, Debug, PartialEq)]
pub struct RefFile {
    pub top_k: usize,
    /// Token chunks, each fed to the model from an empty cache.
    pub chunks: Vec<Vec<u32>>,
    /// Scored positions in chunk order: for a chunk of length L the positions
    /// 0..L-1 predict tokens 1..L.
    pub positions: Vec<RefPosition>,
}

impl RefFile {
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(REF_MAGIC)?;
        w.write_all(&REF_VERSION.to_le_bytes())?;
        w.write_all(&(self.top_k as u32).to_le_bytes())?;
        w.write_all(&(self.chunks.len() as u32).to_le_bytes())?;
        for c in &self.chunks {
            w.write_all(&(c.len() as u32).to_le_bytes())?;
            for &t in c {
                w.write_all(&t.to_le_bytes())?;
            }
        }
        w.write_all(&(self.positions.len() as u64).to_le_bytes())?;
        for p in &self.positions {
            if p.top_ids.len() != self.top_k || p.top_logprobs.len() != self.top_k {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "position does not hold exactly top_k entries",
                ));
            }
            w.write_all(&p.target_logprob.to_le_bytes())?;
            for &id in &p.top_ids {
                w.write_all(&id.to_le_bytes())?;
            }
            for &lp in &p.top_logprobs {
                w.write_all(&lp.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != REF_MAGIC {
            return Err(invalid("not an incord-eval reference file"));
        }
        let version = read_u32(r)?;
        if version != REF_VERSION {
            return Err(invalid(&format!("unsupported reference version {version}")));
        }
        let top_k = read_u32(r)? as usize;
        let n_chunks = read_u32(r)? as usize;
        let mut chunks = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            let len = read_u32(r)? as usize;
            let mut c = Vec::with_capacity(len);
            for _ in 0..len {
                c.push(read_u32(r)?);
            }
            chunks.push(c);
        }
        let n_pos = read_u64(r)? as usize;
        let expected: usize = chunks.iter().map(|c| c.len().saturating_sub(1)).sum();
        if n_pos != expected {
            return Err(invalid(&format!(
                "reference holds {n_pos} positions but its chunks imply {expected}"
            )));
        }
        let mut positions = Vec::with_capacity(n_pos);
        for _ in 0..n_pos {
            let target_logprob = read_f32(r)?;
            let mut top_ids = Vec::with_capacity(top_k);
            for _ in 0..top_k {
                top_ids.push(read_u32(r)?);
            }
            let mut top_logprobs = Vec::with_capacity(top_k);
            for _ in 0..top_k {
                top_logprobs.push(read_f32(r)?);
            }
            positions.push(RefPosition {
                target_logprob,
                top_ids,
                top_logprobs,
            });
        }
        Ok(Self {
            top_k,
            chunks,
            positions,
        })
    }
}

/// Aggregated comparison of a candidate against the reference.
#[derive(Clone, Debug)]
pub struct Score {
    pub positions: usize,
    pub kl_mean: f64,
    pub kl_median: f64,
    pub kl_p99: f64,
    pub kl_max: f64,
    /// Fraction of positions whose argmax matches the reference argmax.
    pub top1_agreement: f64,
    pub ppl_ref: f64,
    pub ppl_cand: f64,
}

/// Accumulates per-position results into a [`Score`].
#[derive(Default)]
pub struct ScoreAccumulator {
    kls: Vec<f64>,
    agree: usize,
    nll_ref: f64,
    nll_cand: f64,
}

impl ScoreAccumulator {
    /// Record one position. `cand_logprobs` are full-vocabulary log-probs,
    /// `target` is the true next token.
    pub fn add(&mut self, r: &RefPosition, cand_logprobs: &[f32], target: u32) {
        self.kls.push(kl_topk(r, cand_logprobs));
        if argmax(cand_logprobs) == r.top_ids[0] {
            self.agree += 1;
        }
        self.nll_ref -= r.target_logprob as f64;
        self.nll_cand -= cand_logprobs[target as usize] as f64;
    }

    pub fn finish(mut self) -> Score {
        let n = self.kls.len();
        assert!(n > 0, "no positions scored");
        let kl_mean = self.kls.iter().sum::<f64>() / n as f64;
        self.kls.sort_by(f64::total_cmp);
        let pct = |q: f64| self.kls[((n as f64 - 1.0) * q).round() as usize];
        Score {
            positions: n,
            kl_mean,
            kl_median: pct(0.5),
            kl_p99: pct(0.99),
            kl_max: self.kls[n - 1],
            top1_agreement: self.agree as f64 / n as f64,
            ppl_ref: (self.nll_ref / n as f64).exp(),
            ppl_cand: (self.nll_cand / n as f64).exp(),
        }
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(logits: &[f32]) -> Vec<f32> {
        let mut v = logits.to_vec();
        log_softmax(&mut v);
        v
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let v = lp(&[1.0, 2.0, 3.0, -50.0, 100.0]);
        let s: f64 = v.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "sum = {s}");
    }

    #[test]
    fn top_k_is_sorted_and_deterministic() {
        let v = [0.1f32, 0.5, 0.5, -1.0, 0.9];
        let (ids, lps) = top_k(&v, 3);
        assert_eq!(ids, vec![4, 1, 2]); // tie 0.5: lower id first
        assert_eq!(lps, vec![0.9, 0.5, 0.5]);
    }

    #[test]
    fn kl_of_identical_distributions_is_zero() {
        let v = lp(&[0.3, -1.2, 2.5, 0.0, 1.1, -0.7]);
        let r = RefPosition::from_logprobs(&v, 2, 3);
        assert!(kl_topk(&r, &v) < 1e-9);
    }

    #[test]
    fn kl_matches_exact_formula_when_k_is_full_vocab() {
        let p = lp(&[0.3, -1.2, 2.5, 0.0]);
        let q = lp(&[0.1, -0.9, 2.0, 0.4]);
        let exact: f64 = p
            .iter()
            .zip(&q)
            .map(|(&a, &b)| (a as f64).exp() * (a as f64 - b as f64))
            .sum();
        let r = RefPosition::from_logprobs(&p, 0, 4);
        assert!((kl_topk(&r, &q) - exact).abs() < 1e-6);
    }

    #[test]
    fn topk_kl_is_a_lower_bound_of_full_kl() {
        let p = lp(&[3.0, 2.0, 1.0, 0.5, 0.2, -1.0, -2.0]);
        let q = lp(&[1.0, 2.5, 0.0, 1.5, -0.5, 0.5, -1.0]);
        let full = kl_topk(&RefPosition::from_logprobs(&p, 0, 7), &q);
        let part = kl_topk(&RefPosition::from_logprobs(&p, 0, 3), &q);
        assert!(part <= full + 1e-12 && part > 0.0, "part {part} full {full}");
    }

    #[test]
    fn ref_file_round_trips() {
        let a = lp(&[0.3, -1.2, 2.5, 0.0]);
        let b = lp(&[1.0, 0.0, -1.0, 2.0]);
        let f = RefFile {
            top_k: 2,
            chunks: vec![vec![5, 6, 7]],
            positions: vec![
                RefPosition::from_logprobs(&a, 1, 2),
                RefPosition::from_logprobs(&b, 3, 2),
            ],
        };
        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();
        let g = RefFile::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(f, g);
    }

    #[test]
    fn ref_file_rejects_position_count_mismatch() {
        let a = lp(&[0.3, -1.2]);
        let f = RefFile {
            top_k: 2,
            chunks: vec![vec![1, 2, 3]], // implies 2 positions
            positions: vec![RefPosition::from_logprobs(&a, 0, 2)],
        };
        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();
        assert!(RefFile::read_from(&mut buf.as_slice()).is_err());
    }

    #[test]
    fn score_of_reference_against_itself_is_perfect() {
        let dists = [lp(&[0.3, -1.2, 2.5, 0.0]), lp(&[2.0, 1.0, 0.0, -1.0])];
        let mut acc = ScoreAccumulator::default();
        for (i, d) in dists.iter().enumerate() {
            let r = RefPosition::from_logprobs(d, i as u32, 4);
            acc.add(&r, d, i as u32);
        }
        let s = acc.finish();
        assert!(s.kl_mean < 1e-9);
        assert_eq!(s.top1_agreement, 1.0);
        assert!((s.ppl_ref - s.ppl_cand).abs() < 1e-9);
    }
}
