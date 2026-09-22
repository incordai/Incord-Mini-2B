# Quantization Roadmap — smaller model, same quality

Goal: shrink Incord-Mini-2B well below today's 2.68 GB Q8 file while keeping
output quality measurably equal to the full-precision model. Every milestone
is gated by a quality test against a fixed reference, so no change lands
unless the numbers prove it.

## Starting point

| Fact | Value |
|---|---|
| Architecture | 42 layers, pure GQA attention (16 Q heads / 2 KV heads, head_dim 128), hidden 2048, SwiGLU MLP 6144, vocab 130,560 |
| Output head | separate from the embedding (`tie_word_embeddings = false`) |
| Weights | ~1.98 B in layers (MLP is 80% of each layer) + 267 M embedding + 267 M output head |
| Shipped file | `model_q8.incord`, 2.68 GB, Q8 everywhere (8.5 bits/weight incl. scales) |
| Quantizer | round-to-nearest, groups of 32 along the output dim, one f16 scale per group |
| File format | every weight carries its own format byte, so mixed precision needs no format change |

## How quality is measured

Tool: `incord-eval` (Milestone 0).

1. **Reference** — the model converted at F16 (bf16 → f16 is exact for these
   weights). `incord-eval reference` runs it over the evaluation text and
   saves, per position, the top-256 log-probabilities and the log-probability
   of the true next token.
2. **Candidate** — any `.incord` file. `incord-eval score` runs it over the
   same text and compares against the saved reference.

Reported per candidate:

| Metric | Meaning |
|---|---|
| **Mean KL(ref ‖ cand)** | how far the candidate's next-token distribution drifts from the reference. Main gate. |
| p99 / max KL | worst-case drift |
| **Top-1 agreement** | % of positions where both models pick the same most-likely token |
| PPL ratio | candidate perplexity ÷ reference perplexity on the true text |
| File size | bytes on disk |

The evaluation text and the calibration text (used from Milestone 4) are
separate sets, so calibration can never overfit the score. Results are logged
in [`quantization-results.md`](quantization-results.md).

## Milestones

Each milestone: its own branch, backups of every edited file, unit tests,
the quality gate, then merge to `main` and push.

Order: the quality milestones (M3, M4, M5) run before the GPU port (M2),
because they decide which formats the GPU path must support.

| # | Milestone | Proof required before it counts as done | Status |
|---|---|---|---|
| M0 | **Quality harness + baselines** — all-position logits, `incord-eval`, F16 reference, baseline scores for current Q8 and plain Q4 | ref vs itself gives KL = 0 and 100% agreement; two runs give identical numbers; unit tests for KL/softmax math | **done** — KL 0.000000, 100% (see results) |
| M1 | **Better rounding** — Q4 uses all 16 levels (today −8 is never used) and searches the best scale per group | unit test: error ≤ old Q4 on every group; KL(new Q4) < KL(old Q4) | **done** — KL 0.224 → 0.190 |
| M2 | **GPU: port to cortex-v2 + CUDA + packed weights** — move `gpu_inference.rs` / `gpu_loader.rs` to cortex-v2 (Burn 0.22 base, runtime `Device`, one code path for Vulkan / Metal / CUDA); keep Q4/Q8 packed on the GPU (cortex-v2 dequantizes inside the matmul kernel) instead of expanding to f32 (today ~4 bytes/weight in VRAM, so the Q8 model cannot fit a 6 GB card) | GPU logits match CPU within tolerance on Vulkan **and** CUDA; Q4 using level −8 verified on GPU; VRAM measured before/after | planned |
| M3 | **Hadamard rotation** — rotate weights offline, run a fast Walsh–Hadamard transform on activations (block size fits 2048 and 6144) | F16 + rotation gives KL ≈ 0 (math is exact); KL(rotated Q4) < KL(M1 Q4) | **done** — F16+rot KL 0.000001; Q4 KL 0.190 → 0.177 |
| M4 | **Calibration + GPTQ** — capture layer inputs on calibration text, Hessian-aware rounding | KL(GPTQ + rotation Q4) < KL(rotation Q4); output stays standard Q4 (no new kernels) | planned |
| M5 | **Mixed precision** — per-tensor bit choice from measured sensitivity (embedding, output head, attention, MLP) | best KL at each target size; size/KL table published | planned |
| M6 | **Sub-4-bit formats** — Q3 / Q2 kernels (CPU + GPU) for the least sensitive tensors | kernel output = dequantized reference; KL table vs size | planned |
| M7 | **Error recovery by training** — LoRA on the quantized model, then QAT + distillation in Cortex | needs the model defined as a trainable Cortex module first; KL must beat M5 at equal size | research |
| M8 | **Codebook / trellis 2-bit** — only if M5–M7 still leave a gap | KL at 2 bits beats M6 | optional |

## Expected sizes (estimates, to be replaced by measured numbers)

| Setup | Size |
|---|---|
| Today: Q8 everywhere | 2.68 GB |
| Q4 layers + Q8 embedding + Q8 output head | ~1.68 GB |
| + 3-bit MLP | ~1.48 GB |
| + 2-bit MLP | ~1.29 GB |

On a 2B model, 3-bit and below usually cost visible quality unless recovered
by training (M7); the gates decide what actually ships.
