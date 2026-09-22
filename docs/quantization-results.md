# Quantization Results

Measured with `incord-eval` (see [quantization-roadmap.md](quantization-roadmap.md)).
Every row is scored against the same F16 reference of MiniCPM5-2B.

## Evaluation setup

| Item | Value |
|---|---|
| Reference | F16 conversion of the original bf16 safetensors (5.04 GB) |
| Evaluation text | `eval/corpus/`: English prose, English science, Chinese prose, Rust code |
| Chunks | 8 × 512 tokens (2 per file), 4,088 scored positions |
| Reference stored | top-256 log-probabilities per position + true-token log-probability |
| Machine | i5-11500 (6C/12T, AVX-512), 16 GB RAM, CPU inference |

## Results

KL = mean KL(reference ‖ candidate) in nats/token; lower is better.
Top-1 = % of positions where the candidate's most likely token equals the reference's.

| Model | Milestone | File size | KL mean | KL p99 | Top-1 | PPL ratio |
|---|---|---|---|---|---|---|
| F16 (reference vs itself) | M0 self-check | 5.04 GB | 0.000000 | 0.000000 | 100.00% | 1.0000 |
| Q8 everywhere (shipped, RTN) | M0 baseline | 2.68 GB | 0.000994 | — | 98.09% | 1.0024 |
| Q4 everywhere, RTN | M0 baseline | 1.42 GB | 0.224029 | — | 71.84% | 1.2255 |
| Q4 everywhere, MSE rounding | M1 | 1.42 GB | 0.189946 | 0.993595 | 73.34% | 1.2042 |

Reference perplexity on the evaluation text: 22.3615.

## Notes per milestone

- **M0** — the shipped `model_q8.incord` is byte-identical to a fresh
  conversion, so these numbers describe the model in use today. Prefill was
  made token-tiled during M0 (weights streamed once per 32 tokens instead of
  once per token): 512-token prefill of the F16 model went from 282 s to
  26 s, and the batched output head from 62 s to 1.3 s per 511 positions.
- **M1** — group scales chosen by squared-error search instead of
  `absmax / 7`: KL −15%, top-1 +1.5 points, same size and kernels. Plain
  scalar Q4 on this 2B model is still far from Q8 quality; rotation (M3) and
  Hessian-aware rounding (M4) target that gap.
