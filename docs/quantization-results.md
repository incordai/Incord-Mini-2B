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
| F16 + rotation (exactness check) | M3 | 5.04 GB | 0.000001 | — | 99.93% | 1.0001 |
| Q4 everywhere, MSE + rotation | M3 | 1.42 GB | 0.177090 | 0.944713 | 76.00% | 1.1785 |
| Q4 everywhere, MSE + rotation + GPTQ | M4 | 1.42 GB | 0.068763 | — | 84.05% | 1.0604 |

Reference perplexity on the evaluation text: 22.3615.

## M5 — sensitivity map

Base: every weight Q8 with MSE rounding + rotation (converter, no GPTQ).
One part at a time dropped to Q4. "KL / GB" = KL added per GB saved —
lower means cheaper to compress.

| Q4 applied to | Size | Saves | KL mean | Top-1 | KL / GB |
|---|---|---|---|---|---|
| nothing (all Q8, MSE + rotation) | 2.678 GB | — | 0.000533 | 98.31% | — |
| embedding | 2.545 GB | 0.133 GB | 0.001709 | 97.14% | 0.009 |
| MLP, layers 14–27 | 2.414 GB | 0.264 GB | 0.017095 | 91.17% | 0.063 |
| gate + up, all layers | 2.150 GB | 0.528 GB | 0.040670 | 87.70% | 0.076 |
| MLP, layers 28–41 | 2.414 GB | 0.264 GB | 0.029689 | 89.73% | 0.11 |
| MLP, layers 0–13 | 2.414 GB | 0.264 GB | 0.043223 | 88.14% | 0.16 |
| output head | 2.545 GB | 0.133 GB | 0.021493 | 89.41% | 0.16 |
| down, all layers | 2.414 GB | 0.264 GB | 0.047677 | 87.11% | 0.18 |
| o | 2.590 GB | 0.088 GB | 0.022137 | 90.75% | 0.25 |
| q, k, v | 2.568 GB | 0.110 GB | 0.036890 | 88.67% | 0.33 |

Findings: the embedding and the middle-layer MLPs are cheap to compress;
attention (q/k/v, o), the output head and the first layers are fragile.
MSE rounding + rotation alone already halves the error of the shipped Q8
(0.000994 → 0.000533) at the same size.

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
- **M3** — randomized Hadamard rotation of every matmul weight, the output
  head and the embedding (`--rotate`). The F16 + rotation model scores
  KL 0.000001, so the transform is exact on the real model. On Q4 it cuts KL
  a further 7% (0.190 → 0.177) and lifts top-1 to 76.0%. The gain is
  smaller than rotation papers report because this Q4 layout already gives
  every 32 weights of an input row their own scale; rotation is expected to
  compound with GPTQ (M4). Rotated models run on the CPU path; the GPU path
  refuses them with a clear error until M2 adds the transform there.
- **M4** — `incord-quantize`: sequential, layer-by-layer GPTQ. Calibration
  text (`eval/calib`, 19 chunks / 9,728 tokens, never the scored text) runs
  through each layer; `H = XᵀX` is collected per projection input; the
  layer's 7 weights are quantized with Hessian-aware rounding (act-order,
  1% damping, 128-row blocks, same MSE group scales); the calibration
  stream then advances through the *quantized* layer. KL drops 61%
  (0.177 → 0.069), top-1 rises to 84.05%, perplexity is within 6% of the
  full-precision model — at the same 1.42 GB and with no kernel change.
  Runtime ~43 min on the i5-11500. The embedding and head still use plain
  MSE rounding; mixed precision (M5) addresses them.
  An end-to-end check (F16 through the tool, no quantization) produced a
  file byte-identical to the converter's F16 output.
