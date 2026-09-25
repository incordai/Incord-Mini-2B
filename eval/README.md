# Evaluation text

Fixed text used by `incord-eval` to measure quantization quality
(see [docs/quantization-roadmap.md](../docs/quantization-roadmap.md)).

- `corpus/` — scored text. Every candidate model is compared against the F16
  reference on exactly these tokens.
- `calib/` — calibration text for Hessian-aware rounding (roadmap M4). Kept
  separate from `corpus/` so calibration never tunes the model to the scored
  text.

Do not edit these files: changing them changes every score. Add new files
instead and rebuild the reference.

| File | Source |
|---|---|
| `corpus/en_prose_pride_and_prejudice.txt` | Excerpt, *Pride and Prejudice*, Jane Austen (Project Gutenberg #1342, public domain) |
| `corpus/en_science_origin_of_species.txt` | Excerpt, *On the Origin of Species*, Charles Darwin (Project Gutenberg #1228, public domain) |
| `corpus/zh_journey_to_the_west.txt` | Excerpt, 西遊記, Wu Cheng'en (Project Gutenberg #23962, public domain) |
| `corpus/code_rust_gpu_loader.txt` | Copy of `crates/incord-llm/src/gpu_loader.rs` from this repository |
| `calib/en_prose_sherlock_holmes.txt` | Excerpt, *The Adventures of Sherlock Holmes*, Arthur Conan Doyle (Project Gutenberg #1661, public domain) |
| `calib/zh_dream_of_red_chamber.txt` | Excerpt, 紅樓夢, Cao Xueqin (Project Gutenberg #24264, public domain) |
| `calib/code_rust_tokenizer.txt` | Copy of `crates/incord-llm/src/tokenizer.rs` from this repository |

## Task benchmarks

KL and perplexity show how far a model's predictions drift from the original's.
They don't show whether it still answers questions correctly. `incord-eval tasks`
covers that with graded benchmarks. The data is downloaded, not committed:

```text
bash eval/fetch-tasks.sh                      # -> eval-data/tasks (git-ignored)
incord-eval tasks --model eval-data/model_f16.incord --tokenizer tokenizer.json \
    --items eval-data/tasks-f16.jsonl --max-new 512
incord-eval tasks --model <candidate.incord> ... --items eval-data/tasks-cand.jsonl --max-new 512
incord-eval compare --ref eval-data/tasks-f16.jsonl --cand eval-data/tasks-cand.jsonl
```

| Task | Questions (default) | Scored by |
|---|---|---|
| ARC-Challenge (test, CC BY-SA 4.0) | 300 of 1172 | log p of answer letter |
| MMLU (test, MIT) | 570 of 14042, all subjects | log p of answer letter |
| HellaSwag (validation, MIT) | 400 of 10042 | log p of ending per character |
| WinoGrande-xl (validation, CC BY) | 400 of 1267 | log p of the sentence after the blank |
| GSM8K (test, MIT) | 100 of 1319 | greedy answer, final number must match |

Questions are spread evenly over each set, so every model gets the same ones.
The items file records each answer, and a stopped run resumes where it left
off. `compare` counts the questions a candidate lost or gained against the
reference. With a few hundred questions, one standard error is 2-3 points,
so treat smaller accuracy differences as noise. The lost/gained counts are
the sharper signal.
