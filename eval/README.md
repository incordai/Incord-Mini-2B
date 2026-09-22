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
