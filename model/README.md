---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language:
  - en
  - zh
  - multilingual
library_name: rust
tags:
  - text-generation
  - image-text-to-text
  - video-text-to-text
  - multimodal
  - vision
  - rust
  - pure-rust
  - no-python
  - quantized
  - deltanet
  - hybrid-attention
  - gpu
  - vulkan
  - metal
pipeline_tag: image-text-to-text
model-index:
  - name: Incord-Mini-2B
    results: []
---

# Incord-Mini-2B (Incord)

Pure Rust multimodal inference engine and standalone server. No Python, no CUDA, no PyTorch, no external C++ ML frameworks. Single executable + quantized model weights = portable AI that runs locally on Windows, Linux, and macOS.

---

## Directory Contents

This `model` folder is completely self-contained with everything needed to run text and multimodal inference:

| File | Size | Description |
|------|------|-------------|
| **`incord.exe`** | ~70 MB | Standalone CLI inference engine executable. Auto-loads model and tokenizer from this folder. |
| **`incord-server.exe`** | ~36 MB | Standalone OpenAI-compatible HTTP API server executable (`/v1/chat/completions`, `/v1/models`). |
| **`model_q8.incord`** / **`model.incord`** | ~2.55 GB | 8-bit quantized weights (42 layers: hybrid DeltaNet + Full Attention, vocab=130,560, hidden=2048). |
| **`tokenizer.json`** | ~12.8 MB | Vocabulary, BPE tokenizer, and chat templates. |
| **`README.md`** | — | This documentation guide. |

---

## Quick Start (CLI Inference)

Open PowerShell or Command Prompt, navigate to this folder, and run `incord.exe`:

```powershell
cd model

# 1. Basic prompt (uses default prompt or pass your own)
.\incord.exe --prompt "Explain the theory of relativity in simple terms."

# 2. Fast direct answer (disable thinking block)
.\incord.exe --no-think --prompt "What is the capital of France?"

# 3. Force CPU-only mode
.\incord.exe --cpu --prompt "Write a quick Rust hello world program."

# 4. Stream thinking reasoning steps in real-time
.\incord.exe --show-think --prompt "Solve: If 3x + 7 = 22, what is x?"

# 5. Control generation token limits
.\incord.exe --max-tokens 512 --think-budget 256 --prompt "Summarize renewable energy."

# 6. Deterministic output (greedy sampling)
.\incord.exe --greedy --no-think --prompt "Count from 1 to 5."
```

---

## CLI Options & Flags

When running `incord.exe`, the executable automatically searches the current directory for `model_q8.incord` (or `model.incord`) and `tokenizer.json`.

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt <TEXT>` | `"Hello, how are you?"` | The input user prompt. |
| `--cpu` | `false` | Force CPU-only inference (skips GPU probe and auto-detection). |
| `--max-tokens <N>` | `1024` | Maximum new tokens to generate. |
| `--think-budget <N>` | `1024` | Maximum thinking tokens allowed before forcing final answer. |
| `--no-think` | `false` | Disable `<think>` reasoning block for faster, direct responses. |
| `--show-think` | `false` | Print `<think>` tokens to stderr as they are generated. |
| `--greedy` | `false` | Greedy decoding (temperature = 0.0). |
| `--image <PATH>` | `None` | Path to image file for vision understanding. |
| `--video <DIR>` | `None` | Path to folder containing video frame images. |
| `--model <PATH>` | Auto-detected | Custom path to `.incord` model weights file. |
| `--tokenizer <PATH>` | Auto-detected | Custom path to `tokenizer.json`. |

---

## OpenAI-Compatible HTTP Server (`incord-server.exe`)

A standalone HTTP API server (`incord-server.exe`, Release build ~14.5 MB) is provided alongside the CLI runner. It supports OpenAI-compatible endpoints, concurrent slot pooling, and streaming token generation.

### 1. Configuration & Start

The server reads configuration from `incord-server.toml` (located in the same directory) and requires two environment variables:

```powershell
cd model

# 1. Set environment variables (operator chosen)
$env:INCORD_ADMIN_TOKEN   = "<OPERATOR_ADMIN_TOKEN>"
$env:INCORD_SERVER_SECRET = "<OPERATOR_SERVER_SECRET>"

# 2. Launch the server
.\incord-server.exe incord-server.toml
```

The server loads `model_q8.incord` eagerly into memory (~35s) and listens on `http://127.0.0.1:8080`.

### 2. Health & Model Discovery

Check server readiness:
```powershell
curl http://127.0.0.1:8080/health
# Response: {"status":"ok"}
```

List available models:
```powershell
curl http://127.0.0.1:8080/v1/models
```

### 3. Generate API Key (Admin API)

To generate an API key for client chat completions:
```powershell
curl -X POST http://127.0.0.1:8080/v1/admin/keys `
  -H "Authorization: Bearer <OPERATOR_ADMIN_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{"name": "client-1", "rpm_limit": 60, "tpm_limit": 60000}'
```

### 4. Send Chat Completion (`curl`)

```powershell
curl http://127.0.0.1:8080/v1/chat/completions `
  -H "Authorization: Bearer <KEY_FROM_STEP_3>" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "incord-mini-2b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain Rust ownership in two sentences."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### 5. Use with Python `openai` SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="<KEY_FROM_STEP_3>"
)

response = client.chat.completions.create(
    model="incord-mini-2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do DeltaNet layers work in language models?"}
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

---

## Hardware & System Requirements

### RAM & System Limits
The engine detects system RAM at startup and clamps limits safely:
- **Available RAM < 4 GB**: Caps max tokens to 512 (think budget 256).
- **Available RAM 4–8 GB**: Caps max tokens to 1024 (think budget 512).
- **Available RAM > 8 GB**: Allows full generation (up to 2048+ tokens).

### GPU VRAM Behavior
- **Backend**: Vulkan / WGPU on Windows & Linux, Metal on macOS.
- **VRAM Requirements**:
  - Full model loading onto GPU requires **~3.4 GB free VRAM** (model weights + KV cache).
  - If available GPU memory is less than 3.4 GB (e.g. on a 6 GB GPU where Windows and desktop apps consume ~3.2 GB, leaving ~2.8 GB free), the engine will detect the VRAM boundary and safely fall back to CPU execution.
  - To skip GPU allocation attempts entirely and run immediately on CPU, pass the `--cpu` flag.

---

## Technical Architecture

- **Model Architecture**: Hybrid DeltaNet (Linear Attention) + Full Attention (Grouped Query Attention).
- **Quantization**: Symmetric 8-bit quantization (`Q8`) with per-group scaling.
- **Tokenizer**: Byte-level BPE with 130,560 vocabulary size, handling thinking tags (`<think> ... </think>`) and role markers (`<|im_start|>`, `<|im_end|>`).
- **Dependencies**: 100% self-contained pure Rust binary linked with `rayon`, `tokenizers`, `wgpu`, and `cortex`.
