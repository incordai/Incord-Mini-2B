# Incord-Mini-2B

Pure Rust multimodal inference engine and standalone server. No Python, no CUDA, no PyTorch, no external ML frameworks. Single executable + quantized model weights = portable AI that runs locally on Windows, Linux, and macOS.

**GPU accelerated** — auto-detects Vulkan (Windows/Linux) or Metal (macOS) GPU and runs inference on it. Falls back to CPU if no GPU available. **Smart system awareness** — detects RAM and CPU at startup and adjusts generation limits automatically.

## License

This project is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## What It Does

Incord-Mini-2B is a 2-billion parameter multimodal language model with built-in vision. It can:

- **Text generation** — answer questions, write code, reason through problems
- **Image understanding** — describe photos, answer questions about images
- **Video understanding** — analyze frame sequences, describe motion and temporal changes
- **Thinking mode** — extended chain-of-thought reasoning with configurable budget (`low`, `medium`, `high`)

## Architecture

Incord-Mini-2B is built on a pure-Rust high-performance Transformer architecture:

| Component | Details |
|-----------|---------|
| **Parameters** | 2B total |
| **Hidden dim** | 2048 |
| **Intermediate dim** | 6144 (SwiGLU) |
| **Layers** | 42 |
| **Attention Heads** | 16 Q heads (head_dim = 128) |
| **KV Heads** | 2 KV heads (Grouped Query Attention, 8:1 ratio) |
| **Vocabulary** | 130,560 tokens |
| **Context** | 128K tokens natively (RoPE theta = 5,000,000.0) |
| **Normalization** | RMSNorm (eps = 1e-6) |

### Vision Encoder
- ViT patch embedding with learned positional embeddings
- 2D spatial RoPE
- 2x2 spatial merger
- **Images**: single frame processed through patch embedding
- **Video**: temporal pairs processed over consecutive frames

## GPU Support

Incord-Mini-2B automatically detects and uses your GPU via the Burn framework's wgpu backend:

- **Windows/Linux**: Vulkan
- **macOS**: Metal (Apple Silicon and Intel)

```
Attempting GPU inference...
GPU initialized successfully
VRAM probe: 256MB OK
Weights loaded to GPU in 5.9s
```

GPU gives a **~2.5x speedup** over CPU. Use `--cpu` to force CPU-only inference.

## Smart System Awareness

Incord-Mini-2B detects your system at startup and automatically adjusts generation limits:

```
Incord-Mini-2B - Pure Rust Multimodal Inference Engine
System: 16101 MB RAM (8271 MB free), 12 threads
```

| Available RAM | Think Budget | Max Tokens | Behavior |
|---------------|-------------|------------|----------|
| < 4 GB | 128 (cap 256) | 256 (cap 512) | Minimal generation, warning displayed |
| 4-8 GB | 256 (cap 1024) | 512 (cap 1024) | Constrained, warning displayed |
| 8-12 GB | 1024 (cap 2048) | 1024 (cap 2048) | Normal operation |
| >= 12 GB | 2048 (cap 8192) | 2048 (cap 8192) | Full capability |

Hard caps apply even to explicit user values. Supports **Windows** (wmic), **Linux** (/proc/meminfo), and **macOS** (sysctl/vm_stat).

## Weight Formats

| Format | Size | Description |
|--------|------|-------------|
| **`model_q8.incord`** / **`model.incord`** | ~2.55 GB | 8-bit quantized weights (42 layers, vocab=130,560, hidden=2048) |

Multi-threaded GEMV/GEMM via rayon for CPU inference with AVX-512 SIMD kernels where supported.

## Platform Support

| Platform | Binary | GPU Backend | Status |
|----------|--------|-------------|--------|
| **Windows x86_64** | `incord.exe` | Vulkan | Tested |
| **Linux x86_64** | `incord` | Vulkan | Supported |
| **macOS aarch64** | `incord` | Metal | Supported |

Pre-built binaries and weights are available on the [GitHub Releases](https://github.com/incordai/Incord-Mini-2B/releases) page.

## Quick Start (CLI Inference)

1. Download files from [GitHub Releases](https://github.com/incordai/Incord-Mini-2B/releases):
   - `model.incord` (or `model_q8.incord`)
   - `tokenizer.json`
   - `incord.exe` (Windows) or build from source (Linux/macOS)

2. Place them in the same directory and run:

```bash
# Text generation (auto-detects GPU)
incord --prompt "Explain quantum computing" --max-tokens 500

# Force CPU-only
incord --prompt "Hello" --cpu

# Image understanding
incord --prompt "What's in this image?" --image photo.jpg

# Video understanding (directory of frame images)
incord --prompt "What happens in this video?" --video frames_dir/

# Thinking mode (default, extended reasoning)
incord --prompt "Solve: integral of x^2 * e^x dx" --think-budget 2048

# No-think mode (faster, direct answers)
incord --prompt "What is 2+2?" --no-think

# Greedy decoding (deterministic output)
incord --prompt "Hello" --greedy
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | Input prompt (default: "Hello, how are you?") |
| `--image PATH` | Path to an image file (PNG/JPG) |
| `--video PATH` | Path to directory of frame images (PNG/JPG, sorted by name) |
| `--max-tokens N` | Max tokens to generate (default: 1024) |
| `--think-budget N` | Max thinking tokens before forcing answer (default: 1024) |
| `--no-think` | Disable thinking mode (direct answers) |
| `--show-think` | Display thinking tokens on stderr |
| `--greedy` | Greedy decoding (temperature=0, not recommended with thinking mode) |
| `--cpu` | Force CPU inference (skip GPU auto-detection) |

## HTTP Server & OpenAI-Compatible API

In addition to CLI inference, Incord includes a standalone OpenAI-compatible server (`incord-server.exe`):

```bash
# Start server
incord-server.exe config/incord-server.toml
```

Send standard chat completions:

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer incord_testkey" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "incord-mini-2b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "reasoning_effort": "high"
  }'
```

Supported model identifiers:
- `incord-mini-2b` (canonical)
- `incord`
- `incord-2b`

## Building from Source

```bash
# CPU-only (all platforms)
cargo build --release

# GPU — Windows/Linux (Vulkan)
cargo build --release --features gpu

# GPU — macOS (Metal)
cargo build --release --features gpu-metal
```

## Repository

- **GitHub**: [https://github.com/incordai/Incord-Mini-2B](https://github.com/incordai/Incord-Mini-2B)
