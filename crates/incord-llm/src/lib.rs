pub mod config;
pub mod e8;
pub mod eval;
pub mod gemv;
pub mod gemv_batch;
#[cfg(feature = "quantize")]
pub mod gptq;
pub mod generate;
pub mod plan;
pub mod quant;
pub mod rotate;
pub mod save;
pub mod simd;
pub mod store;
pub mod system;
pub mod tasks;
pub mod tokenizer;
pub mod vision;

#[cfg(any(feature = "gpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_inference;
#[cfg(any(feature = "gpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_kernels;
#[cfg(any(feature = "gpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub mod gpu_loader;

// incord's bundled standalone HTTP server. Downstream consumers (e.g. the
// top-level incord-server) bring their own server stack and disable this
// via `default-features = false` to skip the axum / redb / argon2 deps.
#[cfg(feature = "server")]
pub mod server;
