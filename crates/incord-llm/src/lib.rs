pub mod config;
pub mod gemv;
pub mod gemv_batch;
pub mod generate;
pub mod save;
pub mod simd;
pub mod system;
pub mod tokenizer;
pub mod vision;

#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
pub mod gpu_inference;
#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
pub mod gpu_loader;

// qor4b's bundled standalone HTTP server. Downstream consumers (e.g. the
// top-level qora-server) bring their own server stack and disable this
// via `default-features = false` to skip the axum / redb / argon2 deps.
#[cfg(feature = "server")]
pub mod server;
