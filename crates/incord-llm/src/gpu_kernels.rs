//! Our own GPU kernels, launched on cortex-v2 tensors through a backend
//! extension (compiled by CubeCL for CUDA, wgpu/Vulkan/DX12 and Metal).
//!
//! `q4_gemv`: decode-time `y[1, n] = x[1, k] · W[k, n]` for incord's Q4
//! layout (block-32 symmetric along n, signed nibbles packed 8 per u32 along
//! n). cortex's generic quantized matmul runs this single-row case through a
//! batch-matmul kernel at ~1-2 ms per layer-sized matrix when its input comes
//! from the previous op; this kernel does it in ~0.05-0.2 ms.
//!
//! Layout: one thread owns one packed word (8 output columns) over a slice of
//! the k rows, so neighbouring threads read neighbouring words of a row
//! (coalesced). Slices write partial sums `[slices, n]`, summed afterwards.

// `Tensor` in this module is CubeCL's kernel tensor; cortex's high-level
// tensor is always written out as `cortex::tensor::Tensor`.
use cortex::backend::cubecl::dtype_to_storage_type;
use cortex::backend::tensor::{FloatTensor, QuantizedTensor};
#[cfg(feature = "gpu-cuda")]
use cortex::backend::Cuda;
#[cfg(feature = "gpu-metal")]
use cortex::backend::Metal;
#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
use cortex::backend::Wgpu;
use cortex::backend::{Dispatch, backend_extension};
use cortex::tensor::{DType, Shape};
use cortex_cubecl::{CubeBackend, CubeRuntime, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// Signed value of nibble `i` of `w` (two's complement 4-bit).
#[cube]
fn nibble(w: u32, #[comptime] i: u32) -> f32 {
    let v = (w >> (4 * i)) & 15;
    f32::cast_from(v) - f32::cast_from(v >> 3) * 16.0
}

#[cube(launch)]
fn q4_gemv_kernel<S: Float>(
    x: &Tensor<f32>,
    data: &Tensor<u32>,
    scales: &Tensor<S>,
    partial: &mut Tensor<f32>,
    #[define(S)] _scale_dtype: StorageType,
) {
    let word = ABSOLUTE_POS_X as usize;
    let slice = ABSOLUTE_POS_Y as usize;
    let k = data.shape(0);
    let words = data.shape(1);
    let slices = partial.shape(0);
    if word >= words || slice >= slices {
        terminate!();
    }
    let groups = words / 4; // one scale per 32 columns = 4 words
    let g = word / 4;
    let rows = (k + slices - 1) / slices;
    let r0 = slice * rows;
    let mut r1 = r0 + rows;
    if r1 > k {
        r1 = k;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut a4 = 0.0f32;
    let mut a5 = 0.0f32;
    let mut a6 = 0.0f32;
    let mut a7 = 0.0f32;
    for r in r0..r1 {
        let w = data[r * words + word];
        let s = f32::cast_from(scales[r * groups + g]) * x[r];
        a0 += s * nibble(w, 0u32);
        a1 += s * nibble(w, 1u32);
        a2 += s * nibble(w, 2u32);
        a3 += s * nibble(w, 3u32);
        a4 += s * nibble(w, 4u32);
        a5 += s * nibble(w, 5u32);
        a6 += s * nibble(w, 6u32);
        a7 += s * nibble(w, 7u32);
    }
    let base = slice * words * 8 + word * 8;
    partial[base] = a0;
    partial[base + 1] = a1;
    partial[base + 2] = a2;
    partial[base + 3] = a3;
    partial[base + 4] = a4;
    partial[base + 5] = a5;
    partial[base + 6] = a6;
    partial[base + 7] = a7;
}

// One registration per build: the extension lists exactly the GPU backends
// this build contains (Metal builds also contain the wgpu backend).
#[cfg_attr(all(feature = "gpu", feature = "gpu-cuda", not(feature = "gpu-metal")), backend_extension(Wgpu, Cuda))]
#[cfg_attr(all(feature = "gpu", not(feature = "gpu-cuda"), not(feature = "gpu-metal")), backend_extension(Wgpu))]
#[cfg_attr(all(feature = "gpu-cuda", not(feature = "gpu"), not(feature = "gpu-metal")), backend_extension(Cuda))]
#[cfg_attr(all(feature = "gpu-metal", not(feature = "gpu-cuda")), backend_extension(Wgpu, Metal))]
#[cfg_attr(all(feature = "gpu-metal", feature = "gpu-cuda"), backend_extension(Wgpu, Metal, Cuda))]
pub trait IncordKernels: cortex::backend::Backend {
    /// Partial sums `[slices, n]` of `x[1, k] · w[k, n]` (w: incord Q4 layout).
    fn q4_gemv_partial(x: FloatTensor<Self>, w: QuantizedTensor<Self>, slices: usize) -> FloatTensor<Self>;
}

impl<R: CubeRuntime> IncordKernels for CubeBackend<R> {
    fn q4_gemv_partial(x: FloatTensor<Self>, w: QuantizedTensor<Self>, slices: usize) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let (data, scales) = w.quantized_handles().expect("q4_gemv needs a quantized weight");
        let words = data.meta.shape()[1];
        let n = words * 8;
        let buffer = x.client.empty(slices * n * DType::F32.size());
        let partial = CubeTensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            Shape::from(vec![slices, n]),
            buffer,
            DType::F32,
        );
        let cube_dim = CubeDim { x: 64, y: 4, z: 1 };
        let count = CubeCount::Static(words.div_ceil(64) as u32, slices.div_ceil(4) as u32, 1);
        let scale_dtype = dtype_to_storage_type(scales.dtype);
        q4_gemv_kernel::launch(
            &partial.client,
            count,
            cube_dim,
            x.into_tensor_arg(),
            data.into_tensor_arg(),
            scales.into_tensor_arg(),
            partial.clone().into_tensor_arg(),
            scale_dtype,
        );
        partial
    }
}

/// `x[1, k] · w[k, n]` for a Q4 weight uploaded by `gpu_loader`.
pub fn q4_gemv(x: cortex::tensor::Tensor<2>, w: cortex::tensor::Tensor<2>) -> cortex::tensor::Tensor<2> {
    let k = x.dims()[1];
    // Enough row slices to occupy the GPU; each slice covers >= 64 rows.
    let slices = (k / 64).clamp(1, 64);
    let partial = cortex::tensor::Tensor::<2>::from_dispatch(<Dispatch as IncordKernels>::q4_gemv_partial(
        x.into_dispatch(),
        w.into_dispatch(),
        slices,
    ));
    partial.sum_dim(0)
}

#[cfg(test)]
mod tests {
    //! The kernel against incord's CPU gemv on the real model's shapes.

    use crate::gemv::{Q4Weight, Weight, gemv};
    use crate::gpu_inference::gpu_device;
    use crate::quant::{Rounding, quantize_q4};

    fn noise(len: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * amp
            })
            .collect()
    }

    #[test]
    fn q4_gemv_matches_cpu_on_model_shapes() {
        let dev = gpu_device();
        // (k, n): q/o, k/v, gate/up, down, and a k that is not a multiple of 64.
        for (k, n) in [(2048usize, 2048usize), (2048, 256), (2048, 6144), (6144, 2048), (100, 64)] {
            let w = noise(k * n, (k * 31 + n) as u64, 0.05);
            let x = noise(k, 7, 1.0);
            let (packed, scales) = quantize_q4(&w, k, n, Rounding::Rtn);
            let weight = Weight::Q4(Q4Weight { packed: packed.into(), scales: scales.into(), k, n });
            let cpu = gemv(&x, &weight);
            let mat = crate::gpu_loader::mat_to_gpu(&weight, &dev).unwrap();
            assert!(mat.q4);
            let xt = cortex::tensor::Tensor::<2>::from_data(cortex::tensor::TensorData::new(x, [1, k]), &dev);
            let gpu = mat.apply(xt).into_data().to_vec::<f32>().unwrap();
            assert_eq!(gpu.len(), n);
            let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = cpu.iter().zip(&gpu).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(diff <= 1e-4 * scale, "[{k}x{n}] max |cpu - gpu| {diff} vs scale {scale}");
        }
    }
}
