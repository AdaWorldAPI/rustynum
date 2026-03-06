//! NVIDIA CUDA backend — cuBLAS + tensor cores.
//!
//! Feature gate: `cuda`
//! Requires: CUDA toolkit + NVIDIA driver.
//!
//! Use on:
//!   - Laptops with 3050Ti/4060/etc (discrete, limited VRAM)
//!   - Servers with A100/H100 (datacenter)
//!
//! What this provides over wgpu on NVIDIA hardware:
//!   - cuBLAS sgemm: 3-5x faster than hand-written compute shader
//!   - Tensor cores: INT8/BF16 matmul at peak throughput
//!   - CUDA-specific memory management (pinned host, async copy)
//!
//! The wgpu backend also works on NVIDIA via Vulkan, but without
//! cuBLAS/tensor core access. Use this for maximum NVIDIA performance.

use crate::traits::*;
use rustynum_core::simd::{HdrResult, PreciseMode};

pub struct CudaBackend {
    info: BackendInfo,
    // cudarc device handle will go here
    // device: cudarc::driver::CudaDevice,
}

impl CudaBackend {
    /// Initialize CUDA backend. Returns None if no NVIDIA GPU found.
    pub fn new() -> Option<Self> {
        // TODO: use cudarc to enumerate devices
        //
        // let device = cudarc::driver::CudaDevice::new(0).ok()?;
        // let props = device.get_device_properties();
        //
        // For now: detect via cudarc when the dependency resolves.

        log::info!("rustynum-gpu/cuda: CUDA backend not yet implemented");
        None

        // Skeleton for when cudarc is wired up:
        //
        // Some(CudaBackend {
        //     info: BackendInfo {
        //         name: format!("CUDA: {}", props.name()),
        //         device_kind: DeviceKind::DiscreteGpu,
        //         available_memory: props.total_global_mem() as usize,
        //         unified_memory: false, // discrete GPU, PCIe boundary
        //         int8_tops: estimate_nvidia_int8_tops(&props),
        //     },
        //     device,
        // })
    }
}

impl ComputeBackend for CudaBackend {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn estimate(&self, op: OpHint, batch_size: usize, element_bytes: usize) -> f64 {
        let data_size = batch_size * element_bytes;
        let vram = self.info.available_memory;

        // If data doesn't fit in VRAM, GPU path is worse than CPU
        // (would need streaming, which kills throughput)
        if data_size > vram {
            return 0.0;
        }

        match op {
            // cuBLAS matmul is the CUDA sweet spot
            OpHint::MatmulF32 => 100.0 * batch_size as f64,
            // Tensor cores for INT8 dot product
            OpHint::DotI8 => 200.0 * batch_size as f64,
            // Hamming: CUDA can do it but no special advantage over wgpu
            OpHint::HammingDistance | OpHint::HammingPrefix => 30.0 * batch_size as f64,
            OpHint::ElementwiseF32 => 20.0 * batch_size as f64,
            OpHint::ReduceF32 => 5.0 * batch_size as f64,
        }
    }

    fn sgemm(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _beta: f32,
    ) -> bool {
        // TODO: cudarc cublas sgemm
        // cudarc::cublas::Cublas::new(self.device.clone())
        //     .sgemm(...)
        false
    }

    fn gemm_i8(
        &self,
        _a: &[i8],
        _b: &[i8],
        _c: &mut [i32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> bool {
        // TODO: cudarc cublas int8 gemm (tensor cores)
        false
    }

    fn hamming_batch(
        &self,
        _query: &[u8],
        _database: &[u8],
        _num_rows: usize,
        _row_bytes: usize,
    ) -> Option<Vec<u64>> {
        // TODO: custom CUDA kernel for XOR + __popc
        // PTX is trivial — similar to the WGSL shader.
        None
    }

    fn hdr_cascade_search(
        &self,
        _query: &[u8],
        _database: &[u8],
        _vec_bytes: usize,
        _num_vectors: usize,
        _threshold: u64,
        _precise_mode: PreciseMode,
    ) -> Option<Vec<HdrResult>> {
        // TODO: CUDA Stroke 1 + CPU Strokes 2-3
        // Same pattern as wgpu but with CUDA dispatch.
        None
    }
}
