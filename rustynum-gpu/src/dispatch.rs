//! Auto-detect available compute backends and pick the best one per operation.
//!
//! The dispatch table is built once at startup. Each operation queries all
//! available backends for throughput estimates and picks the winner.
//!
//! ```text
//!                      Laptop (11gen)           NUC (185H)            Cloud (Sapphire)
//! ──────────────────────────────────────────────────────────────────────────────────────
//! Hamming (batch)    → CPU AVX-512 (200x)     → wgpu Xe-LPG          → CPU AVX-512
//! Hamming (single)   → CPU AVX-512            → CPU AVX2             → CPU AVX-512
//! MatMul F32         → CUDA cuBLAS            → wgpu Xe-LPG          → CPU rustyblas
//! MatMul INT8        → CUDA tensor cores      → Level Zero NPU       → CPU VNNI
//! HDR cascade        → CPU AVX-512            → wgpu Stroke1 + CPU   → CPU AVX-512
//! ```
//!
//! Self-configuring. No user decision required.

use crate::backends::cpu::CpuBackend;
use crate::traits::*;
use rustynum_core::simd::{HdrResult, PreciseMode};
use std::sync::OnceLock;

/// All detected backends, ordered by preference.
struct BackendRegistry {
    backends: Vec<Box<dyn ComputeBackend>>,
}

static REGISTRY: OnceLock<BackendRegistry> = OnceLock::new();

fn registry() -> &'static BackendRegistry {
    REGISTRY.get_or_init(|| {
        let mut backends: Vec<Box<dyn ComputeBackend>> = Vec::new();

        // CPU is always first (and always available)
        backends.push(Box::new(CpuBackend::new()));
        log::info!("rustynum-gpu: registered CPU backend");

        // Try wgpu (universal GPU)
        #[cfg(feature = "wgpu-backend")]
        {
            if let Some(wgpu) = crate::backends::wgpu_backend::WgpuBackend::new() {
                log::info!("rustynum-gpu: registered wgpu backend: {}", wgpu.info().name);
                backends.push(Box::new(wgpu));
            }
        }

        // Try CUDA (NVIDIA-specific)
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda) = crate::backends::cuda_backend::CudaBackend::new() {
                log::info!("rustynum-gpu: registered CUDA backend: {}", cuda.info().name);
                backends.push(Box::new(cuda));
            }
        }

        // Try Level Zero (Intel NPU + Xe)
        #[cfg(feature = "level-zero")]
        {
            if let Some(ze) = crate::backends::level_zero::LevelZeroBackend::new() {
                log::info!(
                    "rustynum-gpu: registered Level Zero backend: {}",
                    ze.info().name
                );
                backends.push(Box::new(ze));
            }
        }

        log::info!(
            "rustynum-gpu: {} backend(s) available",
            backends.len()
        );

        BackendRegistry { backends }
    })
}

/// Pick the best backend for a given operation.
fn best_backend(op: OpHint, batch_size: usize, element_bytes: usize) -> &'static dyn ComputeBackend {
    let reg = registry();
    reg.backends
        .iter()
        .max_by(|a, b| {
            let ea = a.estimate(op, batch_size, element_bytes);
            let eb = b.estimate(op, batch_size, element_bytes);
            ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|b| b.as_ref())
        .unwrap() // CPU is always present
}

// ════════════════════════════════════════════════════════════
// Public API — same signatures as rustynum-core, auto-dispatched
// ════════════════════════════════════════════════════════════

/// Batch Hamming distance with auto-dispatch.
///
/// Picks CPU, wgpu, CUDA, or Level Zero based on batch size and hardware.
pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    let reg = registry();

    // Try backends in throughput order (highest estimate first)
    let mut candidates: Vec<_> = reg
        .backends
        .iter()
        .map(|b| {
            let est = b.estimate(OpHint::HammingDistance, num_rows, row_bytes);
            (b.as_ref(), est)
        })
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (backend, _est) in &candidates {
        if let Some(result) = backend.hamming_batch(query, database, num_rows, row_bytes) {
            return result;
        }
    }

    // Should never reach here — CPU always implements hamming_batch
    unreachable!("CPU backend must implement hamming_batch")
}

/// HDR cascade search (Belichtungsmesser) with auto-dispatch.
///
/// On AVX-512 systems: pure CPU (already 200x faster than naive GPU).
/// On Meteor Lake: GPU Stroke 1 + CPU Strokes 2-3.
/// Falls through to CPU if GPU unavailable or batch too small.
pub fn hdr_cascade_search(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    num_vectors: usize,
    threshold: u64,
    precise_mode: PreciseMode,
) -> Vec<HdrResult> {
    let reg = registry();

    let mut candidates: Vec<_> = reg
        .backends
        .iter()
        .map(|b| {
            let est = b.estimate(OpHint::HammingPrefix, num_vectors, vec_bytes);
            (b.as_ref(), est)
        })
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (backend, _est) in &candidates {
        if let Some(result) = backend.hdr_cascade_search(
            query,
            database,
            vec_bytes,
            num_vectors,
            threshold,
            precise_mode,
        ) {
            return result;
        }
    }

    unreachable!("CPU backend must implement hdr_cascade_search")
}

/// Report all detected backends and their capabilities.
pub fn list_backends() -> Vec<&'static BackendInfo> {
    registry().backends.iter().map(|b| b.info()).collect()
}

/// Report which backend would be selected for a given operation.
pub fn which_backend(op: OpHint, batch_size: usize, element_bytes: usize) -> &'static BackendInfo {
    best_backend(op, batch_size, element_bytes).info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_backends() {
        env_logger::try_init().ok();
        let backends = list_backends();
        println!("Detected {} backend(s):", backends.len());
        for b in &backends {
            println!("  {:?}", b);
        }
        assert!(!backends.is_empty(), "CPU backend must always be present");
        assert_eq!(backends[0].device_kind, DeviceKind::Cpu);
    }

    #[test]
    fn test_dispatch_hamming() {
        let vec_bytes = 1250;
        let num_vectors = 10;
        let query = vec![0xAA_u8; vec_bytes];
        let database = vec![0x55_u8; vec_bytes * num_vectors];

        let distances = hamming_batch(&query, &database, num_vectors, vec_bytes);
        assert_eq!(distances.len(), num_vectors);
        // 0xAA ^ 0x55 = 0xFF → 8 bits per byte × 1250 bytes = 10000
        for &d in &distances {
            assert_eq!(d, 10000);
        }
    }

    #[test]
    fn test_which_backend() {
        let info = which_backend(OpHint::HammingPrefix, 100_000, 1250);
        println!("Best backend for 100K Hamming prefix: {:?}", info);
    }
}
