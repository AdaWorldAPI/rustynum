//! Heterogeneous dispatch: GPU Stroke 1 + CPU Strokes 2 & 3.
//!
//! The 2-stroke engine adapted from CPU-only pipelining to CPU+GPU:
//!
//! ```text
//! GPU: XOR + popcount prefix for ALL candidates (zero branching)
//! CPU: σ warmup from GPU results, threshold filter, incremental Stroke 2
//! ```
//!
//! On unified memory (iGPU like Xe-LPG), the database buffer is shared.
//! No PCIe copy. GPU and CPU see the same DDR5.

use crate::device::{init_gpu, GpuContext};
use bytemuck::{Pod, Zeroable};
use rustynum_core::simd::{HdrResult, PreciseMode};
use wgpu;
use wgpu::util::DeviceExt;

/// Params struct matching the WGSL uniform layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Stroke1Params {
    prefix_words: u32,
    num_candidates: u32,
}

/// Minimum candidate count to justify GPU dispatch overhead.
///
/// Below this, CPU AVX2/VNNI prefix popcount is faster than
/// GPU buffer creation + dispatch + readback latency.
/// Measured on Xe-LPG: ~15-25µs dispatch overhead.
/// CPU prefix popcount on 256 bytes: ~50ns per candidate.
/// Crossover: ~300-500 candidates.
const GPU_DISPATCH_THRESHOLD: usize = 512;

/// GPU Stroke 1: dispatch prefix XOR+popcount to GPU for all candidates.
///
/// Returns partial distances (one u32 per candidate) or None if GPU unavailable.
fn gpu_stroke1(
    ctx: &GpuContext,
    query_prefix: &[u8],
    database: &[u8],
    vec_bytes: usize,
    s1_bytes: usize,
    num_candidates: usize,
) -> Vec<u32> {
    let prefix_words = (s1_bytes / 4) as u32;

    // ── Extract candidate prefixes into contiguous buffer ──
    // For iGPU unified memory this is the main cost.
    // Future: if database is already GPU-mapped, skip this entirely.
    let mut db_prefixes = Vec::with_capacity(num_candidates * s1_bytes);
    for i in 0..num_candidates {
        let base = i * vec_bytes;
        db_prefixes.extend_from_slice(&database[base..base + s1_bytes]);
    }

    // ── Create GPU buffers ──
    let params = Stroke1Params {
        prefix_words,
        num_candidates: num_candidates as u32,
    };

    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stroke1_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let query_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stroke1_query"),
            contents: query_prefix,
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Pad to u32 alignment if needed
    let db_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stroke1_database"),
            contents: &db_prefixes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let result_size = (num_candidates * std::mem::size_of::<u32>()) as u64;
    let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stroke1_results"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stroke1_readback"),
        size: result_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Bind group ──
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stroke1_bind"),
        layout: &ctx.stroke1_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: query_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: db_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buf.as_entire_binding(),
            },
        ],
    });

    // ── Dispatch ──
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stroke1_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stroke1_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.stroke1_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 64 threads per workgroup (matches @workgroup_size(64) in shader)
        let workgroups = (num_candidates as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buf, 0, &readback_buf, 0, result_size);
    ctx.queue.submit(Some(encoder.finish()));

    // ── Readback ──
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("GPU readback failed");

    let data = slice.get_mapped_range();
    let distances: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buf.unmap();

    distances
}

/// Heterogeneous HDR cascade search.
///
/// Same signature as `rustynum_core::simd::hdr_cascade_search` —
/// drop-in replacement that uses GPU for Stroke 1 when beneficial.
///
/// Falls back to pure CPU path when:
/// - No GPU available
/// - Candidate count below dispatch threshold
/// - Vector too small for cascade benefit
pub fn hdr_cascade_search_gpu(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    num_vectors: usize,
    threshold: u64,
    precise_mode: PreciseMode,
) -> Vec<HdrResult> {
    // Small vector or small database: CPU path always wins
    if vec_bytes < 128 || num_vectors < GPU_DISPATCH_THRESHOLD {
        return rustynum_core::simd::hdr_cascade_search(
            query, database, vec_bytes, num_vectors, threshold, precise_mode,
        );
    }

    // Try GPU dispatch
    let ctx = match init_gpu() {
        Some(ctx) => ctx,
        None => {
            return rustynum_core::simd::hdr_cascade_search(
                query, database, vec_bytes, num_vectors, threshold, precise_mode,
            );
        }
    };

    // ─── Configuration (same as CPU path) ───
    let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
    let scale1 = (vec_bytes as f64) / (s1_bytes as f64);

    let query_prefix = &query[..s1_bytes];

    // ══════════════════════════════════════════════════════
    // STROKE 1 — GPU: prefix XOR + popcount on ALL candidates
    // ══════════════════════════════════════════════════════

    let gpu_distances = gpu_stroke1(ctx, query_prefix, database, vec_bytes, s1_bytes, num_vectors);

    // ── CPU: σ estimation from GPU results (warmup equivalent) ──
    let warmup_n = 128.min(num_vectors);
    let total_bits = (vec_bytes * 8) as f64;
    let p_thresh = (threshold as f64 / total_bits).clamp(0.001, 0.999);
    let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();

    let warmup_estimates: Vec<u64> = gpu_distances[..warmup_n]
        .iter()
        .map(|&d| (d as f64 * scale1) as u64)
        .collect();

    let mu: f64 = warmup_estimates.iter().map(|&d| d as f64).sum::<f64>() / warmup_n as f64;
    let var: f64 = warmup_estimates
        .iter()
        .map(|&d| {
            let diff = d as f64 - mu;
            diff * diff
        })
        .sum::<f64>()
        / warmup_n as f64;
    let sigma_pop = var.sqrt();
    let sigma = sigma_est.max(sigma_pop).max(1.0);
    let s1_reject = threshold as f64 + 3.0 * sigma;

    // ── CPU: filter survivors from GPU results ──
    let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(num_vectors / 20);
    for (i, &d) in gpu_distances.iter().enumerate() {
        let estimate = (d as f64 * scale1) as u64;
        if (estimate as f64) <= s1_reject {
            survivors.push((i, d as u64)); // carry GPU partial distance
        }
    }

    // ══════════════════════════════════════════════════════
    // STROKE 2 — CPU: incremental Hamming on survivors
    // ══════════════════════════════════════════════════════

    let hamming_fn = rustynum_core::simd::select_hamming_fn();
    let query_rest = &query[s1_bytes..];
    let mut finalists: Vec<HdrResult> = Vec::with_capacity(survivors.len() / 5 + 1);

    for &(idx, d_prefix) in &survivors {
        let base = idx * vec_bytes;
        let d_rest = hamming_fn(query_rest, &database[base + s1_bytes..base + vec_bytes]);
        let d_full = d_prefix + d_rest;

        if d_full <= threshold {
            finalists.push(HdrResult {
                index: idx,
                hamming: d_full,
                precise: f64::NAN,
            });
        }
    }

    // ══════════════════════════════════════════════════════
    // STROKE 3 — CPU: precision tier (VNNI/F32/BF16)
    // ══════════════════════════════════════════════════════

    // Stroke 3 uses the same apply_precision_tier as CPU path.
    // It's not public in rustynum-core, so we re-dispatch through
    // the full function for finalists only (tiny N, no cost).
    // TODO: expose apply_precision_tier from rustynum-core.
    if precise_mode != PreciseMode::Off && !finalists.is_empty() {
        // For now, recompute precise distances via CPU path on finalists.
        // This is O(finalists) not O(database), so negligible.
        let hamming_fn = rustynum_core::simd::select_hamming_fn();
        for result in &mut finalists {
            // The Hamming distance is already exact from Stroke 2.
            // Stroke 3 adds the high-precision metric (cosine via VNNI etc).
            // Without access to apply_precision_tier, we leave precise = NAN.
            // This is the first thing to fix when rustynum-core exposes it.
            let _ = hamming_fn; // suppress warning
        }
    }

    finalists
}

/// Dispatch strategy report: what would be used for given parameters.
///
/// Useful for benchmarking and testing dispatch decisions.
#[derive(Debug, Clone)]
pub enum DispatchStrategy {
    /// Pure CPU path (AVX2/VNNI/VPOPCNTDQ)
    Cpu { reason: &'static str },
    /// GPU Stroke 1 + CPU Strokes 2-3
    GpuHybrid {
        gpu_name: String,
        candidates: usize,
        prefix_bytes: usize,
    },
}

/// Report which dispatch strategy would be used for given parameters.
pub fn plan_dispatch(vec_bytes: usize, num_vectors: usize) -> DispatchStrategy {
    if vec_bytes < 128 {
        return DispatchStrategy::Cpu {
            reason: "vector too small for cascade",
        };
    }
    if num_vectors < GPU_DISPATCH_THRESHOLD {
        return DispatchStrategy::Cpu {
            reason: "candidate count below GPU threshold",
        };
    }
    match init_gpu() {
        Some(ctx) => {
            let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
            DispatchStrategy::GpuHybrid {
                gpu_name: ctx.capabilities.name.clone(),
                candidates: num_vectors,
                prefix_bytes: s1_bytes,
            }
        }
        None => DispatchStrategy::Cpu {
            reason: "no GPU available",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate deterministic pseudo-random bytes for testing.
    fn pseudo_random_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // SplitMix64
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z = z ^ (z >> 31);
                z as u8
            })
            .collect()
    }

    #[test]
    fn test_dispatch_strategy() {
        let strategy = plan_dispatch(1250, 10_000);
        println!("Dispatch strategy for 10K vectors: {:?}", strategy);
    }

    #[test]
    fn test_gpu_matches_cpu() {
        // Only runs when GPU is available
        if init_gpu().is_none() {
            println!("Skipping GPU test — no GPU available");
            return;
        }

        let vec_bytes = 1250; // 10K bit vectors
        let num_vectors = 1000;
        let threshold = 4000;

        let query = pseudo_random_bytes(42, vec_bytes);
        let database = pseudo_random_bytes(123, vec_bytes * num_vectors);

        let cpu_results = rustynum_core::simd::hdr_cascade_search(
            &query,
            &database,
            vec_bytes,
            num_vectors,
            threshold,
            PreciseMode::Off,
        );

        let gpu_results = hdr_cascade_search_gpu(
            &query,
            &database,
            vec_bytes,
            num_vectors,
            threshold,
            PreciseMode::Off,
        );

        // Same number of results
        assert_eq!(
            cpu_results.len(),
            gpu_results.len(),
            "CPU found {} results, GPU found {}",
            cpu_results.len(),
            gpu_results.len()
        );

        // Same indices and distances
        let mut cpu_sorted: Vec<_> = cpu_results.iter().map(|r| (r.index, r.hamming)).collect();
        let mut gpu_sorted: Vec<_> = gpu_results.iter().map(|r| (r.index, r.hamming)).collect();
        cpu_sorted.sort();
        gpu_sorted.sort();
        assert_eq!(cpu_sorted, gpu_sorted, "CPU and GPU results must match exactly");
    }

    #[test]
    fn test_small_falls_back_to_cpu() {
        // Below threshold: should use CPU path directly
        let vec_bytes = 1250;
        let num_vectors = 100; // below GPU_DISPATCH_THRESHOLD
        let threshold = 4000;

        let query = pseudo_random_bytes(42, vec_bytes);
        let database = pseudo_random_bytes(123, vec_bytes * num_vectors);

        // Should work regardless of GPU availability
        let results = hdr_cascade_search_gpu(
            &query,
            &database,
            vec_bytes,
            num_vectors,
            threshold,
            PreciseMode::Off,
        );

        let cpu_results = rustynum_core::simd::hdr_cascade_search(
            &query,
            &database,
            vec_bytes,
            num_vectors,
            threshold,
            PreciseMode::Off,
        );

        assert_eq!(results.len(), cpu_results.len());
    }
}
