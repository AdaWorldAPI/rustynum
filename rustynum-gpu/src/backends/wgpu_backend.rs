//! Universal GPU backend via wgpu (Vulkan/DX12/Metal).
//!
//! Works on: Intel Xe-LPG/Xe2, NVIDIA (via Vulkan), AMD, Apple, Adreno.
//! No vendor SDK required. Just a GPU driver.
//!
//! Feature gate: `wgpu-backend`

use crate::traits::*;
use bytemuck::{Pod, Zeroable};
use rustynum_core::simd::{self, HdrResult, PreciseMode};
use std::sync::OnceLock;
use wgpu;
use wgpu::util::DeviceExt;

// ────────────────────────────────────────────────────────────
// GPU context — created once, reused for all dispatches
// ────────────────────────────────────────────────────────────

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: BackendInfo,
    stroke1_pipeline: wgpu::ComputePipeline,
    stroke1_bind_group_layout: wgpu::BindGroupLayout,
}

static WGPU_CTX: OnceLock<Option<WgpuContext>> = OnceLock::new();

fn get_context() -> Option<&'static WgpuContext> {
    WGPU_CTX
        .get_or_init(|| pollster::block_on(init_wgpu()))
        .as_ref()
}

async fn init_wgpu() -> Option<WgpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
        ..Default::default()
    });

    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).collect();
    if adapters.is_empty() {
        log::info!("rustynum-gpu/wgpu: no adapters found");
        return None;
    }

    // Prefer iGPU (unified memory, zero copy), then dGPU
    let adapter = adapters
        .iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
        .or_else(|| {
            adapters
                .iter()
                .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        })
        .or(adapters.first())?;

    let wgpu_info = adapter.get_info();
    log::info!(
        "rustynum-gpu/wgpu: {} ({:?} via {:?})",
        wgpu_info.name,
        wgpu_info.device_type,
        wgpu_info.backend
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rustynum-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .ok()?;

    let limits = adapter.limits();

    let device_kind = match wgpu_info.device_type {
        wgpu::DeviceType::IntegratedGpu => DeviceKind::IntegratedGpu,
        wgpu::DeviceType::DiscreteGpu => DeviceKind::DiscreteGpu,
        _ => DeviceKind::Cpu,
    };

    let info = BackendInfo {
        name: format!("wgpu: {}", wgpu_info.name),
        device_kind,
        available_memory: limits.max_buffer_size as usize,
        unified_memory: device_kind == DeviceKind::IntegratedGpu,
        int8_tops: match device_kind {
            DeviceKind::IntegratedGpu => 5.0, // rough Xe-LPG estimate
            DeviceKind::DiscreteGpu => 20.0,  // rough mid-range estimate
            _ => 0.0,
        },
    };

    // Stroke 1 pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stroke1"),
        source: wgpu::ShaderSource::Wgsl(include_str!("stroke1.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stroke1_layout"),
        entries: &[
            bgl_entry(0, wgpu::BufferBindingType::Uniform),
            bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("stroke1_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stroke1"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("stroke1"),
        compilation_options: Default::default(),
        cache: None,
    });

    Some(WgpuContext {
        device,
        queue,
        info,
        stroke1_pipeline: pipeline,
        stroke1_bind_group_layout: bind_group_layout,
    })
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ────────────────────────────────────────────────────────────
// Stroke 1 GPU dispatch
// ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Stroke1Params {
    prefix_words: u32,
    num_candidates: u32,
}

fn gpu_stroke1_prefix(
    ctx: &WgpuContext,
    query_prefix: &[u8],
    database: &[u8],
    vec_bytes: usize,
    s1_bytes: usize,
    num_candidates: usize,
) -> Vec<u32> {
    let prefix_words = (s1_bytes / 4) as u32;

    // Extract contiguous prefixes for GPU
    let mut db_prefixes = Vec::with_capacity(num_candidates * s1_bytes);
    for i in 0..num_candidates {
        let base = i * vec_bytes;
        db_prefixes.extend_from_slice(&database[base..base + s1_bytes]);
    }

    let params = Stroke1Params {
        prefix_words,
        num_candidates: num_candidates as u32,
    };

    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let query_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("query"),
            contents: query_prefix,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let db_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("database"),
            contents: &db_prefixes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let result_size = (num_candidates * 4) as u64;
    let result_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("results"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: result_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stroke1_bind"),
        layout: &ctx.stroke1_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: query_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: db_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: result_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stroke1"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.stroke1_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((num_candidates as u32 + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result_buf, 0, &readback_buf, 0, result_size);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().expect("GPU readback failed");

    let data = slice.get_mapped_range();
    let distances: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    readback_buf.unmap();
    distances
}

// ────────────────────────────────────────────────────────────
// WgpuBackend — implements ComputeBackend
// ────────────────────────────────────────────────────────────

/// Minimum candidates to justify GPU dispatch overhead.
const GPU_DISPATCH_THRESHOLD: usize = 512;

pub struct WgpuBackend;

impl WgpuBackend {
    /// Try to create a wgpu backend. Returns None if no GPU available.
    pub fn new() -> Option<Self> {
        get_context().map(|_| WgpuBackend)
    }
}

impl ComputeBackend for WgpuBackend {
    fn info(&self) -> &BackendInfo {
        // Safe: WgpuBackend::new() only succeeds if context exists
        &get_context().unwrap().info
    }

    fn estimate(&self, op: OpHint, batch_size: usize, _element_bytes: usize) -> f64 {
        if batch_size < GPU_DISPATCH_THRESHOLD {
            return 0.0; // too small, CPU wins
        }
        match op {
            OpHint::HammingPrefix => 50.0 * batch_size as f64, // GPU Stroke 1 sweet spot
            OpHint::HammingDistance => 30.0 * batch_size as f64,
            OpHint::MatmulF32 => 10.0 * batch_size as f64, // wgpu matmul is decent
            OpHint::DotI8 => 5.0 * batch_size as f64,
            OpHint::ElementwiseF32 => 2.0 * batch_size as f64,
            OpHint::ReduceF32 => 1.0 * batch_size as f64, // reductions are CPU-friendly
        }
    }

    fn hamming_batch(
        &self,
        query: &[u8],
        database: &[u8],
        num_rows: usize,
        row_bytes: usize,
    ) -> Option<Vec<u64>> {
        if num_rows < GPU_DISPATCH_THRESHOLD {
            return None; // let CPU handle it
        }
        let ctx = get_context()?;
        let distances = gpu_stroke1_prefix(ctx, query, database, row_bytes, row_bytes, num_rows);
        Some(distances.into_iter().map(|d| d as u64).collect())
    }

    fn hdr_cascade_search(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        num_vectors: usize,
        threshold: u64,
        precise_mode: PreciseMode,
    ) -> Option<Vec<HdrResult>> {
        if vec_bytes < 128 || num_vectors < GPU_DISPATCH_THRESHOLD {
            return None;
        }

        let ctx = get_context()?;

        // ── Configuration (same as CPU path) ──
        let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
        let scale1 = (vec_bytes as f64) / (s1_bytes as f64);

        // ═══ STROKE 1: GPU prefix popcount ═══
        let gpu_distances =
            gpu_stroke1_prefix(ctx, &query[..s1_bytes], database, vec_bytes, s1_bytes, num_vectors);

        // ── σ estimation from GPU results ──
        let warmup_n = 128.min(num_vectors);
        let total_bits = (vec_bytes * 8) as f64;
        let p_thresh = (threshold as f64 / total_bits).clamp(0.001, 0.999);
        let sigma_est =
            (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();

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
        let sigma = sigma_est.max(var.sqrt()).max(1.0);
        let s1_reject = threshold as f64 + 3.0 * sigma;

        // ── Filter survivors ──
        let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(num_vectors / 20);
        for (i, &d) in gpu_distances.iter().enumerate() {
            let estimate = (d as f64 * scale1) as u64;
            if (estimate as f64) <= s1_reject {
                survivors.push((i, d as u64));
            }
        }

        // ═══ STROKE 2: CPU incremental Hamming on survivors ═══
        let hamming_fn = simd::select_hamming_fn();
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

        // ═══ STROKE 3: CPU precision tier ═══
        // TODO: expose apply_precision_tier from rustynum-core
        let _ = precise_mode;

        Some(finalists)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_backend_detection() {
        env_logger::try_init().ok();
        match WgpuBackend::new() {
            Some(backend) => println!("wgpu backend: {:?}", backend.info()),
            None => println!("No GPU available (CI/headless)"),
        }
    }
}
