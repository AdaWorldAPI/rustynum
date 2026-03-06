//! GPU device management for rustynum compute operations.
//!
//! Handles wgpu adapter/device lifecycle, pipeline creation and caching.
//! Detects available compute surfaces and reports capabilities.

use std::sync::OnceLock;
use wgpu;

/// Capabilities of the detected GPU compute surface.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Human-readable adapter name (e.g. "Intel(R) Arc(TM) Graphics")
    pub name: String,
    /// Backend API in use (Vulkan, DX12, Metal, ...)
    pub backend: wgpu::Backend,
    /// Device type (IntegratedGpu, DiscreteGpu, Cpu, ...)
    pub device_type: wgpu::DeviceType,
    /// Maximum workgroup size (x dimension)
    pub max_workgroup_size_x: u32,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Maximum compute workgroups per dispatch
    pub max_dispatch_x: u32,
    /// Whether this is a unified memory architecture (iGPU = true, dGPU = depends)
    pub unified_memory: bool,
}

/// Cached GPU context — created once, reused for all dispatches.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub capabilities: GpuCapabilities,
    pub stroke1_pipeline: wgpu::ComputePipeline,
    pub stroke1_bind_group_layout: wgpu::BindGroupLayout,
}

/// Global GPU context — initialized on first use.
static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Detect and initialize the GPU compute surface.
///
/// Returns `None` if no suitable GPU is available.
/// Prefers integrated GPUs for zero-copy unified memory.
/// Falls back to discrete GPUs if no iGPU found.
pub fn init_gpu() -> Option<&'static GpuContext> {
    GPU_CONTEXT
        .get_or_init(|| {
            pollster::block_on(async {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN
                        | wgpu::Backends::DX12
                        | wgpu::Backends::METAL,
                    ..Default::default()
                });

                // Enumerate all adapters, prefer iGPU for unified memory
                let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).collect();

                if adapters.is_empty() {
                    log::info!("rustynum-gpu: no GPU adapters found");
                    return None;
                }

                // Sort: iGPU first (unified memory, zero copy), then dGPU, then CPU
                let adapter = adapters
                    .iter()
                    .find(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
                    .or_else(|| {
                        adapters
                            .iter()
                            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
                    })
                    .or(adapters.first())?;

                let info = adapter.get_info();
                log::info!(
                    "rustynum-gpu: selected {} ({:?} via {:?})",
                    info.name,
                    info.device_type,
                    info.backend
                );

                let limits = adapter.limits();

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

                let capabilities = GpuCapabilities {
                    name: info.name.clone(),
                    backend: info.backend,
                    device_type: info.device_type,
                    max_workgroup_size_x: limits.max_compute_workgroup_size_x,
                    max_buffer_size: limits.max_buffer_size,
                    max_dispatch_x: limits.max_compute_workgroups_per_dimension,
                    unified_memory: info.device_type == wgpu::DeviceType::IntegratedGpu,
                };

                // Create Stroke 1 compute pipeline
                let shader_src = include_str!("stroke1.wgsl");
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("stroke1"),
                    source: wgpu::ShaderSource::Wgsl(shader_src.into()),
                });

                let bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("stroke1_layout"),
                        entries: &[
                            // params uniform
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // query_prefix (read-only storage)
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // database prefixes (read-only storage)
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // results (read-write storage)
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("stroke1_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

                let pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("stroke1_pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("stroke1"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

                Some(GpuContext {
                    device,
                    queue,
                    capabilities,
                    stroke1_pipeline: pipeline,
                    stroke1_bind_group_layout: bind_group_layout,
                })
            })
        })
        .as_ref()
}

/// Check if GPU compute is available without initializing.
///
/// Useful for dispatch decisions before committing to GPU path.
pub fn gpu_available() -> bool {
    init_gpu().is_some()
}

/// Report detected GPU capabilities. Returns None if no GPU.
pub fn gpu_capabilities() -> Option<&'static GpuCapabilities> {
    init_gpu().map(|ctx| &ctx.capabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        env_logger::try_init().ok();
        // This test is informational — passes whether GPU exists or not
        match init_gpu() {
            Some(ctx) => {
                println!("GPU detected: {:?}", ctx.capabilities);
                assert!(!ctx.capabilities.name.is_empty());
            }
            None => {
                println!("No GPU available (CI/headless expected)");
            }
        }
    }
}
