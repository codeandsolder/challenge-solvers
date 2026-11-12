//! GPU adapter selection, pipeline compilation, bind groups and context initialization.
//!
//! Keep device initialization apart from the steady-state submission hot paths.
//! The public `GpuContext` type and all protocol-facing behavior remain in the parent module.

use super::shaders::{
    altcha_shader_source, cerberus_shader_source, generic_single_block_shader_source,
    multi_layout_shader_source, shader_source,
};
use super::{GpuContext, GpuError};
use core::{future::Future, pin::Pin};
use std::{borrow::Cow, string::String, vec::Vec};

struct GpuPipelines {
    anubis: wgpu::ComputePipeline,
    generic: wgpu::ComputePipeline,
    double: wgpu::ComputePipeline,
    goaway: wgpu::ComputePipeline,
    binary: wgpu::ComputePipeline,
    cerberus: wgpu::ComputePipeline,
    altcha: wgpu::ComputePipeline,
}

impl GpuPipelines {
    fn create(device: &wgpu::Device, majority_variant: bool) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-anubis"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(majority_shader_source(
                shader_source(),
                majority_variant,
            ))),
        });
        let anubis = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("challenge-solvers-anubis"),
            layout: None,
            module: &shader,
            entry_point: Some("solve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let generic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-sha256-single-block"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(majority_shader_source(
                generic_single_block_shader_source(),
                majority_variant,
            ))),
        });
        let generic = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("challenge-solvers-sha256-single-block"),
            layout: None,
            module: &generic_shader,
            entry_point: Some("solve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let multi_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-sha256-multi-layout"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(majority_shader_source(
                multi_layout_shader_source(),
                majority_variant,
            ))),
        });
        let create_multi = |label, mode| {
            let constants = [("MODE", mode)];
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &multi_shader,
                entry_point: Some("solve"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..wgpu::PipelineCompilationOptions::default()
                },
                cache: None,
            })
        };
        let double = create_multi("challenge-solvers-sha256-double-block", 1.0);
        let goaway = create_multi("challenge-solvers-sha256-goaway", 2.0);
        let binary = create_multi("challenge-solvers-sha256-binary", 3.0);

        let cerberus_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-cerberus"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(majority_shader_source(
                cerberus_shader_source(),
                majority_variant,
            ))),
        });
        let cerberus = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("challenge-solvers-cerberus"),
            layout: None,
            module: &cerberus_shader,
            entry_point: Some("solve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let altcha_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-altcha"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(majority_shader_source(
                altcha_shader_source(),
                majority_variant,
            ))),
        });
        let altcha = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("challenge-solvers-altcha"),
            layout: None,
            module: &altcha_shader,
            entry_point: Some("solve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            anubis,
            generic,
            double,
            goaway,
            binary,
            cerberus,
            altcha,
        }
    }
}

/// Select an algebraically equivalent SHA-256 majority expression while
/// preserving all other generated WGSL, including counter formatting and
/// protocol comparisons. This is only used by opt-in GPU experiments.
fn majority_shader_source(source: String, alternative: bool) -> String {
    if alternative {
        source.replace("((a&b)^(a&c)^(b&c))", "((a&b)|(c&(a|b)))")
    } else {
        source
    }
}

struct GpuBuffers {
    anubis_input: wgpu::Buffer,
    generic_input: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuBuffers {
    fn create(device: &wgpu::Device) -> Self {
        let anubis_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("challenge-solvers-anubis-input"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let generic_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("challenge-solvers-sha256-single-block-input"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("challenge-solvers-anubis-output"),
            size: 48,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("challenge-solvers-anubis-staging"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            anubis_input,
            generic_input,
            output,
            staging,
        }
    }
}

struct GpuBindings {
    anubis: wgpu::BindGroup,
    generic: wgpu::BindGroup,
    double: wgpu::BindGroup,
    goaway: wgpu::BindGroup,
    binary: wgpu::BindGroup,
    cerberus: wgpu::BindGroup,
    altcha: wgpu::BindGroup,
}

impl GpuBindings {
    fn create(device: &wgpu::Device, pipelines: &GpuPipelines, buffers: &GpuBuffers) -> Self {
        let create =
            |label: &'static str, pipeline: &wgpu::ComputePipeline, input: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffers.output.as_entire_binding(),
                        },
                    ],
                })
            };
        Self {
            anubis: create(
                "challenge-solvers-anubis",
                &pipelines.anubis,
                &buffers.anubis_input,
            ),
            generic: create(
                "challenge-solvers-sha256-single-block",
                &pipelines.generic,
                &buffers.generic_input,
            ),
            double: create(
                "challenge-solvers-sha256-double-block",
                &pipelines.double,
                &buffers.generic_input,
            ),
            goaway: create(
                "challenge-solvers-sha256-goaway",
                &pipelines.goaway,
                &buffers.generic_input,
            ),
            binary: create(
                "challenge-solvers-sha256-binary",
                &pipelines.binary,
                &buffers.generic_input,
            ),
            cerberus: create(
                "challenge-solvers-cerberus",
                &pipelines.cerberus,
                &buffers.generic_input,
            ),
            altcha: create(
                "challenge-solvers-altcha",
                &pipelines.altcha,
                &buffers.generic_input,
            ),
        }
    }
}

fn select_gpu_adapter(
    instance: &wgpu::Instance,
) -> core::pin::Pin<Box<dyn core::future::Future<Output = Result<wgpu::Adapter, GpuError>> + '_>> {
    Box::pin(async move {
        if let Ok(requested) = std::env::var("WGPU_ADAPTER_NAME") {
            let requested = requested.to_lowercase();
            let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;
            let available = adapters
                .iter()
                .map(|adapter| adapter.get_info().name)
                .collect::<Vec<_>>();
            return adapters
                .into_iter()
                .find(|adapter| {
                    let info = adapter.get_info();
                    info.device_type != wgpu::DeviceType::Cpu
                        && info.name.to_lowercase().contains(&requested)
                })
                .ok_or_else(|| {
                    GpuError::new(format!(
                        "WGPU_ADAPTER_NAME={requested:?} matched no GPU adapter; available: {}",
                        available.join(", ")
                    ))
                });
        }

        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..wgpu::RequestAdapterOptions::default()
            })
            .await
            .map_err(|e| GpuError::new(format!("request GPU adapter: {e}")))
    })
}

impl GpuContext {
    /// Enumerate usable non-CPU GPU adapters.
    pub fn adapter_names() -> Pin<Box<dyn Future<Output = Vec<String>>>> {
        Box::pin(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let mut names = instance
                .enumerate_adapters(wgpu::Backends::VULKAN)
                .await
                .into_iter()
                .filter_map(|adapter| {
                    let info = adapter.get_info();
                    (info.device_type != wgpu::DeviceType::Cpu).then_some(info.name)
                })
                .collect::<Vec<_>>();
            names.sort();
            names.dedup();
            names
        })
    }

    /// Create a GPU solver. `WGPU_ADAPTER_NAME` forces a matching adapter; otherwise
    /// the high-performance adapter selected by `wgpu` is used.
    ///
    /// # Errors
    ///
    /// Returns an error if no requested/usable adapter is available or the device cannot be created.
    pub fn create() -> Pin<Box<dyn Future<Output = Result<Self, GpuError>>>> {
        Self::create_with_timestamp_requirement(false)
    }

    /// Create a GPU context with hardware timestamp-query support for profiling.
    ///
    /// This is opt-in: regular solver contexts do not request extra device features.
    ///
    /// # Errors
    ///
    /// Returns an error if the selected adapter cannot provide timestamp queries.
    pub fn create_with_timestamps() -> Pin<Box<dyn Future<Output = Result<Self, GpuError>>>> {
        Self::create_with_timestamp_requirement(true)
    }

    /// Recompile all SHA-256 pipelines with an algebraically equivalent
    /// majority formula on the same GPU device and queue.
    /// Shared buffers are used sequentially, as with normal GPU solves.
    ///
    /// Production kernels and routing remain unchanged.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn with_alternative_majority(&self) -> Self {
        let pipelines = GpuPipelines::create(&self.device, true);
        let buffers = GpuBuffers {
            anubis_input: self.in_buf.clone(),
            generic_input: self.generic_in_buf.clone(),
            output: self.out_buf.clone(),
            staging: self.staging.clone(),
        };
        Self::from_resources(
            self.device.clone(),
            self.queue.clone(),
            pipelines,
            buffers,
            self.max_wgs,
            self.adapter.clone(),
        )
    }

    fn create_with_timestamp_requirement(
        timestamps: bool,
    ) -> Pin<Box<dyn Future<Output = Result<Self, GpuError>>>> {
        Box::pin(async move {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = select_gpu_adapter(&instance).await?;
            let info = adapter.get_info();
            let adapter_name = format!("{} ({:?}, {})", info.name, info.backend, info.driver);
            let limits = adapter.limits();
            let mut required_features = adapter.features() & wgpu::Features::SHADER_INT64;
            if timestamps {
                if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                    return Err(GpuError::new(format!(
                        "GPU adapter {adapter_name} does not support timestamp queries"
                    )));
                }
                required_features |= wgpu::Features::TIMESTAMP_QUERY;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_features,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    ..wgpu::DeviceDescriptor::default()
                })
                .await
                .map_err(|e| GpuError::new(format!("request GPU device: {e}")))?;

            let pipelines = GpuPipelines::create(&device, false);
            let buffers = GpuBuffers::create(&device);
            Ok(Self::from_resources(
                device,
                queue,
                pipelines,
                buffers,
                limits.max_compute_workgroups_per_dimension,
                adapter_name,
            ))
        })
    }

    fn from_resources(
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipelines: GpuPipelines,
        buffers: GpuBuffers,
        max_wgs: u32,
        adapter: String,
    ) -> Self {
        let bindings = GpuBindings::create(&device, &pipelines, &buffers);
        Self {
            device,
            queue,
            pipeline: pipelines.anubis,
            in_buf: buffers.anubis_input,
            generic_pipeline: pipelines.generic,
            double_pipeline: pipelines.double,
            goaway_pipeline: pipelines.goaway,
            binary_pipeline: pipelines.binary,
            cerberus_pipeline: pipelines.cerberus,
            altcha_pipeline: pipelines.altcha,
            linear_sha1_pipeline: std::sync::OnceLock::new(),
            sha1_decimal_pipeline: std::sync::OnceLock::new(),
            altcha_sha64_pipeline: std::sync::OnceLock::new(),
            generic_in_buf: buffers.generic_input,
            out_buf: buffers.output,
            staging: buffers.staging,
            bind_group: bindings.anubis,
            generic_bind_group: bindings.generic,
            double_bind_group: bindings.double,
            goaway_bind_group: bindings.goaway,
            binary_bind_group: bindings.binary,
            cerberus_bind_group: bindings.cerberus,
            altcha_bind_group: bindings.altcha,
            max_wgs,
            adapter,
        }
    }
}
