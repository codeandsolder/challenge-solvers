//! WebGPU/native GPU solver backend for challenge-solvers message families.
//!
//! The same Rust implementation targets native `wgpu` backends and browser WebGPU on `wasm32`.

use core::{fmt, future::Future, num::NonZeroU8, pin::Pin};
use std::{borrow::Cow, string::String, vec::Vec};

use bytemuck::{Pod, Zeroable};

mod scrypt;
pub use scrypt::{ScryptKdfGpu, ScryptRomixGpu};

use crate::{high_u32, low_u32};
use sha2::{Digest, Sha256};

const WG_SIZE: u32 = 64;
const STEPS: u32 = 128;
const HASHES_PER_WG: u64 = 64 * 128;
const CANCEL_INTERVAL: u32 = 16;
const COUNTER_SPACE: u64 = 1_u64 << 54;

fn gpu_u32(value: u64, field: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::new(format!("{field} exceeds the GPU u32 ABI")))
}

fn gpu_usize_u32(value: usize, field: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::new(format!("{field} exceeds the GPU u32 ABI")))
}

fn append_fmt(output: &mut String, args: fmt::Arguments<'_>) {
    if fmt::Write::write_fmt(output, args).is_err() {
        unreachable!("writing to String cannot fail");
    }
}

fn be_words_4(bytes: &[u8; 16]) -> [u32; 4] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn be_words_8(bytes: &[u8; 32]) -> [u32; 8] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn be_words_16(bytes: &[u8; 64]) -> [u32; 16] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn le_words_16(bytes: &[u8; 64]) -> [u32; 16] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

#[inline]
fn gpu_batch_target(expected_work: u64, quantum: u64) -> u64 {
    expected_work.saturating_mul(3).div_ceil(10).max(quantum)
}

/// Error returned by the GPU solver.
#[derive(Debug, Clone)]
pub struct GpuError(String);

impl GpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

/// Successful GPU proof-of-work solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSolution {
    /// Solver nonce.
    pub nonce: u64,
    /// Solver digest/output words.
    pub hash: [u32; 8],
    /// Candidate hashes submitted through the winning batch.
    ///
    /// Workgroups stop cooperatively after a hit, so this is an upper bound on hashes actually executed.
    pub dispatched_hashes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    words: [u32; 16],
}

/// Reusable GPU device/context shared by all GPU solver instances.
///
/// Native builds use the best backend selected by `wgpu`; wasm builds use browser WebGPU.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    in_buf: wgpu::Buffer,
    generic_pipeline: wgpu::ComputePipeline,
    double_pipeline: wgpu::ComputePipeline,
    goaway_pipeline: wgpu::ComputePipeline,
    binary_pipeline: wgpu::ComputePipeline,
    cerberus_pipeline: wgpu::ComputePipeline,
    altcha_pipeline: wgpu::ComputePipeline,
    generic_in_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generic_bind_group: wgpu::BindGroup,
    double_bind_group: wgpu::BindGroup,
    goaway_bind_group: wgpu::BindGroup,
    binary_bind_group: wgpu::BindGroup,
    cerberus_bind_group: wgpu::BindGroup,
    altcha_bind_group: wgpu::BindGroup,
    max_wgs: u32,
    adapter: String,
}

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
            let required_features = if timestamps {
                if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                    return Err(GpuError::new(format!(
                        "GPU adapter {adapter_name} does not support timestamp queries"
                    )));
                }
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            };
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

    /// Submit a small representative batch to force lazy browser/driver shader setup.
    ///
    /// Browser WebGPU implementations may defer substantial pipeline work until the first submit.
    /// Extensions can call this once during startup so the first real challenge does not pay that cost.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn warm_up(&self) -> Result<(), GpuError> {
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&crate::sha256::IV);
        params.words[10] = 19 * 8;
        params.words[11] = u32::MAX;
        params.words[12] = u32::MAX;
        params.words[13] = 8192; // Exercise the fast kernel in the warmup workgroup.
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.generic_in_buf, 0, &[0; 256]);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("challenge-solvers-anubis-warmup"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("challenge-solvers-anubis-warmup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(64, 1, 1);
        }
        for (label, pipeline, bind_group) in [
            (
                "challenge-solvers-sha256-single-block-warmup",
                &self.generic_pipeline,
                &self.generic_bind_group,
            ),
            (
                "challenge-solvers-sha256-double-block-warmup",
                &self.double_pipeline,
                &self.double_bind_group,
            ),
            (
                "challenge-solvers-sha256-goaway-warmup",
                &self.goaway_pipeline,
                &self.goaway_bind_group,
            ),
            (
                "challenge-solvers-sha256-binary-warmup",
                &self.binary_pipeline,
                &self.binary_bind_group,
            ),
            (
                "challenge-solvers-cerberus-warmup",
                &self.cerberus_pipeline,
                &self.cerberus_bind_group,
            ),
            (
                "challenge-solvers-altcha-warmup",
                &self.altcha_pipeline,
                &self.altcha_bind_group,
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let _ = self.read_result().await?;
        Ok(())
    }

    /// Human-readable adapter/backend description.
    #[must_use]
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Solve an Anubis SHA-256 challenge.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_anubis(
        &self,
        prefix: &[u8],
        difficulty: NonZeroU8,
    ) -> Result<GpuSolution, GpuError> {
        self.solve_anubis_with_limit(prefix, difficulty, COUNTER_SPACE)
            .await?
            .ok_or_else(|| GpuError::new("GPU search space exhausted"))
    }

    /// Solve an Anubis SHA-256 challenge while dispatching at most `max_hashes` candidates.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_anubis_with_limit(
        &self,
        prefix: &[u8],
        difficulty: NonZeroU8,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if !prefix.len().is_multiple_of(64) {
            return Err(GpuError::new(format!(
                "GPU Anubis solver requires a block-aligned prefix, got {} bytes",
                prefix.len()
            )));
        }
        if difficulty.get() > 16 {
            return Err(GpuError::new("Anubis GPU difficulty must be <= 16"));
        }
        let bitlen = prefix
            .len()
            .checked_add(19)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| GpuError::new("challenge is too large for the GPU kernel"))?;

        let mid = midstate(prefix);
        let mask = crate::compute_mask_zero_nibbles(difficulty);
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&mid);
        params.words[10] = bitlen;
        params.words[11] = high_u32(mask);
        params.words[12] = low_u32(mask);
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let expected = 16_u64.saturating_pow(u32::from(difficulty.get()));
        let desired = gpu_batch_target(expected, HASHES_PER_WG);
        let limit = max_hashes.min(COUNTER_SPACE);
        let mut base = 0_u64;

        while base < limit {
            let remaining = limit - base;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            params.words[8] = low_u32(base);
            params.words[9] = high_u32(base);
            params.words[13] = gpu_u32(batch, "fast SHA-256 batch size")?;
            self.queue
                .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("challenge-solvers-anubis-solve"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("challenge-solvers-anubis-solve"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = base + batch;
            if values[0] != 0 {
                let counter = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let nonce = counter_to_decimal_nonce(counter);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify(prefix, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid Anubis proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            base = dispatched;
        }

        Ok(None)
    }

    /// Solve a parsed Anubis challenge descriptor.
    #[cfg(feature = "anubis")]
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_anubis_descriptor(
        &self,
        descriptor: &crate::anubis::ChallengeDescriptor,
    ) -> Result<GpuSolution, GpuError> {
        if descriptor.rules().algorithm() == "preact" {
            return Err(GpuError::new(
                "preact challenges do not require GPU proof-of-work",
            ));
        }
        if !descriptor.supported() {
            return Err(GpuError::new("unsupported Anubis algorithm"));
        }
        let difficulty = NonZeroU8::new(descriptor.rules().difficulty())
            .ok_or_else(|| GpuError::new("Anubis difficulty must be non-zero"))?;
        self.solve_anubis(descriptor.challenge().as_ref().as_bytes(), difficulty)
            .await
    }

    async fn solve_single_block_inner<const TYPE: u8>(
        &self,
        message: &crate::message::SingleBlockMessage,
        target: u64,
        mask: u64,
        limit: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if !matches!(
            TYPE,
            crate::solver::SOLVE_TYPE_LT
                | crate::solver::SOLVE_TYPE_GT
                | crate::solver::SOLVE_TYPE_MASK
                | crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS
        ) {
            return Err(GpuError::new("unsupported solve type"));
        }

        if TYPE == crate::solver::SOLVE_TYPE_MASK
            && target == 0
            && single_block_fast_compatible(message)
        {
            return self
                .solve_fast_single_block_message(message, mask, limit)
                .await;
        }

        self.solve_generic_single_block::<TYPE>(message, target, mask, limit)
            .await
    }

    async fn solve_fast_single_block_message(
        &self,
        message: &crate::message::SingleBlockMessage,
        mask: u64,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let mut params = Params { words: [0; 16] };
        params.words[..8].copy_from_slice(&message.prefix_state);
        params.words[10] = message.message[15];
        params.words[11] = high_u32(mask);
        params.words[12] = if cfg!(feature = "compare-64bit") {
            low_u32(mask)
        } else {
            0
        };
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let desired = gpu_batch_target(
            expected_work::<{ crate::solver::SOLVE_TYPE_MASK }>(0, mask),
            HASHES_PER_WG,
        );
        let limit = max_hashes.min(COUNTER_SPACE);
        let mut base = 0_u64;

        while base < limit {
            let remaining = limit - base;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            params.words[8] = low_u32(base);
            params.words[9] = high_u32(base);
            params.words[13] = gpu_u32(batch, "fast SHA-256 batch size")?;
            self.queue
                .write_buffer(&self.in_buf, 0, bytemuck::bytes_of(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("challenge-solvers-sha256-fast-single-block"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("challenge-solvers-sha256-fast-single-block"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = base + batch;
            if values[0] != 0 {
                let counter = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let nonce = counter_to_decimal_nonce(counter);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify_fast_single_block(message, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid fast-path proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            base = dispatched;
        }

        Ok(None)
    }

    async fn solve_generic_single_block<const TYPE: u8>(
        &self,
        message: &crate::message::SingleBlockMessage,
        target: u64,
        mask: u64,
        max_hashes: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let search_space = single_block_search_space(message);
        let limit = max_hashes.min(search_space);
        if limit == 0 {
            return Ok(None);
        }

        let start = if message.nonce_addend == 0 {
            100_000_000_u64
        } else {
            0
        };
        let mut params = [0_u32; 40];
        params[..8].copy_from_slice(&message.prefix_state);
        params[8..24].copy_from_slice(&message.message);
        params[25] = gpu_usize_u32(message.digit_index, "digit index")?;
        params[26] = low_u32(message.nonce_addend);
        params[27] = high_u32(message.nonce_addend);
        params[28] = high_u32(target);
        params[29] = low_u32(target);
        params[30] = high_u32(mask);
        params[31] = low_u32(mask);
        params[32] = u32::from(TYPE);
        params[33] = u32::from(message.no_trailing_zeros)
            | (u32::from(cfg!(feature = "compare-64bit")) << 1);
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);

        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut offset = 0_u64;

        while offset < limit {
            let remaining = limit - offset;
            let batch = desired
                .min(remaining)
                .min(u64::from(self.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let candidate_start = start + offset;
            let candidate_end = start + offset + batch;
            params[24] = gpu_u32(candidate_start, "candidate start")?;
            params[34] = gpu_u32(candidate_end, "candidate end")?;
            self.queue
                .write_buffer(&self.generic_in_buf, 0, bytemuck::cast_slice(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("challenge-solvers-sha256-generic-single-block"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("challenge-solvers-sha256-generic-single-block"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.generic_pipeline);
                pass.set_bind_group(0, &self.generic_bind_group, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
            self.queue.submit([encoder.finish()]);

            let values = self.read_result().await?;
            let dispatched = offset.saturating_add(batch);
            if values[0] != 0 {
                let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
                let hash: [u32; 8] = values[3..11]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid GPU result layout"))?;
                if !verify_single_block::<TYPE>(message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid generic proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: dispatched,
                }));
            }
            offset = dispatched;
        }

        Ok(None)
    }

    async fn dispatch_multi(
        &self,
        params: &[u32; 64],
        workgroups: u32,
    ) -> Result<Option<(u64, [u32; 8])>, GpuError> {
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(&self.generic_in_buf, 0, bytemuck::cast_slice(params));
        let (label, pipeline, bind_group) = match params[0] {
            1 => (
                "challenge-solvers-sha256-double-block",
                &self.double_pipeline,
                &self.double_bind_group,
            ),
            2 => (
                "challenge-solvers-sha256-goaway",
                &self.goaway_pipeline,
                &self.goaway_bind_group,
            ),
            3 => (
                "challenge-solvers-sha256-binary",
                &self.binary_pipeline,
                &self.binary_bind_group,
            ),
            mode => {
                return Err(GpuError::new(format!(
                    "invalid SHA-256 GPU layout mode {mode}"
                )));
            }
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let values = self.read_result().await?;
        if values[0] == 0 {
            return Ok(None);
        }
        let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
        let hash = values[3..11]
            .try_into()
            .map_err(|_| GpuError::new("invalid GPU result layout"))?;
        Ok(Some((nonce, hash)))
    }

    async fn dispatch_cerberus(
        &self,
        params: &[u32; 64],
        workgroups: u32,
    ) -> Result<Option<(u64, [u32; 8])>, GpuError> {
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(&self.generic_in_buf, 0, bytemuck::cast_slice(params));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("challenge-solvers-cerberus"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("challenge-solvers-cerberus"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cerberus_pipeline);
            pass.set_bind_group(0, &self.cerberus_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let values = self.read_result().await?;
        if values[0] == 0 {
            return Ok(None);
        }
        let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
        let hash = values[3..11]
            .try_into()
            .map_err(|_| GpuError::new("invalid GPU Cerberus result layout"))?;
        Ok(Some((nonce, hash)))
    }

    async fn dispatch_altcha(
        &self,
        params: &[u32],
        workgroups: u32,
    ) -> Result<Option<(u64, [u32; 8])>, GpuError> {
        if params.len() < 64 {
            return Err(GpuError::new("incomplete ALTCHA GPU parameter buffer"));
        }
        let param_bytes = u64::try_from(params.len())
            .ok()
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| GpuError::new("ALTCHA GPU parameter buffer size overflow"))?;
        let limits = self.device.limits();
        if param_bytes > limits.max_storage_buffer_binding_size
            || param_bytes > limits.max_buffer_size
        {
            return Err(GpuError::new(
                "ALTCHA GPU parameter buffer exceeds device storage limits",
            ));
        }
        // Existing layouts use the resident 256-byte buffer. Arbitrary
        // PBKDF2 salts require their padded U1 blocks appended after it;
        // bind an exact-size storage input rather than silently truncating.
        let variable_buffer = (params.len() > 64).then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("challenge-solvers-altcha-variable-input"),
                size: param_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let input = variable_buffer.as_ref().unwrap_or(&self.generic_in_buf);
        self.queue.write_buffer(&self.out_buf, 0, &[0; 48]);
        self.queue
            .write_buffer(input, 0, bytemuck::cast_slice(params));
        let variable_binding = variable_buffer.as_ref().map(|buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("challenge-solvers-altcha-variable-bind-group"),
                layout: &self.altcha_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.out_buf.as_entire_binding(),
                    },
                ],
            })
        });
        let binding = variable_binding.as_ref().unwrap_or(&self.altcha_bind_group);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("challenge-solvers-altcha"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("challenge-solvers-altcha"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.altcha_pipeline);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, &self.staging, 0, 48);
        self.queue.submit([encoder.finish()]);
        let values = self.read_result().await?;
        if values[0] == 0 {
            return Ok(None);
        }
        let nonce = (u64::from(values[2]) << 32) | u64::from(values[1]);
        let hash = values[3..11]
            .try_into()
            .map_err(|_| GpuError::new("invalid GPU Altcha result layout"))?;
        Ok(Some((nonce, hash)))
    }

    async fn read_result(&self) -> Result<Vec<u32>, GpuError> {
        let slice = self.staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::new(format!("poll GPU: {e}")))?;

        rx.await
            .map_err(|_| GpuError::new("GPU map callback dropped"))?
            .map_err(|e| GpuError::new(format!("map GPU result: {e}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|e| GpuError::new(format!("read mapped GPU result: {e}")))?;
        let values = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        self.staging.unmap();
        Ok(values)
    }
}

/// Async GPU counterpart of the CPU single-block SHA-256 solver family.
///
/// Anubis, mCaptcha and Cap.js all route through this protocol-agnostic message type.
#[derive(Debug, Clone)]
pub struct SingleBlockSolver {
    message: crate::message::SingleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::SingleBlockMessage> for SingleBlockSolver {
    fn from(message: crate::message::SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl SingleBlockSolver {
    /// Set the maximum number of candidates dispatched by this solver.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    ///
    /// GPU workgroups cooperatively stop after a hit, so this is an upper bound on candidates
    /// actually executed by the device.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this SHA-256 single-block message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let result = gpu
            .solve_single_block_inner::<TYPE>(&self.message, target, mask, self.limit)
            .await?;
        self.attempted_nonces = result.as_ref().map_or_else(
            || self.limit.min(single_block_search_space(&self.message)),
            |solution| solution.dispatched_hashes.min(self.limit),
        );
        Ok(result)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU counterpart of the CPU double-block decimal SHA-256 solver.
pub struct DoubleBlockSolver {
    message: crate::message::DoubleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: crate::message::DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl DoubleBlockSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this double-block decimal message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let search_space = if self.message.nonce_addend == 0 {
            900_000_000_u64
        } else {
            1_000_000_000
        };
        let limit = self.limit.min(search_space);
        let start = if self.message.nonce_addend == 0 {
            100_000_000_u64
        } else {
            0
        };
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 1;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = gpu_u32(start + offset, "double-block candidate start")?;
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[16..24].copy_from_slice(&self.message.prefix_state);
            p[24..40].copy_from_slice(&self.message.message);
            let mut hot_state = self.message.prefix_state;
            let mut hot_prefix = [0_u32; 13];
            hot_prefix.copy_from_slice(&self.message.message[..13]);
            crate::sha256::ingest_message_prefix::<13>(&mut hot_state, hot_prefix);
            p[44..52].copy_from_slice(&hot_state);
            p[40] = low_u32(self.message.nonce_addend);
            p[41] = high_u32(self.message.nonce_addend);
            let bitlen = self.message.message_length.saturating_mul(8);
            p[42] = high_u32(bitlen);
            p[43] = low_u32(bitlen);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_double_block::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid double-block proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// GPU router for decimal SHA-256 messages, mirroring [`crate::message::DecimalMessage`].
pub enum DecimalSolver {
    /// Single-block message.
    SingleBlock(SingleBlockSolver),
    /// Double-block message.
    DoubleBlock(DoubleBlockSolver),
}

impl From<crate::message::DecimalMessage> for DecimalSolver {
    fn from(message: crate::message::DecimalMessage) -> Self {
        match message {
            crate::message::DecimalMessage::SingleBlock(m) => Self::SingleBlock(m.into()),
            crate::message::DecimalMessage::DoubleBlock(m) => Self::DoubleBlock(m.into()),
        }
    }
}

impl DecimalSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        match self {
            Self::SingleBlock(s) => s.set_limit(limit),
            Self::DoubleBlock(s) => s.set_limit(limit),
        }
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        match self {
            Self::SingleBlock(s) => s.attempted_nonces(),
            Self::DoubleBlock(s) => s.attempted_nonces(),
        }
    }
    /// Solve a decimal SHA-256 message.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        match self {
            Self::SingleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
            Self::DoubleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
        }
    }
}

/// Async GPU `GoAway` SHA-256 solver.
pub struct GoAwaySolver {
    message: crate::message::GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::GoAwayMessage> for GoAwaySolver {
    fn from(message: crate::message::GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a `GoAway` message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let limit = self.limit.min(u64::from(u32::MAX) + 1);
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 2;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = gpu_u32(offset, "candidate offset")?;
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[16..24].copy_from_slice(&self.message.challenge);
            p[24] = self.message.high_word;
            let mut hot_state = crate::sha256::IV;
            let mut invariant = [0_u32; 9];
            invariant[..8].copy_from_slice(&self.message.challenge);
            invariant[8] = self.message.high_word;
            crate::sha256::ingest_message_prefix::<9>(&mut hot_state, invariant);
            p[40..48].copy_from_slice(&hot_state);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_goaway::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid GoAway proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU solver for binary SHA-256 nonce messages.
pub struct BinarySolver {
    message: crate::message::BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
}
impl From<crate::message::BinaryMessage> for BinarySolver {
    fn from(message: crate::message::BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}
impl BinarySolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a binary-nonce SHA-256 message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let bits = u32::from(self.message.nonce_byte_count.get()) * 8;
        let space = if bits == 64 { u64::MAX } else { 1_u64 << bits };
        let limit = self.limit.min(space);
        let (templates, used) = binary_templates(&self.message);
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut base = 0_u64;
        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 3;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = low_u32(base);
            p[4] = high_u32(base);
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[10] = used;
            p[11] = u32::from(self.message.nonce_byte_count.get());
            p[12] = gpu_usize_u32(self.message.salt_residual_len, "salt residual length")?;
            p[16..24].copy_from_slice(&self.message.prefix_state);
            p[24..40].copy_from_slice(&templates[0]);
            p[40..56].copy_from_slice(&templates[1]);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = base + batch;
                if !verify_binary::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid binary proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU counterpart of the CPU Cerberus BLAKE3 solver.
pub struct CerberusSolver {
    message: crate::message::CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::CerberusMessage> for CerberusSolver {
    fn from(message: crate::message::CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl CerberusSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve a Cerberus message on the GPU.
    ///
    /// Cerberus uses a zero target plus bit mask; this mirrors the CPU backend.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if target != 0 {
            return Err(GpuError::new("Cerberus GPU solver requires target == 0"));
        }
        let _ = TYPE;
        let expected = 1_u64.checked_shl(mask.count_ones()).unwrap_or(u64::MAX);
        let desired = gpu_batch_target(expected, HASHES_PER_WG);
        let search_space = match &self.message {
            crate::message::CerberusMessage::Decimal(_) => 1_000_000_000_u64,
            crate::message::CerberusMessage::Binary(_) => u64::from(u32::MAX) + 1,
        };
        let limit = self.limit.min(search_space);
        let mut base = 0_u64;

        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG)
                .min(u64::from(u32::MAX));
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[1] = low_u32(base);
            p[2] = gpu_u32(batch, "batch size")?;
            p[3] = high_u32(mask);
            p[4] = low_u32(mask);

            match &self.message {
                crate::message::CerberusMessage::Decimal(message) => {
                    p[0] = 0;
                    p[8..16].copy_from_slice(&message.prefix_state);
                    for (i, chunk) in message.salt_residual.as_chunks::<4>().0.iter().enumerate() {
                        p[16 + i] = u32::from_le_bytes(*chunk);
                    }
                    p[32] = gpu_usize_u32(message.salt_residual_len, "salt residual length")?;
                    p[33] = gpu_usize_u32(message.salt_residual_len, "salt residual length")? + 9;
                    p[34] = message.flags;
                    p[35] = low_u32(message.nonce_addend);
                    p[36] = high_u32(message.nonce_addend);
                }
                crate::message::CerberusMessage::Binary(message) => {
                    p[0] = 1;
                    p[8..16].copy_from_slice(&message.midstate);
                    p[16] = message.first_word;
                }
            }

            if let Some((nonce, hash)) = gpu.dispatch_cerberus(&p, wgs).await? {
                self.attempted_nonces = base + batch;
                if !verify_cerberus(&self.message, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid Cerberus proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|solution| solution.nonce))
    }
}

/// Reusable midstate and at-most-one partial compression block for
/// SHA-256(salt || nonce || variable-counter). Entire fixed prefixes are
/// compressed once on CPU; each GPU candidate completes at most two blocks.
struct AltchaShaPrefix {
    fixed_bytes: Vec<u8>,
    state: [u32; 8],
    tail: [u8; 64],
    tail_len: usize,
    fixed_len: u64,
}

impl AltchaShaPrefix {
    fn new(fixed_bytes: Vec<u8>) -> Result<Self, GpuError> {
        let fixed_len = u64::try_from(fixed_bytes.len())
            .map_err(|_| GpuError::new("ALTCHA SHA-256 prefix length does not fit in u64"))?;
        // SHA-256's 64-bit bit length must also accommodate a maximum
        // 16-digit decimal counter, without overflow.
        if fixed_len > (u64::MAX / 8) - 16 {
            return Err(GpuError::new(
                "ALTCHA SHA-256 message exceeds 64-bit bit length",
            ));
        }
        let mut state = crate::sha256::IV;
        let (chunks, remainder) = fixed_bytes.as_chunks::<64>();
        for chunk in chunks {
            let words: [u32; 16] = core::array::from_fn(|i| {
                u32::from_be_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ])
            });
            crate::sha256::digest_block(&mut state, &words);
        }
        let tail_len = remainder.len();
        let mut tail = [0_u8; 64];
        tail[..tail_len].copy_from_slice(remainder);
        Ok(Self {
            fixed_bytes,
            state,
            tail,
            tail_len,
            fixed_len,
        })
    }
}

/// Variable-length PBKDF2-HMAC-SHA256 message representation.
///
/// The nonce is the fixed part of the candidate-dependent HMAC key. For
/// passwords longer than a SHA-256 block, its prefix state is reused by
/// the GPU to hash the complete nonce+counter key before applying HMAC.
///
/// The salt's RFC 8018 block index and SHA-256 padding are prepared once.
/// Unlike its HMAC chaining state, these message blocks are independent of
/// the candidate password and may be shared by every GPU invocation.
struct AltchaPbkdf2Input {
    salt: Vec<u8>,
    key_prefix: AltchaShaPrefix,
    u1_blocks: Vec<u32>,
}

impl AltchaPbkdf2Input {
    fn new(salt: &[u8], nonce: &[u8]) -> Result<Self, GpuError> {
        let key_prefix = AltchaShaPrefix::new(nonce.to_vec())?;
        let salt_len = salt
            .len()
            .checked_add(4)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 salt length overflow"))?;
        let padded_len = salt_len
            .checked_add(9)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 padding length overflow"))?
            .div_ceil(64)
            .checked_mul(64)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 padded size overflow"))?;
        let hmac_msg_len = u64::try_from(salt_len)
            .map_err(|_| GpuError::new("ALTCHA PBKDF2 salt length exceeds u64"))?
            .checked_add(64)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 inner length overflow"))?;
        let bit_len = hmac_msg_len
            .checked_mul(8)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 exceeds SHA-256 length limit"))?;
        let mut padded = vec![0_u8; padded_len];
        padded[..salt.len()].copy_from_slice(salt);
        padded[salt.len() + 3] = 1; // PBKDF2 block index INT(1), BE32.
        padded[salt_len] = 0x80;
        padded[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());
        let (words, _) = padded.as_chunks::<4>();
        let u1_blocks = words.iter().map(|word| u32::from_be_bytes(*word)).collect();
        Ok(Self {
            salt: salt.to_vec(),
            key_prefix,
            u1_blocks,
        })
    }
}

/// Async GPU counterpart of the CPU Altcha SHA-256/PBKDF2 solver.
pub struct AltchaSha256Solver {
    message: crate::message::AltchaMessage,
    attempted_nonces: u64,
    start_nonce: u64,
    limit: u64,
    string_counter: bool,
    prefix_nibbles: Vec<u8>,
    sha_prefix: Option<AltchaShaPrefix>,
    pbkdf2_input: Option<AltchaPbkdf2Input>,
    output_key_length: usize,
    verified_key: Option<Vec<u8>>,
}

impl From<crate::message::AltchaMessage> for AltchaSha256Solver {
    fn from(message: crate::message::AltchaMessage) -> Self {
        let output_key_length = message.key_length.get() as usize;
        Self {
            message,
            attempted_nonces: 0,
            start_nonce: 0,
            limit: u64::MAX,
            string_counter: false,
            prefix_nibbles: Vec::new(),
            sha_prefix: None,
            pbkdf2_input: None,
            output_key_length,
            verified_key: None,
        }
    }
}

impl AltchaSha256Solver {
    /// Construct a SHA-256 ALTCHA solver for arbitrary-length binary
    /// salt/nonce data. Pre-hash complete blocks on CPU once; the GPU
    /// performs only the counter-dependent SHA-256 compression.
    ///
    /// # Errors
    ///
    /// Rejects message lengths beyond SHA-256's 64-bit length field.
    pub fn for_sha256_prefix(
        salt: &[u8],
        nonce: &[u8],
        cost: core::num::NonZeroU32,
    ) -> Result<Self, GpuError> {
        let length = salt
            .len()
            .checked_add(nonce.len())
            .ok_or_else(|| GpuError::new("ALTCHA SHA-256 salt and nonce length overflow"))?;
        let mut fixed = Vec::with_capacity(length);
        fixed.extend_from_slice(salt);
        fixed.extend_from_slice(nonce);
        let sha_prefix = AltchaShaPrefix::new(fixed)?;
        // The fixed CPU message struct is retained for existing PBKDF2
        // layouts. The SHA path below uses the general prefix exclusively.
        let message = crate::message::AltchaMessage {
            nonce: [0; 16],
            salt: [0; 16],
            cost,
            pbkdf2: false,
            key_length: core::num::NonZeroU32::new(32)
                .ok_or_else(|| GpuError::new("invalid SHA-256 output length"))?,
        };
        let mut solver = Self::from(message);
        solver.sha_prefix = Some(sha_prefix);
        Ok(solver)
    }

    /// Construct a PBKDF2-HMAC-SHA256 search over arbitrary-length binary
    /// salt and nonce. GPU candidates derive their own HMAC keys (including
    /// SHA-256 hashing of passwords exceeding 64 bytes), process all U1 salt
    /// blocks and run the remaining PBKDF2 iterations on-device.
    ///
    /// # Errors
    ///
    /// Rejects input lengths incompatible with SHA-256 bit-length encoding.
    pub fn for_pbkdf2(
        salt: &[u8],
        nonce: &[u8],
        cost: core::num::NonZeroU32,
    ) -> Result<Self, GpuError> {
        let input = AltchaPbkdf2Input::new(salt, nonce)?;
        let message = crate::message::AltchaMessage {
            nonce: [0; 16],
            salt: [0; 16],
            cost,
            pbkdf2: true,
            key_length: core::num::NonZeroU32::new(32)
                .ok_or_else(|| GpuError::new("invalid PBKDF2 output length"))?,
        };
        let mut solver = Self::from(message);
        solver.pbkdf2_input = Some(input);
        Ok(solver)
    }

    /// Request any PBKDF2-SHA256 derived key size allowed by RFC 8018.
    /// The first 32 bytes are computed for each candidate; additional
    /// PBKDF2 blocks are computed on GPU only after earlier prefix blocks
    /// match. The complete derived key is independently computed on CPU
    /// only for the winning candidate.
    ///
    /// # Errors
    ///
    /// Rejects zero length, non-PBKDF2 inputs, and lengths beyond RFC 8018.
    pub fn set_pbkdf2_key_length(&mut self, length: usize) -> Result<(), GpuError> {
        let max_len = u64::from(u32::MAX) * 32;
        if !self.message.pbkdf2 || length == 0 || !u64::try_from(length).is_ok_and(|n| n <= max_len)
        {
            return Err(GpuError::new("invalid PBKDF2-SHA256 derived key length"));
        }
        self.output_key_length = length;
        self.verified_key = None;
        Ok(())
    }

    /// The complete independently validated answer from the last successful
    /// PBKDF2 GPU solve, rather than just the first SHA-256 output block.
    #[must_use]
    pub fn verified_key(&self) -> Option<&[u8]> {
        self.verified_key.as_deref()
    }

    /// Require an exact hexadecimal prefix of the derived output, including
    /// half-byte prefixes. SHA-256 accepts up to 64 nibbles; the general
    /// PBKDF2 solver also accepts prefixes spanning multiple digest blocks.
    ///
    /// The input is nibble values (0..=15), rather than ASCII hex digits.
    ///
    /// # Errors
    ///
    /// Rejects malformed nibbles and multi-block prefixes on unsupported inputs.
    pub fn set_prefix_nibbles(&mut self, nibbles: &[u8]) -> Result<(), GpuError> {
        if (nibbles.len() > 64 && self.pbkdf2_input.is_none())
            || nibbles.iter().any(|&nibble| nibble > 15)
        {
            return Err(GpuError::new(
                "ALTCHA GPU prefix requires valid hex nibbles and a multi-block PBKDF2 input beyond 64",
            ));
        }
        self.prefix_nibbles.clear();
        self.prefix_nibbles.extend_from_slice(nibbles);
        Ok(())
    }

    /// Use ALTCHA v2 decimal-string counter encoding rather than binary u32.
    ///
    /// Covers the complete JavaScript-safe integer counter domain (0..=2^53-1).
    pub const fn set_string_counter(&mut self) {
        self.string_counter = true;
    }

    /// Set the inclusive search starting counter, allowing disjoint GPU
    /// partitions and exact tests across the u32 rollover.
    pub const fn set_start_nonce(&mut self, start: u64) {
        self.start_nonce = start;
    }

    /// Set the exclusive upper counter bound (up to the encoding's domain).
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve an Altcha SHA-256 or PBKDF2 challenge on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let counter_limit = if self.string_counter {
            1_u64 << 53
        } else {
            u64::from(u32::MAX) + 1
        };
        let limit = self.limit.min(counter_limit);
        self.verified_key = None;
        let desired = gpu_batch_target(
            expected_work_full64::<TYPE>(target, mask),
            u64::from(WG_SIZE),
        );
        let max_batch = u64::from(gpu.max_wgs) * u64::from(WG_SIZE);
        let mut base = self.start_nonce;
        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(max_batch)
                .min(u64::from(u32::MAX));
            let wgs = gpu_u32(batch.div_ceil(u64::from(WG_SIZE)).max(1), "workgroup count")?;
            let mut p = vec![0_u32; 64];
            if let Some(input) = &self.pbkdf2_input {
                p.extend_from_slice(&input.u1_blocks);
            }
            p[0] = u32::from(self.message.pbkdf2);
            p[1] = low_u32(base);
            p[2] = gpu_u32(batch, "batch size")?;
            p[10] = high_u32(base);
            p[11] = u32::from(self.string_counter);
            p[3] = u32::from(TYPE);
            p[4] = high_u32(target);
            p[5] = low_u32(target);
            p[6] = high_u32(mask);
            p[7] = low_u32(mask);
            p[8] = self.message.cost.get();
            p[9] = self.message.key_length.get().min(32);
            p[12] = gpu_usize_u32(self.prefix_nibbles.len(), "ALTCHA prefix length")?;
            // Compare the first SHA-256 digest block for all candidates.
            // Additional PBKDF2 blocks are needed only when all 64 initial
            // nibbles matched; store their expected values after U1 blocks.
            for (index, nibble) in self.prefix_nibbles.iter().take(64).enumerate() {
                p[32 + index / 8] |= u32::from(*nibble) << ((7 - index % 8) * 4);
            }
            p[16..20].copy_from_slice(&be_words_4(&self.message.salt));
            p[20..24].copy_from_slice(&be_words_4(&self.message.nonce));
            if let Some(prefix) = self
                .sha_prefix
                .as_ref()
                .or_else(|| self.pbkdf2_input.as_ref().map(|input| &input.key_prefix))
            {
                p[24] = if self.pbkdf2_input.is_some() { 2 } else { 1 };
                p[13] = low_u32(prefix.fixed_len);
                p[14] = high_u32(prefix.fixed_len);
                p[15] = gpu_usize_u32(prefix.tail_len, "SHA-256 partial block length")?;
                p[40..48].copy_from_slice(&prefix.state);
                p[48..64].copy_from_slice(&be_words_16(&prefix.tail));
            }
            if let Some(input) = &self.pbkdf2_input {
                p[25] = gpu_usize_u32(input.u1_blocks.len() / 16, "PBKDF2 salt block count")?;
                p[26] = gpu_usize_u32(input.salt.len(), "PBKDF2 salt length")?;
            }
            if self.prefix_nibbles.len() > 64 {
                if self.pbkdf2_input.is_none() {
                    return Err(GpuError::new(
                        "multi-block prefixes require general PBKDF2 input",
                    ));
                }
                p[28] = gpu_usize_u32(p.len(), "PBKDF2 prefix buffer offset")?;
                let suffix_nibbles = &self.prefix_nibbles[64..];
                let suffix_words = suffix_nibbles.len().div_ceil(8);
                let start = p.len();
                p.resize(start + suffix_words, 0);
                for (index, nibble) in suffix_nibbles.iter().enumerate() {
                    p[start + index / 8] |= u32::from(*nibble) << ((7 - index % 8) * 4);
                }
            }

            if let Some((nonce, hash)) = gpu.dispatch_altcha(&p, wgs).await? {
                self.attempted_nonces = base + batch - self.start_nonce;
                let Some(verified_key) = verify_altcha::<TYPE>(self, nonce, hash, target, mask)
                else {
                    return Err(GpuError::new("GPU returned an invalid Altcha proof"));
                };
                self.verified_key = Some(verified_key);
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit.saturating_sub(self.start_nonce);
        Ok(None)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|solution| solution.nonce))
    }
}

const fn single_block_search_space(message: &crate::message::SingleBlockMessage) -> u64 {
    if message.nonce_addend == 0 {
        900_000_000
    } else {
        1_000_000_000
    }
}

const fn single_block_fast_compatible(message: &crate::message::SingleBlockMessage) -> bool {
    if message.digit_index != 10
        || message.nonce_addend != 1_000_000_000_000_000_000
        || message.no_trailing_zeros
    {
        return false;
    }
    let words = &message.message;
    words[0] == 0x3130_3030
        && words[1] == 0x3030_3030
        && (words[2] & 0xffff_0000) == 0x3030_0000
        && words[4] == 0x0000_0080
}

fn expected_work<const TYPE: u8>(target: u64, mask: u64) -> u64 {
    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
        return u32::try_from(target)
            .ok()
            .and_then(|bits| 1_u64.checked_shl(bits))
            .unwrap_or(u64::MAX);
    }
    let compare_mask = if cfg!(feature = "compare-64bit") {
        mask
    } else {
        mask & 0xffff_ffff_0000_0000
    };
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        return 1_u64
            .checked_shl(compare_mask.count_ones())
            .unwrap_or(u64::MAX);
    }
    let bits = if cfg!(feature = "compare-64bit") {
        64
    } else {
        32
    };
    let t = if bits == 64 {
        u128::from(target)
    } else {
        u128::from(target >> 32)
    };
    let space = 1_u128 << bits;
    let success = if TYPE == crate::solver::SOLVE_TYPE_LT {
        t.max(1)
    } else {
        space.saturating_sub(t).saturating_sub(1).max(1)
    };
    u64::try_from((space / success).min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
}

fn midstate(prefix: &[u8]) -> [u32; 8] {
    let mut state = crate::sha256::IV;
    for block in prefix.as_chunks::<64>().0 {
        let words = be_words_16(block);
        crate::sha256::digest_block(&mut state, &words);
    }
    state
}

fn counter_to_decimal_nonce(counter: u64) -> u64 {
    let mut nonce = 1_u64;
    for digit in (0..18).rev() {
        nonce = nonce * 10 + ((counter >> (digit * 3)) & 7);
    }
    nonce
}

fn verify(prefix: &[u8], nonce: u64, hash: [u32; 8], mask: u64) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(nonce.to_string().as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let expected = be_words_8(&digest);
    expected == hash && (((u64::from(hash[0]) << 32) | u64::from(hash[1])) & mask) == 0
}

fn digest_single_block_message(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
) -> Option<[u32; 8]> {
    let suffix = nonce.checked_sub(message.nonce_addend)?;
    if suffix >= 1_000_000_000 {
        return None;
    }
    let mut words = message.message;
    let mut x = u32::try_from(suffix).ok()?;
    for j in 0..9 {
        let digit = x % 10;
        x /= 10;
        let pos = message.digit_index + 8 - j;
        let word = pos / 4;
        let shift = (3 - (pos % 4)) * 8;
        words[word] = (words[word] & !(0xff_u32 << shift)) | ((u32::from(b'0') + digit) << shift);
    }
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    Some(state)
}

fn result_matches<const TYPE: u8>(hash: [u32; 8], target: u64, mask: u64) -> bool {
    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
        return crate::hash_has_leading_zero_bits(&hash, target);
    }
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    if cfg!(feature = "compare-64bit") {
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    } else {
        let value = hash[0];
        let target = high_u32(target);
        let mask = high_u32(mask);
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    }
}

fn verify_single_block<const TYPE: u8>(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    digest_single_block_message(message, nonce)
        .is_some_and(|expected| expected == hash && result_matches::<TYPE>(hash, target, mask))
}

fn verify_fast_single_block(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    mask: u64,
) -> bool {
    let nonce = nonce.to_string();
    if nonce.len() != 19 {
        return false;
    }
    let mut block = [0_u8; 64];
    block[..19].copy_from_slice(nonce.as_bytes());
    block[19] = 0x80;
    block[60..64].copy_from_slice(&message.message[15].to_be_bytes());
    let words = be_words_16(&block);
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    state == hash && result_matches::<{ crate::solver::SOLVE_TYPE_MASK }>(hash, 0, mask)
}

fn verify_cerberus(
    message: &crate::message::CerberusMessage,
    nonce: u64,
    hash: [u32; 8],
    mask: u64,
) -> bool {
    let expected = match message {
        crate::message::CerberusMessage::Decimal(message) => {
            let Some(mut suffix) = nonce.checked_sub(message.nonce_addend) else {
                return false;
            };
            if suffix >= 1_000_000_000 {
                return false;
            }
            let mut bytes = message.salt_residual;
            for i in (0..9).rev() {
                bytes[message.salt_residual_len + i] = (suffix % 10).to_le_bytes()[0] + b'0';
                suffix /= 10;
            }
            let words = le_words_16(&bytes);
            let Ok(salt_residual_len) = u32::try_from(message.salt_residual_len) else {
                return false;
            };
            crate::blake3::compress8(
                &message.prefix_state,
                &words,
                0,
                salt_residual_len + 9,
                message.flags,
            )
        }
        crate::message::CerberusMessage::Binary(message) => {
            if high_u32(nonce) != message.first_word {
                return false;
            }
            let mut words = [0_u32; 16];
            words[0] = message.first_word;
            words[1] = low_u32(nonce);
            crate::blake3::compress8(
                &message.midstate,
                &words,
                0,
                8,
                crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
            )
        }
    };
    expected == hash && (((u64::from(hash[0]) << 32) | u64::from(hash[1])) & mask) == 0
}

fn expected_work_full64<const TYPE: u8>(target: u64, mask: u64) -> u64 {
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        return 1_u64.checked_shl(mask.count_ones()).unwrap_or(u64::MAX);
    }
    let space = 1_u128 << 64;
    let t = u128::from(target);
    let success = if TYPE == crate::solver::SOLVE_TYPE_LT {
        t.max(1)
    } else {
        space.saturating_sub(t).saturating_sub(1).max(1)
    };
    u64::try_from((space / success).min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
}

fn result_matches_full64<const TYPE: u8>(hash: [u32; 8], target: u64, mask: u64) -> bool {
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    if TYPE == crate::solver::SOLVE_TYPE_LT {
        value < target
    } else if TYPE == crate::solver::SOLVE_TYPE_GT {
        value > target
    } else {
        (value & mask) == (target & mask)
    }
}

fn verify_altcha<const TYPE: u8>(
    solver: &AltchaSha256Solver,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> Option<Vec<u8>> {
    let message = &solver.message;
    let prefix_nibbles = &solver.prefix_nibbles;
    if (!solver.string_counter && nonce > u64::from(u32::MAX))
        || (solver.string_counter && nonce >= (1_u64 << 53))
        || prefix_nibbles.len() > solver.output_key_length.saturating_mul(2)
    {
        return None;
    }
    let suffix = if solver.string_counter {
        nonce.to_string().into_bytes()
    } else {
        nonce.to_be_bytes()[4..].to_vec()
    };
    let mut derived_key = if message.pbkdf2 {
        let input = solver.pbkdf2_input.as_ref();
        let mut password = Vec::with_capacity(
            input.map_or(16, |value| value.key_prefix.fixed_bytes.len()) + suffix.len(),
        );
        if let Some(value) = input {
            password.extend_from_slice(&value.key_prefix.fixed_bytes);
        } else {
            password.extend_from_slice(&message.nonce);
        }
        password.extend_from_slice(&suffix);
        let salt = input.map_or(message.salt.as_slice(), |value| value.salt.as_slice());
        let mut output = Vec::new();
        let len = solver.output_key_length.max(32);
        output.try_reserve_exact(len).ok()?;
        output.resize(len, 0);
        pbkdf2::pbkdf2::<pbkdf2::hmac::Hmac<Sha256>>(
            &password,
            salt,
            message.cost.get(),
            &mut output,
        )
        .ok()?;
        output
    } else {
        let mut first = Vec::with_capacity(
            solver
                .sha_prefix
                .as_ref()
                .map_or(32, |prefix| prefix.fixed_bytes.len())
                + suffix.len(),
        );
        if let Some(prefix) = &solver.sha_prefix {
            first.extend_from_slice(&prefix.fixed_bytes);
        } else {
            first.extend_from_slice(&message.salt);
            first.extend_from_slice(&message.nonce);
        }
        first.extend_from_slice(&suffix);
        let mut digest: [u8; 32] = Sha256::digest(&first).into();
        // ALTCHA v2 truncates only the final digest, after all iterations.
        for _ in 1..message.cost.get() {
            digest = Sha256::digest(digest).into();
        }
        digest.to_vec()
    };
    let first_digest: [u8; 32] = derived_key[..32].try_into().ok()?;
    let first_hash = be_words_8(&first_digest);
    let prefix_matches = prefix_nibbles.iter().enumerate().all(|(index, expected)| {
        let byte = derived_key[index / 2];
        let actual = if index.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0x0f
        };
        actual == *expected
    });
    let condition = if TYPE == crate::solver::SOLVE_TYPE_MASK && !prefix_nibbles.is_empty() {
        true // Full digest prefix supersedes the 64-bit target/mask.
    } else {
        result_matches_full64::<TYPE>(hash, target, mask)
    };
    if first_hash != hash || !prefix_matches || !condition {
        return None;
    }
    derived_key.truncate(solver.output_key_length);
    Some(derived_key)
}

fn sha256_core_shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    include_str!("shaders/sha256_core.wgsl").replace("__SHA256_ROUNDS__", &rounds)
}

fn altcha_shader_source() -> String {
    include_str!("shaders/altcha.wgsl")
        .replace("__SHA256_CORE__", &sha256_core_shader_source())
        .replace("__WORKGROUP_SIZE__", &WG_SIZE.to_string())
}

fn cerberus_shader_source() -> String {
    const PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
    let mut schedule = [[0_usize; 16]; 7];
    schedule[0] = core::array::from_fn(|i| i);
    for round in 1..7 {
        schedule[round] = core::array::from_fn(|i| schedule[round - 1][PERM[i]]);
    }
    let mut rounds = String::new();
    let gs = [
        (0, 4, 8, 12, 0, 1),
        (1, 5, 9, 13, 2, 3),
        (2, 6, 10, 14, 4, 5),
        (3, 7, 11, 15, 6, 7),
        (0, 5, 10, 15, 8, 9),
        (1, 6, 11, 12, 10, 11),
        (2, 7, 8, 13, 12, 13),
        (3, 4, 9, 14, 14, 15),
    ];
    for sched in &schedule {
        for &(a, b, c, d, x, y) in &gs {
            let mx = sched[x];
            let my = sched[y];
            append_fmt(
                &mut rounds,
                format_args!(
                    "v[{a}]=v[{a}]+v[{b}]+m[{mx}];v[{d}]=rotr(v[{d}]^v[{a}],16u);v[{c}]=v[{c}]+v[{d}];v[{b}]=rotr(v[{b}]^v[{c}],12u);v[{a}]=v[{a}]+v[{b}]+m[{my}];v[{d}]=rotr(v[{d}]^v[{a}],8u);v[{c}]=v[{c}]+v[{d}];v[{b}]=rotr(v[{b}]^v[{c}],7u);\n"
                ),
            );
        }
    }
    format!(
        r"
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn compress(cv:array<u32,8>,m:array<u32,16>,blen:u32,flags:u32)->array<u32,8>{{
 var v=array<u32,16>(cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7],0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0u,0u,blen,flags);
 {rounds}
 return array<u32,8>(v[0]^v[8],v[1]^v[9],v[2]^v[10],v[3]^v[11],v[4]^v[12],v[5]^v[13],v[6]^v[14],v[7]^v[15]);
}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=gid.x*{STEPS}u;
 var decimal_m=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31]);
 if(P[0]==0u&&first<P[2]){{
  var x=P[1]+first;for(var j=0u;j<9u;j=j+1u){{let digit=x%10u;x=x/10u;let pos=P[32]+8u-j;let wi=pos>>2u;let sh=(pos&3u)*8u;decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);}}
 }}
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let idx=first+step;if(idx>=P[2]){{return;}}let candidate=P[1]+idx;
  let cv=array<u32,8>(P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15]);
  var m:array<u32,16>;var nlo=candidate;var nhi=0u;var h:array<u32,8>;
  if(P[0]==0u){{
   m=decimal_m;h=compress(cv,m,P[33],P[34]);nlo=P[35]+candidate;let carry=select(0u,1u,nlo<P[35]);nhi=P[36]+carry;
   if(step+1u<{STEPS}u&&idx+1u<P[2]){{var j=0u;loop{{let pos=P[32]+8u-j;let wi=pos>>2u;let sh=(pos&3u)*8u;let digit=(decimal_m[wi]>>sh)&0xffu;if(digit<0x39u){{decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}decimal_m[wi]=(decimal_m[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}}}}}
  }}else{{
   m=array<u32,16>(P[16],candidate,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);h=compress(cv,m,8u,10u);nhi=P[16];
  }}
  if((h[0]&P[3])==0u&&(h[1]&P[4])==0u){{if(atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=nlo;R.nonce_hi=nhi;for(var i=0u;i<8u;i=i+1u){{R.hash[i]=h[i];}}}}}}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

fn multi_layout_rounds(start: usize) -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate().skip(start) {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    rounds
}

fn multi_layout_shader_source() -> String {
    let rounds = multi_layout_rounds(0);
    let rounds9 = multi_layout_rounds(9);
    let rounds13 = multi_layout_rounds(13);
    format!(
        r"
const IV:array<u32,8>=array<u32,8>(0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
override MODE:u32;
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h:array<u32,8>)->bool{{
 let ty=P[1];
 if(ty==8u){{
  let bits=P[7];if(P[6]!=0u||bits>256u){{return false;}}let full_words=bits>>5u;
  for(var i=0u;i<8u;i=i+1u){{if(i<full_words&&h[i]!=0u){{return false;}}}}
  let remaining=bits&31u;if(remaining==0u){{return true;}}
  return (h[full_words]>>(32u-remaining))==0u;
 }}
 let full=(P[2]&1u)!=0u;
 if(ty==1u){{return select(h[0]<P[6],less64(h[0],h[1],P[6],P[7]),full);}}
 if(ty==2u){{return select(h[0]>P[6],greater64(h[0],h[1],P[6],P[7]),full);}}
 let hi=(h[0]&P[8])==(P[6]&P[8]);return hi&&select(true,(h[1]&P[9])==(P[7]&P[9]),full);
}}
fn compress(s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds}
 return array<u32,8>(s[0]+a,s[1]+b,s[2]+c,s[3]+d,s[4]+e,s[5]+f,s[6]+g,s[7]+h);
}}
fn compress9(feed:array<u32,8>,s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds9}
 return array<u32,8>(feed[0]+a,feed[1]+b,feed[2]+c,feed[3]+d,feed[4]+e,feed[5]+f,feed[6]+g,feed[7]+h);
}}
fn compress13(feed:array<u32,8>,s:array<u32,8>,block:array<u32,16>)->array<u32,8>{{
 var w0=block[0];var w1=block[1];var w2=block[2];var w3=block[3];var w4=block[4];var w5=block[5];var w6=block[6];var w7=block[7];
 var w8=block[8];var w9=block[9];var w10=block[10];var w11=block[11];var w12=block[12];var w13=block[13];var w14=block[14];var w15=block[15];
 var a=s[0];var b=s[1];var c=s[2];var d=s[3];var e=s[4];var f=s[5];var g=s[6];var h=s[7];
 {rounds13}
 return array<u32,8>(feed[0]+a,feed[1]+b,feed[2]+c,feed[3]+d,feed[4]+e,feed[5]+f,feed[6]+g,feed[7]+h);
}}
fn publish(nlo:u32,nhi:u32,h:array<u32,8>){{if(target_ok(h)&&atomicExchange(&R.flag,1u)==0u){{R.nonce_lo=nlo;R.nonce_hi=nhi;for(var i=0u;i<8u;i=i+1u){{R.hash[i]=h[i];}}}}}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=gid.x*{STEPS}u;
 var decimal_block=array<u32,16>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39]);
 var binary_blocks=array<u32,32>(P[24],P[25],P[26],P[27],P[28],P[29],P[30],P[31],P[32],P[33],P[34],P[35],P[36],P[37],P[38],P[39],P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47],P[48],P[49],P[50],P[51],P[52],P[53],P[54],P[55]);
 if(MODE==1u&&first<P[5]){{
  var x=P[3]+first;
  for(var j=0u;j<9u;j=j+1u){{let digit=x%10u;x=x/10u;let pos=62u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);}}
 }}
 if(MODE==3u&&first<P[5]){{
  let blo=P[3]+first;let bc=select(0u,1u,blo<P[3]);let bhi=P[4]+bc;
  for(var j=0u;j<P[11];j=j+1u){{let byte=select((blo>>(j*8u))&0xffu,(bhi>>((j-4u)*8u))&0xffu,j>=4u);let pos=P[12]+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;binary_blocks[wi]=(binary_blocks[wi]&~(0xffu<<sh))|(byte<<sh);}}
 }}
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let idx=first+step;if(idx>=P[5]){{return;}}
  let clo=P[3]+idx;let carry=select(0u,1u,clo<P[3]);let chi=P[4]+carry;let mode=MODE;
  if(mode==1u){{
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);let hs=array<u32,8>(P[44],P[45],P[46],P[47],P[48],P[49],P[50],P[51]);let mid=compress13(st,hs,decimal_block);
   var tail=array<u32,16>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,P[42],P[43]);let h=compress(mid,tail);
   let nlo=P[40]+clo;let nc=select(0u,1u,nlo<P[40]);publish(nlo,P[41]+nc,h);
   if(step+1u<{STEPS}u&&idx+1u<P[5]){{
    var j=0u;loop{{let pos=62u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let digit=(decimal_block[wi]>>sh)&0xffu;if(digit<0x39u){{decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}decimal_block[wi]=(decimal_block[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}}}
   }}
  }} else if(mode==2u){{
   let block=array<u32,16>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23],P[24],clo,0x80000000u,0u,0u,0u,0u,320u);
   let hs=array<u32,8>(P[40],P[41],P[42],P[43],P[44],P[45],P[46],P[47]);publish(clo,P[24],compress9(IV,hs,block));
  }} else if(mode==3u){{
   let b0=array<u32,16>(binary_blocks[0],binary_blocks[1],binary_blocks[2],binary_blocks[3],binary_blocks[4],binary_blocks[5],binary_blocks[6],binary_blocks[7],binary_blocks[8],binary_blocks[9],binary_blocks[10],binary_blocks[11],binary_blocks[12],binary_blocks[13],binary_blocks[14],binary_blocks[15]);
   let st=array<u32,8>(P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);var h=compress(st,b0);
   if(P[10]==2u){{let b1=array<u32,16>(binary_blocks[16],binary_blocks[17],binary_blocks[18],binary_blocks[19],binary_blocks[20],binary_blocks[21],binary_blocks[22],binary_blocks[23],binary_blocks[24],binary_blocks[25],binary_blocks[26],binary_blocks[27],binary_blocks[28],binary_blocks[29],binary_blocks[30],binary_blocks[31]);h=compress(h,b1);}}
   publish(clo,chi,h);
   if(step+1u<{STEPS}u&&idx+1u<P[5]){{var j=0u;loop{{let pos=P[12]+j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let byte=(binary_blocks[wi]>>sh)&0xffu;if(byte<0xffu){{binary_blocks[wi]=(binary_blocks[wi]&~(0xffu<<sh))|((byte+1u)<<sh);break;}}binary_blocks[wi]=binary_blocks[wi]&~(0xffu<<sh);j=j+1u;if(j==P[11]){{break;}}}}}}
  }}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

fn verify_double_block<const TYPE: u8>(
    message: &crate::message::DoubleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let Some(mut suffix) = nonce.checked_sub(message.nonce_addend) else {
        return false;
    };
    if suffix >= 1_000_000_000 {
        return false;
    }
    let mut block = message.message;
    for j in (0..9).rev() {
        let digit = u32::from((suffix % 10).to_le_bytes()[0]);
        suffix /= 10;
        let pos = crate::message::DoubleBlockMessage::DIGIT_IDX + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        block[wi] = (block[wi] & !(0xffu32 << sh)) | ((u32::from(b'0') + digit) << sh);
    }
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &block);
    let mut tail = [0u32; 16];
    let bits = message.message_length * 8;
    tail[14] = high_u32(bits);
    tail[15] = low_u32(bits);
    crate::sha256::digest_block(&mut state, &tail);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn verify_goaway<const TYPE: u8>(
    message: &crate::message::GoAwayMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    if high_u32(nonce) != message.high_word {
        return false;
    }
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&message.challenge);
    block[8] = message.high_word;
    block[9] = low_u32(nonce);
    block[10] = 0x8000_0000;
    block[15] = 320;
    let mut state = crate::sha256::IV;
    crate::sha256::digest_block(&mut state, &block);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn binary_templates(message: &crate::message::BinaryMessage) -> ([[u32; 16]; 2], u32) {
    let mut bytes = [[0u8; 64]; 2];
    bytes[0][..message.salt_residual_len]
        .copy_from_slice(&message.salt_residual[..message.salt_residual_len]);
    let mut ptr = message.salt_residual_len;
    let mut cur = 0usize;
    for _ in 0..message.nonce_byte_count.get() {
        ptr += 1;
        if ptr == 64 {
            cur = 1;
            ptr = 0;
        }
    }
    bytes[cur][ptr] = 0x80;
    ptr += 1;
    if ptr + 8 > 64 {
        cur = 1;
    }
    bytes[cur][56..64].copy_from_slice(
        &(u64::try_from(message.message_length)
            .unwrap_or(u64::MAX)
            .saturating_mul(8))
        .to_be_bytes(),
    );
    (
        core::array::from_fn(|b| be_words_16(&bytes[b])),
        u32::try_from(cur + 1).unwrap_or(2),
    )
}
fn verify_binary<const TYPE: u8>(
    message: &crate::message::BinaryMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let (blocks, used) = binary_templates(message);
    let mut flat = [0u32; 32];
    flat[..16].copy_from_slice(&blocks[0]);
    flat[16..].copy_from_slice(&blocks[1]);
    let bytes = nonce.to_le_bytes();
    for (j, byte) in bytes
        .iter()
        .enumerate()
        .take(usize::from(message.nonce_byte_count.get()))
    {
        let pos = message.salt_residual_len + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        flat[wi] = (flat[wi] & !(0xffu32 << sh)) | (u32::from(*byte) << sh);
    }
    let mut state = message.prefix_state;
    let blocks = flat.as_chunks::<16>().0;
    crate::sha256::digest_block(&mut state, &blocks[0]);
    if used == 2 {
        crate::sha256::digest_block(&mut state, &blocks[1]);
    }
    state == hash && result_matches::<TYPE>(hash, target, mask)
}

fn generic_single_block_shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    format!(
        r"
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<storage,read> P:array<u32>;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
fn less64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah<bh||(ah==bh&&al<bl);}}
fn greater64(ah:u32,al:u32,bh:u32,bl:u32)->bool{{return ah>bh||(ah==bh&&al>bl);}}
fn target_ok(h:array<u32,8>)->bool{{
 let ty=P[32];
 if(ty==8u){{
  let bits=P[29];if(P[28]!=0u||bits>256u){{return false;}}let full_words=bits>>5u;
  for(var i=0u;i<8u;i=i+1u){{if(i<full_words&&h[i]!=0u){{return false;}}}}
  let remaining=bits&31u;if(remaining==0u){{return true;}}
  return (h[full_words]>>(32u-remaining))==0u;
 }}
 let full=(P[33]&2u)!=0u;
 if(ty==1u){{return select(h[0]<P[28],less64(h[0],h[1],P[28],P[29]),full);}}
 if(ty==2u){{return select(h[0]>P[28],greater64(h[0],h[1],P[28],P[29]),full);}}
 let hi=(h[0]&P[30])==(P[28]&P[30]);
 return hi&&select(true,(h[1]&P[31])==(P[29]&P[31]),full);
}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let first=P[24]+gid.x*{STEPS}u;if(first>=P[34]){{return;}}
 var words=array<u32,16>(P[8],P[9],P[10],P[11],P[12],P[13],P[14],P[15],P[16],P[17],P[18],P[19],P[20],P[21],P[22],P[23]);
 // Format once per invocation. The remaining candidates are generated by an in-place ASCII decimal odometer.
 var x=first;
 for(var j=0u;j<9u;j=j+1u){{
  let digit=x%10u;x=x/10u;let pos=P[25]+8u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;
  words[wi]=(words[wi]&~(0xffu<<sh))|((0x30u+digit)<<sh);
 }}
 let last_pos=P[25]+8u;let last_wi=last_pos>>2u;let last_sh=(3u-(last_pos&3u))*8u;
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if((step&{}u)==0u&&atomicLoad(&R.flag)!=0u){{return;}}
  let candidate=first+step;if(candidate>=P[34]){{return;}}
  let trailing_zero=((words[last_wi]>>last_sh)&0xffu)==0x30u;
  if(!((P[33]&1u)!=0u&&trailing_zero)){{
   var w0=words[0];var w1=words[1];var w2=words[2];var w3=words[3];var w4=words[4];var w5=words[5];var w6=words[6];var w7=words[7];
   var w8=words[8];var w9=words[9];var w10=words[10];var w11=words[11];var w12=words[12];var w13=words[13];var w14=words[14];var w15=words[15];
   var a=P[0];var b=P[1];var c=P[2];var d=P[3];var e=P[4];var f=P[5];var g=P[6];var h=P[7];
   {rounds}
   let out=array<u32,8>(P[0]+a,P[1]+b,P[2]+c,P[3]+d,P[4]+e,P[5]+f,P[6]+g,P[7]+h);
   if(target_ok(out)){{if(atomicExchange(&R.flag,1u)==0u){{
    let nlo=P[26]+candidate;let carry=select(0u,1u,nlo<P[26]);R.nonce_lo=nlo;R.nonce_hi=P[27]+carry;
    for(var i=0u;i<8u;i=i+1u){{R.hash[i]=out[i];}}
   }}}}
  }}
  if(step+1u<{STEPS}u&&candidate+1u<P[34]){{
   var j=0u;
   loop{{
    let pos=P[25]+8u-j;let wi=pos>>2u;let sh=(3u-(pos&3u))*8u;let digit=(words[wi]>>sh)&0xffu;
    if(digit<0x39u){{words[wi]=(words[wi]&~(0xffu<<sh))|((digit+1u)<<sh);break;}}
    words[wi]=(words[wi]&~(0xffu<<sh))|(0x30u<<sh);j=j+1u;if(j==9u){{break;}}
   }}
  }}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

fn shader_source() -> String {
    let mut rounds = String::new();
    for (t, k) in crate::sha256::K32.iter().enumerate() {
        let i = t & 15;
        if t >= 16 {
            append_fmt(
                &mut rounds,
                format_args!(
                    "w{i}=(rotr(w{},17u)^rotr(w{},19u)^(w{}>>10u))+w{}+(rotr(w{},7u)^rotr(w{},18u)^(w{}>>3u))+w{i};\n",
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 2) & 15,
                    (t - 7) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15,
                    (t - 15) & 15
                ),
            );
        }
        append_fmt(
            &mut rounds,
            format_args!(
                "{{let t1=h+(rotr(e,6u)^rotr(e,11u)^rotr(e,25u))+(g^(e&(f^g)))+0x{k:08x}u+w{i};let t2=(rotr(a,2u)^rotr(a,13u)^rotr(a,22u))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}}\n"
            ),
        );
    }
    format!(
        r"
struct InParams {{ s0:u32,s1:u32,s2:u32,s3:u32,s4:u32,s5:u32,s6:u32,s7:u32,base_lo:u32,base_hi:u32,bitlen:u32,mask0:u32,mask1:u32,batch_size:u32 }};
struct OutBuf {{ flag:atomic<u32>,nonce_lo:u32,nonce_hi:u32,hash:array<u32,8> }};
@group(0) @binding(0) var<uniform> P:InParams;
@group(0) @binding(1) var<storage,read_write> R:OutBuf;
fn rotr(x:u32,n:u32)->u32{{return (x>>n)|(x<<(32u-n));}}
@compute @workgroup_size({WG_SIZE})
fn solve(@builtin(global_invocation_id) gid:vec3<u32>){{
 let local_first=gid.x*{STEPS}u;if(local_first>=P.batch_size){{return;}}
 let gid_lo=gid.x<<7u;let gid_hi=gid.x>>25u;let base_lo=P.base_lo+gid_lo;var base_hi=P.base_hi+gid_hi;if(base_lo<P.base_lo){{base_hi=base_hi+1u;}}
 let d2=(base_lo>>6u)&7u;let d3=(base_lo>>9u)&7u;let d4=(base_lo>>12u)&7u;let d5=(base_lo>>15u)&7u;let d6=(base_lo>>18u)&7u;let d7=(base_lo>>21u)&7u;let d8=(base_lo>>24u)&7u;let d9=(base_lo>>27u)&7u;let d10=((base_lo>>30u)|(base_hi<<2u))&7u;let d11=(base_hi>>1u)&7u;let d12=(base_hi>>4u)&7u;let d13=(base_hi>>7u)&7u;let d14=(base_hi>>10u)&7u;let d15=(base_hi>>13u)&7u;let d16=(base_hi>>16u)&7u;let d17=(base_hi>>19u)&7u;
 let nw0=0x31303030u|(d17<<16u)|(d16<<8u)|d15;let nw1=0x30303030u|(d14<<24u)|(d13<<16u)|(d12<<8u)|d11;let nw2=0x30303030u|(d10<<24u)|(d9<<16u)|(d8<<8u)|d7;let nw3=0x30303030u|(d6<<24u)|(d5<<16u)|(d4<<8u)|d3;var nw4=0x30303080u|(d2<<24u);
 const GM:array<u32,7>=array<u32,7>(0x00000100u,0x00000200u,0x00000400u,0x00010000u,0x00020000u,0x00040000u,0x01000000u);
 for(var step=0u;step<{STEPS}u;step=step+1u){{
  if(local_first+step>=P.batch_size){{return;}}
  if((step&{}u)==0u && atomicLoad(&R.flag)!=0u){{return;}}
  var w0=nw0;var w1=nw1;var w2=nw2;var w3=nw3;var w4=nw4;
  var w5=0u;var w6=0u;var w7=0u;var w8=0u;var w9=0u;var w10=0u;var w11=0u;var w12=0u;var w13=0u;var w14=0u;var w15=P.bitlen;
  var a=P.s0;var b=P.s1;var c=P.s2;var d=P.s3;var e=P.s4;var f=P.s5;var g=P.s6;var h=P.s7;
  {rounds}
  let r0=P.s0+a;let r1=P.s1+b;
  if((r0&P.mask0)==0u&&(r1&P.mask1)==0u){{if(atomicExchange(&R.flag,1u)==0u){{let gray=step^(step>>1u);let clo=base_lo+gray;let carry=select(0u,1u,clo<base_lo);R.nonce_lo=clo;R.nonce_hi=base_hi+carry;R.hash[0]=r0;R.hash[1]=r1;R.hash[2]=P.s2+c;R.hash[3]=P.s3+d;R.hash[4]=P.s4+e;R.hash[5]=P.s5+f;R.hash[6]=P.s6+g;R.hash[7]=P.s7+h;}}}}
  let ns=step+1u;if(ns<{STEPS}u){{nw4=nw4^GM[countTrailingZeros(ns)];}}
 }}
}}
",
        CANCEL_INTERVAL - 1
    )
}

#[cfg(test)]
mod sha512_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbkdf2_u1_salt_blocks_have_correct_padding_and_bit_length() -> Result<(), GpuError> {
        for length in [
            0_usize, 1, 51, 52, 55, 56, 59, 60, 63, 64, 115, 116, 119, 120, 127, 128, 4097,
        ] {
            let salt: Vec<u8> = (0..length).map(|i| i as u8).collect();
            let input = AltchaPbkdf2Input::new(&salt, &[0x15; 96])?;
            let packed: Vec<u8> = input
                .u1_blocks
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect();
            assert_eq!(&packed[..length], salt);
            assert_eq!(&packed[length..length + 4], &[0, 0, 0, 1]);
            assert_eq!(packed[length + 4], 0x80);
            assert!(packed[length + 5..packed.len() - 8].iter().all(|b| *b == 0));
            let bit_len = u64::from_be_bytes(
                packed[packed.len() - 8..]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid SHA padding tail"))?,
            );
            assert_eq!(
                bit_len,
                (u64::try_from(length).map_err(|_| GpuError::new("invalid length"))? + 68) * 8
            );
            assert_eq!(packed.len() % 64, 0);
        }
        Ok(())
    }

    #[test]
    fn counter_conversion_is_decimal_octal_digits() {
        assert_eq!(counter_to_decimal_nonce(0), 1_000_000_000_000_000_000);
        assert_eq!(
            counter_to_decimal_nonce(0o1_234_567),
            1_000_000_000_001_234_567
        );
        assert!(counter_to_decimal_nonce(COUNTER_SPACE - 1) < u64::MAX);
    }

    #[test]
    fn full_digest_result_match_checks_past_first_64_bits() {
        let mut hash = [0_u32; 8];
        hash[2] = 0x7fff_ffff;
        assert!(result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >(hash, 65, u64::MAX,));
        hash[2] = 0x8000_0000;
        assert!(!result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >(hash, 65, u64::MAX,));
        assert!(!result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >([0; 8], 257, u64::MAX,));
    }
}
