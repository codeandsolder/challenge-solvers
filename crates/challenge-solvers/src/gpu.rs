//! WebGPU/native GPU solver backend for challenge-solvers message families.
//!
//! The same Rust implementation targets native `wgpu` backends and browser WebGPU on `wasm32`.

use core::{fmt, future::Future, num::NonZeroU8, pin::Pin};
use std::{borrow::Cow, string::String, vec::Vec};

use bytemuck::{Pod, Zeroable};

mod linear_sha1;
pub use linear_sha1::{LinearSha1GpuSolver, LinearSha1Solution};
mod sha1_decimal;
pub use sha1_decimal::{Sha1DecimalGpuSolver, Sha1DecimalSolution};

mod altcha_sha512;
pub use altcha_sha512::{AltchaSha64Algorithm, AltchaSha64GpuSolver, AltchaSha64Solution};

mod scrypt;
pub use scrypt::{ScryptKdfGpu, ScryptRomixGpu};

mod solvers;
pub use solvers::{
    AltchaSha256Solver, BinarySolver, CerberusSolver, DecimalSolver, DoubleBlockSolver,
    GoAwaySolver, SingleBlockSolver,
};

mod shaders;
use shaders::{
    altcha_shader_source, cerberus_shader_source, generic_single_block_shader_source,
    multi_layout_shader_source, shader_source,
};

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
    // Protocol-specific shader pipelines are compiled once per logical context
    // on first use. Unused GPU families do not increase startup overhead.
    linear_sha1_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    sha1_decimal_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    altcha_sha64_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
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
    /// Features actually enabled on this GPU device, for selecting optional
    /// acceleration kernels and reporting accurate adapter capabilities.
    ///
    /// This is the negotiated device feature set, not an adapter-wide claim:
    /// callers must check it before selecting kernels that require additional
    /// features such as native 64-bit shader integers.
    #[must_use]
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
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

#[cfg(test)]
mod sha512_tests;

#[cfg(test)]
mod tests {
    use super::*;

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
