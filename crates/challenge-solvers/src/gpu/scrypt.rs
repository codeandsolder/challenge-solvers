//! GPU core of RFC 7914 scrypt: batched independent `ROMix` lanes.
//!
//! The caller performs PBKDF2-HMAC-SHA256 before/after `ROMix`. A lane is one
//! 128*r-byte segment B[i], so p independent segments can run in parallel.
//! This separation makes GPU scratch occupancy measurable without coupling
//! the `ROMix` kernel to a particular challenge protocol or counter format.

use std::borrow::Cow;

mod kdf;
pub use kdf::ScryptKdfGpu;

use super::{GpuContext, GpuError};

/// Reusable scrypt `ROMix` pipeline, independent of proof-of-work protocols.
///
/// Construct once per GPU context. The shader is compiled lazily only when
/// scrypt work is actually requested.
pub struct ScryptRomixGpu<'a> {
    gpu: &'a GpuContext,
    layout: wgpu::BindGroupLayout,
    fill: wgpu::ComputePipeline,
    mix: wgpu::ComputePipeline,
}

/// Device-side time spent in each of the two scrypt `ROMix` compute passes.
/// Raw GPU timestamp deltas are kept separate from host submission latency.
#[derive(Debug, Clone, Copy)]
pub struct ScryptRomixTimings {
    /// Timestamp ticks covering the fill pass.
    pub fill_ticks: u64,
    /// Timestamp ticks covering the mix pass.
    pub mix_ticks: u64,
    /// Nanoseconds per GPU tick as reported by the device queue.
    pub timestamp_period_ns: f32,
}

impl<'a> ScryptRomixGpu<'a> {
    /// Create reusable RFC 7914 `ROMix` GPU pipelines.
    #[must_use]
    pub fn new(gpu: &'a GpuContext) -> Self {
        Self::compile_wgsl(gpu, include_str!("../shaders/scrypt_romix.wgsl"))
    }

    /// Compile the same `ROMix` implementation from a supplied WGSL source.
    ///
    /// Only available to internal benchmark tools: this allows shader
    /// revisions to be measured interleaved on one device and queue without
    /// exposing shader substitution to production callers.
    #[cfg(feature = "internals")]
    #[must_use]
    pub fn new_from_wgsl(gpu: &'a GpuContext, source: &str) -> Self {
        Self::compile_wgsl(gpu, source)
    }

    fn compile_wgsl(gpu: &'a GpuContext, source: &str) -> Self {
        let device = &gpu.device;
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("challenge-solvers-scrypt-romix-layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("challenge-solvers-scrypt-romix-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-scrypt-romix-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let create = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("challenge-solvers-scrypt-romix"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            gpu,
            layout,
            fill: create("fill"),
            mix: create("mix"),
        }
    }

    /// Maximum independent `ROMix` segments per batch under actual WebGPU
    /// storage-buffer and dispatch limits. Each lane uses 128*r*N bytes
    /// of random-access scratch, plus 2*128*r bytes for state and `BlockMix`.
    ///
    /// # Errors
    ///
    /// Rejects non-scrypt parameters and scratch sizes exceeding the
    /// device's maximum individual storage-buffer binding.
    pub fn batch_lanes(&self, n: u32, r: u32) -> Result<usize, GpuError> {
        if n < 2 || !n.is_power_of_two() || r == 0 {
            return Err(GpuError::new(
                "scrypt ROMix requires power-of-two N>=2 and r>=1",
            ));
        }
        let block_bytes = u64::from(r)
            .checked_mul(128)
            .ok_or_else(|| GpuError::new("scrypt block size overflow"))?;
        let scratch_bytes = block_bytes
            .checked_mul(u64::from(n))
            .ok_or_else(|| GpuError::new("scrypt scratch size overflow"))?;
        let limits = self.gpu.device.limits();
        let per_binding_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let max_lanes = (per_binding_limit / scratch_bytes)
            .min(per_binding_limit / block_bytes)
            .min(u64::from(self.gpu.max_wgs) * 64);
        usize::try_from(max_lanes)
            .ok()
            .filter(|&count| count > 0)
            .ok_or_else(|| GpuError::new(format!(
                "scrypt scratch ({scratch_bytes} bytes/lane) exceeds device storage binding limit ({per_binding_limit} bytes)"
            )))
    }

    /// Execute RFC 7914 `ROMix` over any number of independent 128*r-byte
    /// input segments. Each segment is a separate lane, dynamically batched
    /// according to device memory limits.
    ///
    /// Input must be the B bytes produced by the initial PBKDF2-HMAC-SHA256
    /// operation. Output becomes input to the final PBKDF2 operation.
    ///
    /// # Errors
    ///
    /// Rejects invalid sizes, and reports GPU dispatch/readback failures.
    pub async fn mix_blocks(&self, blocks: &[u8], n: u32, r: u32) -> Result<Vec<u8>, GpuError> {
        self.mix_blocks_inner(blocks, n, r, false)
            .await
            .map(|(result, _)| result)
    }

    /// Run `ROMix` and report GPU fill/mix pass timestamps for each device-sized batch.
    ///
    /// Requires a context created by [`GpuContext::create_with_timestamps`].
    /// Normal [`Self::mix_blocks`] calls do not request timestamp features.
    ///
    /// # Errors
    ///
    /// Reports invalid input, unavailable GPU timestamp queries or device failures.
    pub async fn mix_blocks_with_timestamps(
        &self,
        blocks: &[u8],
        n: u32,
        r: u32,
    ) -> Result<(Vec<u8>, Vec<ScryptRomixTimings>), GpuError> {
        self.mix_blocks_inner(blocks, n, r, true).await
    }

    async fn mix_blocks_inner(
        &self,
        blocks: &[u8],
        n: u32,
        r: u32,
        profile: bool,
    ) -> Result<(Vec<u8>, Vec<ScryptRomixTimings>), GpuError> {
        let lanes_per_batch = self.batch_lanes(n, r)?;
        let block_bytes = usize::try_from(r)
            .ok()
            .and_then(|v| v.checked_mul(128))
            .ok_or_else(|| GpuError::new("scrypt block size exceeds host pointer size"))?;
        if !blocks.len().is_multiple_of(block_bytes) {
            return Err(GpuError::new(
                "scrypt initial B length must be multiple of 128*r",
            ));
        }
        if blocks.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut result = Vec::new();
        let mut timings = Vec::new();
        result
            .try_reserve_exact(blocks.len())
            .map_err(|e| GpuError::new(format!("reserve scrypt output: {e}")))?;
        let chunk_bytes = lanes_per_batch
            .checked_mul(block_bytes)
            .ok_or_else(|| GpuError::new("scrypt batch chunk byte length overflow"))?;
        for batch in blocks.chunks(chunk_bytes) {
            let lane_count = batch.len() / block_bytes;
            let (gpu_chunk, timing) = self.mix_batch(batch, n, r, lane_count, profile).await?;
            result.extend_from_slice(&gpu_chunk);
            if let Some(timing) = timing {
                timings.push(timing);
            }
        }
        Ok((result, timings))
    }

    fn bind_buffers(
        &self,
        uniform: &wgpu::Buffer,
        state: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        mixed: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("challenge-solvers-scrypt-romix-bindings"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scratch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mixed.as_entire_binding(),
                    },
                ],
            })
    }

    async fn mix_batch(
        &self,
        initial: &[u8],
        n: u32,
        r: u32,
        lanes: usize,
        profile: bool,
    ) -> Result<(Vec<u8>, Option<ScryptRomixTimings>), GpuError> {
        let gpu = self.gpu;
        let device = &gpu.device;
        let lanes_u32 = u32::try_from(lanes)
            .map_err(|_| GpuError::new("scrypt batch lane count exceeds GPU u32 range"))?;
        let data_bytes = u64::try_from(initial.len())
            .map_err(|_| GpuError::new("scrypt B byte length exceeds GPU limits"))?;
        let scratch_bytes = data_bytes
            .checked_mul(u64::from(n))
            .ok_or_else(|| GpuError::new("scrypt batch scratch byte length overflow"))?;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("challenge-solvers-scrypt-parameters"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::cast_slice(&[n, r, lanes_u32, 0_u32]));
        let create = |label: &'static str, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let data = create(
            "challenge-solvers-scrypt-B",
            data_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let scratch = create(
            "challenge-solvers-scrypt-V",
            scratch_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let tmp = create(
            "challenge-solvers-scrypt-Y",
            data_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let staging = create(
            "challenge-solvers-scrypt-readback",
            data_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let timestamps = profile
            .then(|| RomixTimestampCapture::new(device))
            .transpose()?;
        gpu.queue.write_buffer(&data, 0, initial);
        let bind = self.bind_buffers(&uniform, &data, &scratch, &tmp);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("challenge-solvers-scrypt-romix"),
        });
        for (step, (label, pipeline)) in
            [("scrypt-V-fill", &self.fill), ("scrypt-ROMix", &self.mix)]
                .into_iter()
                .enumerate()
        {
            let begin = if step == 0 { 0 } else { 2 };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timestamps.as_ref().map(|capture| capture.writes(begin)),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(lanes_u32.div_ceil(64), 1, 1);
        }
        if let Some(capture) = &timestamps {
            capture.resolve(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(&data, 0, &staging, 0, data_bytes);
        gpu.queue.submit([encoder.finish()]);
        let output = readback_scrypt(device, &staging).await?;
        let timing = match timestamps {
            Some(capture) => Some(capture.read(gpu).await?),
            None => None,
        };
        Ok((output, timing))
    }
}

// Profiling buffers are allocated only when a timestamp-capable GPU context
// explicitly requests them; regular ROMix work has no extra GPU allocations.
struct RomixTimestampCapture {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl RomixTimestampCapture {
    fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(GpuError::new(
                "scrypt profiling requires GPU timestamp queries",
            ));
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("scrypt-romix-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 4,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 32,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            queries,
            resolved: buffer(
                "scrypt-romix-timestamp-resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "scrypt-romix-timestamp-readback",
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
        })
    }

    fn writes(&self, first: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        }
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.queries, 0..4, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.readback, 0, 32);
    }

    async fn read(self, gpu: &GpuContext) -> Result<ScryptRomixTimings, GpuError> {
        let raw = readback_scrypt(&gpu.device, &self.readback).await?;
        let (chunks, remainder) = raw.as_chunks::<8>();
        if !remainder.is_empty() {
            return Err(GpuError::new("scrypt timestamp malformed byte count"));
        }
        let bytes: [[u8; 8]; 4] = chunks
            .try_into()
            .map_err(|_| GpuError::new("scrypt timestamp readback is not four u64 values"))?;
        let [fill_start, fill_end, mix_start, mix_end] = bytes.map(u64::from_le_bytes);
        Ok(ScryptRomixTimings {
            fill_ticks: fill_end
                .checked_sub(fill_start)
                .ok_or_else(|| GpuError::new("scrypt fill GPU timestamps went backwards"))?,
            mix_ticks: mix_end
                .checked_sub(mix_start)
                .ok_or_else(|| GpuError::new("scrypt mix GPU timestamps went backwards"))?,
            timestamp_period_ns: gpu.queue.get_timestamp_period(),
        })
    }
}

pub(super) async fn readback_scrypt(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
) -> Result<Vec<u8>, GpuError> {
    let slice = staging.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuError::new(format!("poll scrypt GPU: {e}")))?;
    receiver
        .await
        .map_err(|_| GpuError::new("scrypt GPU map callback dropped"))?
        .map_err(|e| GpuError::new(format!("map scrypt GPU output: {e}")))?;
    let mapped = slice
        .get_mapped_range()
        .map_err(|e| GpuError::new(format!("read scrypt GPU output: {e}")))?;
    let output = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    Ok(output)
}
