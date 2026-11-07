//! GPU core of RFC 7914 scrypt: batched independent `ROMix` lanes.
//!
//! The caller performs PBKDF2-HMAC-SHA256 before/after `ROMix`. A lane is one
//! 128*r-byte segment B[i], so p independent segments can run in parallel.
//! This separation makes GPU scratch occupancy measurable without coupling
//! the `ROMix` kernel to a particular challenge protocol or counter format.

use std::borrow::Cow;

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

impl<'a> ScryptRomixGpu<'a> {
    /// Create reusable RFC 7914 `ROMix` GPU pipelines.
    #[must_use]
    pub fn new(gpu: &'a GpuContext) -> Self {
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/scrypt_romix.wgsl"
            ))),
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
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(blocks.len())
            .map_err(|e| GpuError::new(format!("reserve scrypt output: {e}")))?;
        let chunk_bytes = lanes_per_batch
            .checked_mul(block_bytes)
            .ok_or_else(|| GpuError::new("scrypt batch chunk byte length overflow"))?;
        for batch in blocks.chunks(chunk_bytes) {
            let lane_count = batch.len() / block_bytes;
            let gpu_chunk = self.mix_batch(batch, n, r, lane_count).await?;
            result.extend_from_slice(&gpu_chunk);
        }
        Ok(result)
    }

    async fn mix_batch(
        &self,
        initial: &[u8],
        n: u32,
        r: u32,
        lanes: usize,
    ) -> Result<Vec<u8>, GpuError> {
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
        gpu.queue.write_buffer(&data, 0, initial);
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("challenge-solvers-scrypt-romix-bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tmp.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("challenge-solvers-scrypt-romix"),
        });
        for (label, pipeline) in [("scrypt-V-fill", &self.fill), ("scrypt-ROMix", &self.mix)] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(lanes_u32.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&data, 0, &staging, 0, data_bytes);
        gpu.queue.submit([encoder.finish()]);
        readback(device, &staging).await
    }
}

async fn readback(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<Vec<u8>, GpuError> {
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
