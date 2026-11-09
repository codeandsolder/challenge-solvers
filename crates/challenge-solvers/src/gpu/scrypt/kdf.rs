//! Complete scrypt KDF on WebGPU: PBKDF2, `ROMix`, and final PBKDF2.
//!
//! Only HMAC ipad/opad SHA256 chaining states are prepared on CPU, once per
//! password. Both key-stretching phases and all `ROMix` scratch access remain
//! device-side, with a single final key readback.

use std::borrow::Cow;

use sha2::{Digest as _, Sha256};

use super::{
    super::{GpuContext, GpuError},
    ScryptRomixGpu, readback_scrypt,
};
use crate::sha256::{IV, digest_block};

/// GPU scrypt, including both PBKDF2-HMAC-SHA256 stages and RFC 7914 `ROMix`.
///
/// Each scrypt p-lane is independently mixed; large p is split into
/// GPU-memory-bounded `ROMix` batches without copying intermediate B to host.
pub struct ScryptKdfGpu<'a> {
    romix: ScryptRomixGpu<'a>,
    layout: wgpu::BindGroupLayout,
    initial: wgpu::ComputePipeline,
    final_stage: wgpu::ComputePipeline,
}

impl<'a> ScryptKdfGpu<'a> {
    /// Compile the complete scrypt KDF pipelines on the selected GPU.
    #[must_use]
    pub fn new(gpu: &'a GpuContext) -> Self {
        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("challenge-solvers-scrypt-kdf-layout"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("challenge-solvers-scrypt-kdf-pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("challenge-solvers-scrypt-kdf"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                include_str!("../../shaders/scrypt_pbkdf2.wgsl").replace(
                    "__SHA256_CORE__",
                    &super::super::sha256_core_shader_source(),
                ),
            )),
        });
        let create = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("challenge-solvers-scrypt-kdf"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            romix: ScryptRomixGpu::new(gpu),
            layout,
            initial: create("initial"),
            final_stage: create("finish"),
        }
    }

    /// Run both one-iteration PBKDF2-HMAC-SHA256 stages and RFC 7914 `ROMix`
    /// entirely on GPU. Returns the requested derived-key bytes.
    ///
    /// The host prepares candidate-independent ipad/opad SHA chaining states
    /// but performs no PBKDF2 iterations or intermediate B readback.
    ///
    /// # Errors
    ///
    /// Reports invalid RFC 7914 parameters, lengths exceeding actual
    /// requested device storage limits, or GPU dispatch failures.
    pub async fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let spec = KdfSpec::new(&self.romix, salt.len(), n, r, p, key_length)?;
        let key_input = kdf_input(password, salt)?;
        let gpu = self.romix.gpu;
        let device = &gpu.device;
        let buffers = KdfBuffers::new(gpu, &spec, &key_input)?;
        let key_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("challenge-solvers-scrypt-kdf-bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.config.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.key_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.result.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("challenge-solvers-scrypt-kdf-end-to-end"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scrypt-initial-pbkdf2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.initial);
            pass.set_bind_group(0, &key_binding, &[]);
            pass.dispatch_workgroups(spec.initial_workgroups, 1, 1);
        }
        self.encode_romix(&mut encoder, &buffers, &spec);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scrypt-final-pbkdf2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.final_stage);
            pass.set_bind_group(0, &key_binding, &[]);
            pass.dispatch_workgroups(spec.final_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffers.result, 0, &buffers.staging, 0, spec.key_bytes);
        gpu.queue.submit([encoder.finish()]);
        let data = readback_scrypt(device, &buffers.staging).await?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(spec.key_length)
            .map_err(|error| GpuError::new(format!("allocate derived key: {error}")))?;
        for word in data.as_chunks::<4>().0 {
            let native = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            result.extend_from_slice(&native.to_be_bytes());
        }
        result.truncate(spec.key_length);
        Ok(result)
    }

    fn encode_romix(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &KdfBuffers,
        spec: &KdfSpec,
    ) {
        let gpu = self.romix.gpu;
        let device = &gpu.device;
        let scratch = allocate(
            device,
            "challenge-solvers-scrypt-kdf-scratch",
            spec.scratch_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let temporary = allocate(
            device,
            "challenge-solvers-scrypt-kdf-intermediate",
            spec.temporary_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let mut offset = 0_u32;
        while offset < spec.p {
            let lanes = spec.lanes_per_batch.min(spec.p - offset);
            let uniform = allocate(
                device,
                "challenge-solvers-scrypt-kdf-romix-params",
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            gpu.queue.write_buffer(
                &uniform,
                0,
                bytemuck::cast_slice(&[spec.n, spec.r, lanes, offset]),
            );
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("challenge-solvers-scrypt-kdf-romix-binding"),
                layout: &self.romix.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scratch.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: temporary.as_entire_binding(),
                    },
                ],
            });
            for (label, pipeline) in [
                ("scrypt-kdf-romix-fill", &self.romix.fill),
                ("scrypt-kdf-romix-mix", &self.romix.mix),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(lanes.div_ceil(64), 1, 1);
            }
            offset += lanes;
        }
    }
}

const fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
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

fn allocate(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

struct KdfSpec {
    n: u32,
    r: u32,
    p: u32,
    key_length: usize,
    key_bytes: u64,
    b_bytes: u64,
    scratch_bytes: u64,
    temporary_bytes: u64,
    lanes_per_batch: u32,
    salt_len: u32,
    initial_workgroups: u32,
    final_workgroups: u32,
}

impl KdfSpec {
    fn new(
        romix: &ScryptRomixGpu<'_>,
        salt_length: usize,
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<Self, GpuError> {
        if p == 0 || u64::from(r) * u64::from(p) >= 0x4000_0000 {
            return Err(GpuError::new("scrypt p*r must be nonzero and below 2^30"));
        }
        let lanes_limit = romix.batch_lanes(n, r)?;
        let gpu = romix.gpu;
        if key_length == 0
            || !u64::try_from(key_length).is_ok_and(|k| k <= u64::from(u32::MAX) * 32)
        {
            return Err(GpuError::new("scrypt output length exceeds RFC 7914"));
        }
        let salt_len = u32::try_from(salt_length)
            .map_err(|_| GpuError::new("scrypt salt too large for u32 GPU indexing"))?;
        let b_bytes = u64::from(r) * u64::from(p) * 128;
        let b_words = b_bytes / 4;
        let key_blocks = key_length.div_ceil(32);
        let key_bytes = u64::try_from(key_blocks)
            .ok()
            .and_then(|blocks| blocks.checked_mul(32))
            .ok_or_else(|| GpuError::new("scrypt key buffer overflow"))?;
        let limits = gpu.device.limits();
        let storage = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        if b_bytes > storage || key_bytes > storage {
            return Err(GpuError::new(
                "scrypt B or key output exceeds device storage binding limit",
            ));
        }
        let initial_blocks = b_words / 8;
        let initial_workgroups = u32::try_from(initial_blocks.div_ceil(64))
            .map_err(|_| GpuError::new("scrypt initial dispatch overflow"))?;
        let final_workgroups = u32::try_from(key_blocks.div_ceil(64))
            .map_err(|_| GpuError::new("scrypt final dispatch overflow"))?;
        if initial_workgroups > gpu.max_wgs || final_workgroups > gpu.max_wgs {
            return Err(GpuError::new("scrypt PBKDF2 exceeds device dispatch limit"));
        }
        let lanes_per_batch = u32::try_from(
            lanes_limit.min(usize::try_from(p).map_err(|_| GpuError::new("scrypt p conversion"))?),
        )
        .map_err(|_| GpuError::new("scrypt batch lanes exceed u32"))?;
        let temporary_bytes = u64::from(lanes_per_batch) * u64::from(r) * 128;
        let scratch_bytes = temporary_bytes * u64::from(n);
        Ok(Self {
            n,
            r,
            p,
            key_length,
            key_bytes,
            b_bytes,
            scratch_bytes,
            temporary_bytes,
            lanes_per_batch,
            salt_len,
            initial_workgroups,
            final_workgroups,
        })
    }
}

struct KdfBuffers {
    config: wgpu::Buffer,
    key_input: wgpu::Buffer,
    b: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl KdfBuffers {
    fn new(gpu: &GpuContext, spec: &KdfSpec, input: &[u32]) -> Result<Self, GpuError> {
        let input_bytes = u64::try_from(input.len())
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| GpuError::new("scrypt HMAC input length overflow"))?;
        let storage = gpu.device.limits().max_storage_buffer_binding_size;
        if input_bytes > storage {
            return Err(GpuError::new("scrypt salt input exceeds GPU binding limit"));
        }
        let salt_blocks = u32::try_from((input.len() - 16) / 16)
            .map_err(|_| GpuError::new("scrypt padded salt block count overflow"))?;
        let params = [
            spec.n,
            spec.r,
            spec.p,
            spec.salt_len,
            salt_blocks,
            u32::try_from(spec.key_bytes / 32)
                .map_err(|_| GpuError::new("scrypt key blocks overflow"))?,
            u32::try_from(spec.b_bytes / 4)
                .map_err(|_| GpuError::new("scrypt B words overflow"))?,
            0_u32,
        ];
        let device = &gpu.device;
        let config = allocate(
            device,
            "scrypt-kdf-config",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let key_input = allocate(
            device,
            "scrypt-kdf-ipad-opad-salt",
            input_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        gpu.queue
            .write_buffer(&config, 0, bytemuck::cast_slice(&params));
        gpu.queue
            .write_buffer(&key_input, 0, bytemuck::cast_slice(input));
        Ok(Self {
            config,
            key_input,
            b: allocate(
                device,
                "scrypt-kdf-B",
                spec.b_bytes,
                wgpu::BufferUsages::STORAGE,
            ),
            result: allocate(
                device,
                "scrypt-kdf-output",
                spec.key_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: allocate(
                device,
                "scrypt-kdf-staging",
                spec.key_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        })
    }
}

fn kdf_input(password: &[u8], salt: &[u8]) -> Result<Vec<u32>, GpuError> {
    let key: Vec<u8> = if password.len() > 64 {
        Sha256::digest(password).to_vec()
    } else {
        password.to_vec()
    };
    let mut padded_key = [0_u8; 64];
    padded_key[..key.len()].copy_from_slice(&key);
    let ipad = hmac_state(&padded_key, 0x36);
    let opad = hmac_state(&padded_key, 0x5c);
    let count = salt
        .len()
        .checked_add(4)
        .ok_or_else(|| GpuError::new("scrypt salt length overflow"))?;
    let padded = count
        .checked_add(9)
        .and_then(|n| n.div_ceil(64).checked_mul(64))
        .ok_or_else(|| GpuError::new("scrypt padded salt overflow"))?;
    let bit_len = u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_add(64))
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| GpuError::new("scrypt SHA256 message length overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(padded)
        .map_err(|err| GpuError::new(format!("reserve padded scrypt salt: {err}")))?;
    bytes.resize(padded, 0);
    bytes[..salt.len()].copy_from_slice(salt);
    bytes[salt.len() + 3] = 1;
    bytes[count] = 0x80;
    bytes[padded - 8..].copy_from_slice(&bit_len.to_be_bytes());
    let mut words = Vec::with_capacity(16 + padded / 4);
    words.extend_from_slice(&ipad);
    words.extend_from_slice(&opad);
    for chunk in bytes.as_chunks::<4>().0 {
        words.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

fn hmac_state(key: &[u8; 64], mask: u8) -> [u32; 8] {
    let words: [u32; 16] = core::array::from_fn(|i| {
        let index = i * 4;
        u32::from_be_bytes([
            key[index] ^ mask,
            key[index + 1] ^ mask,
            key[index + 2] ^ mask,
            key[index + 3] ^ mask,
        ])
    });
    let mut state = IV;
    digest_block(&mut state, &words);
    state
}
