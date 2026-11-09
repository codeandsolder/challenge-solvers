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

/// GPU scrypt, including both PBKDF2-HMAC-SHA256 stages and RFC 7914 ``ROMix``.
///
/// Each scrypt p-lane is independently mixed; large p is split into
/// GPU-memory-bounded ``ROMix`` batches without copying intermediate B to host.
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

    /// Execute one complete RFC 7914 scrypt KDF on GPU, with no intermediate
    /// device readback.
    ///
    /// # Errors
    ///
    /// Propagates invalid parameters or GPU dispatch/readback failures.
    pub async fn derive_key(
        &self,
        password: &[u8],
        salt: &[u8],
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<Vec<u8>, GpuError> {
        self.derive_keys(&[password], salt, n, r, p, key_length)
            .await?
            .pop()
            .ok_or_else(|| GpuError::new("scrypt single-key GPU output missing"))
    }

    /// Maximum passwords per GPU dispatch for given RFC 7914 parameters.
    ///
    /// This derives occupancy from the actual requested WebGPU storage and
    /// compute limits: B, output, candidate HMAC states, and both PBKDF2
    /// dispatches. Scratch V may be smaller than the full set of lanes
    /// because `ROMix` transparently processes independent lane batches.
    ///
    /// # Errors
    ///
    /// Rejects parameters that cannot fit even one candidate on the device.
    pub fn batch_candidates(
        &self,
        salt_length: usize,
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<usize, GpuError> {
        let spec = KdfSpec::new(&self.romix, salt_length, n, r, p, 1, key_length)?;
        let limits = self.romix.gpu.device.limits();
        let storage = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let salt_template_bytes = salt_length
            .checked_add(13)
            .and_then(|n| n.div_ceil(64).checked_mul(64))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| GpuError::new("scrypt padded salt exceeds host byte range"))?;
        if salt_template_bytes >= storage {
            return Err(GpuError::new(
                "scrypt padded salt leaves no storage for candidate HMAC states",
            ));
        }
        let dispatch_capacity = u64::from(self.romix.gpu.max_wgs)
            .checked_mul(64)
            .ok_or_else(|| GpuError::new("scrypt GPU dispatch limit overflow"))?;
        let max = [
            storage / spec.b_bytes,
            storage / spec.key_bytes,
            (storage - salt_template_bytes) / 64,
            dispatch_capacity / (spec.b_bytes / 32),
            dispatch_capacity / (spec.key_bytes / 32),
            u64::from(u32::MAX),
        ]
        .into_iter()
        .min()
        .unwrap_or(0);
        usize::try_from(max)
            .ok()
            .filter(|&value| value > 0)
            .ok_or_else(|| GpuError::new("scrypt candidate memory or dispatch exceeds this GPU"))
    }

    /// Derive independent RFC 7914 scrypt keys for any batch of passwords
    /// sharing salt, N, r, p and key length. Independent passwords execute
    /// in one submission per automatically sized GPU resource batch.
    ///
    /// CPU preparation is only 2 SHA256 HMAC ipad/opad chaining states per
    /// password, plus a single shared padded salt template. Both PBKDF2
    /// stages and all N-step `ROMix` passes are run on GPU. The full B state
    /// stays on GPU even when the candidate*p lanes span multiple scratch
    /// batches; only final derived keys are read back.
    ///
    /// Empty input returns an empty set of keys. Large sets are transparently
    /// split into batches chosen by `batch_candidates()` to respect the device
    /// storage buffer and dispatch size limits.
    ///
    /// # Errors
    ///
    /// Reports invalid RFC7914 parameters, device storage/dispatch limits,
    /// and GPU execution or readback errors.
    pub async fn derive_keys(
        &self,
        passwords: &[&[u8]],
        salt: &[u8],
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        if passwords.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self.batch_candidates(salt.len(), n, r, p, key_length)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(passwords.len())
            .map_err(|error| GpuError::new(format!("allocate candidate result set: {error}")))?;
        for chunk in passwords.chunks(batch) {
            result.extend(self.derive_batch(chunk, salt, n, r, p, key_length).await?);
        }
        Ok(result)
    }

    async fn derive_batch(
        &self,
        passwords: &[&[u8]],
        salt: &[u8],
        n: u32,
        r: u32,
        p: u32,
        key_length: usize,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        if passwords.is_empty() {
            return Ok(Vec::new());
        }
        let spec = KdfSpec::new(
            &self.romix,
            salt.len(),
            n,
            r,
            p,
            passwords.len(),
            key_length,
        )?;
        let key_input = kdf_input(passwords, salt)?;
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
            label: Some("challenge-solvers-scrypt-kdf-batched-passwords"),
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
        let candidate_bytes = usize::try_from(spec.key_bytes / u64::from(spec.candidates))
            .map_err(|_| GpuError::new("scrypt per-candidate output bytes overflow"))?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(passwords.len())
            .map_err(|error| GpuError::new(format!("allocate scrypt keys: {error}")))?;
        for candidate in data.chunks_exact(candidate_bytes) {
            let mut key = Vec::new();
            key.try_reserve_exact(spec.key_length)
                .map_err(|error| GpuError::new(format!("allocate scrypt key: {error}")))?;
            for word in candidate.as_chunks::<4>().0 {
                key.extend_from_slice(&u32::from_le_bytes(*word).to_be_bytes());
            }
            key.truncate(spec.key_length);
            results.push(key);
        }
        Ok(results)
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
        while offset < spec.total_lanes {
            let lanes = spec.lanes_per_batch.min(spec.total_lanes - offset);
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
    candidates: u32,
    total_lanes: u32,
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
        candidate_count: usize,
        key_length: usize,
    ) -> Result<Self, GpuError> {
        if p == 0 || r == 0 || u64::from(r) * u64::from(p) >= 0x4000_0000 {
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
        let candidates = u32::try_from(candidate_count)
            .map_err(|_| GpuError::new("scrypt candidate count exceeds GPU u32 range"))?;
        if candidates == 0 {
            return Err(GpuError::new("scrypt candidate count must be positive"));
        }
        let total_lanes = u64::from(candidates)
            .checked_mul(u64::from(p))
            .and_then(|lanes| u32::try_from(lanes).ok())
            .ok_or_else(|| GpuError::new("scrypt total p*passwords lanes exceed GPU range"))?;
        let b_bytes = u64::from(total_lanes)
            .checked_mul(u64::from(r))
            .and_then(|lanes| lanes.checked_mul(128))
            .ok_or_else(|| GpuError::new("scrypt B buffer length overflow"))?;
        let key_blocks = key_length.div_ceil(32);
        let key_bytes = u64::try_from(key_blocks)
            .ok()
            .and_then(|blocks| blocks.checked_mul(32))
            .and_then(|bytes| bytes.checked_mul(u64::from(candidates)))
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
        let initial_workgroups = u32::try_from((b_bytes / 32).div_ceil(64))
            .map_err(|_| GpuError::new("scrypt initial dispatch overflow"))?;
        let final_blocks = u64::try_from(key_blocks)
            .ok()
            .and_then(|n| n.checked_mul(u64::from(candidates)))
            .ok_or_else(|| GpuError::new("scrypt final digest blocks overflow"))?;
        let final_workgroups = u32::try_from(final_blocks.div_ceil(64))
            .map_err(|_| GpuError::new("scrypt final dispatch overflow"))?;
        if initial_workgroups > gpu.max_wgs || final_workgroups > gpu.max_wgs {
            return Err(GpuError::new("scrypt PBKDF2 exceeds device dispatch limit"));
        }
        let lanes_per_batch = u32::try_from(
            lanes_limit.min(
                usize::try_from(total_lanes)
                    .map_err(|_| GpuError::new("scrypt total lanes exceed host usize"))?,
            ),
        )
        .map_err(|_| GpuError::new("scrypt batch lanes exceed u32"))?;
        let temporary_bytes = u64::from(lanes_per_batch)
            .checked_mul(u64::from(r))
            .and_then(|v| v.checked_mul(128))
            .ok_or_else(|| GpuError::new("scrypt temporary buffer overflow"))?;
        let scratch_bytes = temporary_bytes
            .checked_mul(u64::from(n))
            .ok_or_else(|| GpuError::new("scrypt ROMix scratch buffer overflow"))?;
        Ok(Self {
            n,
            r,
            p,
            candidates,
            total_lanes,
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
        let key_words = usize::try_from(spec.candidates)
            .ok()
            .and_then(|n| n.checked_mul(16))
            .ok_or_else(|| GpuError::new("scrypt candidate HMAC state count overflow"))?;
        let salt_words = input
            .len()
            .checked_sub(key_words)
            .filter(|n| n.is_multiple_of(16))
            .ok_or_else(|| GpuError::new("scrypt malformed shared salt template"))?;
        let salt_blocks = u32::try_from(salt_words / 16)
            .map_err(|_| GpuError::new("scrypt padded salt block count overflow"))?;
        let params = [
            spec.n,
            spec.r,
            spec.p,
            spec.salt_len,
            salt_blocks,
            u32::try_from(spec.key_bytes / u64::from(spec.candidates) / 32)
                .map_err(|_| GpuError::new("scrypt per-candidate key blocks overflow"))?,
            u32::try_from(spec.b_bytes / u64::from(spec.candidates) / 4)
                .map_err(|_| GpuError::new("scrypt per-candidate B words overflow"))?,
            spec.candidates,
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

fn kdf_input(passwords: &[&[u8]], salt: &[u8]) -> Result<Vec<u32>, GpuError> {
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
    let words_needed = passwords
        .len()
        .checked_mul(16)
        .and_then(|n| n.checked_add(padded / 4))
        .ok_or_else(|| GpuError::new("scrypt HMAC chaining states buffer overflow"))?;
    let mut words = Vec::new();
    words
        .try_reserve_exact(words_needed)
        .map_err(|error| GpuError::new(format!("allocate HMAC states: {error}")))?;
    for password in passwords {
        let key: Vec<u8> = if password.len() > 64 {
            Sha256::digest(password).to_vec()
        } else {
            password.to_vec()
        };
        let mut padded_key = [0_u8; 64];
        padded_key[..key.len()].copy_from_slice(&key);
        words.extend_from_slice(&hmac_state(&padded_key, 0x36));
        words.extend_from_slice(&hmac_state(&padded_key, 0x5c));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(padded)
        .map_err(|err| GpuError::new(format!("reserve padded scrypt salt: {err}")))?;
    bytes.resize(padded, 0);
    bytes[..salt.len()].copy_from_slice(salt);
    bytes[salt.len() + 3] = 1;
    bytes[count] = 0x80;
    bytes[padded - 8..].copy_from_slice(&bit_len.to_be_bytes());
    for chunk in bytes.as_chunks::<4>().0 {
        words.push(u32::from_be_bytes(*chunk));
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
