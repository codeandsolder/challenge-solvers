//! Native-u64 GPU ALTCHA SHA-384/SHA-512 and PBKDF2-HMAC-SHA-384/SHA-512.
//! The shader processes candidate passwords on GPU, including arbitrary binary
//! salts/nonces and multi-block PBKDF2 prefixes. Independently verify the
//! winning candidate on CPU before returning a derived key.

use core::num::NonZeroU32;
use std::{borrow::Cow, vec::Vec};

use sha2::{Digest, Sha384, Sha512};

use super::{
    GpuContext, GpuError, WG_SIZE, gpu_batch_target, gpu_u32, gpu_usize_u32, high_u32, low_u32,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Supported 64-bit SHA-2 ALTCHA derivation modes.
pub enum AltchaSha64Algorithm {
    /// Iterated SHA-384; truncate only the final digest.
    Sha384,
    /// Iterated SHA-512; truncate only the final digest.
    Sha512,
    /// PBKDF2-HMAC-SHA-384, arbitrary positive derived key length.
    Pbkdf2Sha384,
    /// PBKDF2-HMAC-SHA-512, arbitrary positive derived key length.
    Pbkdf2Sha512,
}

impl AltchaSha64Algorithm {
    const fn pbkdf2(self) -> bool {
        matches!(self, Self::Pbkdf2Sha384 | Self::Pbkdf2Sha512)
    }

    const fn digest_width(self) -> usize {
        match self {
            Self::Sha384 | Self::Pbkdf2Sha384 => 48,
            Self::Sha512 | Self::Pbkdf2Sha512 => 64,
        }
    }

    const fn sha512(self) -> bool {
        matches!(self, Self::Sha512 | Self::Pbkdf2Sha512)
    }

    const fn initial_state(self) -> [u64; 8] {
        if self.sha512() {
            [
                0x6a09_e667_f3bc_c908,
                0xbb67_ae85_84ca_a73b,
                0x3c6e_f372_fe94_f82b,
                0xa54f_f53a_5f1d_36f1,
                0x510e_527f_ade6_82d1,
                0x9b05_688c_2b3e_6c1f,
                0x1f83_d9ab_fb41_bd6b,
                0x5be0_cd19_137e_2179,
            ]
        } else {
            [
                0xcbbb_9d5d_c105_9ed8,
                0x629a_292a_367c_d507,
                0x9159_015a_3070_dd17,
                0x152f_ecd8_f70e_5939,
                0x6733_2667_ffc0_0b31,
                0x8eb4_4a87_6858_1511,
                0xdb0c_2e0d_64f9_8fa7,
                0x47b5_481d_befa_4fa4,
            ]
        }
    }
}

/// Correct and independently verified ALTCHA SHA-384/512 GPU solution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AltchaSha64Solution {
    /// Found candidate nonce (u32 counter or JavaScript-safe decimal).
    pub nonce: u64,
    /// Full derived key, independently recomputed by `RustCrypto`.
    pub derived_key: Vec<u8>,
    /// Upper bound on GPU candidate invocations across dispatched batches.
    pub dispatched_hashes: u64,
}

/// General native-int64 ALTCHA GPU nonce solver.
pub struct AltchaSha64GpuSolver {
    algorithm: AltchaSha64Algorithm,
    salt: Vec<u8>,
    nonce: Vec<u8>,
    cost: NonZeroU32,
    key_length: usize,
    prefix_nibbles: Vec<u8>,
    decimal_counter: bool,
    start_nonce: u64,
    limit: u64,
    attempted: u64,
}

impl AltchaSha64GpuSolver {
    /// Construct arbitrary-salt, arbitrary-nonce SHA-384/512 or PBKDF2 solver.
    ///
    /// # Errors
    /// Returns an error for invalid key lengths, overflowed KDF limits,
    /// or messages that cannot fit a GPU storage/length ABI.
    pub fn new(
        algorithm: AltchaSha64Algorithm,
        salt: &[u8],
        nonce: &[u8],
        cost: NonZeroU32,
        key_length: usize,
    ) -> Result<Self, GpuError> {
        let width = algorithm.digest_width();
        let valid = if algorithm.pbkdf2() {
            u64::try_from(key_length)
                .is_ok_and(|n| n > 0 && n <= u64::from(u32::MAX) * width as u64)
        } else {
            key_length > 0 && key_length <= width
        };
        if !valid {
            return Err(GpuError::new(
                "invalid ALTCHA SHA384/512 derived key length",
            ));
        }
        let first_len = if algorithm.pbkdf2() {
            nonce.len()
        } else {
            salt.len()
                .checked_add(nonce.len())
                .ok_or_else(|| GpuError::new("ALTCHA salt/nonce length overflow"))?
        };
        if first_len > u32::MAX as usize - 160 || salt.len() > u32::MAX as usize - 160 {
            return Err(GpuError::new("ALTCHA message size exceeds GPU ABI"));
        }
        Ok(Self {
            algorithm,
            salt: salt.to_vec(),
            nonce: nonce.to_vec(),
            cost,
            key_length,
            prefix_nibbles: Vec::new(),
            decimal_counter: false,
            start_nonce: 0,
            limit: u64::from(u32::MAX) + 1,
            attempted: 0,
        })
    }

    /// Require an exact derived-key hexadecimal prefix, including odd nibbles.
    ///
    /// # Errors
    /// Rejects invalid nibbles or a prefix longer than the derived key.
    pub fn set_prefix_nibbles(&mut self, prefix: &[u8]) -> Result<(), GpuError> {
        if prefix.iter().any(|n| *n > 15) || prefix.len() > self.key_length.saturating_mul(2) {
            return Err(GpuError::new("invalid ALTCHA SHA384/512 prefix"));
        }
        self.prefix_nibbles = prefix.to_vec();
        Ok(())
    }

    /// Use ALTCHA v2 decimal-string nonce encoding (0..=2^53-1).
    pub const fn set_string_counter(&mut self) {
        self.decimal_counter = true;
        self.limit = 1_u64 << 53;
    }

    /// Set the inclusive nonce search start.
    pub const fn set_start_nonce(&mut self, nonce: u64) {
        self.start_nonce = nonce;
    }

    /// Set the exclusive nonce search limit.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched through this solver.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted
    }

    fn password(&self, counter: u64) -> Vec<u8> {
        let mut value = Vec::with_capacity(self.nonce.len() + 20);
        value.extend_from_slice(&self.nonce);
        if self.decimal_counter {
            value.extend_from_slice(counter.to_string().as_bytes());
        } else {
            let encoded = counter.to_be_bytes();
            value.extend_from_slice(&encoded[4..]);
        }
        value
    }

    fn derive(&self, counter: u64) -> Vec<u8> {
        let password = self.password(counter);
        let mut result = vec![0u8; self.key_length];
        if self.algorithm.pbkdf2() {
            if self.algorithm.sha512() {
                pbkdf2::pbkdf2_hmac::<Sha512>(&password, &self.salt, self.cost.get(), &mut result);
            } else {
                pbkdf2::pbkdf2_hmac::<Sha384>(&password, &self.salt, self.cost.get(), &mut result);
            }
            return result;
        }
        let mut message = Vec::with_capacity(self.salt.len() + password.len());
        message.extend_from_slice(&self.salt);
        message.extend_from_slice(&password);
        let mut digest = if self.algorithm.sha512() {
            Sha512::digest(&message).to_vec()
        } else {
            Sha384::digest(&message).to_vec()
        };
        for _ in 1..self.cost.get() {
            digest = if self.algorithm.sha512() {
                Sha512::digest(&digest).to_vec()
            } else {
                Sha384::digest(&digest).to_vec()
            };
        }
        digest.truncate(self.key_length);
        digest
    }

    fn prefix_matches(&self, derived: &[u8]) -> bool {
        self.prefix_nibbles.iter().enumerate().all(|(i, expected)| {
            let b = derived[i / 2];
            (if i % 2 == 0 { b >> 4 } else { b & 15 }) == *expected
        })
    }

    fn pack_be(bytes: &[u8], dest: &mut Vec<u32>) {
        for chunk in bytes.chunks(4) {
            let mut block = [0u8; 4];
            block[..chunk.len()].copy_from_slice(chunk);
            dest.push(u32::from_be_bytes(block));
        }
    }

    fn params(&self, base: u64, batch: u32) -> Result<Vec<u32>, GpuError> {
        let data: Cow<'_, [u8]> = if self.algorithm.pbkdf2() {
            Cow::Borrowed(&self.nonce)
        } else {
            let mut bytes = Vec::with_capacity(self.salt.len() + self.nonce.len());
            bytes.extend_from_slice(&self.salt);
            bytes.extend_from_slice(&self.nonce);
            Cow::Owned(bytes)
        };
        let mut p = vec![0_u32; 32];
        p[0] = u32::from(self.algorithm.sha512());
        p[1] = low_u32(base);
        p[2] = batch;
        p[3] = high_u32(base);
        p[4] = self.cost.get();
        p[5] = gpu_usize_u32(data.len(), "SHA512 nonce/prefix length")?;
        p[6] = gpu_usize_u32(self.prefix_nibbles.len(), "SHA512 prefix nibble length")?;
        p[7] = u32::from(self.decimal_counter);
        p[8] = gpu_usize_u32(self.salt.len(), "SHA512 salt length")?;
        p[9] = gpu_usize_u32(self.key_length, "SHA512 derived length")?;
        p[10] = 32;
        Self::pack_be(&data, &mut p);
        p[11] = gpu_usize_u32(p.len(), "SHA512 nibble offset")?;
        for chunk in self.prefix_nibbles.chunks(8) {
            let mut word = 0u32;
            for (i, n) in chunk.iter().enumerate() {
                word |= u32::from(*n) << ((7 - i) * 4);
            }
            p.push(word);
        }
        p[12] = gpu_usize_u32(p.len(), "SHA512 salt offset")?;
        if self.algorithm.pbkdf2() {
            Self::pack_be(&self.salt, &mut p);
        }
        p[13] = u32::from(self.algorithm.pbkdf2());
        for (i, word) in self.algorithm.initial_state().iter().enumerate() {
            p[16 + 2 * i] = high_u32(*word);
            p[17 + 2 * i] = low_u32(*word);
        }
        Ok(p)
    }

    async fn dispatch_batch(
        &self,
        gpu: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        base: u64,
        batch: u64,
    ) -> Result<Option<u64>, GpuError> {
        let count = gpu_u32(batch, "SHA512 batch size")?;
        let params = self.params(base, count)?;
        let bytes = u64::try_from(params.len())
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| GpuError::new("SHA512 GPU input buffer overflow"))?;
        let limits = gpu.device.limits();
        if bytes > limits.max_storage_buffer_binding_size || bytes > limits.max_buffer_size {
            return Err(GpuError::new(
                "ALTCHA SHA512 input exceeds GPU storage limits",
            ));
        }
        let input = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("altcha-sha512-params"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&input, 0, bytemuck::cast_slice(&params));
        gpu.queue.write_buffer(&gpu.out_buf, 0, &[0u8; 16]);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("altcha-sha512-bind"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.out_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("altcha-sha512-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("altcha-sha512-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &gpu.staging, 0, 16);
        gpu.queue.submit([encoder.finish()]);
        let result = gpu.read_result().await?;
        if result[0] == 0 {
            return Ok(None);
        }
        let nonce = (u64::from(result[2]) << 32) | u64::from(result[1]);
        if nonce < base || nonce >= base + batch {
            return Err(GpuError::new("GPU SHA512 returned out-of-batch nonce"));
        }
        Ok(Some(nonce))
    }

    /// Run the complete GPU candidate search with independent CPU proof check.
    ///
    /// # Errors
    /// Returns an error if native shader-int64 is unsupported, the GPU fails,
    /// or the winning nonce fails the full `RustCrypto` derived-key oracle.
    pub async fn solve(
        &mut self,
        gpu: &GpuContext,
    ) -> Result<Option<AltchaSha64Solution>, GpuError> {
        if !gpu.device.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(GpuError::new(
                "selected GPU device has no SHADER_INT64 feature",
            ));
        }
        let pipeline = gpu.altcha_sha64_pipeline.get_or_init(|| {
            let source = format!(
                "{}\n{}",
                include_str!("../shaders/sha512_core.wgsl"),
                include_str!("../shaders/altcha_sha512.wgsl")
                    .replace("__WG__", &WG_SIZE.to_string())
            );
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("altcha-sha512"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("altcha-sha512"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("solve"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        });
        let max_domain = if self.decimal_counter {
            1_u64 << 53
        } else {
            u64::from(u32::MAX) + 1
        };
        let limit = self.limit.min(max_domain);
        let desired = gpu_batch_target(
            1_u64
                .checked_shl(
                    u32::try_from(self.prefix_nibbles.len().saturating_mul(4)).unwrap_or(64),
                )
                .unwrap_or(u64::MAX),
            u64::from(WG_SIZE),
        );
        let max_batch = u64::from(gpu.max_wgs) * u64::from(WG_SIZE);
        let mut base = self.start_nonce;
        while base < limit {
            let batch = (limit - base)
                .min(max_batch)
                .min(desired)
                .min(u64::from(u32::MAX));
            let found = self.dispatch_batch(gpu, pipeline, base, batch).await?;
            self.attempted = self.attempted.saturating_add(batch);
            if let Some(nonce) = found {
                let derived = self.derive(nonce);
                if !self.prefix_matches(&derived) {
                    return Err(GpuError::new(
                        "GPU SHA512 answer failed independent CPU verification",
                    ));
                }
                return Ok(Some(AltchaSha64Solution {
                    nonce,
                    derived_key: derived,
                    dispatched_hashes: self.attempted,
                }));
            }
            base += batch;
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix_of(key: &[u8], nibbles: usize) -> Vec<u8> {
        (0..nibbles)
            .map(|i| {
                if i % 2 == 0 {
                    key[i / 2] >> 4
                } else {
                    key[i / 2] & 15
                }
            })
            .collect()
    }

    #[test]
    fn sha384_sha512_cpu_oracles_cover_all_altcha_modes() -> Result<(), GpuError> {
        for algorithm in [
            AltchaSha64Algorithm::Sha384,
            AltchaSha64Algorithm::Sha512,
            AltchaSha64Algorithm::Pbkdf2Sha384,
            AltchaSha64Algorithm::Pbkdf2Sha512,
        ] {
            let salt = (0..137u16).map(|i| i as u8).collect::<Vec<_>>();
            let nonce = (0..151u16).map(|i| (i * 3) as u8).collect::<Vec<_>>();
            let width = algorithm.digest_width();
            let keylen = if algorithm.pbkdf2() {
                width * 2 + 5
            } else {
                width - 3
            };
            let mut solver = AltchaSha64GpuSolver::new(
                algorithm,
                &salt,
                &nonce,
                NonZeroU32::new(3).expect("positive cost"),
                keylen,
            )?;
            let counter = 149_u64;
            let key = solver.derive(counter);
            assert_eq!(key.len(), keylen);
            let prefix = prefix_of(&key, keylen * 2 - 1);
            solver.set_prefix_nibbles(&prefix)?;
            assert!(solver.prefix_matches(&key));
            solver.set_string_counter();
            let different = solver.derive(counter);
            assert_ne!(different, key);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires physical Vulkan GPU with SHADER_INT64"]
    fn sha384_sha512_pbkdf2_gpu_all_modes_with_reference() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        assert!(gpu.altcha_sha64_pipeline.get().is_none());
        assert!(
            gpu.device.features().contains(wgpu::Features::SHADER_INT64),
            "real GPU does not support SHADER_INT64"
        );
        let cases = [
            (
                AltchaSha64Algorithm::Sha384,
                0usize,
                0usize,
                48usize,
                2u32,
                false,
                13u64,
            ),
            (AltchaSha64Algorithm::Sha512, 64, 128, 64, 3, false, 13),
            (
                AltchaSha64Algorithm::Sha384,
                191,
                160,
                37,
                1,
                true,
                (1u64 << 32) + 13,
            ),
            (AltchaSha64Algorithm::Sha512, 131, 13, 44, 4, true, 17),
            (
                AltchaSha64Algorithm::Pbkdf2Sha384,
                127,
                160,
                130,
                2,
                false,
                13,
            ),
            (
                AltchaSha64Algorithm::Pbkdf2Sha512,
                112,
                128,
                193,
                3,
                true,
                (1u64 << 32) + 13,
            ),
            (AltchaSha64Algorithm::Pbkdf2Sha384, 0, 0, 1, 1, false, 0),
            (AltchaSha64Algorithm::Pbkdf2Sha512, 192, 0, 64, 2, false, 13),
        ];
        let mut original_pipeline = None;
        for (algorithm, salt_len, nonce_len, key_len, cost, string_counter, counter) in cases {
            let salt = (0..salt_len)
                .map(|i| ((i * 7 + 13) & 255) as u8)
                .collect::<Vec<_>>();
            let nonce = (0..nonce_len)
                .map(|i| ((i * 11 + 21) & 255) as u8)
                .collect::<Vec<_>>();
            let mut solver = AltchaSha64GpuSolver::new(
                algorithm,
                &salt,
                &nonce,
                NonZeroU32::new(cost).expect("positive iteration count"),
                key_len,
            )?;
            if string_counter {
                solver.set_string_counter();
            }
            let reference = solver.derive(counter);
            let prefix = prefix_of(&reference, key_len.saturating_mul(2).min(257));
            solver.set_prefix_nibbles(&prefix)?;
            solver.set_start_nonce(counter);
            solver.set_limit(counter + 1);
            let proof = super::super::sha512_tests::block_on(solver.solve(&gpu))?
                .expect("GPU target nonce is in the one-candidate search domain");
            assert_eq!(proof.nonce, counter, "algorithm {algorithm:?}");
            assert_eq!(proof.derived_key, reference, "algorithm {algorithm:?}");
            assert_eq!(proof.dispatched_hashes, 1);
            let current = gpu
                .altcha_sha64_pipeline
                .get()
                .expect("ALTCHA SHA64 WGPU pipeline must be initialized");
            if let Some(first) = original_pipeline {
                assert!(
                    std::ptr::eq(first, current),
                    "SHA64 GPU pipeline must persist across modes and challenges"
                );
            } else {
                original_pipeline = Some(current);
            }
            println!(
                "verified {algorithm:?} salt={salt_len} nonce={nonce_len} key={key_len} cost={cost} decimal={string_counter}"
            );
        }
        Ok(())
    }
}
