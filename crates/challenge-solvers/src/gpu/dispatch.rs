//! GPU work submission, result readback and public solve entry points.
//!
//! The public context is defined in the parent module; this module only
//! implements its steady-state GPU methods.

use super::{
    COUNTER_SPACE, GpuContext, GpuError, GpuSolution, HASHES_PER_WG, Params, WG_SIZE,
    counter_to_decimal_nonce, expected_work, gpu_batch_target, gpu_u32, gpu_usize_u32, high_u32,
    low_u32, midstate, single_block_fast_compatible, single_block_search_space, verify,
    verify_fast_single_block, verify_single_block,
};
use core::num::NonZeroU8;
use std::vec::Vec;

impl GpuContext {
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

    /// Compile all optional general-WGPU protocol shaders on this persistent device.
    ///
    /// General SHA-256/BLAKE3 pipelines already live for the full context lifetime.
    /// These optional SHA-1 and native-u64 pipelines are normally lazy; invoking
    /// this once during service startup prevents shader/pipeline compilation on
    /// the first protocol request. The device driver may still defer lower-level
    /// execution specialization until first dispatch.
    ///
    /// Unsupported native-u64 pipelines remain absent rather than requesting
    /// a feature that the negotiated adapter does not provide.
    pub fn preload_protocol_pipelines(&self) {
        let _ = self.cached_linear_sha1_pipeline();
        let _ = self.cached_sha1_decimal_pipeline();
        if self
            .device
            .features()
            .contains(wgpu::Features::SHADER_INT64)
        {
            let _ = self.cached_altcha_sha64_pipeline();
            let _ = self.cached_friendly_blake2b_pipeline();
        }
    }

    /// Prepare optional protocol pipelines and exercise the Friendly `BLAKE2b`
    /// compute kernel once, so the driver compiles its actual executable
    /// before the first real Friendly challenge arrives.
    ///
    /// This is intended for the single general GPU worker at service startup,
    /// before any regular GPU jobs are queued.
    ///
    /// # Errors
    /// Propagates GPU submission or readback failure during shader warm-up.
    pub async fn preload_and_warm_protocol_pipelines(&self) -> Result<(), GpuError> {
        self.preload_protocol_pipelines();
        if self
            .device
            .features()
            .contains(wgpu::Features::SHADER_INT64)
        {
            // Threshold zero ensures a no-result warmup; it still exercises
            // the actual hash shader with a complete 64-lane workgroup.
            let solver = super::FriendlyBlake2bGpu::new(self)?;
            let found = solver.scan(&[0u8; 128], 0, 0, WG_SIZE, 0).await?;
            if found.is_some() {
                return Err(GpuError::new(
                    "Friendly shader warm-up returned unexpected proof",
                ));
            }
        }
        Ok(())
    }

    /// Borrow the WGPU device negotiated for this context.
    ///
    /// Other independent solver pipelines can use the same logical device,
    /// retaining their own buffers while avoiding a second device request.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the queue associated with this GPU device.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Maximum workgroups per dispatch dimension supported by the adapter.
    #[must_use]
    pub const fn max_workgroups_per_dimension(&self) -> u32 {
        self.max_wgs
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

    pub(super) async fn solve_single_block_inner<const TYPE: u8>(
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

    pub(super) async fn dispatch_multi(
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

    pub(super) async fn dispatch_cerberus(
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

    pub(super) async fn dispatch_altcha(
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

    pub(super) async fn read_result(&self) -> Result<Vec<u32>, GpuError> {
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
