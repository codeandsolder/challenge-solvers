//! General GPU search for SHA-1(UTF-8/byte prefix || decimal u32 counter).
//!
//! RustCrypto compresses complete fixed-prefix blocks once on the host;
//! every candidate on GPU finishes at most two SHA-1 blocks. Arbitrary prefix
//! bytes and full exact 160-bit digest matching are supported.

use std::borrow::Cow;

use sha1::{Digest, Sha1, block_api::compress};

use super::{GpuContext, GpuError, WG_SIZE, gpu_u32};

const INITIAL: [u32; 5] = [
    0x6745_2301,
    0xefcd_ab89,
    0x98ba_dcfe,
    0x1032_5476,
    0xc3d2_e1f0,
];

/// GPU SHA-1 exact-digest proof for a decimal uint32 counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sha1DecimalSolution {
    /// Counter whose SHA-1 digest equals the complete requested digest.
    pub nonce: u32,
    /// Number of GPU candidate invocations dispatched.
    pub dispatched_hashes: u64,
}

/// Complete-domain SHA-1 exact-digest GPU search, independent of protocol.
pub struct Sha1DecimalGpuSolver {
    prefix: Vec<u8>,
    target: [u8; 20],
    initial_state: [u32; 5],
    tail: [u8; 64],
    tail_len: usize,
    prefix_len: u64,
    start: u64,
    limit: u64,
    attempted: u64,
}

impl Sha1DecimalGpuSolver {
    /// Precompute full SHA-1 prefix blocks once, retaining at most 63 bytes
    /// for GPU candidate-specific final blocks.
    ///
    /// # Errors
    /// Rejects SHA-1 messages that would overflow the 64-bit bit length.
    pub fn new(prefix: &[u8], target: [u8; 20]) -> Result<Self, GpuError> {
        let prefix_len = u64::try_from(prefix.len())
            .map_err(|_| GpuError::new("SHA-1 fixed prefix does not fit u64"))?;
        if prefix_len > u64::MAX / 8 - 10 {
            return Err(GpuError::new(
                "SHA-1 prefix plus u32 nonce exceeds bit length",
            ));
        }
        let mut initial_state = INITIAL;
        let (blocks, rest) = prefix.as_chunks::<64>();
        compress(&mut initial_state, blocks);
        let mut tail = [0u8; 64];
        tail[..rest.len()].copy_from_slice(rest);
        Ok(Self {
            prefix: prefix.to_vec(),
            target,
            initial_state,
            tail,
            tail_len: rest.len(),
            prefix_len,
            start: 0,
            limit: 1u64 << 32,
            attempted: 0,
        })
    }

    /// Inclusive start of candidate domain.
    pub const fn set_start_nonce(&mut self, start: u64) {
        self.start = start;
    }

    /// Exclusive end of candidate domain.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Total candidate invocations submitted.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted
    }

    fn verifies(&self, nonce: u32) -> bool {
        let mut hash = Sha1::new();
        hash.update(&self.prefix);
        hash.update(nonce.to_string().as_bytes());
        hash.finalize().as_slice() == self.target
    }

    fn params(&self, base: u32, count: u32) -> Result<[u32; 31], GpuError> {
        let mut params = [0u32; 31];
        params[0] = base;
        params[1] = count;
        params[2] = gpu_u32(self.tail_len as u64, "SHA1 prefix tail length")?;
        params[3] = self.prefix_len as u32;
        params[4] = (self.prefix_len >> 32) as u32;
        params[5..10].copy_from_slice(&self.initial_state);
        for (index, bytes) in self.tail.as_chunks::<4>().0.iter().enumerate() {
            params[10 + index] = u32::from_be_bytes(*bytes);
        }
        for (index, bytes) in self.target.as_chunks::<4>().0.iter().enumerate() {
            params[26 + index] = u32::from_be_bytes(*bytes);
        }
        Ok(params)
    }

    async fn dispatch(
        &self,
        gpu: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        input: &wgpu::Buffer,
        base: u64,
        count: u32,
    ) -> Result<Option<u32>, GpuError> {
        let start = gpu_u32(base, "SHA1 decimal start")?;
        let params = self.params(start, count)?;
        gpu.queue
            .write_buffer(input, 0, bytemuck::cast_slice(&params));
        gpu.queue.write_buffer(&gpu.out_buf, 0, &[0; 16]);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sha1-decimal-bind"),
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
                label: Some("sha1-decimal-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sha1-decimal"),
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
        let candidate = result[1];
        if u64::from(candidate) < base || u64::from(candidate) >= base + u64::from(count) {
            return Err(GpuError::new("SHA1 GPU counter outside dispatched batch"));
        }
        if !self.verifies(candidate) {
            return Err(GpuError::new(
                "SHA1 GPU candidate failed independent RustCrypto digest",
            ));
        }
        Ok(Some(candidate))
    }

    /// Search across the complete u32 decimal-counter domain, or a caller
    /// provided subwindow, comparing the full SHA-1 digest exactly.
    ///
    /// # Errors
    /// Reports GPU compilation/submission/readback failure or invalid result.
    pub async fn solve(
        &mut self,
        gpu: &GpuContext,
    ) -> Result<Option<Sha1DecimalSolution>, GpuError> {
        let pipeline = gpu.sha1_decimal_pipeline.get_or_init(|| {
            let source = format!(
                "{}\n{}",
                include_str!("../shaders/sha1_core.wgsl"),
                include_str!("../shaders/sha1_decimal.wgsl"),
            );
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("sha1-decimal"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("sha1-decimal"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("solve"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        });
        let input = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sha1-decimal-params"),
            size: 31 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let limit = self.limit.min(1_u64 << 32);
        let batch_limit = (u64::from(gpu.max_wgs) * u64::from(WG_SIZE)).min(131_072);
        if batch_limit == 0 {
            return Err(GpuError::new("SHA1 GPU cannot dispatch workgroups"));
        }
        let mut base = self.start;
        while base < limit {
            // u32 nonce addition in shader must never cross rollover.
            let count = gpu_u32(batch_limit.min(limit - base), "SHA1 decimal batch")?;
            let candidate = self.dispatch(gpu, pipeline, &input, base, count).await?;
            self.attempted += u64::from(count);
            if let Some(nonce) = candidate {
                return Ok(Some(Sha1DecimalSolution {
                    nonce,
                    dispatched_hashes: self.attempted,
                }));
            }
            base += u64::from(count);
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_compression_and_cpu_verifier_cover_boundaries() -> Result<(), GpuError> {
        for length in [0, 1, 40, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 4096] {
            let bytes = (0..length)
                .map(|i| (i * 73 % 251) as u8)
                .collect::<Vec<_>>();
            let nonce = 4294967295u32;
            let mut hash = Sha1::new();
            hash.update(&bytes);
            hash.update(nonce.to_string().as_bytes());
            let target: [u8; 20] = hash.finalize().into();
            let solver = Sha1DecimalGpuSolver::new(&bytes, target)?;
            assert_eq!(solver.tail_len, length % 64);
            assert!(solver.verifies(nonce));
            assert!(!solver.verifies(nonce - 1));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical Vulkan/WGPU adapter"]
    fn sha1_decimal_gpu_arbitrary_prefix_block_boundaries() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        assert!(gpu.sha1_decimal_pipeline.get().is_none());
        let mut pipeline_original = None;
        for length in [
            0, 1, 10, 40, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 4096,
        ] {
            let prefix = (0..length)
                .map(|i| (i * 73 % 251) as u8)
                .collect::<Vec<_>>();
            for target in [0_u32, 17, 99, 1_000, u32::MAX] {
                let mut h = Sha1::new();
                h.update(&prefix);
                h.update(target.to_string().as_bytes());
                let digest: [u8; 20] = h.finalize().into();
                let mut solver = Sha1DecimalGpuSolver::new(&prefix, digest)?;
                solver.set_start_nonce(u64::from(target));
                solver.set_limit(u64::from(target) + 1);
                let proof = super::super::sha512_tests::block_on(solver.solve(&gpu))?
                    .expect("the sole nonce must match complete SHA1 digest");
                assert_eq!(proof.nonce, target);
                assert_eq!(solver.attempted_nonces(), 1);
                let pipeline = gpu
                    .sha1_decimal_pipeline
                    .get()
                    .expect("shader pipeline initialized");
                if let Some(first) = pipeline_original {
                    assert!(std::ptr::eq(first, pipeline));
                } else {
                    pipeline_original = Some(pipeline);
                }
            }
            println!("SHA1 decimal GPU prefix length {length} all boundary nonces PASS");
        }
        let mut none = Sha1DecimalGpuSolver::new(b"abc", [0; 20])?;
        none.set_limit(16);
        assert!(super::super::sha512_tests::block_on(none.solve(&gpu))?.is_none());
        Ok(())
    }
}
