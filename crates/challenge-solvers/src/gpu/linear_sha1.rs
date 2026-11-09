//! GPU search for Linear.pub's SHA-1 browser cookie proof.
//! The complete valid protocol domain (40-byte ASCII hexadecimal seed plus
//! decimal u32 counter) fits one SHA-1 block and requires only baseline WGPU.

use std::borrow::Cow;

use sha1::{Digest, Sha1};

use super::{GpuContext, GpuError, WG_SIZE, gpu_u32};

/// Full validated proof for a Linear.pub cookie challenge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearSha1Solution {
    /// Valid decimal counter.
    pub nonce: u32,
    /// Upper bound on the number of GPU candidate invocations dispatched.
    pub dispatched_hashes: u64,
}

/// Compute a Linear.pub proof on a general WGPU adapter.
pub struct LinearSha1GpuSolver {
    seed: [u8; 40],
    digest_offset: usize,
    start: u64,
    limit: u64,
    attempted: u64,
}

impl LinearSha1GpuSolver {
    /// Construct a solver for a protocol-valid 40-hex-digit ASCII seed.
    ///
    /// # Errors
    /// Rejects malformed seeds or an impossible digest offset.
    pub fn new(seed: &[u8]) -> Result<Self, GpuError> {
        let seed: [u8; 40] = seed
            .try_into()
            .map_err(|_| GpuError::new("Linear.pub seed must be 40 bytes"))?;
        if !seed.iter().all(u8::is_ascii_hexdigit) {
            return Err(GpuError::new("Linear.pub seed must contain ASCII hex"));
        }
        let digit = seed[0];
        let offset = match digit {
            b'0'..=b'9' => usize::from(digit - b'0'),
            b'a'..=b'f' => usize::from(digit - b'a') + 10,
            b'A'..=b'F' => usize::from(digit - b'A') + 10,
            _ => return Err(GpuError::new("invalid Linear.pub seed offset")),
        };
        Ok(Self {
            seed,
            digest_offset: offset,
            start: 0,
            limit: 1_u64 << 32,
            attempted: 0,
        })
    }

    /// Inclusive starting counter, allowing searches in arbitrary windows.
    pub const fn set_start_nonce(&mut self, start: u64) {
        self.start = start;
    }

    /// Exclusive counter limit. The protocol's decimal u32 domain is enforced.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of GPU candidate invocations dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted
    }

    fn verifies(&self, nonce: u32) -> bool {
        let mut hash = Sha1::new();
        hash.update(self.seed);
        hash.update(nonce.to_string().as_bytes());
        let digest = hash.finalize();
        digest[self.digest_offset] == 0xb0 && digest[self.digest_offset + 1] == 0x0b
    }

    async fn dispatch(
        &self,
        gpu: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        params: &wgpu::Buffer,
        base: u64,
        count: u32,
    ) -> Result<Option<u32>, GpuError> {
        let base_u32 = gpu_u32(base, "Linear.pub counter")?;
        let mut values = [0u32; 14];
        values[0] = base_u32;
        values[1] = count;
        values[2] = u32::try_from(self.digest_offset)
            .map_err(|_| GpuError::new("invalid Linear.pub digest offset"))?;
        for (idx, four) in self.seed.as_chunks::<4>().0.iter().enumerate() {
            values[4 + idx] = u32::from_be_bytes(*four);
        }
        gpu.queue
            .write_buffer(params, 0, bytemuck::cast_slice(&values));
        gpu.queue.write_buffer(&gpu.out_buf, 0, &[0; 16]);
        let bindings = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("linear-sha1-bind"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
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
                label: Some("linear-sha1-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear-sha1-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(count.div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&gpu.out_buf, 0, &gpu.staging, 0, 16);
        gpu.queue.submit([encoder.finish()]);
        let result = gpu.read_result().await?;
        if result[0] == 0 {
            return Ok(None);
        }
        let nonce = result[1];
        let end = base + u64::from(count);
        if u64::from(nonce) < base || u64::from(nonce) >= end {
            return Err(GpuError::new(
                "Linear.pub GPU returned counter outside batch",
            ));
        }
        if !self.verifies(nonce) {
            return Err(GpuError::new("Linear.pub GPU returned invalid SHA-1 proof"));
        }
        Ok(Some(nonce))
    }

    /// Dispatch search windows until the first GPU-valid proof is found.
    /// A parallel GPU can return any valid nonce within a window, not
    /// necessarily the smallest one.
    ///
    /// # Errors
    /// Reports an invalid seed, GPU submission failure, or a nonmatching proof.
    pub async fn solve(
        &mut self,
        gpu: &GpuContext,
    ) -> Result<Option<LinearSha1Solution>, GpuError> {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("linear-sha1"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/linear_sha1.wgsl"
                ))),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("linear-sha1"),
                layout: None,
                module: &shader,
                entry_point: Some("solve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear-sha1-params"),
            size: 14 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut base = self.start;
        let limit = self.limit.min(1_u64 << 32);
        let max_dispatch = u64::from(gpu.max_wgs) * u64::from(WG_SIZE);
        let target = 131_072_u64.min(max_dispatch);
        if target == 0 {
            return Err(GpuError::new("GPU cannot dispatch SHA-1 workgroups"));
        }
        while base < limit {
            let batch = (limit - base).min(target);
            let count = gpu_u32(batch, "Linear.pub batch count")?;
            let found = self.dispatch(gpu, &pipeline, &params, base, count).await?;
            self.attempted += batch;
            if let Some(nonce) = found {
                return Ok(Some(LinearSha1Solution {
                    nonce,
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

    const LIVE_SEED: &[u8; 40] = b"72BBE3B063367AB2F323D46D2ADA8A8A29BB5B58";

    #[test]
    fn seed_validation_and_rustcrypto_reference() -> Result<(), GpuError> {
        assert!(LinearSha1GpuSolver::new(b"bad").is_err());
        assert!(LinearSha1GpuSolver::new(b"Z2BBE3B063367AB2F323D46D2ADA8A8A29BB5B58").is_err());
        let solver = LinearSha1GpuSolver::new(LIVE_SEED)?;
        assert!(solver.verifies(29_224));
        assert!(!solver.verifies(29_223));
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical Vulkan/WGPU adapter"]
    fn linear_sha1_gpu_all_offsets_and_counter_boundaries() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        for offset in 0..16u8 {
            let seed = format!("{offset:X}{}", "0".repeat(39));
            let mut solver = LinearSha1GpuSolver::new(seed.as_bytes())?;
            solver.set_limit(500_000);
            let result = super::super::sha512_tests::block_on(solver.solve(&gpu))?
                .expect("every seed should have SHA1 16-bit match within 500k candidates");
            assert!(solver.verifies(result.nonce), "digest byte offset {offset}");
            assert!(result.nonce < 500_000);
            println!(
                "Linear.pub SHA-1 GPU offset {offset}: nonce={}",
                result.nonce
            );
        }
        let mut live = LinearSha1GpuSolver::new(LIVE_SEED)?;
        live.set_limit(500_000);
        let proof = super::super::sha512_tests::block_on(live.solve(&gpu))?
            .expect("captured seed should solve");
        assert!(live.verifies(proof.nonce));

        for start in [999_900_000_u64, 3_999_000_000] {
            let mut solver = LinearSha1GpuSolver::new(LIVE_SEED)?;
            solver.set_start_nonce(start);
            solver.set_limit(start + 600_000);
            let result = super::super::sha512_tests::block_on(solver.solve(&gpu))?
                .expect("long decimal candidate window should contain a match");
            assert!(u64::from(result.nonce) >= start);
            assert!(solver.verifies(result.nonce));
            println!(
                "Linear.pub SHA-1 GPU high-counter {} -> {}",
                start, result.nonce
            );
        }
        let mut empty = LinearSha1GpuSolver::new(LIVE_SEED)?;
        empty.set_limit(1);
        assert!(super::super::sha512_tests::block_on(empty.solve(&gpu))?.is_none());
        assert_eq!(empty.attempted_nonces(), 1);
        Ok(())
    }
}
