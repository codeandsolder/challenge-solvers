//! Experimental fixed-block BLAKE2b-256 GPU primitive for Friendly Captcha v1.
//! The producer passes the exact 128-byte padded v1 message; byte 120 is the
//! independently searchable bank and bytes 124..128 the LE u32 nonce.

use super::{GpuContext, GpuError, WG_SIZE};
use std::borrow::Cow;

/// Initialized native-u64 BLAKE2b-256 pipeline for repeated GPU searches.
pub struct FriendlyBlake2bGpu<'a> {
    gpu: &'a GpuContext,
    pipeline: &'a wgpu::ComputePipeline,
    input: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl<'a> FriendlyBlake2bGpu<'a> {
    /// Create the native-u64 pipeline once for subsequent scans.
    ///
    /// # Errors
    /// Returns a capability error if the GPU lacks 64-bit integer operations.
    pub fn new(gpu: &'a GpuContext) -> Result<Self, GpuError> {
        if !gpu.device.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(GpuError::new("Friendly BLAKE2b requires GPU SHADER_INT64"));
        }
        let pipeline = gpu.cached_friendly_blake2b_pipeline();
        let input = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-input"),
            size: 35 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("friendly-blake2b-bind"),
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
        Ok(Self {
            gpu,
            pipeline,
            input,
            bind,
        })
    }

    /// Maximum candidates accepted in one GPU dispatch.
    #[must_use]
    pub fn batch_capacity(&self) -> u64 {
        u64::from(self.gpu.max_wgs) * u64::from(WG_SIZE)
    }

    /// Evaluate an exact fixed-size work window from one solution bank.
    /// Returned nonce is any matching candidate, not necessarily the lowest.
    /// Threshold zero runs a fixed-work no-solution benchmark.
    ///
    /// # Errors
    /// Rejects invalid nonce ranges and propagates device execution errors.
    pub async fn scan(
        &self,
        padded: &[u8; 128],
        bank: u8,
        base: u32,
        count: u32,
        threshold: u32,
    ) -> Result<Option<u32>, GpuError> {
        let domain_end = u64::from(base) + u64::from(count);
        let max_batch = u64::from(self.gpu.max_wgs) * u64::from(WG_SIZE);
        if count == 0 || domain_end > u64::from(u32::MAX) + 1 || u64::from(count) > max_batch {
            return Err(GpuError::new(
                "Friendly candidate range exceeds GPU or u32 domain",
            ));
        }
        let mut packed = [0u32; 35];
        for (slot, bytes) in packed[..32].iter_mut().zip(padded.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*bytes);
        }
        packed[30] = (packed[30] & !0xff) | u32::from(bank);
        packed[32] = base;
        packed[33] = count;
        packed[34] = threshold;
        self.gpu
            .queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&packed));
        self.gpu.queue.write_buffer(&self.gpu.out_buf, 0, &[0; 16]);
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("friendly-blake2b-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("friendly-blake2b-search"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(WG_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.gpu.out_buf, 0, &self.gpu.staging, 0, 16);
        self.gpu.queue.submit([encoder.finish()]);
        let result = self.gpu.read_result().await?;
        if result[0] == 0 {
            return Ok(None);
        }
        let candidate = result[1];
        if u64::from(candidate) < u64::from(base) || u64::from(candidate) >= domain_end {
            return Err(GpuError::new("Friendly GPU nonce outside dispatched batch"));
        }
        Ok(Some(candidate))
    }
}

impl GpuContext {
    /// Compile this optional protocol pipeline once and retain it on the GPU context.
    pub(super) fn cached_friendly_blake2b_pipeline(&self) -> &wgpu::ComputePipeline {
        self.friendly_blake2b_pipeline.get_or_init(|| {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("friendly-blake2b-256"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                        "../shaders/friendly_blake2b.wgsl"
                    ))),
                });
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("friendly-blake2b-256"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("solve"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake2::{Blake2b256, Digest};
    use std::{hint::black_box, time::Instant};

    fn puzzle_input() -> [u8; 128] {
        let mut input = [0u8; 128];
        for (i, byte) in input[..64].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        input[12] = 1;
        input[14] = 16;
        input[15] = 128;
        input
    }

    fn cpu_matches(padded: &[u8; 128], bank: u8, nonce: u32, threshold: u32) -> bool {
        let mut candidate = *padded;
        candidate[120] = bank;
        candidate[124..128].copy_from_slice(&nonce.to_le_bytes());
        let digest = Blake2b256::digest(candidate);
        u32::from_le_bytes(digest[..4].try_into().expect("digest prefix")) < threshold
    }

    #[test]
    #[ignore = "requires a physical WGPU adapter with SHADER_INT64"]
    fn friendly_gpu_pipeline_preloading_reuses_compilation() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        assert!(gpu.friendly_blake2b_pipeline.get().is_none());
        assert!(gpu.linear_sha1_pipeline.get().is_none());
        assert!(gpu.sha1_decimal_pipeline.get().is_none());
        assert!(gpu.altcha_sha64_pipeline.get().is_none());

        let started = Instant::now();
        gpu.preload_protocol_pipelines();
        let preload_ms = started.elapsed().as_secs_f64() * 1e3;
        let pipeline = gpu
            .friendly_blake2b_pipeline
            .get()
            .ok_or_else(|| GpuError::new("missing preloaded Friendly BLAKE2b pipeline"))?;
        assert!(gpu.linear_sha1_pipeline.get().is_some());
        assert!(gpu.sha1_decimal_pipeline.get().is_some());
        assert!(gpu.altcha_sha64_pipeline.get().is_some());

        let started = Instant::now();
        gpu.preload_protocol_pipelines();
        let second_preload_us = started.elapsed().as_secs_f64() * 1e6;

        let mut times = Vec::new();
        for _ in 0..12 {
            let started = Instant::now();
            let solver = FriendlyBlake2bGpu::new(&gpu)?;
            assert!(std::ptr::eq(pipeline, solver.pipeline));
            times.push(started.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "FRIENDLY_GPU_CACHE {{\"preload_all_ms\":{preload_ms:.5},\"repeat_preload_us\":{second_preload_us:.5},\"friendly_instance_median_us\":{:.5}}}",
            times[times.len() / 2]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_matches_independent_cpu() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let solver = FriendlyBlake2bGpu::new(&gpu)?;
        let input = puzzle_input();
        for bank in [0, 1, 7, 255] {
            for base in [0, 1, 4_096, 65_534, u32::MAX - 127] {
                let result = super::super::sha512_tests::block_on(solver.scan(
                    &input,
                    bank,
                    base,
                    128,
                    u32::MAX,
                ))?;
                let nonce = result.expect("most BLAKE2b digests should match u32::MAX");
                assert!(cpu_matches(&input, bank, nonce, u32::MAX));
                assert!(u64::from(nonce) >= u64::from(base));
                assert!(u64::from(nonce) < u64::from(base) + 128);
            }
        }
        assert_eq!(
            super::super::sha512_tests::block_on(solver.scan(&input, 1, 0, 256, 0))?,
            None
        );
        let threshold = 1u32 << 24;
        let found =
            super::super::sha512_tests::block_on(solver.scan(&input, 5, 0, 65_536, threshold))?
                .expect("expected one 8-bit prefix hit in a 65k nonce window");
        assert!(cpu_matches(&input, 5, found, threshold));
        println!(
            "FRIENDLY_GPU_PROOF_PASS adapter={} nonce={found}",
            gpu.adapter()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_fixed_work_vs_rustcrypto() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let start_init = Instant::now();
        let solver = FriendlyBlake2bGpu::new(&gpu)?;
        let init_ms = start_init.elapsed().as_secs_f64() * 1e3;
        let input = puzzle_input();
        let bank = 3;
        let mut cpu_input = input;
        cpu_input[120] = bank;
        for count in [16_384u32, 65_536, 262_144] {
            for trial in 0..5 {
                let started = Instant::now();
                let result =
                    super::super::sha512_tests::block_on(solver.scan(&input, bank, 0, count, 1))?;
                assert!(result.is_none());
                let gpu_ns = started.elapsed().as_nanos();
                let started = Instant::now();
                for nonce in 0..count {
                    cpu_input[124..128].copy_from_slice(&nonce.to_le_bytes());
                    black_box(Blake2b256::digest(black_box(&cpu_input)));
                }
                let cpu_ns = started.elapsed().as_nanos();
                println!(
                    "FRIENDLY_GPU_BENCH {{\"adapter\":\"{}\",\"count\":{},\"trial\":{},\"gpu_ns\":{},\"cpu_ns\":{},\"gpu_rate\":{},\"cpu_rate\":{},\"init_ms\":{}}}",
                    gpu.adapter(),
                    count,
                    trial,
                    gpu_ns,
                    cpu_ns,
                    f64::from(count) * 1e9 / gpu_ns as f64,
                    f64::from(count) * 1e9 / cpu_ns as f64,
                    init_ms
                );
            }
        }
        Ok(())
    }
}
