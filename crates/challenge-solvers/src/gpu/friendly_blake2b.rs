//! Experimental fixed-block BLAKE2b-256 GPU primitive for Friendly Captcha v1.
//! The producer passes the exact 128-byte padded v1 message; byte 120 is the
//! independently searchable bank and bytes 124..128 the LE u32 nonce.

use super::{GpuContext, GpuError, WG_SIZE};
use std::borrow::Cow;

// Keep each 2D GPU submission within the worst-case work already performed by
// the previous 16-bank x 262144-nonce scheduler. The per-bank workgroup is
// rounded to WG_SIZE, so partial workgroups count toward the budget.
const MAX_CANDIDATE_INVOCATIONS_PER_DISPATCH: u64 = 16 * 262_144;

fn planned_bank_rows(candidates: u32, max_workgroups: u32) -> Result<usize, GpuError> {
    if candidates == 0 || max_workgroups == 0 {
        return Err(GpuError::new(
            "Friendly GPU bank group has zero work or device capacity",
        ));
    }
    let per_bank = u64::from(candidates).div_ceil(u64::from(WG_SIZE)) * u64::from(WG_SIZE);
    let rows = (MAX_CANDIDATE_INVOCATIONS_PER_DISPATCH / per_bank)
        .max(1)
        .min(u64::from(max_workgroups))
        .min(u64::from(u8::MAX) + 1);
    usize::try_from(rows).map_err(|_| GpuError::new("Friendly GPU bank count too large"))
}

/// Initialized native-u64 BLAKE2b-256 pipeline for repeated GPU searches.
/// Independent puzzle banks share one 2D dispatch and one staging readback.
pub struct FriendlyBlake2bGpu<'a> {
    gpu: &'a GpuContext,
    pipeline: &'a wgpu::ComputePipeline,
    input: wgpu::Buffer,
    bind: wgpu::BindGroup,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
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
            size: (36 + u64::from(u8::MAX) + 1) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_size = (u64::from(u8::MAX) + 1) * 16;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-bank-results"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("friendly-blake2b-bank-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
                    resource: output.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            gpu,
            pipeline,
            input,
            bind,
            output,
            staging,
        })
    }

    /// Maximum candidates accepted in one GPU dispatch.
    #[must_use = "the returned capacity bounds the batch size"]
    pub fn batch_capacity(&self) -> u64 {
        u64::from(self.gpu.max_wgs) * u64::from(WG_SIZE)
    }

    /// Recommend a GPU bank group sized to the candidate window and device.
    /// A short nonce window can use more rows without increasing total work.
    ///
    /// # Errors
    /// Rejects a zero candidate window or a device lacking compute capacity.
    pub fn bank_group_capacity(&self, candidates: u32) -> Result<usize, GpuError> {
        planned_bank_rows(candidates, self.gpu.max_wgs)
    }

    /// Evaluate an exact nonce window independently for each requested bank,
    /// in one 2D GPU dispatch and staging-buffer readback.
    ///
    /// Banks are tagged by their original 8-bit ID; each result remains in the
    /// same order as the provided IDs. Threshold zero runs a fixed-work search.
    ///
    /// # Errors
    /// Rejects duplicate bank IDs, empty selections, nonce domain overflow,
    /// or invalid GPU workgroup counts. Device and readback failures propagate.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "exclusive scanner borrow prevents overlapping GPU staging maps and readbacks"
    )]
    pub async fn scan_banks(
        &mut self,
        padded: &[u8; 128],
        banks: &[u8],
        base: u32,
        count: u32,
        threshold: u32,
    ) -> Result<Vec<Option<u32>>, GpuError> {
        let domain_end = u64::from(base) + u64::from(count);
        let max_batch = self.batch_capacity();
        let bank_rows = u32::try_from(banks.len())
            .map_err(|_| GpuError::new("Friendly GPU bank count too large"))?;
        if banks.is_empty()
            || bank_rows > self.gpu.max_wgs
            || banks.len() > usize::from(u8::MAX) + 1
            || count == 0
            || domain_end > u64::from(u32::MAX) + 1
            || u64::from(count) > max_batch
        {
            return Err(GpuError::new(
                "Friendly bank selection or candidate window invalid",
            ));
        }
        let mut seen = [false; 256];
        for &bank in banks {
            if std::mem::replace(&mut seen[usize::from(bank)], true) {
                return Err(GpuError::new("duplicate Friendly Captcha bank ID"));
            }
        }
        let mut packed = [0u32; 36 + 256];
        for (slot, bytes) in packed[..32].iter_mut().zip(padded.as_chunks::<4>().0) {
            *slot = u32::from_le_bytes(*bytes);
        }
        packed[32] = base;
        packed[33] = count;
        packed[34] = threshold;
        packed[35] = bank_rows;
        for (slot, bank) in packed[36..].iter_mut().zip(banks) {
            *slot = u32::from(*bank);
        }
        self.gpu
            .queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&packed));
        let result_bytes =
            u64::try_from(banks.len()).map_err(|_| GpuError::new("too many banks"))? * 16;
        self.gpu.queue.write_buffer(
            &self.output,
            0,
            &vec![
                0;
                usize::try_from(result_bytes)
                    .map_err(|_| GpuError::new("result size overflow"))?
            ],
        );
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("friendly-blake2b-batched-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("friendly-blake2b-batched-banks"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(count.div_ceil(WG_SIZE), bank_rows, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.staging, 0, result_bytes);
        self.gpu.queue.submit([encoder.finish()]);
        let values = self.gpu.read_staging(&self.staging).await?;
        let mut result = Vec::with_capacity(banks.len());
        for words in values.as_chunks::<4>().0.iter().take(banks.len()) {
            if words[0] == 0 {
                result.push(None);
            } else {
                let candidate = words[1];
                if u64::from(candidate) < u64::from(base) || u64::from(candidate) >= domain_end {
                    return Err(GpuError::new("Friendly GPU nonce outside dispatched batch"));
                }
                result.push(Some(candidate));
            }
        }
        Ok(result)
    }

    /// Convenience wrapper for one bank, used by fixed-work experiments.
    ///
    /// # Errors
    /// Propagates invalid work window, GPU dispatch and readback errors.
    pub async fn scan(
        &mut self,
        padded: &[u8; 128],
        bank: u8,
        base: u32,
        count: u32,
        threshold: u32,
    ) -> Result<Option<u32>, GpuError> {
        self.scan_banks(padded, &[bank], base, count, threshold)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| GpuError::new("missing bank result"))
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
        super::super::sha512_tests::block_on(gpu.preload_and_warm_protocol_pipelines())?;
        let warm_dispatch_ms = started.elapsed().as_secs_f64() * 1e3;

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
            "FRIENDLY_GPU_CACHE {{\"preload_all_ms\":{preload_ms:.5},\"first_dispatch_ms\":{warm_dispatch_ms:.5},\"repeat_preload_us\":{second_preload_us:.5},\"friendly_instance_median_us\":{:.5}}}",
            times[times.len() / 2]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_matches_independent_cpu() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
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
        // Exercise 2D dispatch, independently indexed outputs, per-bank GPU
        // atomics, and bank 255 (outside the 0..254 full-puzzle range).
        let banks = [0, 7, 23, 128, 254, 255];
        let found_many = super::super::sha512_tests::block_on(solver.scan_banks(
            &input,
            &banks,
            u32::MAX - 127,
            128,
            u32::MAX,
        ))?;
        assert_eq!(found_many.len(), banks.len());
        for (bank, result) in banks.into_iter().zip(found_many) {
            let nonce = result
                .ok_or_else(|| anyhow::anyhow!("expected near-certain matching candidate"))?;
            assert!(cpu_matches(&input, bank, nonce, u32::MAX));
            assert!(nonce >= u32::MAX - 127);
        }
        assert_eq!(
            super::super::sha512_tests::block_on(solver.scan_banks(&input, &banks, 0, 256, 0,))?,
            vec![None; banks.len()]
        );
        assert!(
            super::super::sha512_tests::block_on(solver.scan_banks(&input, &[5, 5], 0, 128, 12,))
                .is_err()
        );
        assert!(
            super::super::sha512_tests::block_on(solver.scan_banks(&input, &[], 0, 128, 12,))
                .is_err()
        );
        // The protocol needs at most 255 banks, but the low-level primitive
        // supports the complete u8 bank-ID domain, including 255.
        let all_banks: Vec<u8> = (u8::MIN..=u8::MAX).collect();
        let all_found = super::super::sha512_tests::block_on(solver.scan_banks(
            &input,
            &all_banks,
            0,
            128,
            u32::MAX,
        ))?;
        assert_eq!(all_found.len(), 256);
        for (bank, hit) in all_banks.into_iter().zip(all_found) {
            let nonce = hit.ok_or_else(|| anyhow::anyhow!("bank {bank} yielded no nonce"))?;
            assert!(cpu_matches(&input, bank, nonce, u32::MAX));
            assert!(nonce < 128);
        }
        println!(
            "FRIENDLY_GPU_PROOF_PASS adapter={} nonce={found}",
            gpu.adapter()
        );
        Ok(())
    }

    #[test]
    fn friendly_gpu_group_budget_and_device_limits() -> anyhow::Result<()> {
        assert!(planned_bank_rows(0, 65535).is_err());
        assert!(planned_bank_rows(8192, 0).is_err());
        assert_eq!(planned_bank_rows(8192, 65535)?, 256);
        assert_eq!(planned_bank_rows(8192, 4)?, 4);
        assert_eq!(planned_bank_rows(32772, 65535)?, 127);
        assert_eq!(planned_bank_rows(131086, 65535)?, 31);
        assert_eq!(planned_bank_rows(262144, 65535)?, 16);
        assert_eq!(planned_bank_rows(u32::MAX, 65535)?, 1);
        assert_eq!(planned_bank_rows(1, 1)?, 1);
        for candidates in [1, 64, 65, 8192, 32772, 131086, 262144, u32::MAX] {
            let rows = planned_bank_rows(candidates, 65535)?;
            assert!((1..=256).contains(&rows));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_bank_group_shape_sweep() -> anyhow::Result<()> {
        // Hold the exact number of candidate hashes constant while varying
        // GPU row grouping. Alternating the order avoids systematic GPU
        // frequency and thermal drift favoring one particular shape.
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
        let input = puzzle_input();
        for (banks_count, candidates, sizes) in [
            (255usize, 8_192u32, &[16usize, 32, 64, 128, 255][..]),
            (16usize, 131_086u32, &[1usize, 4, 8, 16][..]),
        ] {
            let banks: Vec<u8> = (0..banks_count)
                .map(u8::try_from)
                .collect::<Result<_, _>>()?;
            // Fixed threshold zero prevents randomly early successes from
            // changing active-bank count; all groups do exactly equal work.
            for trial in 0..8 {
                let order: Vec<_> = if trial % 2 == 0 {
                    sizes.to_vec()
                } else {
                    sizes.iter().copied().rev().collect()
                };
                for group_size in order {
                    let started = Instant::now();
                    let mut submitted = 0u64;
                    for group in banks.chunks(group_size) {
                        let results = super::super::sha512_tests::block_on(
                            solver.scan_banks(&input, group, 0, candidates, 0),
                        )?;
                        assert_eq!(results, vec![None; group.len()]);
                        submitted += u64::from(candidates) * u64::try_from(group.len())?;
                    }
                    assert_eq!(
                        submitted,
                        u64::try_from(banks_count)? * u64::from(candidates)
                    );
                    println!(
                        "FRIENDLY_GROUP_SWEEP {{\"banks\":{banks_count},\"candidates\":{candidates},\"group_size\":{group_size},\"trial\":{trial},\"gpu_ms\":{:.6}}}",
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_batched_versus_sequential_banks() -> anyhow::Result<()> {
        const COUNT: u32 = 262_144;
        const ROUNDS: usize = 10;
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
        let input = puzzle_input();
        let banks: Vec<u8> = (0..16).collect();
        // Both cases perform the same 16*262144 BLAKE2b hashes, threshold 0.
        // This deliberately isolates GPU submission/readback overhead from
        // winning-nonce distribution and difficulty-dependent early returns.
        for trial in 0..ROUNDS {
            let batch_first = trial % 2 == 0;
            let mut sequential_ns = 0;
            let mut batched_ns = 0;
            for batched in [batch_first, !batch_first] {
                let started = Instant::now();
                if batched {
                    let result = super::super::sha512_tests::block_on(
                        solver.scan_banks(&input, &banks, 0, COUNT, 0),
                    )?;
                    assert_eq!(result, vec![None; banks.len()]);
                } else {
                    for &bank in &banks {
                        let result = super::super::sha512_tests::block_on(
                            solver.scan(&input, bank, 0, COUNT, 0),
                        )?;
                        assert_eq!(result, None);
                    }
                }
                if batched {
                    batched_ns = started.elapsed().as_nanos();
                } else {
                    sequential_ns = started.elapsed().as_nanos();
                }
            }
            println!(
                "FRIENDLY_BANK_AB {{\"trial\":{trial},\"batched_first\":{batch_first},\"banks\":16,\"candidates_per_bank\":{COUNT},\"sequential_ns\":{sequential_ns},\"batched_ns\":{batched_ns}}}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a physical SHADER_INT64 GPU"]
    fn friendly_blake2b_gpu_fixed_work_vs_rustcrypto() -> anyhow::Result<()> {
        let gpu = super::super::sha512_tests::block_on(GpuContext::create())?;
        let start_init = Instant::now();
        let mut solver = FriendlyBlake2bGpu::new(&gpu)?;
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
