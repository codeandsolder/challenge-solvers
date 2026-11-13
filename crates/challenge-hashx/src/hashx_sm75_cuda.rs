use std::sync::Arc;

use anyhow::{Context, anyhow};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use crate::{
    hashx::{GpuProgram, difficulty_masks},
    hashx_sm75::{build_solve_cubin, template_solve_cubin},
};

const WARP_SIZE: u32 = 32;
const MAX_BATCH: u32 = 262_144;

pub struct Sm75Cuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    solution: CudaSlice<u32>,
    adapter: String,
}

impl Sm75Cuda {
    pub(crate) fn create(context: Arc<CudaContext>, name: String) -> anyhow::Result<Self> {
        let stream = context.default_stream();
        let solution = stream
            .alloc_zeros::<u32>(1)
            .context("allocate CUDA HashX solution buffer")?;

        // Pay the CUDA driver's one-time cubin/module setup cost during backend creation,
        // not on the first real challenge. This also verifies that the embedded SM75 shell
        // is loadable by the selected driver before we advertise the backend as ready.
        let warm_module = context
            .load_module(Ptx::from_binary(template_solve_cubin()))
            .context("warm sm_75 CUDA module loader")?;
        warm_module
            .load_function("kern")
            .context("verify embedded sm_75 HashX kernel")?;
        drop(warm_module);

        Ok(Self {
            context,
            stream,
            solution,
            adapter: name,
        })
    }

    pub(crate) fn adapter(&self) -> &str {
        &self.adapter
    }

    pub(crate) fn solve(&mut self, program: &GpuProgram, difficulty: u32) -> anyhow::Result<u32> {
        let cubin = build_solve_cubin(program)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin))
            .context("load generated sm_75 HashX cubin")?;
        let function = module
            .load_function("kern")
            .context("load sm_75 HashX kernel")?;
        self.stream
            .memcpy_htod(&[u32::MAX], &mut self.solution)
            .context("reset CUDA HashX solution")?;

        let (mask0, mask1) = difficulty_masks(difficulty);
        let batch_size = batch_size(difficulty);
        let mut base = 0_u64;
        while u32::try_from(base).is_ok() {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u32::try_from(u64::from(batch_size).min(remaining))
                .context("sm_75 HashX batch exceeds u32")?;
            debug_assert_eq!(batch % WARP_SIZE, 0);
            let base32 = u32::try_from(base).context("sm_75 HashX base exceeds u32")?;
            self.launch(&function, base32, batch, mask0, mask1)?;
            let mut answer = [u32::MAX];
            self.stream
                .memcpy_dtoh(&self.solution, &mut answer)
                .context("queue sm_75 HashX solution readback")?;
            self.stream
                .synchronize()
                .context("synchronize sm_75 HashX solve and readback")?;
            if answer[0] != u32::MAX {
                return Ok(answer[0]);
            }
            base += u64::from(batch);
        }
        Err(anyhow!("sm_75 HashX search exhausted u32 nonce space"))
    }

    fn launch(
        &mut self,
        function: &cudarc::driver::CudaFunction,
        base: u32,
        batch: u32,
        mask0: u32,
        mask1: u32,
    ) -> anyhow::Result<()> {
        self.launch_with_block_size(function, base, batch, mask0, mask1, WARP_SIZE)
    }

    #[expect(
        unsafe_code,
        reason = "CUDA kernel launch crosses cudarc's audited unsafe driver boundary"
    )]
    fn launch_with_block_size(
        &mut self,
        function: &cudarc::driver::CudaFunction,
        base: u32,
        batch: u32,
        mask0: u32,
        mask1: u32,
        block_size: u32,
    ) -> anyhow::Result<()> {
        assert_eq!(
            batch % block_size,
            0,
            "CUDA launch must cover exactly the requested range"
        );
        let config = LaunchConfig {
            grid_dim: (batch / block_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `kern` is emitted by `hashx_sm75` from a fixed cubin template whose
        // parameter ABI is `(u32 *solution, u32 base, u32 mask0, u32 mask1)`. The
        // generated kernel only touches `solution[0]`; the remaining arguments are
        // copied by value. `self.solution` stays alive until the synchronized launch
        // completes, and grid/block dimensions are warp-aligned by construction.
        unsafe {
            self.stream
                .launch_builder(function)
                .arg(&mut self.solution)
                .arg(&base)
                .arg(&mask0)
                .arg(&mask1)
                .launch(config)
        }
        .context("launch sm_75 HashX kernel")?;
        Ok(())
    }
}

fn batch_size(difficulty: u32) -> u32 {
    let expected = 1_u32 << difficulty.min(31);
    let wanted = expected.div_ceil(4).clamp(WARP_SIZE, MAX_BATCH);
    wanted.div_ceil(WARP_SIZE) * WARP_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real GPU: exercise all launch shapes against independent native `HashX`
    /// results before comparing fixed-work kernel-only throughput.
    #[test]
    #[ignore = "requires physical NVIDIA SM75 GPU"]
    fn sm75_block_occupancy_hardware() -> anyhow::Result<()> {
        const FULL_BATCH: u32 = 1_048_576;
        const ROUNDS: u32 = 32;
        const TRIALS: usize = 5;

        use std::time::Instant;

        use crate::hashx::{describe, valid_prefix};
        use crate::hashx_sm75::{build_partial_finalizer_cubin, build_solve_cubin_with_block_size};

        let seed = b"HashX CUDA sm75 occupancy experiment 2026-11-13";
        let (hashx, program) = describe(seed)?;
        let context = CudaContext::new(0)?;
        let mut gpu = Sm75Cuda::create(context, "SM75 occupancy".to_owned())?;
        let oracle_masks = difficulty_masks(12);
        let batch = 32_768u32;
        let base_offsets = [0u32, 1_234_560u32];
        let expected = base_offsets.map(|base| {
            (0..batch)
                .find(|&nonce| {
                    let hash = hashx.hash_bytes(u64::from(base + nonce));
                    valid_prefix(&hash, 12)
                })
                .map_or(u32::MAX, |nonce| base + nonce)
        });
        assert!(expected.iter().all(|&nonce| nonce != u32::MAX));

        for (threads, partial) in [
            (32u32, false),
            (32, true),
            (64, false),
            (128, false),
            (256, false),
            (512, false),
            (512, true),
        ] {
            let build_started = Instant::now();
            let cubin = if partial {
                build_partial_finalizer_cubin(&program, threads)?
            } else {
                build_solve_cubin_with_block_size(&program, threads)?
            };
            let build_us = build_started.elapsed().as_secs_f64() * 1e6;
            let module = gpu.context.load_module(Ptx::from_binary(cubin))?;
            let kernel = module.load_function("kern")?;
            for (&base, &want) in base_offsets.iter().zip(expected.iter()) {
                gpu.stream.memcpy_htod(&[u32::MAX], &mut gpu.solution)?;
                gpu.launch_with_block_size(
                    &kernel,
                    base,
                    batch,
                    oracle_masks.0,
                    oracle_masks.1,
                    threads,
                )?;
                let mut got = [u32::MAX];
                gpu.stream.memcpy_dtoh(&gpu.solution, &mut got)?;
                gpu.stream.synchronize()?;
                assert_eq!(got[0], want, "block_size={threads} base={base}");
            }

            // Reject successful solutions so every active thread hashes;
            // this is the exact same SASS with the normal search predicate.
            // Each trial measures 32 full passes across 1Mi candidates, and
            // the stream synchronization is inside the measured interval.
            let (no_solution_lo, no_solution_hi) = difficulty_masks(64);
            let mut times_ms = [0f64; TRIALS];
            for sample in 0..TRIALS + 2 {
                gpu.stream.memcpy_htod(&[u32::MAX], &mut gpu.solution)?;
                gpu.stream.synchronize()?;
                let begin = Instant::now();
                for _ in 0..ROUNDS {
                    gpu.launch_with_block_size(
                        &kernel,
                        0,
                        FULL_BATCH,
                        no_solution_lo,
                        no_solution_hi,
                        threads,
                    )?;
                }
                gpu.stream.synchronize()?;
                let ms = begin.elapsed().as_secs_f64() * 1000.0;
                if sample >= 2 {
                    times_ms[sample - 2] = ms;
                }
            }
            times_ms.sort_by(f64::total_cmp);
            let median_ms = times_ms[TRIALS / 2];
            let throughput_mhs = f64::from(FULL_BATCH) * f64::from(ROUNDS) / (median_ms * 1000.0);
            println!(
                "SM75_BLOCK_BENCH threads={threads} partial={partial} MHs={throughput_mhs:.5} median_ms={median_ms:.5} build_us={build_us:.2} goldens=2"
            );
        }
        Ok(())
    }

    /// Compare complete challenge latency including cubin generation,
    /// module load, batches/readback and independent CPU minimum-nonce checks.
    #[test]
    #[ignore = "requires physical NVIDIA SM75 GPU"]
    fn sm75_paired_real_challenge_hardware() -> anyhow::Result<()> {
        use crate::hashx::{describe, valid_prefix};
        use crate::hashx_sm75::build_solve_cubin_with_block_size;
        use std::time::Instant;

        let ctx = CudaContext::new(0)?;
        let mut gpu = Sm75Cuda::create(ctx, "SM75 real challenge".to_owned())?;
        let difficulty = 18;
        let (mask0, mask1) = difficulty_masks(difficulty);
        let batch_size = 65_536u32;
        let mut data = Vec::new();
        for seed_number in 0_u32..20 {
            let seed = seed_number.to_le_bytes();
            let (hashx, program) = describe(&seed)?;
            // CPU oracle is the actual native HashX implementation, not another
            // GPU kernel or the shared generated instruction interpreter.
            let wanted = (0..4_194_304u32).find(|&nonce| {
                let hash = hashx.hash_bytes(u64::from(nonce));
                valid_prefix(&hash, difficulty)
            });
            let Some(wanted) = wanted else {
                continue;
            };
            let variants = if seed_number % 2 == 0 {
                [(32u32, false), (32, true), (512, false), (512, true)]
            } else {
                [(512u32, true), (512, false), (32, true), (32, false)]
            };
            for (block_size, partial) in variants {
                let started = Instant::now();
                let cubin = if partial {
                    crate::hashx_sm75::build_partial_finalizer_cubin(&program, block_size)?
                } else {
                    build_solve_cubin_with_block_size(&program, block_size)?
                };
                let module = gpu.context.load_module(Ptx::from_binary(cubin))?;
                let function = module.load_function("kern")?;
                let mut got = None;
                for base in (0..4_194_304u32).step_by(batch_size as usize) {
                    gpu.stream.memcpy_htod(&[u32::MAX], &mut gpu.solution)?;
                    gpu.launch_with_block_size(
                        &function, base, batch_size, mask0, mask1, block_size,
                    )?;
                    let mut answer = [u32::MAX];
                    gpu.stream.memcpy_dtoh(&gpu.solution, &mut answer)?;
                    gpu.stream.synchronize()?;
                    if answer[0] != u32::MAX {
                        got = Some(answer[0]);
                        break;
                    }
                }
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    got,
                    Some(wanted),
                    "block_size={block_size} seed={seed_number}"
                );
                data.push((block_size, partial, elapsed_ms));
                println!(
                    "SM75_REAL_PAIR seed={seed_number} threads={block_size} partial={partial} ms={elapsed_ms:.6} nonce={wanted}"
                );
            }
        }
        let mut grouped = [
            ((32u32, false), Vec::new()),
            ((32, true), Vec::new()),
            ((512, false), Vec::new()),
            ((512, true), Vec::new()),
        ];
        for (block, partial, ms) in data {
            let group = grouped
                .iter_mut()
                .find(|((b, p), _)| *b == block && *p == partial)
                .context("registered benchmark configuration")?;
            group.1.push(ms);
        }
        assert!(grouped.iter().all(|(_, samples)| samples.len() >= 16));
        for ((threads, partial), times) in &mut grouped {
            times.sort_by(f64::total_cmp);
            println!(
                "SM75_REAL_SUMMARY threads={threads} partial={partial} n={} median_ms={:.6} mean_ms={:.6}",
                times.len(),
                times[times.len() / 2],
                times.iter().sum::<f64>() / f64::from(u32::try_from(times.len())?),
            );
        }
        Ok(())
    }

    /// Raw sustained throughput should not be inferred from a single
    /// challenge program: opcode and branch patterns vary by seed.
    #[test]
    #[ignore = "requires physical NVIDIA SM75 GPU"]
    fn sm75_multi_seed_hardware() -> anyhow::Result<()> {
        const BATCH: u32 = 1_048_576;
        const REPEATS: u32 = 32;

        use crate::hashx::{describe, valid_prefix};
        use crate::hashx_sm75::build_solve_cubin_with_block_size;
        use std::time::Instant;

        let ctx = CudaContext::new(0)?;
        let mut gpu = Sm75Cuda::create(ctx, "SM75 seed sweep".into())?;
        let (mask0, mask1) = difficulty_masks(64);
        let mut stats = [Vec::<f64>::new(), Vec::<f64>::new()];
        for seed in 0_u32..16 {
            let (hashx, program) = describe(&seed.to_le_bytes())?;
            let expected = (0..32768u32)
                .find(|&n| valid_prefix(&hashx.hash_bytes(u64::from(n)), 12))
                .context("HashX benchmark seed lacked a low-difficulty answer")?;
            let variants = if seed % 2 == 0 {
                [32u32, 512]
            } else {
                [512u32, 32]
            };
            for threads in variants {
                let cubin = build_solve_cubin_with_block_size(&program, threads)?;
                let module = gpu.context.load_module(Ptx::from_binary(cubin))?;
                let kernel = module.load_function("kern")?;
                // Ensure independently exact minimum nonce, not merely a
                // plausible digest, before timing this seed's kernel.
                gpu.stream.memcpy_htod(&[u32::MAX], &mut gpu.solution)?;
                let (v0, v1) = difficulty_masks(12);
                gpu.launch_with_block_size(&kernel, 0, 32768, v0, v1, threads)?;
                let mut result = [u32::MAX];
                gpu.stream.memcpy_dtoh(&gpu.solution, &mut result)?;
                gpu.stream.synchronize()?;
                assert_eq!(result[0], expected, "seed={seed} threads={threads}");
                let mut elapsed = [0f64; 4];
                for t in 0..5 {
                    gpu.stream.memcpy_htod(&[u32::MAX], &mut gpu.solution)?;
                    gpu.stream.synchronize()?;
                    let started = Instant::now();
                    for _ in 0..REPEATS {
                        gpu.launch_with_block_size(&kernel, 0, BATCH, mask0, mask1, threads)?;
                    }
                    gpu.stream.synchronize()?;
                    if t > 0 {
                        elapsed[t - 1] = started.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                elapsed.sort_by(f64::total_cmp);
                let middle = f64::midpoint(elapsed[1], elapsed[2]);
                let rate = f64::from(BATCH) * f64::from(REPEATS) / (middle * 1000.0);
                stats[usize::from(threads != 32)].push(rate);
                println!(
                    "SM75_SEED_MATRIX seed={seed} threads={threads} rate_MHs={rate:.5} oracle={expected}"
                );
            }
        }
        for (i, values) in stats.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            println!(
                "SM75_SEED_MATRIX_SUMMARY threads={} n={} median_MHs={:.5} min_MHs={:.5} max_MHs={:.5}",
                if i == 0 { 32 } else { 512 },
                values.len(),
                f64::midpoint(values[7], values[8]),
                values[0],
                values[15],
            );
        }
        Ok(())
    }

    #[test]
    fn batches_are_warp_aligned_and_bounded() {
        for difficulty in 1..=64 {
            let batch = batch_size(difficulty);
            assert_eq!(batch % WARP_SIZE, 0);
            assert!((WARP_SIZE..=MAX_BATCH).contains(&batch));
        }
    }
}
