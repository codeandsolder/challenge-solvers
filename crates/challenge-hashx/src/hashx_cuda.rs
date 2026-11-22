use std::sync::Arc;

use anyhow::{Context, anyhow};
use cudarc::{
    driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use hashwx::HashX;

use crate::{
    hashx::{GpuProgram, difficulty_masks, gpu_sentinel_nonce},
    hashx_sass::{SassTarget, build_solve_cubin, template_solve_cubin},
};

const WARP_SIZE: u32 = 32;
const MAX_BATCH: u32 = 262_144;

pub struct SassCuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    solution: CudaSlice<u32>,
    adapter: String,
    target: SassTarget,
}

impl SassCuda {
    pub(crate) fn create_for_adapter(info: &wgpu::AdapterInfo) -> anyhow::Result<Option<Self>> {
        if info.vendor != 0x10de {
            return Ok(None);
        }

        let count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(error) => {
                tracing::debug!(%error, "CUDA driver unavailable for native HashX backend");
                return Ok(None);
            }
        };
        let mut matches = Vec::new();
        for ordinal in 0..count {
            let Ok(cuda_ordinal) = usize::try_from(ordinal) else {
                tracing::debug!(ordinal, "CUDA device ordinal is negative");
                continue;
            };
            let context = match CudaContext::new(cuda_ordinal) {
                Ok(context) => context,
                Err(error) => {
                    tracing::debug!(ordinal, %error, "failed to inspect CUDA device");
                    continue;
                }
            };
            let name = context.name()?;
            if name != info.name {
                continue;
            }
            let capability = context.compute_capability()?;
            matches.push((context, name, capability));
        }

        let Some((context, name, capability)) = matches.pop() else {
            return Ok(None);
        };
        if !matches.is_empty() {
            tracing::warn!(
                adapter = %info.name,
                "multiple CUDA devices match the selected wgpu adapter name; disabling native HashX backend"
            );
            return Ok(None);
        }

        if let Some(target) = SassTarget::from_capability(capability) {
            Ok(Some(Self::create(context, name, target)?))
        } else {
            tracing::debug!(?capability, adapter = %info.name, "no native HashX encoder for CUDA architecture");
            Ok(None)
        }
    }

    pub(crate) const fn implementation(&self) -> &'static str {
        self.target.implementation()
    }

    fn create(context: Arc<CudaContext>, name: String, target: SassTarget) -> anyhow::Result<Self> {
        let stream = context.default_stream();
        let solution = stream
            .alloc_zeros::<u32>(1)
            .context("allocate CUDA HashX solution buffer")?;

        // Pay the CUDA driver's one-time cubin/module setup cost during backend creation,
        // not on the first real challenge. This also verifies that the embedded architecture-specific shell
        // is loadable by the selected driver before we advertise the backend as ready.
        let warm_module = context
            .load_module(Ptx::from_binary(template_solve_cubin(target)))
            .context("warm CUDA module loader")?;
        warm_module
            .load_function("kern")
            .context("verify embedded native CUDA HashX kernel")?;
        drop(warm_module);

        Ok(Self {
            context,
            stream,
            solution,
            adapter: name,
            target,
        })
    }

    pub(crate) fn adapter(&self) -> &str {
        &self.adapter
    }

    pub(crate) fn solve(
        &mut self,
        program: &GpuProgram,
        hashx: &HashX,
        difficulty: u32,
    ) -> anyhow::Result<u32> {
        let cubin = build_solve_cubin(program, self.target, WARP_SIZE)?;
        let module = self
            .context
            .load_module(Ptx::from_binary(cubin))
            .context("load generated HashX cubin")?;
        let function = module
            .load_function("kern")
            .context("load native CUDA HashX kernel")?;
        self.stream
            .memcpy_htod(&[u32::MAX], &mut self.solution)
            .context("reset CUDA HashX solution")?;

        let (mask0, mask1) = difficulty_masks(difficulty);
        let batch_size = batch_size(difficulty);
        let mut base = 0_u64;
        while u32::try_from(base).is_ok() {
            let remaining = u64::from(u32::MAX) + 1 - base;
            let batch = u32::try_from(u64::from(batch_size).min(remaining))
                .context("native CUDA HashX batch exceeds u32")?;
            debug_assert_eq!(batch % WARP_SIZE, 0);
            let base32 = u32::try_from(base).context("native CUDA HashX base exceeds u32")?;
            self.launch(&function, base32, batch, mask0, mask1, WARP_SIZE)?;
            let mut answer = [u32::MAX];
            self.stream
                .memcpy_dtoh(&self.solution, &mut answer)
                .context("queue native CUDA HashX solution readback")?;
            self.stream
                .synchronize()
                .context("synchronize native CUDA HashX solve and readback")?;
            if answer[0] != u32::MAX {
                return Ok(answer[0]);
            }
            base += u64::from(batch);
        }
        gpu_sentinel_nonce(hashx, difficulty)
            .ok_or_else(|| anyhow!("native CUDA HashX search exhausted u32 nonce space"))
    }

    #[expect(
        unsafe_code,
        reason = "CUDA kernel launch crosses cudarc's audited unsafe driver boundary"
    )]
    fn launch(
        &mut self,
        function: &cudarc::driver::CudaFunction,
        base: u32,
        batch: u32,
        mask0: u32,
        mask1: u32,
        block_size: u32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            block_size > 0 && batch.is_multiple_of(block_size),
            "CUDA batch must contain complete blocks"
        );
        let config = LaunchConfig {
            grid_dim: (batch / block_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `kern` is emitted by `hashx_sass` from a fixed cubin template whose
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
        .context("launch native CUDA HashX kernel")?;
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

    #[test]
    fn batches_are_warp_aligned_and_bounded() {
        for difficulty in 1..=64 {
            let batch = batch_size(difficulty);
            assert_eq!(batch % WARP_SIZE, 0);
            assert!((WARP_SIZE..=MAX_BATCH).contains(&batch));
        }
    }
}

#[cfg(test)]
mod physical {
    use super::*;
    use crate::hashx::{build_compiled_seed, describe};
    use cudarc::driver::sys::CUevent_flags;
    use std::{fmt::Write as _, time::Instant};

    const BLOCKS: [u32; 5] = [32, 64, 128, 256, 512];
    const VALIDATION_BATCH: u32 = 8192;
    const THROUGHPUT_BATCH: u32 = 1_048_576;
    const LAUNCHES: u32 = 16;
    const SAMPLES: usize = 5;

    #[test]
    #[ignore = "writes disassembly fixtures to a dedicated benchmark working directory"]
    fn emit_disassembly_fixtures() -> anyhow::Result<()> {
        let (_, program) = describe(&[0, 1, 2, 3])?;
        for (target, filename) in [
            (SassTarget::Sm75, "reference-sm75.cubin"),
            (SassTarget::Sm86, "reference-sm86.cubin"),
        ] {
            std::fs::write(filename, build_solve_cubin(&program, target, 32)?)?;
        }
        let seed = std::array::from_fn::<_, 32, _>(|index| index.to_le_bytes()[0]);
        let (compiled, program) = describe(&seed)?;
        std::fs::write(
            "reference-sm86-museum.cubin",
            build_solve_cubin(&program, SassTarget::Sm86, 32)?,
        )?;
        let mut rows = String::new();
        for base in [0, 1_234_567] {
            for difficulty in [8, 12, 16, 18, 20, 40] {
                let want = oracle(
                    &compiled,
                    base,
                    THROUGHPUT_BATCH,
                    difficulty_masks(difficulty),
                );
                writeln!(rows, "{base}\t{THROUGHPUT_BATCH}\t{difficulty}\t{want}")?;
            }
        }
        std::fs::write("oracle-museum.tsv", rows)?;
        Ok(())
    }

    fn backend() -> anyhow::Result<SassCuda> {
        let context = CudaContext::new(0)?;
        let target = SassTarget::from_capability(context.compute_capability()?)
            .context("physical comparison requires a supported CUDA device")?;
        let name = context.name()?;
        eprintln!("device={name} target={target:?}");
        SassCuda::create(context, name, target)
    }

    fn oracle(reference: &HashX, base: u32, batch: u32, masks: (u32, u32)) -> u32 {
        let mask = u64::from(masks.0) | (u64::from(masks.1) << 32);
        (base..base + batch)
            .find(|&nonce| reference.hash(u64::from(nonce)) & mask == 0)
            .unwrap_or(u32::MAX)
    }

    fn check_range(
        backend: &mut SassCuda,
        function: &cudarc::driver::CudaFunction,
        base: u32,
        block: u32,
        masks: (u32, u32),
        expected: u32,
    ) -> anyhow::Result<()> {
        backend
            .stream
            .memcpy_htod(&[u32::MAX], &mut backend.solution)?;
        backend.launch(function, base, VALIDATION_BATCH, masks.0, masks.1, block)?;
        let mut actual = [u32::MAX];
        backend.stream.memcpy_dtoh(&backend.solution, &mut actual)?;
        backend.stream.synchronize()?;
        anyhow::ensure!(
            actual[0] == expected,
            "physical mismatch block={block} base={base} masks={masks:?} actual={} expected={expected}",
            actual[0]
        );
        Ok(())
    }

    fn median(values: &mut [f64; SAMPLES]) -> f64 {
        values.sort_by(f64::total_cmp);
        values[SAMPLES / 2]
    }

    fn throughput(
        backend: &mut SassCuda,
        function: &cudarc::driver::CudaFunction,
        block: u32,
    ) -> anyhow::Result<(f64, f64)> {
        let mut wall = [0.0; SAMPLES];
        let mut device = [0.0; SAMPLES];
        backend
            .stream
            .memcpy_htod(&[u32::MAX], &mut backend.solution)?;
        backend.launch(function, 0, THROUGHPUT_BATCH, u32::MAX, u32::MAX, block)?;
        backend.stream.synchronize()?;
        for (w, d) in wall.iter_mut().zip(&mut device) {
            let start = Instant::now();
            let begin = backend
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for launch in 0..LAUNCHES {
                backend.launch(
                    function,
                    launch * THROUGHPUT_BATCH,
                    THROUGHPUT_BATCH,
                    u32::MAX,
                    u32::MAX,
                    block,
                )?;
            }
            let end = backend
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            backend.stream.synchronize()?;
            *w = start.elapsed().as_secs_f64();
            *d = f64::from(begin.elapsed_ms(&end)?) / 1000.0;
        }
        let mut result = [0];
        backend.stream.memcpy_dtoh(&backend.solution, &mut result)?;
        backend.stream.synchronize()?;
        anyhow::ensure!(
            result == [u32::MAX],
            "unexpected 64-bit proof in benchmark range"
        );
        let count = f64::from(THROUGHPUT_BATCH) * f64::from(LAUNCHES) / 1e6;
        Ok((count / median(&mut device), count / median(&mut wall)))
    }

    #[test]
    #[ignore = "requires physical SM75 or SM86 CUDA device"]
    fn validate_and_benchmark_native_sass() -> anyhow::Result<()> {
        let mut backend = backend()?;
        let mut verified = 0;
        for seed in 0_u32..32 {
            let bytes = seed.to_le_bytes();
            let (compiled, program) = describe(&bytes)?;
            let (_, _, ir) = build_compiled_seed(&bytes)?;
            let reference = HashX::interpret_program(&ir);
            let masks = [difficulty_masks(12), difficulty_masks(16), (0, 0xf0)];
            let cases: Vec<_> = [0, 1_234_560, u32::MAX - 16_383]
                .into_iter()
                .flat_map(|base| masks.into_iter().map(move |mask| (base, mask)))
                .map(|(base, mask)| (base, mask, oracle(&reference, base, VALIDATION_BATCH, mask)))
                .collect();
            for block in BLOCKS {
                let start = Instant::now();
                let cubin = build_solve_cubin(&program, backend.target, block)?;
                let compile_us = start.elapsed().as_secs_f64() * 1e6;
                let start = Instant::now();
                let module = backend.context.load_module(Ptx::from_binary(cubin))?;
                let function = module.load_function("kern")?;
                let load_us = start.elapsed().as_secs_f64() * 1e6;
                for &(base, mask, expected) in &cases {
                    check_range(&mut backend, &function, base, block, mask, expected)?;
                    verified += 1;
                }
                if seed < 8 {
                    let (gpu, wall) = throughput(&mut backend, &function, block)?;
                    eprintln!(
                        "SASS seed={seed} block={block} compile_us={compile_us:.3} load_us={load_us:.3} gpu_mhs={gpu:.3} wall_mhs={wall:.3}"
                    );
                }
            }
            let expected = (0..=u32::MAX)
                .find(|&nonce| {
                    reference.hash(u64::from(nonce)) & u64::from(difficulty_masks(18).0) == 0
                })
                .context("no 18-bit solution in nonce space")?;
            // Also exercise the released search interface with a fresh cubin.
            let start = Instant::now();
            let actual = backend.solve(&program, &compiled, 18)?;
            let us = start.elapsed().as_secs_f64() * 1e6;
            anyhow::ensure!(
                actual == expected,
                "full solve mismatch seed={seed}: {actual} != {expected}"
            );
            eprintln!("SOLVE seed={seed} nonce={actual} elapsed_us={us:.3}");
        }
        eprintln!("VERIFIED_RANGE_CHECKS={verified} VERIFIED_FULL_SOLVES=32");
        Ok(())
    }
}
