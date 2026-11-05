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
    ) -> anyhow::Result<()> {
        let config = LaunchConfig {
            grid_dim: (batch / WARP_SIZE, 1, 1),
            block_dim: (WARP_SIZE, 1, 1),
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

    #[test]
    fn batches_are_warp_aligned_and_bounded() {
        for difficulty in 1..=64 {
            let batch = batch_size(difficulty);
            assert_eq!(batch % WARP_SIZE, 0);
            assert!((WARP_SIZE..=MAX_BATCH).contains(&batch));
        }
    }
}
