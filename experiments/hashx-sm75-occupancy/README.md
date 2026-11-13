# HashX SM75 GPU optimization vs tevador — 2026-11-13

**This is HashX, not HashWX.** In
[tevador's HashWX design §3](https://github.com/tevador/hashwx/blob/master/doc/design.md)
the author reports a *non-public specialized CUDA HashX solver* running
**2,200 MH/s** on an RTX 5060 Ti. The 5.3 MH/s CPU figure is Anubis's
interpreted HashX on Ryzen 3700X (16 threads). Our source implementation
cannot be directly compared with the non-public solver.

Test target: Asus RTX 2060 Max-Q SM75, 30 SMs and **30 W current ceiling**
instead of requested/default 65 W. During full HashX workload, 49 GPU
telemetry samples had 100% utilization, **690 MHz median SM clock** and
**27.86 W median board draw**. USB-C reports sink and 3.25A. Presence
of a simultaneously attached barrel charger has *not* been verified.

## Released baseline

Our challenge-hashx crate already compiles new HashX GpuPrograms directly
to Turing SASS via a fixed cubin shell, loads via cudarc and uses the native
GPU core rather than the slower generic Vulkan interpreter. It has ~35
registers/thread, zero shared memory, and uses a 32-thread CTA with
the CTA indexing stride baked into generated SASS.

This branch explores alternative CTA sizes and a first-word-only final
SipRound without changing the released default or API.

## Results (physical RTX 2060, 2026-11-13)

All benchmark variants matched independent native CPU HashX minimum-nonce
answers for zero/nonzero base challenges (d12), and a 20-seed paired d18
matrix. Workload for raw throughput: 32 x 1,048,576 fixed-work hashes
per timing, no success until the end, five trials per shape after warm-up.

| CUDA threads/block | Single-seed raw MH/s |
| ---: | ---: |
| 32 (production) | 351.397 |
| 64 | 346.380 |
| 128 | 348.828 |
| 256 | 369.653 |
| 512 | 374.780 |

Sixteen independently generated challenge programs, interleaved, showed
32-thread **360.878 MH/s median** (343.351–377.870) versus 512-thread
**379.676 MH/s median** (360.555–398.519). Full d18 challenge latency,
including program construction, CUDA module load, batch loop and readback
on 20 paired seeds: 32 threads **1.0068ms mean** vs 512 threads **1.0417ms**.
Wider blocks improve raw fixed-work by ~5.2% but *do not* improve typical
complete challenge latency, so the original block size stays unchanged.

A test-only first-digest-word final SipRound removes dead arithmetic:
a+=b; c+=d; d=rotl16(d)^c; a=rotl32(a)+d.
Original vs partial GPU SASS at 32 threads: 362.83 vs 368.12 MH/s;
at 512 threads: 374.94 vs 379.61 MH/s. The ~1–1.5% raw increase and
noisy per-request numbers are not sufficient to justify production integration.

The author's 2200 MH/s GPU runs at 180 W stock; our physical GPU is
currently capped to 30 W. 360/30 and 2200/180 are ~12.0 and 12.2
MH/s per watt of *power-limit allowance*. This is illustrative only,
**not a measured cross-generation energy-efficiency comparison.**

## Reproduction

Use the usual managed sccache Cargo wrapper and Rust nightly-2026-11-03.
Run ignored physical GPU tests only on a real NVIDIA SM75 machine:

    cargo +nightly-2026-11-03 test --locked --release -p challenge-hashx --features gpu sm75_ -- --ignored --nocapture --test-threads=1

The benchmark source is in this branch; none of the experimental block-size
or partial finalizer logic replaces the public production code. All actual
timings/telemetry and output logs are *outside the source repo*, at:
 /home/jan/benchmarks/challenge-solvers/hashx-sm75-occupancy-20261113/
with physical-release.log, real-paired.log, finalizer-ab.log,
multi-seed.log, telemetry-multiseed.csv.

## Next work

1. Determine Asus barrel versus USB-C charging configuration and why the
   GPU has a 30 W enforced ceiling while reporting 65 W default.
2. If normal full-power operation is restored, repeat the 16-seed baseline
   before performing additional kernel micro-optimizations.
3. Investigate the prior 365–386 MH/s direct-SASS versus ~540–820 MH/s
   specialized PTX/NVRTC throughput discrepancy with the *same* hardware
   and operating regime, but retain direct SASS for cheap challenge startup.

No production merge recommended from this checkpoint. Related progress:
[Notion HashX GPU SM75 project page](https://app.notion.com/p/417f74be90208127a1e5f23ad37272f1).
