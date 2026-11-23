# challenge-hashx

Reusable Anubis HashX challenge solver and verifier, independent of the
`challenge-disasterer` HTTP server. This package belongs to the
[`challenge-solvers`](https://github.com/codeandsolder/challenge-solvers) workspace.

- Default (Rust 1.99): native `hashwx` JIT CPU search, exact HashX proof validation,
  deterministic program generation, nonce and difficulty checks.
- `gpu`: Vulkan/WGSL, Vulkan SPIR-V persistent backends, and native NVIDIA Turing and Ampere
  SM75/SM86 CUDA SASS backend, with the same HashX IR and verifier as the CPU path.
  This is an optional native hardware path; it is not silently substituted for
  generic GPU hardware.

```toml
[dependencies]
challenge-hashx = { git = "https://github.com/codeandsolder/challenge-solvers", branch = "main" }
```

```rust
use challenge_hashx::{hashx, hashx_cpu_jit};

fn solve(data_hex: String) -> anyhow::Result<String> {
    let challenge = hashx::Challenge { data: data_hex, difficulty: 4 };
    let result = hashx_cpu_jit::solve(&challenge, 1)?;
    anyhow::ensure!(hashx::verify(&challenge, &result.answer)?);
    Ok(result.answer.nonce)
}
```

HashX proof and search-result types are defined by this crate; no SHA/Argon2/Anubis protocol dependency is pulled in merely to serialize a HashX answer.

The app manages queueing, CPU worker budgeting, GPU lifecycle and routing. This
crate only handles actual HashX computation. The bundled SPIR-V shaders and
SM75/SM86 cubins are source-controlled runtime assets, not newly generated build
outputs. Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

## Direct SASS for Ampere (SM86)

The same typed instruction emitter handles Turing SM75 and Ampere SM86;
only device-specific memory reduction and the cubin template differ.
The driver selects an exact CUDA compute capability after matching the wgpu
adapter, and reports `cuda-sm75-sass` or `cuda-sm86-sass`. Other architectures
continue using the existing Vulkan path. The runtime needs the CUDA driver,
not a toolkit, nvcc, ptxas or NVRTC. The safe warmup shell is loaded but not
executed; each challenge patches the live instruction prefix and, on SM86,
the actual EXIT metadata. The existing SM75 golden bytes are unchanged.

The embedded SM86 cubin was built with CUDA 13.1 ptxas 13.1.115 and has
40 registers, no shared/local/stack memory, and the established 20-byte
parameter ABI. Its SHA-256 is
`b2a358dc88e22529c98bea72b2520f892ca96b87176fd37097924b1ee9e4c896`.
Reproduce it with:

```sh
bash tools/build-sm86-template.sh /usr/local/cuda/bin/ptxas /path/to/output.cubin
cmp src/hashx_sm86_solve_template.cubin /path/to/output.cubin
```

The script fails closed if ptxas changes the expected ELF layout. The final
inert shell starts with EXIT; dummy PTX stores exist only to establish a
compiler-generated register allocation and text capacity, and are replaced
with dead NOPs. SM86 uses the compiler-derived uniform address descriptor
load and GPU-scope RED.E.MIN.STRONG.GPU. The old SM75 reduction encodes
system scope on SM86; it is not reused there.

### Physical results, RTX 3080 (2026-11-22)

Driver 591.86; native Windows CUDA Rust test on a desktop RTX 3080 (SM86,
68 SMs); WSL CUDA 13.1 for the independent matched C++ driver comparison.
The test generates a distinct program for 32 seeds; independently
interpreted CPU minima match all 1,440 bounded ranges (three masks, three
offsets including near the upper nonce boundary, five block sizes) and all
32 complete difficulty-18 searches. SM75 reference byte hashes and the
1,024-program fixed-shell fit tests passed. The measured first eight
seeds use five median event-timed windows per block geometry, each
16 x 1,048,576 candidates, difficulty 64 to avoid early termination.

| Threads/block | Device MH/s | Wall MH/s |
| --- | ---: | ---: |
| 32 | 2258.212 | 2218.935 |
| 64 | 2213.771 | 2173.907 |
| 128 | 2283.848 | 2251.092 |
| 256 | 2267.031 | 2226.118 |
| 512 | 2288.167 | 2235.951 |

Medians over the seed medians. Overall median code emission 32.7 us;
CUDA module load/function lookup 176.8 us; complete fresh difficulty-18
solve 2.591 ms (range 0.566–15.770 ms under the observed host load).
Retain 32 threads/block for production: the raw geometry sweep does not
establish enough end-to-end benefit to change it.

The independent comparison uses seed bytes 0..31, the same 32-bit nonce
ranges, predicate-only result, 32-thread geometry, 16 large launches and
event timing; three alternating-order trials. Direct Rust SM86 SASS 2107.267
MH/s versus offline nvcc 4370.269 MH/s (0.845 s offline compiler wall time).
Each implementation matched all 12 reference minima every round. This is
~48% of this measured AOT throughput, not a universal hardware bound.
It is not comparable to the previous museum short batches returning every
hash. Raw logs: [benchmarks/sm86-20261122](benchmarks/sm86-20261122/).

Run the focused physical test on an exact supported CUDA architecture:
```sh
cargo test --locked -p challenge-hashx --release --features gpu --lib \
  hashx_cuda::physical::validate_and_benchmark_native_sass -- --ignored --nocapture
```

The separate ignored `emit_disassembly_fixtures` test writes cubins and a
CPU-generated `oracle-museum.tsv` into the current working directory.
Build tests with `--no-run`, then invoke the test executable with
`--exact hashx_cuda::physical::emit_disassembly_fixtures --ignored --nocapture`
from an explicit benchmark output directory, not a source checkout.
The offline source is the default-seed `kernel.cu` from the
[museum generator](https://github.com/codeandsolder/museum-of-jit/pull/38),
with only its final `hash_batch` entry replaced by the following:
```cpp
extern "C" __global__ void kern(unsigned* solution, unsigned base,
                                unsigned mask0, unsigned mask1) {
    unsigned nonce = base + blockIdx.x * blockDim.x + threadIdx.x;
    U64 h = hash(nonce, nullptr);
    if (!(unsigned(h) & mask0) && !(unsigned(h >> 32) & mask1))
        atomicMin(solution, nonce);
}
```
Compile `nvcc --cubin -O3 -arch=sm_86 -std=c++17` and use
`tools/compare-cubin.cpp` (CUDA Driver API, compiled with the toolkit
headers and linked to libcuda) to test both files with the same oracle
TSV. The tool assumes the audited 32-thread, four-argument ABI.

### CPU baseline on the same RTX 3080 machine (2026-11-23)

The Windows host is a Ryzen 7 3700X, eight physical cores and 16 logical
threads. The reusable `examples/cpu_baseline.rs` uses the authoritative
native Rust `HashX` JIT, the default 00..1f fixed seed, 1,048,576
contiguous nonces, and 11 post-warmup median samples. All worker counts
produced the same complete-batch checksum.

| CPU workers | Native JIT MH/s | SM86 GPU / CPU |
| --- | ---: | ---: |
| 1 | 14.461 | 156.2 |
| 2 | 28.642 | 78.8 |
| 4 | 55.384 | 40.8 |
| 8 | 78.226 | 28.9 |
| 12 | 96.846 | 23.3 |
| 16 | 107.423 | 21.0 |

GPU reference: 2,258 MH/s measured with the same real RTX 3080 using
32-thread blocks; this GPU test uses predicate-only reduction and
16 x 1,048,576 candidates, whereas the CPU tests materialize a
checksum and aggregate across worker threads. These are useful
sustained-throughput comparisons rather than end-to-end latencies.

The same executable also calls the real `hashx_cpu_jit::solve`
for 32 independently generated four-byte seeds 0..31 and three
difficulties, verifying each answer with the shared verifier:

| Difficulty | CPU 1 worker | 8 workers | 12 workers | 16 workers |
| --- | ---: | ---: | ---: | ---: |
| 12 | 0.334 ms | 0.607 ms | 0.836 ms | 1.006 ms |
| 18 | 18.721 ms | 3.322 ms | 2.667 ms | 2.870 ms |
| 20 | 58.283 ms | 9.344 ms | 6.738 ms | 7.015 ms |

The physical SM86 GPU test was extended to exactly the same 32 seeds,
at difficulty 12, 18 and 20, and validates every minimum nonce with
an independently interpreted CPU oracle:

| Difficulty | Fastest CPU solve | SM86 GPU solve | Outcome |
| --- | ---: | ---: | --- |
| 12 | 0.334 ms (1 worker) | 1.834 ms | CPU 5.5x faster |
| 18 | 2.667 ms (12 workers) | 2.853 ms | Approximately even |
| 20 | 6.738 ms (12 workers) | 5.062 ms | GPU 1.33x faster |

These are medians of 32 separate fresh puzzles on the same Windows
machine, with CPU and GPU passes run sequentially. A previous SM86
difficulty-18 pass gave 2.591 ms; these sub-millisecond differences
are not meaningful absent tightly interleaved host-load control.
The CPU benchmark includes per-challenge program compilation and
thread startup; GPU includes emitter, cubin module loading and
driver dispatch. A 21x sustained-throughput advantage is therefore
not a 21x latency advantage on short challenges. The cross-over is
near 18–20 difficulty bits with this implementation and system load.

The 96 GPU full-solve checks and original 1,440 range checks all
passed. Raw GPU latencies and seed-by-seed nonces:
[benchmarks/sm86-20261122/gpu-difficulty-20261123.txt](benchmarks/sm86-20261122/gpu-difficulty-20261123.txt).

Source and raw results:
`examples/cpu_baseline.rs` and
[benchmarks/sm86-20261122/cpu-3700x-windows-20261123.txt](benchmarks/sm86-20261122/cpu-3700x-windows-20261123.txt).
