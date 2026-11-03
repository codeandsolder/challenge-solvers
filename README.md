# challenge-solvers

Feature-configurable proof-of-work solvers used by `challenge-disasterer`.

The crate intentionally separates three layers:

1. shared message, SHA-256, BLAKE3, and solver primitives;
2. optional portable-SIMD architecture dispatch and stateful wide-solver wrappers;
3. optional challenge/protocol modules.

The challenge modules are independent Cargo features: `altcha`, `anubis`, `capjs`,
`cerberus`, `goaway`, and `mcaptcha`. `all-challenges` enables all six. Enabling or
disabling a protocol does not create a separate hashing/SIMD implementation: all
protocols reuse the same solver core.

Backend capability is a separate axis. `portable-simd` is enabled by default and turns on
the optimized x86/WASM SIMD backends; disabling it leaves the public solver aliases backed
by the scalar `safe` implementation and compiles on stable Rust. `gpu` implies
`portable-simd`. This lets a consumer choose protocols and acceleration independently rather
than selecting a second solver crate or duplicating challenge code.

The stable scalar surface also exposes `solve_decimal_sha256_zero_nibbles`, a bounded
protocol-agnostic decimal SHA-256 solver that reports exact attempted work.

With `portable-simd` enabled on x86-64, the portable wide backend selects 16 logical `u32` lanes behind an
AVX-512F target-feature boundary, 8 lanes behind AVX2, and then falls through to
SHA-NI or scalar implementations where applicable. The AVX2 and AVX-512 variants
share one state-machine implementation and one set of portable-SIMD kernels; the
architecture-specific code is limited to runtime feature detection and tiny target
feature thunks.

WebAssembly currently keeps its SIMD128 facade while using the same portable kernels
for the supported solver families. Further consolidation should happen by extending
the shared backend contract rather than copying challenge logic.

See `UPSTREAM.md` for provenance and retained-license details.
