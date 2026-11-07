# challenge-solvers

Feature-configurable proof-of-work solvers used by `challenge-disasterer`.

The crate intentionally separates three layers:

1. shared message, SHA-256, BLAKE3, and solver primitives;
2. required portable-SIMD execution with runtime architecture dispatch and stateful wide-solver wrappers;
3. optional challenge/protocol modules.

The challenge modules are independent Cargo features: `altcha`, `anubis`, `capjs`,
`cerberus`, `goaway`, and `mcaptcha`. `all-challenges` enables all six. Enabling or
disabling a protocol does not create a separate hashing/SIMD implementation: all
protocols reuse the same solver core.

Portable SIMD is a required capability and the pinned nightly toolchain is mandatory.
Architecture-specific acceleration is still runtime-dispatched: optimized SIMD kernels
when supported, with scalar safe solvers retained as explicit comparison and hardware
fallback implementations rather than a separate stable-Rust build surface.

solve_decimal_sha256_zero_nibbles remains a bounded, protocol-agnostic
decimal SHA-256 solver that reports exact attempted work.

On x86-64, the portable wide backend selects 16 logical u32 lanes behind an
AVX-512F target-feature boundary, 8 lanes behind AVX2, and then falls through to
SHA-NI or scalar implementations where applicable. The AVX2 and AVX-512 variants
share one state-machine implementation and one set of portable-SIMD kernels; the
architecture-specific code is limited to runtime feature detection and tiny target
feature thunks.

WebAssembly currently keeps its SIMD128 facade while using the same portable kernels
for the supported solver families. Further consolidation should happen by extending
the shared backend contract rather than copying challenge logic.

See `UPSTREAM.md` for provenance and retained-license details.
