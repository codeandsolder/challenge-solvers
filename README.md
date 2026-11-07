# challenge-solvers

Reusable high-performance Rust proof-of-work solver implementations and
cryptographic search machinery extracted from
[challenge-disasterer](https://github.com/codeandsolder/challenge-disasterer).

## Workspace

- [challenge-solvers](crates/challenge-solvers) — shared SHA-256/BLAKE3 search
  engines, per-protocol message encoders and verifier utilities, and optimized
  CPU/GPU backends for ALTCHA, Anubis, Cap.js, Cerberus, go-away and mCaptcha.
- [challenge-protocols](crates/challenge-protocols) — reusable Anubis (classic and WASM),
  ALTCHA, Cap.js, Cerberus, go-away, haphash, mCaptcha and Linear.pub challenge
  parsing, solve/verify, backend selection and protocol response construction.
- [challenge-hashx](crates/challenge-hashx) — standalone HashX challenge verification, native JIT and optional Vulkan/SM75 CUDA backends (without the HTTP service).
- [anubis-argon2](crates/anubis-argon2) — specialized Anubis Argon2id
  computation and tests.
- [vendor/argon2-fast](vendor/argon2-fast) — locally modified Argon2 reference
  dependency used by the generic ALTCHA Argon2id adapter and Anubis differential tests. Licensing and
  upstream attribution are retained with the source.

The workspace retains the original source history (imported as Git subtrees).
See individual packages for features and backends. The HTTP service,
worker queues, routing, and telemetry remain in
[challenge-disasterer](https://github.com/codeandsolder/challenge-disasterer).
Its migration to these reusable Git dependencies is maintained in that
repository; this workspace owns the algorithm implementations and protocol APIs.

## Use from another Rust project

The protocol crate is importable directly from this repository, independently
of the HTTP service. Until registry publication, use the repository branch; Cargo.lock records the
selected Git commit. For production consumers, replace `branch` with the exact
validated `rev` (the workspace resolves its own internal solver dependency):

```toml
[dependencies]
challenge-protocols = { git = "https://github.com/codeandsolder/challenge-solvers", branch = "main" }
```

```rust
use challenge_protocols::anubis;

let challenge = anubis::Challenge {
    data: "example".to_owned(),
    difficulty: 1,
};
let result = anubis::solve_cpu(&challenge, 1_000_000)
    .expect("valid bounded challenge");
assert!(anubis::verify(&challenge, &result.answer));
```

The pinned nightly toolchain and portable SIMD are required by the solver
workspace, including CPU-only configurations. Disabling optional GPU support or
protocol adapters does not disable the SIMD implementation.

The challenge-protocols package also provides `classic_pow::solve` for ALTCHA,
Cap.js, Cerberus, go-away, haphash, mCaptcha, and Linear.pub, plus Anubis WASM
SHA-256 and Argon2id. Protocol parsing, proof encoding and verification remain
inside the crate; HTTP dispatch, routing and telemetry do not.

## Performance policy

**Portable SIMD is mandatory, not an optional backend.** The workspace
currently pins nightly Rust for `portable_simd`, with architecture-specific
optimizations where measurements justify them. Scalar backends exist to test
correctness, portability, and fallback, not to replace optimized algorithms.

## CI and dependency maintenance

Each pull request runs portable-SIMD CPU regression tests on the pinned nightly
toolchain, plus the shared strict Rust/Python policy. Every Rust-compiling job
uses [the shared mandatory sccache workflow](https://github.com/codeandsolder/ci-workflows);
a separate gate checks the age of newly introduced Cargo.lock dependencies.

More expensive all-feature/GPU backend, AArch64 and WASM SIMD128 checks run on
release tags. Explicit scalar solvers are retained for hardware fallback and
performance comparisons, not for a separate stable-Rust build.

[Renovate](renovate.json) tracks Cargo and immutable GitHub Actions references.
Ordinary dependencies have a 72-hour age hold. Reviewed shared CI and strict
policy changes are exempt from that hold using the central
[trust-root preset](https://github.com/codeandsolder/ci-workflows/blob/main/renovate-ci-trust-root.json).

## Status

Development migration: source packages are extracted and will be consumed as
pinned Git dependencies before any registry publication. Package naming,
documentation, independent protocol-crate boundaries and publishing are not yet
stable. Do not treat this repository as having published public crate versions.
