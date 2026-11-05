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
  dependency retained for the Anubis Argon2 regression suite. Licensing and
  upstream attribution are retained with the source.

The workspace retains the original source history (imported as Git subtrees).
See the individual packages for feature sets and backends. The service layer (HTTP endpoints, queues, routing, telemetry) remains in
challenge-disasterer. The service is not yet wired to consume these extracted
crates; the removal of its former in-tree solver sources is the next step.

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

The default `challenge-protocols` feature enables the optimized portable-SIMD
kernels and specialized Anubis Argon2id engine on pinned nightly Rust. To
consume all protocol implementations on stable Rust 1.99 instead, disable the
default feature:

```toml
challenge-protocols = { git = "https://github.com/codeandsolder/challenge-solvers", branch = "main", default-features = false }
```

This retains the same protocol API, including Anubis Argon2id, using a portable
RustCrypto implementation for memory-hard hashing and scalar SHA search.

`challenge-protocols` also provides `classic_pow::solve` for ALTCHA,
Cap.js, Cerberus, go-away, haphash, mCaptcha, and Linear.pub, plus Anubis WASM
SHA-256 and Argon2id. Protocol parsing, proof encoding and verification remain
inside the crate; HTTP dispatch, routing and telemetry do not.

## Performance policy

**Portable SIMD is a default backend, not an afterthought.** The workspace
currently pins nightly Rust for `portable_simd`, with architecture-specific
optimizations where measurements justify them. Scalar backends exist to test
correctness, portability, and fallback, not to replace optimized algorithms.

## CI and dependency maintenance

Each pull request runs portable-SIMD CPU regression tests on the pinned nightly
toolchain, plus the shared strict Rust/Python policy. Every Rust-compiling job
uses [the shared mandatory sccache workflow](https://github.com/codeandsolder/ci-workflows);
a separate gate checks the age of newly introduced Cargo.lock dependencies.

More expensive all-feature/GPU backend, stable Rust 1.99 scalar, AArch64 and
WASM SIMD128 checks run on release tags. The scalar lane is a portability
check, **not** a replacement for the optimized SIMD implementation.

[Renovate](renovate.json) tracks Cargo and immutable GitHub Actions references.
Ordinary dependencies have a 72-hour age hold. Reviewed shared CI and strict
policy changes are exempt from that hold using the central
[trust-root preset](https://github.com/codeandsolder/ci-workflows/blob/main/renovate-ci-trust-root.json).

## Status

Development migration: source packages are extracted and will be consumed as
pinned Git dependencies before any registry publication. Package naming,
documentation, independent protocol-crate boundaries and publishing are not yet
stable. Do not treat this repository as having published public crate versions.
