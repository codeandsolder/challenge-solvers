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
- [anubis-argon2](crates/anubis-argon2) — specialized Anubis Argon2id
  computation and tests.
- [vendor/argon2-fast](vendor/argon2-fast) — locally modified Argon2 reference
  dependency retained for the Anubis Argon2 regression suite. Licensing and
  upstream attribution are retained with the source.

The workspace retains the original source history (imported as Git subtrees).
See the individual packages for feature sets and backends. The service layer (HTTP endpoints, queues, routing, telemetry) remains in
challenge-disasterer. The service is not yet wired to consume these extracted
crates; the removal of its former in-tree solver sources is the next step.

## Performance policy

**Portable SIMD is a default backend, not an afterthought.** The workspace
currently pins nightly Rust for `portable_simd`, with architecture-specific
optimizations where measurements justify them. Scalar backends exist to test
correctness, portability, and fallback, not to replace optimized algorithms.

## Status

Development migration: source packages are extracted and will be consumed as
pinned Git dependencies before any registry publication. Package naming,
documentation, independent protocol-crate boundaries and publishing are not yet
stable. Do not treat this repository as having published public crate versions.
