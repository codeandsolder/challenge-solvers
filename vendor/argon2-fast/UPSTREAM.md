# Upstream provenance

This crate is a local copy of the Argon2 crate from RustCrypto/password-hashes
PR #937 at commit c3d11f0dee50ce54738c249e6038c05f355127c8
("argon2: add AVX2 block compression backend"). The PR is based on RustCrypto
master after Argon2 0.6.0 and keeps the public API, output, and MSRV 1.85.

The AVX2 backend follows the optimized official Argon2 reference implementation,
including segment-resident SIMD state and the fill-block round layout. The
reference source used by our independent comparator is pinned at
P-H-C/phc-winner-argon2@f57e61e19229e23c4445b85494dbf7c07de721cb.

RustCrypto Argon2 remains MIT OR Apache-2.0. The reference optimized AVX2 source
is available under CC0-1.0 OR Apache-2.0.

Local challenge-disasterer measurements and production-service validation are
recorded in the project worklog rather than duplicated here so this provenance
file does not go stale when benchmark methodology changes.
