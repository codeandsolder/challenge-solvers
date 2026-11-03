# Solver core provenance

`challenge-solvers` is the in-tree protocol PoW engine for `challenge-disasterer`.

It was imported on 2026-10-10 from `codeandsolder/pow-buster-imp` commit
`fa7ff0b29f6737a75fba49766a81951caae92d62`, which descends from
`eternal-flame-AD/pow-buster`. The imported lineage remains Apache-2.0; see
`LICENSE` in this directory.

Included from that lineage are the message builders, protocol machinery, SHA-256 and
BLAKE3 CPU paths, runtime CPU dispatch, ALTCHA/PBKDF2 support, Cerberus formats, and
the generalized WebGPU solver backend. `challenge-disasterer` owns service, routing,
live-protocol verification, benchmarking, and deployment; the old standalone HTTP,
browser-addon, static-UI, CLI, and WASM-FFI packaging remains excluded.

## Current CPU architecture

The historical hand-separated AVX2 and AVX-512 solver state machines have since been
replaced by one portable-SIMD implementation. The stateful solver layer is generic
over a backend contract; x86-64 AVX2 and AVX-512 differ only in runtime feature
detection, logical lane width (8 versus 16 `u32` lanes), and tiny audited
`#[target_feature]` thunks. SHA-NI remains a separate backend because it exposes real
SHA instructions that portable SIMD cannot express.

The lane widths are intentionally part of the backend rather than inferred inside a
single `Simd<T, N>` instantiation: fixed-work measurements on Ryzen 9 4900HS (Zen 2)
and EPYC 9655 (Zen 5) showed 8 lanes are optimal for AVX2 while 16 lanes nearly double
throughput on AVX-512. The shared source therefore performs runtime backend selection
without duplicating challenge state machines.

## Validation

Native x86 implementations are exercised by the deterministic hardware matrix
according to host capabilities (safe, AVX2, SHA-NI, AVX-512, plus runtime
auto-dispatch), and GPU implementations are covered on real adapters. The
WebAssembly SIMD128 backend is compiled and run separately under `wasm32-wasip1`
with `+simd128`.

For comparison with preexisting work, `tools/preexisting-baseline` pins
`eternal-flame-AD/pow-buster@d359adb3a692ba6be27fc5c1d5efdbbd7439434d` in an
independent Cargo workspace so the historical git dependency never becomes a runtime
or workspace dependency.
