# challenge-hashx

Reusable Anubis HashX challenge solver and verifier, independent of the
`challenge-disasterer` HTTP server. This package belongs to the
[`challenge-solvers`](https://github.com/codeandsolder/challenge-solvers) workspace.

- Default (Rust 1.99): native `hashwx` JIT CPU search, exact HashX proof validation,
  deterministic program generation, nonce and difficulty checks.
- `gpu`: Vulkan/WGSL, Vulkan SPIR-V persistent backends, and native NVIDIA Turing
  SM75 CUDA SASS backend, with the same HashX IR and verifier as the CPU path.
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
SM75 cubin are source-controlled runtime assets, not newly generated build
outputs. Attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
