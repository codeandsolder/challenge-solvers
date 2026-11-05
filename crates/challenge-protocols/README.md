# challenge-protocols

Reusable challenge-solvers-independent-of-HTTP protocol handling for
Anubis (classic and WASM), ALTCHA, Cap.js, Cerberus, go-away, haphash,
mCaptcha and Linear.pub. These protocols share the portable-SIMD-first
`challenge-solvers` engine. CPU execution is available by default,
and `gpu` enables optional accelerated wgpu paths.

## Example

```rust
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let result = challenge_protocols::classic_pow::solve(
        "goaway",
        json!({
            "challenge": "0123456789abcdef".repeat(4),
            "difficulty": 0
        }),
        Some("auto"),
    )?;

    println!("{}", result.answer["token"]);
    Ok(())
}
```

The library exposes protocol-native results with duration metrics; it does
not host HTTP endpoints, queue work, perform service calibration or maintain
telemetry. Those remain in [challenge-disasterer](https://github.com/codeandsolder/challenge-disasterer).

## Current packaging

Git consumption only until independent crate publishing is reviewed.
`hashwx-rs` remains a pinned dependency for Cap.js's HashWX variant;
its eventual repository placement is undecided. Portable SIMD and
autovectorization are required fast paths, not optional afterthoughts.
