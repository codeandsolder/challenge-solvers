use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail, ensure};
use challenge_solvers::solver::SOLVE_TYPE_MASK;
use challenge_solvers::{
    capjs::ChallengeDescriptor as CapJsChallenge,
    message::{CapJSEmitter, DecimalMessage},
};
use hashwx::HashWx;
use rug::Integer;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{Implementation, JS_MAX_SAFE_INTEGER, Result, decimal_solver};

#[derive(Debug, Deserialize)]
struct CapV2Envelope {
    token: String,
    format: u8,
    challenges: Vec<CapV2Challenge>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "protocol")]
enum CapV2Challenge {
    #[serde(rename = "sha256-pow")]
    Sha256Pow { payload: CapV2Sha256 },
    #[serde(rename = "rsw")]
    Rsw { payload: CapV2Rsw },
    #[serde(rename = "hashwx")]
    HashWx { payload: CapV2HashWx },
    #[serde(rename = "instrumentation")]
    Instrumentation { payload: Value },
}

#[derive(Debug, Deserialize)]
struct CapV2Sha256 {
    salt: String,
    target: String,
}

#[derive(Debug, Deserialize)]
struct CapV2Rsw {
    #[serde(rename = "N")]
    modulus: String,
    x: String,
    t: u32,
}

#[derive(Debug, Deserialize)]
struct CapV2HashWx {
    c: String,
    d: u64,
    n: u32,
}

/// Reject an unsupported challenge before spending CPU time on any earlier
/// subchallenge in a mixed format-2 envelope. Both CPU and GPU routes use this
/// same preflight so neither can partially execute a challenge we cannot solve.
fn parse_cap_v2_envelope(challenge: Value) -> anyhow::Result<CapV2Envelope> {
    let envelope: CapV2Envelope =
        serde_json::from_value(challenge).context("invalid Cap.js format-2 challenge")?;
    ensure!(envelope.format == 2, "invalid Cap.js format-2 marker");
    ensure!(
        !envelope.challenges.is_empty(),
        "Cap.js format-2 challenge list is empty"
    );
    ensure!(
        !envelope
            .challenges
            .iter()
            .any(|challenge| matches!(challenge, CapV2Challenge::Instrumentation { .. })),
        "Cap.js instrumentation is a browser-environment probe, not a proof-of-work mode"
    );
    Ok(envelope)
}

/// Validate a complete format-2 envelope for a caller that may split work
/// across compute backends. Reject browser-only instrumentation before any
/// partial SHA/RSW/HashWX work is scheduled.
pub(super) fn preflight_format_two(challenge: &Value) -> anyhow::Result<()> {
    parse_cap_v2_envelope(challenge.clone()).map(|_| ())
}

fn solve_cap_decimal_cpu(
    prefix: &[u8],
    target: u64,
    mask: u64,
    implementation: Implementation,
) -> anyhow::Result<(f64, u64)> {
    let mut attempted = 0_u64;
    for bank in 0..=u32::MAX {
        let Some((message, fixup)) = DecimalMessage::new_f64(prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(u64::MAX);
        let result = solver.solve(target, SOLVE_TYPE_MASK, mask);
        attempted = attempted.saturating_add(solver.attempted());
        if let Some((nonce, _)) = result {
            let nonce = match fixup {
                Some(value) => value.fixup(nonce),
                None => js_safe_integer_as_f64(nonce)?,
            };
            return Ok((nonce, attempted));
        }
    }
    bail!("Cap.js SHA-256 search exhausted the lossless JavaScript-number domain")
}

pub(super) fn solve(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    if challenge.get("format").and_then(Value::as_u64) == Some(2) {
        return solve_capjs_v2(challenge, implementation);
    }
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let rules = *descriptor.rules();
    let token = descriptor.token;
    let emitter = CapJSEmitter::new(token.as_bytes());
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(rules.count);
    for index in 0..rules.count {
        let mut salt = vec![0_u8; rules.salt_length];
        let mut targets = [0_u32; 2];
        let subchallenge = u32::try_from(index)
            .context("Cap.js subchallenge index exceeds u32")?
            .checked_add(1)
            .context("Cap.js subchallenge index overflow")?;
        emitter.emit(&mut salt, &mut targets, subchallenge);
        let mask = rules
            .mask()
            .context("Cap.js difficulty exceeds the supported 64-bit target width")?;
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, attempted) = solve_cap_decimal_cpu(&salt, target, mask, implementation)?;
        attempted_nonces = attempted_nonces.saturating_add(attempted);
        solutions.push(nonce);
    }
    let elapsed = started.elapsed();
    let solve_ms = elapsed.as_secs_f64() * 1000.0;
    let (elapsed_us, hashrate) = elapsed_work_metadata(elapsed, attempted_nonces);
    let answer = json!({
        "token": token,
        "solutions": solutions,
        "_meta": {
            "elapsed_us": elapsed_us,
            "attempted_nonces": attempted_nonces,
            "hashrate": hashrate,
        },
        "attempted_nonces_total": attempted_nonces,
    });
    Ok(Result {
        answer,
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn cap_v2_prefix(target: &str) -> anyhow::Result<(u64, u64)> {
    ensure!(
        !target.is_empty(),
        "Cap.js SHA-256 target must not be empty"
    );
    ensure!(target.len() <= 16, "Cap.js SHA-256 target exceeds 64 bits");
    ensure!(
        target.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "Cap.js SHA-256 target must be hexadecimal"
    );
    let bits = u32::try_from(target.len())?.saturating_mul(4);
    let value = u64::from_str_radix(target, 16).context("invalid Cap.js SHA-256 target")?;
    let shift = 64_u32.saturating_sub(bits);
    let mask = if bits == 64 {
        u64::MAX
    } else {
        u64::MAX << shift
    };
    Ok((value << shift, mask))
}

fn solve_cap_v2_sha256(
    payload: &CapV2Sha256,
    implementation: Implementation,
) -> anyhow::Result<(Value, u64)> {
    let (target, mask) = cap_v2_prefix(&payload.target)?;
    let (nonce, attempted) =
        solve_cap_decimal_cpu(payload.salt.as_bytes(), target, mask, implementation)?;
    Ok((json!({ "nonce": nonce }), attempted))
}

/// Parse protocol-defined unsigned hexadecimal integers without accepting signs,
/// whitespace, prefixes or other formats not permitted by the Cap.js RSW wire format.
fn parse_rsw_hex(text: &str) -> Option<Integer> {
    if text.is_empty() || !text.bytes().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    Integer::from_str_radix(text, 16).ok()
}

/// Repeated modular squaring via GMP's native modular exponentiation.
///
/// Every block represents exactly count consecutive squarings, as
/// x^(2^count) mod N. Bound the exponent size independently of rounds.
/// This remains correct for odd and even positive moduli of arbitrary size.
fn repeated_mod_squaring(
    mut value: Integer,
    modulus: &Integer,
    mut rounds: u32,
) -> anyhow::Result<Integer> {
    const SQUARES_PER_BLOCK: u32 = 8_192;
    value.modulo_mut(modulus);
    while rounds != 0 {
        let count = rounds.min(SQUARES_PER_BLOCK);
        let exponent = Integer::from(1) << count;
        value = value
            .pow_mod(&exponent, modulus)
            .map_err(|_| anyhow!("Cap.js RSW modular exponentiation failed"))?;
        rounds -= count;
    }
    Ok(value)
}

fn solve_cap_v2_rsw(payload: &CapV2Rsw) -> anyhow::Result<Value> {
    ensure!(payload.t > 0, "Cap.js RSW iteration count must be positive");
    let modulus = parse_rsw_hex(&payload.modulus).context("invalid Cap.js RSW modulus")?;
    let value = parse_rsw_hex(&payload.x).context("invalid Cap.js RSW input")?;
    ensure!(modulus > 1, "Cap.js RSW modulus is too small");
    let result = repeated_mod_squaring(value, &modulus, payload.t)?;
    Ok(json!({ "y": result.to_string_radix(16) }))
}

fn hashwx_seed(challenge: &[u8; 32], block: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(challenge);
    digest.update(block.to_le_bytes());
    digest.finalize().into()
}

fn solve_cap_v2_hashwx(payload: &CapV2HashWx) -> anyhow::Result<(Value, u64)> {
    ensure!(
        payload.d > 0 && payload.d <= 1_000_000_000,
        "Cap.js HashWX difficulty out of range"
    );
    ensure!(
        (1..=1_048_576).contains(&payload.n),
        "Cap.js HashWX nonces-per-hash out of range"
    );
    let challenge = hex::decode(&payload.c).context("invalid Cap.js HashWX challenge hex")?;
    let challenge: [u8; 32] = challenge
        .try_into()
        .map_err(|_| anyhow!("Cap.js HashWX challenge must be 32 bytes"))?;
    let target = u64::MAX / payload.d;
    let per_seed = u64::from(payload.n);
    let mut current_block = 0_u64;
    let mut hashwx = HashWx::compile(&hashwx_seed(&challenge, current_block))
        .context("compile Cap.js HashWX program")?;
    let mut attempted = 0_u64;
    let mut nonce = 0_u64;
    loop {
        let block = nonce / per_seed;
        if block != current_block {
            hashwx = HashWx::compile(&hashwx_seed(&challenge, block))
                .context("compile Cap.js HashWX program")?;
            current_block = block;
        }
        attempted = attempted.saturating_add(1);
        if hashwx.hash(nonce) <= target {
            return Ok((json!({ "nonce": nonce.to_string() }), attempted));
        }
        if nonce == u64::MAX {
            break;
        }
        nonce += 1;
    }
    bail!("Cap.js HashWX search exhausted the u64 nonce space")
}

fn cap_v2_backend_name(sha_backend: Option<&str>, used_hashwx: bool, used_rsw: bool) -> String {
    let mut backends = Vec::with_capacity(3);
    if let Some(sha_backend) = sha_backend {
        backends.push(sha_backend);
    }
    if used_hashwx {
        backends.push("hashwx-native");
    }
    if used_rsw {
        backends.push("rsw");
    }
    backends.join("+")
}

fn solve_capjs_v2(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let envelope = parse_cap_v2_envelope(challenge)?;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(envelope.challenges.len());
    let mut used_sha256 = false;
    let mut used_hashwx = false;
    let mut used_rsw = false;
    for challenge in &envelope.challenges {
        let solution = match challenge {
            CapV2Challenge::Sha256Pow { payload } => {
                used_sha256 = true;
                let (solution, attempted) = solve_cap_v2_sha256(payload, implementation)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Rsw { payload } => {
                used_rsw = true;
                solve_cap_v2_rsw(payload)?
            }
            CapV2Challenge::HashWx { payload } => {
                used_hashwx = true;
                let (solution, attempted) = solve_cap_v2_hashwx(payload)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Instrumentation { payload } => {
                let _ = payload;
                bail!(
                    "Cap.js instrumentation is a browser-environment probe, not a proof-of-work mode"
                )
            }
        };
        solutions.push(solution);
    }
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let implementation_name = cap_v2_backend_name(
        used_sha256.then_some(implementation.name()),
        used_hashwx,
        used_rsw,
    );
    Ok(Result {
        answer: json!({
            "token": envelope.token,
            "solutions": solutions,
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation_name,
    })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "value is explicitly constrained to JavaScript's exactly representable integer range"
)]
fn js_safe_integer_as_f64(value: u64) -> anyhow::Result<f64> {
    ensure!(
        value <= JS_MAX_SAFE_INTEGER,
        "Cap.js nonce exceeds JavaScript's maximum safe integer"
    );
    Ok(value as f64)
}

fn elapsed_work_metadata(elapsed: Duration, attempted: u64) -> (u64, u64) {
    let elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    let nanos = elapsed.as_nanos();
    let hashrate = u128::from(attempted)
        .saturating_mul(1_000_000_000)
        .checked_div(nanos)
        .map_or(0, |per_second| {
            u64::try_from(per_second).unwrap_or(u64::MAX)
        });
    (elapsed_us, hashrate)
}

#[cfg(feature = "gpu")]
async fn solve_cap_decimal_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    prefix: &[u8],
    target: u64,
    mask: u64,
) -> anyhow::Result<(f64, u64)> {
    let mut attempted = 0_u64;
    for bank in 0..=u32::MAX {
        let Some((message, fixup)) = DecimalMessage::new_f64(prefix, bank) else {
            break;
        };
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(u64::MAX);
        let result = solver.solve::<SOLVE_TYPE_MASK>(gpu, target, mask).await?;
        attempted = attempted.saturating_add(solver.attempted_nonces());
        if let Some(solution) = result {
            let nonce = match fixup {
                Some(value) => value.fixup(solution.nonce),
                None => js_safe_integer_as_f64(solution.nonce)?,
            };
            return Ok((nonce, attempted));
        }
    }
    bail!("Cap.js SHA-256 GPU search exhausted the lossless JavaScript-number domain")
}

#[cfg(feature = "gpu")]
async fn solve_capjs_v2_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let envelope = parse_cap_v2_envelope(challenge)?;

    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(envelope.challenges.len());
    let mut used_sha256 = false;
    let mut used_hashwx = false;
    let mut used_rsw = false;

    for challenge in &envelope.challenges {
        let solution = match challenge {
            CapV2Challenge::Sha256Pow { payload } => {
                used_sha256 = true;
                let (target, mask) = cap_v2_prefix(&payload.target)?;
                let (nonce, attempted) =
                    solve_cap_decimal_gpu(gpu, payload.salt.as_bytes(), target, mask).await?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                json!({ "nonce": nonce })
            }
            CapV2Challenge::Rsw { payload } => {
                used_rsw = true;
                solve_cap_v2_rsw(payload)?
            }
            CapV2Challenge::HashWx { payload } => {
                used_hashwx = true;
                let (solution, attempted) = solve_cap_v2_hashwx(payload)?;
                attempted_nonces = attempted_nonces.saturating_add(attempted);
                solution
            }
            CapV2Challenge::Instrumentation { payload } => {
                let _ = payload;
                bail!(
                    "Cap.js instrumentation is a browser-environment probe, not a proof-of-work mode"
                )
            }
        };
        solutions.push(solution);
    }

    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "token": envelope.token,
            "solutions": solutions,
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: cap_v2_backend_name(used_sha256.then_some("wgpu"), used_hashwx, used_rsw),
    })
}

#[cfg(feature = "gpu")]
pub(super) async fn solve_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    if challenge.get("format").and_then(Value::as_u64) == Some(2) {
        return solve_capjs_v2_gpu(gpu, challenge).await;
    }
    let descriptor: CapJsChallenge =
        serde_json::from_value(challenge).context("invalid Cap.js challenge")?;
    let rules = *descriptor.rules();
    let token = descriptor.token.clone();
    let emitter = CapJSEmitter::new(token.as_bytes());
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solutions = Vec::with_capacity(rules.count);
    for index in 0..rules.count {
        let mut salt = vec![0_u8; rules.salt_length];
        let mut targets = [0_u32; 2];
        let subchallenge = u32::try_from(index)
            .context("Cap.js subchallenge index exceeds u32")?
            .checked_add(1)
            .context("Cap.js subchallenge index overflow")?;
        emitter.emit(&mut salt, &mut targets, subchallenge);
        let mask = rules
            .mask()
            .context("Cap.js difficulty exceeds the supported 64-bit target width")?;
        let target = (u64::from(targets[0]) << 32) | u64::from(targets[1]);
        let (nonce, attempted) = solve_cap_decimal_gpu(gpu, &salt, target, mask).await?;
        attempted_nonces = attempted_nonces.saturating_add(attempted);
        solutions.push(nonce);
    }
    let elapsed = started.elapsed();
    let solve_ms = elapsed.as_secs_f64() * 1000.0;
    let (elapsed_us, hashrate) = elapsed_work_metadata(elapsed, attempted_nonces);
    Ok(Result {
        answer: json!({
            "token": token,
            "solutions": solutions,
            "_meta": {
                "elapsed_us": elapsed_us,
                "attempted_nonces": attempted_nonces,
                "hashrate": hashrate,
            },
            "attempted_nonces_total": attempted_nonces,
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_target_prefix_accepts_nibbles_and_rejects_invalid_shapes() -> anyhow::Result<()> {
        assert_eq!(
            cap_v2_prefix("a")?,
            (0xa000_0000_0000_0000, 0xf000_0000_0000_0000)
        );
        assert_eq!(
            cap_v2_prefix("12ab")?,
            (0x12ab_0000_0000_0000, 0xffff_0000_0000_0000)
        );
        assert!(cap_v2_prefix("").is_err());
        assert!(cap_v2_prefix("g").is_err());
        assert!(cap_v2_prefix("00000000000000000").is_err());
        Ok(())
    }

    #[test]
    fn rsw_native_blocks_match_direct_squaring() -> anyhow::Result<()> {
        // Full 8,192-square block boundary, even moduli, odd moduli, and
        // unreduced inputs. The oracle uses a distinct GMP operation sequence.
        for bits in [2_u32, 32, 64, 129, 1024, 2048] {
            for offset in [1_u8, 2_u8] {
                let modulus = (Integer::from(1) << bits) - Integer::from(offset);
                for rounds in [0_u32, 1, 2, 17, 129, 8_192, 8_193] {
                    for input in [Integer::from(0), Integer::from(1), modulus.clone() + 7] {
                        let mut expected = input.clone();
                        expected.modulo_mut(&modulus);
                        for _ in 0..rounds {
                            expected.square_mut();
                            expected.modulo_mut(&modulus);
                        }
                        assert_eq!(
                            repeated_mod_squaring(input, &modulus, rounds)?,
                            expected,
                            "bits={bits} offset={offset} rounds={rounds}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rsw_unsigned_hex_rejects_non_wire_formats() {
        for invalid in ["", "-2", "+2", " 2", "0x2", "2_", "GG"] {
            assert!(parse_rsw_hex(invalid).is_none(), "{invalid}");
        }
        assert_eq!(parse_rsw_hex("000f"), Some(Integer::from(15)));
        assert_eq!(parse_rsw_hex("ABC"), Some(Integer::from(0xABC)));
    }

    #[test]
    fn rsw_format_two_response_matches_wire_contract() -> anyhow::Result<()> {
        let result = solve(
            json!({
                "token": "fixture-token",
                "format": 2,
                "challenges": [{
                    "protocol": "rsw",
                    "payload": { "N": "11", "x": "2", "t": 3 }
                }]
            }),
            Implementation::Safe,
        )?;
        assert_eq!(result.implementation, "rsw");
        assert_eq!(result.answer["token"], "fixture-token");
        assert_eq!(result.answer["solutions"], json!([{ "y": "1" }]));
        assert_eq!(result.answer["attempted_nonces_total"], 0);
        Ok(())
    }

    #[test]
    fn format_two_instrumentation_is_not_a_proof_of_work_mode() -> anyhow::Result<()> {
        let error = solve(
            json!({
                "token": "fixture-token",
                "format": 2,
                "challenges": [{ "protocol": "instrumentation", "payload": {} }]
            }),
            Implementation::Auto,
        )
        .err()
        .context("instrumentation must be rejected")?;
        assert!(error.to_string().contains("browser-environment probe"));
        Ok(())
    }

    #[test]
    fn format_two_instrumentation_fails_before_prior_expensive_work() -> anyhow::Result<()> {
        // The RSW subchallenge must never be evaluated. Its invalid payload
        // makes an early validation failure deterministic rather than timed.
        let challenge = json!({
            "token": "fixture-token",
            "format": 2,
            "challenges": [
                { "protocol": "rsw", "payload": { "N": "invalid", "x": "2", "t": 0 } },
                { "protocol": "instrumentation", "payload": {} }
            ]
        });
        for error in [
            parse_cap_v2_envelope(challenge.clone())
                .err()
                .context("preflight must reject instrumentation")?
                .to_string(),
            solve(challenge, Implementation::Auto)
                .err()
                .context("CPU solver must reject instrumentation")?
                .to_string(),
        ] {
            assert!(
                error.contains("browser-environment probe"),
                "expected early instrumentation rejection, got {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn nonce_conversion_enforces_javascript_lossless_integer_limit() -> anyhow::Result<()> {
        assert_eq!(
            js_safe_integer_as_f64(JS_MAX_SAFE_INTEGER)?,
            9_007_199_254_740_991.0
        );
        assert!(js_safe_integer_as_f64(JS_MAX_SAFE_INTEGER + 1).is_err());
        Ok(())
    }

    #[test]
    fn hashwx_payload_rejects_unrepresentable_work_parameters() {
        let mut payload = CapV2HashWx {
            c: "00".repeat(32),
            d: 0,
            n: 1,
        };
        assert!(solve_cap_v2_hashwx(&payload).is_err());
        payload.d = 1;
        payload.n = 0;
        assert!(solve_cap_v2_hashwx(&payload).is_err());
        payload.n = 1_048_577;
        assert!(solve_cap_v2_hashwx(&payload).is_err());
    }
}
