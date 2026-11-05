use std::{
    num::{NonZeroU8, NonZeroU64},
    time::Instant,
};

use anyhow::{Context, anyhow, bail, ensure};
use challenge_solvers::{
    build_mcaptcha_prefix,
    capjs::ChallengeDescriptor as CapJsChallenge,
    cerberus::ChallengeDescriptor as CerberusChallenge,
    compute_mask_goaway, compute_mask_zero_nibbles, compute_target_mcaptcha, extract128_be,
    goaway::GoAwayConfig,
    message::{CapJSEmitter, DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LEADING_ZERO_BITS, SOLVE_TYPE_MASK},
};
use serde_json::{Value, json};

use super::{
    AltchaChallenge, AltchaCounterMode, CapV2Challenge, CapV2Envelope, DEFAULT_LIMIT,
    HaphashChallenge, McaptchaRequest, Result, altcha_counter_mode, altcha_fast_case,
    cap_v2_backend_name, cap_v2_prefix, elapsed_work_metadata, encode_words_be, encode_words_le,
    finish_goaway, js_safe_integer_as_f64, solve_cap_v2_hashwx, solve_cap_v2_rsw,
};

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

pub async fn solve(
    gpu: &challenge_solvers::gpu::GpuContext,
    kind: &str,
    challenge: Value,
) -> anyhow::Result<Result> {
    match kind {
        "cerberus" => solve_cerberus_gpu(gpu, challenge).await,
        "goaway" => solve_goaway_gpu(gpu, challenge).await,
        "capjs" => solve_capjs_gpu(gpu, challenge).await,
        "mcaptcha" => solve_mcaptcha_gpu(gpu, challenge).await,
        "haphash" => solve_haphash_gpu(gpu, challenge).await,
        "altcha" => solve_altcha_gpu(gpu, challenge).await,
        _ => bail!("unsupported classic GPU solver: {kind}"),
    }
}

async fn solve_cerberus_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    ensure!(
        descriptor.difficulty().get() <= 16,
        "Cerberus difficulty above 16 dubits is unsupported by upstream clients"
    );
    let started = Instant::now();
    let mut solution = None;
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = descriptor.build_msg(bank) else {
            break;
        };
        let mut solver = challenge_solvers::gpu::CerberusSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver
            .solve::<SOLVE_TYPE_MASK>(gpu, 0, descriptor.mask())
            .await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("Cerberus GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "solution": solution.nonce.to_string(),
            "response": encode_words_le(solution.hash),
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

async fn solve_goaway_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    ensure!(
        config.difficulty() <= 256,
        "go-away difficulty above the 256-bit SHA-256 digest width is unsatisfiable"
    );
    let accelerated_mask = u8::try_from(config.difficulty())
        .ok()
        .and_then(NonZeroU8::new)
        .filter(|difficulty| difficulty.get() <= 64)
        .map(compute_mask_goaway);
    let challenge_bytes: [u8; 64] = config
        .challenge()
        .as_bytes()
        .try_into()
        .context("go-away challenge must be exactly 64 bytes")?;
    let mut message =
        GoAwayMessage::new_hex(&challenge_bytes, 0).context("invalid go-away hex challenge")?;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solution = None;
    for high_word in 0_u32.. {
        message.set_high_word(high_word);
        let mut solver = challenge_solvers::gpu::GoAwaySolver::from(message.clone());
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = if let Some(mask) = accelerated_mask {
            solver.solve::<SOLVE_TYPE_MASK>(gpu, 0, mask).await?
        } else {
            solver
                .solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(gpu, config.difficulty(), u64::MAX)
                .await?
        };
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("go-away GPU search exhausted configured limit")?;
    Ok(finish_goaway(
        &config,
        "wgpu",
        (solution.nonce, solution.hash),
        attempted_nonces,
        started,
    ))
}

async fn solve_capjs_v2_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let envelope: CapV2Envelope =
        serde_json::from_value(challenge).context("invalid Cap.js format-2 challenge")?;
    ensure!(envelope.format == 2, "invalid Cap.js format-2 marker");
    ensure!(
        !envelope.challenges.is_empty(),
        "Cap.js format-2 challenge list is empty"
    );

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

async fn solve_capjs_gpu(
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

async fn solve_haphash_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let challenge: HaphashChallenge =
        serde_json::from_value(challenge).context("invalid haphash challenge")?;
    ensure!(
        challenge.difficulty <= 64,
        "haphash difficulty above the 64-nibble SHA-256 digest width is unsatisfiable"
    );
    let prefix = challenge.prefix.as_bytes();
    ensure!(!prefix.is_empty(), "haphash prefix must not be empty");
    let fast_mask = if challenge.difficulty == 0 {
        Some(0)
    } else {
        NonZeroU8::new(challenge.difficulty)
            .filter(|difficulty| difficulty.get() <= 16)
            .map(compute_mask_zero_nibbles)
    };
    let full_digest_bits = u64::from(challenge.difficulty) * 4;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(prefix, bank) else {
            break;
        };
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = if let Some(mask) = fast_mask {
            solver.solve::<SOLVE_TYPE_MASK>(gpu, 0, mask).await?
        } else {
            solver
                .solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(gpu, full_digest_bits, u64::MAX)
                .await?
        };
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
        if let Some(solution) = result {
            return Ok(Result {
                answer: json!({
                    "nonce": solution.nonce.to_string(),
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms: started.elapsed().as_secs_f64() * 1000.0,
                implementation: "wgpu".to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("haphash GPU search exhausted configured limit"))
}

async fn solve_mcaptcha_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let request: McaptchaRequest =
        serde_json::from_value(challenge).context("invalid mCaptcha request")?;
    let difficulty_factor = NonZeroU64::new(u64::from(request.config.difficulty_factor))
        .context("mCaptcha difficulty factor must be positive")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt)
        .context("mCaptcha string length exceeds the wire format")?;
    let target = compute_target_mcaptcha(difficulty_factor);
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(&prefix, bank) else {
            break;
        };
        let mut solver = challenge_solvers::gpu::DecimalSolver::from(message);
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve::<SOLVE_TYPE_GT>(gpu, target, u64::MAX).await?;
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted_nonces());
        if let Some(solution) = result {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({
                    "string": request.config.string,
                    "result": extract128_be(solution.hash).to_string(),
                    "nonce": solution.nonce,
                    "key": request.key,
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms,
                implementation: "wgpu".to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("mCaptcha GPU search exhausted configured limit"))
}

async fn solve_altcha_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    ensure!(
        altcha_counter_mode(&challenge)? == AltchaCounterMode::Uint32,
        "ALTCHA WGPU supports uint32 counter mode only"
    );
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let (message, target, mask, key_length) = altcha_fast_case(&challenge)
        .context("ALTCHA challenge shape is unsupported by the WGPU fast path")?;
    let mut solver = challenge_solvers::gpu::AltchaSha256Solver::from(message);
    solver.set_limit(u64::from(u32::MAX) + 1);
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let derived_key = encode_words_be(solution.hash);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Result {
        answer: json!({
            "counter": solution.nonce,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted_nonces(),
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}
