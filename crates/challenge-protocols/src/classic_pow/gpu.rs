use std::{
    num::{NonZeroU8, NonZeroU64},
    time::Instant,
};

use anyhow::{Context, anyhow, bail, ensure};
use challenge_solvers::{
    build_mcaptcha_prefix,
    cerberus::ChallengeDescriptor as CerberusChallenge,
    compute_mask_goaway, compute_mask_zero_nibbles, compute_target_mcaptcha, extract128_be,
    goaway::GoAwayConfig,
    message::{DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LEADING_ZERO_BITS, SOLVE_TYPE_MASK},
};
use serde_json::{Value, json};

use super::{
    DEFAULT_LIMIT, HaphashChallenge, McaptchaRequest, Result, encode_words_le, finish_goaway,
};

/// Solve a classic proof-of-work protocol using the shared GPU context.
///
/// # Errors
/// Rejects unsupported protocols or malformed challenges, and propagates
/// device execution, nonce-budget and proof-validation errors.
pub async fn solve(
    gpu: &challenge_solvers::gpu::GpuContext,
    kind: &str,
    challenge: Value,
) -> anyhow::Result<Result> {
    match kind {
        "cerberus" => solve_cerberus_gpu(gpu, challenge).await,
        "goaway" => solve_goaway_gpu(gpu, challenge).await,
        "capjs" => super::capjs::solve_gpu(gpu, challenge).await,
        "mcaptcha" => solve_mcaptcha_gpu(gpu, challenge).await,
        "haphash" => solve_haphash_gpu(gpu, challenge).await,
        "altcha" => super::altcha::solve_gpu(gpu, challenge).await,
        "linear" => crate::linear_pow::solve_gpu(gpu, challenge).await,
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
