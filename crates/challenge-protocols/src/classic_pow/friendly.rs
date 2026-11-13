//! Friendly Captcha v1: independently implemented BLAKE2b-256 puzzle search.
//!
//! The signed puzzle is opaque: this module solves its public computation,
//! leaving signature, expiry, site binding, and replay checks to the issuer.

use std::{
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::Instant,
};

use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use blake2::{Blake2b256, Digest};
use num_traits::ToPrimitive;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Implementation, Result};
#[cfg(feature = "gpu")]
use challenge_solvers::gpu::{FriendlyBlake2bGpu, GpuContext};

#[derive(Deserialize)]
#[serde(untagged)]
enum PuzzleInput {
    Encoded(String),
    Object { puzzle: String },
}

impl PuzzleInput {
    fn encoded(&self) -> &str {
        match self {
            Self::Encoded(puzzle) | Self::Object { puzzle } => puzzle,
        }
    }
}

fn request(challenge: Value) -> anyhow::Result<PuzzleInput> {
    serde_json::from_value(challenge)
        .context("Friendly Captcha request must be a puzzle string or {puzzle: string}")
}

struct Puzzle<'a> {
    signature: &'a str,
    encoded: &'a str,
    padded: [u8; 128],
    solutions: u8,
    threshold: u32,
}

/// The v1 difficulty byte represents an exponential 32-bit threshold.
///
/// # Errors
/// The public difficulty domain maps entirely inside `u32`; errors indicate
/// a violated numeric assumption rather than silently altering the threshold.
fn threshold(difficulty: u8) -> anyhow::Result<u32> {
    let scaled = ((255.999 - f64::from(difficulty)) / 8.0).exp2();
    let result = scaled
        .to_u32()
        .context("Friendly Captcha difficulty does not map to u32")?;
    ensure!(result > 0, "Friendly Captcha threshold must be positive");
    Ok(result)
}

fn parse(puzzle: &str) -> anyhow::Result<Puzzle<'_>> {
    let (signature, encoded) = puzzle
        .split_once('.')
        .context("Friendly Captcha puzzle must be signature.base64")?;
    ensure!(!signature.is_empty(), "Friendly Captcha signature is empty");
    let bytes = STANDARD
        .decode(encoded)
        .context("invalid Friendly Captcha puzzle base64")?;
    ensure!(
        (32..=64).contains(&bytes.len()),
        "Friendly Captcha v1 puzzle must contain 32..=64 bytes"
    );
    ensure!(
        bytes[12] == 1,
        "unsupported Friendly Captcha puzzle version {}",
        bytes[12]
    );
    ensure!(
        bytes[14] > 0,
        "Friendly Captcha solution count must be nonzero"
    );
    let mut padded = [0_u8; 128];
    padded[..bytes.len()].copy_from_slice(&bytes);
    Ok(Puzzle {
        signature,
        encoded,
        padded,
        solutions: bytes[14],
        threshold: threshold(bytes[15])?,
    })
}

const fn diagnostics(seconds: u16) -> [u8; 3] {
    let duration = seconds.to_be_bytes();
    [0, duration[0], duration[1]]
}

fn matches(padded: &[u8; 128], threshold: u32) -> bool {
    let digest = Blake2b256::digest(padded);
    u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) < threshold
}

fn search(puzzle: &Puzzle<'_>, index: u8) -> anyhow::Result<([u8; 8], u64)> {
    let mut input = puzzle.padded;
    // Upstream uses one independently searched 2^32-nonce bank per solution.
    // These disjoint banks also guarantee uniqueness of all returned values.
    input[120] = index;
    for counter in 0..=u32::MAX {
        input[124..128].copy_from_slice(&counter.to_le_bytes());
        if matches(&input, puzzle.threshold) {
            let mut nonce = [0_u8; 8];
            nonce.copy_from_slice(&input[120..128]);
            return Ok((nonce, u64::from(counter) + 1));
        }
    }
    anyhow::bail!("Friendly Captcha nonce bank exhausted for index {index}")
}

/// Verify the computational portion of a Friendly Captcha v1 solution.
///
/// The caller or issuing service must separately validate the puzzle
/// signature, expiration, account/site identity, and replay status.
///
/// # Errors
/// Rejects malformed or unsupported puzzles.
pub fn verify(puzzle: &str, solutions: &[u8]) -> anyhow::Result<bool> {
    let puzzle = parse(puzzle)?;
    if solutions.len() != usize::from(puzzle.solutions) * 8 {
        return Ok(false);
    }
    let mut seen = HashSet::with_capacity(usize::from(puzzle.solutions));
    for solution in solutions.as_chunks::<8>().0 {
        let mut nonce = [0_u8; 8];
        nonce.copy_from_slice(solution);
        if !seen.insert(nonce) {
            return Ok(false);
        }
        let mut input = puzzle.padded;
        input[120..128].copy_from_slice(&nonce);
        if !matches(&input, puzzle.threshold) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Suggested CPU reservation for independently searchable solution banks.
///
/// Cheap puzzles do not spawn needless threads; expensive puzzles can use
/// idle CPU permits, capped by the number of independent solutions.
#[must_use]
pub fn suggested_workers(challenge: &Value, available: usize) -> Option<usize> {
    if available == 0 {
        return None;
    }
    let encoded = challenge
        .as_str()
        .or_else(|| challenge.get("puzzle")?.as_str())?;
    let puzzle = parse(encoded).ok()?;
    let expected_per_bank = (u64::from(u32::MAX) + 1).div_ceil(u64::from(puzzle.threshold));
    let expected_total = expected_per_bank.saturating_mul(u64::from(puzzle.solutions));
    let worthwhile = usize::try_from(expected_total.div_ceil(16_384))
        .unwrap_or(usize::MAX)
        .max(1);
    Some(worthwhile.min(usize::from(puzzle.solutions)).min(available))
}

/// Expected independent candidate-hash evaluations for a valid Friendly v1 puzzle.
/// Used by automatic GPU placement to avoid routing malformed puzzle envelopes.
#[cfg(feature = "gpu")]
pub(super) fn auto_gpu_expected_work(challenge: &Value) -> Option<u128> {
    let request = request(challenge.clone()).ok()?;
    let puzzle = parse(request.encoded()).ok()?;
    let expected_per_bank = (u128::from(u32::MAX) + 1).div_ceil(u128::from(puzzle.threshold));
    Some(u128::from(puzzle.solutions) * expected_per_bank)
}

pub(super) fn solve(
    challenge: Value,
    implementation: Implementation,
    worker_count: usize,
) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "Friendly Captcha BLAKE2b supports only auto and safe CPU implementations"
    );
    let request = request(challenge)?;
    let puzzle = parse(request.encoded())?;
    let started = Instant::now();
    ensure!(
        worker_count > 0,
        "Friendly Captcha requires at least one CPU worker"
    );
    let count = usize::from(puzzle.solutions);
    let workers = worker_count.min(count);
    let mut solutions = vec![None; count];
    if workers == 1 {
        for (i, slot) in solutions.iter_mut().enumerate() {
            *slot = Some(search(&puzzle, u8::try_from(i)?)?);
        }
    } else {
        let next = AtomicUsize::new(0);
        thread::scope(|scope| -> anyhow::Result<()> {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let next = &next;
                let puzzle = &puzzle;
                handles.push(scope.spawn(move || -> anyhow::Result<Vec<_>> {
                    let mut found = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        let result = search(puzzle, u8::try_from(index)?)?;
                        found.push((index, result));
                    }
                    Ok(found)
                }));
            }
            for handle in handles {
                for (index, result) in handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("Friendly Captcha search worker panicked"))??
                {
                    solutions[index] = Some(result);
                }
            }
            Ok(())
        })?;
    }
    let mut bytes = Vec::with_capacity(count * 8);
    let mut attempts = 0_u64;
    for solution in solutions {
        let (nonce, count) = solution.context("Friendly Captcha bank missing a solution")?;
        bytes.extend_from_slice(&nonce);
        attempts = attempts
            .checked_add(count)
            .context("Friendly Captcha attempt count overflow")?;
    }
    Ok(finish(
        &puzzle,
        &bytes,
        attempts,
        started,
        "rustcrypto-blake2b",
    ))
}

fn finish(
    puzzle: &Puzzle<'_>,
    solutions: &[u8],
    attempted_nonces: u64,
    started: Instant,
    implementation: &str,
) -> Result {
    let elapsed = started.elapsed();
    let seconds = u16::try_from(elapsed.as_secs()).unwrap_or(u16::MAX);
    let response = format!(
        "{}.{}.{}.{}",
        puzzle.signature,
        puzzle.encoded,
        STANDARD.encode(solutions),
        STANDARD.encode(diagnostics(seconds))
    );
    Result {
        answer: json!({
            "response": response,
            "attempted_nonces": attempted_nonces,
            "solution_count": puzzle.solutions,
        }),
        solve_ms: elapsed.as_secs_f64() * 1000.0,
        implementation: implementation.to_owned(),
    }
}

/// Search every independent Friendly Captcha v1 bank on the GPU.
/// GPU attempt count is the upper bound of submitted candidates, not
/// the exact number of threads that evaluated a candidate.
///
/// # Errors
/// Rejects malformed puzzles, incompatible devices, exhausted nonce domains
/// and independently unverified GPU proof candidates.
#[cfg(feature = "gpu")]
pub(super) async fn solve_gpu(gpu: &GpuContext, challenge: Value) -> anyhow::Result<Result> {
    let request = request(challenge)?;
    let puzzle = parse(request.encoded())?;
    let started = Instant::now();
    let solver = FriendlyBlake2bGpu::new(gpu)?;
    let mut solutions = Vec::with_capacity(usize::from(puzzle.solutions) * 8);
    let mut attempts = 0u64;
    let batch_capacity = solver.batch_capacity().min(262_144);
    ensure!(
        batch_capacity > 0,
        "Friendly GPU cannot submit any candidates"
    );
    let domain = u64::from(u32::MAX) + 1;
    for bank in 0..puzzle.solutions {
        let mut base = 0u64;
        loop {
            ensure!(base < domain, "Friendly GPU exhausted nonce bank {bank}");
            let count = u32::try_from((domain - base).min(batch_capacity))?;
            let found = solver
                .scan(
                    &puzzle.padded,
                    bank,
                    u32::try_from(base)?,
                    count,
                    puzzle.threshold,
                )
                .await?;
            attempts = attempts
                .checked_add(u64::from(count))
                .context("Friendly GPU dispatched nonce count overflow")?;
            if let Some(nonce) = found {
                let mut input = puzzle.padded;
                input[120] = bank;
                input[124..128].copy_from_slice(&nonce.to_le_bytes());
                ensure!(
                    matches(&input, puzzle.threshold),
                    "Friendly GPU returned a BLAKE2b-256 nonce rejected by RustCrypto"
                );
                solutions.extend_from_slice(&input[120..128]);
                break;
            }
            base += u64::from(count);
        }
    }
    ensure!(
        verify(request.encoded(), &solutions)?,
        "Friendly GPU complete proof failed independent RustCrypto verification"
    );
    Ok(finish(
        &puzzle,
        &solutions,
        attempts,
        started,
        "wgpu-blake2b-256",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    fn block_on_test<F: std::future::Future>(future: F) -> F::Output {
        use std::{
            sync::Arc,
            task::{Context, Poll, Wake, Waker},
        };

        struct ThreadWaker(std::thread::Thread);
        impl Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(result) => return result,
                Poll::Pending => std::thread::park_timeout(std::time::Duration::from_millis(100)),
            }
        }
    }

    fn make_puzzle(difficulty: u8, count: u8, extra: usize) -> String {
        let mut bytes = vec![0_u8; 32 + extra];
        bytes[12] = 1;
        bytes[14] = count;
        bytes[15] = difficulty;
        bytes[24..32].copy_from_slice(b"nonce123");
        format!("fixture.{}", STANDARD.encode(bytes))
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires physical Vulkan GPU with SHADER_INT64"]
    fn friendly_gpu_full_protocol_roundtrip_and_cpu_latency() -> anyhow::Result<()> {
        use std::hint::black_box;

        let gpu = block_on_test(GpuContext::create())?;
        for (difficulty, count, extra) in [(80, 1, 0), (112, 3, 11), (128, 16, 32)] {
            let original = make_puzzle(difficulty, count, extra);
            let challenge = json!({"puzzle": original});
            let started = Instant::now();
            let gpu_result = block_on_test(solve_gpu(&gpu, challenge.clone()))?;
            let gpu_ms = started.elapsed().as_secs_f64() * 1e3;
            let response = gpu_result.answer["response"]
                .as_str()
                .context("GPU response missing")?;
            let fields: Vec<_> = response.split('.').collect();
            ensure!(fields.len() == 4, "unexpected Friendly response parts");
            let proof = STANDARD.decode(fields[2])?;
            ensure!(
                verify(&original, &proof)?,
                "Friendly GPU complete proof invalid"
            );
            for (bank, nonce) in proof.chunks_exact(8).enumerate() {
                ensure!(usize::from(nonce[0]) == bank, "unexpected Friendly bank");
                ensure!(nonce[1..4] == [0, 0, 0], "incorrect reserved nonce bytes");
            }
            println!(
                "FRIENDLY_GPU_FULL {{\"difficulty\":{difficulty},\"banks\":{count},\"gpu_ms\":{gpu_ms},\"solve_ms\":{},\"attempted\":{}}}",
                gpu_result.solve_ms, gpu_result.answer["attempted_nonces"]
            );
            if count == 16 {
                // Pair on the same fixture and same hardware. Each proof is
                // checked independently; no requirement to find the same nonce.
                for trial in 0..5 {
                    let started = Instant::now();
                    let gpu_result = block_on_test(solve_gpu(&gpu, challenge.clone()))?;
                    let gpu_ms = started.elapsed().as_secs_f64() * 1e3;
                    let encoded = gpu_result.answer["response"]
                        .as_str()
                        .context("GPU response missing")?;
                    let proof = STANDARD.decode(
                        encoded
                            .split('.')
                            .nth(2)
                            .context("GPU response fields missing")?,
                    )?;
                    ensure!(verify(&original, &proof)?, "GPU proof invalid");
                    println!(
                        "FRIENDLY_GPU_E2E {{\"difficulty\":{difficulty},\"banks\":{count},\"trial\":{trial},\"gpu_ms\":{gpu_ms}}}"
                    );
                    for workers in [1, 2, 4] {
                        let started = Instant::now();
                        let cpu =
                            black_box(solve(challenge.clone(), Implementation::Auto, workers)?);
                        let cpu_ms = started.elapsed().as_secs_f64() * 1e3;
                        let response = cpu.answer["response"]
                            .as_str()
                            .context("CPU response missing")?;
                        let proof = STANDARD.decode(
                            response
                                .split('.')
                                .nth(2)
                                .context("CPU response fields missing")?,
                        )?;
                        ensure!(verify(&original, &proof)?, "CPU proof invalid");
                        println!(
                            "FRIENDLY_CPU_E2E {{\"difficulty\":{difficulty},\"banks\":{count},\"trial\":{trial},\"workers\":{workers},\"cpu_ms\":{cpu_ms}}}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn threshold_is_monotonic_with_specified_bounds() -> anyhow::Result<()> {
        assert_eq!(threshold(255)?, 1);
        assert_eq!(diagnostics(0x1234), [0, 0x12, 0x34]);
        assert!(threshold(0)? > 4_290_000_000);
        for d in 1..=u8::MAX {
            assert!(threshold(d - 1)? >= threshold(d)?);
        }
        Ok(())
    }

    #[test]
    fn full_response_roundtrip_and_rejection() -> anyhow::Result<()> {
        for extra in [0, 11, 32] {
            let puzzle = make_puzzle(112, 3, extra);
            let result = solve(json!({"puzzle":puzzle}), Implementation::Auto, 1)?;
            let response = result.answer["response"]
                .as_str()
                .context("missing response")?;
            let fields: Vec<_> = response.split('.').collect();
            assert_eq!(fields.len(), 4);
            assert_eq!(fields[0], "fixture");
            assert_eq!(
                fields[1],
                puzzle.split_once('.').context("fixture separator")?.1
            );
            let solutions = STANDARD.decode(fields[2])?;
            assert_eq!(solutions.len(), 24);
            assert!(verify(&puzzle, &solutions)?);
            assert_eq!(STANDARD.decode(fields[3])?.len(), 3);
            let mut duplicated = solutions.clone();
            duplicated[8..16].copy_from_slice(&solutions[0..8]);
            assert!(!verify(&puzzle, &duplicated)?);
            assert!(!verify(&puzzle, &solutions[..16])?);
        }
        Ok(())
    }

    #[test]
    fn independently_captured_official_wasm_solution_verifies() -> anyhow::Result<()> {
        // Captured from the official v1 WASM widget by the MIT-licensed
        // opastorello/friendlycaptcha-solver verification suite.
        const SIGNATURE: &str = "fb929da33c74c6fb67d718727098f29d";
        const PUZZLE: &str = "aqhzM93jWYeGbQijAQwtjQAAAAAAAAAATm3zV+Ldtt0=";
        const SOLUTIONS: &str = concat!(
            "AAAAAM+YAQABAAAALtsBAAIAAAB2IwEAAwAAABsLAwAEAAAAjR8AAAUAAADqzAUABgAAAJQTAwAHAAAAXh0BAAgAAACt0wMACQAAAFAOAAAKAAAAOOcHAAsAAABjVwUADAAAAOpVAAANAAAA4g4EAA4AAABIcQIADwAAAD3oAAAQAAAAr+4AABEAAAAkYgYAEgAAAGOiBAATAAAAVQoEABQAAADkrwQAFQAAAFWqAgAWAAAA0W0EABcAAAC1gwgAGAAAAOopAwAZAAAARdMCABoAAAADEAMAGwAAAATvAAAcAAAAUxcBAB0AAACFTAMAHgAAAInPAAAfAAAA9zIAACAAAAAwBQMAIQAAADYXAgAiAAAAvdoBACMAAAAeRAkAJAAAAAu9AQAlAAAAshsAACYAAAB96wAAJwAAADluAQAoAAAAK2gAACkAAADzKQMAKgAAAAdfAgArAAAA+DwCACwAAABiWAIA",
        );
        let puzzle = format!("{SIGNATURE}.{PUZZLE}");
        let decoded = STANDARD.decode(SOLUTIONS)?;
        assert_eq!(decoded.len(), 45 * 8);
        assert!(verify(&puzzle, &decoded)?);
        let mut altered = decoded.clone();
        altered[7] ^= 0x80;
        assert!(!verify(&puzzle, &altered)?);
        Ok(())
    }

    #[test]
    fn serial_and_parallel_solvers_agree_under_parallel_bank_scheduling() -> anyhow::Result<()> {
        for extra in [0, 32] {
            let puzzle = make_puzzle(96, 8, extra);
            let challenge = json!({"puzzle": puzzle});
            assert_eq!(suggested_workers(&challenge, 1), Some(1));
            assert!(suggested_workers(&challenge, 8).is_some_and(|n| n > 1));
            let reference = solve(challenge.clone(), Implementation::Safe, 1)?;
            let accelerated = solve(challenge.clone(), Implementation::Auto, 1)?;
            let parallel = solve(challenge, Implementation::Auto, 4)?;
            assert_eq!(
                accelerated.answer["response"]
                    .as_str()
                    .and_then(|s| s.split('.').nth(2)),
                reference.answer["response"]
                    .as_str()
                    .and_then(|s| s.split('.').nth(2))
            );
            assert_eq!(
                parallel.answer["response"]
                    .as_str()
                    .and_then(|s| s.split('.').nth(2)),
                reference.answer["response"]
                    .as_str()
                    .and_then(|s| s.split('.').nth(2))
            );
            assert_eq!(
                reference.answer["attempted_nonces"],
                accelerated.answer["attempted_nonces"]
            );
            assert_eq!(
                reference.answer["attempted_nonces"],
                parallel.answer["attempted_nonces"]
            );
            assert_eq!(parallel.answer["solution_count"], 8);
        }
        Ok(())
    }

    #[test]
    fn rejects_malformed_puzzles_and_invalid_cpu_selectors() -> anyhow::Result<()> {
        for puzzle in ["bad", ".YWJj", "x.%%%", "x.AAAA", "x.a.b"] {
            assert!(parse(puzzle).is_err(), "accepted malformed: {puzzle}");
        }
        let mut invalid = vec![0_u8; 32];
        invalid[14] = 1;
        assert!(parse(&format!("sig.{}", STANDARD.encode(&invalid))).is_err());
        invalid[12] = 1;
        invalid[14] = 0;
        assert!(parse(&format!("sig.{}", STANDARD.encode(&invalid))).is_err());
        let request = json!({"puzzle":make_puzzle(0, 1, 0)});
        assert!(solve(request, Implementation::ShaNi, 1).is_err());
        Ok(())
    }
}
