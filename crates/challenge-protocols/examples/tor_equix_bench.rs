//! Reproducible Tor onion-service v1 Equi-X CPU throughput and latency benchmark.
//!
//! Run from the challenge-solvers workspace:
//! cargo +nightly-2026-11-03 run --locked --profile test -j 8
//!   -p challenge-protocols --example `tor_equix_bench` -- 512 3 3
//!
//! Output is JSONL: fixed-work throughput has identical *nonce inputs* for
//! 1/2/4/8 workers; end-to-end first-winner solve is separately labeled.
//! Keep the timed region free of proof verification and reporting.

use std::{env, thread, time::Instant};

use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use blake2::{Blake2b, Digest, digest::consts::U4};
use challenge_protocols::classic_pow;
use equix::{EquiX, Error as EquixError, HashError, SolverMemory};
use num_traits::ToPrimitive;
use serde_json::json;

const PERSONALIZATION: &[u8; 16] = b"Tor hs intro v1\0";
const START_NONCE: u128 = u128::from_le_bytes([0x55; 16]);
const BLINDED_ID: [u8; 32] = [0x11; 32];
const SEED: [u8; 32] = [0xaa; 32];
const WORKERS: [usize; 4] = [1, 2, 4, 8];

#[derive(Default)]
struct WorkerStats {
    attempts: usize,
    invalid_programs: usize,
    solutions: usize,
    accepted: usize,
    checksum: u64,
    witness: Option<([u8; 100], [u8; 16])>,
}

fn challenge_for(nonce: u128, effort: u32) -> [u8; 100] {
    let mut challenge = [0; 100];
    challenge[..16].copy_from_slice(PERSONALIZATION);
    challenge[16..48].copy_from_slice(&BLINDED_ID);
    challenge[48..80].copy_from_slice(&SEED);
    challenge[80..96].copy_from_slice(&nonce.to_le_bytes());
    challenge[96..100].copy_from_slice(&effort.to_be_bytes());
    challenge
}

fn attempt_worker(
    base: u128,
    effort: u32,
    total: usize,
    workers: usize,
    worker: usize,
) -> anyhow::Result<WorkerStats> {
    let mut stats = WorkerStats::default();
    let mut memory = SolverMemory::new();
    for index in (worker..total).step_by(workers) {
        let nonce = base
            .checked_add(u128::try_from(index)?)
            .context("nonce exhausted")?;
        let challenge = challenge_for(nonce, effort);
        stats.attempts += 1;
        match EquiX::new(&challenge) {
            Ok(equix) => {
                for solution in equix.solve_with_memory(&mut memory) {
                    let proof = solution.to_bytes();
                    stats.solutions += 1;
                    let mut hash = Blake2b::<U4>::new();
                    hash.update(challenge);
                    hash.update(proof);
                    let digest = hash.finalize();
                    let v = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
                    stats.accepted += usize::from(v.checked_mul(effort).is_some());
                    // Commutative checksum verifies the identical work set
                    // regardless of how it is partitioned among threads.
                    stats.checksum = stats
                        .checksum
                        .wrapping_add(u64::from_le_bytes(proof[..8].try_into()?) ^ u64::from(v));
                    if stats.witness.is_none() {
                        stats.witness = Some((challenge, proof));
                    }
                }
            }
            Err(EquixError::Hash(HashError::ProgramConstraints)) => {
                stats.invalid_programs += 1;
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(stats)
}

fn fixed_work(total: usize, workers: usize, effort: u32) -> anyhow::Result<(f64, WorkerStats)> {
    let started = Instant::now();
    let parts = thread::scope(|scope| -> anyhow::Result<Vec<WorkerStats>> {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(
                scope.spawn(move || attempt_worker(START_NONCE, effort, total, workers, worker)),
            );
        }
        let mut results = Vec::with_capacity(workers);
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("worker panicked"))??,
            );
        }
        Ok(results)
    })?;
    let millis = started.elapsed().as_secs_f64() * 1000.0;
    let mut combined = WorkerStats::default();
    for part in parts {
        if let Some((challenge, proof)) = part.witness {
            equix::verify_bytes(&challenge, &proof)?;
        }
        combined.attempts += part.attempts;
        combined.invalid_programs += part.invalid_programs;
        combined.solutions += part.solutions;
        combined.accepted += part.accepted;
        combined.checksum = combined.checksum.wrapping_add(part.checksum);
    }
    ensure!(combined.attempts == total, "fixed work count changed");
    Ok((millis, combined))
}

fn emit_fixed(total: usize, repeats: usize) -> anyhow::Result<()> {
    let effort = 2_048;
    let mut reference: Option<(usize, usize, usize, u64)> = None;
    for rep in 0..repeats {
        let workers = if rep % 2 == 0 { WORKERS } else { [8, 4, 2, 1] };
        for worker_count in workers {
            let (millis, result) = fixed_work(total, worker_count, effort)?;
            let signature = (
                result.invalid_programs,
                result.solutions,
                result.accepted,
                result.checksum,
            );
            if let Some(expected) = reference {
                ensure!(
                    expected == signature,
                    "partition changed Equi-X outputs: expected {expected:?}, actual {signature:?}"
                );
            } else {
                reference = Some(signature);
            }
            let per_second =
                total.to_f64().context("attempt count not representable")? * 1_000.0 / millis;
            println!(
                "{}",
                json!({
                    "mode": "fixed_work",
                    "repetition": rep,
                    "workers": worker_count,
                    "effort": effort,
                    "total_candidates": total,
                    "millis": millis,
                    "candidates_per_second": per_second,
                    "invalid_hashx_programs": result.invalid_programs,
                    "equix_solutions": result.solutions,
                    "accepted_solutions": result.accepted,
                    "checksum": format!("{:016x}", result.checksum),
                })
            );
        }
    }
    Ok(())
}

fn emit_solve(repeats: usize) -> anyhow::Result<()> {
    let id = hex::encode(BLINDED_ID);
    let seed = STANDARD.encode(SEED);
    for effort in [100_u32, 1_000] {
        for rep in 0..repeats {
            // Every worker count searches the same deterministic nonce space
            // for this repetition; the first winner need not be the same.
            let offset = u128::try_from(rep)? * 65_536 + u128::from(effort) * 65_536;
            let nonce = START_NONCE
                .checked_add(offset)
                .context("start nonce exhausted")?;
            let start_nonce = hex::encode(nonce.to_le_bytes());
            let workers = if rep % 2 == 0 { WORKERS } else { [8, 4, 2, 1] };
            for worker_count in workers {
                let started = Instant::now();
                let result = classic_pow::solve(
                    "tor-hs-v1",
                    json!({
                        "blinded_id": id,
                        "seed_b64": seed,
                        "effort": effort,
                        "nonce": start_nonce,
                    }),
                    Some("auto"),
                    worker_count,
                )?;
                let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let proof = hex::decode(
                    result.answer["extension_body"]
                        .as_str()
                        .context("missing Tor extension body")?,
                )?;
                ensure!(
                    classic_pow::verify_tor_hs_v1(&id, &seed, &proof)?,
                    "invalid Tor proof"
                );
                let attempts = result.answer["attempted_nonces"]
                    .as_u64()
                    .context("missing attempted nonce count")?;
                println!(
                    "{}",
                    json!({
                        "mode": "first_winner",
                        "repetition": rep,
                        "workers": worker_count,
                        "effort": effort,
                        "starting_nonce": start_nonce,
                        "winning_nonce": result.answer["nonce"],
                        "attempted_nonces": attempts,
                        "millis": result.solve_ms,
                        "wall_millis": wall_ms,
                        "proof_verified": true,
                    })
                );
            }
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let total = args
        .next()
        .map_or(Ok(512), |value| value.parse().context("fixed candidates"))?;
    let repetitions = args
        .next()
        .map_or(Ok(3), |value| value.parse().context("fixed repetitions"))?;
    let solve_repetitions = args
        .next()
        .map_or(Ok(3), |value| value.parse().context("solve repetitions"))?;
    ensure!(
        args.next().is_none(),
        "usage: tor_equix_bench [candidates] [fixed_reps] [solve_reps]"
    );
    ensure!(
        total > 0 && repetitions > 0 && solve_repetitions > 0,
        "all counts must be positive"
    );
    println!(
        "{}",
        json!({
            "mode": "metadata",
            "build_profile": "test opt-level=3 in solver workspace",
            "fixed_candidates": total,
            "fixed_repetitions": repetitions,
            "solve_repetitions": solve_repetitions,
            "initial_nonce": hex::encode(START_NONCE.to_le_bytes()),
        })
    );
    emit_fixed(total, repetitions)?;
    emit_solve(solve_repetitions)
}
