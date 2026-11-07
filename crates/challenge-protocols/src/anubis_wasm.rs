use std::{
    num::NonZeroU8,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Context, ensure};
use challenge_solvers::message::BinaryMessage;
use challenge_solvers::solver::SOLVE_TYPE_MASK;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    anubis::Answer,
    classic_pow::{Implementation, binary_solver},
};

const SEARCH_SPACE: u64 = 1_u64 << 32;
const MAX_DIFFICULTY: u32 = 64;
pub const ARGON2_IMPLEMENTATION: &str = "portable-simd";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Challenge {
    pub data: String,
    pub difficulty: u32,
}

impl Challenge {
    /// Validate the binary challenge framing and difficulty.
    ///
    /// # Errors
    /// Rejects difficulty outside 1–64 bits, malformed hex challenge data,
    /// or a decoded challenge outside the supported nonempty 4096-byte bound.
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_DIFFICULTY).contains(&self.difficulty),
            "difficulty must be in 1..={MAX_DIFFICULTY}"
        );
        let bytes = self.bytes()?;
        ensure!(
            !bytes.is_empty() && bytes.len() <= 4096,
            "challenge data length out of range"
        );
        Ok(())
    }

    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        hex::decode(&self.data).context("Anubis WASM challenge data must be hex")
    }
}

#[derive(Debug)]
pub struct Result {
    pub answer: Answer,
    pub solve_ms: f64,
    pub implementation: String,
}

fn prefix_mask(difficulty: u32) -> u64 {
    debug_assert!((1..=64).contains(&difficulty));
    if difficulty == 64 {
        u64::MAX
    } else {
        u64::MAX << (64 - difficulty)
    }
}

fn has_leading_zero_bits(hash: &[u8; 32], difficulty: u32) -> bool {
    if difficulty > 256 {
        return false;
    }
    let Ok(full_bytes) = usize::try_from(difficulty / 8) else {
        return false;
    };
    let remaining = difficulty % 8;
    if hash[..full_bytes].iter().any(|byte| *byte != 0) {
        return false;
    }
    remaining == 0 || (hash[full_bytes] & (0xff << (8 - remaining))) == 0
}

fn use_little_endian_nonce(data: &[u8]) -> bool {
    data.last().copied().unwrap_or(0) >= 128
}

fn sha256_hash(data: &[u8], nonce: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(if use_little_endian_nonce(data) {
        nonce.to_le_bytes()
    } else {
        nonce.to_be_bytes()
    });
    hasher.finalize().into()
}

// Reusable fixed-shape Anubis Argon2id arena per search worker.
struct Argon2Worker(anubis_argon2::Memory);

impl Argon2Worker {
    fn new() -> Self {
        Self(anubis_argon2::Memory::new())
    }

    fn hash(&mut self, data: &[u8], nonce: u32) -> anyhow::Result<[u8; 32]> {
        let nonce = u64::from(nonce);
        let salt = if use_little_endian_nonce(data) {
            nonce.to_le_bytes()
        } else {
            nonce.to_be_bytes()
        };
        anubis_argon2::hash(data, salt, &mut self.0).context("specialized Argon2id hash failed")
    }
}

fn argon2id_hash(data: &[u8], nonce: u32) -> anyhow::Result<[u8; 32]> {
    Argon2Worker::new().hash(data, nonce)
}

/// Validate a WASM Anubis SHA-256 proof, including protocol nonce byte order.
#[must_use]
pub fn verify_sha256(challenge: &Challenge, answer: &Answer) -> bool {
    if challenge.validate().is_err() {
        return false;
    }
    let Ok(data) = challenge.bytes() else {
        return false;
    };
    let Ok(nonce) = answer.nonce.parse::<u32>() else {
        return false;
    };
    let hash = sha256_hash(&data, nonce);
    hex::encode(hash) == answer.hash && has_leading_zero_bits(&hash, challenge.difficulty)
}

/// Validate a WASM Anubis Argon2id proof against the protocol digest.
#[must_use]
pub fn verify_argon2id(challenge: &Challenge, answer: &Answer) -> bool {
    if challenge.validate().is_err() {
        return false;
    }
    let Ok(data) = challenge.bytes() else {
        return false;
    };
    let Ok(nonce) = answer.nonce.parse::<u32>() else {
        return false;
    };
    let Ok(hash) = argon2id_hash(&data, nonce) else {
        return false;
    };
    hex::encode(hash) == answer.hash && has_leading_zero_bits(&hash, challenge.difficulty)
}

fn sha256_answer(
    challenge: &Challenge,
    data: &[u8],
    search_nonce: u64,
    hash_words: [u32; 8],
) -> anyhow::Result<Answer> {
    let search_nonce = u32::try_from(search_nonce).context("SHA-256 nonce does not fit u32")?;
    let protocol_nonce = if use_little_endian_nonce(data) {
        search_nonce
    } else {
        search_nonce.swap_bytes()
    };
    let hash = hash_words
        .into_iter()
        .flat_map(u32::to_be_bytes)
        .collect::<Vec<_>>();
    let answer = Answer {
        nonce: protocol_nonce.to_string(),
        hash: hex::encode(hash),
    };
    ensure!(
        verify_sha256(challenge, &answer),
        "optimized SHA-256 solver returned an invalid Anubis proof"
    );
    Ok(answer)
}

/// Solve a binary Anubis SHA-256 challenge with a selected CPU backend.
///
/// # Errors
/// Rejects invalid challenge framing or an unsupported backend, exhausted
/// nonce space, or an answer failing independent verification.
pub fn solve_sha256(challenge: &Challenge, implementation: Option<&str>) -> anyhow::Result<Result> {
    challenge.validate()?;
    let data = challenge.bytes()?;
    let implementation = Implementation::parse(implementation.unwrap_or("auto"))?;
    let nonce_bytes = NonZeroU8::new(4).context("four-byte Anubis nonce width is nonzero")?;
    let message = BinaryMessage::new(&data, nonce_bytes)
        .context("Anubis nonce width exceeds the binary solver layout")?;
    let mut solver = binary_solver(implementation, message)?;
    solver.set_limit(SEARCH_SPACE);

    let started = Instant::now();
    let (search_nonce, hash_words) = solver
        .solve(0, SOLVE_TYPE_MASK, prefix_mask(challenge.difficulty))
        .context("Anubis WASM SHA-256 search exhausted u32 nonce space")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let answer = sha256_answer(challenge, &data, search_nonce, hash_words)?;
    Ok(Result {
        answer,
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

#[cfg(feature = "gpu")]
/// Solve a binary Anubis SHA-256 challenge with the GPU backend.
///
/// # Errors
/// Rejects invalid challenges or layout, propagates GPU execution failures,
/// or reports exhausted search space or invalid proof validation.
pub async fn solve_sha256_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: &Challenge,
) -> anyhow::Result<Result> {
    challenge.validate()?;
    let data = challenge.bytes()?;
    let nonce_bytes = NonZeroU8::new(4).context("four-byte Anubis nonce width is nonzero")?;
    let message = BinaryMessage::new(&data, nonce_bytes)
        .context("Anubis nonce width exceeds the binary solver layout")?;
    let mut solver = challenge_solvers::gpu::BinarySolver::from(message);
    solver.set_limit(SEARCH_SPACE);

    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, 0, prefix_mask(challenge.difficulty))
        .await?
        .context("Anubis WASM SHA-256 GPU search exhausted u32 nonce space")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let answer = sha256_answer(challenge, &data, solution.nonce, solution.hash)?;
    Ok(Result {
        answer,
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

/// Solve an Anubis Argon2id challenge using the specified worker count.
///
/// # Errors
/// Rejects malformed challenges, invalid worker counts or backend options,
/// and propagates Argon2 computation, worker coordination or search errors.
pub fn solve_argon2id(
    challenge: &Challenge,
    implementation: Option<&str>,
    worker_count: usize,
) -> anyhow::Result<Result> {
    challenge.validate()?;
    ensure!(
        worker_count > 0,
        "Argon2id worker count must be greater than zero"
    );
    ensure!(
        matches!(implementation, None | Some("portable-simd")),
        "unsupported Argon2id implementation: {}",
        implementation.unwrap_or_default()
    );
    let data = challenge.bytes()?;
    let stride = u64::try_from(worker_count).context("Argon2id worker count overflow")?;
    let best_nonce = AtomicU64::new(SEARCH_SPACE);
    let winner = Mutex::new(None::<(u32, [u8; 32])>);
    let started = Instant::now();

    thread::scope(|scope| -> anyhow::Result<()> {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let best_nonce = &best_nonce;
            let winner = &winner;
            let data = &data;
            handles.push(scope.spawn(move || -> anyhow::Result<()> {
                let mut worker_memory = Argon2Worker::new();
                let mut nonce = u64::try_from(worker).context("Argon2id worker index overflow")?;

                while nonce < SEARCH_SPACE && nonce < best_nonce.load(Ordering::Acquire) {
                    let protocol_nonce = u32::try_from(nonce).context("Argon2id nonce overflow")?;
                    let hash = worker_memory.hash(data, protocol_nonce)?;
                    if has_leading_zero_bits(&hash, challenge.difficulty) {
                        let previous = best_nonce.fetch_min(nonce, Ordering::AcqRel);
                        if nonce < previous {
                            let mut guard = winner
                                .lock()
                                .map_err(|_| anyhow::anyhow!("Argon2id winner lock poisoned"))?;
                            let should_replace = match guard.as_ref() {
                                Some((current, _)) => protocol_nonce < *current,
                                None => true,
                            };
                            if should_replace {
                                *guard = Some((protocol_nonce, hash));
                            }
                        }
                        break;
                    }
                    nonce = nonce.checked_add(stride).unwrap_or(SEARCH_SPACE);
                }
                Ok(())
            }));
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("Argon2id worker panicked"))??;
        }
        Ok(())
    })?;

    let (nonce, hash) = winner
        .into_inner()
        .map_err(|_| anyhow::anyhow!("Argon2id winner lock poisoned"))?
        .context("Anubis Argon2id search exhausted u32 nonce space")?;
    let answer = Answer {
        nonce: nonce.to_string(),
        hash: hex::encode(hash),
    };
    ensure!(
        verify_argon2id(challenge, &answer),
        "Argon2id solver returned an invalid Anubis proof"
    );
    Ok(Result {
        answer,
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
        implementation: ARGON2_IMPLEMENTATION.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_solver_matches_upstream_little_endian_nonce_rule() -> anyhow::Result<()> {
        let challenge = Challenge {
            data: "0011223344556680".to_owned(),
            difficulty: 4,
        };
        let result = solve_sha256(&challenge, Some("safe"))?;
        assert!(verify_sha256(&challenge, &result.answer));
        Ok(())
    }

    #[test]
    fn sha256_solver_matches_upstream_big_endian_nonce_rule() -> anyhow::Result<()> {
        let challenge = Challenge {
            data: "001122334455667f".to_owned(),
            difficulty: 4,
        };
        let result = solve_sha256(&challenge, Some("safe"))?;
        assert!(verify_sha256(&challenge, &result.answer));
        Ok(())
    }

    #[test]
    fn argon2id_solver_matches_upstream_reference() -> anyhow::Result<()> {
        let challenge = Challenge {
            data: "0011223344556677".to_owned(),
            difficulty: 1,
        };
        let result = solve_argon2id(&challenge, None, 1)?;
        assert!(verify_argon2id(&challenge, &result.answer));
        Ok(())
    }

    #[test]
    fn argon2id_rejects_removed_implementation_alias() -> anyhow::Result<()> {
        let challenge = Challenge {
            data: "0011223344556677".to_owned(),
            difficulty: 1,
        };
        assert!(solve_argon2id(&challenge, Some("argon2-0.5.3"), 1).is_err());
        Ok(())
    }

    #[test]
    fn argon2id_reused_memory_matches_fresh_hashes() -> anyhow::Result<()> {
        let data = hex::decode("0011223344556680")?;
        let mut worker_memory = Argon2Worker::new();
        for nonce in [0, 1, 2, 17] {
            assert_eq!(
                worker_memory.hash(&data, nonce)?,
                argon2id_hash(&data, nonce)?
            );
        }
        Ok(())
    }

    #[test]
    fn argon2id_backend_matches_reference_for_both_nonce_endiannesses() -> anyhow::Result<()> {
        use argon2::{Algorithm, Argon2, Params, Version};
        let params = Params::new(19 * 1024, 2, 1, Some(32))?;
        let reference = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        for encoded in ["001122334455667f", "0011223344556680"] {
            let data = hex::decode(encoded)?;
            let mut worker_memory = Argon2Worker::new();
            for nonce in [0_u32, 17] {
                let salt = if use_little_endian_nonce(&data) {
                    u64::from(nonce).to_le_bytes()
                } else {
                    u64::from(nonce).to_be_bytes()
                };
                let mut digest = [0_u8; 32];
                reference.hash_password_into(&data, &salt, &mut digest)?;
                assert_eq!(worker_memory.hash(&data, nonce)?, digest);
            }
        }
        Ok(())
    }

    #[test]
    fn argon2id_parallel_search_preserves_lowest_nonce() -> anyhow::Result<()> {
        let challenge = Challenge {
            data: "0011223344556677".to_owned(),
            difficulty: 3,
        };
        let serial = solve_argon2id(&challenge, None, 1)?;
        let parallel = solve_argon2id(&challenge, None, 4)?;
        assert_eq!(parallel.answer.nonce, serial.answer.nonce);
        assert_eq!(parallel.answer.hash, serial.answer.hash);
        assert!(verify_argon2id(&challenge, &parallel.answer));
        Ok(())
    }
}
