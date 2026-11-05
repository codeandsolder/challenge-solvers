use std::{num::NonZeroU8, time::Instant};

use anyhow::{Context, anyhow};
use challenge_solvers::{
    DecimalSolver, SingleBlockSolver, compute_mask_zero_nibbles,
    message::{DecimalMessage, SingleBlockMessage},
    solver::{SOLVE_TYPE_MASK, Solver},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const HEX: &[u8; 16] = b"0123456789abcdef";

#[derive(Debug, Clone, Deserialize)]
pub struct Challenge {
    pub data: String,
    pub difficulty: u8,
}

impl Challenge {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(1..=16).contains(&self.difficulty) {
            return Err(anyhow!("difficulty must be in 1..=16"));
        }
        if self.data.len() > 4096 {
            return Err(anyhow!("data is too large"));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    pub fn expected_work(&self) -> u64 {
        16_u64.saturating_pow(u32::from(self.difficulty))
    }

    #[cfg(feature = "gpu")]
    pub const fn gpu_compatible(&self) -> bool {
        self.data.len().is_multiple_of(64)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Answer {
    pub nonce: String,
    pub hash: String,
}

#[derive(Debug, Clone)]
pub struct BackendResult {
    pub answer: Answer,
    pub solve_ms: f64,
}

fn mask(difficulty: u8) -> anyhow::Result<u64> {
    let difficulty = NonZeroU8::new(difficulty).context("difficulty cannot be zero")?;
    Ok(compute_mask_zero_nibbles(difficulty))
}

fn hash_hex(words: [u32; 8]) -> String {
    hex::encode(
        words
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect::<Vec<_>>(),
    )
}

pub fn verify(challenge: &Challenge, answer: &Answer) -> bool {
    if challenge.validate().is_err() {
        return false;
    }
    let digest = Sha256::digest(format!("{}{}", challenge.data, answer.nonce).as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push(char::from(HEX[usize::from(byte >> 4)]));
        hex.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    let difficulty = usize::from(challenge.difficulty);
    hex == answer.hash && hex.starts_with(&"0".repeat(difficulty))
}

pub fn solve_cpu(challenge: &Challenge, limit: u64) -> anyhow::Result<BackendResult> {
    challenge.validate()?;
    let message = DecimalMessage::new(challenge.data.as_bytes(), 0)
        .context("unsupported Anubis challenge shape")?;
    let mut solver = DecimalSolver::from(message);
    solver.set_limit(limit);
    let started = Instant::now();
    let (nonce, hash) = solver
        .solve::<SOLVE_TYPE_MASK>(0, mask(challenge.difficulty)?)
        .context("CPU nonce search exhausted configured limit")?;
    let result = BackendResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hash_hex(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
    };
    anyhow::ensure!(
        verify(challenge, &result.answer),
        "CPU solver returned an invalid answer"
    );
    Ok(result)
}

pub fn calibrate_cpu(hash_limit: u64) -> anyhow::Result<f64> {
    let started = Instant::now();
    let mut attempted_total = 0_u64;
    let calibration_mask = mask(5)?;
    for sample in 0_u8..8 {
        let mut prefix = vec![b'a'; 128];
        prefix[0] = sample;
        let message =
            SingleBlockMessage::new(&prefix, 0).context("calibration message rejected")?;
        let mut solver = SingleBlockSolver::from(message);
        solver.set_limit(hash_limit);
        let _ = solver.solve_nonce_only::<SOLVE_TYPE_MASK>(0, calibration_mask);
        attempted_total = attempted_total.saturating_add(solver.attempted_nonces());
    }
    let elapsed = started.elapsed().as_secs_f64();
    anyhow::ensure!(
        attempted_total > 0 && elapsed > 0.0,
        "invalid CPU calibration sample"
    );
    Ok(calibration_work_as_f64(attempted_total) / elapsed)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "CPU calibration throughput is approximate telemetry"
)]
const fn calibration_work_as_f64(work: u64) -> f64 {
    work as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_rejects_invalid_difficulty_even_for_matching_hash() {
        let challenge = Challenge {
            data: "challenge".to_owned(),
            difficulty: 0,
        };
        let nonce = "0";
        let hash = hex::encode(Sha256::digest(
            format!("{}{}", challenge.data, nonce).as_bytes(),
        ));
        let answer = Answer {
            nonce: nonce.to_owned(),
            hash,
        };
        assert!(!verify(&challenge, &answer));
    }
}
