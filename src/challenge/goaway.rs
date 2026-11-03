//! `GoAway` specific protocol structures.

use alloc::string::String;

use crate::{
    compute_mask_goaway,
    message::GoAwayMessage,
    solver::{SOLVE_TYPE_LEADING_ZERO_BITS, SOLVE_TYPE_MASK, Solver},
};

#[derive(serde::Deserialize, Debug)]
/// `GoAway` "js-pow-sha256" `PoW` challenge configuration.
pub struct GoAwayConfig {
    /// The challenge. (JSON key: `challenge`)
    pub challenge: String,
    /// The target. (JSON key: `target`)
    // target: String,
    pub difficulty: u64,
}

impl GoAwayConfig {
    /// Get the challenge of a `GoAway` "js-pow-sha256" `PoW`.
    #[must_use]
    pub fn challenge(&self) -> &str {
        &self.challenge
    }

    /// Get the difficulty of a `GoAway` "js-pow-sha256" `PoW`.
    #[must_use]
    pub const fn difficulty(&self) -> u64 {
        self.difficulty
    }

    /// Estimate the workload of a `GoAway` "js-pow-sha256" `PoW`.
    #[must_use]
    pub const fn estimated_workload(&self) -> u64 {
        if self.difficulty >= 64 {
            u64::MAX
        } else {
            1_u64 << self.difficulty
        }
    }

    /// Solve a `GoAway` "js-pow-sha256" `PoW`.
    #[must_use]
    pub fn solve(&self) -> (Option<(u64, [u32; 8])>, u64) {
        self.solve_with_limit(u64::MAX)
    }

    /// Solve a `GoAway` "js-pow-sha256" `PoW` with a limit.
    pub fn solve_with_limit(&self, limit: u64) -> (Option<(u64, [u32; 8])>, u64) {
        let Some(mut message) = self
            .challenge
            .as_bytes()
            .try_into()
            .ok()
            .and_then(|x| GoAwayMessage::new_hex(x, 0))
        else {
            return (None, 0);
        };

        if self.difficulty > 256 {
            return (None, 0);
        }

        let accelerated_mask = u8::try_from(self.difficulty)
            .ok()
            .and_then(core::num::NonZeroU8::new)
            .filter(|difficulty| difficulty.get() <= 64)
            .map(compute_mask_goaway);

        let mut attempted_nonces = 0_u64;
        for high_word in 0.. {
            message.set_high_word(high_word);
            let mut solver = crate::GoAwaySolver::from(message.clone());
            solver.set_limit(limit.saturating_sub(attempted_nonces));
            let res = if let Some(mask) = accelerated_mask {
                solver.solve::<{ SOLVE_TYPE_MASK }>(0, mask)
            } else {
                solver.solve::<{ SOLVE_TYPE_LEADING_ZERO_BITS }>(self.difficulty, u64::MAX)
            };
            attempted_nonces += solver.attempted_nonces();
            if let Some(res) = res {
                return (Some(res), attempted_nonces);
            }
            if limit <= attempted_nonces {
                break;
            }
        }
        (None, attempted_nonces)
    }
}

#[cfg(test)]
fn hash_nonce(message: &GoAwayMessage, nonce: u64) -> [u32; 8] {
    let mut block = [0_u32; 16];
    block[..8].copy_from_slice(&message.challenge);
    let nonce_bytes = nonce.to_be_bytes();
    block[8] = u32::from_be_bytes([
        nonce_bytes[0],
        nonce_bytes[1],
        nonce_bytes[2],
        nonce_bytes[3],
    ]);
    block[9] = u32::from_be_bytes([
        nonce_bytes[4],
        nonce_bytes[5],
        nonce_bytes[6],
        nonce_bytes[7],
    ]);
    block[10] = 0x8000_0000;
    block[15] = 320;
    let mut state = crate::sha256::IV;
    crate::sha256::digest_block(&mut state, &block);
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goaway_solver_terminates() {
        let config = GoAwayConfig {
            challenge: String::from("abc"),
            difficulty: 20,
        };

        let (result, attempted_nonces) = config.solve_with_limit(1000);
        assert!(result.is_none());
        assert!(attempted_nonces < 2000);
    }

    #[test]
    fn zero_difficulty_is_trivially_solved() -> anyhow::Result<()> {
        let config = GoAwayConfig {
            challenge: "00".repeat(32),
            difficulty: 0,
        };
        let (result, attempted) = config.solve_with_limit(1);
        assert_eq!(attempted, 1);
        let (nonce, _) =
            result.ok_or_else(|| anyhow::anyhow!("difficulty zero returned no solution"))?;
        assert_eq!(nonce, 0);
        Ok(())
    }

    #[test]
    fn difficulty_above_digest_width_is_unsatisfiable_without_work() {
        let config = GoAwayConfig {
            challenge: "00".repeat(32),
            difficulty: 257,
        };
        assert_eq!(config.solve_with_limit(1_000_000), (None, 0));
    }

    #[test]
    fn full_digest_solver_attempts_valid_difficulty_above_64() {
        let config = GoAwayConfig {
            challenge: "00".repeat(32),
            difficulty: 65,
        };
        let (result, attempted) = config.solve_with_limit(1);
        assert_eq!(attempted, 1);
        if let Some((_, hash)) = result {
            assert!(crate::goaway_hash_matches_difficulty(&hash, 65));
        }
    }

    #[test]
    fn scalar_nonce_bytes_match_emitted_big_endian_token_suffix() {
        use sha2::{Digest, Sha256};

        let challenge = [0_u8; 32];
        let message = GoAwayMessage::new_bytes(&challenge, 0);
        let nonce = 0x0102_0304_0506_0708_u64;
        let got = hash_nonce(&message, nonce);
        let mut wire = challenge.to_vec();
        wire.extend_from_slice(&nonce.to_be_bytes());
        let expected: [u8; 32] = Sha256::digest(&wire).into();
        let chunks = expected.as_chunks::<4>().0;
        let expected_words: [u32; 8] =
            core::array::from_fn(|index| u32::from_be_bytes(chunks[index]));
        assert_eq!(got, expected_words);
    }
}
