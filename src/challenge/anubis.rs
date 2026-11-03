//! Anubis specific protocol structures.
use sha2::Digest;

use crate::solve_decimal_sha256_zero_nibbles;
use alloc::string::String;

#[derive(serde::Deserialize, Debug)]
/// Anubis `PoW` challenge descriptor.
pub struct ChallengeDescriptor {
    challenge: ChallengeForm,
    rules: AnubisRules,
}

#[derive(serde::Deserialize, Debug)]
#[serde(untagged)]
/// Anubis `PoW` challenge form.
pub enum ChallengeForm {
    /// Plain challenge.
    Plain(String),
    /// Id wrapped challenge.
    #[serde(rename_all = "camelCase")]
    IdWrapped {
        /// The id.
        id: String,
        /// The random data.
        random_data: String,
    },
}

impl ChallengeForm {
    /// Get the id of an Anubis `PoW` challenge.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Plain(_) => None,
            Self::IdWrapped { id, .. } => Some(id),
        }
    }
}

/// As reference to a string.
impl AsRef<str> for ChallengeForm {
    fn as_ref(&self) -> &str {
        match self {
            Self::Plain(s) => s,
            Self::IdWrapped { random_data, .. } => random_data,
        }
    }
}

impl ChallengeDescriptor {
    /// Estimate the workload of an Anubis `PoW`.
    #[must_use]
    pub fn estimated_workload(&self) -> u64 {
        if self.rules.algorithm == "preact" {
            return 1u64;
        }
        16u64.saturating_pow(self.rules.difficulty.into())
    }

    /// Get the rules of an Anubis `PoW`.
    #[must_use]
    pub const fn rules(&self) -> &AnubisRules {
        &self.rules
    }

    /// Get the hash result key of an Anubis `PoW`.
    #[must_use]
    pub fn hash_result_key(&self) -> &str {
        if self.rules.algorithm == "preact" {
            "result"
        } else {
            "response"
        }
    }

    /// Get the challenge of an Anubis `PoW`.
    #[must_use]
    pub const fn challenge(&self) -> &ChallengeForm {
        &self.challenge
    }

    /// If the Anubis `PoW` is supported.
    #[must_use]
    pub fn supported(&self) -> bool {
        self.rules.algorithm == "fast"
            || self.rules.algorithm == "slow"
            || self.rules.algorithm == "preact"
    }

    /// Solve an Anubis `PoW`.
    #[must_use]
    pub fn solve(&self) -> (Option<(u64, [u32; 8])>, u64) {
        self.solve_with_limit(u64::MAX)
    }

    /// Delay to hold the solution before it will be accepted.
    #[must_use]
    pub fn delay(&self) -> u64 {
        if self.rules.algorithm == "preact" {
            u64::from(self.rules.difficulty) * 100
        } else {
            0
        }
    }

    /// Solve an Anubis `PoW` with a limit.
    #[must_use]
    pub fn solve_with_limit(&self, limit: u64) -> (Option<(u64, [u32; 8])>, u64) {
        if self.rules.algorithm == "preact" {
            let hash: [u8; 32] = sha2::Sha256::digest(self.challenge.as_ref().as_bytes()).into();
            let chunks = hash.as_chunks::<4>().0;
            let hash_arr = core::array::from_fn(|i| u32::from_be_bytes(chunks[i]));
            return (Some((0, hash_arr)), 0);
        }
        solve_decimal_sha256_zero_nibbles(
            self.challenge.as_ref().as_bytes(),
            self.rules.difficulty,
            limit,
        )
        .map_or((None, 0), |search| {
            (
                search
                    .solution
                    .map(|solution| (solution.nonce, solution.digest)),
                search.attempted,
            )
        })
    }
}

#[derive(serde::Deserialize, Debug)]
/// Anubis `PoW` challenge rules.
pub struct AnubisRules {
    /// The algorithm. (JSON key: `algorithm`)
    algorithm: String,
    difficulty: u8,
}

impl AnubisRules {
    /// If the instant is instantly or almost instantly solved and not worth task spawning.
    #[must_use]
    pub fn instant(&self) -> bool {
        self.algorithm == "preact" || self.difficulty < 4
    }

    /// Get the algorithm of an Anubis `PoW`.
    #[must_use]
    pub fn algorithm(&self) -> &str {
        &self.algorithm
    }

    /// Difficulty in leading zero nibbles.
    #[must_use]
    pub const fn difficulty(&self) -> u8 {
        self.difficulty
    }
}
