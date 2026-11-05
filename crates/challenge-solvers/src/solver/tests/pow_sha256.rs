//! mCaptcha's SHA-256 based proof-of-work library (Apache-2.0 licensed, with modifications)
//!
//! <https://github.com/mCaptcha/pow_sha256/blob/master/LICENSE-APACHE>
//!
//! # Example:
//! ```rust
//!   use pow_sha256::{ConfigBuilder, PoW};
//!
//!   fn main() -> Result<(), Box<dyn std::error::Error>> {
//!       let config = ConfigBuilder::default()
//!         .salt("myrandomsaltisnotlongenoug".into())
//!         .build()?;
//!
//!       let phrase = "ironmansucks";
//!
//!       const DIFFICULTY: u32 = 1000;
//!
//!       let work = config.prove_work(&phrase, DIFFICULTY)?;
//!       assert!(config.is_valid_proof(&work, &phrase));
//!       assert!(Config::is_sufficient_difficulty(&work, DIFFICULTY));
//!       Ok(())
//!   }
//! ```

use std::marker::PhantomData;

//use num::Num;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, digest::Update};

/// Proof of work over concrete type `T`. `T` can be any type that implements `serde::Serialize`.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct PoW<T> {
    pub nonce: u64,
    pub result: String,
    _spook: PhantomData<T>,
}

impl<T> PoW<T> {
    pub fn new(nonce: u64, result: String) -> Self {
        Self {
            nonce,
            result,
            _spook: PhantomData,
        }
    }
}

/// Configuration for generating proof of work.
/// Please choose a long, unique value for salt
/// Resistance to dictionary/rainbow attacks depend on uniqueness
/// of the salt
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Debug)]
pub struct Config {
    pub salt: String,
}

impl Config {
    /// Create proof of work over an item of type `T`.
    ///
    /// Make sure difficulty is not too high. A 64 bit difficulty,
    /// for example, takes a long time on a general purpose processor.
    /// Returns `bincode::Error` if serialization fails.
    pub fn prove_work<T>(&self, t: &T, difficulty: u32) -> bincode::Result<PoW<T>>
    where
        T: Serialize,
    {
        bincode::serialize(t).map(|v| self.prove_work_serialized(&v, difficulty))
    }

    /// Create proof of work on an already serialized item of type `T`.
    /// The input is assumed to be serialized using network byte order.
    ///
    /// Make sure difficulty is not too high. A 64 bit difficulty,
    /// for example, takes a long time on a general purpose processor.
    pub fn prove_work_serialized<T>(&self, prefix: &[u8], difficulty: u32) -> PoW<T>
    where
        T: Serialize,
    {
        let prefix_sha = Sha256::new().chain(&self.salt).chain(prefix);
        let mut n = 0;
        let mut result = 0;
        let difficulty = get_difficulty(difficulty);
        while result < difficulty {
            n += 1;
            result = dev::score(prefix_sha.clone(), n);
        }
        PoW {
            nonce: n,
            result: result.to_string(),
            _spook: PhantomData,
        }
    }

    /// Calculate the `PoW` score with the provided input `T`.
    pub fn calculate<T>(&self, pow: &PoW<T>, t: &T) -> bincode::Result<u128>
    where
        T: Serialize,
    {
        bincode::serialize(t).map(|v| self.calculate_serialized(pow, &v))
    }

    /// Calculate the `PoW` score of an already serialized `T` and `self`.
    /// The input is assumed to be serialized using network byte order.
    pub fn calculate_serialized<T>(&self, pow: &PoW<T>, target: &[u8]) -> u128
    where
        T: Serialize,
    {
        dev::score(Sha256::new().chain(&self.salt).chain(target), pow.nonce)
    }

    /// Verifies that the `PoW` is indeed generated out of the phrase provided.
    pub fn is_valid_proof<T>(&self, pow: &PoW<T>, t: &T) -> bool
    where
        T: Serialize,
    {
        self.calculate(pow, t)
            .is_ok_and(|res| pow.result == res.to_string())
    }

    /// Checks if the `PoW` result is of sufficient difficulty
    pub fn is_sufficient_difficulty<T>(pow: &PoW<T>, target_diff: u32) -> bool
    where
        T: Serialize,
    {
        pow.result
            .parse::<u128>()
            .is_ok_and(|res| res >= get_difficulty(target_diff))
    }
}

pub mod dev {
    use super::*;

    pub fn score(prefix_sha: Sha256, nonce: u64) -> u128 {
        first_bytes_as_u128(
            prefix_sha
                .chain(nonce.to_string()) //used to be: to_be_bytes() converts to network endian
                // chain() expexets something that can be converted to &[u8], String is fine
                .finalize()
                .as_slice(),
        )
    }

    fn first_bytes_as_u128(inp: &[u8]) -> u128 {
        u128::from_be_bytes([
            inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8], inp[9],
            inp[10], inp[11], inp[12], inp[13], inp[14], inp[15],
        ])
    }
}

// utility function to get u128 difficulty factor from u32
// javacript isn't capable of represnting u128 so
fn get_difficulty(difficulty_factor: u32) -> u128 {
    u128::MAX - u128::MAX / u128::from(difficulty_factor)
}

#[cfg(test)]
mod test {
    use super::*;

    const DIFFICULTY: u32 = 1000;

    fn get_config() -> Config {
        Config {
            salt: "79ziepia7vhjgviiwjhnend3ofjqocsi2winc4ptqhmkvcajihywxcizewvckg9h6gs4j83v9"
                .into(),
        }
    }

    #[test]
    fn base_functionality() -> anyhow::Result<()> {
        // Let's prove we did work targeting a phrase.
        let phrase = b"Ex nihilo nihil fit.".to_vec();
        let config = get_config();
        let pw = config.prove_work(&phrase, DIFFICULTY)?;
        assert!(config.calculate(&pw, &phrase)? >= get_difficulty(DIFFICULTY));
        assert!(config.is_valid_proof(&pw, &phrase));
        assert!(Config::is_sufficient_difficulty(&pw, DIFFICULTY));
        Ok(())
    }

    #[test]
    fn double_pow() -> anyhow::Result<()> {
        let phrase = "Ex nihilo nihil fit.".to_owned();
        let config = get_config();

        let pw = config.prove_work(&phrase, DIFFICULTY)?;
        let pwpw = config.prove_work(&pw, DIFFICULTY)?;

        assert!(config.calculate(&pw, &phrase)? >= get_difficulty(DIFFICULTY));
        assert!(config.is_valid_proof(&pw, &phrase));
        assert!(Config::is_sufficient_difficulty(&pw, DIFFICULTY));

        assert!(config.calculate(&pwpw, &pw)? >= get_difficulty(DIFFICULTY));
        assert!(config.is_valid_proof(&pwpw, &pw));
        assert!(Config::is_sufficient_difficulty(&pwpw, DIFFICULTY));
        Ok(())
    }

    #[test]
    fn is_not_valid_proof() -> anyhow::Result<()> {
        let phrase = "Ex nihilo nihil fit.".to_owned();
        let phrase2 = "Omne quod movetur ab alio movetur.".to_owned();

        let config = get_config();
        let pw = config.prove_work(&phrase, DIFFICULTY)?;

        let pw2 = config.prove_work(&phrase2, DIFFICULTY)?;

        assert!(!config.is_valid_proof(&pw, &phrase2));
        assert!(!config.is_valid_proof(&pw2, &phrase));
        Ok(())
    }

    fn check_time(prev: u128, current: u128) -> bool {
        prev < current
    }

    #[test]
    fn computation_time_test() -> anyhow::Result<()> {
        use std::time::Instant;
        const DIFFICULTY: u32 = 50000;

        let target = "testing";
        let config = get_config();
        let mut current = Instant::now();
        config.prove_work(&target, DIFFICULTY)?;
        let prev = current.elapsed().as_nanos();

        current = Instant::now();
        config.prove_work(&target, DIFFICULTY * 10)?;
        let tmp = current.elapsed().as_nanos();
        assert!(check_time(prev, tmp));
        Ok(())
    }

    #[test]
    fn serialization_test() -> anyhow::Result<()> {
        let target: u8 = 1;
        let config = get_config();
        let pw = config.prove_work(&target, DIFFICULTY)?;

        let message: (u8, PoW<u8>) = (target, pw);
        let message_ser = bincode::serialize(&message)?;
        let recieved_message: (u8, PoW<u8>) = bincode::deserialize(&message_ser)?;
        assert_eq!(recieved_message, message);
        assert!(Config::is_sufficient_difficulty(&message.1, DIFFICULTY));
        assert!(config.is_valid_proof(&message.1, &target));
        Ok(())
    }
}
