use std::time::Instant;

use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::json;
use sha1::{Digest, Sha1};

use crate::classic_pow::{Implementation, Result};

const DEFAULT_LIMIT: u64 = 4_000_000_000;
const TARGET: [u8; 2] = [0xb0, 0x0b];

#[derive(Debug, Deserialize)]
pub struct Challenge {
    seed: String,
}

pub fn solve(
    challenge: serde_json::Value,
    implementation: Implementation,
) -> anyhow::Result<Result> {
    let challenge: Challenge =
        serde_json::from_value(challenge).context("invalid Linear.pub challenge")?;
    ensure!(
        challenge.seed.len() == 40 && challenge.seed.as_bytes().iter().all(u8::is_ascii_hexdigit),
        "Linear.pub seed must be exactly 40 ASCII hex characters"
    );
    ensure!(
        matches!(implementation, Implementation::Auto),
        "Linear.pub uses RustCrypto SHA-1 and exposes only the auto implementation"
    );
    let offset = challenge
        .seed
        .as_bytes()
        .first()
        .and_then(|byte| (*byte as char).to_digit(16))
        .and_then(|value| usize::try_from(value).ok())
        .context("Linear.pub seed must begin with a hex digit")?;
    ensure!(offset + 1 < 20, "Linear.pub digest offset is out of range");

    let started = Instant::now();
    let (nonce, attempted_nonces) =
        solve_rustcrypto(challenge.seed.as_bytes(), offset, DEFAULT_LIMIT)
            .context("Linear.pub search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cookie_value = format!("{}{nonce}", challenge.seed);

    Ok(Result {
        answer: json!({
            "nonce": nonce,
            "cookie_name": "yX3",
            "cookie_value": cookie_value,
            "cookie": format!("yX3={cookie_value}"),
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: "auto".to_owned(),
    })
}

fn solve_rustcrypto(seed: &[u8], offset: usize, limit: u64) -> Option<(u64, u64)> {
    for nonce in 0..limit {
        let mut hasher = Sha1::new();
        hasher.update(seed);
        let decimal = decimal_bytes(nonce);
        hasher.update(decimal.as_slice());
        let digest = hasher.finalize();
        if digest[offset] == TARGET[0] && digest[offset + 1] == TARGET[1] {
            return Some((nonce, nonce + 1));
        }
    }
    None
}

#[derive(Clone, Copy)]
struct DecimalBytes {
    bytes: [u8; 20],
    start: usize,
}

impl DecimalBytes {
    fn as_slice(&self) -> &[u8] {
        &self.bytes[self.start..]
    }
}

const fn decimal_bytes(mut value: u64) -> DecimalBytes {
    let mut bytes = [0_u8; 20];
    let mut start = bytes.len();
    loop {
        start -= 1;
        bytes[start] = b'0' + (value % 10).to_le_bytes()[0];
        value /= 10;
        if value == 0 {
            break;
        }
    }
    DecimalBytes { bytes, start }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE_SEED: &str = "72BBE3B063367AB2F323D46D2ADA8A8A29BB5B58";

    #[test]
    fn solves_every_live_digest_offset() {
        for (offset, expected_nonce) in [
            160_456_u64,
            26_862,
            24_557,
            3_353,
            126_421,
            78_394,
            46_861,
            17_884,
            120_106,
            13_018,
            28_957,
            171_452,
            35_718,
            62_236,
            12_775,
            22_189,
        ]
        .into_iter()
        .enumerate()
        {
            let seed = format!("{offset:X}{}", "0".repeat(39));
            let result = solve_rustcrypto(seed.as_bytes(), offset, 500_000);
            assert_eq!(result.map(|found| found.0), Some(expected_nonce));
        }
    }

    #[test]
    fn solves_captured_live_challenge() {
        let offset = 7;
        let result = solve_rustcrypto(LIVE_SEED.as_bytes(), offset, 100_000);
        assert_eq!(result.map(|found| found.0), Some(29_224));
    }

    #[test]
    fn protocol_answer_matches_browser_cookie_shape() -> anyhow::Result<()> {
        let answer = solve(json!({"seed": LIVE_SEED}), Implementation::Auto)?;
        assert_eq!(answer.answer["nonce"], 29_224);
        assert_eq!(answer.answer["cookie_name"], "yX3");
        assert_eq!(answer.answer["cookie_value"], format!("{LIVE_SEED}29224"));
        Ok(())
    }

    #[test]
    fn rejects_non_hex_or_wrong_length_seed() {
        assert!(solve(json!({"seed": "abc"}), Implementation::Auto).is_err());
        assert!(
            solve(
                json!({"seed": "Z2BBE3B063367AB2F323D46D2ADA8A8A29BB5B58"}),
                Implementation::Auto,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_non_auto_implementation() {
        assert!(solve(json!({"seed": LIVE_SEED}), Implementation::Safe).is_err());
    }
}
