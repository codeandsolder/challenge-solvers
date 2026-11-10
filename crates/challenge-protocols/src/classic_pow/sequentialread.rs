//! sequentialread `PoW` Bot Deterrent's base64(JSON) Scrypt challenge protocol.
//!
//! The service issues challenges as base64 JSON; its `/Verify` endpoint only
//! needs the original challenge and a minimal big-endian hexadecimal nonce.
//! Stateful challenge registration, expiry, and one-shot use are server-side.

use std::{
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Context, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Implementation, Result};

#[derive(Debug, Deserialize)]
struct Challenge {
    #[serde(rename = "N")]
    cpu_memory_cost: u64,
    r: u32,
    p: u32,
    klen: usize,
    i: String,
    d: String,
    dl: i64,
}

struct Parsed {
    original: String,
    params: scrypt::Params,
    preimage: [u8; 8],
    difficulty: String,
    key_len: usize,
}

fn parse(challenge: &Value) -> anyhow::Result<Parsed> {
    let encoded = match challenge {
        Value::String(text) => text.as_str(),
        Value::Object(fields) => fields
            .get("challenge")
            .and_then(Value::as_str)
            .context("PoW Bot Deterrent expects a base64 challenge string")?,
        _ => bail!("PoW Bot Deterrent expects a base64 challenge string or challenge object"),
    };
    let payload = STANDARD
        .decode(encoded)
        .context("PoW Bot Deterrent challenge is not valid base64")?;
    let challenge: Challenge =
        serde_json::from_slice(&payload).context("invalid PoW Bot Deterrent JSON challenge")?;

    // The official verifier compares *the last len(d) hexadecimal digits*,
    // not a numeric interpretation of the full Scrypt output.
    ensure!(
        challenge.d.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "PoW Bot Deterrent difficulty must contain hexadecimal digits"
    );
    ensure!(
        challenge.d.len() <= challenge.klen.saturating_mul(2),
        "PoW Bot Deterrent difficulty exceeds derived key length"
    );
    ensure!(
        challenge.klen > 0,
        "PoW Bot Deterrent key length must be nonzero"
    );
    let log_n = challenge
        .cpu_memory_cost
        .checked_ilog2()
        .filter(|_| challenge.cpu_memory_cost.is_power_of_two())
        .context("PoW Bot Deterrent N must be a power of two")?;
    let params = scrypt::Params::new(
        u8::try_from(log_n).context("PoW Bot Deterrent N is too large")?,
        challenge.r,
        challenge.p,
    )
    .context("invalid PoW Bot Deterrent Scrypt parameters")?;
    let preimage: [u8; 8] = STANDARD
        .decode(challenge.i)
        .context("PoW Bot Deterrent preimage is not valid base64")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("PoW Bot Deterrent preimage must be exactly 8 bytes"))?;
    // Validate metadata, but don't use dl in the proof threshold: the original
    // server checks only d and the full Scrypt parameters.
    // Difficulty level is metadata; the server enforces the encoded `d` threshold.
    let _ = challenge.dl;
    Ok(Parsed {
        original: encoded.to_owned(),
        params,
        preimage,
        difficulty: challenge.d,
        key_len: challenge.klen,
    })
}

fn nonce_hex(nonce: u64) -> String {
    let full = nonce.to_be_bytes();
    let first = full.iter().position(|byte| *byte != 0).unwrap_or(7);
    hex::encode(&full[first..])
}

fn derive(parsed: &Parsed, nonce: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut key = vec![0_u8; parsed.key_len];
    scrypt::scrypt(nonce, &parsed.preimage, &parsed.params, &mut key)
        .context("PoW Bot Deterrent Scrypt derivation failed")?;
    Ok(key)
}

fn matches_difficulty(parsed: &Parsed, derived: &[u8]) -> bool {
    let digest = hex::encode(derived);
    digest
        .get(digest.len().saturating_sub(parsed.difficulty.len())..)
        .is_some_and(|suffix| suffix <= parsed.difficulty.as_str())
}

/// Independently check the computational part of a response from this protocol.
///
/// The origin server must still check that the issued challenge is registered,
/// not expired, and has not already been consumed.
///
/// # Errors
/// Returns an error for malformed challenges or invalid nonce hexadecimal.
pub fn verify(challenge: &Value, nonce: &str) -> anyhow::Result<bool> {
    let parsed = parse(challenge)?;
    // Official Go verifier accepts up to eight raw nonce bytes, including
    // leading zeros; canonical outputs from this solver use minimal bytes.
    let bytes = hex::decode(nonce).context("invalid PoW Bot Deterrent nonce hex")?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= 8,
        "nonce must be 1 to 8 bytes"
    );
    let key = derive(&parsed, &bytes)?;
    Ok(matches_difficulty(&parsed, &key))
}

#[derive(Clone)]
struct Winner {
    nonce: String,
    derived_key: String,
}

fn solve_worker(
    parsed: &Parsed,
    worker_index: usize,
    workers: usize,
    stopped: &AtomicBool,
    winner: &OnceLock<Winner>,
) -> anyhow::Result<u64> {
    // Official JS starts at worker_id * 2^dl * 1000 + 1. Starting at one
    // for all workers and striding produces equally valid server proofs,
    // without assuming a fixed or capped client-side difficulty.
    let mut nonce = u64::try_from(worker_index)?
        .checked_add(1)
        .context("PoW Bot Deterrent nonce domain exhausted")?;
    let stride = u64::try_from(workers)?;
    let mut attempts = 0_u64;
    while !stopped.load(Ordering::Relaxed) {
        attempts = attempts
            .checked_add(1)
            .context("nonce attempt count overflow")?;
        let bytes = nonce.to_be_bytes();
        let first = bytes.iter().position(|byte| *byte != 0).unwrap_or(7);
        let password = &bytes[first..];
        let derived = match derive(parsed, password) {
            Ok(value) => value,
            Err(err) => {
                stopped.store(true, Ordering::Release);
                return Err(err);
            }
        };
        if matches_difficulty(parsed, &derived) {
            let value = Winner {
                nonce: nonce_hex(nonce),
                derived_key: hex::encode(derived),
            };
            if winner.set(value).is_ok() {
                stopped.store(true, Ordering::Release);
            }
            return Ok(attempts);
        }
        let Some(next) = nonce.checked_add(stride) else {
            break;
        };
        nonce = next;
    }
    Ok(attempts)
}

pub(super) fn solve(
    challenge: &Value,
    implementation: Implementation,
    worker_count: usize,
) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "PoW Bot Deterrent supports only auto/safe CPU implementations"
    );
    ensure!(
        worker_count > 0,
        "PoW Bot Deterrent needs at least one CPU worker"
    );
    let parsed = parse(challenge)?;
    let started = Instant::now();
    let stopped = AtomicBool::new(false);
    let winner = OnceLock::new();
    let attempts = thread::scope(|scope| -> anyhow::Result<u64> {
        let handles: Vec<_> = (0..worker_count)
            .map(|worker_index| {
                let parsed = &parsed;
                let stopped = &stopped;
                let winner = &winner;
                scope.spawn(move || {
                    solve_worker(parsed, worker_index, worker_count, stopped, winner)
                })
            })
            .collect();
        let mut attempts = 0_u64;
        let mut first_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(value)) => {
                    attempts = attempts
                        .checked_add(value)
                        .context("total attempts overflow")?;
                }
                Ok(Err(err)) if first_error.is_none() => first_error = Some(err),
                Err(_) if first_error.is_none() => {
                    stopped.store(true, Ordering::Release);
                    first_error = Some(anyhow::anyhow!("PoW Bot Deterrent worker panicked"));
                }
                Ok(Err(_)) | Err(_) => {}
            }
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(attempts)
    })?;
    let proof = winner
        .get()
        .context("PoW Bot Deterrent nonce domain exhausted")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok(Result {
        answer: json!({
            "challenge": parsed.original,
            "nonce": proof.nonce,
            "derived_key": proof.derived_key,
            "attempted_nonces": attempts,
        }),
        solve_ms,
        implementation: "scrypt".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn challenge_with(difficulty: &str) -> anyhow::Result<Value> {
        let source = json!({
            "N": 16,
            "r": 1,
            "p": 1,
            "klen": 16,
            "i": STANDARD.encode([1_u8,2,3,4,5,6,7,8]),
            "d": difficulty,
            "dl": 4
        });
        Ok(Value::String(STANDARD.encode(serde_json::to_vec(&source)?)))
    }

    #[test]
    fn trivial_challenge_is_valid_and_uses_canonical_nonce() -> anyhow::Result<()> {
        let challenge = challenge_with("ffff")?;
        let result = solve(&challenge, Implementation::Safe, 1)?;
        assert_eq!(result.answer["nonce"], "01");
        assert_eq!(result.answer["attempted_nonces"], 1);
        assert_eq!(
            result.answer["derived_key"],
            "20f8890d88126dbe692b670ec465d0fc"
        );
        assert!(verify(&challenge, "01")?);
        assert!(verify(&challenge, "0001")?); // Go accepts padded raw bytes
        Ok(())
    }

    #[test]
    fn harder_challenge_accepts_only_real_scrypt_proof() -> anyhow::Result<()> {
        let challenge = challenge_with("0fff")?;
        let result = solve(&challenge, Implementation::Auto, 1)?;
        assert_eq!(result.answer["nonce"], "10");
        assert_eq!(
            result.answer["derived_key"],
            "732d9a0d3e86cd94a3f9bfda94f00620"
        );
        assert_eq!(result.answer["attempted_nonces"], 16);
        assert!(verify(&challenge, "10")?);
        assert!(!verify(&challenge, "01")?);
        assert_eq!(result.answer["challenge"], challenge);

        // Parallel search may return a different winning nonce, which is
        // acceptable to the server so long as it verifies.
        let concurrent = solve(&challenge, Implementation::Auto, 2)?;
        assert!(verify(
            &challenge,
            concurrent.answer["nonce"]
                .as_str()
                .context("parallel nonce")?
        )?);
        Ok(())
    }

    #[test]
    fn native_go_issued_challenge_and_multibyte_nonce() -> anyhow::Result<()> {
        // Exact challenge JSON serialized by the official upstream Go model.
        // The first accepted proof was obtained independently with Go's
        // golang.org/x/crypto/scrypt and the original server's suffix check.
        let challenge = Value::String(
            "eyJOIjoxNiwiciI6MSwicCI6MSwia2xlbiI6MTYsImkiOiJBUUlEQkFVR0J3Zz0iLCJkIjoiMDBmZiIsImRsIjo0fQ=="
                .to_owned(),
        );
        let result = solve(&challenge, Implementation::Safe, 1)?;
        assert_eq!(result.answer["nonce"], "01fc");
        assert_eq!(
            result.answer["derived_key"],
            "2cdf77996def2e46fa12da173d9b0042"
        );
        assert_eq!(result.answer["attempted_nonces"], 508);
        assert!(verify(&challenge, "01fc")?);
        assert!(!verify(&challenge, "01")?);
        Ok(())
    }

    #[test]
    fn bad_parameter_shapes_are_rejected() -> anyhow::Result<()> {
        let source = json!({"N": 15, "r": 1, "p": 1, "klen": 16,
            "i": STANDARD.encode([0_u8;8]), "d": "ff", "dl": 8});
        let challenge = Value::String(STANDARD.encode(serde_json::to_vec(&source)?));
        assert!(solve(&challenge, Implementation::Auto, 1).is_err());
        assert!(verify(&challenge_with("ffff")?, "invalid").is_err());
        assert!(solve(&challenge_with("ffff")?, Implementation::ShaNi, 1).is_err());
        Ok(())
    }
}
