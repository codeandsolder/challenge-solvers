//! Friendly Captcha v1: independently implemented BLAKE2b-256 puzzle search.
//!
//! The signed puzzle is opaque: this module solves its public computation,
//! leaving signature, expiry, site binding, and replay checks to the issuer.

use std::{collections::HashSet, time::Instant};

use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use blake2::{Blake2b256, Digest};
use num_traits::ToPrimitive;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Implementation, Result};

#[derive(Deserialize)]
#[serde(untagged)]
enum PuzzleInput {
    Encoded(String),
    Object { puzzle: String },
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

fn diagnostics(seconds: u16) -> [u8; 3] {
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

pub(super) fn solve(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "Friendly Captcha BLAKE2b supports only auto and safe CPU implementations"
    );
    let request: PuzzleInput = serde_json::from_value(challenge)
        .context("Friendly Captcha request must be a puzzle string or {puzzle: string}")?;
    let encoded = match &request {
        PuzzleInput::Encoded(encoded) => encoded,
        PuzzleInput::Object { puzzle } => puzzle,
    };
    let puzzle = parse(encoded)?;
    let started = Instant::now();
    let mut bytes = Vec::with_capacity(usize::from(puzzle.solutions) * 8);
    let mut attempts = 0_u64;
    for i in 0..puzzle.solutions {
        let (nonce, count) = search(&puzzle, i)?;
        bytes.extend_from_slice(&nonce);
        attempts = attempts
            .checked_add(count)
            .context("Friendly Captcha attempt count overflow")?;
    }
    let elapsed = started.elapsed();
    let seconds = u16::try_from(elapsed.as_secs()).unwrap_or(u16::MAX);
    let diagnostics = diagnostics(seconds);
    let response = format!(
        "{}.{}.{}.{}",
        puzzle.signature,
        puzzle.encoded,
        STANDARD.encode(&bytes),
        STANDARD.encode(diagnostics)
    );
    Ok(Result {
        answer: json!({
            "response": response,
            "attempted_nonces": attempts,
            "solution_count": puzzle.solutions,
        }),
        solve_ms: elapsed.as_secs_f64() * 1000.0,
        implementation: "rustcrypto-blake2b".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_puzzle(difficulty: u8, count: u8, extra: usize) -> String {
        let mut bytes = vec![0_u8; 32 + extra];
        bytes[12] = 1;
        bytes[14] = count;
        bytes[15] = difficulty;
        bytes[24..32].copy_from_slice(b"nonce123");
        format!("fixture.{}", STANDARD.encode(bytes))
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
            let result = solve(json!({"puzzle":puzzle}), Implementation::Auto)?;
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
        assert!(solve(request, Implementation::ShaNi).is_err());
        Ok(())
    }
}
