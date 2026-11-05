use std::{num::NonZeroU32, time::Instant};

use anyhow::{Context, bail, ensure};
use argon2::{
    Algorithm as Argon2Algorithm, Argon2, Params as Argon2Params, Version as Argon2Version,
};
use challenge_solvers::{AltchaSha256Solver, message::AltchaMessage, solver::SOLVE_TYPE_MASK};
use pbkdf2::pbkdf2_hmac;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256, Sha384, Sha512};

use super::{DynPow, Implementation, JS_MAX_SAFE_INTEGER, Result, require_x86_feature};

fn altcha_solver(
    implementation: Implementation,
    message: AltchaMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "ALTCHA has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(AltchaSha256Solver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::AltchaSha256Solver::from(message),
        )),
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2AltchaSha256Solver::from(message),
        )),
        #[cfg(all(feature = "portable-simd", target_arch = "x86_64"))]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512AltchaSha256Solver::from(message),
        )),
        _ => bail!("implementation unavailable for ALTCHA"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AltchaCounterMode {
    Uint32,
    String,
}

#[derive(Debug, Deserialize)]
struct AltchaChallenge {
    parameters: AltchaChallengeParameters,
}

#[derive(Debug, Deserialize)]
struct AltchaChallengeParameters {
    algorithm: String,
    cost: u32,
    #[serde(rename = "keyLength")]
    key_length: usize,
    #[serde(rename = "keyPrefix")]
    key_prefix: String,
    #[serde(rename = "memoryCost", default)]
    memory_cost: Option<u32>,
    nonce: String,
    #[serde(default)]
    parallelism: Option<u32>,
    salt: String,
}

fn altcha_counter_mode(challenge: &Value) -> anyhow::Result<AltchaCounterMode> {
    match challenge.get("counterMode").and_then(Value::as_str) {
        None | Some("uint32") => Ok(AltchaCounterMode::Uint32),
        Some("string") => Ok(AltchaCounterMode::String),
        Some(other) => bail!("unsupported ALTCHA counterMode: {other}"),
    }
}

pub(super) fn solve(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    if challenge.get("parameters").is_none() {
        return solve_altcha_v1(challenge, implementation);
    }
    let counter_mode = altcha_counter_mode(&challenge)?;
    let official: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;

    if counter_mode == AltchaCounterMode::String {
        ensure!(
            matches!(implementation, Implementation::Auto | Implementation::Safe),
            "ALTCHA string-counter mode supports auto/safe implementations only"
        );
        return solve_altcha_generic(&official, counter_mode);
    }
    if let Some(result) = try_solve_altcha_fast(&official, implementation)? {
        return Ok(result);
    }
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "this ALTCHA algorithm supports auto/safe implementations only"
    );
    solve_altcha_generic(&official, counter_mode)
}

fn try_solve_altcha_fast(
    challenge: &AltchaChallenge,
    implementation: Implementation,
) -> anyhow::Result<Option<Result>> {
    let Some((message, target, mask, key_length)) = altcha_fast_case(challenge) else {
        return Ok(None);
    };
    let mut solver = altcha_solver(implementation, message)?;
    solver.set_limit(u64::from(u32::MAX) + 1);
    let started = Instant::now();
    let (counter, key) = solver
        .solve(target, SOLVE_TYPE_MASK, mask)
        .context("ALTCHA optimized search exhausted the u32 counter domain")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let derived_key = encode_words_be(key);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Some(Result {
        answer: json!({
            "counter": counter,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted(),
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    }))
}

fn altcha_fast_case(challenge: &AltchaChallenge) -> Option<(AltchaMessage, u64, u64, usize)> {
    let parameters = &challenge.parameters;
    if !matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256")
        || parameters.cost == 0
        || parameters.key_length == 0
        || parameters.key_length > 32
        || parameters.key_prefix.is_empty()
        || !parameters.key_prefix.len().is_multiple_of(2)
        || parameters.key_prefix.len() > 8
    {
        return None;
    }
    let nonce: [u8; 16] = hex::decode(&parameters.nonce)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())?;
    let salt: [u8; 16] = hex::decode(&parameters.salt)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())?;
    let cost = NonZeroU32::new(parameters.cost)?;
    let key_length_u32 = u32::try_from(parameters.key_length).ok()?;
    let key_length = NonZeroU32::new(key_length_u32)?;
    let (target, mask) = prefix_target(&parameters.key_prefix).ok()?;
    let message = AltchaMessage {
        nonce,
        salt,
        cost,
        pbkdf2: parameters.algorithm == "PBKDF2/SHA-256",
        key_length,
    };
    Some((message, target, mask, parameters.key_length))
}
struct AltchaHexPrefix(Vec<u8>);

impl AltchaHexPrefix {
    fn parse(prefix: &str) -> anyhow::Result<Self> {
        let nibbles = prefix
            .bytes()
            .map(|byte| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                _ => bail!("ALTCHA keyPrefix must be hexadecimal"),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self(nibbles))
    }

    fn matches(&self, derived: &[u8]) -> bool {
        self.0.len() <= derived.len() * 2
            && self.0.iter().enumerate().all(|(index, expected)| {
                let byte = derived[index / 2];
                let actual = if index.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                actual == *expected
            })
    }
}

fn altcha_iterative_sha<D: Digest>(initial: &[u8], iterations: u32, key_length: usize) -> Vec<u8> {
    let mut data = initial.to_vec();
    for _ in 0..iterations.max(1) {
        data = D::digest(&data).to_vec();
    }
    data.truncate(key_length.min(data.len()));
    data
}

fn altcha_derive_key(
    parameters: &AltchaChallengeParameters,
    salt: &[u8],
    password: &[u8],
) -> anyhow::Result<Vec<u8>> {
    let key_length = parameters.key_length;
    let algorithm = parameters.algorithm.to_ascii_uppercase();
    match algorithm.as_str() {
        "SHA-256" | "SHA-384" | "SHA-512" => {
            let initial: Vec<_> = salt.iter().chain(password).copied().collect();
            match algorithm.as_str() {
                "SHA-384" => Ok(altcha_iterative_sha::<Sha384>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
                "SHA-512" => Ok(altcha_iterative_sha::<Sha512>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
                _ => Ok(altcha_iterative_sha::<Sha256>(
                    &initial,
                    parameters.cost,
                    key_length,
                )),
            }
        }
        "PBKDF2/SHA-256" | "PBKDF2/SHA-384" | "PBKDF2/SHA-512" => {
            let mut out = vec![0_u8; key_length];
            match algorithm.as_str() {
                "PBKDF2/SHA-384" => {
                    pbkdf2_hmac::<Sha384>(password, salt, parameters.cost, &mut out);
                }
                "PBKDF2/SHA-512" => {
                    pbkdf2_hmac::<Sha512>(password, salt, parameters.cost, &mut out);
                }
                _ => pbkdf2_hmac::<Sha256>(password, salt, parameters.cost, &mut out),
            }
            Ok(out)
        }
        "SCRYPT" => {
            let n = parameters.cost;
            let log_n = n
                .checked_ilog2()
                .filter(|_| n.is_power_of_two())
                .context("ALTCHA scrypt cost must be a power of two")?;
            let params = scrypt::Params::new(
                u8::try_from(log_n).context("ALTCHA scrypt log2(N) exceeds u8")?,
                parameters.memory_cost.unwrap_or(8),
                parameters.parallelism.unwrap_or(1),
            )
            .context("invalid ALTCHA scrypt parameters")?;
            let mut out = vec![0_u8; key_length];
            scrypt::scrypt(password, salt, &params, &mut out)
                .context("ALTCHA scrypt derivation failed")?;
            Ok(out)
        }
        "ARGON2ID" => {
            let params = Argon2Params::new(
                parameters
                    .memory_cost
                    .context("ALTCHA Argon2id requires memoryCost")?,
                parameters.cost,
                parameters.parallelism.unwrap_or(1),
                Some(key_length),
            )
            .context("invalid ALTCHA Argon2id parameters")?;
            let argon2 = Argon2::new(Argon2Algorithm::Argon2id, Argon2Version::V0x13, params);
            let mut out = vec![0_u8; key_length];
            argon2
                .hash_password_into(password, salt, &mut out)
                .context("ALTCHA Argon2id derivation failed")?;
            Ok(out)
        }
        other => bail!("unsupported ALTCHA algorithm: {other}"),
    }
}

fn altcha_generic_result(
    counter: u64,
    derived: Vec<u8>,
    started: Instant,
    implementation: &str,
) -> Result {
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Result {
        answer: json!({
            "counter": counter,
            "derivedKey": hex::encode(derived),
            "time": solve_ms,
            "attempted_nonces": counter.saturating_add(1),
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    }
}

fn solve_altcha_generic(
    challenge: &AltchaChallenge,
    counter_mode: AltchaCounterMode,
) -> anyhow::Result<Result> {
    let parameters = &challenge.parameters;
    let nonce = hex::decode(&parameters.nonce).context("invalid ALTCHA nonce hex")?;
    let salt = hex::decode(&parameters.salt).context("invalid ALTCHA salt hex")?;
    let prefix = AltchaHexPrefix::parse(&parameters.key_prefix)?;
    let started = Instant::now();
    let mut password = Vec::with_capacity(nonce.len() + 16);

    match counter_mode {
        AltchaCounterMode::Uint32 => {
            for counter in 0..=u32::MAX {
                password.clear();
                password.extend_from_slice(&nonce);
                password.extend_from_slice(&counter.to_be_bytes());
                let derived = altcha_derive_key(parameters, &salt, &password)?;
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        u64::from(counter),
                        derived,
                        started,
                        "altcha-generic",
                    ));
                }
            }
            bail!("ALTCHA uint32-counter search exhausted the u32 domain")
        }
        AltchaCounterMode::String => {
            let mut counter_buffer = itoa::Buffer::new();
            for counter in 0..=JS_MAX_SAFE_INTEGER {
                password.clear();
                password.extend_from_slice(&nonce);
                password.extend_from_slice(counter_buffer.format(counter).as_bytes());
                let derived = altcha_derive_key(parameters, &salt, &password)?;
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        counter,
                        derived,
                        started,
                        "altcha-string",
                    ));
                }
            }
            bail!("ALTCHA string-counter search exhausted Number.MAX_SAFE_INTEGER")
        }
    }
}

#[derive(Debug, Deserialize)]
struct AltchaV1Challenge {
    algorithm: String,
    challenge: String,
    maxnumber: Option<u32>,
    salt: String,
    signature: String,
}

fn solve_altcha_v1(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "ALTCHA v1 supports auto/safe implementations only"
    );
    let challenge: AltchaV1Challenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v1 challenge")?;
    ensure!(
        matches!(
            challenge.algorithm.as_str(),
            "SHA-1" | "SHA-256" | "SHA-512"
        ),
        "unsupported ALTCHA v1 algorithm: {}",
        challenge.algorithm
    );
    ensure!(
        !challenge.signature.is_empty(),
        "ALTCHA v1 signature must not be empty"
    );
    let max = challenge.maxnumber.unwrap_or(1_000_000);
    let started = Instant::now();
    for number in 0..=max {
        let input = format!("{}{number}", challenge.salt);
        let candidate = match challenge.algorithm.as_str() {
            "SHA-1" => {
                use sha1::Digest as _;
                hex::encode(sha1::Sha1::digest(input.as_bytes()))
            }
            "SHA-256" => hex::encode(Sha256::digest(input.as_bytes())),
            "SHA-512" => hex::encode(sha2::Sha512::digest(input.as_bytes())),
            _ => unreachable!(),
        };
        if candidate == challenge.challenge {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({ "number": number, "took": solve_ms }),
                solve_ms,
                implementation: implementation.name().to_owned(),
            });
        }
    }
    bail!("ALTCHA v1 search exhausted maxnumber")
}

fn prefix_target(prefix: &str) -> anyhow::Result<(u64, u64)> {
    ensure!(
        !prefix.is_empty() && prefix.len().is_multiple_of(2),
        "ALTCHA keyPrefix must be non-empty whole-byte hex"
    );
    let bytes = hex::decode(prefix).context("invalid ALTCHA keyPrefix hex")?;
    ensure!(
        bytes.len() <= 4,
        "ALTCHA keyPrefix longer than 4 bytes needs the 64-bit comparison build"
    );
    let mut first = [0_u8; 8];
    first[..bytes.len()].copy_from_slice(&bytes);
    let target = u64::from_be_bytes(first);
    let bits = bytes.len() * 8;
    let mask = if bits == 0 {
        0
    } else {
        u64::MAX << (64 - bits)
    };
    Ok((target, mask))
}

fn encode_words_be(words: [u32; 8]) -> String {
    hex::encode(
        words
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect::<Vec<_>>(),
    )
}

#[cfg(feature = "gpu")]
pub(super) async fn solve_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    ensure!(
        altcha_counter_mode(&challenge)? == AltchaCounterMode::Uint32,
        "ALTCHA WGPU supports uint32 counter mode only"
    );
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let (message, target, mask, key_length) = altcha_fast_case(&challenge)
        .context("ALTCHA challenge shape is unsupported by the WGPU fast path")?;
    let mut solver = challenge_solvers::gpu::AltchaSha256Solver::from(message);
    solver.set_limit(u64::from(u32::MAX) + 1);
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let derived_key = encode_words_be(solution.hash);
    let derived_key = &derived_key[..key_length * 2];
    Ok(Result {
        answer: json!({
            "counter": solution.nonce,
            "derivedKey": derived_key,
            "time": solve_ms,
            "attempted_nonces": solver.attempted_nonces(),
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altcha_prefix_target_places_bytes_at_hash_front() -> anyhow::Result<()> {
        let target = prefix_target("12ab").context("valid fixed ALTCHA prefix")?;
        assert_eq!(target, (0x12ab_0000_0000_0000, 0xffff_0000_0000_0000));
        Ok(())
    }

    #[test]
    fn altcha_fast_path_declines_nonstandard_nonce_shape() -> anyhow::Result<()> {
        let challenge = AltchaChallenge {
            parameters: AltchaChallengeParameters {
                algorithm: "SHA-256".to_owned(),
                cost: 1,
                key_length: 32,
                key_prefix: "00".to_owned(),
                memory_cost: None,
                nonce: "00".to_owned(),
                parallelism: None,
                salt: "00".repeat(16),
            },
        };
        assert!(try_solve_altcha_fast(&challenge, Implementation::Safe)?.is_none());
        Ok(())
    }

    #[test]
    fn altcha_string_counter_derivation_covers_js_safe_integer_max() -> anyhow::Result<()> {
        let parameters = AltchaChallengeParameters {
            algorithm: "SHA-256".to_owned(),
            cost: 2,
            key_length: 32,
            key_prefix: "00".to_owned(),
            memory_cost: None,
            nonce: hex::encode((0_u8..16).collect::<Vec<_>>()),
            parallelism: None,
            salt: hex::encode((16_u8..32).collect::<Vec<_>>()),
        };
        let nonce = hex::decode(&parameters.nonce)?;
        let salt = hex::decode(&parameters.salt)?;
        let mut password = nonce;
        password.extend_from_slice(b"9007199254740991");
        let derived = altcha_derive_key(&parameters, &salt, &password)?;
        assert_eq!(
            hex::encode(derived),
            "fb46d9456a5caf40eb17da502cdec45ac202cc4f30f50ee667bddb40904be548"
        );
        Ok(())
    }

    #[test]
    fn altcha_generic_modes_match_upstream_empty_prefix_and_case_insensitive_algorithm()
    -> anyhow::Result<()> {
        let challenge = AltchaChallenge {
            parameters: AltchaChallengeParameters {
                algorithm: "sha-256".to_owned(),
                cost: 1,
                key_length: 32,
                key_prefix: String::new(),
                memory_cost: None,
                nonce: "00".to_owned(),
                parallelism: None,
                salt: "00".repeat(16),
            },
        };
        for (mode, implementation) in [
            (AltchaCounterMode::Uint32, "altcha-generic"),
            (AltchaCounterMode::String, "altcha-string"),
        ] {
            let result = solve_altcha_generic(&challenge, mode)?;
            assert_eq!(result.answer["counter"], 0);
            assert_eq!(result.implementation, implementation);
        }
        Ok(())
    }
}
