use std::{
    borrow::Cow,
    num::NonZeroU32,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Context, bail, ensure};
use argon2::{
    Algorithm as Argon2Algorithm, Argon2, Block as Argon2Block, Params as Argon2Params,
    Version as Argon2Version,
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
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2AltchaSha256Solver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
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

pub(super) fn solve(
    challenge: Value,
    implementation: Implementation,
    worker_count: usize,
) -> anyhow::Result<Result> {
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
        return solve_altcha_generic(&official, counter_mode, worker_count);
    }
    if let Some(result) = try_solve_altcha_fast(&official, implementation)? {
        return Ok(result);
    }
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "this ALTCHA algorithm supports auto/safe implementations only"
    );
    solve_altcha_generic(&official, counter_mode, worker_count)
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

fn altcha_fixed_message(challenge: &AltchaChallenge) -> Option<(AltchaMessage, usize)> {
    let parameters = &challenge.parameters;
    if !matches!(parameters.algorithm.as_str(), "SHA-256" | "PBKDF2/SHA-256")
        || parameters.cost == 0
        || parameters.key_length == 0
        || parameters.key_length > 32
    {
        return None;
    }
    let nonce: [u8; 16] = hex::decode(&parameters.nonce)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())?;
    let salt: [u8; 16] = hex::decode(&parameters.salt)
        .ok()
        .and_then(|bytes| bytes.try_into().ok())?;
    let pbkdf2 = parameters.algorithm == "PBKDF2/SHA-256";
    let message = AltchaMessage {
        nonce,
        salt,
        cost: NonZeroU32::new(parameters.cost)?,
        pbkdf2,
        // The optimized nested SHA kernel takes its *intermediate* digest
        // length from this field. ALTCHA v2 truncates only the final result,
        // so all SHA-256 iterations must consume the complete 32-byte digest.
        key_length: NonZeroU32::new(if pbkdf2 {
            u32::try_from(parameters.key_length).ok()?
        } else {
            32
        })?,
    };
    Some((message, parameters.key_length))
}

fn altcha_fast_case(challenge: &AltchaChallenge) -> Option<(AltchaMessage, u64, u64, usize)> {
    let (message, key_length) = altcha_fixed_message(challenge)?;
    let (target, mask) = prefix_target(&challenge.parameters.key_prefix).ok()?;
    Some((message, target, mask, key_length))
}

#[cfg(feature = "gpu")]
enum AltchaGpuInput {
    /// Prehash arbitrary-length salt/nonce data exactly once on the host.
    Sha256 {
        salt: Vec<u8>,
        nonce: Vec<u8>,
        cost: NonZeroU32,
    },
    /// Retain the specialized fixed-key/fixed-salt fast HMAC path.
    Pbkdf2Fixed(AltchaMessage),
    /// Arbitrary-length HMAC key and salt; full block padding and password
    /// prehashing are handled by the general GPU backend.
    Pbkdf2General {
        salt: Vec<u8>,
        nonce: Vec<u8>,
        cost: NonZeroU32,
    },
}

#[cfg(feature = "gpu")]
struct AltchaGpuCase {
    input: AltchaGpuInput,
    prefix: Vec<u8>,
    target: u64,
    mask: u64,
    key_length: usize,
}

/// A GPU routing cost estimate, expressed in SHA-256 compression-equivalent
/// units. It is a dispatch heuristic, not a claim about the winning nonce:
/// official ALTCHA challenges can contain precomputed exact prefixes.
#[cfg(feature = "gpu")]
pub(super) fn auto_gpu_expected_work(challenge: &Value) -> Option<u128> {
    let counter_mode = altcha_counter_mode(challenge).ok()?;
    let parsed: AltchaChallenge = serde_json::from_value(challenge.clone()).ok()?;
    let gpu_case = altcha_gpu_case(&parsed).ok()?;
    let counter_bits: usize = match counter_mode {
        AltchaCounterMode::Uint32 => 32,
        AltchaCounterMode::String => 53,
    };
    let difficulty_bits = gpu_case.prefix.len().saturating_mul(4).min(counter_bits);
    let candidates = 1_u128 << difficulty_bits;
    let (cost, pbkdf2) = match gpu_case.input {
        AltchaGpuInput::Sha256 { cost, .. } => (cost.get(), false),
        AltchaGpuInput::Pbkdf2Fixed(message) => (message.cost.get(), true),
        AltchaGpuInput::Pbkdf2General { cost, .. } => (cost.get(), true),
    };
    let kdf_factor = if pbkdf2 { 2_u128 } else { 1 };
    Some(
        candidates
            .saturating_mul(u128::from(cost))
            .saturating_mul(kdf_factor),
    )
}

#[cfg(feature = "gpu")]
fn altcha_gpu_case(challenge: &AltchaChallenge) -> anyhow::Result<AltchaGpuCase> {
    let params = &challenge.parameters;
    ensure!(
        params.key_length > 0,
        "ALTCHA GPU key length must be positive"
    );
    if params.algorithm == "SHA-256" {
        ensure!(
            params.key_length <= 32,
            "ALTCHA iterative SHA-256 output cannot exceed 32 bytes"
        );
    } else if params.algorithm == "PBKDF2/SHA-256" {
        ensure!(
            u64::try_from(params.key_length).is_ok_and(|n| n <= u64::from(u32::MAX) * 32),
            "ALTCHA PBKDF2-SHA256 output exceeds RFC 8018"
        );
    }
    let cost = NonZeroU32::new(params.cost).context("ALTCHA GPU cost must be positive")?;
    let input = match params.algorithm.as_str() {
        "SHA-256" => AltchaGpuInput::Sha256 {
            salt: hex::decode(&params.salt).context("invalid ALTCHA SHA-256 salt hex")?,
            nonce: hex::decode(&params.nonce).context("invalid ALTCHA SHA-256 nonce hex")?,
            cost,
        },
        "PBKDF2/SHA-256" => {
            let salt = hex::decode(&params.salt).context("invalid ALTCHA PBKDF2 salt hex")?;
            let nonce = hex::decode(&params.nonce).context("invalid ALTCHA PBKDF2 nonce hex")?;
            if let Some((message, _)) = altcha_fixed_message(challenge) {
                AltchaGpuInput::Pbkdf2Fixed(message)
            } else {
                AltchaGpuInput::Pbkdf2General { salt, nonce, cost }
            }
        }
        _ => bail!("ALTCHA GPU supports SHA-256 and PBKDF2/SHA-256 only"),
    };
    let prefix = AltchaHexPrefix::parse(&params.key_prefix)?;
    ensure!(
        prefix.0.len() <= params.key_length * 2,
        "ALTCHA GPU prefix exceeds the derived key length"
    );
    let mut target = 0_u64;
    for (index, nibble) in prefix.0.iter().take(16).enumerate() {
        target |= u64::from(*nibble) << (60 - index * 4);
    }
    let significant_bits = prefix.0.len().min(16) * 4;
    let mask = if significant_bits == 0 {
        0
    } else {
        u64::MAX << (64 - significant_bits)
    };
    Ok(AltchaGpuCase {
        input,
        prefix: prefix.0,
        target,
        mask,
        key_length: params.key_length,
    })
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

/// Reuses validated Argon2id parameters and the complete memory arena across
/// counter attempts, independent of the ALTCHA counter encoding.
struct AltchaArgon2Worker {
    argon2: Argon2<'static>,
    memory: Vec<Argon2Block>,
    output: Vec<u8>,
}

impl AltchaArgon2Worker {
    fn new(parameters: &AltchaChallengeParameters) -> anyhow::Result<Self> {
        let params = Argon2Params::new(
            parameters
                .memory_cost
                .context("ALTCHA Argon2id requires memoryCost")?,
            parameters.cost,
            parameters.parallelism.unwrap_or(1),
            Some(parameters.key_length),
        )
        .context("invalid ALTCHA Argon2id parameters")?;
        let memory = vec![Argon2Block::default(); params.block_count()];
        Ok(Self {
            argon2: Argon2::new(Argon2Algorithm::Argon2id, Argon2Version::V0x13, params),
            memory,
            output: vec![0; parameters.key_length],
        })
    }

    fn derive(&mut self, password: &[u8], salt: &[u8]) -> anyhow::Result<&[u8]> {
        self.argon2
            .hash_password_into_with_memory(password, salt, &mut self.output, &mut self.memory)
            .context("ALTCHA Argon2id derivation failed")?;
        Ok(&self.output)
    }
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
        "ARGON2ID" => AltchaArgon2Worker::new(parameters)?
            .derive(password, salt)
            .map(<[u8]>::to_vec),
        other => bail!("unsupported ALTCHA algorithm: {other}"),
    }
}

fn altcha_generic_result(
    counter: u64,
    derived: Vec<u8>,
    started: Instant,
    implementation: &str,
    attempted: u64,
) -> Result {
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Result {
        answer: json!({
            "counter": counter,
            "derivedKey": hex::encode(derived),
            "time": solve_ms,
            "attempted_nonces": attempted,
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    }
}

fn solve_altcha_generic(
    challenge: &AltchaChallenge,
    counter_mode: AltchaCounterMode,
    worker_count: usize,
) -> anyhow::Result<Result> {
    let parameters = &challenge.parameters;
    let nonce = hex::decode(&parameters.nonce).context("invalid ALTCHA nonce hex")?;
    let salt = hex::decode(&parameters.salt).context("invalid ALTCHA salt hex")?;
    let prefix = AltchaHexPrefix::parse(&parameters.key_prefix)?;
    if parameters.algorithm.eq_ignore_ascii_case("ARGON2ID") {
        ensure!(
            prefix.0.len() <= parameters.key_length.saturating_mul(2),
            "ALTCHA Argon2id keyPrefix exceeds derived key length"
        );
    }
    let started = Instant::now();
    if worker_count > 1
        && !prefix.0.is_empty()
        && parameters.algorithm.eq_ignore_ascii_case("ARGON2ID")
    {
        return solve_altcha_argon2_parallel(
            parameters,
            counter_mode,
            &nonce,
            &salt,
            &prefix,
            worker_count,
            started,
        );
    }
    let mut password = Vec::with_capacity(nonce.len() + 16);
    let mut argon2_worker = if parameters.algorithm.eq_ignore_ascii_case("ARGON2ID") {
        Some(AltchaArgon2Worker::new(parameters)?)
    } else {
        None
    };

    match counter_mode {
        AltchaCounterMode::Uint32 => {
            for counter in 0..=u32::MAX {
                password.clear();
                password.extend_from_slice(&nonce);
                password.extend_from_slice(&counter.to_be_bytes());
                let derived = if let Some(worker) = argon2_worker.as_mut() {
                    Cow::Borrowed(worker.derive(&password, &salt)?)
                } else {
                    Cow::Owned(altcha_derive_key(parameters, &salt, &password)?)
                };
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        u64::from(counter),
                        derived.into_owned(),
                        started,
                        "altcha-generic",
                        u64::from(counter) + 1,
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
                let derived = if let Some(worker) = argon2_worker.as_mut() {
                    Cow::Borrowed(worker.derive(&password, &salt)?)
                } else {
                    Cow::Owned(altcha_derive_key(parameters, &salt, &password)?)
                };
                if prefix.matches(&derived) {
                    return Ok(altcha_generic_result(
                        counter,
                        derived.into_owned(),
                        started,
                        "altcha-string",
                        counter + 1,
                    ));
                }
            }
            bail!("ALTCHA string-counter search exhausted Number.MAX_SAFE_INTEGER")
        }
    }
}

/// Search disjoint, strided counter sequences with independent Argon2 arenas.
/// A worker may finish after a larger counter succeeds, but all smaller
/// counters are still checked before the smallest valid answer is returned.
fn solve_altcha_argon2_parallel(
    parameters: &AltchaChallengeParameters,
    counter_mode: AltchaCounterMode,
    nonce: &[u8],
    salt: &[u8],
    prefix: &AltchaHexPrefix,
    workers: usize,
    started: Instant,
) -> anyhow::Result<Result> {
    let upper_bound = match counter_mode {
        AltchaCounterMode::Uint32 => u64::from(u32::MAX) + 1,
        AltchaCounterMode::String => JS_MAX_SAFE_INTEGER + 1,
    };
    let stride = u64::try_from(workers).context("ALTCHA worker count overflow")?;
    let best = AtomicU64::new(upper_bound);
    let attempts = AtomicU64::new(0);
    let winner = Mutex::new(None::<(u64, Vec<u8>)>);

    thread::scope(|scope| -> anyhow::Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let best = &best;
            let attempts = &attempts;
            let winner = &winner;
            handles.push(scope.spawn(move || -> anyhow::Result<()> {
                let mut worker = AltchaArgon2Worker::new(parameters)?;
                let mut password = Vec::with_capacity(nonce.len() + 16);
                password.extend_from_slice(nonce);
                let mut buffer = itoa::Buffer::new();
                let mut counter = u64::try_from(index).context("ALTCHA worker index overflow")?;
                while counter < upper_bound && counter < best.load(Ordering::Acquire) {
                    password.truncate(nonce.len());
                    match counter_mode {
                        AltchaCounterMode::Uint32 => {
                            let n = u32::try_from(counter).context("ALTCHA uint32 overflow")?;
                            password.extend_from_slice(&n.to_be_bytes());
                        }
                        AltchaCounterMode::String => {
                            password.extend_from_slice(buffer.format(counter).as_bytes());
                        }
                    }
                    let digest = worker.derive(&password, salt)?;
                    attempts.fetch_add(1, Ordering::Relaxed);
                    if prefix.matches(digest) {
                        let previous = best.fetch_min(counter, Ordering::AcqRel);
                        if counter < previous {
                            let mut guard = winner
                                .lock()
                                .map_err(|_| anyhow::anyhow!("ALTCHA winner lock poisoned"))?;
                            if guard.as_ref().is_none_or(|(current, _)| counter < *current) {
                                *guard = Some((counter, digest.to_vec()));
                            }
                        }
                        break;
                    }
                    counter = counter.checked_add(stride).unwrap_or(upper_bound);
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("ALTCHA Argon2 worker panicked"))??;
        }
        Ok(())
    })?;

    let (counter, digest) = winner
        .into_inner()
        .map_err(|_| anyhow::anyhow!("ALTCHA winner lock poisoned"))?
        .context("ALTCHA Argon2id search exhausted the counter domain")?;
    let implementation = match counter_mode {
        AltchaCounterMode::Uint32 => "altcha-generic",
        AltchaCounterMode::String => "altcha-string",
    };
    Ok(altcha_generic_result(
        counter,
        digest,
        started,
        implementation,
        attempts.load(Ordering::Relaxed),
    ))
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
    let counter_mode = altcha_counter_mode(&challenge)?;
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let gpu_case = altcha_gpu_case(&challenge)?;
    let mut solver = match gpu_case.input {
        AltchaGpuInput::Sha256 { salt, nonce, cost } => {
            challenge_solvers::gpu::AltchaSha256Solver::for_sha256_prefix(&salt, &nonce, cost)?
        }
        AltchaGpuInput::Pbkdf2Fixed(message) => {
            let mut solver = challenge_solvers::gpu::AltchaSha256Solver::from(message);
            solver.set_pbkdf2_key_length(gpu_case.key_length)?;
            solver
        }
        AltchaGpuInput::Pbkdf2General { salt, nonce, cost } => {
            let mut solver =
                challenge_solvers::gpu::AltchaSha256Solver::for_pbkdf2(&salt, &nonce, cost)?;
            solver.set_pbkdf2_key_length(gpu_case.key_length)?;
            solver
        }
    };
    solver.set_prefix_nibbles(&gpu_case.prefix)?;
    if counter_mode == AltchaCounterMode::String {
        solver.set_string_counter();
        solver.set_limit(JS_MAX_SAFE_INTEGER + 1);
    } else {
        solver.set_limit(u64::from(u32::MAX) + 1);
    }
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, gpu_case.target, gpu_case.mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let verified_key = solver
        .verified_key()
        .context("ALTCHA GPU did not retain its independently verified derived key")?;
    let derived_key = hex::encode(
        verified_key
            .get(..gpu_case.key_length)
            .context("ALTCHA GPU returned a short verified derived key")?,
    );
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
    fn altcha_argon2_reused_memory_matches_reference_for_variable_parameters() -> anyhow::Result<()>
    {
        for (memory_kib, passes, lanes, key_length) in [
            (32, 1, 1, 16),
            (61, 2, 1, 32),
            (96, 3, 2, 64),
            (173, 2, 4, 96),
        ] {
            let parameters = AltchaChallengeParameters {
                algorithm: "argon2id".to_owned(),
                cost: passes,
                key_length,
                key_prefix: String::new(),
                memory_cost: Some(memory_kib),
                nonce: "00".repeat(16),
                parallelism: Some(lanes),
                salt: "01".repeat(16),
            };
            let mut worker = AltchaArgon2Worker::new(&parameters)?;
            let params = Argon2Params::new(memory_kib, passes, lanes, Some(key_length))?;
            let reference = Argon2::new(Argon2Algorithm::Argon2id, Argon2Version::V0x13, params);
            for (password, salt) in [
                (&b"sample-1"[..], &b"salt1234"[..]),
                (&b"reused-arena-2"[..], &b"longer-salt-value"[..]),
                (&b"sample-1"[..], &b"salt1234"[..]),
            ] {
                let mut expected = vec![0; key_length];
                reference.hash_password_into(password, salt, &mut expected)?;
                assert_eq!(worker.derive(password, salt)?, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn altcha_argon2_both_counter_modes_use_full_variable_length_output() -> anyhow::Result<()> {
        let challenge = AltchaChallenge {
            parameters: AltchaChallengeParameters {
                algorithm: "argon2id".to_owned(),
                cost: 2,
                key_length: 96,
                key_prefix: String::new(),
                memory_cost: Some(97),
                nonce: "31".repeat(16),
                parallelism: Some(2),
                salt: "43".repeat(16),
            },
        };
        let salt = hex::decode(&challenge.parameters.salt)?;
        let nonce = hex::decode(&challenge.parameters.nonce)?;
        for (mode, counter_suffix) in [
            (AltchaCounterMode::Uint32, &0_u32.to_be_bytes()[..]),
            (AltchaCounterMode::String, &b"0"[..]),
        ] {
            let result = solve_altcha_generic(&challenge, mode, 1)?;
            assert_eq!(result.answer["counter"], 0);
            let mut password = nonce.clone();
            password.extend_from_slice(counter_suffix);
            let expected = altcha_derive_key(&challenge.parameters, &salt, &password)?;
            assert_eq!(result.answer["derivedKey"], hex::encode(expected));
        }
        Ok(())
    }

    #[test]
    fn altcha_argon2_parallel_search_preserves_lowest_valid_counter_and_counts_actual_work()
    -> anyhow::Result<()> {
        for (mode, lanes) in [
            (AltchaCounterMode::Uint32, 1),
            (AltchaCounterMode::String, 2),
        ] {
            let mut challenge = AltchaChallenge {
                parameters: AltchaChallengeParameters {
                    algorithm: "ARGON2ID".to_owned(),
                    cost: 2,
                    key_length: 96,
                    key_prefix: String::new(),
                    memory_cost: Some(97),
                    nonce: "21".repeat(16),
                    parallelism: Some(lanes),
                    salt: "12".repeat(16),
                },
            };
            let salt = hex::decode(&challenge.parameters.salt)?;
            let mut password = hex::decode(&challenge.parameters.nonce)?;
            let target = 9_u64;
            match mode {
                AltchaCounterMode::Uint32 => {
                    password.extend_from_slice(&u32::try_from(target)?.to_be_bytes());
                }
                AltchaCounterMode::String => {
                    password.extend_from_slice(target.to_string().as_bytes());
                }
            }
            let expected = altcha_derive_key(&challenge.parameters, &salt, &password)?;
            challenge.parameters.key_prefix = hex::encode(&expected);

            let serial = solve_altcha_generic(&challenge, mode, 1)?;
            let parallel = solve_altcha_generic(&challenge, mode, 4)?;
            assert_eq!(serial.answer["counter"], target);
            assert_eq!(parallel.answer["counter"], target);
            assert_eq!(parallel.answer["derivedKey"], serial.answer["derivedKey"]);
            assert!(parallel.answer["attempted_nonces"].as_u64().unwrap_or(0) >= target + 1);
            assert_eq!(serial.answer["attempted_nonces"], target + 1);
        }
        Ok(())
    }

    #[test]
    fn altcha_cpu_auto_fast_short_prefix_respects_worker_stack() -> anyhow::Result<()> {
        // Debug builds on AVX2 hosts previously overflowed Tokio's worker
        // stack when the dispatcher inlined all specialized SIMD variants.
        for (algorithm, cost) in [("SHA-256", 2), ("PBKDF2/SHA-256", 16)] {
            let mut challenge = AltchaChallenge {
                parameters: AltchaChallengeParameters {
                    algorithm: algorithm.to_owned(),
                    cost,
                    key_length: 32,
                    key_prefix: String::new(),
                    memory_cost: None,
                    nonce: "2a".repeat(16),
                    parallelism: None,
                    salt: "51".repeat(16),
                },
            };
            let salt = hex::decode(&challenge.parameters.salt)?;
            let mut password = hex::decode(&challenge.parameters.nonce)?;
            password.extend_from_slice(&0_u32.to_be_bytes());
            let expected = altcha_derive_key(&challenge.parameters, &salt, &password)?;
            challenge.parameters.key_prefix = hex::encode(&expected[..1]);
            let solved = try_solve_altcha_fast(&challenge, Implementation::Auto)?
                .context("expected optimized ALTCHA search")?;
            assert_eq!(solved.answer["counter"], 0);
            assert_eq!(solved.answer["derivedKey"], hex::encode(expected));
        }
        Ok(())
    }

    #[test]
    fn altcha_optimized_sha_truncates_only_the_final_digest() -> anyhow::Result<()> {
        for algorithm in ["SHA-256", "PBKDF2/SHA-256"] {
            let mut challenge = AltchaChallenge {
                parameters: AltchaChallengeParameters {
                    algorithm: algorithm.to_owned(),
                    cost: 3,
                    key_length: 12,
                    key_prefix: String::new(),
                    memory_cost: None,
                    nonce: "55".repeat(16),
                    parallelism: None,
                    salt: "cc".repeat(16),
                },
            };
            let salt = hex::decode(&challenge.parameters.salt)?;
            let mut password = hex::decode(&challenge.parameters.nonce)?;
            password.extend_from_slice(&0_u32.to_be_bytes());
            let expected = altcha_derive_key(&challenge.parameters, &salt, &password)?;
            assert_eq!(expected.len(), 12);
            challenge.parameters.key_prefix = hex::encode(&expected[..2]);
            let optimized = try_solve_altcha_fast(&challenge, Implementation::Safe)?
                .context("expected optimized SHA256/PBKDF2 path")?;
            assert_eq!(optimized.answer["counter"], 0);
            assert_eq!(optimized.answer["derivedKey"], hex::encode(&expected));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn altcha_gpu_cost_hint_uses_validated_shape_and_counter_domain() -> anyhow::Result<()> {
        let mut request = serde_json::json!({
            "counterMode": "uint32",
            "parameters": {
                "algorithm": "SHA-256", "cost": 2, "keyLength": 12,
                "keyPrefix": "0", "nonce": "01".repeat(16),
                "salt": "02".repeat(16)
            }
        });
        assert_eq!(auto_gpu_expected_work(&request), Some(32));
        request["parameters"]["keyPrefix"] = serde_json::json!("00");
        assert_eq!(auto_gpu_expected_work(&request), Some(512));
        request["parameters"]["keyPrefix"] = serde_json::json!("000");
        assert_eq!(auto_gpu_expected_work(&request), Some(8_192));
        request["parameters"]["algorithm"] = serde_json::json!("PBKDF2/SHA-256");
        request["parameters"]["cost"] = serde_json::json!(16);
        request["parameters"]["keyPrefix"] = serde_json::json!("00");
        assert_eq!(auto_gpu_expected_work(&request), Some(8_192));
        request["parameters"]["keyPrefix"] = serde_json::json!("a".repeat(24));
        assert_eq!(
            auto_gpu_expected_work(&request),
            Some((1_u128 << 32) * 16 * 2)
        );
        request["counterMode"] = serde_json::json!("string");
        assert_eq!(
            auto_gpu_expected_work(&request),
            Some((1_u128 << 53) * 16 * 2)
        );
        request["parameters"]["algorithm"] = serde_json::json!("SCRYPT");
        assert_eq!(auto_gpu_expected_work(&request), None);
        request["parameters"]["algorithm"] = serde_json::json!("SHA-256");
        request["parameters"]["keyPrefix"] = serde_json::json!("bad!");
        assert_eq!(auto_gpu_expected_work(&request), None);
        request["parameters"]["keyPrefix"] = serde_json::json!("a");
        request["parameters"]["nonce"] = serde_json::json!("00");
        assert_eq!(auto_gpu_expected_work(&request), Some(256));
        request["parameters"]["salt"] = serde_json::json!("01".repeat(130));
        assert_eq!(auto_gpu_expected_work(&request), Some(256));
        request["parameters"]["algorithm"] = serde_json::json!("PBKDF2/SHA-256");
        assert_eq!(auto_gpu_expected_work(&request), Some(512));
        request["parameters"]["nonce"] = serde_json::json!("ab".repeat(128));
        assert_eq!(auto_gpu_expected_work(&request), Some(512));
        request["parameters"]["salt"] = serde_json::json!("");
        assert_eq!(auto_gpu_expected_work(&request), Some(512));
        request["parameters"]["algorithm"] = serde_json::json!("SHA-256");
        request["parameters"]["nonce"] = serde_json::json!("xx");
        assert_eq!(auto_gpu_expected_work(&request), None);
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn altcha_gpu_accepts_odd_and_full_digest_hex_prefixes() -> anyhow::Result<()> {
        let mut challenge = AltchaChallenge {
            parameters: AltchaChallengeParameters {
                algorithm: "SHA-256".to_owned(),
                cost: 2,
                key_length: 32,
                key_prefix: String::new(),
                memory_cost: None,
                nonce: "01".repeat(16),
                parallelism: None,
                salt: "02".repeat(16),
            },
        };
        for prefix in [
            String::new(),
            "0".to_owned(),
            "AbC".to_owned(),
            "12ab34".to_owned(),
            "aBcDeF".repeat(10),
        ] {
            challenge.parameters.key_prefix = prefix.clone();
            let gpu_case = altcha_gpu_case(&challenge)?;
            assert_eq!(gpu_case.prefix.len(), prefix.len());
            assert_eq!(gpu_case.key_length, 32);
            if prefix == "AbC" {
                assert_eq!(gpu_case.target, 0xabc0_0000_0000_0000);
                assert_eq!(gpu_case.mask, 0xfff0_0000_0000_0000);
            }
        }
        challenge.parameters.key_prefix = "a".repeat(65);
        assert!(altcha_gpu_case(&challenge).is_err());
        challenge.parameters.key_prefix = "hello".to_owned();
        assert!(altcha_gpu_case(&challenge).is_err());
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn altcha_gpu_sha_accepts_full_block_boundary_and_large_fixed_inputs() -> anyhow::Result<()> {
        let mut challenge = AltchaChallenge {
            parameters: AltchaChallengeParameters {
                algorithm: "SHA-256".to_owned(),
                cost: 3,
                key_length: 12,
                key_prefix: "abc".to_owned(),
                memory_cost: None,
                nonce: String::new(),
                parallelism: None,
                salt: String::new(),
            },
        };
        for length in [
            0_usize, 1, 16, 55, 56, 63, 64, 65, 119, 120, 127, 128, 129, 1024, 4097,
        ] {
            challenge.parameters.salt = "a1".repeat(length);
            challenge.parameters.nonce = "ef".repeat(length % 19);
            let gpu_case = altcha_gpu_case(&challenge)?;
            assert_eq!(gpu_case.key_length, 12);
            assert_eq!(gpu_case.prefix, vec![10, 11, 12]);
            assert!(matches!(gpu_case.input, AltchaGpuInput::Sha256 { .. }));
        }
        challenge.parameters.salt = "xx".to_owned();
        assert!(altcha_gpu_case(&challenge).is_err());
        challenge.parameters.salt = "aa".to_owned();
        challenge.parameters.nonce = "0".to_owned();
        assert!(altcha_gpu_case(&challenge).is_err());
        Ok(())
    }

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
            let result = solve_altcha_generic(&challenge, mode, 1)?;
            assert_eq!(result.answer["counter"], 0);
            assert_eq!(result.implementation, implementation);
        }
        Ok(())
    }
}
