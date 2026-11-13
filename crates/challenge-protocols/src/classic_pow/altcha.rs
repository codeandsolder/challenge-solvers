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
use argon2_rust::{
    Algorithm as Argon2Algorithm, Argon2, Hasher as Argon2Hasher, Params as Argon2Params,
    Version as Argon2Version,
    params::{Memory as Argon2Memory, TagLen as Argon2TagLen},
};
#[cfg(feature = "gpu")]
use challenge_solvers::gpu::{AltchaSha64Algorithm, AltchaSha64GpuSolver};
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

#[derive(Debug, Deserialize, serde::Serialize)]
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
    /// Native 64-bit SHA-2 family, including HMAC/PBKDF2 with arbitrary
    /// valid salt, nonce, derived length, and counter encodings.
    Sha64(AltchaSha64GpuSolver),
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
    if challenge.get("parameters").is_none() {
        let case = parse_altcha_v1(challenge.clone()).ok()?;
        return Some(u128::from(case.maxnumber.unwrap_or(1_000_000)) + 1);
    }
    let counter_mode = altcha_counter_mode(challenge).ok()?;
    let parsed: AltchaChallenge = serde_json::from_value(challenge.clone()).ok()?;
    let gpu_case = altcha_gpu_case(&parsed).ok()?;
    let counter_bits: usize = match counter_mode {
        AltchaCounterMode::Uint32 => 32,
        AltchaCounterMode::String => 53,
    };
    let difficulty_bits = gpu_case.prefix.len().saturating_mul(4).min(counter_bits);
    let candidates = 1_u128 << difficulty_bits;
    // Eligibility and parameter validity come from the actual GPU parser.
    // All supported ALTCHA v2 algorithms share the same cost parameter.
    let cost = parsed.parameters.cost;
    let pbkdf2 = parsed
        .parameters
        .algorithm
        .to_ascii_uppercase()
        .starts_with("PBKDF2/");
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
    let algorithm = params.algorithm.to_ascii_uppercase();
    if algorithm == "SHA-256" {
        ensure!(
            params.key_length <= 32,
            "ALTCHA iterative SHA-256 output cannot exceed 32 bytes"
        );
    } else if algorithm == "PBKDF2/SHA-256" {
        ensure!(
            u64::try_from(params.key_length).is_ok_and(|n| n <= u64::from(u32::MAX) * 32),
            "ALTCHA PBKDF2-SHA256 output exceeds RFC 8018"
        );
    }
    let cost = NonZeroU32::new(params.cost).context("ALTCHA GPU cost must be positive")?;
    let input = match algorithm.as_str() {
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
        "SHA-384" | "SHA-512" | "PBKDF2/SHA-384" | "PBKDF2/SHA-512" => {
            let mode = match algorithm.as_str() {
                "SHA-384" => AltchaSha64Algorithm::Sha384,
                "SHA-512" => AltchaSha64Algorithm::Sha512,
                "PBKDF2/SHA-384" => AltchaSha64Algorithm::Pbkdf2Sha384,
                "PBKDF2/SHA-512" => AltchaSha64Algorithm::Pbkdf2Sha512,
                _ => unreachable!("validated SHA-384/512 variant"),
            };
            let salt = hex::decode(&params.salt).context("invalid ALTCHA SHA-384/512 salt hex")?;
            let nonce =
                hex::decode(&params.nonce).context("invalid ALTCHA SHA-384/512 nonce hex")?;
            AltchaGpuInput::Sha64(AltchaSha64GpuSolver::new(
                mode,
                &salt,
                &nonce,
                cost,
                params.key_length,
            )?)
        }
        _ => bail!("ALTCHA GPU algorithm not supported: {algorithm}"),
    };
    let prefix = AltchaHexPrefix::parse(&params.key_prefix)?;
    ensure!(
        prefix.0.len() <= params.key_length.saturating_mul(2),
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
fn altcha_scrypt_parameters(
    parameters: &AltchaChallengeParameters,
) -> anyhow::Result<scrypt::Params> {
    let log_n = parameters
        .cost
        .checked_ilog2()
        .filter(|_| parameters.cost.is_power_of_two())
        .context("ALTCHA Scrypt cost must be a power of two")?;
    scrypt::Params::new(
        u8::try_from(log_n).context("ALTCHA Scrypt log2(N) exceeds u8")?,
        parameters.memory_cost.unwrap_or(8),
        parameters.parallelism.unwrap_or(1),
    )
    .context("invalid ALTCHA Scrypt parameters")
}

struct AltchaScryptWorker {
    workspace: scrypt_fast::Workspace,
    output: Vec<u8>,
}

impl AltchaScryptWorker {
    fn new(parameters: &AltchaChallengeParameters) -> anyhow::Result<Self> {
        let params = altcha_scrypt_parameters(parameters)?;
        Ok(Self {
            workspace: scrypt_fast::Workspace::new(&params),
            output: vec![0; parameters.key_length],
        })
    }

    fn derive(&mut self, password: &[u8], salt: &[u8]) -> anyhow::Result<&[u8]> {
        self.workspace
            .derive(password, salt, &mut self.output)
            .context("ALTCHA Scrypt derivation failed")?;
        Ok(&self.output)
    }
}

/// Match the reference Argon2id parameter semantics, but constrain its
/// physical thread budget to the single CPU permit reserved per nonce worker.
fn altcha_argon2_parameters(
    parameters: &AltchaChallengeParameters,
) -> anyhow::Result<Argon2Params> {
    Argon2Params::builder()
        .memory(Argon2Memory::kib(u64::from(
            parameters
                .memory_cost
                .context("ALTCHA Argon2id requires memoryCost")?,
        )))
        .passes(parameters.cost)
        .lanes(parameters.parallelism.unwrap_or(1))
        .threads(1)
        .tag_len(Argon2TagLen::bytes(u64::try_from(parameters.key_length)?))
        .build()
        .context("invalid ALTCHA Argon2id parameters")
}

struct AltchaArgon2Worker {
    hasher: Argon2Hasher,
    output: Vec<u8>,
}

impl AltchaArgon2Worker {
    fn new(parameters: &AltchaChallengeParameters) -> anyhow::Result<Self> {
        let params = altcha_argon2_parameters(parameters)?;
        let mut hasher =
            Argon2::new(Argon2Algorithm::Argon2id, Argon2Version::V0x13, params).hasher();
        // Preserve the previous upfront allocation behavior. This allocation
        // occurs only after the service has acquired its KDF-memory reservation.
        hasher
            .reserve()
            .context("ALTCHA Argon2id arena allocation failed")?;
        Ok(Self {
            hasher,
            output: vec![0; parameters.key_length],
        })
    }

    fn derive(&mut self, password: &[u8], salt: &[u8]) -> anyhow::Result<&[u8]> {
        self.hasher
            .hash_into(password, salt, &mut self.output)
            .context("ALTCHA Argon2id derivation failed")?;
        Ok(&self.output)
    }
}

pub(super) fn memory_requirement(challenge: &Value) -> Option<anyhow::Result<u64>> {
    let algorithm = challenge.pointer("/parameters/algorithm")?.as_str()?;
    if !algorithm.eq_ignore_ascii_case("SCRYPT") && !algorithm.eq_ignore_ascii_case("ARGON2ID") {
        return None;
    }
    Some((|| {
        let parsed: AltchaChallenge =
            serde_json::from_value(challenge.clone()).context("invalid ALTCHA KDF challenge")?;
        let p = parsed.parameters;
        if algorithm.eq_ignore_ascii_case("ARGON2ID") {
            let params = altcha_argon2_parameters(&p)?;
            let arena = u64::from(params.memory_blocks())
                .checked_mul(1024)
                .context("ALTCHA Argon2id arena size overflow")?;
            return arena
                .checked_add(u64::try_from(p.key_length)?)
                .context("ALTCHA Argon2id output size overflow");
        }
        let params = altcha_scrypt_parameters(&p)?;
        super::scrypt_memory_bytes(&params, p.key_length)
    })())
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
        "SCRYPT" => Ok(AltchaScryptWorker::new(parameters)?
            .derive(password, salt)?
            .to_vec()),
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
    let mut scrypt_worker = if parameters.algorithm.eq_ignore_ascii_case("SCRYPT") {
        Some(AltchaScryptWorker::new(parameters)?)
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
                } else if let Some(worker) = scrypt_worker.as_mut() {
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
                } else if let Some(worker) = scrypt_worker.as_mut() {
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

fn parse_altcha_v1(challenge: Value) -> anyhow::Result<AltchaV1Challenge> {
    let challenge: AltchaV1Challenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v1 challenge")?;
    let digest_length = match challenge.algorithm.as_str() {
        "SHA-1" => 40,
        "SHA-256" => 64,
        "SHA-512" => 128,
        other => bail!("unsupported ALTCHA v1 algorithm: {other}"),
    };
    ensure!(
        !challenge.signature.is_empty(),
        "ALTCHA v1 signature must not be empty"
    );
    // The CPU protocol compares lowercase *hex strings*, not decoded bytes.
    // Decode/re-encode parity prevents GPU from accepting noncanonical hex.
    ensure!(
        challenge.challenge.len() == digest_length
            && hex::decode(&challenge.challenge)
                .is_ok_and(|bytes| hex::encode(bytes) == challenge.challenge),
        "ALTCHA v1 challenge must be a canonical lowercase full digest"
    );
    Ok(challenge)
}

#[cfg(feature = "gpu")]
async fn solve_altcha_v1_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    let challenge = parse_altcha_v1(challenge)?;
    let max = u64::from(challenge.maxnumber.unwrap_or(1_000_000)) + 1;
    let target = hex::decode(&challenge.challenge)?;
    let prefix = target
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 15])
        .collect::<Vec<u8>>();
    let started = Instant::now();
    let one = NonZeroU32::new(1).context("positive SHA iteration count")?;
    let number = match challenge.algorithm.as_str() {
        "SHA-256" => {
            let mut solver = challenge_solvers::gpu::AltchaSha256Solver::for_sha256_prefix(
                challenge.salt.as_bytes(),
                &[],
                one,
            )?;
            solver.set_string_counter();
            solver.set_limit(max);
            solver.set_prefix_nibbles(&prefix)?;
            solver
                .solve::<SOLVE_TYPE_MASK>(gpu, 0, 0)
                .await?
                .context("ALTCHA v1 SHA-256 GPU search exhausted maxnumber")?
                .nonce
        }
        "SHA-512" => {
            let mut solver = AltchaSha64GpuSolver::new(
                AltchaSha64Algorithm::Sha512,
                challenge.salt.as_bytes(),
                &[],
                one,
                64,
            )?;
            solver.set_string_counter();
            solver.set_limit(max);
            solver.set_prefix_nibbles(&prefix)?;
            solver
                .solve(gpu)
                .await?
                .context("ALTCHA v1 SHA-512 GPU search exhausted maxnumber")?
                .nonce
        }
        "SHA-1" => {
            let digest: [u8; 20] = target
                .as_slice()
                .try_into()
                .context("ALTCHA v1 SHA1 digest must be 20 bytes")?;
            let mut solver = challenge_solvers::gpu::Sha1DecimalGpuSolver::new(
                challenge.salt.as_bytes(),
                digest,
            )?;
            solver.set_limit(max);
            u64::from(
                solver
                    .solve(gpu)
                    .await?
                    .context("ALTCHA v1 SHA-1 GPU search exhausted maxnumber")?
                    .nonce,
            )
        }
        _ => unreachable!("validated ALTCHA v1 digest algorithm"),
    };
    let digest = match challenge.algorithm.as_str() {
        "SHA-1" => hex::encode(sha1::Sha1::digest(
            format!("{}{number}", challenge.salt).as_bytes(),
        )),
        "SHA-256" => hex::encode(Sha256::digest(
            format!("{}{number}", challenge.salt).as_bytes(),
        )),
        "SHA-512" => hex::encode(Sha512::digest(
            format!("{}{number}", challenge.salt).as_bytes(),
        )),
        _ => unreachable!("validated SHA-256/512 branch"),
    };
    ensure!(
        digest == challenge.challenge && number < max,
        "ALTCHA v1 GPU candidate failed independent full digest validation"
    );
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({"number": number, "took": solve_ms}),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

fn solve_altcha_v1(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "ALTCHA v1 supports auto/safe implementations only"
    );
    let challenge = parse_altcha_v1(challenge)?;
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
async fn solve_sha64_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    mut solver: AltchaSha64GpuSolver,
    prefix: &[u8],
    counter_mode: AltchaCounterMode,
) -> anyhow::Result<Result> {
    solver.set_prefix_nibbles(prefix)?;
    if counter_mode == AltchaCounterMode::String {
        solver.set_string_counter();
    }
    let started = Instant::now();
    let proof = solver
        .solve(gpu)
        .await?
        .context("ALTCHA SHA-384/512 GPU search exhausted configured counter domain")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Result {
        answer: json!({
            "counter": proof.nonce,
            "derivedKey": hex::encode(proof.derived_key),
            "time": solve_ms,
            "attempted_nonces": solver.attempted_nonces(),
        }),
        solve_ms,
        implementation: "wgpu".to_owned(),
    })
}

#[cfg(feature = "gpu")]
pub(super) async fn solve_gpu(
    gpu: &challenge_solvers::gpu::GpuContext,
    challenge: Value,
) -> anyhow::Result<Result> {
    if challenge.get("parameters").is_none() {
        return solve_altcha_v1_gpu(gpu, challenge).await;
    }
    let counter_mode = altcha_counter_mode(&challenge)?;
    let challenge: AltchaChallenge =
        serde_json::from_value(challenge).context("invalid ALTCHA v2 challenge")?;
    let AltchaGpuCase {
        input,
        prefix,
        target,
        mask,
        key_length,
    } = altcha_gpu_case(&challenge)?;
    let mut solver = match input {
        AltchaGpuInput::Sha64(solver) => {
            return solve_sha64_gpu(gpu, solver, &prefix, counter_mode).await;
        }
        AltchaGpuInput::Sha256 { salt, nonce, cost } => {
            challenge_solvers::gpu::AltchaSha256Solver::for_sha256_prefix(&salt, &nonce, cost)?
        }
        AltchaGpuInput::Pbkdf2Fixed(message) => {
            let mut solver = challenge_solvers::gpu::AltchaSha256Solver::from(message);
            solver.set_pbkdf2_key_length(key_length)?;
            solver
        }
        AltchaGpuInput::Pbkdf2General { salt, nonce, cost } => {
            let mut solver =
                challenge_solvers::gpu::AltchaSha256Solver::for_pbkdf2(&salt, &nonce, cost)?;
            solver.set_pbkdf2_key_length(key_length)?;
            solver
        }
    };
    solver.set_prefix_nibbles(&prefix)?;
    if counter_mode == AltchaCounterMode::String {
        solver.set_string_counter();
        solver.set_limit(JS_MAX_SAFE_INTEGER + 1);
    } else {
        solver.set_limit(u64::from(u32::MAX) + 1);
    }
    let started = Instant::now();
    let solution = solver
        .solve::<SOLVE_TYPE_MASK>(gpu, target, mask)
        .await?
        .context("ALTCHA GPU search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let verified_key = solver
        .verified_key()
        .context("ALTCHA GPU did not retain its independently verified derived key")?;
    let derived_key = hex::encode(
        verified_key
            .get(..key_length)
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
    fn altcha_v1_digest_must_be_canonical_full_lowercase() -> anyhow::Result<()> {
        let salt = "české-salt-12";
        for (algorithm, digest) in [
            (
                "SHA-1",
                hex::encode(sha1::Sha1::digest(format!("{salt}17").as_bytes())),
            ),
            (
                "SHA-256",
                hex::encode(Sha256::digest(format!("{salt}17").as_bytes())),
            ),
            (
                "SHA-512",
                hex::encode(Sha512::digest(format!("{salt}17").as_bytes())),
            ),
        ] {
            let good = json!({"algorithm": algorithm, "challenge": digest, "maxnumber": 18, "salt": salt, "signature": "signed"});
            assert!(parse_altcha_v1(good.clone()).is_ok());
            let solved = solve_altcha_v1(good, Implementation::Auto)?;
            assert_eq!(solved.answer["number"], 17);
            for invalid in [
                digest.to_ascii_uppercase(),
                digest[..digest.len() - 2].to_owned(),
                "ab!d".to_owned(),
            ] {
                assert!(parse_altcha_v1(json!({"algorithm": algorithm, "challenge": invalid, "salt": salt, "signature": "signed"})).is_err());
            }
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a physical WGPU adapter with SHADER_INT64"]
    fn altcha_v1_sha1_sha256_sha512_gpu_full_digest_and_boundaries() -> anyhow::Result<()> {
        use std::{
            future::Future,
            sync::Arc,
            task::{Context as TaskContext, Poll, Wake, Waker},
            time::Duration,
        };
        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        fn block_on<F: Future>(future: F) -> F::Output {
            let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
            let mut future = std::pin::pin!(future);
            loop {
                match future.as_mut().poll(&mut TaskContext::from_waker(&waker)) {
                    Poll::Ready(value) => return value,
                    Poll::Pending => std::thread::park_timeout(Duration::from_millis(100)),
                }
            }
        }
        let gpu = block_on(challenge_solvers::gpu::GpuContext::create())?;
        for algorithm in ["SHA-1", "SHA-256", "SHA-512"] {
            for (salt, target) in [
                ("hello-v1-Привет".to_owned(), 17u32),
                ("z".repeat(119), 99u32),
                ("t".repeat(1050), 1000u32),
                (String::new(), 0u32),
            ] {
                let plain = format!("{salt}{target}");
                let digest = match algorithm {
                    "SHA-1" => hex::encode(sha1::Sha1::digest(plain.as_bytes())),
                    "SHA-256" => hex::encode(Sha256::digest(plain.as_bytes())),
                    "SHA-512" => hex::encode(Sha512::digest(plain.as_bytes())),
                    _ => unreachable!("known SHA variant"),
                };
                let challenge = json!({"algorithm":algorithm,"challenge":digest,"salt":salt,"maxnumber":target+1,"signature":"signed"});
                let proof = block_on(solve_altcha_v1_gpu(&gpu, challenge))?;
                assert_eq!(proof.answer["number"], target);
                assert_eq!(proof.implementation, "wgpu");
                println!(
                    "verified ALTCHA v1 {algorithm} utf8_len={} target={target}",
                    salt.len()
                );
            }
        }
        Ok(())
    }

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
            let expected_blocks =
                argon2::Params::new(memory_kib, passes, lanes, Some(key_length))?.block_count();
            assert_eq!(worker.hasher.reserved_blocks(), expected_blocks);
            assert_eq!(worker.hasher.params().threads(), 1);
            let wire = json!({"parameters": {
                "algorithm": "ARGON2ID",
                "cost": passes,
                "memoryCost": memory_kib,
                "parallelism": lanes,
                "keyLength": key_length,
                "nonce": parameters.nonce,
                "salt": parameters.salt,
                "keyPrefix": ""
            }});
            let reserved = crate::classic_pow::memory_hard_requirement("altcha", &wire)
                .context("ALTCHA memory reservation missing")??;
            assert_eq!(
                reserved,
                u64::try_from(expected_blocks)? * 1024 + u64::try_from(key_length)?
            );
            let params = argon2::Params::new(memory_kib, passes, lanes, Some(key_length))?;
            let reference =
                argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
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
    fn altcha_sha64_gpu_parser_and_auto_cost_cover_all_modes() -> anyhow::Result<()> {
        for (algorithm, width, factor) in [
            ("SHA-384", 48, 1_u128),
            ("SHA-512", 64, 1),
            ("PBKDF2/SHA-384", 48, 2),
            ("PBKDF2/SHA-512", 64, 2),
        ] {
            let mut request = serde_json::json!({
                "counterMode": "uint32",
                "parameters": {
                    "algorithm": algorithm,
                    "cost": 3,
                    "keyLength": width,
                    "keyPrefix": "aBc",
                    "nonce": "44".repeat(155),
                    "salt": "55".repeat(143),
                }
            });
            assert_eq!(auto_gpu_expected_work(&request), Some(4096 * 3 * factor));
            request["counterMode"] = json!("string");
            assert_eq!(auto_gpu_expected_work(&request), Some(4096 * 3 * factor));
            request["parameters"]["keyLength"] = json!(1);
            assert_eq!(auto_gpu_expected_work(&request), None);
            request["parameters"]["keyLength"] = json!(width);
            request["parameters"]["salt"] = json!("gg");
            assert_eq!(auto_gpu_expected_work(&request), None);
            request["parameters"]["salt"] = json!("55".repeat(143));
            request["parameters"]["algorithm"] = json!(algorithm.to_ascii_lowercase());
            assert_eq!(auto_gpu_expected_work(&request), Some(4096 * 3 * factor));
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn make_altcha_sha64_protocol_case(
        algorithm: &str,
        key_length: usize,
        cost: u32,
        decimal: bool,
    ) -> anyhow::Result<(Value, Vec<u8>)> {
        let mut parameters = AltchaChallengeParameters {
            algorithm: algorithm.to_owned(),
            cost,
            key_length,
            key_prefix: String::new(),
            memory_cost: None,
            nonce: hex::encode(
                (0_u16..135)
                    .map(|i| u8::try_from((i * 13 + 7) % 256).expect("byte"))
                    .collect::<Vec<_>>(),
            ),
            parallelism: None,
            salt: hex::encode(
                (0_u16..157)
                    .map(|i| u8::try_from((i * 11 + 17) % 256).expect("byte"))
                    .collect::<Vec<_>>(),
            ),
        };
        let mut target: Option<(u64, Vec<u8>)> = None;
        for variant in 0..16_u8 {
            parameters
                .salt
                .replace_range(..2, &format!("{variant:02x}"));
            let salt = hex::decode(&parameters.salt)?;
            let nonce = hex::decode(&parameters.nonce)?;
            let mut previous = Vec::new();
            let mut winner = Vec::new();
            for counter in 0..=7_u32 {
                let mut password = nonce.clone();
                if decimal {
                    password.extend_from_slice(counter.to_string().as_bytes());
                } else {
                    password.extend_from_slice(&counter.to_be_bytes());
                }
                let key = altcha_derive_key(&parameters, &salt, &password)?;
                if counter == 7 {
                    winner = key;
                } else {
                    previous.push(key);
                }
            }
            let prefix = &winner[..2];
            if previous.iter().all(|key| key.get(..2) != Some(prefix)) {
                target = Some((7, winner));
                break;
            }
        }
        let (_, reference) = target.context("unable to find independent test prefix")?;
        parameters.key_prefix = hex::encode(&reference[..2]);
        let challenge = json!({
            "counterMode": if decimal {"string"} else {"uint32"},
            "parameters": parameters
        });
        Ok((challenge, reference))
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a physical GPU with native SHADER_INT64"]
    fn altcha_sha64_gpu_protocol_roundtrip() -> anyhow::Result<()> {
        use std::{
            future::Future,
            sync::Arc,
            task::{Context as TaskContext, Poll, Wake, Waker},
            time::Duration,
        };

        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }
        fn block_on<F: Future>(future: F) -> F::Output {
            let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
            let mut future = std::pin::pin!(future);
            loop {
                match future.as_mut().poll(&mut TaskContext::from_waker(&waker)) {
                    Poll::Ready(result) => return result,
                    Poll::Pending => std::thread::park_timeout(Duration::from_millis(100)),
                }
            }
        }
        let gpu = block_on(challenge_solvers::gpu::GpuContext::create())?;
        for (algorithm, key_length, cost, decimal) in [
            ("SHA-384", 31, 2, false),
            ("SHA-512", 49, 3, true),
            ("PBKDF2/SHA-384", 115, 2, false),
            ("PBKDF2/SHA-512", 147, 2, true),
        ] {
            let (challenge, reference) =
                make_altcha_sha64_protocol_case(algorithm, key_length, cost, decimal)?;
            assert!(auto_gpu_expected_work(&challenge).is_some());
            // The target at counter 7 is known valid, but many independent
            // GPU threads can find a different matching nonce in the same
            // dispatch. Any independently verified proof is correct.
            let parsed: AltchaChallenge = serde_json::from_value(challenge.clone())?;
            let salt = hex::decode(&parsed.parameters.salt)?;
            let nonce_prefix = hex::decode(&parsed.parameters.nonce)?;
            let mut known_password = nonce_prefix.clone();
            if decimal {
                known_password.extend_from_slice(b"7");
            } else {
                known_password.extend_from_slice(&7_u32.to_be_bytes());
            }
            assert_eq!(
                reference,
                altcha_derive_key(&parsed.parameters, &salt, &known_password)?
            );
            let result = block_on(solve_gpu(&gpu, challenge))?;
            let counter = result.answer["counter"]
                .as_u64()
                .context("GPU ALTCHA answer must contain a numeric counter")?;
            assert!(
                counter >= 7,
                "earlier candidates were excluded from this challenge"
            );
            let mut password = nonce_prefix;
            if decimal {
                password.extend_from_slice(counter.to_string().as_bytes());
            } else {
                password.extend_from_slice(&u32::try_from(counter)?.to_be_bytes());
            }
            let derived = altcha_derive_key(&parsed.parameters, &salt, &password)?;
            assert_eq!(
                result.answer["derivedKey"],
                hex::encode(&derived),
                "{algorithm} derived key for returned nonce"
            );
            assert!(
                AltchaHexPrefix::parse(&parsed.parameters.key_prefix)?.matches(&derived),
                "{algorithm} returned proof must match challenge prefix"
            );
            assert_eq!(result.implementation, "wgpu");
            println!("ALTCHA protocol passed {algorithm} decimal={decimal} counter={counter}");
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
