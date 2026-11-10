use std::{
    num::{NonZeroU8, NonZeroU64},
    time::Instant,
};

use anyhow::{Context, anyhow, bail, ensure};
use challenge_solvers::{
    DecimalSolver, build_mcaptcha_prefix,
    cerberus::ChallengeDescriptor as CerberusChallenge,
    compute_mask_goaway, compute_mask_zero_nibbles, compute_target_mcaptcha, extract128_be,
    goaway::GoAwayConfig,
    mcaptcha::PoWConfig,
    message::{CerberusMessage, DecimalMessage, GoAwayMessage},
    solver::{SOLVE_TYPE_GT, SOLVE_TYPE_LEADING_ZERO_BITS, SOLVE_TYPE_MASK, Solver, SolverDyn},
};
use serde::Deserialize;
use serde_json::{Value, json};

mod altcha;
mod capjs;
mod friendly;
mod sequentialread;
mod tor_equix;
/// CPU worker recommendation for independent Friendly Captcha solution banks.
pub use friendly::suggested_workers as friendlycaptcha_suggested_workers;
/// Verify the computational proof of a Friendly Captcha v1 puzzle.
pub use friendly::verify as verify_friendlycaptcha_v1;
/// Independently verify a sequentialread Scrypt `PoW` proof against its challenge.
pub use sequentialread::verify as verify_pow_bot_deterrent;
/// Verify a Tor onion-service v1 proof extension against its descriptor inputs.
pub use tor_equix::verify as verify_tor_hs_v1;
#[cfg(feature = "gpu")]
pub mod gpu;

const DEFAULT_LIMIT: u64 = 4_000_000_000;

#[derive(Debug)]
pub struct Result {
    pub answer: Value,
    pub solve_ms: f64,
    pub implementation: String,
}

#[derive(Debug, Deserialize)]
struct McaptchaRequest {
    config: PoWConfig,
    key: String,
}

#[derive(Debug, Deserialize)]
struct HaphashChallenge {
    prefix: String,
    difficulty: u8,
}

/// Solve one supported classic `PoW` protocol, returning protocol-native JSON.
///
/// # Errors
/// Rejects unsupported protocol/backend combinations or malformed input;
/// propagates nonce-search exhaustion and independent proof-check failures.
pub fn solve(
    kind: &str,
    challenge: Value,
    implementation: Option<&str>,
    worker_count: usize,
) -> anyhow::Result<Result> {
    ensure!(
        worker_count > 0,
        "classic solver worker count must be nonzero"
    );
    let implementation = Implementation::parse(implementation.unwrap_or("auto"))?;
    match kind {
        "cerberus" => solve_cerberus(challenge, implementation),
        "goaway" => solve_goaway(challenge, implementation),
        "capjs" => capjs::solve(challenge, implementation),
        "friendlycaptcha" => friendly::solve(challenge, implementation, worker_count),
        "tor-hs-v1" => tor_equix::solve(challenge, implementation, worker_count),
        "pow-bot-deterrent" => sequentialread::solve(&challenge, implementation, worker_count),
        "mcaptcha" => solve_mcaptcha(challenge, implementation),
        "haphash" => solve_haphash(challenge, implementation),
        "altcha" => altcha::solve(challenge, implementation, worker_count),
        "linear" => crate::linear_pow::solve(challenge, implementation),
        _ => bail!("unsupported classic solver: {kind}"),
    }
}

/// Estimated candidate-cost score for automatic WGPU routing, only for
/// protocol shapes the existing GPU implementation can actually execute.
///
/// This shares the protocol's real parser and GPU eligibility logic so the
/// service does not maintain a duplicate, incomplete list of supported cases.
#[cfg(feature = "gpu")]
#[must_use]
pub fn auto_gpu_expected_work(kind: &str, challenge: &Value) -> Option<u128> {
    match kind {
        "altcha" => altcha::auto_gpu_expected_work(challenge),
        "linear" => crate::linear_pow::auto_gpu_expected_work(challenge),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Implementation {
    Auto,
    Safe,
    Avx2,
    ShaNi,
    Avx512,
}

impl Implementation {
    pub(crate) fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "safe" => Ok(Self::Safe),
            "avx2" => Ok(Self::Avx2),
            "sha-ni" => Ok(Self::ShaNi),
            "avx512" => Ok(Self::Avx512),
            other => bail!("unsupported classic implementation: {other}"),
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Safe => "safe",
            Self::Avx2 => "avx2",
            Self::ShaNi => "sha-ni",
            Self::Avx512 => "avx512",
        }
    }
}

pub trait DynPow {
    fn set_limit(&mut self, limit: u64);
    fn attempted(&self) -> u64;
    fn solve(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])>;
}

impl<T> DynPow for T
where
    T: Solver<Output = [u32; 8]> + SolverDyn<Output = [u32; 8]>,
{
    fn set_limit(&mut self, limit: u64) {
        Solver::set_limit(self, limit);
    }
    fn attempted(&self) -> u64 {
        Solver::attempted_nonces(self)
    }
    fn solve(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, [u32; 8])> {
        SolverDyn::solve_dyn(self, target, ty, mask)
    }
}

fn require_x86_feature(implementation: Implementation) -> anyhow::Result<()> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    match implementation {
        Implementation::Avx2 if !std::arch::is_x86_feature_detected!("avx2") => {
            bail!("AVX2 unavailable on this CPU")
        }
        Implementation::ShaNi if !std::arch::is_x86_feature_detected!("sha") => {
            bail!("SHA-NI unavailable on this CPU")
        }
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 if !std::arch::is_x86_feature_detected!("avx512f") => {
            bail!("AVX-512 unavailable on this CPU")
        }
        _ => {}
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "requested x86 implementation on a non-x86 host"
    );
    Ok(())
}

fn decimal_solver(
    implementation: Implementation,
    message: DecimalMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(DecimalSolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::DecimalSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2DecimalSolver::from(message),
        )),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(
            challenge_solvers::solver::sha_ni::DecimalSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512DecimalSolver::from(message),
        )),
        #[cfg_attr(
            target_arch = "x86_64",
            expect(
                unreachable_patterns,
                reason = "all solver implementations in this match are available on x86_64"
            )
        )]
        _ => bail!("implementation unavailable on this target"),
    }
}

/// Construct an optimized binary SHA-256 CPU search backend.
///
/// # Errors
/// Rejects unsupported ISA selectors, including SHA-NI for binary framing.
pub fn binary_solver(
    implementation: Implementation,
    message: challenge_solvers::message::BinaryMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "binary SHA-256 has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(challenge_solvers::BinarySolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::BinarySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2BinarySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512BinarySolver::from(message),
        )),
        _ => bail!("implementation unavailable for binary SHA-256"),
    }
}

fn goaway_solver(
    implementation: Implementation,
    message: GoAwayMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(challenge_solvers::GoAwaySolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::GoAwaySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2GoAwaySolver::from(message),
        )),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Implementation::ShaNi => Ok(Box::new(
            challenge_solvers::solver::sha_ni::GoAwaySolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512GoAwaySolver::from(message),
        )),
        #[cfg_attr(
            target_arch = "x86_64",
            expect(
                unreachable_patterns,
                reason = "all solver implementations in this match are available on x86_64"
            )
        )]
        _ => bail!("implementation unavailable on this target"),
    }
}

fn cerberus_solver(
    implementation: Implementation,
    message: CerberusMessage,
) -> anyhow::Result<Box<dyn DynPow>> {
    ensure!(
        !matches!(implementation, Implementation::ShaNi),
        "Cerberus has no SHA-NI backend"
    );
    require_x86_feature(implementation)?;
    match implementation {
        Implementation::Auto => Ok(Box::new(challenge_solvers::CerberusSolver::from(message))),
        Implementation::Safe => Ok(Box::new(
            challenge_solvers::solver::safe::CerberusSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx2 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx2CerberusSolver::from(message),
        )),
        #[cfg(target_arch = "x86_64")]
        Implementation::Avx512 => Ok(Box::new(
            challenge_solvers::solver::x86::Avx512CerberusSolver::from(message),
        )),
        _ => bail!("implementation unavailable for Cerberus"),
    }
}

fn solve_cerberus(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let descriptor: CerberusChallenge =
        serde_json::from_value(challenge).context("invalid Cerberus challenge")?;
    ensure!(
        descriptor.difficulty().get() <= 16,
        "Cerberus difficulty above 16 dubits is unsupported by upstream clients"
    );
    let started = Instant::now();
    let mut solution = None;
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = descriptor.build_msg(bank) else {
            break;
        };
        let mut solver = cerberus_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = solver.solve(0, SOLVE_TYPE_MASK, descriptor.mask());
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let (solution, hash) = solution.context("Cerberus search exhausted configured limit")?;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let response = encode_words_le(hash);
    Ok(Result {
        answer: json!({
            "solution": solution.to_string(),
            "response": response,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation.name().to_owned(),
    })
}

fn finish_goaway(
    config: &GoAwayConfig,
    implementation: &str,
    solution: (u64, [u32; 8]),
    attempted_nonces: u64,
    started: Instant,
) -> Result {
    let (nonce, hash) = solution;
    let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
    let token = format!("{}{nonce:016x}", config.challenge());
    let id = hex::encode(
        hash.into_iter()
            .take(4)
            .flat_map(u32::to_be_bytes)
            .collect::<Vec<_>>(),
    );
    Result {
        answer: json!({
            "nonce": nonce.to_string(),
            "token": token,
            "id": id,
            "attempted_nonces": attempted_nonces,
        }),
        solve_ms,
        implementation: implementation.to_owned(),
    }
}

fn solve_goaway(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let config: GoAwayConfig =
        serde_json::from_value(challenge).context("invalid go-away challenge")?;
    ensure!(
        config.challenge().len() == 64,
        "go-away challenge must be exactly 64 characters"
    );
    ensure!(
        config.difficulty() <= 256,
        "go-away difficulty above the 256-bit SHA-256 digest width is unsatisfiable"
    );
    let accelerated_mask = u8::try_from(config.difficulty())
        .ok()
        .and_then(NonZeroU8::new)
        .filter(|difficulty| difficulty.get() <= 64)
        .map(compute_mask_goaway);
    let challenge_bytes: [u8; 64] = config
        .challenge()
        .as_bytes()
        .try_into()
        .context("go-away challenge must be exactly 64 bytes")?;
    let mut message =
        GoAwayMessage::new_hex(&challenge_bytes, 0).context("invalid go-away hex challenge")?;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    let mut solution = None;
    for high_word in 0_u32.. {
        message.set_high_word(high_word);
        let mut solver = goaway_solver(implementation, message.clone())?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        solution = if let Some(mask) = accelerated_mask {
            solver.solve(0, SOLVE_TYPE_MASK, mask)
        } else {
            solver.solve(config.difficulty(), SOLVE_TYPE_LEADING_ZERO_BITS, u64::MAX)
        };
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if solution.is_some() || attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    let solution = solution.context("go-away search exhausted configured limit")?;
    Ok(finish_goaway(
        &config,
        implementation.name(),
        solution,
        attempted_nonces,
        started,
    ))
}

fn solve_haphash(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let challenge: HaphashChallenge =
        serde_json::from_value(challenge).context("invalid haphash challenge")?;
    ensure!(
        challenge.difficulty <= 64,
        "haphash difficulty above the 64-nibble SHA-256 digest width is unsatisfiable"
    );
    let prefix = challenge.prefix.as_bytes();
    ensure!(!prefix.is_empty(), "haphash prefix must not be empty");
    let fast_mask = if challenge.difficulty == 0 {
        Some(0)
    } else {
        NonZeroU8::new(challenge.difficulty)
            .filter(|difficulty| difficulty.get() <= 16)
            .map(compute_mask_zero_nibbles)
    };
    let full_digest_bits = u64::from(challenge.difficulty) * 4;
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;
    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = if let Some(mask) = fast_mask {
            solver.solve(0, SOLVE_TYPE_MASK, mask)
        } else {
            solver.solve(full_digest_bits, SOLVE_TYPE_LEADING_ZERO_BITS, u64::MAX)
        };
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if let Some((nonce, _)) = result {
            return Ok(Result {
                answer: json!({
                    "nonce": nonce.to_string(),
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms: started.elapsed().as_secs_f64() * 1000.0,
                implementation: implementation.name().to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }
    Err(anyhow!("haphash search exhausted configured limit"))
}

fn solve_mcaptcha(challenge: Value, implementation: Implementation) -> anyhow::Result<Result> {
    let request: McaptchaRequest =
        serde_json::from_value(challenge).context("invalid mCaptcha request")?;
    let difficulty_factor = NonZeroU64::new(u64::from(request.config.difficulty_factor))
        .context("mCaptcha difficulty factor must be positive")?;
    let mut prefix = Vec::new();
    build_mcaptcha_prefix(&mut prefix, &request.config.string, &request.config.salt)
        .context("mCaptcha string length exceeds the wire format")?;
    let target = compute_target_mcaptcha(difficulty_factor);
    let started = Instant::now();
    let mut attempted_nonces = 0_u64;

    for bank in 0_u32.. {
        let Some(message) = DecimalMessage::new(&prefix, bank) else {
            break;
        };
        let mut solver = decimal_solver(implementation, message)?;
        solver.set_limit(DEFAULT_LIMIT.saturating_sub(attempted_nonces));
        let result = solver.solve(target, SOLVE_TYPE_GT, u64::MAX);
        attempted_nonces = attempted_nonces.saturating_add(solver.attempted());
        if let Some((nonce, hash)) = result {
            let solve_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Result {
                answer: json!({
                    "string": request.config.string,
                    "result": extract128_be(hash).to_string(),
                    "nonce": nonce,
                    "key": request.key,
                    "attempted_nonces": attempted_nonces,
                }),
                solve_ms,
                implementation: implementation.name().to_owned(),
            });
        }
        if attempted_nonces >= DEFAULT_LIMIT {
            break;
        }
    }

    Err(anyhow!("mCaptcha search exhausted configured limit"))
}

const JS_MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

fn encode_words_le(words: [u32; 8]) -> String {
    hex::encode(
        words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_error<T>(result: anyhow::Result<T>, message: &str) -> anyhow::Result<anyhow::Error> {
        result.err().ok_or_else(|| anyhow::anyhow!("{message}"))
    }

    #[test]
    fn goaway_response_id_is_canonical_sha256_digest_prefix() -> anyhow::Result<()> {
        use sha2::Digest as _;

        let challenge: GoAwayConfig = serde_json::from_value(json!({
            "challenge": "0123456789abcdef".repeat(4),
            "difficulty": 0,
        }))?;
        let nonce = 0x1020_3040_5060_7080_u64;
        let mut input = hex::decode(challenge.challenge())?;
        input.extend_from_slice(&nonce.to_be_bytes());
        let digest: [u8; 32] = sha2::Sha256::digest(&input).into();
        let hash = core::array::from_fn(|index| {
            u32::from_be_bytes([
                digest[index * 4],
                digest[index * 4 + 1],
                digest[index * 4 + 2],
                digest[index * 4 + 3],
            ])
        });

        let result = finish_goaway(&challenge, "safe", (nonce, hash), 1, Instant::now());
        assert_eq!(result.answer["id"], hex::encode(&digest[..16]));
        assert_eq!(
            result.answer["token"],
            format!("{}{nonce:016x}", challenge.challenge())
        );
        assert_eq!(result.answer["nonce"], nonce.to_string());
        Ok(())
    }

    #[test]
    fn haphash_solver_returns_nonce_for_protocol_prefix() -> anyhow::Result<()> {
        let result = solve_haphash(
            json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":1}),
            Implementation::Safe,
        )?;
        assert!(result.answer["nonce"].is_string());
        Ok(())
    }

    #[test]
    fn haphash_full_digest_predicate_covers_zero_through_64_nibbles() {
        let zero = [0_u32; 8];
        assert!(challenge_solvers::hash_has_leading_zero_bits(&zero, 0));
        assert!(challenge_solvers::hash_has_leading_zero_bits(&zero, 256));

        let mut digest = [0_u32; 8];
        digest[2] = 0x0fff_ffff;
        assert!(challenge_solvers::hash_has_leading_zero_bits(&digest, 68));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&digest, 69));

        digest.fill(0);
        digest[7] = 1;
        assert!(challenge_solvers::hash_has_leading_zero_bits(&digest, 255));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&digest, 256));
        assert!(!challenge_solvers::hash_has_leading_zero_bits(&zero, 257));
    }

    #[test]
    fn haphash_zero_difficulty_is_trivial_for_accelerated_solver() -> anyhow::Result<()> {
        let result = solve_haphash(
            json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":0}),
            Implementation::Auto,
        )?;
        assert!(result.answer["nonce"].is_string());
        assert_eq!(result.answer["attempted_nonces"], 1);
        Ok(())
    }

    #[test]
    fn haphash_rejects_unsatisfiable_difficulty_above_digest_width() -> anyhow::Result<()> {
        let error = require_error(
            solve_haphash(
                json!({"prefix":"127.0.0.1;example.test;1759954402;","difficulty":65}),
                Implementation::Safe,
            ),
            "difficulty above 64 nibbles must be rejected",
        )?;
        assert!(error.to_string().contains("64-nibble SHA-256 digest width"));
        Ok(())
    }

    #[test]
    fn altcha_reference_backend_is_not_exposed_by_production() -> anyhow::Result<()> {
        for implementation in ["reference", "altcha-rust"] {
            let error = require_error(
                solve("altcha", json!({"parameters": {}}), Some(implementation), 1),
                "external ALTCHA reference solver must not be selectable in production",
            )?;
            assert!(
                error
                    .to_string()
                    .contains("unsupported classic implementation")
            );
        }
        Ok(())
    }

    #[test]
    fn cerberus_and_goaway_reject_unrepresentable_difficulties() -> anyhow::Result<()> {
        let cerberus = require_error(
            solve_cerberus(
                json!({
                    "challenge": "domain-test",
                    "difficulty": 17,
                    "nonce": 1,
                    "ts": 1,
                    "signature": "domain-test",
                    "version": "0.4.9"
                }),
                Implementation::Safe,
            ),
            "current Cerberus clients only define up to 16 dubits",
        )?;
        assert!(cerberus.to_string().contains("above 16 dubits"));

        let goaway = require_error(
            solve_goaway(
                json!({
                    "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                    "difficulty": 257
                }),
                Implementation::Safe,
            ),
            "go-away support must reject unsatisfiable difficulty above 256 bits",
        )?;
        assert!(goaway.to_string().contains("256-bit SHA-256 digest width"));
        Ok(())
    }

    #[test]
    fn mcaptcha_rejects_zero_difficulty_factor() -> anyhow::Result<()> {
        let error = require_error(
            solve_mcaptcha(
                json!({
                    "config": {
                        "string": "schema-test",
                        "difficulty_factor": 0,
                        "salt": "schema-test-salt"
                    },
                    "key": "schema-test"
                }),
                Implementation::Auto,
            ),
            "zero mCaptcha difficulty must be rejected before target calculation",
        )?;
        assert!(error.to_string().contains("must be positive"));
        Ok(())
    }

    #[test]
    fn protocol_nonce_json_types_match_wire_contracts() -> anyhow::Result<()> {
        let mcaptcha = solve_mcaptcha(
            json!({
                "config": {
                    "string": "schema-test",
                    "difficulty_factor": 1,
                    "salt": "schema-test-salt"
                },
                "key": "schema-test"
            }),
            Implementation::Auto,
        )?;
        assert!(mcaptcha.answer["nonce"].is_u64());

        let goaway = solve_goaway(
            json!({
                "challenge": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "difficulty": 1
            }),
            Implementation::Auto,
        )?;
        assert!(goaway.answer["nonce"].is_string());
        Ok(())
    }
}
