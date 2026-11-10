//! Tor onion-service introduction `PoW` scheme v1 (Equi-X + BLAKE2b-32).
//!
//! The caller supplies the descriptor's blinded identity and seed. The output
//! is the exact 41-byte body of the encrypted INTRODUCE1 extension type 2.
//! Descriptor freshness and the onion service's replay cache remain external.

use std::{
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Context, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use blake2::{Blake2b, Digest, digest::consts::U4};
use equix::{EquiX, Error as EquixError, HashError, SolverMemory};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Implementation, Result};

const PERSONALIZATION: &[u8; 16] = b"Tor hs intro v1\0";
const CHALLENGE_LEN: usize = 16 + 32 + 32 + 16 + 4;
const EXTENSION_LEN: usize = 1 + 16 + 4 + 4 + 16;
const SCHEME: u8 = 1;
/// The enclosing encrypted INTRODUCE1 extension type.
pub const EXTENSION_TYPE: u8 = 2;

#[derive(Deserialize)]
struct Request {
    /// The 32-byte blinded public service ID, hex encoded.
    blinded_id: String,
    /// The descriptor's full 32-byte seed, base64 encoded.
    seed_b64: String,
    /// Client-selected effort committed into the challenge before searching.
    effort: u32,
    /// Optional 16-byte little-endian starting nonce, hex encoded.
    /// When absent a cryptographically random nonce is generated.
    nonce: Option<String>,
}

struct Parameters {
    blinded_id: [u8; 32],
    seed: [u8; 32],
    effort: u32,
}

impl Parameters {
    fn new(blinded_id: &str, seed_b64: &str, effort: u32) -> anyhow::Result<Self> {
        let decoded_id = hex::decode(blinded_id).context("invalid blinded_id hex")?;
        let blinded_id = decoded_id
            .try_into()
            .map_err(|_| anyhow::anyhow!("blinded_id must decode to exactly 32 bytes"))?;
        let decoded_seed = STANDARD.decode(seed_b64).context("invalid seed_b64")?;
        let seed = decoded_seed
            .try_into()
            .map_err(|_| anyhow::anyhow!("seed_b64 must decode to exactly 32 bytes"))?;
        Ok(Self {
            blinded_id,
            seed,
            effort,
        })
    }

    fn challenge(&self, nonce: [u8; 16]) -> [u8; CHALLENGE_LEN] {
        let mut challenge = [0_u8; CHALLENGE_LEN];
        challenge[..16].copy_from_slice(PERSONALIZATION);
        challenge[16..48].copy_from_slice(&self.blinded_id);
        challenge[48..80].copy_from_slice(&self.seed);
        challenge[80..96].copy_from_slice(&nonce);
        challenge[96..100].copy_from_slice(&self.effort.to_be_bytes());
        challenge
    }
}

#[derive(Clone, Copy)]
struct Proof {
    nonce: [u8; 16],
    solution: [u8; 16],
}

impl Proof {
    fn extension_body(self, params: &Parameters) -> [u8; EXTENSION_LEN] {
        let mut extension = [0_u8; EXTENSION_LEN];
        extension[0] = SCHEME;
        extension[1..17].copy_from_slice(&self.nonce);
        extension[17..21].copy_from_slice(&params.effort.to_be_bytes());
        extension[21..25].copy_from_slice(&params.seed[..4]);
        extension[25..41].copy_from_slice(&self.solution);
        extension
    }
}

// BLAKE2b with digest size 4 is NOT truncated BLAKE2b-512.
// Tor interprets those four bytes as a network-order u32.
fn effort_valid(challenge: &[u8; CHALLENGE_LEN], solution: &[u8; 16], effort: u32) -> bool {
    let mut hasher = Blake2b::<U4>::new();
    hasher.update(challenge);
    hasher.update(solution);
    let digest = hasher.finalize();
    let r = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    r.checked_mul(effort).is_some()
}

/// Verify the computational validity of an exact 41-byte Tor v1 proof body.
///
/// The caller must independently check the descriptor's expiry, authenticate
/// its identity and seed, and prevent (seed, nonce) replay.
///
/// # Errors
/// Invalid blinded service identity or seed encoding is a caller error.
pub fn verify(blinded_id: &str, seed_b64: &str, extension_body: &[u8]) -> anyhow::Result<bool> {
    let params = Parameters::new(blinded_id, seed_b64, 0)?;
    if extension_body.len() != EXTENSION_LEN || extension_body[0] != SCHEME {
        return Ok(false);
    }
    if extension_body[21..25] != params.seed[..4] {
        return Ok(false);
    }
    let effort = u32::from_be_bytes(extension_body[17..21].try_into()?);
    let nonce: [u8; 16] = extension_body[1..17].try_into()?;
    let solution: [u8; 16] = extension_body[25..41].try_into()?;
    let params = Parameters { effort, ..params };
    let challenge = params.challenge(nonce);
    if !effort_valid(&challenge, &solution, effort) {
        return Ok(false);
    }
    Ok(equix::verify_bytes(&challenge, &solution).is_ok())
}

fn search_worker(
    params: &Parameters,
    start_nonce: u128,
    index: usize,
    stride: usize,
    stop: &AtomicBool,
    found: &OnceLock<Proof>,
) -> anyhow::Result<u64> {
    let Some(mut nonce) = start_nonce.checked_add(u128::try_from(index)?) else {
        return Ok(0);
    };
    let stride = u128::try_from(stride)?;
    let mut memory = SolverMemory::new();
    let mut attempts = 0_u64;
    while !stop.load(Ordering::Relaxed) {
        attempts = attempts
            .checked_add(1)
            .context("Tor nonce attempt count overflow")?;
        let bytes = nonce.to_le_bytes();
        let challenge = params.challenge(bytes);
        match EquiX::new(&challenge) {
            Ok(equix) => {
                for solution in equix.solve_with_memory(&mut memory) {
                    let proof = solution.to_bytes();
                    if effort_valid(&challenge, &proof, params.effort) {
                        if found
                            .set(Proof {
                                nonce: bytes,
                                solution: proof,
                            })
                            .is_ok()
                        {
                            stop.store(true, Ordering::Release);
                        }
                        return Ok(attempts);
                    }
                }
            }
            // Some nonce-dependent HashX programs are intentionally invalid.
            Err(EquixError::Hash(HashError::ProgramConstraints)) => {}
            Err(err) => {
                stop.store(true, Ordering::Release);
                return Err(err.into());
            }
        }
        let Some(next) = nonce.checked_add(stride) else {
            break;
        };
        nonce = next;
    }
    Ok(attempts)
}

pub(super) fn solve(
    challenge: Value,
    implementation: Implementation,
    worker_count: usize,
) -> anyhow::Result<Result> {
    ensure!(
        matches!(implementation, Implementation::Auto | Implementation::Safe),
        "Tor Equi-X supports only auto/safe CPU selectors"
    );
    ensure!(
        worker_count > 0,
        "Tor Equi-X requires at least one CPU worker"
    );
    let request: Request =
        serde_json::from_value(challenge).context("invalid Tor v1 Equi-X challenge")?;
    let params = Parameters::new(&request.blinded_id, &request.seed_b64, request.effort)?;
    let nonce: [u8; 16] = if let Some(encoded) = request.nonce {
        hex::decode(encoded)
            .context("invalid Tor starting nonce hex")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("Tor starting nonce must be 16 bytes"))?
    } else {
        let mut nonce = [0_u8; 16];
        getrandom::fill(&mut nonce)
            .map_err(|err| anyhow::anyhow!("cannot generate Tor nonce: {err}"))?;
        nonce
    };

    let started = Instant::now();
    let start_nonce = u128::from_le_bytes(nonce);
    let found = OnceLock::new();
    let stop = AtomicBool::new(false);
    let attempts = thread::scope(|scope| -> anyhow::Result<u64> {
        let handles: Vec<_> = (0..worker_count)
            .map(|index| {
                let params = &params;
                let found = &found;
                let stop = &stop;
                scope.spawn(move || {
                    search_worker(params, start_nonce, index, worker_count, stop, found)
                })
            })
            .collect();
        let mut total = 0_u64;
        let mut first_error = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(count)) => {
                    total = total
                        .checked_add(count)
                        .context("Tor total work counter overflow")?;
                }
                Ok(Err(err)) if first_error.is_none() => first_error = Some(err),
                Err(_) if first_error.is_none() => {
                    stop.store(true, Ordering::Release);
                    first_error = Some(anyhow::anyhow!("Tor Equi-X CPU worker panicked"));
                }
                Ok(Err(_)) | Err(_) => {}
            }
        }
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(total)
    })?;
    let proof = found.get().context("Tor Equi-X nonce space exhausted")?;
    let extension = proof.extension_body(&params);
    Ok(Result {
        answer: json!({
            "scheme": SCHEME,
            "extension_type": EXTENSION_TYPE,
            "extension_body": hex::encode(extension),
            "nonce": hex::encode(proof.nonce),
            "effort": params.effort,
            "seed_prefix": hex::encode(&params.seed[..4]),
            "solution": hex::encode(proof.solution),
            "attempted_nonces": attempts,
        }),
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
        implementation: "equix-hashx".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    fn sample_params() -> (String, String) {
        let blinded_id = hex::encode(core::array::from_fn::<u8, 32, _>(|i| {
            u8::try_from(i).unwrap_or(0)
        }));
        let seed = STANDARD.encode([0x5a_u8; 32]);
        (blinded_id, seed)
    }

    #[test]
    fn challenge_framing_and_wire_endianness() -> anyhow::Result<()> {
        let (id, seed) = sample_params();
        let params = Parameters::new(&id, &seed, 0x1234_5678)?;
        let nonce = 0x0112_2334_4556_6778_899a_abbc_cdde_eff0_u128.to_le_bytes();
        let challenge = params.challenge(nonce);
        assert_eq!(&challenge[..16], b"Tor hs intro v1\0");
        assert_eq!(&challenge[16..48], &(0_u8..32).collect::<Vec<_>>());
        assert_eq!(&challenge[48..80], &[0x5a_u8; 32]);
        assert_eq!(&challenge[80..96], &nonce);
        assert_eq!(&challenge[96..100], &[0x12, 0x34, 0x56, 0x78]);
        let extension = Proof {
            nonce,
            solution: [0x77; 16],
        }
        .extension_body(&params);
        assert_eq!(extension.len(), 41);
        assert_eq!(&extension[..1], &[1]);
        assert_eq!(&extension[17..21], &[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(&extension[21..25], &[0x5a; 4]);
        Ok(())
    }

    #[test]
    fn native_blake2b_32_is_not_truncated_blake2b_512() -> anyhow::Result<()> {
        let (id, seed) = sample_params();
        let params = Parameters::new(&id, &seed, 1)?;
        let input = params.challenge([0; 16]);
        let solution = [0_u8; 16];
        let mut h = Blake2b::<U4>::new();
        h.update(input);
        h.update(solution);
        let short = h.finalize();
        let long = blake2::Blake2b512::digest([input.as_slice(), solution.as_slice()].concat());
        assert_ne!(&short[..], &long[..4]);
        Ok(())
    }

    #[test]
    fn generated_proof_verifies_and_tampering_fails() -> anyhow::Result<()> {
        let (id, seed) = sample_params();
        let request = json!({"blinded_id":id, "seed_b64":seed,"effort": 2, "nonce": "00000000000000000000000000000000"});
        let solved = solve(request, Implementation::Auto, 2)?;
        let decoded = hex::decode(solved.answer["extension_body"].as_str().context("body")?)?;
        assert_eq!(decoded.len(), 41);
        assert!(verify(&id, &seed, &decoded)?);
        assert_eq!(solved.answer["seed_prefix"], "5a5a5a5a");
        let mut damaged = decoded.clone();
        damaged[25] ^= 0x40;
        assert!(!verify(&id, &seed, &damaged)?);
        let mut damaged = decoded.clone();
        damaged[0] = 2;
        assert!(!verify(&id, &seed, &damaged)?);
        let mut damaged = decoded.clone();
        damaged[21] ^= 1;
        assert!(!verify(&id, &seed, &damaged)?);
        assert!(!verify(&id, &seed, &decoded[..40])?);
        Ok(())
    }

    #[test]
    fn tor_c_reference_vector_high_effort_is_accepted() -> anyhow::Result<()> {
        // Independently captured from C Tor and published by the Arti
        // tor-hscrypto/tests/pow_v1_vectors.rs compatibility suite.
        let id = "11".repeat(32);
        let seed = STANDARD.encode([0xaa_u8; 32]);
        let nonce = hex::decode("59217255555555555555555555555555")?;
        let solution = hex::decode("0f3db97b9cac20c1771680a1a34848d3")?;
        let mut proof = Vec::with_capacity(EXTENSION_LEN);
        proof.push(1);
        proof.extend_from_slice(&nonce);
        proof.extend_from_slice(&1_000_000_u32.to_be_bytes());
        proof.extend_from_slice(&[0xaa; 4]);
        proof.extend_from_slice(&solution);
        assert!(verify(&id, &seed, &proof)?);
        proof[17..21].copy_from_slice(&100_000_u32.to_be_bytes());
        assert!(!verify(&id, &seed, &proof)?);
        proof[17..21].copy_from_slice(&1_000_000_u32.to_be_bytes());
        proof[25] ^= 0x01;
        assert!(!verify(&id, &seed, &proof)?);
        Ok(())
    }

    #[test]
    fn tor_c_reference_solve_vector_matches_nonce_and_proof() -> anyhow::Result<()> {
        // C Tor test vector: effort 100, seed=AA*32, blind ID=11*32,
        // starting nonce 55*16; expected first solution at 56 55*15.
        let id = "11".repeat(32);
        let seed = STANDARD.encode([0xaa_u8; 32]);
        let request = json!({
            "blinded_id": id,
            "seed_b64": seed,
            "effort": 100,
            "nonce": "55555555555555555555555555555555",
        });
        let solved = solve(request, Implementation::Safe, 1)?;
        assert_eq!(solved.answer["nonce"], "56555555555555555555555555555555");
        assert_eq!(
            solved.answer["solution"],
            "3a4122a240bd7abfc922ab3cbb9479ed"
        );
        let extension = hex::decode(
            solved.answer["extension_body"]
                .as_str()
                .context("extension")?,
        )?;
        assert!(verify(&id, &seed, &extension)?);
        Ok(())
    }

    #[test]
    fn invalid_inputs_rejected() {
        let (id, seed) = sample_params();
        let request = json!({"blinded_id":id,"seed_b64":seed,"effort":1,"nonce":"nothex"});
        assert!(solve(request, Implementation::Auto, 1).is_err());
        assert!(Parameters::new("abcd", &seed, 1).is_err());
        assert!(Parameters::new(&id, "bad", 1).is_err());
    }
}
