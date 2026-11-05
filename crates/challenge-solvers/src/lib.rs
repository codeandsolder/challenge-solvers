#![cfg_attr(feature = "portable-simd", feature(portable_simd))]
#![cfg_attr(feature = "gpu", recursion_limit = "256")]
#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![doc = "Feature-configurable proof-of-work solver core."]
#![deny(missing_docs)]
#![deny(unsafe_code)]

use core::num::{NonZeroU8, NonZeroU64};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "gpu")]
/// GPU solver backend using native `wgpu` or browser WebGPU.
pub mod gpu;

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "wasm32"))]
cfg_if::cfg_if! {
    if #[cfg(feature = "internals")] {
        /// String manipulation functions
        pub mod strings;
    } else {
        mod strings;
    }
}

/// SHA-256 primitives
mod sha256;

/// BLAKE3 primitives
mod blake3;

/// Message builders
pub mod message;

/// Solvers
pub mod solver;

#[cfg(feature = "altcha")]
mod protocol;

#[cfg(feature = "altcha")]
#[path = "challenge/altcha.rs"]
/// ALTCHA protocol types and end-to-end solving helpers.
pub mod altcha;
#[cfg(feature = "anubis")]
#[path = "challenge/anubis.rs"]
/// Anubis protocol types and end-to-end solving helpers.
pub mod anubis;
#[cfg(feature = "capjs")]
#[path = "challenge/capjs.rs"]
/// Cap.js protocol types and end-to-end solving helpers.
pub mod capjs;
#[cfg(feature = "cerberus")]
#[path = "challenge/cerberus.rs"]
/// Cerberus protocol types and end-to-end solving helpers.
pub mod cerberus;
#[cfg(feature = "goaway")]
#[path = "challenge/goaway.rs"]
/// `GoAway` protocol types and end-to-end solving helpers.
pub mod goaway;
#[cfg(feature = "mcaptcha")]
#[path = "challenge/mcaptcha.rs"]
/// mCaptcha protocol types and end-to-end solving helpers.
pub mod mcaptcha;

#[cold]
const fn unlikely() {}

pub(crate) const fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

pub(crate) const fn high_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[expect(
    clippy::inline_always,
    reason = "byte extraction is an inner-loop primitive in architecture-specific solvers"
)]
#[inline(always)]
pub(crate) const fn read_be_word_byte(words: &[u32; 16], byte_index: usize) -> u8 {
    words[byte_index / 4].to_be_bytes()[byte_index % 4]
}

#[cfg(feature = "portable-simd")]
#[expect(
    clippy::inline_always,
    reason = "byte insertion is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) fn write_be_word_byte(words: &mut [u32; 16], byte_index: usize, byte: u8) {
    let shift = (3 - byte_index % 4) * 8;
    let mask = 0xffu32 << shift;
    let word = &mut words[byte_index / 4];
    *word = (*word & !mask) | (u32::from(byte) << shift);
}

#[expect(
    clippy::inline_always,
    reason = "byte extraction is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) const fn read_le_word_byte(words: &[u32; 16], byte_index: usize) -> u8 {
    words[byte_index / 4].to_le_bytes()[byte_index % 4]
}

#[expect(
    clippy::inline_always,
    reason = "byte insertion is an inner-loop primitive in solver kernels"
)]
#[inline(always)]
pub(crate) fn write_le_word_byte(words: &mut [u32; 16], byte_index: usize, byte: u8) {
    let shift = (byte_index % 4) * 8;
    let mask = 0xffu32 << shift;
    let word = &mut words[byte_index / 4];
    *word = (*word & !mask) | (u32::from(byte) << shift);
}

#[cfg(not(feature = "portable-simd"))]
/// Scalar SHA-256 solver for ALTCHA-compatible challenges.
pub type AltchaSha256Solver = crate::solver::safe::AltchaSha256Solver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar single-block SHA-256 solver.
pub type SingleBlockSolver = crate::solver::safe::SingleBlockSolver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar double-block SHA-256 solver.
pub type DoubleBlockSolver = crate::solver::safe::DoubleBlockSolver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar decimal-nonce SHA-256 solver.
pub type DecimalSolver = crate::solver::safe::DecimalSolver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar binary-nonce SHA-256 solver.
pub type BinarySolver = crate::solver::safe::BinarySolver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar `GoAway` SHA-256 solver.
pub type GoAwaySolver = crate::solver::safe::GoAwaySolver;
#[cfg(not(feature = "portable-simd"))]
/// Scalar Cerberus solver.
pub type CerberusSolver = crate::solver::safe::CerberusSolver;

#[cfg(feature = "portable-simd")]
cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        /// ALTCHA SHA-256 solver selected at runtime for this CPU.
        pub type AltchaSha256Solver = crate::solver::x86::AltchaSha256Solver;
        /// Single-block SHA-256 solver selected at runtime for this CPU.
        pub type SingleBlockSolver = crate::solver::x86::SingleBlockSolver;
        /// Double-block SHA-256 solver selected at runtime for this CPU.
        pub type DoubleBlockSolver = crate::solver::x86::DoubleBlockSolver;
        /// Decimal-nonce SHA-256 solver selected at runtime for this CPU.
        pub type DecimalSolver = crate::solver::x86::DecimalSolver;
        /// Binary-nonce SHA-256 solver selected at runtime for this CPU.
        pub type BinarySolver = crate::solver::x86::BinarySolver;
        /// `GoAway` solver selected at runtime for this CPU.
        pub type GoAwaySolver = crate::solver::x86::GoAwaySolver;
        /// Cerberus solver selected at runtime for this CPU.
        pub type CerberusSolver = crate::solver::x86::CerberusSolver;
    } else if #[cfg(target_arch = "x86")] {
        /// ALTCHA SHA-256 scalar fallback on 32-bit x86.
        pub type AltchaSha256Solver = crate::solver::safe::AltchaSha256Solver;
        /// Single-block solver using SHA-NI when available, otherwise scalar.
        pub type SingleBlockSolver = crate::solver::SolverRouter<
            crate::solver::sha_ni::RequiredFeatures,
            crate::message::SingleBlockMessage,
            crate::solver::sha_ni::SingleBlockSolver,
            crate::solver::safe::SingleBlockSolver,
        >;
        /// Double-block solver using SHA-NI when available, otherwise scalar.
        pub type DoubleBlockSolver = crate::solver::SolverRouter<
            crate::solver::sha_ni::RequiredFeatures,
            crate::message::DoubleBlockMessage,
            crate::solver::sha_ni::DoubleBlockSolver,
            crate::solver::safe::DoubleBlockSolver,
        >;
        /// Decimal solver using SHA-NI when available, otherwise scalar.
        pub type DecimalSolver = crate::solver::SolverRouter<
            crate::solver::sha_ni::RequiredFeatures,
            crate::message::DecimalMessage,
            crate::solver::sha_ni::DecimalSolver,
            crate::solver::safe::DecimalSolver,
        >;
        /// Binary solver.
        pub type BinarySolver = crate::solver::safe::BinarySolver;
        /// `GoAway` solver using SHA-NI when available, otherwise scalar.
        pub type GoAwaySolver = crate::solver::SolverRouter<
            crate::solver::sha_ni::RequiredFeatures,
            crate::message::GoAwayMessage,
            crate::solver::sha_ni::GoAwaySolver,
            crate::solver::safe::GoAwaySolver,
        >;
        /// Cerberus solver.
        pub type CerberusSolver = crate::solver::safe::CerberusSolver;
    } else if #[cfg(target_arch = "wasm32")] {
        /// ALTCHA SHA-256 solver.
        pub type AltchaSha256Solver = crate::solver::safe::AltchaSha256Solver;
        /// Single-block SIMD128 solver.
        pub type SingleBlockSolver = crate::solver::simd128::SingleBlockSolver;
        /// Double-block SIMD128 solver.
        pub type DoubleBlockSolver = crate::solver::simd128::DoubleBlockSolver;
        /// Decimal SIMD128 solver.
        pub type DecimalSolver = crate::solver::simd128::DecimalSolver;
        /// Binary scalar solver.
        pub type BinarySolver = crate::solver::safe::BinarySolver;
        /// `GoAway` SIMD128 solver.
        pub type GoAwaySolver = crate::solver::simd128::GoAwaySolver;
        /// Cerberus SIMD128 solver.
        pub type CerberusSolver = crate::solver::simd128::CerberusSolver;
    } else {
        /// ALTCHA SHA-256 scalar solver.
        pub type AltchaSha256Solver = crate::solver::safe::AltchaSha256Solver;
        /// Single-block scalar solver.
        pub type SingleBlockSolver = crate::solver::safe::SingleBlockSolver;
        /// Double-block scalar solver.
        pub type DoubleBlockSolver = crate::solver::safe::DoubleBlockSolver;
        /// Decimal scalar solver.
        pub type DecimalSolver = crate::solver::safe::DecimalSolver;
        /// Binary scalar solver.
        pub type BinarySolver = crate::solver::safe::BinarySolver;
        /// `GoAway` scalar solver.
        pub type GoAwaySolver = crate::solver::safe::GoAwaySolver;
        /// Cerberus scalar solver.
        pub type CerberusSolver = crate::solver::safe::CerberusSolver;
    }
}

/// Build a prefix for mCaptcha `PoW`
pub fn build_mcaptcha_prefix<E: Extend<u8>>(out: &mut E, string: &str, salt: &str) -> Option<()> {
    let string_len = u64::try_from(string.len()).ok()?;
    out.extend(salt.as_bytes().iter().copied());
    out.extend(string_len.to_le_bytes());
    out.extend(string.as_bytes().iter().copied());
    Some(())
}

/// Compute a plausible time for a SHA-256 `PoW` to prevent tainting metrics
///
/// # Arguments
///
/// * `hashes`: The number of hashes that have been computed
///
/// # Returns
///
/// The plausible time in milliseconds
#[expect(
    dead_code,
    reason = "kept as the SHA-256 counterpart to the active plausibility estimators"
)]
pub(crate) const fn compute_plausible_time_sha256(hashes: u64) -> u64 {
    hashes / 512
}

/// Compute the target for an mCaptcha `PoW`
#[must_use]
pub const fn compute_target_mcaptcha(difficulty_factor: NonZeroU64) -> u64 {
    let difficulty_factor = difficulty_factor.get();
    u64::MAX - u64::MAX / difficulty_factor
}

/// Compute a leading-zero-nibble SHA-256 mask (mask & (V\[0] << 32 | V\[1]) == 0)
#[must_use]
pub const fn compute_mask_zero_nibbles(difficulty_factor: NonZeroU8) -> u64 {
    let bits = difficulty_factor.get().saturating_mul(4);
    if bits >= 64 {
        u64::MAX
    } else {
        !(!0u64 >> bits)
    }
}

/// Error returned when a decimal SHA-256 leading-zero request is outside the optimized domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimalSha256Error {
    /// The optimized leading-zero-nibble solver supports at most 16 nibbles.
    DifficultyAbove16(u8),
}

impl core::fmt::Display for DecimalSha256Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DifficultyAbove16(difficulty) => write!(
                formatter,
                "decimal SHA-256 difficulty {difficulty} exceeds the supported 16 leading nibbles"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecimalSha256Error {}

/// One decimal-nonce SHA-256 solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalSha256Solution {
    /// Decimal nonce appended to the prefix.
    pub nonce: u64,
    /// SHA-256 digest as eight big-endian words, matching the solver representation.
    pub digest: [u32; 8],
}

/// Bounded search result and exact amount of work performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalSha256Search {
    /// First solution found within the search budget, if any.
    pub solution: Option<DecimalSha256Solution>,
    /// Exact number of candidate nonces hashed.
    pub attempted: u64,
}

/// Solve `SHA-256(prefix || ASCII_DECIMAL(nonce))` for leading zero nibbles.
///
/// `limit` bounds the number of candidate nonces attempted across all search banks.
/// The returned counter is the exact number of candidates attempted.
///
/// # Errors
///
/// Returns [`DecimalSha256Error::DifficultyAbove16`] when `difficulty` exceeds the
/// optimized solver's 64-bit comparison domain.
pub fn solve_decimal_sha256_zero_nibbles(
    prefix: &[u8],
    difficulty: u8,
    limit: u64,
) -> Result<DecimalSha256Search, DecimalSha256Error> {
    use crate::{
        message::DecimalMessage,
        solver::{SOLVE_TYPE_MASK, Solver},
    };
    use sha2::Digest as _;

    if difficulty > 16 {
        return Err(DecimalSha256Error::DifficultyAbove16(difficulty));
    }
    let Some(difficulty) = NonZeroU8::new(difficulty) else {
        if limit == 0 {
            return Ok(DecimalSha256Search {
                solution: None,
                attempted: 0,
            });
        }
        let mut hasher = sha2::Sha256::new();
        hasher.update(prefix);
        hasher.update(b"0");
        let digest: [u8; 32] = hasher.finalize().into();
        let chunks = digest.as_chunks::<4>().0;
        let digest_words = core::array::from_fn(|index| u32::from_be_bytes(chunks[index]));
        return Ok(DecimalSha256Search {
            solution: Some(DecimalSha256Solution {
                nonce: 0,
                digest: digest_words,
            }),
            attempted: 1,
        });
    };
    let mask = compute_mask_zero_nibbles(difficulty);
    let mut result = None;
    let mut attempted_nonces = 0;
    let mut remaining_limit = limit;

    for search_bank in 0.. {
        let Some(message) = DecimalMessage::new(prefix, search_bank) else {
            break;
        };
        let mut solver = DecimalSolver::from(message);
        solver.set_limit(remaining_limit);
        result = solver.solve::<{ SOLVE_TYPE_MASK }>(0, mask);
        let attempted = solver.attempted_nonces();
        attempted_nonces += attempted;
        remaining_limit = remaining_limit.saturating_sub(attempted);
        if result.is_some() || remaining_limit == 0 {
            break;
        }
    }

    Ok(DecimalSha256Search {
        solution: result.map(|(nonce, digest)| DecimalSha256Solution { nonce, digest }),
        attempted: attempted_nonces,
    })
}

/// Test the full SHA-256 digest against go-away's leading-zero-bit rule.
///
/// Difficulties above 256 are unsatisfiable because SHA-256 has only 256 bits.
#[must_use]
pub fn goaway_hash_matches_difficulty(hash: &[u32; 8], difficulty: u16) -> bool {
    hash_has_leading_zero_bits(hash, u64::from(difficulty))
}

/// Test whether a SHA-256 digest has at least `bits` leading zero bits.
///
/// Values above the 256-bit digest width are unsatisfiable. This is shared by
/// protocols such as go-away and haphash whose verifier difficulty spans the
/// complete digest rather than only the first comparison word.
#[must_use]
pub fn hash_has_leading_zero_bits(hash: &[u32; 8], bits: u64) -> bool {
    if bits > 256 {
        return false;
    }

    let full_words = usize::try_from(bits / 32).unwrap_or(8);
    if hash[..full_words].iter().any(|word| *word != 0) {
        return false;
    }

    let remaining = u32::try_from(bits % 32).unwrap_or(31);
    remaining == 0 || hash[full_words].leading_zeros() >= remaining
}

/// Compute the mask for a `GoAway` `PoW` against the first 64 SHA-256 bits.
#[must_use]
pub const fn compute_mask_goaway(difficulty_factor: NonZeroU8) -> u64 {
    let bits = difficulty_factor.get();
    if bits >= 64 {
        u64::MAX
    } else {
        !(!0u64 >> bits)
    }
}

/// Compute a mask for a Cerberus `PoW` (mask & (V\[0] << 32) == 0)
#[must_use]
pub const fn compute_mask_cerberus(difficulty_factor: NonZeroU8) -> u64 {
    let doubled = difficulty_factor.get().saturating_mul(2);
    let bits = if doubled > 32 { 32 } else { doubled };
    let mask = if bits == 32 {
        u32::MAX
    } else {
        !(!0u32 >> bits)
    };
    let [b0, b1, b2, b3] = mask.swap_bytes().to_be_bytes();
    u64::from_be_bytes([b0, b1, b2, b3, 0, 0, 0, 0])
}

/// Extract top 128 bits from a 64-bit word array
#[must_use]
pub const fn extract128_be(inp: [u32; 8]) -> u128 {
    (inp[0] as u128) << 96 | (inp[1] as u128) << 64 | (inp[2] as u128) << 32 | (inp[3] as u128)
}

/// Encode a sha-256 hash into hex
pub fn encode_hex(out: &mut [u8; 64], inp: [u32; 8]) {
    for w in 0..8 {
        let be_bytes = inp[w].to_be_bytes();
        be_bytes.iter().enumerate().for_each(|(i, b)| {
            let high_nibble = b >> 4;
            let low_nibble = b & 0xf;
            out[w * 8 + i * 2] = if high_nibble < 10 {
                high_nibble + b'0'
            } else {
                high_nibble + b'a' - 10
            };
            out[w * 8 + i * 2 + 1] = if low_nibble < 10 {
                low_nibble + b'0'
            } else {
                low_nibble + b'a' - 10
            };
        });
    }
}

/// Encode a blake3 hash into hex
pub fn encode_hex_le(out: &mut [u8; 64], inp: [u32; 8]) {
    for w in 0..8 {
        let be_bytes = inp[w].to_le_bytes();
        be_bytes.iter().enumerate().for_each(|(i, b)| {
            let high_nibble = b >> 4;
            let low_nibble = b & 0xf;
            out[w * 8 + i * 2] = if high_nibble < 10 {
                high_nibble + b'0'
            } else {
                high_nibble + b'a' - 10
            };
            out[w * 8 + i * 2 + 1] = if low_nibble < 10 {
                low_nibble + b'0'
            } else {
                low_nibble + b'a' - 10
            };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nonzero(value: u8) -> anyhow::Result<NonZeroU8> {
        NonZeroU8::new(value).ok_or_else(|| anyhow::anyhow!("expected nonzero test value"))
    }

    pub fn build_prefix_official<W: std::io::Write>(
        out: &mut W,
        string: &str,
        salt: &str,
    ) -> std::io::Result<()> {
        out.write_all(salt.as_bytes())?;
        if let Err(error) = bincode::serialize_into(out, string) {
            return match *error {
                bincode::ErrorKind::Io(error) => Err(error),
                other => Err(std::io::Error::other(other.to_string())),
            };
        }
        Ok(())
    }

    #[test]
    fn test_encode_hex() {
        let mut out = [0u8; 64];
        encode_hex(
            &mut out,
            [
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
                0x1234_5678,
                0x9abc_def0,
            ],
        );
        assert_eq!(
            &out,
            b"123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0"
        );
    }

    #[test]
    fn test_compute_mask_zero_nibbles() -> anyhow::Result<()> {
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(1)?),
            0xfu64.reverse_bits(),
        );
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(2)?),
            0xffu64.reverse_bits(),
        );
        assert_eq!(
            compute_mask_zero_nibbles(nonzero(3)?),
            0xfffu64.reverse_bits(),
        );
        assert_eq!(compute_mask_zero_nibbles(nonzero(16)?), u64::MAX,);
        Ok(())
    }

    #[test]
    fn decimal_sha256_zero_nibbles_is_bounded_and_verifiable() -> anyhow::Result<()> {
        use sha2::Digest;
        use std::io::Write as _;

        let prefix = b"stable-scalar-boundary";
        let search = solve_decimal_sha256_zero_nibbles(prefix, 1, 4096)?;
        let solution = search.solution.ok_or_else(|| {
            anyhow::anyhow!("difficulty-1 solve unexpectedly exhausted its limit")
        })?;
        let nonce = solution.nonce;
        let digest_words = solution.digest;
        let attempted = search.attempted;
        assert!(attempted <= 4096);

        let mut message = prefix.to_vec();
        write!(message, "{nonce}")?;
        let expected: [u8; 32] = sha2::Sha256::digest(&message).into();
        let mut actual = [0_u8; 32];
        for (chunk, word) in actual.as_chunks_mut::<4>().0.iter_mut().zip(digest_words) {
            *chunk = word.to_be_bytes();
        }
        assert_eq!(actual, expected);
        assert_eq!(actual[0] >> 4, 0);

        let zero = solve_decimal_sha256_zero_nibbles(prefix, 0, 1)?;
        let zero_solution = zero
            .solution
            .ok_or_else(|| anyhow::anyhow!("difficulty-0 solve unexpectedly failed"))?;
        assert_eq!(zero_solution.nonce, 0);
        assert_eq!(zero.attempted, 1);

        let empty_budget = solve_decimal_sha256_zero_nibbles(prefix, 16, 0)?;
        assert!(empty_budget.solution.is_none());
        assert_eq!(empty_budget.attempted, 0);
        assert_eq!(
            solve_decimal_sha256_zero_nibbles(prefix, 17, 1),
            Err(DecimalSha256Error::DifficultyAbove16(17))
        );
        Ok(())
    }

    #[test]
    fn test_bincode_string_serialize() -> anyhow::Result<()> {
        let string = "hello";
        let mut homegrown = Vec::new();
        build_mcaptcha_prefix(&mut homegrown, string, "z")
            .ok_or_else(|| anyhow::anyhow!("small test string length is not representable"))?;
        let mut official = Vec::new();
        build_prefix_official(&mut official, string, "z")?;
        assert_eq!(homegrown, official);
        Ok(())
    }

    #[test]
    fn test_cerberus_mask() -> anyhow::Result<()> {
        fn check_small(hash: &[u8; 32], n: u32) -> bool {
            // <https://github.com/sjtug/cerberus/blob/ee8f903f1311da7022aec68c8686739b40f4a168/pow/src/check_dubit.rs>
            let first_word = u32::from(hash[0]) << 24
                | u32::from(hash[1]) << 16
                | u32::from(hash[2]) << 8
                | u32::from(hash[3]);
            first_word.leading_zeros() >= n * 2
        }

        for i in 1_u8..8 {
            let mask = compute_mask_cerberus(nonzero(i)?);
            eprintln!("mask: {mask:016x}");
        }

        // hash[0] is the LSB of the H0 register

        let mask = compute_mask_cerberus(nonzero(7)?);
        let mask_hi = u32::try_from(mask >> 32)?;
        let mask_lo = u32::try_from(mask & u64::from(u32::MAX))?;
        assert_eq!(mask_lo, 0);
        let hash_partial = (!mask_hi).to_le_bytes();
        eprintln!("hash_partial: {hash_partial:02x?}");
        let mut test = [0; 32];
        test[..4].copy_from_slice(&hash_partial);
        assert!(check_small(&test, 7));
        Ok(())
    }
}

#[cfg(test)]
mod challenge_domain_tests {
    use super::*;

    fn nonzero(value: u8) -> anyhow::Result<NonZeroU8> {
        NonZeroU8::new(value).ok_or_else(|| anyhow::anyhow!("expected nonzero test value"))
    }

    #[test]
    fn goaway_mask_covers_full_u64_domain() -> anyhow::Result<()> {
        assert_eq!(compute_mask_goaway(nonzero(1)?), 1_u64 << 63);
        assert_eq!(compute_mask_goaway(nonzero(33)?), u64::MAX << 31);
        assert_eq!(compute_mask_goaway(nonzero(64)?), u64::MAX);
        Ok(())
    }

    #[test]
    fn goaway_full_digest_predicate_covers_zero_through_256_bits() {
        let all_zero = [0_u32; 8];
        assert!(goaway_hash_matches_difficulty(&all_zero, 0));
        assert!(goaway_hash_matches_difficulty(&all_zero, 256));
        assert!(!goaway_hash_matches_difficulty(&all_zero, 257));

        let hash = [
            0,
            0,
            0x07ff_ffff,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ];
        assert!(goaway_hash_matches_difficulty(&hash, 69));
        assert!(!goaway_hash_matches_difficulty(&hash, 70));
    }

    #[test]
    fn cerberus_mask_clamps_to_the_upstream_32_bit_check() -> anyhow::Result<()> {
        let full = compute_mask_cerberus(nonzero(16)?);
        assert_eq!(full, u64::from(u32::MAX) << 32);
        assert_eq!(compute_mask_cerberus(nonzero(17)?), full);
        assert_eq!(compute_mask_cerberus(nonzero(u8::MAX)?), full);
        Ok(())
    }
}
