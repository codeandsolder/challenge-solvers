#![expect(
    clippy::inline_always,
    reason = "hot solver kernels force inlining across scalar, SIMD, and architecture wrappers to preserve codegen"
)]
macro_rules! impl_decimal_solver {
    ([$single_block_solver:ty, $double_block_solver:ty] => $decimal_solver:ident) => {
        /// Safe decimal nonce solver variant
        pub enum $decimal_solver {
            /// Single block solver variant
            SingleBlock($single_block_solver),
            /// Double block solver variant
            DoubleBlock($double_block_solver),
        }

        impl From<SingleBlockMessage> for $decimal_solver {
            fn from(message: SingleBlockMessage) -> Self {
                Self::SingleBlock(SingleBlockSolver::from(message))
            }
        }

        impl From<DoubleBlockMessage> for $decimal_solver {
            fn from(message: DoubleBlockMessage) -> Self {
                Self::DoubleBlock(DoubleBlockSolver::from(message))
            }
        }

        impl From<DecimalMessage> for $decimal_solver {
            fn from(message: DecimalMessage) -> Self {
                match message {
                    DecimalMessage::SingleBlock(message) => {
                        Self::SingleBlock(<$single_block_solver>::from(message))
                    }
                    DecimalMessage::DoubleBlock(message) => {
                        Self::DoubleBlock(<$double_block_solver>::from(message))
                    }
                }
            }
        }

        impl crate::solver::Solver for $decimal_solver {
            type Output = [u32; 8];
            fn set_limit(&mut self, limit: u64) {
                match self {
                    Self::SingleBlock(solver) => solver.set_limit(limit),
                    Self::DoubleBlock(solver) => solver.set_limit(limit),
                }
            }

            fn attempted_nonces(&self) -> u64 {
                match self {
                    Self::SingleBlock(solver) => solver.attempted_nonces(),
                    Self::DoubleBlock(solver) => solver.attempted_nonces(),
                }
            }

            fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
                match self {
                    Self::SingleBlock(solver) => solver.solve::<TYPE>(target, mask),
                    Self::DoubleBlock(solver) => solver.solve::<TYPE>(target, mask),
                }
            }
        }
    };
}

/// Unified x86-64 solver facade.
#[cfg(feature = "portable-simd")]
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Shared stateful wrappers for portable wide-SIMD x86 backends.
#[cfg(feature = "portable-simd")]
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod wide_backend;

/// Audited x86 target-feature entry points for the portable wide-SIMD kernels.
#[cfg(feature = "portable-simd")]
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod wide_backend_arch;

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
/// SHA-NI solver
pub mod sha_ni;

/// SIMD128 solver
#[cfg(feature = "portable-simd")]
#[cfg(target_arch = "wasm32")]
pub mod simd128;

/// Shared portable-SIMD implementation used by the wide x86 and WASM backends.
#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(crate) mod wide;

#[cfg(all(
    feature = "portable-simd",
    feature = "internals",
    target_arch = "x86_64"
))]
#[doc(hidden)]
pub mod width_bench;

/// Safe solver
pub mod safe;

#[inline(always)]
pub(crate) fn increment_ascii_decimal(digits: &mut [u8]) -> bool {
    for digit in digits.iter_mut().rev() {
        if *digit != b'9' {
            *digit += 1;
            return false;
        }
        *digit = b'0';
    }
    true
}

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
struct PreparedBinaryBlocks {
    blocks: [[u8; 64]; 2],
    current_block: usize,
    first_block: [u32; 16],
    second_block_schedule: [u32; 64],
}

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn prepare_binary_blocks(message: &crate::message::BinaryMessage) -> Option<PreparedBinaryBlocks> {
    let salt = &message.salt_residual[..message.salt_residual_len];
    let mut blocks = [[0u8; 64]; 2];
    blocks[0][..salt.len()].copy_from_slice(salt);
    let mut ptr = salt.len() + usize::from(message.nonce_byte_count.get());
    let mut current_block = ptr / 64;
    ptr %= 64;

    blocks[current_block][ptr] = 0x80;
    ptr += 1;
    if ptr + 8 > 64 {
        current_block = 1;
    }
    let bit_length = u64::try_from(message.message_length).ok()?.checked_mul(8)?;
    blocks[current_block][56..].copy_from_slice(&bit_length.to_be_bytes());

    let first_chunks = blocks[0].as_chunks::<4>().0;
    let first_block = core::array::from_fn(|i| u32::from_be_bytes(first_chunks[i]));
    let mut second_block_schedule = [0; 64];
    if current_block == 1 {
        let second_chunks = blocks[1].as_chunks::<4>().0;
        for (word, chunk) in second_block_schedule[..16].iter_mut().zip(second_chunks) {
            *word = u32::from_be_bytes(*chunk);
        }
        crate::sha256::do_message_schedule_k_w(&mut second_block_schedule);
    }

    Some(PreparedBinaryBlocks {
        blocks,
        current_block,
        first_block,
        second_block_schedule,
    })
}

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn patch_cerberus_decimal_residual(
    mut message: [u8; 64],
    residual_len: usize,
    position_mod: usize,
    resid0: u8,
    resid1: u8,
) -> [u8; 64] {
    match position_mod {
        0 | 3 => {
            message[residual_len] = resid0 + b'0';
            message[residual_len + 8] = resid1 + b'0';
        }
        1 => {
            message[residual_len + 7] = resid0 + b'0';
            message[residual_len + 8] = resid1 + b'0';
        }
        2 => {
            message[residual_len] = resid0 + b'0';
            message[residual_len + 1] = resid1 + b'0';
        }
        _ => unreachable!(),
    }
    message
}

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn cerberus_decimal_nonce(
    position_mod: usize,
    middle_word: u64,
    success_lane: u64,
    resid0: u8,
    resid1: u8,
) -> u64 {
    match position_mod {
        0 => {
            10 * middle_word
                + 100_000 * success_lane
                + 100_000_000 * u64::from(resid0)
                + u64::from(resid1)
        }
        1 => {
            100 * middle_word
                + 1_000_000 * success_lane
                + 10 * u64::from(resid0)
                + u64::from(resid1)
        }
        2 => {
            1000 * middle_word
                + success_lane
                + 100_000_000 * u64::from(resid0)
                + 10_000_000 * u64::from(resid1)
        }
        3 => {
            10000 * middle_word
                + 10 * success_lane
                + 100_000_000 * u64::from(resid0)
                + u64::from(resid1)
        }
        _ => unreachable!(),
    }
}

/// Less than test (such as Anubis)
pub const SOLVE_TYPE_LT: u8 = 1;
/// Greater than test (such as mCaptcha)
pub const SOLVE_TYPE_GT: u8 = 2;
/// Mask test (such as Cap.js)
pub const SOLVE_TYPE_MASK: u8 = 4;
/// Full-digest leading-zero-bit test. `target` is the required bit count and
/// `mask` is ignored.
pub const SOLVE_TYPE_LEADING_ZERO_BITS: u8 = 8;

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const HMAC_IPAD: u8 = 0x36;
#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const HMAC_IPAD32: u32 = u32::from_be_bytes([HMAC_IPAD; 4]);
#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const HMAC_OPAD: u8 = 0x5c;
#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) const HMAC_OPAD32: u32 = u32::from_be_bytes([HMAC_OPAD; 4]);

/// A token for checking CPU features
pub trait CpuIDToken: Default + Copy + Clone + 'static {
    /// Get the CPU feature status
    fn get() -> bool;
}

/// A solver router that can take a solver or a fallback solver based on the CPU features
#[derive(Debug, Copy, Clone)]
#[expect(
    missing_docs,
    reason = "variant fields are internal dispatch state and not part of the public solver contract"
)]
#[cfg_attr(not(feature = "runtime-dispatch"), repr(transparent))]
pub enum SolverRouter<T: CpuIDToken, M, S: Solver + From<M>, F: Solver + From<M>> {
    #[cfg(feature = "runtime-dispatch")]
    Taken {
        solver: S,
        _marker: core::marker::PhantomData<(S, F, T, M)>,
    },
    Fallback {
        solver: F,
        _marker: core::marker::PhantomData<(S, F, T, M)>,
    },
}

impl<T: CpuIDToken, M, S: Solver + From<M>, F: Solver + From<M>> From<M>
    for SolverRouter<T, M, S, F>
{
    fn from(value: M) -> Self {
        #[cfg(feature = "runtime-dispatch")]
        if T::get() {
            return Self::Taken {
                solver: value.into(),
                _marker: core::marker::PhantomData,
            };
        }
        Self::Fallback {
            solver: value.into(),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<O: Copy, T: CpuIDToken, M, S: Solver<Output = O> + From<M>, F: Solver<Output = O> + From<M>>
    Solver for SolverRouter<T, M, S, F>
{
    type Output = O;

    fn set_limit(&mut self, limit: u64) {
        match self {
            #[cfg(feature = "runtime-dispatch")]
            Self::Taken { solver, .. } => solver.set_limit(limit),
            Self::Fallback { solver, .. } => solver.set_limit(limit),
        }
    }
    fn attempted_nonces(&self) -> u64 {
        match self {
            #[cfg(feature = "runtime-dispatch")]
            Self::Taken { solver, .. } => solver.attempted_nonces(),
            Self::Fallback { solver, .. } => solver.attempted_nonces(),
        }
    }

    #[inline]
    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        match self {
            #[cfg(feature = "runtime-dispatch")]
            Self::Taken { solver, .. } => solver.solve_nonce_only::<TYPE>(target, mask),
            Self::Fallback { solver, .. } => solver.solve_nonce_only::<TYPE>(target, mask),
        }
    }

    #[inline]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, Self::Output)> {
        match self {
            #[cfg(feature = "runtime-dispatch")]
            Self::Taken { solver, .. } => solver.solve::<TYPE>(target, mask),
            Self::Fallback { solver, .. } => solver.solve::<TYPE>(target, mask),
        }
    }
}

/// A generic solver trait
pub trait Solver {
    /// The output type
    type Output: Copy;

    /// Returns a valid nonce and its corresponding hash value.
    ///
    /// Supported schemes:
    ///
    /// - `SOLVE_TYPE_LT`: Less than test (such as Anubis)
    /// - `SOLVE_TYPE_GT`: Greater than test (such as mCaptcha)
    /// - `SOLVE_TYPE_MASK`: Mask test (such as Cap.js)
    /// - `SOLVE_TYPE_LEADING_ZERO_BITS`: full SHA-256 leading-zero-bit test;
    ///   `target` contains the required bit count and `mask` is ignored
    ///
    /// Currently bitmasking `SOLVE_TYPE_MASK` with `SOLVE_TYPE_GT` or `SOLVE_TYPE_LT` is not supported and may result in unexpected behavior.
    ///
    /// Returns None when the solver cannot solve the prefix.
    ///
    /// Failure is usually because the key space is exhausted (or presumed exhausted).
    /// It should by design happen extremely rarely for common difficulty settings.
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, Self::Output)>;

    /// Returns a valid nonce without the actual hash.
    ///
    /// A trivial implementation is provided by default.
    #[inline]
    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        self.solve::<TYPE>(target, mask).map(|(nonce, _)| nonce)
    }

    /// Set the limit.
    fn set_limit(&mut self, limit: u64);

    /// Get the attempted nonces.
    fn attempted_nonces(&self) -> u64;
}

/// A dyn-dispatching wrapper for Solver
pub trait SolverDyn {
    /// The output type
    type Output: Copy;

    /// A dynamic dispatching wrapper for solve
    fn solve_dyn(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, Self::Output)>;
    /// A dynamic dispatching wrapper for `solve_nonce_only`
    fn solve_nonce_only_dyn(&mut self, target: u64, ty: u8, mask: u64) -> Option<u64>;
}

impl<S: Solver> SolverDyn for S {
    type Output = S::Output;

    // A dynamic dispatching wrapper for solve
    fn solve_dyn(&mut self, target: u64, ty: u8, mask: u64) -> Option<(u64, Self::Output)> {
        match ty {
            SOLVE_TYPE_LT => self.solve::<SOLVE_TYPE_LT>(target, mask),
            SOLVE_TYPE_GT => self.solve::<SOLVE_TYPE_GT>(target, mask),
            SOLVE_TYPE_LEADING_ZERO_BITS => {
                self.solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(target, mask)
            }
            _ => self.solve::<SOLVE_TYPE_MASK>(target, mask),
        }
    }

    // A dynamic dispatching wrapper for solve_nonce_only
    fn solve_nonce_only_dyn(&mut self, target: u64, ty: u8, mask: u64) -> Option<u64> {
        match ty {
            SOLVE_TYPE_LT => self.solve_nonce_only::<SOLVE_TYPE_LT>(target, mask),
            SOLVE_TYPE_GT => self.solve_nonce_only::<SOLVE_TYPE_GT>(target, mask),
            SOLVE_TYPE_LEADING_ZERO_BITS => {
                self.solve_nonce_only::<SOLVE_TYPE_LEADING_ZERO_BITS>(target, mask)
            }
            _ => self.solve_nonce_only::<SOLVE_TYPE_MASK>(target, mask),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use core::num::NonZeroU8;
    use std::io::Write;

    use anyhow::Context as _;
    use pbkdf2::hmac::Hmac;
    use sha2::{Digest, Sha256};

    mod pow_sha256;

    use crate::{
        compute_mask_cerberus, compute_mask_goaway, compute_mask_zero_nibbles,
        compute_target_mcaptcha, extract128_be, message::IEEE754LosslessFixupPrefix,
    };

    use super::*;

    const MCAPTCHA_TEST_DIFFICULTY: u32 = 100_000;
    const BINARY_TEST_DIFFICULTY: u64 = 100_000;
    const FULL_DIGEST_TEST_BITS: u64 = 4;

    const fn cerberus_small_check(hash: &[u8; 32], n: u32) -> bool {
        u32::from_be_bytes([hash[0], hash[1], hash[2], hash[3]]).leading_zeros() >= n * 2
    }

    /// Extract top 64 bits from a 64-bit word array
    const fn extract64_be(inp: [u32; 8]) -> u64 {
        let hi = inp[0].to_be_bytes();
        let lo = inp[1].to_be_bytes();
        u64::from_be_bytes([hi[0], hi[1], hi[2], hi[3], lo[0], lo[1], lo[2], lo[3]])
    }

    fn validate_decimal_full_digest<S: Solver<Output = [u32; 8]>>(
        solver: &mut S,
        prefix: &[u8],
    ) -> anyhow::Result<()> {
        let (nonce, result) = solver
            .solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(FULL_DIGEST_TEST_BITS, u64::MAX)
            .context("full-digest solver failed")?;
        let mut message = prefix.to_vec();
        write!(message, "{nonce}")?;
        let expected = Sha256::digest(&message);
        let expected_words = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([
                expected[base],
                expected[base + 1],
                expected[base + 2],
                expected[base + 3],
            ])
        });
        assert_eq!(result, expected_words);
        assert!(crate::hash_has_leading_zero_bits(
            &result,
            FULL_DIGEST_TEST_BITS,
        ));
        Ok(())
    }

    fn validate_decimal_mcaptcha<S: Solver<Output = [u32; 8]>>(
        solver: &mut S,
        phrase: &str,
        search_space: u32,
    ) -> anyhow::Result<()> {
        let difficulty = core::num::NonZeroU64::new(u64::from(MCAPTCHA_TEST_DIFFICULTY))
            .context("mCaptcha test difficulty is nonzero")?;
        let target = compute_target_mcaptcha(difficulty);
        let target_u64 = u64::from_be_bytes(target.to_be_bytes()[..8].try_into()?);
        let (nonce, result) = solver
            .solve::<SOLVE_TYPE_GT>(target_u64, u64::MAX)
            .context("GT solver failed")?;
        let result_u128 = extract128_be(result);
        let config = pow_sha256::Config {
            salt: String::from("abc"),
        };
        let response = pow_sha256::PoW::new(nonce, result_u128.to_string());
        let solver_name = core::any::type_name::<S>();

        assert_eq!(
            config.calculate(&response, &phrase)?,
            result_u128,
            "response {response:?} differs for solver {solver_name}, search_space={search_space}",
        );
        assert!(
            config.is_valid_proof(&response, &phrase),
            "{result_u128} is not a valid proof for solver {solver_name}",
        );
        assert!(
            pow_sha256::Config::is_sufficient_difficulty(&response, MCAPTCHA_TEST_DIFFICULTY),
            "{result_u128:016x} is below expected target {target:032x} for solver {solver_name}",
        );
        Ok(())
    }

    fn validate_decimal_anubis<S: Solver<Output = [u32; 8]>>(
        solver: &mut S,
        prefix: &[u8],
        search_space: u32,
        difficulty: NonZeroU8,
    ) -> anyhow::Result<()> {
        let mask = compute_mask_zero_nibbles(difficulty);
        let (nonce, result) = solver
            .solve::<SOLVE_TYPE_MASK>(0, mask)
            .context("Anubis mask solver failed")?;
        let result_u64 = extract64_be(result);
        let result_bytes = result_u64.to_be_bytes();
        let mut message = prefix.to_vec();
        write!(message, "{nonce}")?;
        let expected = Sha256::digest(&message);
        let solver_name = core::any::type_name::<S>();

        assert_eq!(expected[..8], result_bytes);
        assert_eq!(
            result_u64 & mask,
            0,
            "Anubis result {result_u64:016x} misses mask {mask:016x} for solver {solver_name}, search_space={search_space}",
        );
        for nibble_index in 0..usize::from(difficulty.get()) {
            let byte_index = nibble_index / 2;
            let shift = (1 - nibble_index % 2) * 4;
            let nibble = (result_bytes[byte_index] >> shift) & 0x0f;
            assert_eq!(
                nibble, 0,
                "{result_u64:016x} is not valid Anubis proof for solver {solver_name}, nibble={nibble_index}",
            );
        }
        Ok(())
    }

    fn validate_decimal_equality<S: Solver<Output = [u32; 8]>>(
        solver: &mut S,
        search_space: u32,
    ) -> anyhow::Result<()> {
        let target = 0b10111 << (64 - 5);
        let (_, result) = solver
            .solve::<SOLVE_TYPE_MASK>(target, u64::MAX << (64 - 5))
            .context("64-bit equality solver failed")?;
        let result_u128 = extract128_be(result);
        let solver_name = core::any::type_name::<S>();
        assert_eq!(
            result_u128 >> (128 - 5),
            0b10111,
            "equality result {result_u128:016x} differs for solver {solver_name}, search_space={search_space}",
        );

        #[cfg(all(feature = "compare-64bit", target_arch = "x86_64"))]
        {
            let target = 0b10111 << (32 - 5);
            let mask = 0b11111 << (32 - 5);
            let (_, result) = solver
                .solve::<SOLVE_TYPE_MASK>(target, mask)
                .context("32-bit equality solver failed")?;
            let result_u64 = extract64_be(result);
            assert_eq!(
                result_u64 & mask,
                target,
                "32-bit equality result {result_u64:016x} differs for solver {solver_name}, search_space={search_space}",
            );
        }
        Ok(())
    }

    pub fn test_decimal_validator<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8], u32) -> Option<S>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        let anubis_difficulty = NonZeroU8::new(4).context("Anubis test difficulty is nonzero")?;
        for phrase_len in 0..64 {
            let phrase = String::from_iter(std::iter::repeat_n('a', phrase_len));
            let mut prefix = b"abc".to_vec();
            prefix.extend_from_slice(&bincode::serialize(&phrase)?);

            for search_space in [0, 1, 9, 10, 11] {
                let [
                    Some(mut solver),
                    Some(mut anubis_solver),
                    Some(mut eq_solver),
                    Some(mut full_digest_solver),
                ] = core::array::from_fn(|_| factory(&prefix, search_space))
                else {
                    assert_ne!(
                        search_space, 0,
                        "baseline decimal shape must construct every validation solver",
                    );
                    break;
                };

                validate_decimal_full_digest(&mut full_digest_solver, &prefix)?;
                validate_decimal_mcaptcha(&mut solver, &phrase, search_space)?;
                validate_decimal_anubis(
                    &mut anubis_solver,
                    &prefix,
                    search_space,
                    anubis_difficulty,
                )?;
                validate_decimal_equality(&mut eq_solver, search_space)?;
            }
        }
        Ok(())
    }

    pub fn test_binary_validator<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8], NonZeroU8) -> Option<S>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        for nonce_byte_count in [4, 5, 8] {
            for prefix_len in 0..64 {
                let mut msg = Vec::from_iter(b"abc".iter().copied().cycle().take(prefix_len));
                let nonce_byte_count = NonZeroU8::new(nonce_byte_count)
                    .context("binary validator nonce width is nonzero")?;
                let mut solver = factory(&msg, nonce_byte_count)
                    .context("supported binary validator width must construct")?;
                let target = compute_target_mcaptcha(
                    core::num::NonZeroU64::new(BINARY_TEST_DIFFICULTY)
                        .context("binary test difficulty is nonzero")?,
                );
                let (nonce, result) = solver
                    .solve::<SOLVE_TYPE_GT>(target, !0)
                    .context("binary GT solver failed")?;
                msg.extend_from_slice(&nonce.to_le_bytes()[..usize::from(nonce_byte_count.get())]);
                let expected_digest = sha2::Sha256::digest(&msg);
                let mut got_digest = [0; 32];
                for i in 0..8 {
                    got_digest[i * 4..][..4].copy_from_slice(&result[i].to_be_bytes());
                }
                assert_eq!(
                    expected_digest.as_slice(),
                    got_digest.as_slice(),
                    "nonce: {nonce}, prefix_len: {prefix_len}",
                );
                let got_difficulty = u64::from_be_bytes(got_digest[..8].try_into()?);
                assert!(
                    got_difficulty >= target,
                    "got_difficulty: {got_difficulty:016x} < target: {target:016x} (nonce: {nonce}, prefix_len: {prefix_len})",
                );
            }
        }
        Ok(())
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "this test intentionally models JavaScript Number conversion of a u64 nonce"
    )]
    pub fn test_decimal_validator_f64_safe<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8], u32) -> Option<(S, Option<IEEE754LosslessFixupPrefix>)>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        use std::io::Write;
        let mut tested_by_search_space = [0_u32; 3];
        for c in b'a'..=b'c' {
            let salts = [c; 64];
            for len in 1..=64 {
                for search_space in 0..=2 {
                    let u64_target = 0b10111 << (64 - 5);
                    let u64_mask = !0 << (64 - 5);
                    let Some((mut solver, fixup_prefix)) = factory(&salts[..len], search_space)
                    else {
                        assert_ne!(search_space, 0, "baseline f64 shape must remain supported");
                        continue;
                    };
                    tested_by_search_space[usize::try_from(search_space)?] += 1;

                    let (nonce, result) = solver
                        .solve::<SOLVE_TYPE_MASK>(u64_target, u64_mask)
                        .context("f64-safe mask solver failed")?;
                    let mut final_message = salts[..len].to_vec();
                    let mut final_message_f64 = final_message.clone();
                    let final_nonce = fixup_prefix.as_ref().map_or(nonce as f64, |fixup_prefix| {
                        final_message.extend_from_slice(fixup_prefix.as_ref());
                        fixup_prefix.fixup(nonce)
                    });
                    if fixup_prefix.is_some() {
                        assert_ne!(nonce % 10, 0);
                    }
                    write!(final_message, "{nonce:09}")?;
                    write!(final_message_f64, "{final_nonce}")?;
                    assert_eq!(
                        final_message_f64,
                        final_message,
                        "final_nonce: {}, final_nonce_f64: {} (fixup_prefix: {:?})",
                        String::from_utf8_lossy(&final_message),
                        String::from_utf8_lossy(&final_message_f64),
                        fixup_prefix,
                    );
                    let hash_result = sha2::Sha256::digest(&final_message);
                    let hash_result_u64 = u64::from_be_bytes(hash_result[..8].try_into()?);
                    assert_eq!(
                        hash_result_u64 & u64_mask,
                        u64_target,
                        "hash_result_u64: {:016x} (solver: {}, len: {}, message: {:?})",
                        hash_result_u64,
                        core::any::type_name::<S>(),
                        len,
                        String::from_utf8_lossy(&final_message)
                    );
                    assert_eq!(
                        hash_result_u64,
                        ((u64::from(result[0])) << 32 | u64::from(result[1]))
                    );
                }
            }
        }
        assert!(
            tested_by_search_space.into_iter().all(|count| count > 0),
            "each requested f64 working-set value must exercise at least one supported shape"
        );
        Ok(())
    }

    pub fn test_cerberus_decimal_validator<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8]) -> Option<S>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        use std::io::Write;

        for df in (5_u8..=7).chain(core::iter::once(9)) {
            let difficulty = NonZeroU8::new(df).context("Cerberus test difficulty is nonzero")?;
            let mask = compute_mask_cerberus(difficulty);
            eprintln!("mask: {mask:08x}");

            let mut test_seed = [0_u8; 128];
            for (i, byte) in test_seed.iter_mut().enumerate() {
                *byte = b'a'.wrapping_add(u8::try_from(i)?);
            }

            for seed_len in 0..128 {
                let Some(mut solver) = factory(&test_seed[..seed_len]) else {
                    if seed_len < 64 {
                        eprintln!("solver is None for seed_len: {seed_len}");
                    }
                    continue;
                };

                let (nonce, hash) = solver
                    .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, mask)
                    .context("Cerberus decimal solver failed")?;
                let mut msg = test_seed[..seed_len].to_vec();
                write!(&mut msg, "{nonce}")?;

                let mut ref_hasher = blake3::Hasher::new();
                ref_hasher.update(msg.as_slice());
                let ref_hash = ref_hasher.finalize();
                let ref_hash_bytes = ref_hash.as_bytes();
                let ref_hash = core::array::from_fn(|i| {
                    u32::from_le_bytes([
                        ref_hash_bytes[i * 4],
                        ref_hash_bytes[i * 4 + 1],
                        ref_hash_bytes[i * 4 + 2],
                        ref_hash_bytes[i * 4 + 3],
                    ])
                });
                let hit = ((u64::from(ref_hash[0])) << 32 | u64::from(ref_hash[1])) & mask == 0;
                assert_eq!(
                    hash, ref_hash,
                    "incorrect output: {nonce} (seed_len: {seed_len})"
                );
                assert!(hit);
                assert!(cerberus_small_check(ref_hash_bytes, u32::from(df)));
            }
        }
        Ok(())
    }

    pub fn test_cerberus_binary_validator<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8]) -> Option<S>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        for seed in [b"a", b"b"] {
            let seed_hash = ::blake3::hash(seed).to_hex();

            for df in (5_u8..=7).chain(core::iter::once(9)) {
                let difficulty =
                    NonZeroU8::new(df).context("Cerberus test difficulty is nonzero")?;
                let mask = compute_mask_cerberus(difficulty);
                eprintln!("mask: {mask:08x}");

                let mut solver = factory(seed)
                    .with_context(|| format!("solver is unavailable for seed {seed:?}"))?;

                let (nonce, hash) = solver
                    .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, mask)
                    .context("Cerberus binary solver failed")?;
                let mut msg = seed_hash.as_bytes().to_vec();
                msg.extend_from_slice(nonce.rotate_right(32).to_le_bytes().as_slice());

                let mut ref_hasher = blake3::Hasher::new();
                ref_hasher.update(msg.as_slice());
                let ref_hash = ref_hasher.finalize();
                let ref_hash_bytes = ref_hash.as_bytes();
                let ref_hash = core::array::from_fn(|i| {
                    u32::from_le_bytes([
                        ref_hash_bytes[i * 4],
                        ref_hash_bytes[i * 4 + 1],
                        ref_hash_bytes[i * 4 + 2],
                        ref_hash_bytes[i * 4 + 3],
                    ])
                });
                let hit = ((u64::from(ref_hash[0])) << 32 | u64::from(ref_hash[1])) & mask == 0;
                assert_eq!(hash, ref_hash, "incorrect output: {nonce}");
                assert!(hit);
                assert!(cerberus_small_check(ref_hash_bytes, u32::from(df)));
            }
        }
        Ok(())
    }

    pub fn test_goaway_validator<
        S: Solver<Output = [u32; 8]>,
        F: for<'a> FnMut(&'a [u8; 32]) -> S,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        let difficulty = NonZeroU8::new(10).context("GoAway test difficulty is nonzero")?;
        let mask = compute_mask_goaway(difficulty);
        let mut test_prefix = [0_u8; 32];
        for (i, byte) in test_prefix.iter_mut().enumerate() {
            *byte = u8::try_from(i)?;
        }

        let mut solver = factory(&test_prefix);
        let mut eq_solver = factory(&test_prefix);
        let mut full_digest_solver = factory(&test_prefix);

        let (full_nonce, full_result) = full_digest_solver
            .solve::<SOLVE_TYPE_LEADING_ZERO_BITS>(FULL_DIGEST_TEST_BITS, u64::MAX)
            .context("full-digest GoAway solver failed")?;
        let mut full_hasher = Sha256::default();
        full_hasher.update(test_prefix);
        full_hasher.update(full_nonce.to_be_bytes());
        let full_hash = full_hasher.finalize();
        let full_expected = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([
                full_hash[base],
                full_hash[base + 1],
                full_hash[base + 2],
                full_hash[base + 3],
            ])
        });
        assert_eq!(
            full_result, full_expected,
            "full-digest GoAway result mismatch"
        );
        assert!(crate::hash_has_leading_zero_bits(
            &full_result,
            FULL_DIGEST_TEST_BITS,
        ));

        let (nonce, result) = solver
            .solve::<SOLVE_TYPE_MASK>(0, mask)
            .context("GoAway mask solver failed")?;

        let mut hasher = Sha256::default();
        hasher.update(test_prefix);
        hasher.update(nonce.to_be_bytes());
        let hash = hasher.finalize();
        let expected = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([hash[base], hash[base + 1], hash[base + 2], hash[base + 3]])
        });
        assert_eq!(result, expected, "solver returned the wrong GoAway digest");
        let prefix = (u64::from(expected[0]) << 32) | u64::from(expected[1]);
        assert_eq!(prefix & mask, 0, "returned GoAway nonce misses target");

        let eq_target = 0b10111 << (64 - 5);
        let eq_mask = !0 << (64 - 5);
        let (_, eq_result) = eq_solver
            .solve::<{ SOLVE_TYPE_MASK }>(eq_target, eq_mask)
            .context("GoAway equality solver failed")?;
        let eq_result_u128 = extract128_be(eq_result);
        assert_eq!(
            eq_result_u128 >> (128 - 5),
            0b10111,
            "eq_result: {:016x} (solver: {})",
            eq_result_u128,
            core::any::type_name::<S>()
        );

        #[cfg(all(feature = "compare-64bit", target_arch = "x86_64"))]
        {
            let eq_target = 0b10111 << (32 - 5);
            let eq_mask = 0b11111 << (32 - 5);
            let (_, eq_result) = eq_solver
                .solve::<{ SOLVE_TYPE_MASK }>(eq_target, eq_mask)
                .context("GoAway 32-bit equality solver failed")?;
            let eq_result_u64 = extract64_be(eq_result);
            assert_eq!(
                eq_result_u64 & eq_mask,
                eq_target,
                "eq_result_u64: {:016x} (solver: {})",
                eq_result_u64,
                core::any::type_name::<S>()
            );
        }
        Ok(())
    }

    fn validate_altcha_known_answer<S, F>(
        factory: &mut F,
        message: crate::message::AltchaMessage,
        expected_nonce: u64,
        expected_hash: [u32; 8],
    ) -> anyhow::Result<()>
    where
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    {
        let mut solver = factory(message).context("ALTCHA solver unavailable")?;
        let difficulty = NonZeroU8::new(2).context("ALTCHA difficulty is nonzero")?;
        let (nonce, hash) = solver
            .solve::<SOLVE_TYPE_MASK>(0, compute_mask_zero_nibbles(difficulty))
            .context("ALTCHA known-answer solver failed")?;
        assert_eq!(nonce, expected_nonce);
        assert_eq!(hash, expected_hash);
        Ok(())
    }

    fn validate_altcha_known_answers<S, F>(factory: &mut F) -> anyhow::Result<()>
    where
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    {
        validate_altcha_known_answer(
            factory,
            crate::message::AltchaMessage {
                nonce: [
                    0xf0, 0xc5, 0x9c, 0x5f, 0x4b, 0x3a, 0xeb, 0x1f, 0xf3, 0x4d, 0x44, 0xab, 0xd1,
                    0x5f, 0x0f, 0x15,
                ],
                salt: [
                    0x7f, 0x49, 0x8a, 0x9e, 0xb9, 0x4b, 0xb8, 0x4f, 0x3c, 0xc2, 0xa6, 0xb4, 0x8b,
                    0x6e, 0xe6, 0x14,
                ],
                cost: 2000.try_into()?,
                pbkdf2: true,
                key_length: 32.try_into()?,
            },
            78,
            [
                0x0058_b155,
                0x1f52_6b2e,
                0xa174_87e5,
                0xf6f7_edd0,
                0x35ca_3847,
                0x8030_a59d,
                0x7b7b_ee49,
                0xb090_c58e,
            ],
        )?;
        validate_altcha_known_answer(
            factory,
            crate::message::AltchaMessage {
                nonce: [
                    0xd9, 0x8d, 0x4a, 0x93, 0x45, 0x8a, 0x69, 0x7d, 0x4c, 0xc0, 0xe8, 0x29, 0x6b,
                    0x0d, 0x73, 0xc2,
                ],
                salt: [
                    0xdd, 0xd2, 0x93, 0x34, 0xb7, 0x49, 0xfd, 0x0a, 0xd5, 0x4d, 0x6d, 0x00, 0xc5,
                    0xcc, 0x48, 0x0a,
                ],
                cost: 2000.try_into()?,
                pbkdf2: false,
                key_length: 32.try_into()?,
            },
            128,
            [
                0x0064_1b3e,
                0x3f8d_b108,
                0xd345_552e,
                0x8d9f_78fe,
                0x67a7_d0ae,
                0x0faa_2ea0,
                0x701a_0432,
                0x0c40_7bb7,
            ],
        )
    }

    fn validate_altcha_matrix<S, F>(factory: &mut F) -> anyhow::Result<()>
    where
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    {
        let difficulty = NonZeroU8::new(3).context("ALTCHA difficulty is nonzero")?;
        let mask = compute_mask_zero_nibbles(difficulty);
        for q in 0..5 {
            for r in 0..5 {
                for pbkdf2 in [false, true] {
                    let message = crate::message::AltchaMessage {
                        nonce: [
                            q, 0xc5, 0x9c, 0x5f, 0x4b, 0x3a, 0xeb, 0x1f, 0xf3, 0x4d, 0x44, 0xab,
                            0xd1, 0x5f, 0x0f, 0x15,
                        ],
                        salt: [
                            r, 0x49, 0x8a, 0x9e, 0xb9, 0x4b, 0xb8, 0x4f, 0x3c, 0xc2, 0xa6, 0xb4,
                            0x8b, 0x6e, 0xe6, 0x14,
                        ],
                        cost: 64.try_into()?,
                        pbkdf2,
                        key_length: 32.try_into()?,
                    };
                    let mut solver =
                        factory(message.clone()).context("ALTCHA matrix solver unavailable")?;
                    let (nonce, hash) = solver
                        .solve::<SOLVE_TYPE_MASK>(0, mask)
                        .context("ALTCHA matrix solver failed")?;
                    let nonce = u32::try_from(nonce)?;
                    assert_eq!(hash[0] & u32::try_from(mask >> 32)?, 0);

                    let mut expected_hash = [0; 32];
                    if pbkdf2 {
                        let mut password = [0; 20];
                        password[..16].copy_from_slice(&message.nonce);
                        password[16..].copy_from_slice(&nonce.to_be_bytes());
                        ::pbkdf2::pbkdf2::<Hmac<Sha256>>(
                            &password,
                            &message.salt,
                            64,
                            &mut expected_hash,
                        )?;
                    } else {
                        let mut password = [0; 36];
                        password[..16].copy_from_slice(&message.salt);
                        password[16..32].copy_from_slice(&message.nonce);
                        password[32..].copy_from_slice(&nonce.to_be_bytes());
                        let mut data = password.as_slice();
                        for _ in 0..64 {
                            expected_hash = Sha256::digest(data).into();
                            data = &expected_hash;
                        }
                    }
                    let expected_words = core::array::from_fn(|i| {
                        let base = i * 4;
                        u32::from_be_bytes([
                            expected_hash[base],
                            expected_hash[base + 1],
                            expected_hash[base + 2],
                            expected_hash[base + 3],
                        ])
                    });
                    assert_eq!(hash, expected_words);
                }
            }
        }
        Ok(())
    }

    fn validate_altcha_truncated_keys<S, F>(factory: &mut F) -> anyhow::Result<()>
    where
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    {
        let difficulty = NonZeroU8::new(1).context("ALTCHA difficulty is nonzero")?;
        let mask = compute_mask_zero_nibbles(difficulty);
        for key_length in [1_u32, 4, 5, 8, 12, 16, 20, 24, 28, 31] {
            let message = crate::message::AltchaMessage {
                nonce: [0x35; 16],
                salt: [0xa7; 16],
                cost: 3.try_into()?,
                pbkdf2: false,
                key_length: key_length.try_into()?,
            };
            let mut solver =
                factory(message.clone()).context("ALTCHA truncated-key solver unavailable")?;
            let (nonce, hash) = solver
                .solve::<SOLVE_TYPE_MASK>(0, mask)
                .context("truncated-key solver failed")?;
            let nonce = u32::try_from(nonce).context("ALTCHA counter fits u32")?;

            let mut first = [0_u8; 36];
            first[..16].copy_from_slice(&message.salt);
            first[16..32].copy_from_slice(&message.nonce);
            first[32..].copy_from_slice(&nonce.to_be_bytes());
            let mut digest = Sha256::digest(first);
            for _ in 1..message.cost.get() {
                digest = Sha256::digest(&digest[..usize::try_from(key_length)?]);
            }
            let expected = core::array::from_fn(|i| {
                let base = i * 4;
                u32::from_be_bytes([
                    digest[base],
                    digest[base + 1],
                    digest[base + 2],
                    digest[base + 3],
                ])
            });
            assert_eq!(
                hash, expected,
                "ALTCHA nested key_length={key_length}, nonce={nonce}",
            );
        }
        Ok(())
    }

    fn validate_altcha_resume<S, F>(factory: &mut F) -> anyhow::Result<()>
    where
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    {
        let message = crate::message::AltchaMessage {
            nonce: [0x42; 16],
            salt: [0x91; 16],
            cost: 1.try_into()?,
            pbkdf2: false,
            key_length: 32.try_into()?,
        };
        let target_nonce = 4_u32;
        let reference = |counter: u32| {
            let mut bytes = [0_u8; 36];
            bytes[..16].copy_from_slice(&message.salt);
            bytes[16..32].copy_from_slice(&message.nonce);
            bytes[32..].copy_from_slice(&counter.to_be_bytes());
            Sha256::digest(bytes)
        };
        let target_digest = reference(target_nonce);
        let target = u64::from_be_bytes([
            target_digest[0],
            target_digest[1],
            target_digest[2],
            target_digest[3],
            0,
            0,
            0,
            0,
        ]);
        let mask = 0xffff_ffff_0000_0000;
        for counter in 0..target_nonce {
            assert_ne!(
                &reference(counter)[..4],
                &target_digest[..4],
                "resume test needs a unique 32-bit prefix before the target",
            );
        }

        let mut solver = factory(message).context("ALTCHA resume solver unavailable")?;
        solver.set_limit(1);
        assert!(solver.solve::<SOLVE_TYPE_MASK>(target, mask).is_none());
        assert_eq!(solver.attempted_nonces(), 1);

        let final_limit = u64::from(target_nonce) + 1;
        solver.set_limit(final_limit);
        let (nonce, hash) = solver
            .solve::<SOLVE_TYPE_MASK>(target, mask)
            .context("resumed ALTCHA search should reach counter 4")?;
        assert_eq!(nonce, u64::from(target_nonce));
        let expected = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([
                target_digest[base],
                target_digest[base + 1],
                target_digest[base + 2],
                target_digest[base + 3],
            ])
        });
        assert_eq!(hash, expected);
        assert_eq!(solver.attempted_nonces(), final_limit);
        Ok(())
    }

    pub fn test_altcha_validator<
        S: Solver<Output = [u32; 8]>,
        F: FnMut(crate::message::AltchaMessage) -> Option<S>,
    >(
        mut factory: F,
    ) -> anyhow::Result<()> {
        validate_altcha_known_answers::<S, _>(&mut factory)?;
        validate_altcha_matrix::<S, _>(&mut factory)?;
        validate_altcha_truncated_keys::<S, _>(&mut factory)?;
        validate_altcha_resume::<S, _>(&mut factory)
    }
}
