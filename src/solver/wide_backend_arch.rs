//! x86 runtime-targeted entry points for the shared portable-SIMD solver kernels.

use crate::message::{AltchaMessage, CerberusMessage, DoubleBlockMessage, SingleBlockMessage};

/// Internal contract implemented by a concrete x86 SIMD width.
///
/// The implementation owns the only unsafe target-feature boundary. Solver state and
/// protocol dispatch remain architecture-independent in `wide_backend`.
#[doc(hidden)]
pub trait Backend: Copy + Clone + Default + 'static {
    /// Logical SIMD lane count selected by this backend.
    const LANES: usize;
    /// Human-readable backend name.
    const NAME: &'static str;

    /// Whether the running CPU supports this backend.
    fn available() -> bool;

    /// Run the single-block SHA-256 kernel.
    fn solve_single_block<
        const DIGIT_WORD_IDX0: usize,
        const DIGIT_WORD_IDX1_INCREMENT: bool,
        const TYPE: u8,
        const MUTATION_TYPE: u8,
    >(
        message: &mut SingleBlockMessage,
        attempted_nonces: &mut u64,
        limit: u64,
        target: u64,
        mask: u64,
    ) -> Option<u64>;

    /// Run the double-block SHA-256 kernel.
    fn solve_double_block<const TYPE: u8>(
        message: &mut DoubleBlockMessage,
        attempted_nonces: &mut u64,
        limit: u64,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])>;

    /// Run the binary-nonce SHA-256 kernel.
    #[expect(
        clippy::too_many_arguments,
        reason = "backend contract mirrors the complete solver kernel inputs without packing hot-path state"
    )]
    fn solve_binary<
        const TYPE: u8,
        const FIRST_NONCE_WORD_IDX: usize,
        const NEED_SECOND_BLOCK: bool,
    >(
        attempted_nonces: &mut u64,
        limit: u64,
        prefix_state: [u32; 8],
        first_block: &[u32; 16],
        second_block_schedule: &[u32; 64],
        nonce_byte_offset: usize,
        nonce_byte_count: core::num::NonZeroU8,
        target: u64,
        mask: u64,
    ) -> Option<u64>;

    /// Run the `GoAway` SHA-256 kernel.
    fn solve_goaway<const TYPE: u8>(
        attempted_nonces: &mut u64,
        limit: u64,
        challenge: [u32; 8],
        high_word: u32,
        target: u64,
        mask: u64,
    ) -> Option<u64>;

    /// Run the Cerberus decimal BLAKE3 kernel.
    fn solve_cerberus_decimal<
        const CENTER_WORD_IDX: usize,
        const LANE_ID_WORD_IDX: usize,
        const CONSTANT_WORD_COUNT: usize,
    >(
        message: &CerberusMessage,
        msg_tpl: [u32; 16],
        attempted_nonces: &mut u64,
        limit: u64,
        target: u64,
        mask: u64,
    ) -> Option<(u64, u64)>;

    /// Run the Cerberus binary BLAKE3 kernel.
    fn solve_cerberus_binary(
        message: &CerberusMessage,
        attempted_nonces: &mut u64,
        limit: u64,
        target: u64,
        mask: u64,
    ) -> Option<u64>;

    /// Run the ALTCHA SHA-256/PBKDF2 kernel.
    fn solve_altcha<const TYPE: u8>(
        message: &AltchaMessage,
        attempted_nonces: &mut u64,
        limit: u64,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])>;
}

cpufeatures::new!(has_avx2, "avx2");
cpufeatures::new!(has_avx512f, "avx512f");

/// Eight-lane portable-SIMD codegen under an AVX2 target-feature boundary.
#[derive(Debug, Copy, Clone, Default)]
pub struct Avx2;

/// Sixteen-lane portable-SIMD codegen under an AVX-512F target-feature boundary.
#[derive(Debug, Copy, Clone, Default)]
pub struct Avx512;

macro_rules! backend_impl {
    ($backend:ident, $module:ident, $lanes:literal, $detect:ident, $name:literal) => {
        impl Backend for $backend {
            const LANES: usize = $lanes;
            const NAME: &'static str = $name;

            #[inline]
            fn available() -> bool {
                $detect::get()
            }

            fn solve_single_block<
                const DIGIT_WORD_IDX0: usize,
                const DIGIT_WORD_IDX1_INCREMENT: bool,
                const TYPE: u8,
                const MUTATION_TYPE: u8,
            >(
                message: &mut SingleBlockMessage,
                attempted_nonces: &mut u64,
                limit: u64,
                target: u64,
                mask: u64,
            ) -> Option<u64> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_single_block::<
                        DIGIT_WORD_IDX0,
                        DIGIT_WORD_IDX1_INCREMENT,
                        TYPE,
                        MUTATION_TYPE,
                    >(message, attempted_nonces, limit, target, mask)
                }
            }

            fn solve_double_block<const TYPE: u8>(
                message: &mut DoubleBlockMessage,
                attempted_nonces: &mut u64,
                limit: u64,
                target: u64,
                mask: u64,
            ) -> Option<(u64, [u32; 8])> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_double_block::<TYPE>(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }
            }

            fn solve_binary<
                const TYPE: u8,
                const FIRST_NONCE_WORD_IDX: usize,
                const NEED_SECOND_BLOCK: bool,
            >(
                attempted_nonces: &mut u64,
                limit: u64,
                prefix_state: [u32; 8],
                first_block: &[u32; 16],
                second_block_schedule: &[u32; 64],
                nonce_byte_offset: usize,
                nonce_byte_count: core::num::NonZeroU8,
                target: u64,
                mask: u64,
            ) -> Option<u64> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_binary::<TYPE, FIRST_NONCE_WORD_IDX, NEED_SECOND_BLOCK>(
                        attempted_nonces,
                        limit,
                        prefix_state,
                        first_block,
                        second_block_schedule,
                        nonce_byte_offset,
                        nonce_byte_count,
                        target,
                        mask,
                    )
                }
            }

            fn solve_goaway<const TYPE: u8>(
                attempted_nonces: &mut u64,
                limit: u64,
                challenge: [u32; 8],
                high_word: u32,
                target: u64,
                mask: u64,
            ) -> Option<u64> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_goaway::<TYPE>(
                        attempted_nonces,
                        limit,
                        challenge,
                        high_word,
                        target,
                        mask,
                    )
                }
            }

            fn solve_cerberus_decimal<
                const CENTER_WORD_IDX: usize,
                const LANE_ID_WORD_IDX: usize,
                const CONSTANT_WORD_COUNT: usize,
            >(
                message: &CerberusMessage,
                msg_tpl: [u32; 16],
                attempted_nonces: &mut u64,
                limit: u64,
                target: u64,
                mask: u64,
            ) -> Option<(u64, u64)> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_cerberus_decimal::<
                        CENTER_WORD_IDX,
                        LANE_ID_WORD_IDX,
                        CONSTANT_WORD_COUNT,
                    >(message, msg_tpl, attempted_nonces, limit, target, mask)
                }
            }

            fn solve_cerberus_binary(
                message: &CerberusMessage,
                attempted_nonces: &mut u64,
                limit: u64,
                target: u64,
                mask: u64,
            ) -> Option<u64> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_cerberus_binary(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }
            }

            fn solve_altcha<const TYPE: u8>(
                message: &AltchaMessage,
                attempted_nonces: &mut u64,
                limit: u64,
                target: u64,
                mask: u64,
            ) -> Option<(u64, [u32; 8])> {
                if !Self::available() {
                    return None;
                }
                #[expect(
                    unsafe_code,
                    reason = "runtime detection guards the target-feature thunk"
                )]
                // SAFETY: `Self::available()` above proves the target feature required by this thunk.
                unsafe {
                    target::$module::solve_altcha::<TYPE>(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }
            }
        }
    };
}

mod target {
    use super::{AltchaMessage, CerberusMessage, DoubleBlockMessage, SingleBlockMessage};

    macro_rules! target_module {
        ($module:ident, $feature:literal, $lanes:literal) => {
            pub(super) mod $module {
                use super::{AltchaMessage, CerberusMessage, DoubleBlockMessage, SingleBlockMessage};

                #[target_feature(enable = $feature)]
                pub fn solve_single_block<
                    const DIGIT_WORD_IDX0: usize,
                    const DIGIT_WORD_IDX1_INCREMENT: bool,
                    const TYPE: u8,
                    const MUTATION_TYPE: u8,
                >(
                    message: &mut SingleBlockMessage,
                    attempted_nonces: &mut u64,
                    limit: u64,
                    target: u64,
                    mask: u64,
                ) -> Option<u64> {
                    crate::solver::wide::solve_single_block_impl::<
                        $lanes,
                        DIGIT_WORD_IDX0,
                        DIGIT_WORD_IDX1_INCREMENT,
                        TYPE,
                        MUTATION_TYPE,
                    >(message, attempted_nonces, limit, target, mask)
                }

                #[target_feature(enable = $feature)]
                pub fn solve_double_block<const TYPE: u8>(
                    message: &mut DoubleBlockMessage,
                    attempted_nonces: &mut u64,
                    limit: u64,
                    target: u64,
                    mask: u64,
                ) -> Option<(u64, [u32; 8])> {
                    crate::solver::wide::solve_double_block_impl::<$lanes, TYPE>(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }

                #[target_feature(enable = $feature)]
                #[expect(
    clippy::too_many_arguments,
    reason = "backend contract mirrors the complete solver kernel inputs without packing hot-path state"
)]
                pub fn solve_binary<
                    const TYPE: u8,
                    const FIRST_NONCE_WORD_IDX: usize,
                    const NEED_SECOND_BLOCK: bool,
                >(
                    attempted_nonces: &mut u64,
                    limit: u64,
                    prefix_state: [u32; 8],
                    first_block: &[u32; 16],
                    second_block_schedule: &[u32; 64],
                    nonce_byte_offset: usize,
                    nonce_byte_count: core::num::NonZeroU8,
                    target: u64,
                    mask: u64,
                ) -> Option<u64> {
                    crate::solver::wide::solve_binary_impl::<
                        $lanes,
                        TYPE,
                        FIRST_NONCE_WORD_IDX,
                        NEED_SECOND_BLOCK,
                    >(
                        crate::solver::wide::BinaryWork {
                            attempted_nonces,
                            limit,
                            prefix_state,
                            first_block,
                            second_block_schedule,
                            nonce_byte_offset,
                            nonce_byte_count,
                        },
                        target,
                        mask,
                    )
                }

                #[target_feature(enable = $feature)]
                pub fn solve_goaway<const TYPE: u8>(
                    attempted_nonces: &mut u64,
                    limit: u64,
                    challenge: [u32; 8],
                    high_word: u32,
                    target: u64,
                    mask: u64,
                ) -> Option<u64> {
                    crate::solver::wide::solve_goaway_impl::<$lanes, TYPE>(
                        attempted_nonces,
                        limit,
                        challenge,
                        high_word,
                        target,
                        mask,
                    )
                }

                #[target_feature(enable = $feature)]
                pub fn solve_cerberus_decimal<
                    const CENTER_WORD_IDX: usize,
                    const LANE_ID_WORD_IDX: usize,
                    const CONSTANT_WORD_COUNT: usize,
                >(
                    message: &CerberusMessage,
                    msg_tpl: [u32; 16],
                    attempted_nonces: &mut u64,
                    limit: u64,
                    target: u64,
                    mask: u64,
                ) -> Option<(u64, u64)> {
                    crate::solver::wide::solve_cerberus_decimal_wide::<
                        $lanes,
                        CENTER_WORD_IDX,
                        LANE_ID_WORD_IDX,
                        CONSTANT_WORD_COUNT,
                    >(message, msg_tpl, attempted_nonces, limit, target, mask)
                }

                #[target_feature(enable = $feature)]
                pub fn solve_cerberus_binary(
                    message: &CerberusMessage,
                    attempted_nonces: &mut u64,
                    limit: u64,
                    target: u64,
                    mask: u64,
                ) -> Option<u64> {
                    crate::solver::wide::solve_cerberus_binary_wide::<$lanes>(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }

                #[target_feature(enable = $feature)]
                pub fn solve_altcha<const TYPE: u8>(
                    message: &AltchaMessage,
                    attempted_nonces: &mut u64,
                    limit: u64,
                    target: u64,
                    mask: u64,
                ) -> Option<(u64, [u32; 8])> {
                    crate::solver::wide::solve_altcha_impl::<$lanes, TYPE>(
                        message,
                        attempted_nonces,
                        limit,
                        target,
                        mask,
                    )
                }
            }
        };
    }

    target_module!(avx2, "avx2", 8);
    target_module!(avx512, "avx512f", 16);
}

backend_impl!(Avx2, avx2, 8, has_avx2, "AVX2");
backend_impl!(Avx512, avx512, 16, has_avx512f, "AVX-512");
