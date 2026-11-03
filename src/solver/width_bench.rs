//! Internal benchmark hooks for comparing portable-SIMD logical lane counts.

use crate::message::{GoAwayMessage, SingleBlockMessage};

const MUTATION_TYPE_UNALIGNED: u8 = 0;
const MUTATION_TYPE_ALIGNED: u8 = 1;
const MUTATION_TYPE_OCTAL: u8 = 2;
const MUTATION_TYPE_ALIGNED_OCTAL: u8 = MUTATION_TYPE_ALIGNED | MUTATION_TYPE_OCTAL;
const MUTATION_TYPE_UNALIGNED_OCTAL: u8 = MUTATION_TYPE_UNALIGNED | MUTATION_TYPE_OCTAL;

/// Run the shared single-block SHA-256 solver at a fixed portable-SIMD lane count.
#[inline(always)]
pub fn solve_single_block<const LANES: usize, const TYPE: u8>(
    message: &mut SingleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    let lane_id_0_word_idx = message.digit_index / 4;
    let lane_id_1_word_idx = (message.digit_index + 1) / 4;

    macro_rules! dispatch {
        ($idx0:literal, $idx1_inc:literal) => {{
            if message.digit_index % 4 == 2 {
                if message.no_trailing_zeros || message.approx_working_set_count.get() >= 100 {
                    super::wide::solve_single_block_impl::<
                        LANES,
                        $idx0,
                        $idx1_inc,
                        TYPE,
                        MUTATION_TYPE_ALIGNED_OCTAL,
                    >(message, attempted_nonces, limit, target, mask)
                } else {
                    super::wide::solve_single_block_impl::<
                        LANES,
                        $idx0,
                        $idx1_inc,
                        TYPE,
                        MUTATION_TYPE_ALIGNED,
                    >(message, attempted_nonces, limit, target, mask)
                }
            } else if message.no_trailing_zeros {
                super::wide::solve_single_block_impl::<
                    LANES,
                    $idx0,
                    $idx1_inc,
                    TYPE,
                    MUTATION_TYPE_UNALIGNED_OCTAL,
                >(message, attempted_nonces, limit, target, mask)
            } else {
                super::wide::solve_single_block_impl::<
                    LANES,
                    $idx0,
                    $idx1_inc,
                    TYPE,
                    MUTATION_TYPE_UNALIGNED,
                >(message, attempted_nonces, limit, target, mask)
            }
        }};
        ($idx0:literal) => {{
            if lane_id_0_word_idx == lane_id_1_word_idx {
                dispatch!($idx0, false)
            } else {
                dispatch!($idx0, true)
            }
        }};
    }

    match lane_id_0_word_idx {
        0 => dispatch!(0),
        1 => dispatch!(1),
        2 => dispatch!(2),
        3 => dispatch!(3),
        4 => dispatch!(4),
        5 => dispatch!(5),
        6 => dispatch!(6),
        7 => dispatch!(7),
        8 => dispatch!(8),
        9 => dispatch!(9),
        10 => dispatch!(10),
        11 => dispatch!(11),
        12 => dispatch!(12),
        13 => dispatch!(13),
        _ => None,
    }
}

/// Run the shared go-away SHA-256 solver at a fixed portable-SIMD lane count.
#[inline(always)]
pub fn solve_goaway<const LANES: usize, const TYPE: u8>(
    message: &GoAwayMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    super::wide::solve_goaway_impl::<LANES, TYPE>(
        attempted_nonces,
        limit,
        message.challenge,
        message.high_word,
        target,
        mask,
    )
}

#[target_feature(enable = "avx2")]
fn solve_single_block_avx2<const TYPE: u8>(
    message: &mut SingleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    solve_single_block::<8, TYPE>(message, attempted_nonces, limit, target, mask)
}

#[target_feature(enable = "avx512f")]
fn solve_single_block_avx512<const TYPE: u8>(
    message: &mut SingleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    solve_single_block::<16, TYPE>(message, attempted_nonces, limit, target, mask)
}

/// Runtime-select the natural x86 portable-SIMD width from one shared implementation.
#[expect(
    unsafe_code,
    reason = "runtime feature detection guards the target-feature thunks"
)]
pub fn solve_single_block_auto<const TYPE: u8>(
    message: &mut SingleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: runtime detection proves AVX-512F is present.
        unsafe { solve_single_block_avx512::<TYPE>(message, attempted_nonces, limit, target, mask) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection proves AVX2 is present.
        unsafe { solve_single_block_avx2::<TYPE>(message, attempted_nonces, limit, target, mask) }
    } else {
        solve_single_block::<4, TYPE>(message, attempted_nonces, limit, target, mask)
    }
}

#[target_feature(enable = "avx2")]
fn solve_goaway_avx2<const TYPE: u8>(
    message: &GoAwayMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    solve_goaway::<8, TYPE>(message, attempted_nonces, limit, target, mask)
}

#[target_feature(enable = "avx512f")]
fn solve_goaway_avx512<const TYPE: u8>(
    message: &GoAwayMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    solve_goaway::<16, TYPE>(message, attempted_nonces, limit, target, mask)
}

/// Runtime-select the natural x86 portable-SIMD width for go-away.
#[expect(
    unsafe_code,
    reason = "runtime feature detection guards the target-feature thunks"
)]
pub fn solve_goaway_auto<const TYPE: u8>(
    message: &GoAwayMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: runtime detection proves AVX-512F is present.
        unsafe { solve_goaway_avx512::<TYPE>(message, attempted_nonces, limit, target, mask) }
    } else if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime detection proves AVX2 is present.
        unsafe { solve_goaway_avx2::<TYPE>(message, attempted_nonces, limit, target, mask) }
    } else {
        solve_goaway::<4, TYPE>(message, attempted_nonces, limit, target, mask)
    }
}
