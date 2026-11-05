use core::simd::u32x4;

use crate::message::{DecimalMessage, DoubleBlockMessage, GoAwayMessage, SingleBlockMessage};

cpufeatures::new!(sse41sha, "sha", "sse4.1");

#[inline]
fn usize_u32(value: usize) -> u32 {
    let Ok(value) = u32::try_from(value) else {
        unreachable!("SHA-NI lane index exceeds u32")
    };
    value
}

#[inline]
const fn decimal_pair_ascii(input: u32) -> u32 {
    assert!(input < 100);
    let high_digit = (input / 10).to_le_bytes()[0];
    let low_digit = (input % 10).to_le_bytes()[0];
    u32::from_be_bytes([0, 0, high_digit + b'0', low_digit + b'0'])
}

struct SingleBlockLaneIdPlucker<
    'a,
    const DIGIT_WORD_IDX0_DIV_4: usize,
    const DIGIT_WORD_IDX0_MOD_4: usize,
    const DIGIT_WORD_IDX1_INC: bool,
> {
    lane_0_or_value: &'a [u32; 4],
    lane_1_or_value: &'a [u32; 4],
}

impl<
    const DIGIT_WORD_IDX0_DIV_4: usize,
    const DIGIT_WORD_IDX0_MOD_4: usize,
    const DIGIT_WORD_IDX1_INC: bool,
> SingleBlockLaneIdPlucker<'_, DIGIT_WORD_IDX0_DIV_4, DIGIT_WORD_IDX0_MOD_4, DIGIT_WORD_IDX1_INC>
{
    #[inline(always)]
    const fn fetch_msg_or(&self, idx: usize, lane: usize) -> u32 {
        if idx == DIGIT_WORD_IDX0_DIV_4 * 4 + DIGIT_WORD_IDX0_MOD_4 {
            self.lane_0_or_value[lane]
        } else if DIGIT_WORD_IDX1_INC
            && idx == DIGIT_WORD_IDX0_DIV_4 * 4 + DIGIT_WORD_IDX0_MOD_4 + 1
        {
            self.lane_1_or_value[lane]
        } else {
            0
        }
    }
}

impl<
    const DIGIT_WORD_IDX0_DIV_4: usize,
    const DIGIT_WORD_IDX0_MOD_4: usize,
    const DIGIT_WORD_IDX1_INC: bool,
> crate::sha256::sha_ni::Plucker
    for SingleBlockLaneIdPlucker<
        '_,
        DIGIT_WORD_IDX0_DIV_4,
        DIGIT_WORD_IDX0_MOD_4,
        DIGIT_WORD_IDX1_INC,
    >
{
    #[inline(always)]
    fn pluck_qword0(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w | u32x4::from_array(core::array::from_fn(|i| self.fetch_msg_or(i, lane)))
    }

    #[inline(always)]
    fn pluck_qword1(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w | u32x4::from_array(core::array::from_fn(|i| self.fetch_msg_or(i + 4, lane)))
    }

    #[inline(always)]
    fn pluck_qword2(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w | u32x4::from_array(core::array::from_fn(|i| self.fetch_msg_or(i + 8, lane)))
    }

    #[inline(always)]
    fn pluck_qword3(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w | u32x4::from_array(core::array::from_fn(|i| self.fetch_msg_or(i + 12, lane)))
    }
}

struct DoubleBlockLaneIdPlucker<'a> {
    lane_index_value_v: &'a [u32; 4],
}

impl crate::sha256::sha_ni::Plucker for DoubleBlockLaneIdPlucker<'_> {
    #[inline(always)]
    fn pluck_qword3(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w | u32x4::from_array([0, self.lane_index_value_v[lane], 0, 0])
    }
}

struct GoAwayLaneIdPlucker;

impl crate::sha256::sha_ni::Plucker for GoAwayLaneIdPlucker {
    #[inline(always)]
    fn pluck_qword2(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w + u32x4::from_array([0, usize_u32(lane), 0, 0])
    }
}

#[derive(Debug, Copy, Clone)]
/// Required features for SHA-NI solver.
pub struct RequiredFeatures;

impl Default for RequiredFeatures {
    fn default() -> Self {
        Self
    }
}

impl crate::solver::CpuIDToken for RequiredFeatures {
    fn get() -> bool {
        sse41sha::get()
    }
}

/// SHA-NI decimal nonce single block solver.
///
///
/// Current implementation: 4 way multi-issue with 4-round hotstart granularity.
pub struct SingleBlockSolver {
    message: SingleBlockMessage,

    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::SingleBlockSolver> for SingleBlockSolver {
    fn from(solver: super::safe::SingleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<SingleBlockMessage> for SingleBlockSolver {
    fn from(message: SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl crate::solver::Solver for SingleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        if self.message.no_trailing_zeros {
            self.solve_impl::<TYPE, true>(target, mask)
        } else {
            self.solve_impl::<TYPE, false>(target, mask)
        }
    }
}

impl SingleBlockSolver {
    #[expect(
        unsafe_code,
        reason = "runtime CPU feature detection proves the required target features before this audited boundary"
    )]
    fn solve_impl<const TYPE: u8, const NO_TRAILING_ZEROS: bool>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let lane_id_0_word_idx = self.message.digit_index / 4;

        let target = target & mask;
        for i in (self.message.digit_index..).take(9) {
            crate::write_be_word_byte(&mut self.message.message, i, b'0');
        }
        if NO_TRAILING_ZEROS {
            crate::write_be_word_byte(
                &mut self.message.message,
                self.message.digit_index + 8,
                b'1',
            );
        }

        let lane_id_1_word_idx = (self.message.digit_index + 1) / 4;

        macro_rules! dispatch {
            ($idx0_0:literal, $idx0_1:literal, $idx0_2:literal, $lane_id_1_word_idx_inc:literal) => {
                // SAFETY: Solver::solve checked SSE4.1 + SHA-NI before entering this dispatch.
                unsafe {
                    // Mirror the AVX-512 reduced-radix path for large Anubis-style searches:
                    // digits 0-7 are valid decimal and form a 3-bit alphabet, enabling one-bit Gray-code mutations.
                    let use_reduced_radix =
                        !NO_TRAILING_ZEROS && self.message.approx_working_set_count.get() >= 100;
                    if self.message.digit_index % 4 == 2 {
                        if use_reduced_radix {
                            self.solve_inner::<
                                                { $idx0_0 }, { $idx0_1 }, { $idx0_2 },
                                                { $lane_id_1_word_idx_inc }, TYPE,
                                                NO_TRAILING_ZEROS, true, true,
                                            >(target, mask)
                        } else {
                            self.solve_inner::<
                                                { $idx0_0 }, { $idx0_1 }, { $idx0_2 },
                                                { $lane_id_1_word_idx_inc }, TYPE,
                                                NO_TRAILING_ZEROS, true, false,
                                            >(target, mask)
                        }
                    } else if use_reduced_radix {
                        self.solve_inner::<
                                            { $idx0_0 }, { $idx0_1 }, { $idx0_2 },
                                            { $lane_id_1_word_idx_inc }, TYPE,
                                            NO_TRAILING_ZEROS, false, true,
                                        >(target, mask)
                    } else {
                        self.solve_inner::<
                                            { $idx0_0 }, { $idx0_1 }, { $idx0_2 },
                                            { $lane_id_1_word_idx_inc }, TYPE,
                                            NO_TRAILING_ZEROS, false, false,
                                        >(target, mask)
                    }
                }
            };
            ($idx0_0:literal, $idx0_1:literal, $idx0_2:literal) => {
                if lane_id_1_word_idx == lane_id_0_word_idx {
                    dispatch!($idx0_0, $idx0_1, $idx0_2, false)
                } else {
                    dispatch!($idx0_0, $idx0_1, $idx0_2, true)
                }
            };
        }

        let nonce = match lane_id_0_word_idx {
            0 => dispatch!(0, 0, 0),
            1 => dispatch!(0, 0, 1),
            2 => dispatch!(0, 0, 2),
            3 => dispatch!(0, 0, 3),
            4 => dispatch!(4, 1, 0),
            5 => dispatch!(4, 1, 1),
            6 => dispatch!(4, 1, 2),
            7 => dispatch!(4, 1, 3),
            8 => dispatch!(8, 2, 0),
            9 => dispatch!(8, 2, 1),
            10 => dispatch!(8, 2, 2),
            11 => dispatch!(8, 2, 3),
            12 => dispatch!(12, 3, 0),
            13 => dispatch!(12, 3, 1),
            _ => return None,
        }?;

        let mut final_sha_state = self.message.prefix_state;
        crate::sha256::digest_block(&mut final_sha_state, &self.message.message);

        Some((nonce + self.message.nonce_addend, final_sha_state))
    }

    #[inline(never)]
    #[target_feature(enable = "sse4.1,sha")]
    #[expect(
        clippy::too_many_lines,
        reason = "SHA-NI decimal hot loop keeps lane state, Gray-code mutation, and target comparison together for codegen"
    )]
    fn solve_inner<
        const DIGIT_WORD_IDX0_DIV_4_TIMES_4: usize,
        const DIGIT_WORD_IDX0_DIV_4: usize,
        const DIGIT_WORD_IDX0_MOD_4: usize,
        const DIGIT_WORD_IDX1_INC: bool,
        const TYPE: u8,
        const NO_TRAILING_ZEROS: bool,
        const ON_REGISTER_BOUNDARY: bool,
        const REDUCED_RADIX: bool,
    >(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        if REDUCED_RADIX {
            for i in (self.message.digit_index + 2..).take(7) {
                crate::write_be_word_byte(&mut self.message.message, i, b'0');
            }
        }

        let mut partial_state = self.message.prefix_state;
        crate::sha256::ingest_message_prefix::<{ DIGIT_WORD_IDX0_DIV_4_TIMES_4 }>(
            &mut partial_state,
            core::array::from_fn(|i| self.message.message[i]),
        );
        let prepared_state = crate::sha256::sha_ni::prepare_state(&partial_state);
        let lane_id_0_byte_idx = self.message.digit_index % 4;
        let lane_id_1_byte_idx = (self.message.digit_index + 1) % 4;
        let gray_ops: [(usize, u32); 21] = core::array::from_fn(|bit| {
            let group = bit / 3;
            let logical_digit = 6 - group;
            let byte_index = self.message.digit_index + logical_digit + 2;
            let word_index = byte_index / 4;
            let shift = (3 - byte_index % 4) * 8;
            (word_index, (1u32 << (bit % 3)) << shift)
        });

        // move AB into position for feedback
        let feedback_ab = crate::sha256::sha_ni::feedback_ab(
            self.message.prefix_state[0],
            self.message.prefix_state[1],
        );
        let feedback_state = crate::sha256::sha_ni::prepare_state(&self.message.prefix_state);

        let prefix_start = if REDUCED_RADIX { 0u32 } else { 10u32 };
        let prefix_end = if REDUCED_RADIX { 92u32 } else { 96u32 };
        for nonce_prefix_start in (prefix_start..=prefix_end).step_by(4) {
            let lane_index_values = [
                decimal_pair_ascii(nonce_prefix_start),
                decimal_pair_ascii(nonce_prefix_start + 1),
                decimal_pair_ascii(nonce_prefix_start + 2),
                decimal_pair_ascii(nonce_prefix_start + 3),
            ];

            let lane_id_1_or_value = core::array::from_fn(|i| {
                (lane_index_values[i] & 0xff) << ((3 - lane_id_1_byte_idx) * 8)
            });

            let lane_id_0_or_value = core::array::from_fn(|i| {
                let mut r = (lane_index_values[i] >> 8) << ((3 - lane_id_0_byte_idx) * 8);
                if !DIGIT_WORD_IDX1_INC {
                    r |= lane_id_1_or_value[i];
                }
                r
            });

            #[cfg(target_feature = "avx2")]
            let mut itoa_buf = *b"0000\x80000";

            let mut next_inner_key: u32 = if NO_TRAILING_ZEROS { 2 } else { 1 };
            let inner_iteration_end: u32 = if REDUCED_RADIX { 1 << 21 } else { 10_000_000 };
            while next_inner_key <= inner_iteration_end {
                let valid_lanes =
                    usize::try_from(self.limit.saturating_sub(self.attempted_nonces).min(4))
                        .ok()?;
                if valid_lanes == 0 {
                    return None;
                }

                let mut state0 = prepared_state;
                let mut state1 = prepared_state;
                let mut state2 = prepared_state;
                let mut state3 = prepared_state;

                crate::sha256::sha_ni::multiway_arx_abef_cdgh::<{ DIGIT_WORD_IDX0_DIV_4 }, 4, _>(
                    &mut [&mut state0, &mut state1, &mut state2, &mut state3],
                    &self.message.message,
                    SingleBlockLaneIdPlucker::<
                        DIGIT_WORD_IDX0_DIV_4,
                        DIGIT_WORD_IDX0_MOD_4,
                        DIGIT_WORD_IDX1_INC,
                    > {
                        lane_0_or_value: &lane_id_0_or_value,
                        lane_1_or_value: &lane_id_1_or_value,
                    },
                );

                // The normal predicates only need AB. Full-digest leading-zero
                // checks keep both SHA-NI state vectors live and feed forward all words.
                for state in [&mut state0, &mut state1, &mut state2, &mut state3] {
                    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                        for i in 0..2 {
                            state[i] =
                                crate::sha256::sha_ni::add_words(state[i], feedback_state[i]);
                        }
                    } else {
                        state[0] = crate::sha256::sha_ni::add_words(state[0], feedback_ab);
                    }
                }

                let success_lane_idx = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                    [state0, state1, state2, state3].iter().position(|state| {
                        let hash = crate::sha256::sha_ni::extract_state(*state);
                        crate::hash_has_leading_zero_bits(&hash, target)
                    })
                } else {
                    let cmp_fn = |x: &u64, y: &u64| {
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            x > y
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            x < y
                        } else {
                            x & mask == y & mask
                        }
                    };
                    let result_abs = [
                        crate::sha256::sha_ni::extract_ab(state0[0]),
                        crate::sha256::sha_ni::extract_ab(state1[0]),
                        crate::sha256::sha_ni::extract_ab(state2[0]),
                        crate::sha256::sha_ni::extract_ab(state3[0]),
                    ];
                    result_abs.iter().position(|x| cmp_fn(x, &target))
                };

                if let Some(success_lane_idx) = success_lane_idx.filter(|&lane| lane < valid_lanes)
                {
                    crate::unlikely();
                    self.attempted_nonces += u64::try_from(success_lane_idx + 1).ok()?;

                    let nonce_prefix = nonce_prefix_start + usize_u32(success_lane_idx);

                    // stamp the lane ID back onto the message
                    crate::write_be_word_byte(
                        &mut self.message.message,
                        self.message.digit_index,
                        (nonce_prefix / 10).to_le_bytes()[0] + b'0',
                    );
                    crate::write_be_word_byte(
                        &mut self.message.message,
                        self.message.digit_index + 1,
                        (nonce_prefix % 10).to_le_bytes()[0] + b'0',
                    );

                    let mut prev_inner_key = next_inner_key - 1;
                    let decimal_inner_key = if REDUCED_RADIX {
                        let mut value = 0u64;
                        let mut reduced_key = {
                            let ordinal = next_inner_key - 1;
                            ordinal ^ (ordinal >> 1)
                        };
                        for m in (0..7u32).map(|i| 10u64.pow(i)) {
                            let output = reduced_key % 8;
                            reduced_key /= 8;
                            value += u64::from(output) * m;
                        }
                        value
                    } else {
                        if NO_TRAILING_ZEROS && prev_inner_key % 10 == 0 {
                            prev_inner_key -= 1;
                        }
                        u64::from(prev_inner_key)
                    };

                    return Some(u64::from(nonce_prefix) * 10u64.pow(7) + decimal_inner_key);
                }

                if REDUCED_RADIX {
                    let step = next_inner_key;
                    let (word_index, xor_mask) =
                        gray_ops[usize::from(step.trailing_zeros().min(20).to_le_bytes()[0])];
                    self.message.message[word_index] ^= xor_mask;
                } else {
                    if NO_TRAILING_ZEROS && next_inner_key % 10 == 0 {
                        next_inner_key += 1;
                    }
                    #[cfg(target_feature = "avx2")]
                    {
                        crate::strings::format_decimal_8::<7, false, 0x80>(
                            &mut itoa_buf,
                            next_inner_key,
                        );
                        if ON_REGISTER_BOUNDARY {
                            let inner_word_idx =
                                DIGIT_WORD_IDX0_DIV_4 * 4 + DIGIT_WORD_IDX0_MOD_4 + 1;
                            self.message.message[inner_word_idx] = u32::from_be_bytes([
                                itoa_buf[0],
                                itoa_buf[1],
                                itoa_buf[2],
                                itoa_buf[3],
                            ]);
                            self.message.message[inner_word_idx + 1] = u32::from_be_bytes([
                                itoa_buf[4],
                                itoa_buf[5],
                                itoa_buf[6],
                                itoa_buf[7],
                            ]);
                        } else {
                            for i in 0..7 {
                                crate::write_be_word_byte(
                                    &mut self.message.message,
                                    self.message.digit_index + i + 2,
                                    itoa_buf[i],
                                );
                            }
                        }
                    }
                    #[cfg(not(target_feature = "avx2"))]
                    {
                        let mut key_copy = next_inner_key;
                        for i in (0..7).rev() {
                            let digit = key_copy % 10;
                            key_copy /= 10;
                            crate::write_be_word_byte(
                                &mut self.message.message,
                                self.message.digit_index + i + 2,
                                digit.to_le_bytes()[0] + b'0',
                            );
                        }
                    }
                }

                self.attempted_nonces += u64::try_from(valid_lanes).ok()?;
                if self.attempted_nonces >= self.limit {
                    return None;
                }
                next_inner_key += 1;
            }
        }
        None
    }
}

/// SHA-NI decimal nonce double block solver.
///
///
/// Current implementation: 4 way multi-issue with 4-round hotstart granularity.
pub struct DoubleBlockSolver {
    message: DoubleBlockMessage,
    attempted_nonces: u64,

    limit: u64,
}

impl From<super::safe::DoubleBlockSolver> for DoubleBlockSolver {
    fn from(solver: super::safe::DoubleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl crate::solver::Solver for DoubleBlockSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    #[expect(
        unsafe_code,
        reason = "runtime CPU feature detection proves the required target features before this audited boundary"
    )]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        // SAFETY: the feature check above proves SSE4.1 and SHA-NI are available.
        unsafe { self.solve_impl::<TYPE>(target, mask) }
    }
}

impl DoubleBlockSolver {
    #[target_feature(enable = "sse4.1,sha")]
    #[expect(
        clippy::too_many_lines,
        reason = "SHA-NI double-block hot loop keeps four lane states and terminal compression local for codegen"
    )]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if self.attempted_nonces >= self.limit {
            return None;
        }

        let target = target & mask;

        // The first two decimal digits are supplied by the SHA-NI lanes. Keep the remaining
        // seven digits directly in the message and advance them as an ASCII decimal odometer.
        // This preserves the full decimal search order while avoiding seven div/mod operations
        // and seven stores for every four candidate hashes.
        for logical_idx in (DoubleBlockMessage::DIGIT_IDX + 2)..63 {
            crate::write_be_word_byte(&mut self.message.message, logical_idx, b'0');
        }

        // Keep the candidate block alive across iterations. Only the seven mutable
        // ASCII digits change in the hot loop; rebuilding/copying the other 13 words
        // every four hashes is pure input-generation overhead.
        let mut msg0 = [0; 16];
        msg0[..13].copy_from_slice(&self.message.message[..13]);
        msg0[14] = self.message.message[14];
        msg0[15] = self.message.message[15];

        let iv_state = crate::sha256::sha_ni::prepare_state(&self.message.prefix_state);
        let mut prefix_state = self.message.prefix_state;
        crate::sha256::sha2_arx::<0>(&mut prefix_state, &self.message.message[..12]);
        let prepared_state = crate::sha256::sha_ni::prepare_state(&prefix_state);

        let mut terminal_message = [0; 16];
        let bit_length = self.message.message_length.checked_mul(8)?;
        terminal_message[14] = crate::high_u32(bit_length);
        terminal_message[15] = crate::low_u32(bit_length);

        for nonce_prefix_start in (10u32..=96).step_by(4) {
            let lane_index_value_v = [
                decimal_pair_ascii(nonce_prefix_start) | self.message.message[13],
                decimal_pair_ascii(nonce_prefix_start + 1) | self.message.message[13],
                decimal_pair_ascii(nonce_prefix_start + 2) | self.message.message[13],
                decimal_pair_ascii(nonce_prefix_start + 3) | self.message.message[13],
            ];

            for inner_key in 0..10_000_000u64 {
                let valid_lanes =
                    usize::try_from(self.limit.saturating_sub(self.attempted_nonces).min(4))
                        .ok()?;
                if valid_lanes == 0 {
                    return None;
                }

                let mut states0 = prepared_state;
                let mut states1 = prepared_state;
                let mut states2 = prepared_state;
                let mut states3 = prepared_state;

                crate::sha256::sha_ni::multiway_arx_abef_cdgh::<3, 4, DoubleBlockLaneIdPlucker>(
                    &mut [&mut states0, &mut states1, &mut states2, &mut states3],
                    &msg0,
                    DoubleBlockLaneIdPlucker {
                        lane_index_value_v: &lane_index_value_v,
                    },
                );

                for s in [&mut states0, &mut states1, &mut states2, &mut states3] {
                    s.iter_mut()
                        .zip(iv_state.iter())
                        .for_each(|(state, iv_state)| {
                            *state = crate::sha256::sha_ni::add_words(*state, *iv_state);
                        });
                }

                let saved_states = [states0, states1, states2, states3];
                let save_abs = [states0[0], states1[0], states2[0], states3[0]];

                // this isn't really SIMD so we can't really amortize the cost of fetching message schedule
                // so let's compute it with sha-ni
                crate::sha256::sha_ni::multiway_arx_abef_cdgh::<0, 4, _>(
                    &mut [&mut states0, &mut states1, &mut states2, &mut states3],
                    &terminal_message,
                    (),
                );

                if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                    for (state, saved) in [
                        (&mut states0, &saved_states[0]),
                        (&mut states1, &saved_states[1]),
                        (&mut states2, &saved_states[2]),
                        (&mut states3, &saved_states[3]),
                    ] {
                        for i in 0..2 {
                            state[i] = crate::sha256::sha_ni::add_words(state[i], saved[i]);
                        }
                    }
                } else {
                    states0[0] = crate::sha256::sha_ni::add_words(states0[0], save_abs[0]);
                    states1[0] = crate::sha256::sha_ni::add_words(states1[0], save_abs[1]);
                    states2[0] = crate::sha256::sha_ni::add_words(states2[0], save_abs[2]);
                    states3[0] = crate::sha256::sha_ni::add_words(states3[0], save_abs[3]);
                }

                let success_lane_idx = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                    [states0, states1, states2, states3]
                        .iter()
                        .position(|state| {
                            let hash = crate::sha256::sha_ni::extract_state(*state);
                            crate::hash_has_leading_zero_bits(&hash, target)
                        })
                } else {
                    let final_abs = [
                        crate::sha256::sha_ni::extract_ab(states0[0]),
                        crate::sha256::sha_ni::extract_ab(states1[0]),
                        crate::sha256::sha_ni::extract_ab(states2[0]),
                        crate::sha256::sha_ni::extract_ab(states3[0]),
                    ];
                    let cmp_fn = |x: &u64, y: &u64| {
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            x > y
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            x < y
                        } else {
                            x & mask == y & mask
                        }
                    };
                    final_abs.iter().position(|x| cmp_fn(x, &target))
                };

                if let Some(success_lane_idx) = success_lane_idx.filter(|&lane| lane < valid_lanes)
                {
                    crate::unlikely();
                    self.attempted_nonces += u64::try_from(success_lane_idx + 1).ok()?;

                    let nonce_prefix = nonce_prefix_start + usize_u32(success_lane_idx);
                    self.message.message[13] = lane_index_value_v[success_lane_idx];
                    self.message.message[14] = msg0[14];
                    self.message.message[15] = msg0[15];

                    // recompute the hash from the beginning
                    // this prevents the compiler from having to compute the final B-H registers alive in tight loops
                    let mut final_sha_state = self.message.prefix_state;
                    crate::sha256::digest_block(&mut final_sha_state, &self.message.message);
                    crate::sha256::digest_block(&mut final_sha_state, &terminal_message);

                    let computed_nonce = u64::from(nonce_prefix) * 10u64.pow(7)
                        + inner_key
                        + self.message.nonce_addend;

                    // the nonce is the 8 digits in the message, plus the first two digits recomputed from the lane index
                    return Some((computed_nonce, final_sha_state));
                }

                // Advance the seven mutable decimal bytes in-place. Carries are rare, so
                // this touches 10/9 bytes on average instead of rebuilding all seven digits.
                for logical_idx in ((DoubleBlockMessage::DIGIT_IDX + 2)..63).rev() {
                    let byte = crate::read_be_word_byte(&msg0, logical_idx);
                    if byte != b'9' {
                        crate::write_be_word_byte(&mut msg0, logical_idx, byte + 1);
                        break;
                    }
                    crate::write_be_word_byte(&mut msg0, logical_idx, b'0');
                }

                self.attempted_nonces += u64::try_from(valid_lanes).ok()?;

                if self.attempted_nonces >= self.limit {
                    return None;
                }
            }
        }
        crate::unlikely();

        None
    }
}

impl_decimal_solver!(
    [SingleBlockSolver, DoubleBlockSolver] => DecimalSolver
);

/// SHA-NI `GoAway` solver.
///
///
/// Current implementation: 4 way multi-issued solver with 4-round hotstart granularity.
pub struct GoAwaySolver {
    message: GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<super::safe::GoAwaySolver> for GoAwaySolver {
    fn from(solver: super::safe::GoAwaySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
        }
    }
}

impl From<GoAwayMessage> for GoAwaySolver {
    fn from(message: GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    const MSG_LEN: u32 = 10 * 4 * 8;

    #[cold]
    #[inline(never)]
    fn solve_scalar_tail<const TYPE: u8>(
        &mut self,
        mut low_word: u64,
        search_end: u64,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let mut message_block = [0; 16];
        message_block[0..8].copy_from_slice(&self.message.challenge);
        message_block[8] = self.message.high_word;
        message_block[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
        message_block[15] = Self::MSG_LEN as _;

        while low_word < search_end {
            let low_word_u32 = crate::low_u32(low_word);
            message_block[9] = low_word_u32;
            let mut final_sha_state = crate::sha256::IV;
            crate::sha256::digest_block(&mut final_sha_state, &message_block);
            self.attempted_nonces += 1;

            let state_ab = (u64::from(final_sha_state[0]) << 32) | u64::from(final_sha_state[1]);
            let hit = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                crate::hash_has_leading_zero_bits(&final_sha_state, target)
            } else if TYPE == crate::solver::SOLVE_TYPE_GT {
                state_ab > target
            } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                state_ab < target
            } else {
                state_ab & mask == target
            };
            if hit {
                crate::unlikely();
                return Some((
                    (u64::from(self.message.high_word) << 32) | u64::from(low_word_u32),
                    final_sha_state,
                ));
            }
            low_word += 1;
        }

        None
    }
}

impl crate::solver::Solver for GoAwaySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[expect(
        unsafe_code,
        reason = "runtime CPU feature detection proves the required target features before this audited boundary"
    )]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !<RequiredFeatures as crate::solver::CpuIDToken>::get() {
            return None;
        }
        // SAFETY: the feature check above proves SSE4.1 and SHA-NI are available.
        unsafe { self.solve_impl::<TYPE>(target, mask) }
    }
}

impl GoAwaySolver {
    #[target_feature(enable = "sse4.1,sha")]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let target = target & mask;

        let mut prefix_state = crate::sha256::IV;
        crate::sha256::ingest_message_prefix(&mut prefix_state, self.message.challenge);
        let prepared_state = crate::sha256::sha_ni::prepare_state(&prefix_state);

        let feedback_ab =
            crate::sha256::sha_ni::feedback_ab(crate::sha256::IV[0], crate::sha256::IV[1]);
        let feedback_state = crate::sha256::sha_ni::prepare_state(&crate::sha256::IV);

        let mut message_block = [0; 16];
        message_block[0..8].copy_from_slice(&self.message.challenge);
        message_block[8] = self.message.high_word;
        message_block[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
        message_block[15] = Self::MSG_LEN as _;

        macro_rules! try_full_batch {
            ($low_word_u32:expr) => {{
                let low_word_u32 = $low_word_u32;
                let mut states0 = prepared_state;
                let mut states1 = prepared_state;
                let mut states2 = prepared_state;
                let mut states3 = prepared_state;

                message_block[9] = low_word_u32;

                crate::sha256::sha_ni::multiway_arx_abef_cdgh::<2, 4, _>(
                    &mut [&mut states0, &mut states1, &mut states2, &mut states3],
                    &message_block,
                    GoAwayLaneIdPlucker,
                );

                for state in [&mut states0, &mut states1, &mut states2, &mut states3] {
                    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                        for i in 0..2 {
                            state[i] =
                                crate::sha256::sha_ni::add_words(state[i], feedback_state[i]);
                        }
                    } else {
                        state[0] = crate::sha256::sha_ni::add_words(state[0], feedback_ab);
                    }
                }

                let success_lane_idx = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                    [states0, states1, states2, states3]
                        .iter()
                        .position(|state| {
                            let hash = crate::sha256::sha_ni::extract_state(*state);
                            crate::hash_has_leading_zero_bits(&hash, target)
                        })
                } else {
                    let result_abs = [
                        crate::sha256::sha_ni::extract_ab(states0[0]),
                        crate::sha256::sha_ni::extract_ab(states1[0]),
                        crate::sha256::sha_ni::extract_ab(states2[0]),
                        crate::sha256::sha_ni::extract_ab(states3[0]),
                    ];
                    let cmp_fn = |x: &u64, y: &u64| {
                        if TYPE == crate::solver::SOLVE_TYPE_GT {
                            x > y
                        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                            x < y
                        } else {
                            x & mask == y & mask
                        }
                    };
                    result_abs.iter().position(|x| cmp_fn(x, &target))
                };

                self.attempted_nonces += 4;

                if let Some(success_lane_idx) = success_lane_idx {
                    crate::unlikely();

                    let mut output_msg: [u32; 16] = [0; 16];

                    let final_low_word = low_word_u32.wrapping_add(usize_u32(success_lane_idx));
                    output_msg[..8].copy_from_slice(&self.message.challenge);
                    output_msg[8] = self.message.high_word;
                    output_msg[9] = final_low_word;
                    output_msg[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
                    output_msg[15] = Self::MSG_LEN as _;

                    let mut final_sha_state = crate::sha256::IV;
                    crate::sha256::digest_block(&mut final_sha_state, &output_msg);

                    return Some((
                        (u64::from(self.message.high_word) << 32) | u64::from(final_low_word),
                        final_sha_state,
                    ));
                }
            }};
        }

        let search_end = self.limit.min(u64::from(u32::MAX) + 1);
        let mut low_word = self.attempted_nonces.min(search_end);
        let full_end = low_word + (search_end - low_word) / 4 * 4;

        while low_word < full_end {
            try_full_batch!(crate::low_u32(low_word));
            low_word += 4;
        }

        if low_word < search_end {
            return self.solve_scalar_tail::<TYPE>(low_word, search_end, target, mask);
        }

        crate::unlikely();
        None
    }
}

#[cfg(all(target_feature = "sse4.1", target_feature = "sha"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::Solver;

    #[test]
    fn test_goaway_vector_tail_attempt_limits() {
        let challenge = [b'a'; 32];
        let mut solver = GoAwaySolver::from(GoAwayMessage::new_bytes(&challenge, 0));
        solver.set_limit(1);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), 1);

        solver.set_limit(5);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), 5);

        let mut fresh_solver = GoAwaySolver::from(GoAwayMessage::new_bytes(&challenge, 0));
        fresh_solver.set_limit(6);
        assert!(
            fresh_solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(fresh_solver.attempted_nonces(), 6);

        fresh_solver.set_limit(9);
        assert!(
            fresh_solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(fresh_solver.attempted_nonces(), 9);
    }

    #[test]
    fn test_goaway_scalar_tail_returns_exact_digest() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};

        let mut challenge = [0_u8; 32];
        for (i, byte) in challenge.iter_mut().enumerate() {
            *byte = u8::try_from(i)?;
        }
        let nonce = 5u64;
        let mut hasher = Sha256::new();
        hasher.update(challenge);
        hasher.update(nonce.to_be_bytes());
        let hash = hasher.finalize();
        let target = u64::from_be_bytes(hash[..8].try_into()?);
        let expected = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([hash[base], hash[base + 1], hash[base + 2], hash[base + 3]])
        });

        let mut solver = GoAwaySolver::from(GoAwayMessage::new_bytes(&challenge, 0));
        solver.set_limit(nonce + 1);
        let (found_nonce, found_hash) = solver
            .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(target, u64::MAX)
            .ok_or_else(|| anyhow::anyhow!("tail candidate did not match exact target"))?;
        assert_eq!(found_nonce, nonce);
        assert_eq!(found_hash, expected);
        assert_eq!(solver.attempted_nonces(), nonce + 1);
        Ok(())
    }

    #[test]
    fn test_solve_decimal() -> anyhow::Result<()> {
        crate::solver::tests::test_decimal_validator::<DecimalSolver, _>(
            |prefix, search_space| {
                if let Some(solver) = SingleBlockMessage::new(prefix, search_space).map(Into::into)
                {
                    Some(DecimalSolver::SingleBlock(solver))
                } else {
                    DoubleBlockMessage::new(prefix, search_space).map(Into::into)
                }
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_word_aligned_decimal_mutation() -> Result<(), &'static str> {
        use sha2::{Digest as _, Sha256};

        // A 38-byte prefix puts the nine decimal nonce bytes at offset 38. The
        // seven inner digits therefore start exactly on a u32 word boundary,
        // exercising the specialized two-word update path.
        let prefix = [b'a'; 38];
        let message = SingleBlockMessage::new(&prefix, 0).ok_or("single-block message")?;
        assert_eq!(message.digit_index % 4, 2);
        assert!(message.approx_working_set_count.get() < 100);

        // Inner key 1 is the second four-lane batch. Target it exactly so the
        // test cannot pass unless the first failed batch mutates the message
        // bytes to the next decimal value correctly.
        let expected_nonce = 100_000_001u64;
        let mut input = prefix.to_vec();
        input.extend_from_slice(expected_nonce.to_string().as_bytes());
        let expected_digest = Sha256::digest(&input);
        let target = u64::from_be_bytes([
            expected_digest[0],
            expected_digest[1],
            expected_digest[2],
            expected_digest[3],
            expected_digest[4],
            expected_digest[5],
            expected_digest[6],
            expected_digest[7],
        ]);

        let mut solver = SingleBlockSolver::from(message);
        solver.set_limit(8);
        let (nonce, digest) = solver
            .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(target, u64::MAX)
            .ok_or("word-aligned decimal target not found")?;

        assert_eq!(nonce, expected_nonce);
        let expected_words: [u32; 8] = core::array::from_fn(|word| {
            u32::from_be_bytes([
                expected_digest[word * 4],
                expected_digest[word * 4 + 1],
                expected_digest[word * 4 + 2],
                expected_digest[word * 4 + 3],
            ])
        });
        assert_eq!(digest, expected_words);
        Ok(())
    }

    #[test]
    fn test_anubis_reduced_radix_nonce_path() -> anyhow::Result<()> {
        use core::num::NonZeroU8;
        use std::io::Write;

        use sha2::Digest;

        let prefix = [b'a'; 128];
        let message = SingleBlockMessage::new(&prefix, 0)
            .ok_or_else(|| anyhow::anyhow!("single-block message is invalid"))?;
        assert!(message.approx_working_set_count.get() >= 100);

        let mut solver = SingleBlockSolver::from(message);
        solver.set_limit(4_000_000);
        let mask = crate::compute_mask_zero_nibbles(
            NonZeroU8::new(4).ok_or_else(|| anyhow::anyhow!("difficulty is nonzero"))?,
        );
        let (nonce, hash) = solver
            .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, mask)
            .ok_or_else(|| anyhow::anyhow!("reduced-radix SHA-NI path found no Anubis solution"));

        let mut input = prefix.to_vec();
        write!(input, "{nonce}")?;
        let expected = sha2::Sha256::digest(&input);
        let actual: Vec<u8> = hash.into_iter().flat_map(u32::to_be_bytes).collect();
        assert_eq!(actual, expected.as_slice());
        assert_eq!(u64::from_be_bytes(actual[..8].try_into()?) & mask, 0);
        Ok(())
    }

    #[test]
    fn test_anubis_reduced_radix_wraps_cleanly() -> anyhow::Result<()> {
        let prefix = [b'a'; 128];
        let message = SingleBlockMessage::new(&prefix, 0)
            .ok_or_else(|| anyhow::anyhow!("single-block message is invalid"))?;
        assert!(message.approx_working_set_count.get() >= 100);

        let mut solver = SingleBlockSolver::from(message);
        const INNER_STATES: u64 = 1 << 21;
        solver.set_limit(INNER_STATES * 4);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), INNER_STATES * 4);

        for logical in (solver.message.digit_index + 2..).take(7) {
            assert_eq!(
                crate::read_be_word_byte(&solver.message.message, logical),
                b'0'
            );
        }
        Ok(())
    }

    #[test]
    fn test_solve_decimal_f64() -> anyhow::Result<()> {
        crate::solver::tests::test_decimal_validator_f64_safe::<DecimalSolver, _>(
            |prefix, search_space| {
                if let Some((solver, p)) =
                    SingleBlockMessage::new_f64(prefix, search_space).map(|(x, p)| (x.into(), p))
                {
                    Some((DecimalSolver::SingleBlock(solver), p))
                } else {
                    DoubleBlockMessage::new(prefix, search_space)
                        .map(|x| (DecimalSolver::DoubleBlock(x.into()), None))
                }
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_solve_goaway() -> anyhow::Result<()> {
        crate::solver::tests::test_goaway_validator::<GoAwaySolver, _>(|prefix| {
            GoAwaySolver::from(GoAwayMessage::new(
                core::array::from_fn(|i| {
                    u32::from_be_bytes([
                        prefix[i * 4],
                        prefix[i * 4 + 1],
                        prefix[i * 4 + 2],
                        prefix[i * 4 + 3],
                    ])
                }),
                0,
            ))
        })?;
        Ok(())
    }
}
