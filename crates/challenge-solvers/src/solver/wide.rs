use core::simd::{
    Simd,
    cmp::{SimdPartialEq as _, SimdPartialOrd as _},
};

#[inline]
fn usize_u32(value: usize) -> u32 {
    let Ok(value) = u32::try_from(value) else {
        unreachable!("SIMD/search index exceeds u32");
    };
    value
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn usize_u64(value: usize) -> u64 {
    let Ok(value) = u64::try_from(value) else {
        unreachable!("SIMD/search index exceeds u64");
    };
    value
}

#[inline(always)]
fn target_match<const LANES: usize, const TYPE: u8>(
    a: Simd<u32, LANES>,
    b: Simd<u32, LANES>,
    target: u64,
    mask: u64,
) -> core::simd::Mask<i32, LANES> {
    #[cfg(not(feature = "compare-64bit"))]
    let _ = b;

    let target = target & mask;
    let target_hi = crate::high_u32(target);
    let mask_hi = crate::high_u32(mask);

    #[cfg(not(feature = "compare-64bit"))]
    {
        if TYPE == crate::solver::SOLVE_TYPE_GT {
            a.simd_gt(Simd::splat(target_hi))
        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
            a.simd_lt(Simd::splat(target_hi))
        } else {
            (a & Simd::splat(mask_hi)).simd_eq(Simd::splat(target_hi & mask_hi))
        }
    }

    #[cfg(feature = "compare-64bit")]
    {
        let target_lo = crate::low_u32(target);
        let mask_lo = crate::low_u32(mask);
        let hi_eq = a.simd_eq(Simd::splat(target_hi));
        if TYPE == crate::solver::SOLVE_TYPE_GT {
            a.simd_gt(Simd::splat(target_hi)) | (hi_eq & b.simd_gt(Simd::splat(target_lo)))
        } else if TYPE == crate::solver::SOLVE_TYPE_LT {
            a.simd_lt(Simd::splat(target_hi)) | (hi_eq & b.simd_lt(Simd::splat(target_lo)))
        } else {
            (a & Simd::splat(mask_hi)).simd_eq(Simd::splat(target_hi & mask_hi))
                & (b & Simd::splat(mask_lo)).simd_eq(Simd::splat(target_lo & mask_lo))
        }
    }
}

#[inline(always)]
fn leading_zero_match<const LANES: usize>(
    state: &[Simd<u32, LANES>; 8],
    bits: u64,
) -> core::simd::Mask<i32, LANES> {
    if bits > 256 {
        return core::simd::Mask::splat(false);
    }

    let full_words = usize::from((bits / 32).to_le_bytes()[0]);
    let mut matched = core::simd::Mask::splat(true);
    for word in &state[..full_words] {
        matched &= word.simd_eq(Simd::splat(0));
    }

    let remaining = crate::low_u32(bits % 32);
    if remaining != 0 {
        let mask = u32::MAX << (32 - remaining);
        matched &= (state[full_words] & Simd::splat(mask)).simd_eq(Simd::splat(0));
    }
    matched
}

#[cfg(target_arch = "x86_64")]
pub struct BinaryWork<'a> {
    pub attempted_nonces: &'a mut u64,
    pub limit: u64,
    pub prefix_state: [u32; 8],
    pub first_block: &'a [u32; 16],
    pub second_block_schedule: &'a [u32; 64],
    pub nonce_byte_offset: usize,
    pub nonce_byte_count: core::num::NonZeroU8,
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn solve_binary_impl<
    const LANES: usize,
    const TYPE: u8,
    const FIRST_NONCE_WORD_IDX: usize,
    const NEED_SECOND_BLOCK: bool,
>(
    work: BinaryWork<'_>,
    target: u64,
    mask: u64,
) -> Option<u64> {
    let BinaryWork {
        attempted_nonces,
        limit,
        prefix_state,
        first_block,
        second_block_schedule,
        nonce_byte_offset,
        nonce_byte_count,
    } = work;
    let target = target & mask;
    let lane_shift = (3 - nonce_byte_offset % 4) * 8;
    let lane_ids =
        Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| usize_u32(lane) << lane_shift));

    let mut memo_state = prefix_state;
    crate::sha256::ingest_message_prefix::<FIRST_NONCE_WORD_IDX>(
        &mut memo_state,
        core::array::from_fn(|i| first_block[i]),
    );

    let candidate_space = 256u64.saturating_pow(u32::from(nonce_byte_count.get()));
    let search_end = limit.min(candidate_space);
    let mut cursor = (*attempted_nonces).min(search_end);
    while cursor < search_end {
        let x = cursor >> 8;
        let mut block_template = *first_block;
        let x_bytes = x.to_le_bytes();
        for (i, byte) in x_bytes
            .iter()
            .take(usize::from(nonce_byte_count.get()) - 1)
            .enumerate()
        {
            let byte_index = nonce_byte_offset + 1 + i;
            let shift = (3 - byte_index % 4) * 8;
            block_template[byte_index / 4] |= u32::from(*byte) << shift;
        }

        let mut lane_base = usize::from((cursor & 0xff).to_le_bytes()[0]);
        while lane_base < 256 && cursor < search_end {
            let valid_lanes = (search_end - cursor)
                .min(usize_u64(LANES))
                .min(usize_u64(256 - lane_base));
            let valid_lanes = usize::try_from(valid_lanes).ok()?;
            let lane_values = lane_ids + Simd::splat(usize_u32(lane_base) << lane_shift);

            let mut state = core::array::from_fn(|i| Simd::splat(memo_state[i]));
            let mut message: [Simd<u32, LANES>; 16] =
                core::array::from_fn(|i| Simd::splat(block_template[i]));
            message[FIRST_NONCE_WORD_IDX] |= lane_values;

            crate::sha256::portable::multiway_arx::<LANES, FIRST_NONCE_WORD_IDX>(
                &mut state,
                &mut message,
            );

            if NEED_SECOND_BLOCK {
                for i in 0..8 {
                    state[i] += Simd::splat(prefix_state[i]);
                }
                let save_a = state[0];
                #[cfg(feature = "compare-64bit")]
                let save_b = state[1];

                crate::sha256::portable::bcst_multiway_arx::<LANES, 0>(
                    &mut state,
                    second_block_schedule,
                );
                state[0] += save_a;
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += save_b;
                }
            } else {
                state[0] += Simd::splat(prefix_state[0]);
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += Simd::splat(prefix_state[1]);
                }
            }

            let met_target = target_match::<LANES, TYPE>(state[0], state[1], target, mask);

            cursor += u64::try_from(valid_lanes).ok()?;
            *attempted_nonces = cursor;

            if met_target.any() {
                crate::unlikely();
                let lanes = met_target.to_array();
                if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                    return Some((x << 8) | usize_u64(lane_base + success_lane));
                }
            }

            lane_base += valid_lanes;
        }
    }

    None
}

#[inline(always)]
pub fn solve_goaway_impl<const LANES: usize, const TYPE: u8>(
    attempted_nonces: &mut u64,
    limit: u64,
    challenge: [u32; 8],
    high_word: u32,
    target: u64,
    mask: u64,
) -> Option<u64> {
    const MSG_LEN: u32 = 10 * 4 * 8;

    let lane_ids = Simd::<u32, LANES>::from_array(core::array::from_fn(usize_u32));
    let mut prefix_state = crate::sha256::IV;
    crate::sha256::ingest_message_prefix(&mut prefix_state, challenge);
    let mut partial_state = prefix_state;
    crate::sha256::sha2_arx::<8>(&mut partial_state, &[high_word]);

    let search_end = limit.min(u64::from(u32::MAX) + 1);
    let lanes_u64 = u64::try_from(LANES).ok()?;
    let mut low_word = (*attempted_nonces).min(search_end);
    while low_word < search_end {
        let valid_lanes = usize::try_from((search_end - low_word).min(lanes_u64)).ok()?;
        let mut state = core::array::from_fn(|i| Simd::splat(partial_state[i]));
        let mut message = [
            Simd::splat(challenge[0]),
            Simd::splat(challenge[1]),
            Simd::splat(challenge[2]),
            Simd::splat(challenge[3]),
            Simd::splat(challenge[4]),
            Simd::splat(challenge[5]),
            Simd::splat(challenge[6]),
            Simd::splat(challenge[7]),
            Simd::splat(high_word),
            Simd::splat(crate::low_u32(low_word)) + lane_ids,
            Simd::splat(u32::from_be_bytes([0x80, 0, 0, 0])),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(0),
            Simd::splat(MSG_LEN),
        ];
        crate::sha256::portable::multiway_arx::<LANES, 9>(&mut state, &mut message);
        if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
            for (word, iv) in state.iter_mut().zip(crate::sha256::IV) {
                *word += Simd::splat(iv);
            }
        } else {
            state[0] += Simd::splat(crate::sha256::IV[0]);
            #[cfg(feature = "compare-64bit")]
            {
                state[1] += Simd::splat(crate::sha256::IV[1]);
            }
        }

        let met_target = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
            leading_zero_match(&state, target)
        } else {
            target_match::<LANES, TYPE>(state[0], state[1], target, mask)
        };
        low_word += u64::try_from(valid_lanes).ok()?;
        *attempted_nonces = low_word;

        if met_target.any() {
            crate::unlikely();
            let lanes = met_target.to_array();
            if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                let final_low_word = crate::low_u32(low_word - u64::try_from(valid_lanes).ok()?)
                    + usize_u32(success_lane);
                return Some((u64::from(high_word) << 32) | u64::from(final_low_word));
            }
        }
    }
    None
}

#[expect(
    clippy::too_many_lines,
    reason = "fixed-round SIMD SHA-256 kernel keeps hot-loop state and reconstruction adjacent for codegen and auditability"
)]
#[inline(always)]
pub fn solve_double_block_impl<const LANES: usize, const TYPE: u8>(
    message: &mut crate::message::DoubleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    if *attempted_nonces >= limit {
        return None;
    }

    for (ix, i) in (crate::message::DoubleBlockMessage::DIGIT_IDX..)
        .take(9)
        .enumerate()
    {
        crate::write_be_word_byte(&mut message.message, i, if ix >= 2 { b'1' } else { b'0' });
    }

    let mut partial_state = message.prefix_state;
    crate::sha256::sha2_arx::<0>(&mut partial_state, &message.message[..13]);

    let bit_length = message.message_length.checked_mul(8)?;
    let mut terminal_message_schedule = [0; 64];
    terminal_message_schedule[14] = crate::high_u32(bit_length);
    terminal_message_schedule[15] = crate::low_u32(bit_length);
    crate::sha256::do_message_schedule_k_w(&mut terminal_message_schedule);

    let mut itoa_buf = *b"1111\x80111";
    // Preserve the historical wide-backend search bounds exactly: the AVX2
    // path covers 11x8 prefixes and AVX-512 covers 5x16 prefixes.
    let prefix_set_count = 90 / LANES;
    for prefix_set_index in 0..prefix_set_count {
        let lane_index_value = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| {
            let prefix = 10 + prefix_set_index * LANES + lane;
            let tens = usize_u32(prefix / 10) + u32::from(b'0');
            let ones = usize_u32(prefix % 10) + u32::from(b'0');
            message.message[13] | (tens << 8) | ones
        }));

        for next_inner_key in 1..=0o10_000_000u32 {
            let cum0 = u32::from_ne_bytes([itoa_buf[0], itoa_buf[1], itoa_buf[2], itoa_buf[3]]);
            let cum1 = u32::from_ne_bytes([itoa_buf[4], itoa_buf[5], itoa_buf[6], itoa_buf[7]]);

            let mut state = core::array::from_fn(|i| Simd::splat(partial_state[i]));
            let mut blocks: [Simd<u32, LANES>; 16] = core::array::from_fn(|i| {
                if i == 13 {
                    lane_index_value
                } else if i == 14 {
                    Simd::splat(cum0)
                } else if i == 15 {
                    Simd::splat(cum1)
                } else {
                    Simd::splat(message.message[i])
                }
            });

            crate::sha256::portable::multiway_arx::<LANES, 13>(&mut state, &mut blocks);
            for i in 0..8 {
                state[i] += Simd::splat(message.prefix_state[i]);
            }

            let feedforward = state;
            crate::sha256::portable::bcst_multiway_arx::<LANES, 14>(
                &mut state,
                &terminal_message_schedule,
            );
            if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                for (word, saved) in state.iter_mut().zip(feedforward) {
                    *word += saved;
                }
            } else {
                state[0] += feedforward[0];
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += feedforward[1];
                }
            }

            let met_target = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                leading_zero_match(&state, target)
            } else {
                target_match::<LANES, TYPE>(state[0], state[1], target, mask)
            };
            let valid_lanes = usize::try_from(
                limit
                    .saturating_sub(*attempted_nonces)
                    .min(u64::try_from(LANES).ok()?),
            )
            .ok()?;
            if valid_lanes == 0 {
                return None;
            }
            if met_target.any() {
                crate::unlikely();
                let lanes = met_target.to_array();
                if let Some(success_lane_idx) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                    *attempted_nonces += u64::try_from(success_lane_idx + 1).ok()?;
                    let nonce_prefix = 10 + LANES * prefix_set_index + success_lane_idx;
                    message.message[14] = cum0;
                    message.message[15] = cum1;
                    let digit_idx = crate::message::DoubleBlockMessage::DIGIT_IDX;
                    crate::write_be_word_byte(
                        &mut message.message,
                        digit_idx,
                        (nonce_prefix / 10).to_le_bytes()[0] + b'0',
                    );
                    crate::write_be_word_byte(
                        &mut message.message,
                        digit_idx + 1,
                        (nonce_prefix % 10).to_le_bytes()[0] + b'0',
                    );

                    let mut final_sha_state = message.prefix_state;
                    crate::sha256::digest_block(&mut final_sha_state, &message.message);
                    let mut terminal_message = [0; 16];
                    terminal_message[14] = crate::high_u32(bit_length);
                    terminal_message[15] = crate::low_u32(bit_length);
                    crate::sha256::digest_block(&mut final_sha_state, &terminal_message);

                    let mut decimal_inner_key = 0u64;
                    let mut key_octal = next_inner_key - 1;
                    for place in (0..7u32).map(|i| 10u64.pow(i)) {
                        let output = (key_octal % 8) + 1;
                        key_octal /= 8;
                        decimal_inner_key += u64::from(output) * place;
                    }
                    let computed_nonce = u64::try_from(nonce_prefix).ok()? * 10u64.pow(7)
                        + decimal_inner_key
                        + message.nonce_addend;
                    return Some((computed_nonce, final_sha_state));
                }
            }

            *attempted_nonces += u64::try_from(valid_lanes).ok()?;
            if *attempted_nonces >= limit {
                return None;
            }
            crate::strings::to_octal_7::<true, 0x80, 1>(&mut itoa_buf, next_inner_key);
        }
    }

    crate::unlikely();
    None
}

#[expect(
    clippy::too_many_lines,
    reason = "fixed-round SIMD SHA-256 kernel keeps compile-time layout cases in one hot loop for codegen"
)]
#[inline(always)]
pub fn solve_single_block_impl<
    const LANES: usize,
    const DIGIT_WORD_IDX0: usize,
    const DIGIT_WORD_IDX1_INCREMENT: bool,
    const TYPE: u8,
    const MUTATION_TYPE: u8,
>(
    message: &mut crate::message::SingleBlockMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    const MUTATION_TYPE_ALIGNED: u8 = 1;
    const MUTATION_TYPE_OCTAL: u8 = 2;

    let aligned = MUTATION_TYPE & MUTATION_TYPE_ALIGNED != 0;
    let octal = MUTATION_TYPE & MUTATION_TYPE_OCTAL != 0;

    let mut partial_state = message.prefix_state;
    crate::sha256::ingest_message_prefix::<DIGIT_WORD_IDX0>(
        &mut partial_state,
        core::array::from_fn(|i| message.message[i]),
    );

    // Reset the nine nonce bytes so a reused solver cannot inherit a previous hit.
    for (ix, i) in (message.digit_index..).take(9).enumerate() {
        crate::write_be_word_byte(
            &mut message.message,
            i,
            if ix >= 2 && octal { b'1' } else { b'0' },
        );
    }

    let lane_id_0_byte_idx = message.digit_index % 4;
    let lane_id_1_byte_idx = (message.digit_index + 1) % 4;
    let lane_id_0_shift = (3 - lane_id_0_byte_idx) * 8;
    let lane_id_1_shift = (3 - lane_id_1_byte_idx) * 8;

    // Preserve the historical backend search ranges exactly. AVX2 searched
    // decimal-like prefixes 10..=97 for both mutation alphabets; AVX-512
    // searched 10..=89 for decimal and 00..=95 for octal.
    let (prefix_start, prefix_set_count) = if LANES == 16 {
        if octal { (0usize, 6usize) } else { (10, 5) }
    } else {
        (10, 90 / LANES)
    };

    for prefix_set_index in 0..prefix_set_count {
        if *attempted_nonces >= limit {
            return None;
        }

        let prefix_base = prefix_start + prefix_set_index * LANES;
        let lane_id_0_or_value = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| {
            let prefix = prefix_base + lane;
            (usize_u32(prefix / 10) + u32::from(b'0')) << lane_id_0_shift
        }));
        let lane_id_1_or_value = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| {
            let prefix = prefix_base + lane;
            (usize_u32(prefix % 10) + u32::from(b'0')) << lane_id_1_shift
        }));
        let lane_id_0_or_value_v = if DIGIT_WORD_IDX1_INCREMENT {
            lane_id_0_or_value
        } else {
            lane_id_0_or_value | lane_id_1_or_value
        };

        let mut inner_key_buf = if octal {
            *b"1111\x80111"
        } else {
            *b"0000\x80000"
        };
        let max_inner_iterations = if octal {
            0o10_000_000u64
        } else {
            10_000_000u64
        };
        let inner_iteration_end = limit
            .saturating_sub(*attempted_nonces)
            .div_ceil(u64::try_from(LANES).ok()?)
            .min(max_inner_iterations);
        let inner_iteration_end = u32::try_from(inner_iteration_end).ok()?;

        for next_inner_key in 1..=inner_iteration_end {
            let inner_word_0 = u32::from_ne_bytes([
                inner_key_buf[0],
                inner_key_buf[1],
                inner_key_buf[2],
                inner_key_buf[3],
            ]);
            let inner_word_1 = u32::from_ne_bytes([
                inner_key_buf[4],
                inner_key_buf[5],
                inner_key_buf[6],
                inner_key_buf[7],
            ]);

            let mut blocks: [Simd<u32, LANES>; 16] = core::array::from_fn(|idx| {
                if idx == DIGIT_WORD_IDX0 {
                    Simd::splat(message.message[idx]) | lane_id_0_or_value_v
                } else if DIGIT_WORD_IDX1_INCREMENT && idx == DIGIT_WORD_IDX0 + 1 {
                    Simd::splat(message.message[idx]) | lane_id_1_or_value
                } else if aligned && idx == DIGIT_WORD_IDX0 + 1 {
                    Simd::splat(inner_word_0)
                } else if aligned && idx == DIGIT_WORD_IDX0 + 2 {
                    Simd::splat(inner_word_1)
                } else {
                    Simd::splat(message.message[idx])
                }
            });
            let mut state = core::array::from_fn(|i| Simd::splat(partial_state[i]));
            crate::sha256::portable::multiway_arx::<LANES, DIGIT_WORD_IDX0>(
                &mut state,
                &mut blocks,
            );
            if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                for (word, saved) in state.iter_mut().zip(message.prefix_state) {
                    *word += Simd::splat(saved);
                }
            } else {
                state[0] += Simd::splat(message.prefix_state[0]);
                #[cfg(feature = "compare-64bit")]
                {
                    state[1] += Simd::splat(message.prefix_state[1]);
                }
            }

            let met_target = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                leading_zero_match(&state, target)
            } else {
                target_match::<LANES, TYPE>(state[0], state[1], target, mask)
            };
            let valid_lanes = usize::try_from(
                limit
                    .saturating_sub(*attempted_nonces)
                    .min(u64::try_from(LANES).ok()?),
            )
            .ok()?;

            if met_target.any() {
                let lanes = met_target.to_array();
                if let Some(success_lane_idx) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                    crate::unlikely();
                    *attempted_nonces += u64::try_from(success_lane_idx + 1).ok()?;
                    let nonce_prefix = prefix_base + success_lane_idx;

                    if aligned {
                        message.message[DIGIT_WORD_IDX0 + 1] = inner_word_0;
                        message.message[DIGIT_WORD_IDX0 + 2] = inner_word_1;
                    }
                    crate::write_be_word_byte(
                        &mut message.message,
                        message.digit_index,
                        (nonce_prefix / 10).to_le_bytes()[0] + b'0',
                    );
                    crate::write_be_word_byte(
                        &mut message.message,
                        message.digit_index + 1,
                        (nonce_prefix % 10).to_le_bytes()[0] + b'0',
                    );

                    let decimal_inner_key = if octal {
                        let mut decimal = 0u64;
                        let mut key_octal = next_inner_key - 1;
                        for place in (0..7u32).map(|i| 10u64.pow(i)) {
                            let output = (key_octal % 8) + 1;
                            key_octal /= 8;
                            decimal += u64::from(output) * place;
                        }
                        decimal
                    } else {
                        u64::from(next_inner_key - 1)
                    };
                    return Some(
                        u64::try_from(nonce_prefix).ok()? * 10u64.pow(7) + decimal_inner_key,
                    );
                }
            }

            *attempted_nonces += u64::try_from(valid_lanes).ok()?;
            if *attempted_nonces >= limit {
                return None;
            }

            if aligned && octal {
                crate::strings::to_octal_7::<true, 0x80, 1>(&mut inner_key_buf, next_inner_key);
            } else if aligned {
                crate::strings::format_decimal_8::<7, true, 0x80>(
                    &mut inner_key_buf,
                    next_inner_key,
                );
            } else {
                let mut key_copy = next_inner_key;
                for i in (0..7).rev() {
                    let radix = if octal { 8 } else { 10 };
                    let output = key_copy % radix;
                    key_copy /= radix;
                    crate::write_be_word_byte(
                        &mut message.message,
                        message.digit_index + i + 2,
                        output.to_le_bytes()[0] + if octal { b'1' } else { b'0' },
                    );
                }
            }
        }
    }

    crate::unlikely();
    None
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
const fn altcha_be_word(bytes: &[u8; 16], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn altcha_compare_mask<const TYPE: u8>(mask: u64) -> u64 {
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        mask
    } else {
        u64::MAX
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn solve_altcha_nested_impl<const LANES: usize, const TYPE: u8, const KEY_WORDS: usize>(
    message: &crate::message::AltchaMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    const PAD_WORD: u32 = 0x8000_0000;

    let salt: [u32; 4] = core::array::from_fn(|i| altcha_be_word(&message.salt, i));
    let nonce: [u32; 4] = core::array::from_fn(|i| altcha_be_word(&message.nonce, i));
    let zero = Simd::<u32, LANES>::splat(0);
    let pad = Simd::<u32, LANES>::splat(PAD_WORD);
    let search_end = limit.min(u64::from(u32::MAX) + 1);
    let mut cursor = (*attempted_nonces).min(search_end);

    while cursor < search_end {
        let batch_start = cursor;
        let valid_lanes = usize::try_from((search_end - cursor).min(usize_u64(LANES))).ok()?;
        let counters = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| {
            crate::low_u32(batch_start + usize_u64(lane))
        }));
        let mut blocks = [
            Simd::splat(salt[0]),
            Simd::splat(salt[1]),
            Simd::splat(salt[2]),
            Simd::splat(salt[3]),
            Simd::splat(nonce[0]),
            Simd::splat(nonce[1]),
            Simd::splat(nonce[2]),
            Simd::splat(nonce[3]),
            counters,
            pad,
            zero,
            zero,
            zero,
            zero,
            zero,
            Simd::splat((32 + 4) * 8),
        ];

        for round in 0..message.cost.get() {
            let mut state =
                core::array::from_fn(|i| Simd::<u32, LANES>::splat(crate::sha256::IV[i]));
            crate::sha256::portable::multiway_arx::<LANES, 0>(&mut state, &mut blocks);
            let digest: [Simd<u32, LANES>; 8] =
                core::array::from_fn(|i| state[i] + Simd::splat(crate::sha256::IV[i]));

            if round + 1 == message.cost.get() {
                blocks[..8].copy_from_slice(&digest);
            } else {
                blocks = core::array::from_fn(|word| {
                    if word < KEY_WORDS {
                        digest[word]
                    } else if word == KEY_WORDS {
                        pad
                    } else if word == 15 {
                        Simd::splat(usize_u32(KEY_WORDS * 4 * 8))
                    } else {
                        zero
                    }
                });
            }
        }

        let met_target = target_match::<LANES, TYPE>(
            blocks[0],
            blocks[1],
            target,
            altcha_compare_mask::<TYPE>(mask),
        );
        cursor += u64::try_from(valid_lanes).ok()?;
        *attempted_nonces = cursor;

        if met_target.any() {
            crate::unlikely();
            let lanes = met_target.to_array();
            if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                return Some((
                    batch_start + u64::try_from(success_lane).ok()?,
                    core::array::from_fn(|word| blocks[word][success_lane]),
                ));
            }
        }
    }

    None
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
#[expect(
    clippy::similar_names,
    reason = "HMAC terminology conventionally names the inner and outer pad midstates ipad and opad"
)]
#[expect(
    clippy::too_many_lines,
    reason = "PBKDF2 SIMD hot loop keeps ipad/opad midstates and digest chaining local to avoid per-round state plumbing"
)]
fn solve_altcha_pbkdf2_impl<const LANES: usize, const TYPE: u8>(
    message: &crate::message::AltchaMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    const PAD_WORD: u32 = 0x8000_0000;

    let salt: [u32; 4] = core::array::from_fn(|i| altcha_be_word(&message.salt, i));
    let nonce: [u32; 4] = core::array::from_fn(|i| altcha_be_word(&message.nonce, i));
    let zero = Simd::<u32, LANES>::splat(0);
    let pad = Simd::<u32, LANES>::splat(PAD_WORD);
    let search_end = limit.min(u64::from(u32::MAX) + 1);
    let mut cursor = (*attempted_nonces).min(search_end);

    while cursor < search_end {
        let batch_start = cursor;
        let valid_lanes = usize::try_from((search_end - cursor).min(usize_u64(LANES))).ok()?;
        let counters = Simd::<u32, LANES>::from_array(core::array::from_fn(|lane| {
            crate::low_u32(batch_start + usize_u64(lane))
        }));

        let mut ipad_blocks = [Simd::<u32, LANES>::splat(crate::solver::HMAC_IPAD32); 16];
        let mut opad_blocks = [Simd::<u32, LANES>::splat(crate::solver::HMAC_OPAD32); 16];
        for i in 0..4 {
            ipad_blocks[i] ^= Simd::splat(nonce[i]);
            opad_blocks[i] ^= Simd::splat(nonce[i]);
        }
        ipad_blocks[4] ^= counters;
        opad_blocks[4] ^= counters;

        let mut midstate_ipad =
            core::array::from_fn(|i| Simd::<u32, LANES>::splat(crate::sha256::IV[i]));
        crate::sha256::portable::multiway_arx::<LANES, 0>(&mut midstate_ipad, &mut ipad_blocks);
        let mut midstate_opad =
            core::array::from_fn(|i| Simd::<u32, LANES>::splat(crate::sha256::IV[i]));
        crate::sha256::portable::multiway_arx::<LANES, 0>(&mut midstate_opad, &mut opad_blocks);
        for i in 0..8 {
            midstate_ipad[i] += Simd::splat(crate::sha256::IV[i]);
            midstate_opad[i] += Simd::splat(crate::sha256::IV[i]);
        }

        let mut blocks = [
            Simd::splat(salt[0]),
            Simd::splat(salt[1]),
            Simd::splat(salt[2]),
            Simd::splat(salt[3]),
            Simd::splat(1),
            pad,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            Simd::splat(512 + ((16 + 4) * 8)),
        ];
        let mut result = [zero; 8];

        for _ in 0..message.cost.get() {
            let mut state = midstate_ipad;
            crate::sha256::portable::multiway_arx::<LANES, 0>(&mut state, &mut blocks);
            blocks = [
                state[0] + midstate_ipad[0],
                state[1] + midstate_ipad[1],
                state[2] + midstate_ipad[2],
                state[3] + midstate_ipad[3],
                state[4] + midstate_ipad[4],
                state[5] + midstate_ipad[5],
                state[6] + midstate_ipad[6],
                state[7] + midstate_ipad[7],
                pad,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                Simd::splat(512 + 256),
            ];

            let mut state = midstate_opad;
            crate::sha256::portable::multiway_arx::<LANES, 0>(&mut state, &mut blocks);
            blocks = [
                state[0] + midstate_opad[0],
                state[1] + midstate_opad[1],
                state[2] + midstate_opad[2],
                state[3] + midstate_opad[3],
                state[4] + midstate_opad[4],
                state[5] + midstate_opad[5],
                state[6] + midstate_opad[6],
                state[7] + midstate_opad[7],
                pad,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                Simd::splat(512 + 256),
            ];
            for i in 0..8 {
                result[i] ^= blocks[i];
            }
        }

        let met_target = target_match::<LANES, TYPE>(
            result[0],
            result[1],
            target,
            altcha_compare_mask::<TYPE>(mask),
        );
        cursor += u64::try_from(valid_lanes).ok()?;
        *attempted_nonces = cursor;

        if met_target.any() {
            crate::unlikely();
            let lanes = met_target.to_array();
            if let Some(success_lane) = (0..valid_lanes).find(|&lane| lanes[lane]) {
                return Some((
                    batch_start + u64::try_from(success_lane).ok()?,
                    core::array::from_fn(|word| result[word][success_lane]),
                ));
            }
        }
    }

    None
}

#[inline]
#[cfg(target_arch = "x86_64")]
pub fn solve_altcha_fallback<const TYPE: u8>(
    message: &crate::message::AltchaMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    use crate::solver::Solver as _;

    let mut fallback = crate::solver::safe::AltchaSha256Solver::from(message.clone());
    fallback.attempted_nonces = *attempted_nonces;
    fallback.limit = limit;
    let result = fallback.solve::<TYPE>(target, mask);
    *attempted_nonces = fallback.attempted_nonces;
    result
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn solve_altcha_impl<const LANES: usize, const TYPE: u8>(
    message: &crate::message::AltchaMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, [u32; 8])> {
    if message.pbkdf2 {
        solve_altcha_pbkdf2_impl::<LANES, TYPE>(message, attempted_nonces, limit, target, mask)
    } else {
        match message.key_length.get().min(32) {
            4 => solve_altcha_nested_impl::<LANES, TYPE, 1>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            8 => solve_altcha_nested_impl::<LANES, TYPE, 2>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            12 => solve_altcha_nested_impl::<LANES, TYPE, 3>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            16 => solve_altcha_nested_impl::<LANES, TYPE, 4>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            20 => solve_altcha_nested_impl::<LANES, TYPE, 5>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            24 => solve_altcha_nested_impl::<LANES, TYPE, 6>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            28 => solve_altcha_nested_impl::<LANES, TYPE, 7>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            32 => solve_altcha_nested_impl::<LANES, TYPE, 8>(
                message,
                attempted_nonces,
                limit,
                target,
                mask,
            ),
            _ => solve_altcha_fallback::<TYPE>(message, attempted_nonces, limit, target, mask),
        }
    }
}

trait CerberusCompression<const LANES: usize> {
    fn compress<const CONSTANT_WORD_COUNT: usize, const PATCH: usize>(
        state: &mut [Simd<u32, LANES>; 16],
        template: &[u32; 16],
        patch: Simd<u32, LANES>,
    );
}

struct PortableCerberusCompression;

impl<const LANES: usize> CerberusCompression<LANES> for PortableCerberusCompression {
    #[inline(always)]
    fn compress<const CONSTANT_WORD_COUNT: usize, const PATCH: usize>(
        state: &mut [Simd<u32, LANES>; 16],
        template: &[u32; 16],
        patch: Simd<u32, LANES>,
    ) {
        crate::blake3::portable::compress::<LANES, CONSTANT_WORD_COUNT, PATCH>(
            state, template, patch,
        );
    }
}

#[inline(always)]
fn cerberus_lane_words<const LANES: usize>(base: usize) -> Simd<u32, LANES> {
    Simd::from_array(core::array::from_fn(|lane| {
        let mut value = base + lane;
        let ones = (value % 10).to_le_bytes()[0] + b'0';
        value /= 10;
        let tens = (value % 10).to_le_bytes()[0] + b'0';
        value /= 10;
        let hundreds = (value % 10).to_le_bytes()[0] + b'0';
        u32::from_be_bytes([ones, tens, hundreds, 0])
    }))
}

#[inline(always)]
fn first_zero_mask_lane<const LANES: usize>(
    word: Simd<u32, LANES>,
    mask: u32,
    valid_lanes: usize,
) -> Option<usize> {
    let hits = (word & Simd::splat(mask))
        .simd_eq(Simd::splat(0))
        .to_bitmask();
    let valid_mask = if valid_lanes == 64 {
        u64::MAX
    } else {
        (1u64 << valid_lanes) - 1
    };
    let hits = hits & valid_mask;
    if hits == 0 {
        None
    } else {
        usize::try_from(hits.trailing_zeros()).ok()
    }
}

fn solve_cerberus_decimal_impl<
    const LANES: usize,
    const CENTER_WORD_IDX: usize,
    const LANE_ID_WORD_IDX: usize,
    const CONSTANT_WORD_COUNT: usize,
    B: CerberusCompression<LANES>,
>(
    message: &crate::message::CerberusMessage,
    msg_tpl: [u32; 16],
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, u64)> {
    debug_assert_eq!(target, 0);

    let crate::message::CerberusMessage::Decimal(message) = message else {
        return None;
    };

    let mut msg = [0u32; 16];
    msg[..=CENTER_WORD_IDX + 1].copy_from_slice(&msg_tpl[..=CENTER_WORD_IDX + 1]);
    let prepared_state = crate::blake3::ingest_message_prefix(
        message.prefix_state,
        &msg[..CONSTANT_WORD_COUNT],
        0,
        u32::try_from(message.salt_residual_len).ok()? + 9,
        message.flags,
    );
    let state_base: [Simd<u32, LANES>; 16] =
        core::array::from_fn(|i| Simd::splat(prepared_state[i]));
    let mask_hi = crate::high_u32(mask);
    let lane_group_count = 1000 / LANES;
    let lanes_u64 = u64::try_from(LANES).ok()?;
    let pair_width = lanes_u64 * 2;

    for lane_id_idx in 0..lane_group_count {
        if *attempted_nonces >= limit {
            return None;
        }

        let mut lane_id_value = cerberus_lane_words::<LANES>(lane_id_idx * LANES);
        if CENTER_WORD_IDX < LANE_ID_WORD_IDX {
            lane_id_value >>= Simd::splat(8);
        }
        let patch = Simd::splat(msg[LANE_ID_WORD_IDX]) | lane_id_value;

        let lut = &crate::strings::DIGIT_LUT_10000_LE_EVEN;
        let remaining = limit - *attempted_nonces;
        let lut_len_u64 = u64::try_from(lut.len()).ok()?;
        let full_pairs = usize::try_from((remaining / pair_width).min(lut_len_u64)).ok()?;

        for (i, word) in lut.iter().take(full_pairs).enumerate() {
            msg[CENTER_WORD_IDX] = *word;
            let mut state0 = state_base;
            B::compress::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(&mut state0, &msg, patch);

            msg[CENTER_WORD_IDX] = *word | u32::from_be_bytes([1, 0, 0, 0]);
            let mut state1 = state_base;
            B::compress::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(&mut state1, &msg, patch);

            *attempted_nonces += lanes_u64;
            if let Some(success_lane) = first_zero_mask_lane(state0[0], mask_hi, LANES) {
                crate::unlikely();
                return Some((
                    u64::try_from(i).ok()? * 2,
                    u64::try_from(lane_id_idx).ok()? * lanes_u64
                        + u64::try_from(success_lane).ok()?,
                ));
            }

            *attempted_nonces += lanes_u64;
            if let Some(success_lane) = first_zero_mask_lane(state1[0], mask_hi, LANES) {
                crate::unlikely();
                return Some((
                    u64::try_from(i).ok()? * 2 + 1,
                    u64::try_from(lane_id_idx).ok()? * lanes_u64
                        + u64::try_from(success_lane).ok()?,
                ));
            }
        }

        if full_pairs < lut.len() {
            let tail = usize::try_from(limit - *attempted_nonces).ok()?;
            if tail == 0 {
                return None;
            }
            debug_assert!(tail < LANES * 2);
            let i = full_pairs;
            let word = lut[i];

            msg[CENTER_WORD_IDX] = word;
            let mut state0 = state_base;
            B::compress::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(&mut state0, &msg, patch);
            let valid0 = tail.min(LANES);
            *attempted_nonces += u64::try_from(valid0).ok()?;
            if let Some(success_lane) = first_zero_mask_lane(state0[0], mask_hi, valid0) {
                return Some((
                    u64::try_from(i).ok()? * 2,
                    u64::try_from(lane_id_idx).ok()? * lanes_u64
                        + u64::try_from(success_lane).ok()?,
                ));
            }
            if valid0 < LANES {
                return None;
            }

            let valid1 = tail - LANES;
            if valid1 == 0 {
                return None;
            }
            msg[CENTER_WORD_IDX] = word | u32::from_be_bytes([1, 0, 0, 0]);
            let mut state1 = state_base;
            B::compress::<CONSTANT_WORD_COUNT, LANE_ID_WORD_IDX>(&mut state1, &msg, patch);
            *attempted_nonces += u64::try_from(valid1).ok()?;
            if let Some(success_lane) = first_zero_mask_lane(state1[0], mask_hi, valid1) {
                return Some((
                    u64::try_from(i).ok()? * 2 + 1,
                    u64::try_from(lane_id_idx).ok()? * lanes_u64
                        + u64::try_from(success_lane).ok()?,
                ));
            }
            return None;
        }
    }

    None
}

fn solve_cerberus_binary_impl<const LANES: usize, B: CerberusCompression<LANES>>(
    message: &crate::message::CerberusMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    debug_assert_eq!(target, 0);

    let crate::message::CerberusMessage::Binary(message) = message else {
        return None;
    };

    let mut msg = [0; 16];
    msg[0] = message.first_word;
    let prepared_state = crate::blake3::ingest_message_prefix(
        message.midstate,
        &msg[..1],
        0,
        8,
        crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
    );
    let state_base: [Simd<u32, LANES>; 16] =
        core::array::from_fn(|i| Simd::splat(prepared_state[i]));
    let mask_hi = crate::high_u32(mask);
    let search_end = limit.min(u64::from(u32::MAX) + 1);
    let lanes_u64 = u64::try_from(LANES).ok()?;
    let mut cursor = (*attempted_nonces).min(search_end);
    let lane_offsets = Simd::<u32, LANES>::from_array(core::array::from_fn(usize_u32));
    let mut nonce = Simd::splat(crate::low_u32(cursor)) + lane_offsets;
    let nonce_increment = Simd::splat(usize_u32(LANES));

    while cursor < search_end {
        let batch_start = cursor;
        let valid_lanes = usize::try_from((search_end - cursor).min(lanes_u64)).ok()?;
        let mut state = state_base;
        B::compress::<1, 1>(&mut state, &msg, nonce);
        cursor += u64::try_from(valid_lanes).ok()?;
        *attempted_nonces = cursor;

        if let Some(success_lane) = first_zero_mask_lane(state[0], mask_hi, valid_lanes) {
            crate::unlikely();
            return Some(
                (batch_start + u64::try_from(success_lane).ok()?)
                    | (u64::from(message.first_word) << 32),
            );
        }
        nonce += nonce_increment;
    }

    None
}

#[inline(always)]
pub fn solve_cerberus_decimal_wide<
    const LANES: usize,
    const CENTER_WORD_IDX: usize,
    const LANE_ID_WORD_IDX: usize,
    const CONSTANT_WORD_COUNT: usize,
>(
    message: &crate::message::CerberusMessage,
    msg_tpl: [u32; 16],
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<(u64, u64)> {
    solve_cerberus_decimal_impl::<
        LANES,
        CENTER_WORD_IDX,
        LANE_ID_WORD_IDX,
        CONSTANT_WORD_COUNT,
        PortableCerberusCompression,
    >(message, msg_tpl, attempted_nonces, limit, target, mask)
}

#[inline(always)]
pub fn solve_cerberus_binary_wide<const LANES: usize>(
    message: &crate::message::CerberusMessage,
    attempted_nonces: &mut u64,
    limit: u64,
    target: u64,
    mask: u64,
) -> Option<u64> {
    solve_cerberus_binary_impl::<LANES, PortableCerberusCompression>(
        message,
        attempted_nonces,
        limit,
        target,
        mask,
    )
}

#[cfg(test)]
mod full_digest_tests {
    use super::*;

    #[test]
    fn leading_zero_match_checks_words_beyond_first_64_bits() {
        let mut state = core::array::from_fn(|_| Simd::<u32, 4>::splat(0));
        state[2] = Simd::from_array([0x7fff_ffff, 0x8000_0000, 0, 0]);
        assert_eq!(
            leading_zero_match(&state, 65).to_array(),
            [true, false, true, true]
        );
        assert_eq!(
            leading_zero_match(&state, 256).to_array(),
            [false, false, true, true]
        );
        assert_eq!(
            leading_zero_match(&state, 257).to_array(),
            [false, false, false, false]
        );
    }
}
