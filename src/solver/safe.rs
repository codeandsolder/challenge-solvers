use crate::message::{
    AltchaMessage, BinaryMessage, CerberusMessage, DecimalMessage, DoubleBlockMessage,
    GoAwayMessage, SingleBlockMessage,
};

/// Safe decimal nonce single block solver.
///
///
/// Current implementation: generic sha2 crate fallback.
pub struct SingleBlockSolver {
    pub(super) message: SingleBlockMessage,

    pub(super) attempted_nonces: u64,

    pub(super) limit: u64,
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

impl SingleBlockSolver {
    fn solve_impl<const TYPE: u8, const NO_TRAILING_ZEROS: bool>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let mut message_be = [0u8; 64];
        for i in 0..16 {
            message_be[i * 4..i * 4 + 4].copy_from_slice(&self.message.message[i].to_be_bytes());
        }
        let target = target & mask;

        for nonzero_digit in 1_u8..=9 {
            let start = self.message.digit_index;
            let varying_range = if NO_TRAILING_ZEROS {
                message_be[start..start + 8].fill(b'0');
                message_be[start + 8] = b'0' + nonzero_digit;
                start..start + 8
            } else {
                message_be[start] = b'0' + nonzero_digit;
                message_be[start + 1..start + 9].fill(b'0');
                start + 1..start + 9
            };

            for key in 0_u64..100_000_000 {
                if self.attempted_nonces >= self.limit {
                    return None;
                }

                let mut state = self.message.prefix_state;
                sha2::block_api::compress256(&mut state, core::array::from_ref(&message_be));
                self.attempted_nonces += 1;

                let pass = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                    crate::hash_has_leading_zero_bits(&state, target)
                } else if TYPE == crate::solver::SOLVE_TYPE_GT {
                    u64::from(state[0]) << 32 | u64::from(state[1]) > target
                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                    u64::from(state[0]) << 32 | u64::from(state[1]) < target
                } else {
                    (u64::from(state[0]) << 32 | u64::from(state[1])) & mask == target & mask
                };

                if pass {
                    let mut transformed_key = key;
                    if NO_TRAILING_ZEROS {
                        transformed_key *= 10;
                        transformed_key += u64::from(nonzero_digit);
                    } else {
                        transformed_key += 100_000_000 * u64::from(nonzero_digit);
                    }
                    return Some((transformed_key + self.message.nonce_addend, state));
                }

                if key + 1 < 100_000_000 {
                    let wrapped = crate::solver::increment_ascii_decimal(
                        &mut message_be[varying_range.clone()],
                    );
                    debug_assert!(!wrapped);
                }
            }
        }

        None
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
        if self.message.no_trailing_zeros {
            self.solve_impl::<TYPE, true>(target, mask)
        } else {
            self.solve_impl::<TYPE, false>(target, mask)
        }
    }
}

/// Safe decimal nonce double block solver.
///
///
/// Current implementation: generic sha2 crate fallback.
pub struct DoubleBlockSolver {
    pub(super) message: DoubleBlockMessage,
    pub(super) attempted_nonces: u64,

    pub(super) limit: u64,
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

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if self.attempted_nonces >= self.limit {
            return None;
        }
        let target = target & mask;

        let mut buffer = [0u8; 64];
        for i in 0..16 {
            buffer[i * 4..i * 4 + 4].copy_from_slice(&self.message.message[i].to_be_bytes());
        }

        let bit_length = self.message.message_length.checked_mul(8)?;
        let mut buffer2 = [0u8; 64];
        buffer2[56..].copy_from_slice(&bit_length.to_be_bytes());
        let mut terminal_message_schedule = [0; 64];
        terminal_message_schedule[14] = crate::high_u32(bit_length);
        terminal_message_schedule[15] = crate::low_u32(bit_length);
        crate::sha256::do_message_schedule_k_w(&mut terminal_message_schedule);

        let first_key: u64 = if self.message.nonce_addend == 0 {
            100_000_000
        } else {
            0
        };
        let digit_start = DoubleBlockMessage::DIGIT_IDX;
        buffer[digit_start..digit_start + 9].fill(b'0');
        if first_key != 0 {
            buffer[digit_start] = b'1';
        }

        for key in first_key..1_000_000_000 {
            let mut state = self.message.prefix_state;
            sha2::block_api::compress256(&mut state, &[buffer]);

            let feedforward = state;

            crate::sha256::sha2_arx_without_constants::<0, 64>(
                &mut state,
                terminal_message_schedule,
            );

            if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                for (word, saved) in state.iter_mut().zip(feedforward) {
                    *word = word.wrapping_add(saved);
                }
            } else {
                state[0] = state[0].wrapping_add(feedforward[0]);
                state[1] = state[1].wrapping_add(feedforward[1]);
            }

            let ab = u64::from(state[0]) << 32 | u64::from(state[1]);

            self.attempted_nonces += 1;

            let pass = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                crate::hash_has_leading_zero_bits(&state, target)
            } else if TYPE == crate::solver::SOLVE_TYPE_GT {
                ab > target
            } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                ab < target
            } else {
                ab & mask == target & mask
            };
            if pass {
                crate::unlikely();

                let mut state = self.message.prefix_state;
                sha2::block_api::compress256(&mut state, &[buffer, buffer2]);
                return Some((key + self.message.nonce_addend, state));
            }

            if self.attempted_nonces >= self.limit {
                return None;
            }

            if key + 1 < 1_000_000_000 {
                let wrapped = crate::solver::increment_ascii_decimal(
                    &mut buffer[digit_start..digit_start + 9],
                );
                debug_assert!(!wrapped);
            }
        }

        crate::unlikely();

        None
    }
}

impl_decimal_solver!(
    [SingleBlockSolver, DoubleBlockSolver] => DecimalSolver
);

/// Safe `GoAway` solver.
///
/// Current implementation: generic sha2 crate fallback.
pub struct GoAwaySolver {
    pub(super) message: GoAwayMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
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
}

impl crate::solver::Solver for GoAwaySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let target = target & mask;

        let mut buffer = [[0u8; 64]; 16];
        for i in 0..8 {
            buffer[0][i * 4..i * 4 + 4].copy_from_slice(&self.message.challenge[i].to_be_bytes());
        }
        buffer[0][32..36].copy_from_slice(&self.message.high_word.to_be_bytes());
        buffer[0][40] = 0x80;
        buffer[0][60..64].copy_from_slice(&(Self::MSG_LEN).to_be_bytes());

        let search_end = self.limit.min(u64::from(u32::MAX) + 1);
        let mut low_word = self.attempted_nonces.min(search_end);
        while low_word < search_end {
            let key = crate::low_u32(low_word);
            buffer[0][36..40].copy_from_slice(&key.to_be_bytes());

            let mut state = crate::sha256::IV;
            sha2::block_api::compress256(&mut state, core::array::from_ref(&buffer[0]));

            let state_ab = u64::from(state[0]) << 32 | u64::from(state[1]);
            self.attempted_nonces += 1;

            let pass = if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
                crate::hash_has_leading_zero_bits(&state, target)
            } else if TYPE == crate::solver::SOLVE_TYPE_GT {
                state_ab > target
            } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                state_ab < target
            } else {
                state_ab & mask == target & mask
            };
            if pass {
                crate::unlikely();

                return Some((
                    u64::from(self.message.high_word) << 32 | u64::from(key),
                    state,
                ));
            }

            low_word += 1;
        }
        crate::unlikely();
        None
    }
}

/// Safe binary nonce solver.
///
/// Output: nonce in little endian order
///
/// Current implementation: generic sha2 crate fallback.
pub struct BinarySolver {
    pub(super) message: BinaryMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
}

impl From<BinaryMessage> for BinarySolver {
    fn from(message: BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl crate::solver::Solver for BinarySolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let salt = &self.message.salt_residual[..self.message.salt_residual_len];
        let mut blocks = [[0u8; 64]; 2];
        blocks[0][..salt.len()].copy_from_slice(salt);
        let mut ptr = salt.len();
        let mut cur_block = 0;

        for _ in 0..self.message.nonce_byte_count.get() {
            blocks[cur_block][ptr] = 0;
            ptr += 1;
            if ptr == 64 {
                cur_block = 1;
                ptr = 0;
            }
        }
        blocks[cur_block][ptr] = 0x80;
        ptr += 1;
        if ptr + 8 > 64 {
            cur_block = 1;
        }
        blocks[cur_block][(64 - 8)..].copy_from_slice(
            &(u64::try_from(self.message.message_length)
                .ok()?
                .checked_mul(8)?)
            .to_be_bytes(),
        );

        let used_blocks = &mut blocks[..=cur_block];

        for x in 0..(self
            .limit
            .min(256u64.saturating_pow(u32::from(self.message.nonce_byte_count.get())))
            .max(1))
        {
            let mut state = self.message.prefix_state;
            let nonce_bytes = &x.to_le_bytes()[..usize::from(self.message.nonce_byte_count.get())];
            for (i, byte) in nonce_bytes.iter().enumerate() {
                let byte_index = self.message.salt_residual_len + i;
                used_blocks[byte_index / 64][byte_index % 64] = *byte;
            }

            sha2::block_api::compress256(&mut state, used_blocks);
            self.attempted_nonces += 1;

            let cmp_fn = |x: u64, y: u64| {
                if TYPE == crate::solver::SOLVE_TYPE_GT {
                    x > y
                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                    x < y
                } else {
                    x & mask == y & mask
                }
            };
            if cmp_fn(u64::from(state[0]) << 32 | u64::from(state[1]), target) {
                return Some((x, state));
            }

            if self.attempted_nonces >= self.limit {
                return None;
            }
        }

        None
    }
}

/// Safe Cerberus solver.
///
///
/// Current implementation: scalar fallback.
pub struct CerberusSolver {
    message: CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<CerberusMessage> for CerberusSolver {
    fn from(message: CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: !0,
        }
    }
}

impl crate::solver::Solver for CerberusSolver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        debug_assert_eq!(target, 0);

        let remaining_limit = self.limit.saturating_sub(self.attempted_nonces);

        match &self.message {
            CerberusMessage::Decimal(message) => {
                let mut msg = core::array::from_fn(|i| {
                    u32::from_le_bytes([
                        message.salt_residual[i * 4],
                        message.salt_residual[i * 4 + 1],
                        message.salt_residual[i * 4 + 2],
                        message.salt_residual[i * 4 + 3],
                    ])
                });
                assert!(
                    message.salt_residual_len + 8 < message.salt_residual.len(),
                    "there must be at least 9 bytes of headroom for the nonce"
                );
                for byte_index in message.salt_residual_len..message.salt_residual_len + 9 {
                    crate::write_le_word_byte(&mut msg, byte_index, b'0');
                }
                let search_limit = remaining_limit.min(1_000_000_000);
                for nonce in 0u64..search_limit {
                    let hash = crate::blake3::compress8(
                        &message.prefix_state,
                        &msg,
                        0,
                        u32::try_from(message.salt_residual_len).ok()? + 9,
                        message.flags,
                    );
                    self.attempted_nonces += 1;
                    if (u64::from(hash[0]) << 32 | u64::from(hash[1])) & mask == 0 {
                        crate::unlikely();

                        return Some((nonce + message.nonce_addend, hash));
                    }

                    if nonce + 1 < search_limit {
                        let start = message.salt_residual_len;
                        let end = start + 9;
                        let mut wrapped = true;
                        for byte_index in (start..end).rev() {
                            let byte = crate::read_le_word_byte(&msg, byte_index);
                            if byte != b'9' {
                                crate::write_le_word_byte(&mut msg, byte_index, byte + 1);
                                wrapped = false;
                                break;
                            }
                            crate::write_le_word_byte(&mut msg, byte_index, b'0');
                        }
                        debug_assert!(!wrapped);
                    }
                }
            }
            CerberusMessage::Binary(message) => {
                let mut msg = [0; 16];
                msg[0] = message.first_word;
                let search_limit = remaining_limit.min(u64::from(u32::MAX) + 1);
                for nonce in 0..search_limit {
                    msg[1] = crate::low_u32(nonce);

                    let hash = crate::blake3::compress8(
                        &message.midstate,
                        &msg,
                        0,
                        8,
                        crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
                    );
                    self.attempted_nonces += 1;
                    if (u64::from(hash[0]) << 32 | u64::from(hash[1])) & mask == 0 {
                        crate::unlikely();

                        return Some((u64::from(msg[1]) | (u64::from(msg[0]) << 32), hash));
                    }
                }
            }
        }

        None
    }
}

/// Safe Altcha SHA-256 solver
pub struct AltchaSha256Solver {
    pub(super) message: AltchaMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
}

impl From<AltchaMessage> for AltchaSha256Solver {
    fn from(message: AltchaMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl AltchaSha256Solver {
    fn solve_nested_impl<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let mut blocks = [0u8; 64];
        let mut subblocks = [0u8; 64];
        blocks[0..16].copy_from_slice(&self.message.salt);
        blocks[16..32].copy_from_slice(&self.message.nonce);
        blocks[32 + 4] = 0x80;
        blocks[60..].copy_from_slice(&((32u32 + 4) * 8).to_be_bytes());
        let key_length = self.message.key_length.get().min(32);

        let search_end = self.limit.min(u64::from(u32::MAX) + 1);
        let mut counter = self.attempted_nonces.min(search_end);
        while counter < search_end {
            let counter_u32 = crate::low_u32(counter);
            let mut state = crate::sha256::IV;
            blocks[32..(32 + 4)].copy_from_slice(&counter_u32.to_be_bytes());
            sha2::block_api::compress256(&mut state, core::array::from_ref(&blocks));

            for _ in 1..self.message.cost.get() {
                for i in 0..8 {
                    subblocks[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_be_bytes());
                }
                subblocks[usize::try_from(key_length).ok()?] = 0x80;
                subblocks[(usize::try_from(key_length).ok()? + 1)..40].fill(0);
                subblocks[60..].copy_from_slice(&(key_length * 8).to_be_bytes());
                state = crate::sha256::IV;
                sha2::block_api::compress256(&mut state, core::array::from_ref(&subblocks));
            }

            self.attempted_nonces = counter + 1;
            let cmp64_fn = |x: u64, y: u64| {
                if TYPE == crate::solver::SOLVE_TYPE_GT {
                    x > y
                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                    x < y
                } else {
                    x & mask == y & mask
                }
            };
            if cmp64_fn(u64::from(state[0]) << 32 | u64::from(state[1]), target) {
                return Some((counter, state));
            }
            counter += 1;
        }
        None
    }

    fn solve_pbkdf2_impl<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        let mut buffer = [0; 32];
        let mut password = [0; 16 + 4];
        password[..16].copy_from_slice(&self.message.nonce);

        let search_end = self.limit.min(u64::from(u32::MAX) + 1);
        let mut counter = self.attempted_nonces.min(search_end);
        while counter < search_end {
            let counter_u32 = crate::low_u32(counter);
            password[16..16 + 4].copy_from_slice(&counter_u32.to_be_bytes());
            pbkdf2::pbkdf2_hmac::<sha2::Sha256>(
                &password,
                &self.message.salt,
                self.message.cost.get(),
                &mut buffer,
            );

            self.attempted_nonces = counter + 1;
            let cmp64_fn = |x: u64, y: u64| {
                if TYPE == crate::solver::SOLVE_TYPE_GT {
                    x > y
                } else if TYPE == crate::solver::SOLVE_TYPE_LT {
                    x < y
                } else {
                    x & mask == y & mask
                }
            };
            if cmp64_fn(
                u64::from_be_bytes([
                    buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6],
                    buffer[7],
                ]),
                target,
            ) {
                return Some((
                    counter,
                    core::array::from_fn(|i| {
                        u32::from_be_bytes([
                            buffer[i * 4],
                            buffer[i * 4 + 1],
                            buffer[i * 4 + 2],
                            buffer[i * 4 + 3],
                        ])
                    }),
                ));
            }
            counter += 1;
        }

        None
    }
}

impl crate::solver::Solver for AltchaSha256Solver {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if self.message.pbkdf2 {
            self.solve_pbkdf2_impl::<TYPE>(target, mask)
        } else {
            self.solve_nested_impl::<TYPE>(target, mask)
        }
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU8;

    use crate::message::{CerberusBinaryMessage, CerberusDecimalMessage};
    use crate::solver::Solver as _;

    use super::*;

    #[test]
    fn test_solve_decimal() -> anyhow::Result<()> {
        crate::solver::tests::test_decimal_validator::<DecimalSolver, _>(
            |prefix, search_space| {
                SingleBlockMessage::new(prefix, search_space)
                    .map(Into::into)
                    .map_or_else(
                        || DoubleBlockMessage::new(prefix, search_space).map(Into::into),
                        |solver| Some(DecimalSolver::SingleBlock(solver)),
                    )
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_solve_cerberus_decimal() -> anyhow::Result<()> {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_decimal_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Decimal(CerberusDecimalMessage::new(prefix, i)?).into())
            })?;
        }
        Ok(())
    }
    #[test]
    fn test_solve_cerberus_binary() -> anyhow::Result<()> {
        for i in 0..=1 {
            crate::solver::tests::test_cerberus_binary_validator::<CerberusSolver, _>(|prefix| {
                Some(CerberusMessage::Binary(CerberusBinaryMessage::new(prefix, i)).into())
            })?;
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
    fn test_solve_binary() -> anyhow::Result<()> {
        crate::solver::tests::test_binary_validator::<BinarySolver, _>(
            |prefix, nonce_byte_count| {
                BinaryMessage::new(prefix, nonce_byte_count).map(BinarySolver::from)
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

    #[test]
    fn test_goaway_resumes_after_limit_increase() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};

        let challenge = [b'a'; 32];
        let target_nonce = 4u64;
        let mut hasher = Sha256::new();
        hasher.update(challenge);
        hasher.update(target_nonce.to_be_bytes());
        let hash = hasher.finalize();
        let target = u64::from_be_bytes(hash[..8].try_into()?);

        let mut solver = GoAwaySolver::from(GoAwayMessage::new_bytes(&challenge, 0));

        solver.set_limit(1);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(target, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), 1);

        solver.set_limit(5);
        let (found_nonce, found_hash) = solver
            .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(target, u64::MAX)
            .ok_or_else(|| anyhow::anyhow!("resumed search did not reach nonce 4"))?;
        let expected = core::array::from_fn(|i| {
            let base = i * 4;
            u32::from_be_bytes([hash[base], hash[base + 1], hash[base + 2], hash[base + 3]])
        });
        assert_eq!(found_nonce, target_nonce);
        assert_eq!(found_hash, expected);
        assert_eq!(solver.attempted_nonces(), 5);
        Ok(())
    }

    #[test]
    fn test_exact_attempt_limits() -> anyhow::Result<()> {
        fn exhaust_one<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) {
            solver.set_limit(1);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.attempted_nonces(), 1);
        }

        let salt = [b'a'; 64];
        exhaust_one(SingleBlockSolver::from(
            SingleBlockMessage::new(&salt, 0)
                .ok_or_else(|| anyhow::anyhow!("single-block test fixture is invalid"))?,
        ));
        let double = (0..64)
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .ok_or_else(|| anyhow::anyhow!("no double-block test fixture shape"))?;
        exhaust_one(DoubleBlockSolver::from(double));
        let nonce_width =
            NonZeroU8::new(4).ok_or_else(|| anyhow::anyhow!("nonce width is nonzero"))?;
        exhaust_one(BinarySolver::from(
            BinaryMessage::new(&salt, nonce_width)
                .ok_or_else(|| anyhow::anyhow!("four-byte binary test fixture is invalid"))?,
        ));
        let challenge: &[u8; 32] = (&salt[..32]).try_into()?;
        exhaust_one(GoAwaySolver::from(GoAwayMessage::new_bytes(challenge, 0)));
        exhaust_one(AltchaSha256Solver::from(AltchaMessage {
            nonce: [1; 16],
            salt: [2; 16],
            cost: 1.try_into()?,
            pbkdf2: false,
            key_length: 32.try_into()?,
        }));
        Ok(())
    }

    #[test]
    fn test_solve_altcha() -> anyhow::Result<()> {
        crate::solver::tests::test_altcha_validator::<AltchaSha256Solver, _>(|message| {
            Some(AltchaSha256Solver::from(message))
        })?;
        Ok(())
    }
}
