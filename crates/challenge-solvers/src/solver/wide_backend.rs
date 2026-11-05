use crate::message::{
    AltchaMessage, BinaryMessage, CerberusMessage, DecimalMessage, DoubleBlockMessage,
    GoAwayMessage, SingleBlockMessage,
};

use core::marker::PhantomData;

use super::wide_backend_arch::Backend;

/// Portable wide-SIMD decimal nonce single block solver.
///
///
/// Current implementation: backend-width SIMD with 1-round hotstart granularity.
pub struct SingleBlockSolver<B: Backend> {
    message: SingleBlockMessage,

    attempted_nonces: u64,

    limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<super::safe::SingleBlockSolver> for SingleBlockSolver<B> {
    fn from(solver: super::safe::SingleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> From<SingleBlockMessage> for SingleBlockSolver<B> {
    fn from(message: SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
            _backend: PhantomData,
        }
    }
}

const MUTATION_TYPE_UNALIGNED: u8 = 0;
const MUTATION_TYPE_ALIGNED: u8 = 1;
const MUTATION_TYPE_OCTAL: u8 = 2;
const MUTATION_TYPE_ALIGNED_OCTAL: u8 = MUTATION_TYPE_ALIGNED | MUTATION_TYPE_OCTAL;
const MUTATION_TYPE_UNALIGNED_OCTAL: u8 = MUTATION_TYPE_UNALIGNED | MUTATION_TYPE_OCTAL;

impl<B: Backend> SingleBlockSolver<B> {
    #[inline(always)]
    fn solve_impl<
        const DIGIT_WORD_IDX0: usize,
        const DIGIT_WORD_IDX1_INCREMENT: bool,
        const TYPE: u8,
        const MUTATION_TYPE: u8,
    >(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        B::solve_single_block::<DIGIT_WORD_IDX0, DIGIT_WORD_IDX1_INCREMENT, TYPE, MUTATION_TYPE>(
            &mut self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl<B: Backend> crate::solver::Solver for SingleBlockSolver<B> {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !B::available() {
            return None;
        }
        if self.attempted_nonces >= self.limit {
            return None;
        }
        let target = target & mask;

        // the official default difficulty is 5e6, so we design for 1e8
        // and there should almost always be a valid solution within our supported solution space
        // pgeom(5 * 16e7, 1/5e7, lower=F) = 0.03%
        // pgeom(16e7, 1/5e7, lower=F) = 20%, which is too much so we need the prefix to change as well

        // pre-compute an OR to apply to the message to add the lane ID
        let lane_id_0_word_idx = self.message.digit_index / 4;
        let lane_id_1_word_idx = (self.message.digit_index + 1) / 4;

        macro_rules! dispatch {
            ($idx0:literal, $idx1_inc:literal) => {
                if self.message.digit_index % 4 == 2 {
                    // if we have to much search space it doesn't matter
                    // use the octal kernel
                    if self.message.no_trailing_zeros
                        || self.message.approx_working_set_count.get() >= 100
                    {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED_OCTAL>(
                            target, mask,
                        )
                    } else {
                        self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_ALIGNED>(
                            target, mask,
                        )
                    }
                } else if self.message.no_trailing_zeros {
                    self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED_OCTAL>(
                        target, mask,
                    )
                } else {
                    self.solve_impl::<$idx0, $idx1_inc, TYPE, MUTATION_TYPE_UNALIGNED>(target, mask)
                }
            };
            ($idx0:literal) => {
                if lane_id_0_word_idx == lane_id_1_word_idx {
                    dispatch!($idx0, false)
                } else {
                    dispatch!($idx0, true)
                }
            };
        }

        let nonce = match lane_id_0_word_idx {
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
            _ => return None,
        }?;

        Some(nonce + self.message.nonce_addend)
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;

        // recompute the hash from the beginning
        // this prevents the compiler from having to compute the final B-H registers alive in tight loops
        let mut final_sha_state = self.message.prefix_state;
        crate::sha256::digest_block(&mut final_sha_state, &self.message.message);

        Some((nonce, final_sha_state))
    }
}

/// Portable wide-SIMD decimal nonce double block solver.
///
///
/// Current implementation: backend-width SIMD with 1-round hotstart granularity.
pub struct DoubleBlockSolver<B: Backend> {
    message: DoubleBlockMessage,
    attempted_nonces: u64,

    limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<super::safe::DoubleBlockSolver> for DoubleBlockSolver<B> {
    fn from(solver: super::safe::DoubleBlockSolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> From<DoubleBlockMessage> for DoubleBlockSolver<B> {
    fn from(message: DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> crate::solver::Solver for DoubleBlockSolver<B> {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline]
    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !B::available() {
            return None;
        }
        // SAFETY: the feature check above proves Portable wide-SIMD is available.
        self.solve_impl::<TYPE>(target, mask)
    }
}

impl<B: Backend> DoubleBlockSolver<B> {
    #[inline(always)]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        B::solve_double_block::<TYPE>(
            &mut self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

/// Decimal nonce solver selecting the single- or double-block layout.
pub enum DecimalSolver<B: Backend> {
    /// Single-block message.
    SingleBlock(SingleBlockSolver<B>),
    /// Double-block message.
    DoubleBlock(DoubleBlockSolver<B>),
}

impl<B: Backend> From<SingleBlockMessage> for DecimalSolver<B> {
    fn from(message: SingleBlockMessage) -> Self {
        Self::SingleBlock(message.into())
    }
}

impl<B: Backend> From<DoubleBlockMessage> for DecimalSolver<B> {
    fn from(message: DoubleBlockMessage) -> Self {
        Self::DoubleBlock(message.into())
    }
}

impl<B: Backend> From<DecimalMessage> for DecimalSolver<B> {
    fn from(message: DecimalMessage) -> Self {
        match message {
            DecimalMessage::SingleBlock(message) => Self::SingleBlock(message.into()),
            DecimalMessage::DoubleBlock(message) => Self::DoubleBlock(message.into()),
        }
    }
}

impl<B: Backend> crate::solver::Solver for DecimalSolver<B> {
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

/// Portable wide-SIMD binary nonce solver.
///
/// Output: nonce in little endian order
///
/// Current implementation: backend-width SIMD with 1-round hotstart granularity.
pub struct BinarySolver<B: Backend> {
    message: BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<BinaryMessage> for BinarySolver<B> {
    fn from(message: BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> From<crate::solver::safe::BinarySolver> for BinarySolver<B> {
    fn from(solver: crate::solver::safe::BinarySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> BinarySolver<B> {
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "const-generic SHA block specialization keeps hot solver inputs explicit"
    )]
    fn solve_impl<
        const TYPE: u8,
        const FIRST_NONCE_WORD_IDX: usize,
        const NEED_SECOND_BLOCK: bool,
    >(
        &mut self,
        prefix_state: [u32; 8],
        first_block: &[u32; 16],
        second_block_schedule: &[u32; 64],
        nonce_byte_offset: usize,
        nonce_byte_count: core::num::NonZeroU8,
        target: u64,
        mask: u64,
    ) -> Option<u64> {
        B::solve_binary::<TYPE, FIRST_NONCE_WORD_IDX, NEED_SECOND_BLOCK>(
            &mut self.attempted_nonces,
            self.limit,
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

impl<B: Backend> crate::solver::Solver for BinarySolver<B> {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !B::available() {
            return None;
        }
        // A one-byte search has only 256 candidates, and a nonce crossing a SHA block boundary
        // needs a distinct shuffle layout. Keep those rare cases on the scalar fallback instead of
        // burdening the hot SIMD path with extra setup and branches.
        if (self.message.nonce_byte_count.get() == 1)
            || (self.message.salt_residual_len + usize::from(self.message.nonce_byte_count.get())
                > 64)
        {
            crate::unlikely();
            let mut solver = crate::solver::safe::BinarySolver::from(self.message.clone());
            return solver.solve::<TYPE>(target, mask);
        }

        let super::PreparedBinaryBlocks {
            mut blocks,
            current_block,
            first_block,
            second_block_schedule,
        } = super::prepare_binary_blocks(&self.message)?;
        let used_blocks = &mut blocks[..=current_block];

        macro_rules! dispatch {
            ($skipped_rounds:expr) => {
                if current_block == 1 {
                    if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, true>(
                        self.message.prefix_state,
                        &first_block,
                        &second_block_schedule,
                        self.message.salt_residual_len,
                        self.message.nonce_byte_count,
                        target,
                        mask,
                    ) {
                        let mut final_sha_state = self.message.prefix_state;
                        for i in 0..usize::from(self.message.nonce_byte_count.get()) {
                            used_blocks[0][self.message.salt_residual_len + i] =
                                nonce.to_le_bytes()[i];
                        }
                        sha2::block_api::compress256(&mut final_sha_state, &used_blocks);
                        return Some((nonce, final_sha_state));
                    }
                } else if let Some(nonce) = self.solve_impl::<TYPE, { $skipped_rounds }, false>(
                    self.message.prefix_state,
                    &first_block,
                    &[0; 64],
                    self.message.salt_residual_len,
                    self.message.nonce_byte_count,
                    target,
                    mask,
                ) {
                    let mut final_sha_state = self.message.prefix_state;
                    for i in 0..usize::from(self.message.nonce_byte_count.get()) {
                        used_blocks[0][self.message.salt_residual_len + i] = nonce.to_le_bytes()[i];
                    }
                    sha2::block_api::compress256(&mut final_sha_state, &used_blocks);
                    return Some((nonce, final_sha_state));
                }
            };
        }

        match self.message.salt_residual_len / 4 {
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
            14 => dispatch!(14),
            15 => dispatch!(15),
            _ => unreachable!(),
        }

        crate::unlikely();

        None
    }
}

/// Portable wide-SIMD `GoAway` solver.
///
///
/// Current implementation: backend-width SIMD with 1-round hotstart granularity.
pub struct GoAwaySolver<B: Backend> {
    message: GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<super::safe::GoAwaySolver> for GoAwaySolver<B> {
    fn from(solver: super::safe::GoAwaySolver) -> Self {
        Self {
            message: solver.message,
            attempted_nonces: solver.attempted_nonces,
            limit: solver.limit,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> From<GoAwayMessage> for GoAwaySolver<B> {
    fn from(message: GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> GoAwaySolver<B> {
    const MSG_LEN: u32 = 10 * 4 * 8;

    #[inline(always)]
    fn solve_nonce_only_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        B::solve_goaway::<TYPE>(
            &mut self.attempted_nonces,
            self.limit,
            self.message.challenge,
            self.message.high_word,
            target,
            mask,
        )
    }
}

impl<B: Backend> crate::solver::Solver for GoAwaySolver<B> {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    #[inline(always)]
    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !B::available() {
            return None;
        }
        // SAFETY: the feature check above proves Portable wide-SIMD is available.
        self.solve_nonce_only_impl::<TYPE>(target, mask)
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        let mut output_msg = [0; 16];
        let nonce = self.solve_nonce_only::<TYPE>(target, mask)?;
        output_msg[..8].copy_from_slice(&self.message.challenge);
        output_msg[8] = crate::high_u32(nonce);
        output_msg[9] = crate::low_u32(nonce);
        output_msg[10] = u32::from_be_bytes([0x80, 0, 0, 0]);
        output_msg[15] = Self::MSG_LEN as _;

        let mut final_sha_state = crate::sha256::IV;
        crate::sha256::digest_block(&mut final_sha_state, &output_msg);

        Some((nonce, final_sha_state))
    }
}

/// Portable wide-SIMD Cerberus solver.
///
/// Current implementation: 9-digit out-of-order kernel with dual-wavefront backend-width SIMD and quarter-round hotstart granularity.
pub struct CerberusSolver<B: Backend> {
    message: CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<CerberusMessage> for CerberusSolver<B> {
    fn from(message: CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: !0,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> CerberusSolver<B> {
    /// Set the limit.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Get the attempted nonces.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
}

impl<B: Backend> CerberusSolver<B> {
    #[inline(always)]
    fn solve_decimal_impl<
        const CENTER_WORD_IDX: usize,
        const LANE_ID_WORD_IDX: usize,
        const CONSTANT_WORD_COUNT: usize,
    >(
        &mut self,
        msg_tpl: [u32; 16],
        target: u64,
        mask: u64,
    ) -> Option<(u64, u64)> {
        B::solve_cerberus_decimal::<CENTER_WORD_IDX, LANE_ID_WORD_IDX, CONSTANT_WORD_COUNT>(
            &self.message,
            msg_tpl,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }

    #[inline(always)]
    fn solve_binary_impl(&mut self, target: u64, mask: u64) -> Option<u64> {
        B::solve_cerberus_binary(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl<B: Backend> crate::solver::Solver for CerberusSolver<B> {
    type Output = [u32; 8];
    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve_nonce_only<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<u64> {
        if !B::available() {
            return None;
        }
        match &self.message {
            CerberusMessage::Binary(_) => self.solve_binary_impl(target, mask),
            CerberusMessage::Decimal(message) => {
                // two digits as lane ID, N=\x00, ? is prefix
                // position % 4 =0: |1234|5678|NNN9
                // position % 4 =1: |123?|4567|NN89
                // position % 4 =2: |12??|3456|N789
                // position % 4 =3: |1???|2345|6789

                let center_word_idx = message.salt_residual_len / 4 + 1;
                let nonce_addend = message.nonce_addend;
                let salt_residual = message.salt_residual;
                let salt_residual_len = message.salt_residual_len;
                let position_mod = message.salt_residual_len % 4;

                for resid0 in 0..10_u8 {
                    for resid1 in 0..10_u8 {
                        if self.attempted_nonces >= self.limit {
                            return None;
                        }
                        let msg = super::patch_cerberus_decimal_residual(
                            salt_residual,
                            salt_residual_len,
                            position_mod,
                            resid0,
                            resid1,
                        );
                        let msg = core::array::from_fn(|i| {
                            u32::from_le_bytes([
                                msg[i * 4],
                                msg[i * 4 + 1],
                                msg[i * 4 + 2],
                                msg[i * 4 + 3],
                            ])
                        });

                        macro_rules! dispatch {
                            ($center_word_idx:literal) => {
                                if position_mod < 2 {
                                    self.solve_decimal_impl::<$center_word_idx, { $center_word_idx - 1 }, {$center_word_idx - 1}>(
                                        msg, target, mask,
                                    )
                                } else {
                                    self.solve_decimal_impl::<$center_word_idx, { $center_word_idx + 1 }, $center_word_idx>(
                                        msg, target, mask,
                                    )
                                }
                            };
                        }

                        if let Some((middle_word, success_lane_idx)) = match center_word_idx {
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
                            14 => dispatch!(14),
                            15 => dispatch!(15),
                            _ => unreachable!(),
                        } {
                            return Some(
                                nonce_addend
                                    + super::cerberus_decimal_nonce(
                                        position_mod,
                                        middle_word,
                                        success_lane_idx,
                                        resid0,
                                        resid1,
                                    ),
                            );
                        }
                    }
                }

                None
            }
        }
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if let Some(nonce) = self.solve_nonce_only::<TYPE>(target, mask) {
            match &self.message {
                CerberusMessage::Decimal(message) => {
                    let mut msg = message.salt_residual;

                    let mut nonce_copy = nonce;
                    for i in (0..9).rev() {
                        msg[message.salt_residual_len + i] =
                            (nonce_copy % 10).to_le_bytes()[0] + b'0';
                        nonce_copy /= 10;
                    }

                    let msg = core::array::from_fn(|i| {
                        u32::from_le_bytes([
                            msg[i * 4],
                            msg[i * 4 + 1],
                            msg[i * 4 + 2],
                            msg[i * 4 + 3],
                        ])
                    });

                    let hash = crate::blake3::compress8(
                        &message.prefix_state,
                        &msg,
                        0,
                        u32::try_from(message.salt_residual_len).ok()? + 9,
                        message.flags,
                    );

                    Some((nonce, hash))
                }
                CerberusMessage::Binary(message) => {
                    let mut msg = [0; 16];
                    msg[0] = message.first_word;
                    msg[1] = crate::low_u32(nonce);
                    let hash = crate::blake3::compress8(
                        &message.midstate,
                        &msg,
                        0,
                        8,
                        crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
                    );
                    Some((u64::from(msg[1]) | u64::from(msg[0]) << 32, hash))
                }
            }
        } else {
            None
        }
    }
}

/// Portable wide-SIMD Altcha SHA-256 solver
pub struct AltchaSha256Solver<B: Backend> {
    pub(super) message: AltchaMessage,
    pub(super) attempted_nonces: u64,
    pub(super) limit: u64,
    _backend: PhantomData<B>,
}

impl<B: Backend> From<AltchaMessage> for AltchaSha256Solver<B> {
    fn from(message: AltchaMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> AltchaSha256Solver<B> {
    #[inline(always)]
    fn solve_impl<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        B::solve_altcha::<TYPE>(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }

    fn solve_fallback<const TYPE: u8>(
        &mut self,
        target: u64,
        mask: u64,
    ) -> Option<(u64, [u32; 8])> {
        crate::solver::wide::solve_altcha_fallback::<TYPE>(
            &self.message,
            &mut self.attempted_nonces,
            self.limit,
            target,
            mask,
        )
    }
}

impl<B: Backend> crate::solver::Solver for AltchaSha256Solver<B> {
    type Output = [u32; 8];

    fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    fn solve<const TYPE: u8>(&mut self, target: u64, mask: u64) -> Option<(u64, [u32; 8])> {
        if !B::available() {
            return self.solve_fallback::<TYPE>(target, mask);
        }

        self.solve_impl::<TYPE>(target, mask)
    }
}
