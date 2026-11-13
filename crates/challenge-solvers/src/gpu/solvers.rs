//! Generic GPU protocol solvers for SHA-256 and Cerberus message formats.
use super::{
    GpuContext, GpuError, GpuSolution, HASHES_PER_WG, binary_templates, expected_work,
    gpu_batch_target, gpu_u32, gpu_usize_u32, high_u32, low_u32, single_block_search_space,
    verify_binary, verify_cerberus, verify_double_block, verify_goaway,
};

/// Async GPU counterpart of the CPU single-block SHA-256 solver family.
///
/// Anubis, mCaptcha and Cap.js all route through this protocol-agnostic message type.
#[derive(Debug, Clone)]
pub struct SingleBlockSolver {
    message: crate::message::SingleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::SingleBlockMessage> for SingleBlockSolver {
    fn from(message: crate::message::SingleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl SingleBlockSolver {
    /// Set the maximum number of candidates dispatched by this solver.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    ///
    /// GPU workgroups cooperatively stop after a hit, so this is an upper bound on candidates
    /// actually executed by the device.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this SHA-256 single-block message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let result = gpu
            .solve_single_block_inner::<TYPE>(&self.message, target, mask, self.limit)
            .await?;
        self.attempted_nonces = result.as_ref().map_or_else(
            || self.limit.min(single_block_search_space(&self.message)),
            |solution| solution.dispatched_hashes.min(self.limit),
        );
        Ok(result)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU counterpart of the CPU double-block decimal SHA-256 solver.
pub struct DoubleBlockSolver {
    message: crate::message::DoubleBlockMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::DoubleBlockMessage> for DoubleBlockSolver {
    fn from(message: crate::message::DoubleBlockMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl DoubleBlockSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve this double-block decimal message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let search_space = if self.message.nonce_addend == 0 {
            900_000_000_u64
        } else {
            1_000_000_000
        };
        let limit = self.limit.min(search_space);
        let start = if self.message.nonce_addend == 0 {
            100_000_000_u64
        } else {
            0
        };
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 1;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = gpu_u32(start + offset, "double-block candidate start")?;
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[16..24].copy_from_slice(&self.message.prefix_state);
            p[24..40].copy_from_slice(&self.message.message);
            let mut hot_state = self.message.prefix_state;
            let mut hot_prefix = [0_u32; 13];
            hot_prefix.copy_from_slice(&self.message.message[..13]);
            crate::sha256::ingest_message_prefix::<13>(&mut hot_state, hot_prefix);
            p[44..52].copy_from_slice(&hot_state);
            p[40] = low_u32(self.message.nonce_addend);
            p[41] = high_u32(self.message.nonce_addend);
            let bitlen = self.message.message_length.saturating_mul(8);
            p[42] = high_u32(bitlen);
            p[43] = low_u32(bitlen);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_double_block::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid double-block proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// GPU router for decimal SHA-256 messages, mirroring [`crate::message::DecimalMessage`].
pub enum DecimalSolver {
    /// Single-block message.
    SingleBlock(SingleBlockSolver),
    /// Double-block message.
    DoubleBlock(DoubleBlockSolver),
}

impl From<crate::message::DecimalMessage> for DecimalSolver {
    fn from(message: crate::message::DecimalMessage) -> Self {
        match message {
            crate::message::DecimalMessage::SingleBlock(m) => Self::SingleBlock(m.into()),
            crate::message::DecimalMessage::DoubleBlock(m) => Self::DoubleBlock(m.into()),
        }
    }
}

impl DecimalSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        match self {
            Self::SingleBlock(s) => s.set_limit(limit),
            Self::DoubleBlock(s) => s.set_limit(limit),
        }
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        match self {
            Self::SingleBlock(s) => s.attempted_nonces(),
            Self::DoubleBlock(s) => s.attempted_nonces(),
        }
    }
    /// Solve a decimal SHA-256 message.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        match self {
            Self::SingleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
            Self::DoubleBlock(s) => s.solve::<TYPE>(gpu, target, mask).await,
        }
    }
}

/// Async GPU `GoAway` SHA-256 solver.
pub struct GoAwaySolver {
    message: crate::message::GoAwayMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::GoAwayMessage> for GoAwaySolver {
    fn from(message: crate::message::GoAwayMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl GoAwaySolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a `GoAway` message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let limit = self.limit.min(u64::from(u32::MAX) + 1);
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut offset = 0_u64;
        while offset < limit {
            let batch = desired
                .min(limit - offset)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 2;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = gpu_u32(offset, "candidate offset")?;
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[16..24].copy_from_slice(&self.message.challenge);
            p[24] = self.message.high_word;
            let mut hot_state = crate::sha256::IV;
            let mut invariant = [0_u32; 9];
            invariant[..8].copy_from_slice(&self.message.challenge);
            invariant[8] = self.message.high_word;
            crate::sha256::ingest_message_prefix::<9>(&mut hot_state, invariant);
            p[40..48].copy_from_slice(&hot_state);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = offset + batch;
                if !verify_goaway::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid GoAway proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            offset += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU solver for binary SHA-256 nonce messages.
pub struct BinarySolver {
    message: crate::message::BinaryMessage,
    attempted_nonces: u64,
    limit: u64,
}
impl From<crate::message::BinaryMessage> for BinarySolver {
    fn from(message: crate::message::BinaryMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}
impl BinarySolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }
    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }
    /// Solve a binary-nonce SHA-256 message on the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        let bits = u32::from(self.message.nonce_byte_count.get()) * 8;
        let space = if bits == 64 { u64::MAX } else { 1_u64 << bits };
        let limit = self.limit.min(space);
        let (templates, used) = binary_templates(&self.message);
        let desired = gpu_batch_target(expected_work::<TYPE>(target, mask), HASHES_PER_WG);
        let mut base = 0_u64;
        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG);
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[0] = 3;
            p[1] = u32::from(TYPE);
            p[2] = u32::from(cfg!(feature = "compare-64bit"));
            p[3] = low_u32(base);
            p[4] = high_u32(base);
            p[5] = gpu_u32(batch, "batch size")?;
            p[6] = high_u32(target);
            p[7] = low_u32(target);
            p[8] = high_u32(mask);
            p[9] = low_u32(mask);
            p[10] = used;
            p[11] = u32::from(self.message.nonce_byte_count.get());
            p[12] = gpu_usize_u32(self.message.salt_residual_len, "salt residual length")?;
            p[16..24].copy_from_slice(&self.message.prefix_state);
            p[24..40].copy_from_slice(&templates[0]);
            p[40..56].copy_from_slice(&templates[1]);
            if let Some((nonce, hash)) = gpu.dispatch_multi(&p, wgs).await? {
                self.attempted_nonces = base + batch;
                if !verify_binary::<TYPE>(&self.message, nonce, hash, target, mask) {
                    return Err(GpuError::new("GPU returned an invalid binary proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }
    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|x| x.nonce))
    }
}

/// Async GPU counterpart of the CPU Cerberus BLAKE3 solver.
pub struct CerberusSolver {
    message: crate::message::CerberusMessage,
    attempted_nonces: u64,
    limit: u64,
}

impl From<crate::message::CerberusMessage> for CerberusSolver {
    fn from(message: crate::message::CerberusMessage) -> Self {
        Self {
            message,
            attempted_nonces: 0,
            limit: u64::MAX,
        }
    }
}

impl CerberusSolver {
    /// Set the maximum number of candidates dispatched.
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve a Cerberus message on the GPU.
    ///
    /// Cerberus uses a zero target plus bit mask; this mirrors the CPU backend.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<GpuSolution>, GpuError> {
        if target != 0 {
            return Err(GpuError::new("Cerberus GPU solver requires target == 0"));
        }
        let _ = TYPE;
        let expected = 1_u64.checked_shl(mask.count_ones()).unwrap_or(u64::MAX);
        let desired = gpu_batch_target(expected, HASHES_PER_WG);
        let search_space = match &self.message {
            crate::message::CerberusMessage::Decimal(_) => 1_000_000_000_u64,
            crate::message::CerberusMessage::Binary(_) => u64::from(u32::MAX) + 1,
        };
        let limit = self.limit.min(search_space);
        let mut base = 0_u64;

        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(u64::from(gpu.max_wgs) * HASHES_PER_WG)
                .min(u64::from(u32::MAX));
            let wgs = gpu_u32(batch.div_ceil(HASHES_PER_WG).max(1), "workgroup count")?;
            let mut p = [0_u32; 64];
            p[1] = low_u32(base);
            p[2] = gpu_u32(batch, "batch size")?;
            p[3] = high_u32(mask);
            p[4] = low_u32(mask);

            match &self.message {
                crate::message::CerberusMessage::Decimal(message) => {
                    p[0] = 0;
                    p[8..16].copy_from_slice(&message.prefix_state);
                    for (i, chunk) in message.salt_residual.as_chunks::<4>().0.iter().enumerate() {
                        p[16 + i] = u32::from_le_bytes(*chunk);
                    }
                    p[32] = gpu_usize_u32(message.salt_residual_len, "salt residual length")?;
                    p[33] = gpu_usize_u32(message.salt_residual_len, "salt residual length")? + 9;
                    p[34] = message.flags;
                    p[35] = low_u32(message.nonce_addend);
                    p[36] = high_u32(message.nonce_addend);
                }
                crate::message::CerberusMessage::Binary(message) => {
                    p[0] = 1;
                    p[8..16].copy_from_slice(&message.midstate);
                    p[16] = message.first_word;
                }
            }

            if let Some((nonce, hash)) = gpu.dispatch_cerberus(&p, wgs).await? {
                self.attempted_nonces = base + batch;
                if !verify_cerberus(&self.message, nonce, hash, mask) {
                    return Err(GpuError::new("GPU returned an invalid Cerberus proof"));
                }
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit;
        Ok(None)
    }

    /// Solve and return only the nonce.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError`] if GPU setup, submission, synchronization, or result readback fails.
    pub async fn solve_nonce_only<const TYPE: u8>(
        &mut self,
        gpu: &GpuContext,
        target: u64,
        mask: u64,
    ) -> Result<Option<u64>, GpuError> {
        Ok(self
            .solve::<TYPE>(gpu, target, mask)
            .await?
            .map(|solution| solution.nonce))
    }
}
