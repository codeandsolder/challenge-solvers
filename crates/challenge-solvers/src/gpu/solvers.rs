//! Public SHA-256/BLAKE3 GPU protocol solver wrappers and ALTCHA proof verification.
use super::{
    Digest, Future, GpuContext, GpuError, GpuSolution, HASHES_PER_WG, Pod, Sha256, Vec, WG_SIZE,
    Zeroable, be_words_4, be_words_8, be_words_16, binary_templates, expected_work,
    expected_work_full64, gpu_batch_target, gpu_u32, gpu_usize_u32, high_u32, low_u32,
    result_matches_full64, single_block_search_space, verify_binary, verify_cerberus,
    verify_double_block, verify_goaway,
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

/// Reusable midstate and at-most-one partial compression block for
/// SHA-256(salt || nonce || variable-counter). Entire fixed prefixes are
/// compressed once on CPU; each GPU candidate completes at most two blocks.
struct AltchaShaPrefix {
    fixed_bytes: Vec<u8>,
    state: [u32; 8],
    tail: [u8; 64],
    tail_len: usize,
    fixed_len: u64,
}

impl AltchaShaPrefix {
    fn new(fixed_bytes: Vec<u8>) -> Result<Self, GpuError> {
        let fixed_len = u64::try_from(fixed_bytes.len())
            .map_err(|_| GpuError::new("ALTCHA SHA-256 prefix length does not fit in u64"))?;
        // SHA-256's 64-bit bit length must also accommodate a maximum
        // 16-digit decimal counter, without overflow.
        if fixed_len > (u64::MAX / 8) - 16 {
            return Err(GpuError::new(
                "ALTCHA SHA-256 message exceeds 64-bit bit length",
            ));
        }
        let mut state = crate::sha256::IV;
        let (chunks, remainder) = fixed_bytes.as_chunks::<64>();
        for chunk in chunks {
            let words: [u32; 16] = core::array::from_fn(|i| {
                u32::from_be_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ])
            });
            crate::sha256::digest_block(&mut state, &words);
        }
        let tail_len = remainder.len();
        let mut tail = [0_u8; 64];
        tail[..tail_len].copy_from_slice(remainder);
        Ok(Self {
            fixed_bytes,
            state,
            tail,
            tail_len,
            fixed_len,
        })
    }
}

/// Variable-length PBKDF2-HMAC-SHA256 message representation.
///
/// The nonce is the fixed part of the candidate-dependent HMAC key. For
/// passwords longer than a SHA-256 block, its prefix state is reused by
/// the GPU to hash the complete nonce+counter key before applying HMAC.
///
/// The salt's RFC 8018 block index and SHA-256 padding are prepared once.
/// Unlike its HMAC chaining state, these message blocks are independent of
/// the candidate password and may be shared by every GPU invocation.
struct AltchaPbkdf2Input {
    salt: Vec<u8>,
    key_prefix: AltchaShaPrefix,
    u1_blocks: Vec<u32>,
}

impl AltchaPbkdf2Input {
    fn new(salt: &[u8], nonce: &[u8]) -> Result<Self, GpuError> {
        let key_prefix = AltchaShaPrefix::new(nonce.to_vec())?;
        let salt_len = salt
            .len()
            .checked_add(4)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 salt length overflow"))?;
        let padded_len = salt_len
            .checked_add(9)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 padding length overflow"))?
            .div_ceil(64)
            .checked_mul(64)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 padded size overflow"))?;
        let hmac_msg_len = u64::try_from(salt_len)
            .map_err(|_| GpuError::new("ALTCHA PBKDF2 salt length exceeds u64"))?
            .checked_add(64)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 inner length overflow"))?;
        let bit_len = hmac_msg_len
            .checked_mul(8)
            .ok_or_else(|| GpuError::new("ALTCHA PBKDF2 exceeds SHA-256 length limit"))?;
        let mut padded = vec![0_u8; padded_len];
        padded[..salt.len()].copy_from_slice(salt);
        padded[salt.len() + 3] = 1; // PBKDF2 block index INT(1), BE32.
        padded[salt_len] = 0x80;
        padded[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());
        let (words, _) = padded.as_chunks::<4>();
        let u1_blocks = words.iter().map(|word| u32::from_be_bytes(*word)).collect();
        Ok(Self {
            salt: salt.to_vec(),
            key_prefix,
            u1_blocks,
        })
    }
}

/// Async GPU counterpart of the CPU Altcha SHA-256/PBKDF2 solver.
pub struct AltchaSha256Solver {
    message: crate::message::AltchaMessage,
    attempted_nonces: u64,
    start_nonce: u64,
    limit: u64,
    string_counter: bool,
    prefix_nibbles: Vec<u8>,
    sha_prefix: Option<AltchaShaPrefix>,
    pbkdf2_input: Option<AltchaPbkdf2Input>,
    output_key_length: usize,
    verified_key: Option<Vec<u8>>,
}

impl From<crate::message::AltchaMessage> for AltchaSha256Solver {
    fn from(message: crate::message::AltchaMessage) -> Self {
        let output_key_length = message.key_length.get() as usize;
        Self {
            message,
            attempted_nonces: 0,
            start_nonce: 0,
            limit: u64::MAX,
            string_counter: false,
            prefix_nibbles: Vec::new(),
            sha_prefix: None,
            pbkdf2_input: None,
            output_key_length,
            verified_key: None,
        }
    }
}

impl AltchaSha256Solver {
    /// Construct a SHA-256 ALTCHA solver for arbitrary-length binary
    /// salt/nonce data. Pre-hash complete blocks on CPU once; the GPU
    /// performs only the counter-dependent SHA-256 compression.
    ///
    /// # Errors
    ///
    /// Rejects message lengths beyond SHA-256's 64-bit length field.
    pub fn for_sha256_prefix(
        salt: &[u8],
        nonce: &[u8],
        cost: core::num::NonZeroU32,
    ) -> Result<Self, GpuError> {
        let length = salt
            .len()
            .checked_add(nonce.len())
            .ok_or_else(|| GpuError::new("ALTCHA SHA-256 salt and nonce length overflow"))?;
        let mut fixed = Vec::with_capacity(length);
        fixed.extend_from_slice(salt);
        fixed.extend_from_slice(nonce);
        let sha_prefix = AltchaShaPrefix::new(fixed)?;
        // The fixed CPU message struct is retained for existing PBKDF2
        // layouts. The SHA path below uses the general prefix exclusively.
        let message = crate::message::AltchaMessage {
            nonce: [0; 16],
            salt: [0; 16],
            cost,
            pbkdf2: false,
            key_length: core::num::NonZeroU32::new(32)
                .ok_or_else(|| GpuError::new("invalid SHA-256 output length"))?,
        };
        let mut solver = Self::from(message);
        solver.sha_prefix = Some(sha_prefix);
        Ok(solver)
    }

    /// Construct a PBKDF2-HMAC-SHA256 search over arbitrary-length binary
    /// salt and nonce. GPU candidates derive their own HMAC keys (including
    /// SHA-256 hashing of passwords exceeding 64 bytes), process all U1 salt
    /// blocks and run the remaining PBKDF2 iterations on-device.
    ///
    /// # Errors
    ///
    /// Rejects input lengths incompatible with SHA-256 bit-length encoding.
    pub fn for_pbkdf2(
        salt: &[u8],
        nonce: &[u8],
        cost: core::num::NonZeroU32,
    ) -> Result<Self, GpuError> {
        let input = AltchaPbkdf2Input::new(salt, nonce)?;
        let message = crate::message::AltchaMessage {
            nonce: [0; 16],
            salt: [0; 16],
            cost,
            pbkdf2: true,
            key_length: core::num::NonZeroU32::new(32)
                .ok_or_else(|| GpuError::new("invalid PBKDF2 output length"))?,
        };
        let mut solver = Self::from(message);
        solver.pbkdf2_input = Some(input);
        Ok(solver)
    }

    /// Request any PBKDF2-SHA256 derived key size allowed by RFC 8018.
    /// The first 32 bytes are computed for each candidate; additional
    /// PBKDF2 blocks are computed on GPU only after earlier prefix blocks
    /// match. The complete derived key is independently computed on CPU
    /// only for the winning candidate.
    ///
    /// # Errors
    ///
    /// Rejects zero length, non-PBKDF2 inputs, and lengths beyond RFC 8018.
    pub fn set_pbkdf2_key_length(&mut self, length: usize) -> Result<(), GpuError> {
        let max_len = u64::from(u32::MAX) * 32;
        if !self.message.pbkdf2 || length == 0 || !u64::try_from(length).is_ok_and(|n| n <= max_len)
        {
            return Err(GpuError::new("invalid PBKDF2-SHA256 derived key length"));
        }
        self.output_key_length = length;
        self.verified_key = None;
        Ok(())
    }

    /// The complete independently validated answer from the last successful
    /// PBKDF2 GPU solve, rather than just the first SHA-256 output block.
    #[must_use]
    pub fn verified_key(&self) -> Option<&[u8]> {
        self.verified_key.as_deref()
    }

    /// Require an exact hexadecimal prefix of the derived output, including
    /// half-byte prefixes. SHA-256 accepts up to 64 nibbles; the general
    /// PBKDF2 solver also accepts prefixes spanning multiple digest blocks.
    ///
    /// The input is nibble values (0..=15), rather than ASCII hex digits.
    ///
    /// # Errors
    ///
    /// Rejects malformed nibbles and multi-block prefixes on unsupported inputs.
    pub fn set_prefix_nibbles(&mut self, nibbles: &[u8]) -> Result<(), GpuError> {
        if (nibbles.len() > 64 && self.pbkdf2_input.is_none())
            || nibbles.iter().any(|&nibble| nibble > 15)
        {
            return Err(GpuError::new(
                "ALTCHA GPU prefix requires valid hex nibbles and a multi-block PBKDF2 input beyond 64",
            ));
        }
        self.prefix_nibbles.clear();
        self.prefix_nibbles.extend_from_slice(nibbles);
        Ok(())
    }

    /// Use ALTCHA v2 decimal-string counter encoding rather than binary u32.
    ///
    /// Covers the complete JavaScript-safe integer counter domain (0..=2^53-1).
    pub const fn set_string_counter(&mut self) {
        self.string_counter = true;
    }

    /// Set the inclusive search starting counter, allowing disjoint GPU
    /// partitions and exact tests across the u32 rollover.
    pub const fn set_start_nonce(&mut self, start: u64) {
        self.start_nonce = start;
    }

    /// Set the exclusive upper counter bound (up to the encoding's domain).
    pub const fn set_limit(&mut self, limit: u64) {
        self.limit = limit;
    }

    /// Number of candidates dispatched so far.
    #[must_use]
    pub const fn attempted_nonces(&self) -> u64 {
        self.attempted_nonces
    }

    /// Solve an Altcha SHA-256 or PBKDF2 challenge on the GPU.
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
        let counter_limit = if self.string_counter {
            1_u64 << 53
        } else {
            u64::from(u32::MAX) + 1
        };
        let limit = self.limit.min(counter_limit);
        self.verified_key = None;
        let desired = gpu_batch_target(
            expected_work_full64::<TYPE>(target, mask),
            u64::from(WG_SIZE),
        );
        let max_batch = u64::from(gpu.max_wgs) * u64::from(WG_SIZE);
        let mut base = self.start_nonce;
        while base < limit {
            let batch = desired
                .min(limit - base)
                .min(max_batch)
                .min(u64::from(u32::MAX));
            let wgs = gpu_u32(batch.div_ceil(u64::from(WG_SIZE)).max(1), "workgroup count")?;
            let mut p = vec![0_u32; 64];
            if let Some(input) = &self.pbkdf2_input {
                p.extend_from_slice(&input.u1_blocks);
            }
            p[0] = u32::from(self.message.pbkdf2);
            p[1] = low_u32(base);
            p[2] = gpu_u32(batch, "batch size")?;
            p[10] = high_u32(base);
            p[11] = u32::from(self.string_counter);
            p[3] = u32::from(TYPE);
            p[4] = high_u32(target);
            p[5] = low_u32(target);
            p[6] = high_u32(mask);
            p[7] = low_u32(mask);
            p[8] = self.message.cost.get();
            p[9] = self.message.key_length.get().min(32);
            p[12] = gpu_usize_u32(self.prefix_nibbles.len(), "ALTCHA prefix length")?;
            // Compare the first SHA-256 digest block for all candidates.
            // Additional PBKDF2 blocks are needed only when all 64 initial
            // nibbles matched; store their expected values after U1 blocks.
            for (index, nibble) in self.prefix_nibbles.iter().take(64).enumerate() {
                p[32 + index / 8] |= u32::from(*nibble) << ((7 - index % 8) * 4);
            }
            p[16..20].copy_from_slice(&be_words_4(&self.message.salt));
            p[20..24].copy_from_slice(&be_words_4(&self.message.nonce));
            if let Some(prefix) = self
                .sha_prefix
                .as_ref()
                .or_else(|| self.pbkdf2_input.as_ref().map(|input| &input.key_prefix))
            {
                p[24] = if self.pbkdf2_input.is_some() { 2 } else { 1 };
                p[13] = low_u32(prefix.fixed_len);
                p[14] = high_u32(prefix.fixed_len);
                p[15] = gpu_usize_u32(prefix.tail_len, "SHA-256 partial block length")?;
                p[40..48].copy_from_slice(&prefix.state);
                p[48..64].copy_from_slice(&be_words_16(&prefix.tail));
            }
            if let Some(input) = &self.pbkdf2_input {
                p[25] = gpu_usize_u32(input.u1_blocks.len() / 16, "PBKDF2 salt block count")?;
                p[26] = gpu_usize_u32(input.salt.len(), "PBKDF2 salt length")?;
            }
            if self.prefix_nibbles.len() > 64 {
                if self.pbkdf2_input.is_none() {
                    return Err(GpuError::new(
                        "multi-block prefixes require general PBKDF2 input",
                    ));
                }
                p[28] = gpu_usize_u32(p.len(), "PBKDF2 prefix buffer offset")?;
                let suffix_nibbles = &self.prefix_nibbles[64..];
                let suffix_words = suffix_nibbles.len().div_ceil(8);
                let start = p.len();
                p.resize(start + suffix_words, 0);
                for (index, nibble) in suffix_nibbles.iter().enumerate() {
                    p[start + index / 8] |= u32::from(*nibble) << ((7 - index % 8) * 4);
                }
            }

            if let Some((nonce, hash)) = gpu.dispatch_altcha(&p, wgs).await? {
                self.attempted_nonces = base + batch - self.start_nonce;
                let Some(verified_key) = verify_altcha::<TYPE>(self, nonce, hash, target, mask)
                else {
                    return Err(GpuError::new("GPU returned an invalid Altcha proof"));
                };
                self.verified_key = Some(verified_key);
                return Ok(Some(GpuSolution {
                    nonce,
                    hash,
                    dispatched_hashes: self.attempted_nonces,
                }));
            }
            base += batch;
        }
        self.attempted_nonces = limit.saturating_sub(self.start_nonce);
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

fn verify_altcha<const TYPE: u8>(
    solver: &AltchaSha256Solver,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> Option<Vec<u8>> {
    let message = &solver.message;
    let prefix_nibbles = &solver.prefix_nibbles;
    if (!solver.string_counter && nonce > u64::from(u32::MAX))
        || (solver.string_counter && nonce >= (1_u64 << 53))
        || prefix_nibbles.len() > solver.output_key_length.saturating_mul(2)
    {
        return None;
    }
    let suffix = if solver.string_counter {
        nonce.to_string().into_bytes()
    } else {
        nonce.to_be_bytes()[4..].to_vec()
    };
    let mut derived_key = if message.pbkdf2 {
        let input = solver.pbkdf2_input.as_ref();
        let mut password = Vec::with_capacity(
            input.map_or(16, |value| value.key_prefix.fixed_bytes.len()) + suffix.len(),
        );
        if let Some(value) = input {
            password.extend_from_slice(&value.key_prefix.fixed_bytes);
        } else {
            password.extend_from_slice(&message.nonce);
        }
        password.extend_from_slice(&suffix);
        let salt = input.map_or(message.salt.as_slice(), |value| value.salt.as_slice());
        let mut output = Vec::new();
        let len = solver.output_key_length.max(32);
        output.try_reserve_exact(len).ok()?;
        output.resize(len, 0);
        pbkdf2::pbkdf2::<pbkdf2::hmac::Hmac<Sha256>>(
            &password,
            salt,
            message.cost.get(),
            &mut output,
        )
        .ok()?;
        output
    } else {
        let mut first = Vec::with_capacity(
            solver
                .sha_prefix
                .as_ref()
                .map_or(32, |prefix| prefix.fixed_bytes.len())
                + suffix.len(),
        );
        if let Some(prefix) = &solver.sha_prefix {
            first.extend_from_slice(&prefix.fixed_bytes);
        } else {
            first.extend_from_slice(&message.salt);
            first.extend_from_slice(&message.nonce);
        }
        first.extend_from_slice(&suffix);
        let mut digest: [u8; 32] = Sha256::digest(&first).into();
        // ALTCHA v2 truncates only the final digest, after all iterations.
        for _ in 1..message.cost.get() {
            digest = Sha256::digest(digest).into();
        }
        digest.to_vec()
    };
    let first_digest: [u8; 32] = derived_key[..32].try_into().ok()?;
    let first_hash = be_words_8(&first_digest);
    let prefix_matches = prefix_nibbles.iter().enumerate().all(|(index, expected)| {
        let byte = derived_key[index / 2];
        let actual = if index.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0x0f
        };
        actual == *expected
    });
    let condition = if TYPE == crate::solver::SOLVE_TYPE_MASK && !prefix_nibbles.is_empty() {
        true // Full digest prefix supersedes the 64-bit target/mask.
    } else {
        result_matches_full64::<TYPE>(hash, target, mask)
    };
    if first_hash != hash || !prefix_matches || !condition {
        return None;
    }
    derived_key.truncate(solver.output_key_length);
    Some(derived_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbkdf2_u1_salt_blocks_have_correct_padding_and_bit_length() -> Result<(), GpuError> {
        for length in [
            0_usize, 1, 51, 52, 55, 56, 59, 60, 63, 64, 115, 116, 119, 120, 127, 128, 4097,
        ] {
            let salt: Vec<u8> = (0..length).map(|i| i as u8).collect();
            let input = AltchaPbkdf2Input::new(&salt, &[0x15; 96])?;
            let packed: Vec<u8> = input
                .u1_blocks
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect();
            assert_eq!(&packed[..length], salt);
            assert_eq!(&packed[length..length + 4], &[0, 0, 0, 1]);
            assert_eq!(packed[length + 4], 0x80);
            assert!(packed[length + 5..packed.len() - 8].iter().all(|b| *b == 0));
            let bit_len = u64::from_be_bytes(
                packed[packed.len() - 8..]
                    .try_into()
                    .map_err(|_| GpuError::new("invalid SHA padding tail"))?,
            );
            assert_eq!(
                bit_len,
                (u64::try_from(length).map_err(|_| GpuError::new("invalid length"))? + 68) * 8
            );
            assert_eq!(packed.len() % 64, 0);
        }
        Ok(())
    }
}
