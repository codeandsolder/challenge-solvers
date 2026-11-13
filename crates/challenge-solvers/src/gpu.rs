//! WebGPU/native GPU solver backend for challenge-solvers message families.
//!
//! The same Rust implementation targets native `wgpu` backends and browser WebGPU on `wasm32`.

use core::fmt;
use std::{string::String, vec::Vec};

use bytemuck::{Pod, Zeroable};

mod linear_sha1;
pub use linear_sha1::{LinearSha1GpuSolver, LinearSha1Solution};
mod sha1_decimal;
pub use sha1_decimal::{Sha1DecimalGpuSolver, Sha1DecimalSolution};

mod altcha_sha512;
pub use altcha_sha512::{AltchaSha64Algorithm, AltchaSha64GpuSolver, AltchaSha64Solution};

mod scrypt;
pub use scrypt::{ScryptKdfGpu, ScryptRomixGpu};

mod solvers;
pub use solvers::{
    AltchaSha256Solver, BinarySolver, CerberusSolver, DecimalSolver, DoubleBlockSolver,
    GoAwaySolver, SingleBlockSolver,
};

mod shaders;
use shaders::sha256_core_shader_source;

mod dispatch;
mod resources;

use crate::{high_u32, low_u32};
use sha2::{Digest, Sha256};

const WG_SIZE: u32 = 64;
const STEPS: u32 = 128;
const HASHES_PER_WG: u64 = 64 * 128;
const CANCEL_INTERVAL: u32 = 16;
const COUNTER_SPACE: u64 = 1_u64 << 54;

fn gpu_u32(value: u64, field: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::new(format!("{field} exceeds the GPU u32 ABI")))
}

fn gpu_usize_u32(value: usize, field: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| GpuError::new(format!("{field} exceeds the GPU u32 ABI")))
}

fn append_fmt(output: &mut String, args: fmt::Arguments<'_>) {
    if fmt::Write::write_fmt(output, args).is_err() {
        unreachable!("writing to String cannot fail");
    }
}

fn be_words_4(bytes: &[u8; 16]) -> [u32; 4] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn be_words_8(bytes: &[u8; 32]) -> [u32; 8] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn be_words_16(bytes: &[u8; 64]) -> [u32; 16] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_be_bytes(chunks[i]))
}

fn le_words_16(bytes: &[u8; 64]) -> [u32; 16] {
    let chunks = bytes.as_chunks::<4>().0;
    core::array::from_fn(|i| u32::from_le_bytes(chunks[i]))
}

#[inline]
fn gpu_batch_target(expected_work: u64, quantum: u64) -> u64 {
    expected_work.saturating_mul(3).div_ceil(10).max(quantum)
}

/// Error returned by the GPU solver.
#[derive(Debug, Clone)]
pub struct GpuError(String);

impl GpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

/// Successful GPU proof-of-work solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSolution {
    /// Solver nonce.
    pub nonce: u64,
    /// Solver digest/output words.
    pub hash: [u32; 8],
    /// Candidate hashes submitted through the winning batch.
    ///
    /// Workgroups stop cooperatively after a hit, so this is an upper bound on hashes actually executed.
    pub dispatched_hashes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    words: [u32; 16],
}

/// Reusable GPU device/context shared by all GPU solver instances.
///
/// Native builds use the best backend selected by `wgpu`; wasm builds use browser WebGPU.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    in_buf: wgpu::Buffer,
    generic_pipeline: wgpu::ComputePipeline,
    double_pipeline: wgpu::ComputePipeline,
    goaway_pipeline: wgpu::ComputePipeline,
    binary_pipeline: wgpu::ComputePipeline,
    cerberus_pipeline: wgpu::ComputePipeline,
    altcha_pipeline: wgpu::ComputePipeline,
    // Protocol-specific shader pipelines are compiled once per logical context
    // on first use. Unused GPU families do not increase startup overhead.
    linear_sha1_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    sha1_decimal_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    altcha_sha64_pipeline: std::sync::OnceLock<wgpu::ComputePipeline>,
    generic_in_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generic_bind_group: wgpu::BindGroup,
    double_bind_group: wgpu::BindGroup,
    goaway_bind_group: wgpu::BindGroup,
    binary_bind_group: wgpu::BindGroup,
    cerberus_bind_group: wgpu::BindGroup,
    altcha_bind_group: wgpu::BindGroup,
    max_wgs: u32,
    adapter: String,
}

const fn single_block_search_space(message: &crate::message::SingleBlockMessage) -> u64 {
    if message.nonce_addend == 0 {
        900_000_000
    } else {
        1_000_000_000
    }
}

const fn single_block_fast_compatible(message: &crate::message::SingleBlockMessage) -> bool {
    if message.digit_index != 10
        || message.nonce_addend != 1_000_000_000_000_000_000
        || message.no_trailing_zeros
    {
        return false;
    }
    let words = &message.message;
    words[0] == 0x3130_3030
        && words[1] == 0x3030_3030
        && (words[2] & 0xffff_0000) == 0x3030_0000
        && words[4] == 0x0000_0080
}

fn expected_work<const TYPE: u8>(target: u64, mask: u64) -> u64 {
    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
        return u32::try_from(target)
            .ok()
            .and_then(|bits| 1_u64.checked_shl(bits))
            .unwrap_or(u64::MAX);
    }
    let compare_mask = if cfg!(feature = "compare-64bit") {
        mask
    } else {
        mask & 0xffff_ffff_0000_0000
    };
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        return 1_u64
            .checked_shl(compare_mask.count_ones())
            .unwrap_or(u64::MAX);
    }
    let bits = if cfg!(feature = "compare-64bit") {
        64
    } else {
        32
    };
    let t = if bits == 64 {
        u128::from(target)
    } else {
        u128::from(target >> 32)
    };
    let space = 1_u128 << bits;
    let success = if TYPE == crate::solver::SOLVE_TYPE_LT {
        t.max(1)
    } else {
        space.saturating_sub(t).saturating_sub(1).max(1)
    };
    u64::try_from((space / success).min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
}

fn midstate(prefix: &[u8]) -> [u32; 8] {
    let mut state = crate::sha256::IV;
    for block in prefix.as_chunks::<64>().0 {
        let words = be_words_16(block);
        crate::sha256::digest_block(&mut state, &words);
    }
    state
}

fn counter_to_decimal_nonce(counter: u64) -> u64 {
    let mut nonce = 1_u64;
    for digit in (0..18).rev() {
        nonce = nonce * 10 + ((counter >> (digit * 3)) & 7);
    }
    nonce
}

fn verify(prefix: &[u8], nonce: u64, hash: [u32; 8], mask: u64) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(nonce.to_string().as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let expected = be_words_8(&digest);
    expected == hash && (((u64::from(hash[0]) << 32) | u64::from(hash[1])) & mask) == 0
}

fn digest_single_block_message(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
) -> Option<[u32; 8]> {
    let suffix = nonce.checked_sub(message.nonce_addend)?;
    if suffix >= 1_000_000_000 {
        return None;
    }
    let mut words = message.message;
    let mut x = u32::try_from(suffix).ok()?;
    for j in 0..9 {
        let digit = x % 10;
        x /= 10;
        let pos = message.digit_index + 8 - j;
        let word = pos / 4;
        let shift = (3 - (pos % 4)) * 8;
        words[word] = (words[word] & !(0xff_u32 << shift)) | ((u32::from(b'0') + digit) << shift);
    }
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    Some(state)
}

fn result_matches<const TYPE: u8>(hash: [u32; 8], target: u64, mask: u64) -> bool {
    if TYPE == crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS {
        return crate::hash_has_leading_zero_bits(&hash, target);
    }
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    if cfg!(feature = "compare-64bit") {
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    } else {
        let value = hash[0];
        let target = high_u32(target);
        let mask = high_u32(mask);
        if TYPE == crate::solver::SOLVE_TYPE_LT {
            value < target
        } else if TYPE == crate::solver::SOLVE_TYPE_GT {
            value > target
        } else {
            (value & mask) == (target & mask)
        }
    }
}

fn verify_single_block<const TYPE: u8>(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    digest_single_block_message(message, nonce)
        .is_some_and(|expected| expected == hash && result_matches::<TYPE>(hash, target, mask))
}

fn verify_fast_single_block(
    message: &crate::message::SingleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    mask: u64,
) -> bool {
    let nonce = nonce.to_string();
    if nonce.len() != 19 {
        return false;
    }
    let mut block = [0_u8; 64];
    block[..19].copy_from_slice(nonce.as_bytes());
    block[19] = 0x80;
    block[60..64].copy_from_slice(&message.message[15].to_be_bytes());
    let words = be_words_16(&block);
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &words);
    state == hash && result_matches::<{ crate::solver::SOLVE_TYPE_MASK }>(hash, 0, mask)
}

fn verify_cerberus(
    message: &crate::message::CerberusMessage,
    nonce: u64,
    hash: [u32; 8],
    mask: u64,
) -> bool {
    let expected = match message {
        crate::message::CerberusMessage::Decimal(message) => {
            let Some(mut suffix) = nonce.checked_sub(message.nonce_addend) else {
                return false;
            };
            if suffix >= 1_000_000_000 {
                return false;
            }
            let mut bytes = message.salt_residual;
            for i in (0..9).rev() {
                bytes[message.salt_residual_len + i] = (suffix % 10).to_le_bytes()[0] + b'0';
                suffix /= 10;
            }
            let words = le_words_16(&bytes);
            let Ok(salt_residual_len) = u32::try_from(message.salt_residual_len) else {
                return false;
            };
            crate::blake3::compress8(
                &message.prefix_state,
                &words,
                0,
                salt_residual_len + 9,
                message.flags,
            )
        }
        crate::message::CerberusMessage::Binary(message) => {
            if high_u32(nonce) != message.first_word {
                return false;
            }
            let mut words = [0_u32; 16];
            words[0] = message.first_word;
            words[1] = low_u32(nonce);
            crate::blake3::compress8(
                &message.midstate,
                &words,
                0,
                8,
                crate::blake3::FLAG_CHUNK_END | crate::blake3::FLAG_ROOT,
            )
        }
    };
    expected == hash && (((u64::from(hash[0]) << 32) | u64::from(hash[1])) & mask) == 0
}

fn expected_work_full64<const TYPE: u8>(target: u64, mask: u64) -> u64 {
    if TYPE == crate::solver::SOLVE_TYPE_MASK {
        return 1_u64.checked_shl(mask.count_ones()).unwrap_or(u64::MAX);
    }
    let space = 1_u128 << 64;
    let t = u128::from(target);
    let success = if TYPE == crate::solver::SOLVE_TYPE_LT {
        t.max(1)
    } else {
        space.saturating_sub(t).saturating_sub(1).max(1)
    };
    u64::try_from((space / success).min(u128::from(u64::MAX))).unwrap_or(u64::MAX)
}

fn result_matches_full64<const TYPE: u8>(hash: [u32; 8], target: u64, mask: u64) -> bool {
    let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
    if TYPE == crate::solver::SOLVE_TYPE_LT {
        value < target
    } else if TYPE == crate::solver::SOLVE_TYPE_GT {
        value > target
    } else {
        (value & mask) == (target & mask)
    }
}

fn verify_double_block<const TYPE: u8>(
    message: &crate::message::DoubleBlockMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let Some(mut suffix) = nonce.checked_sub(message.nonce_addend) else {
        return false;
    };
    if suffix >= 1_000_000_000 {
        return false;
    }
    let mut block = message.message;
    for j in (0..9).rev() {
        let digit = u32::from((suffix % 10).to_le_bytes()[0]);
        suffix /= 10;
        let pos = crate::message::DoubleBlockMessage::DIGIT_IDX + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        block[wi] = (block[wi] & !(0xffu32 << sh)) | ((u32::from(b'0') + digit) << sh);
    }
    let mut state = message.prefix_state;
    crate::sha256::digest_block(&mut state, &block);
    let mut tail = [0u32; 16];
    let bits = message.message_length * 8;
    tail[14] = high_u32(bits);
    tail[15] = low_u32(bits);
    crate::sha256::digest_block(&mut state, &tail);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn verify_goaway<const TYPE: u8>(
    message: &crate::message::GoAwayMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    if high_u32(nonce) != message.high_word {
        return false;
    }
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&message.challenge);
    block[8] = message.high_word;
    block[9] = low_u32(nonce);
    block[10] = 0x8000_0000;
    block[15] = 320;
    let mut state = crate::sha256::IV;
    crate::sha256::digest_block(&mut state, &block);
    state == hash && result_matches::<TYPE>(hash, target, mask)
}
fn binary_templates(message: &crate::message::BinaryMessage) -> ([[u32; 16]; 2], u32) {
    let mut bytes = [[0u8; 64]; 2];
    bytes[0][..message.salt_residual_len]
        .copy_from_slice(&message.salt_residual[..message.salt_residual_len]);
    let mut ptr = message.salt_residual_len;
    let mut cur = 0usize;
    for _ in 0..message.nonce_byte_count.get() {
        ptr += 1;
        if ptr == 64 {
            cur = 1;
            ptr = 0;
        }
    }
    bytes[cur][ptr] = 0x80;
    ptr += 1;
    if ptr + 8 > 64 {
        cur = 1;
    }
    bytes[cur][56..64].copy_from_slice(
        &(u64::try_from(message.message_length)
            .unwrap_or(u64::MAX)
            .saturating_mul(8))
        .to_be_bytes(),
    );
    (
        core::array::from_fn(|b| be_words_16(&bytes[b])),
        u32::try_from(cur + 1).unwrap_or(2),
    )
}
fn verify_binary<const TYPE: u8>(
    message: &crate::message::BinaryMessage,
    nonce: u64,
    hash: [u32; 8],
    target: u64,
    mask: u64,
) -> bool {
    let (blocks, used) = binary_templates(message);
    let mut flat = [0u32; 32];
    flat[..16].copy_from_slice(&blocks[0]);
    flat[16..].copy_from_slice(&blocks[1]);
    let bytes = nonce.to_le_bytes();
    for (j, byte) in bytes
        .iter()
        .enumerate()
        .take(usize::from(message.nonce_byte_count.get()))
    {
        let pos = message.salt_residual_len + j;
        let wi = pos / 4;
        let sh = (3 - (pos % 4)) * 8;
        flat[wi] = (flat[wi] & !(0xffu32 << sh)) | (u32::from(*byte) << sh);
    }
    let mut state = message.prefix_state;
    let blocks = flat.as_chunks::<16>().0;
    crate::sha256::digest_block(&mut state, &blocks[0]);
    if used == 2 {
        crate::sha256::digest_block(&mut state, &blocks[1]);
    }
    state == hash && result_matches::<TYPE>(hash, target, mask)
}

#[cfg(test)]
mod sha512_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_conversion_is_decimal_octal_digits() {
        assert_eq!(counter_to_decimal_nonce(0), 1_000_000_000_000_000_000);
        assert_eq!(
            counter_to_decimal_nonce(0o1_234_567),
            1_000_000_000_001_234_567
        );
        assert!(counter_to_decimal_nonce(COUNTER_SPACE - 1) < u64::MAX);
    }

    #[test]
    fn full_digest_result_match_checks_past_first_64_bits() {
        let mut hash = [0_u32; 8];
        hash[2] = 0x7fff_ffff;
        assert!(result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >(hash, 65, u64::MAX,));
        hash[2] = 0x8000_0000;
        assert!(!result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >(hash, 65, u64::MAX,));
        assert!(!result_matches::<
            { crate::solver::SOLVE_TYPE_LEADING_ZERO_BITS },
        >([0; 8], 257, u64::MAX,));
    }
}
