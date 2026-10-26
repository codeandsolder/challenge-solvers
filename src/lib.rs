#![feature(portable_simd)]
#![allow(clippy::inline_always)] // Hot SIMD helpers must inline into target-feature wrappers.

//! Focused Argon2id v1.3 engine for Anubis' fixed m=19456 KiB, t=2, p=1 parameters.
//!
//! The hot 1 KiB compression state is represented as portable four-lane
//! `u64` vectors from the start. This leaves register layout visible to LLVM
//! without baking x86 instructions into the algorithm. The data-independent
//! reference indices in the first half-pass are computed once process-wide and
//! shared by every reusable worker arena.

use blake2::{Blake2b256, Blake2b512, Digest};
use core::simd::{Simd, simd_swizzle};
use std::sync::OnceLock;

pub const MEMORY_KIB: usize = 19 * 1024;
const PASSES: usize = 2;
const SYNC_POINTS: usize = 4;
const SEGMENT_LENGTH: usize = MEMORY_KIB / SYNC_POINTS;
const WORDS: usize = 128;
const VECS: usize = WORDS / 4;
const ADDRESSES_IN_BLOCK: usize = WORDS;
const VERSION: u32 = 0x13;
const ARGON2_ID: u32 = 2;

type V = Simd<u64, 4>;
type State = [V; VECS];

#[derive(Clone, Copy)]
enum Backend {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
}

impl Backend {
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                Self::Avx2
            } else {
                Self::Portable
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::Portable
        }
    }
}

const ZERO_VEC: V = V::from_array([0; 4]);

#[derive(Clone, Copy)]
#[repr(align(64))]
struct Block([u64; WORDS]);

impl Block {
    const ZERO: Self = Self([0; WORDS]);
}

/// Reusable 19 MiB Argon2 memory arena.
pub struct Memory {
    blocks: Vec<Block>,
    independent_refs: &'static [usize],
    backend: Backend,
}

impl Memory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: vec![Block::ZERO; MEMORY_KIB],
            independent_refs: independent_refs(),
            backend: Backend::detect(),
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash one Anubis candidate with Argon2id v1.3, m=19456 KiB, t=2, p=1.
///
/// Returns `None` only if an internal reference index would alias the output
/// block, which is forbidden by the Argon2 addressing rules.
#[must_use]
pub fn hash(password: &[u8], salt: [u8; 8], memory: &mut Memory) -> Option<[u8; 32]> {
    let password_len = u32::try_from(password.len()).ok()?;
    let h0 = initial_hash(password, password_len, salt);
    initialize(&h0, &mut memory.blocks);
    fill(&mut memory.blocks, memory.independent_refs, memory.backend)?;
    Some(finalize(&memory.blocks[MEMORY_KIB - 1]))
}

#[allow(unsafe_code)]
fn fill(memory: &mut [Block], independent_refs: &[usize], backend: Backend) -> Option<()> {
    match backend {
        Backend::Portable => fill_all(memory, independent_refs),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Avx2 => {
            // SAFETY: `Backend::Avx2` is created only after runtime AVX2 detection.
            unsafe { fill_all_avx2(memory, independent_refs) }
        }
    }
}

#[inline(always)]
fn fill_all(memory: &mut [Block], independent_refs: &[usize]) -> Option<()> {
    for pass in 0..PASSES {
        for slice in 0..SYNC_POINTS {
            fill_segment(memory, independent_refs, pass, slice)?;
        }
    }
    Some(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
fn fill_all_avx2(memory: &mut [Block], independent_refs: &[usize]) -> Option<()> {
    fill_all(memory, independent_refs)
}

fn initial_hash(password: &[u8], password_len: u32, salt: [u8; 8]) -> [u8; 64] {
    let mut h = Blake2b512::new();
    h.update(1_u32.to_le_bytes());
    h.update(32_u32.to_le_bytes());
    h.update(19_456_u32.to_le_bytes());
    h.update(2_u32.to_le_bytes());
    h.update(VERSION.to_le_bytes());
    h.update(ARGON2_ID.to_le_bytes());
    h.update(password_len.to_le_bytes());
    h.update(password);
    h.update(8_u32.to_le_bytes());
    h.update(salt);
    h.update(0_u32.to_le_bytes());
    h.update(0_u32.to_le_bytes());
    h.finalize().into()
}

fn initialize(h0: &[u8; 64], memory: &mut [Block]) {
    for (block, index) in memory[..2].iter_mut().zip([0_u32, 1]) {
        let mut bytes = [0_u8; 1024];
        h_prime_1024(h0, &index.to_le_bytes(), &0_u32.to_le_bytes(), &mut bytes);
        for (word, chunk) in block.0.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *word = u64::from_le_bytes(*chunk);
        }
    }
}

fn h_prime_1024(a: &[u8], b: &[u8], c: &[u8], out: &mut [u8; 1024]) {
    let mut h = Blake2b512::new();
    h.update(1024_u32.to_le_bytes());
    h.update(a);
    h.update(b);
    h.update(c);
    let mut last: [u8; 64] = h.finalize().into();

    for chunk in out[..960].as_chunks_mut::<32>().0 {
        chunk.copy_from_slice(&last[..32]);
        last = Blake2b512::digest(last).into();
    }
    out[960..].copy_from_slice(&last);
}

fn finalize(block: &Block) -> [u8; 32] {
    let mut h = Blake2b256::new();
    h.update(32_u32.to_le_bytes());
    for word in block.0 {
        h.update(word.to_le_bytes());
    }
    h.finalize().into()
}

#[inline(always)]
fn fill_segment(
    memory: &mut [Block],
    independent_refs: &[usize],
    pass: usize,
    slice: usize,
) -> Option<()> {
    let data_independent = pass == 0 && slice < 2;
    let first_block = usize::from(pass == 0 && slice == 0) * 2;
    let first_cur = slice * SEGMENT_LENGTH + first_block;
    let prev = if slice == 0 && first_block == 0 {
        first_cur + MEMORY_KIB - 1
    } else {
        first_cur - 1
    };
    let mut state = load(&memory[prev]);
    let mut prev_word = memory[prev].0[0];
    let reference_area_base = if pass == 0 {
        slice * SEGMENT_LENGTH
    } else {
        MEMORY_KIB - SEGMENT_LENGTH
    };
    let reference_start = if pass != 0 && slice != SYNC_POINTS - 1 {
        (slice + 1) * SEGMENT_LENGTH
    } else {
        0
    };

    for (cur, block) in (first_cur..).zip(first_block..SEGMENT_LENGTH) {
        let reference = if data_independent {
            independent_refs[slice * SEGMENT_LENGTH + block]
        } else {
            reference_index(prev_word, reference_area_base + block - 1, reference_start)
        };

        if !compress_at(&mut state, memory, reference, cur, pass != 0) {
            return None;
        }
        prev_word = memory[cur].0[0];
    }
    Some(())
}

#[inline(always)]
fn reference_index(rand: u64, reference_area: usize, start: usize) -> usize {
    let mut map = rand & u64::from(u32::MAX);
    map = map.wrapping_mul(map) >> 32;
    let relative = reference_area - 1 - ((reference_area as u64).wrapping_mul(map) >> 32) as usize;
    let position = start + relative;
    if position >= MEMORY_KIB {
        position - MEMORY_KIB
    } else {
        position
    }
}

fn independent_refs() -> &'static [usize] {
    static REFS: OnceLock<Box<[usize]>> = OnceLock::new();
    REFS.get_or_init(build_independent_refs)
}

fn build_independent_refs() -> Box<[usize]> {
    let mut refs = vec![0; SEGMENT_LENGTH * 2].into_boxed_slice();
    for slice in 0..2 {
        let mut address = Block::ZERO;
        let mut input = Block::ZERO;
        input.0[..6].copy_from_slice(&[
            0,
            0,
            slice as u64,
            MEMORY_KIB as u64,
            PASSES as u64,
            u64::from(ARGON2_ID),
        ]);
        let first_block = usize::from(slice == 0) * 2;
        if first_block != 0 {
            update_address(&mut address, &mut input);
        }
        for block in first_block..SEGMENT_LENGTH {
            let address_index = block % ADDRESSES_IN_BLOCK;
            if address_index == 0 {
                update_address(&mut address, &mut input);
            }
            refs[slice * SEGMENT_LENGTH + block] = reference_index(
                address.0[address_index],
                slice * SEGMENT_LENGTH + block - 1,
                0,
            );
        }
    }
    refs
}

fn update_address(address: &mut Block, input: &mut Block) {
    input.0[6] = input.0[6].wrapping_add(1);
    let mut state = [ZERO_VEC; VECS];
    let mut first = Block::ZERO;
    compress(&mut state, input, &mut first, false);
    state.fill(ZERO_VEC);
    compress(&mut state, &first, address, false);
}

#[inline(always)]
fn compress_at(
    state: &mut State,
    memory: &mut [Block],
    reference: usize,
    current: usize,
    with_xor: bool,
) -> bool {
    if reference == current || reference >= memory.len() || current >= memory.len() {
        return false;
    }

    if with_xor {
        for (i, value) in state.iter_mut().enumerate() {
            *value ^= load_vec(&memory[reference], i);
            let saved = *value ^ load_vec(&memory[current], i);
            store_vec(&mut memory[current], i, saved);
        }
    } else {
        for (i, value) in state.iter_mut().enumerate() {
            *value ^= load_vec(&memory[reference], i);
            store_vec(&mut memory[current], i, *value);
        }
    }

    for i in 0..4 {
        round_rows(state, i);
    }
    for i in 0..4 {
        round_columns(state, i);
    }

    for (i, value) in state.iter_mut().enumerate() {
        *value ^= load_vec(&memory[current], i);
        store_vec(&mut memory[current], i, *value);
    }
    true
}

#[inline(always)]
fn load(block: &Block) -> State {
    core::array::from_fn(|i| load_vec(block, i))
}

#[inline(always)]
fn load_vec(block: &Block, i: usize) -> V {
    let base = i * 4;
    V::from_array([
        block.0[base],
        block.0[base + 1],
        block.0[base + 2],
        block.0[base + 3],
    ])
}

#[inline(always)]
fn store_vec(block: &mut Block, i: usize, value: V) {
    let base = i * 4;
    let lanes = value.to_array();
    block.0[base] = lanes[0];
    block.0[base + 1] = lanes[1];
    block.0[base + 2] = lanes[2];
    block.0[base + 3] = lanes[3];
}

#[inline(always)]
fn compress(state: &mut State, rhs: &Block, next: &mut Block, with_xor: bool) {
    if with_xor {
        for (i, value) in state.iter_mut().enumerate() {
            *value ^= load_vec(rhs, i);
            store_vec(next, i, *value ^ load_vec(next, i));
        }
    } else {
        for (i, value) in state.iter_mut().enumerate() {
            *value ^= load_vec(rhs, i);
            store_vec(next, i, *value);
        }
    }

    for i in 0..4 {
        round_rows(state, i);
    }
    for i in 0..4 {
        round_columns(state, i);
    }

    for (i, value) in state.iter_mut().enumerate() {
        *value ^= load_vec(next, i);
        store_vec(next, i, *value);
    }
}

const A0: usize = 0;
const A1: usize = 1;
const B0: usize = 2;
const B1: usize = 3;
const C0: usize = 4;
const C1: usize = 5;
const D0: usize = 6;
const D1: usize = 7;

#[inline(always)]
fn round_rows(state: &mut State, i: usize) {
    const IDX: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let mut v = IDX.map(|k| state[8 * i + k]);
    round(&mut v, diagonalize_rows, undiagonalize_rows);
    for (k, value) in IDX.into_iter().zip(v) {
        state[8 * i + k] = value;
    }
}

#[inline(always)]
fn round_columns(state: &mut State, i: usize) {
    let mut v = core::array::from_fn(|j| state[4 * j + i]);
    round(&mut v, diagonalize_columns, undiagonalize_columns);
    for (j, value) in v.into_iter().enumerate() {
        state[4 * j + i] = value;
    }
}

#[inline(always)]
fn round(v: &mut [V; 8], diagonalize: fn(&mut [V; 8]), undiagonalize: fn(&mut [V; 8])) {
    g1(v);
    g2(v);
    diagonalize(v);
    g1(v);
    g2(v);
    undiagonalize(v);
}

#[inline(always)]
fn blamka(x: V, y: V) -> V {
    let lo = V::splat(u64::from(u32::MAX));
    let product = (x & lo) * (y & lo);
    x + y + product + product
}

#[inline(always)]
fn rotr<const N: u32>(x: V) -> V {
    (x >> u64::from(N)) | (x << u64::from(64 - N))
}

#[inline(always)]
fn g1(v: &mut [V; 8]) {
    for (a, b, c, d) in [(A0, B0, C0, D0), (A1, B1, C1, D1)] {
        v[a] = blamka(v[a], v[b]);
        v[d] = rotr::<32>(v[d] ^ v[a]);
        v[c] = blamka(v[c], v[d]);
        v[b] = rotr::<24>(v[b] ^ v[c]);
    }
}

#[inline(always)]
fn g2(v: &mut [V; 8]) {
    for (a, b, c, d) in [(A0, B0, C0, D0), (A1, B1, C1, D1)] {
        v[a] = blamka(v[a], v[b]);
        v[d] = rotr::<16>(v[d] ^ v[a]);
        v[c] = blamka(v[c], v[d]);
        v[b] = rotr::<63>(v[b] ^ v[c]);
    }
}

#[inline(always)]
fn diagonalize_rows(v: &mut [V; 8]) {
    for index in [B0, B1] {
        v[index] = simd_swizzle!(v[index], [1, 2, 3, 0]);
    }
    for index in [C0, C1] {
        v[index] = simd_swizzle!(v[index], [2, 3, 0, 1]);
    }
    for index in [D0, D1] {
        v[index] = simd_swizzle!(v[index], [3, 0, 1, 2]);
    }
}

#[inline(always)]
fn undiagonalize_rows(v: &mut [V; 8]) {
    for index in [B0, B1] {
        v[index] = simd_swizzle!(v[index], [3, 0, 1, 2]);
    }
    for index in [C0, C1] {
        v[index] = simd_swizzle!(v[index], [2, 3, 0, 1]);
    }
    for index in [D0, D1] {
        v[index] = simd_swizzle!(v[index], [1, 2, 3, 0]);
    }
}

#[inline(always)]
fn diagonalize_columns(v: &mut [V; 8]) {
    let b0 = v[B0];
    let b1 = v[B1];
    v[B0] = simd_swizzle!(b0, b1, [1, 4, 3, 6]);
    v[B1] = simd_swizzle!(b0, b1, [5, 0, 7, 2]);
    v.swap(C0, C1);
    let d0 = v[D0];
    let d1 = v[D1];
    v[D0] = simd_swizzle!(d0, d1, [5, 0, 7, 2]);
    v[D1] = simd_swizzle!(d0, d1, [1, 4, 3, 6]);
}

#[inline(always)]
fn undiagonalize_columns(v: &mut [V; 8]) {
    let b0 = v[B0];
    let b1 = v[B1];
    v[B0] = simd_swizzle!(b0, b1, [5, 0, 7, 2]);
    v[B1] = simd_swizzle!(b0, b1, [1, 4, 3, 6]);
    v.swap(C0, C1);
    let d0 = v[D0];
    let d1 = v[D1];
    v[D0] = simd_swizzle!(d0, d1, [1, 4, 3, 6]);
    v[D1] = simd_swizzle!(d0, d1, [5, 0, 7, 2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon2::{Argon2, Block as ReferenceBlock};

    fn reference_hash(
        reference: &Argon2<'_>,
        memory: &mut [ReferenceBlock],
        data: &[u8],
        salt: [u8; 8],
    ) -> [u8; 32] {
        let mut expected = [0_u8; 32];
        let result = reference.hash_password_into_with_memory(data, &salt, &mut expected, memory);
        assert!(result.is_ok());
        expected
    }

    #[test]
    fn matches_reference_for_reused_memory() {
        let data = b"portable SIMD says hello";
        let reference = Argon2::default();
        let mut reference_memory =
            vec![ReferenceBlock::default(); reference.params().block_count()];
        let mut memory = Memory::new();

        for nonce in [0_u64, 1, 2, 17, u64::from(u32::MAX)] {
            let salt = nonce.to_le_bytes();
            assert_eq!(
                hash(data, salt, &mut memory),
                Some(reference_hash(
                    &reference,
                    &mut reference_memory,
                    data,
                    salt
                ))
            );
        }
    }

    #[test]
    fn matches_reference_for_varied_anubis_inputs() {
        let reference = Argon2::default();
        let mut reference_memory =
            vec![ReferenceBlock::default(); reference.params().block_count()];
        let mut memory = Memory::new();

        for len in [1_usize, 7, 32, 255, 1024, 4096] {
            let data = (0..len)
                .map(|i| u8::try_from((i * 131 + len * 17) & 0xff).unwrap_or(0))
                .collect::<Vec<_>>();
            for salt in [
                u64::try_from(len)
                    .unwrap_or(0)
                    .wrapping_mul(0x9e37_79b9)
                    .to_le_bytes(),
                u64::try_from(len)
                    .unwrap_or(0)
                    .wrapping_mul(0xd1b5_4a32)
                    .to_be_bytes(),
            ] {
                assert_eq!(
                    hash(&data, salt, &mut memory),
                    Some(reference_hash(
                        &reference,
                        &mut reference_memory,
                        &data,
                        salt,
                    ))
                );
            }
        }
    }
}
