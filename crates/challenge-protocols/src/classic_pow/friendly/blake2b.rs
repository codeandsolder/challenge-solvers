//! Dedicated 128-byte BLAKE2b-256 compression for Friendly Captcha v1.
//!
//! Every v1 candidate is exactly one full BLAKE2b block; the only changing
//! message word is the final 64-bit nonce.  We retain the decoded fixed words
//! and evaluate only digest word zero, which contains the complete 32-bit
//! leading-word predicate.  The RustCrypto implementation remains the oracle.

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 3, 9, 4, 0, 7, 5, 15, 14, 1, 13],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline(always)]
fn g<const A: usize, const B: usize, const C: usize, const D: usize>(
    v: &mut [u64; 16],
    x: u64,
    y: u64,
) {
    v[A] = v[A].wrapping_add(v[B]).wrapping_add(x);
    v[D] = (v[D] ^ v[A]).rotate_right(32);
    v[C] = v[C].wrapping_add(v[D]);
    v[B] = (v[B] ^ v[C]).rotate_right(24);
    v[A] = v[A].wrapping_add(v[B]).wrapping_add(y);
    v[D] = (v[D] ^ v[A]).rotate_right(16);
    v[C] = v[C].wrapping_add(v[D]);
    v[B] = (v[B] ^ v[C]).rotate_right(63);
}

#[derive(Clone)]
pub(super) struct SingleBlock {
    words: [u64; 16],
}

impl SingleBlock {
    pub(super) fn new(message: &[u8; 128]) -> Self {
        let mut words = [0_u64; 16];
        for (slot, chunk) in words.iter_mut().zip(message.as_chunks::<8>().0) {
            *slot = u64::from_le_bytes(*chunk);
        }
        Self { words }
    }

    /// Return the first little-endian 32 digest bits of BLAKE2b-256 for an
    /// exactly 128-byte final message block, with the specified last 8 bytes.
    #[inline]
    pub(super) fn first_u32(&self, nonce: u64) -> u32 {
        let mut v = [0_u64; 16];
        v[..8].copy_from_slice(&IV);
        // Parameter block: digest length 32, fanout 1, depth 1, no key.
        v[0] ^= 0x0101_0020;
        v[8..].copy_from_slice(&IV);
        v[12] ^= 128; // one full, final block: total byte count
        v[14] = !v[14]; // last block flag

        let mut words = self.words;
        words[15] = nonce;
        for round in 0..12 {
            let s = SIGMA[round % 10];
            g::<0, 4, 8, 12>(&mut v, words[s[0]], words[s[1]]);
            g::<1, 5, 9, 13>(&mut v, words[s[2]], words[s[3]]);
            g::<2, 6, 10, 14>(&mut v, words[s[4]], words[s[5]]);
            g::<3, 7, 11, 15>(&mut v, words[s[6]], words[s[7]]);
            g::<0, 5, 10, 15>(&mut v, words[s[8]], words[s[9]]);
            g::<1, 6, 11, 12>(&mut v, words[s[10]], words[s[11]]);
            g::<2, 7, 8, 13>(&mut v, words[s[12]], words[s[13]]);
            g::<3, 4, 9, 14>(&mut v, words[s[14]], words[s[15]]);
        }
        (IV[0] ^ 0x0101_0020 ^ v[0] ^ v[8]) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake2::{Blake2b256, Digest};

    #[test]
    fn one_block_matches_rustcrypto_for_all_nonce_positions_and_inputs() {
        for seed in 0..16_u64 {
            let mut padded = [0_u8; 128];
            for (i, byte) in padded.iter_mut().enumerate() {
                *byte = seed
                    .wrapping_mul(17)
                    .wrapping_add(u64::try_from(i).unwrap_or(0)) as u8;
            }
            let block = SingleBlock::new(&padded);
            for nonce in [
                0_u64,
                1,
                u64::MAX,
                0x0123_4567_89ab_cdef,
                seed,
                0xdead_beef_0000_0000 | seed,
                0x8000_0000_0000_0000 | seed,
            ] {
                padded[120..128].copy_from_slice(&nonce.to_le_bytes());
                let digest = Blake2b256::digest(padded);
                let expected = u32::from_le_bytes(digest[..4].try_into().unwrap_or([0; 4]));
                assert_eq!(
                    block.first_u32(nonce),
                    expected,
                    "seed={seed} nonce={nonce:#x}"
                );
            }
        }
    }
}
