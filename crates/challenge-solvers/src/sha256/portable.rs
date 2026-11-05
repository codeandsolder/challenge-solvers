//! Portable-SIMD SHA-256 compression core shared by architecture wrappers.
use super::K32;
use core::simd::Simd;
#[macro_use]
#[path = "loop_macros.rs"]
mod loop_macros;
#[inline(always)]
fn ror<const N: usize>(x: Simd<u32, N>, s: u32) -> Simd<u32, N> {
    (x >> s) | (x << (32 - s))
}
#[inline(always)]
fn ss0<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 7) ^ ror(x, 18) ^ (x >> 3)
}
#[inline(always)]
fn ss1<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 17) ^ ror(x, 19) ^ (x >> 10)
}
#[inline(always)]
fn bs0<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 2) ^ ror(x, 13) ^ ror(x, 22)
}
#[inline(always)]
fn bs1<const N: usize>(x: Simd<u32, N>) -> Simd<u32, N> {
    ror(x, 6) ^ ror(x, 11) ^ ror(x, 25)
}
#[inline(always)]
fn ch<const N: usize>(x: Simd<u32, N>, y: Simd<u32, N>, z: Simd<u32, N>) -> Simd<u32, N> {
    (x & y) ^ ((!x) & z)
}
#[inline(always)]
fn maj<const N: usize>(x: Simd<u32, N>, y: Simd<u32, N>, z: Simd<u32, N>) -> Simd<u32, N> {
    (x & y) ^ (x & z) ^ (y & z)
}
#[cfg_attr(target_arch = "wasm32", inline(never))]
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub fn multiway_arx<const N: usize, const BEGIN: usize>(
    state: &mut [Simd<u32, N>; 8],
    block: &mut [Simd<u32, N>; 16],
) {
    let [a, b, c, d, e, f, g, h] = &mut *state;
    repeat64!(i, {
        if i >= BEGIN {
            let w = if i < 16 {
                block[i]
            } else {
                let x = ss1(block[(i - 2) % 16])
                    + block[(i - 7) % 16]
                    + ss0(block[(i - 15) % 16])
                    + block[i % 16];
                block[i % 16] = x;
                x
            };
            let t1 = *h + bs1(*e) + ch(*e, *f, *g) + Simd::splat(K32[i]) + w;
            let t2 = bs0(*a) + maj(*a, *b, *c);
            *h = *g;
            *g = *f;
            *f = *e;
            *e = *d + t1;
            *d = *c;
            *c = *b;
            *b = *a;
            *a = t1 + t2;
        }
    });
}
#[cfg_attr(target_arch = "wasm32", inline(never))]
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub fn bcst_multiway_arx<const N: usize, const LEAD: usize>(
    state: &mut [Simd<u32, N>; 8],
    wk: &[u32; 64],
) {
    let [a, b, c, d, e, f, g, h] = &mut *state;
    repeat64!(i, {
        let w = Simd::splat(if i < LEAD { K32[i] } else { wk[i] });
        let t1 = *h + bs1(*e) + ch(*e, *f, *g) + w;
        let t2 = bs0(*a) + maj(*a, *b, *c);
        *h = *g;
        *g = *f;
        *f = *e;
        *e = *d + t1;
        *d = *c;
        *c = *b;
        *b = *a;
        *a = t1 + t2;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::{IV, deterministic_test_word, digest_block, do_message_schedule_k_w};

    fn check_digest_block_equivalence<const LANES: usize>() -> Result<(), core::num::TryFromIntError>
    {
        let mut states = [[0_u32; 8]; LANES];
        let mut blocks = [[0_u32; 16]; LANES];
        for lane_index in 0..LANES {
            let lane = u32::try_from(lane_index)?;
            for (word, state) in states[lane_index].iter_mut().enumerate() {
                *state = deterministic_test_word(lane, u32::try_from(word)?, 0x5354_4154);
            }
            for (word, block) in blocks[lane_index].iter_mut().enumerate() {
                *block = deterministic_test_word(lane, u32::try_from(word)?, 0x424c_4f43);
            }
        }
        let mut state_simd: [Simd<u32, LANES>; 8] = core::array::from_fn(|word| {
            Simd::from_array(core::array::from_fn(|lane| states[lane][word]))
        });
        let state_save = state_simd;
        let mut block_simd: [Simd<u32, LANES>; 16] = core::array::from_fn(|word| {
            Simd::from_array(core::array::from_fn(|lane| blocks[lane][word]))
        });

        for lane in 0..LANES {
            digest_block(&mut states[lane], &blocks[lane]);
        }
        multiway_arx::<LANES, 0>(&mut state_simd, &mut block_simd);
        for word in 0..8 {
            state_simd[word] += state_save[word];
        }

        let actual: [[u32; 8]; LANES] =
            core::array::from_fn(|lane| core::array::from_fn(|word| state_simd[word][lane]));
        assert_eq!(states, actual);
        Ok(())
    }

    fn check_broadcast_block<const LANES: usize>() {
        let mut schedule = [0u32; 64];
        schedule[0] = u32::from_be_bytes(*b"abc\x80");
        schedule[15] = 3 * 8;
        do_message_schedule_k_w(&mut schedule);

        let mut state: [Simd<u32, LANES>; 8] = core::array::from_fn(|word| Simd::splat(IV[word]));
        let state_save = state;
        bcst_multiway_arx::<LANES, 0>(&mut state, &schedule);
        for word in 0..8 {
            state[word] += state_save[word];
        }

        let expected = [
            0xba78_16bf,
            0x8f01_cfea,
            0x4141_40de,
            0x5dae_2223,
            0xb003_61a3,
            0x9617_7a9c,
            0xb410_ff61,
            0xf200_15ad,
        ];
        for lane in 0..LANES {
            let actual = core::array::from_fn(|word| state[word][lane]);
            assert_eq!(actual, expected, "lane {lane}");
        }
    }

    #[test]
    fn multiway_matches_scalar_for_8_and_16_lanes() -> Result<(), core::num::TryFromIntError> {
        check_digest_block_equivalence::<8>()?;
        check_digest_block_equivalence::<16>()?;
        Ok(())
    }

    #[test]
    fn broadcast_schedule_matches_nist_vector_for_8_and_16_lanes() {
        check_broadcast_block::<8>();
        check_broadcast_block::<16>();
    }
}
