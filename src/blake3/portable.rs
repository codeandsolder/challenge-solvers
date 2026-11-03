//! Portable-SIMD BLAKE3 compression core shared by architecture wrappers.
use super::MESSAGE_SCHEDULE;
use core::simd::Simd;
#[inline(always)]
fn ror<const N: usize>(x: Simd<u32, N>, s: u32) -> Simd<u32, N> {
    // The shifted halves are disjoint, so XOR is equivalent to OR and gives
    // LLVM the same straight-line operation used by the former AVX2 kernel.
    (x >> s) ^ (x << (32 - s))
}
#[inline(always)]
pub(super) fn g<const N: usize>(
    a: &mut Simd<u32, N>,
    b: &mut Simd<u32, N>,
    c: &mut Simd<u32, N>,
    d: &mut Simd<u32, N>,
    x: Simd<u32, N>,
    y: Simd<u32, N>,
) {
    *a = *a + *b + x;
    *d = ror(*d ^ *a, 16);
    *c += *d;
    *b = ror(*b ^ *c, 12);
    *a = *a + *b + y;
    *d = ror(*d ^ *a, 8);
    *c += *d;
    *b = ror(*b ^ *c, 7);
}
#[cfg_attr(target_arch = "wasm32", inline(never))]
#[cfg_attr(not(target_arch = "wasm32"), inline(always))]
pub fn compress<const N: usize, const CONSTANT_WORD_COUNT: usize, const PATCH: usize>(
    v: &mut [Simd<u32, N>; 16],
    tpl: &[u32; 16],
    patch: Simd<u32, N>,
) {
    repeat7!(round, {
        macro_rules! mix {
            ($a:literal,$b:literal,$c:literal,$d:literal,$x:literal,$y:literal) => {{
                let ix = MESSAGE_SCHEDULE[round][$x];
                let iy = MESSAGE_SCHEDULE[round][$y];
                let x = if ix == PATCH {
                    patch
                } else {
                    Simd::splat(tpl[ix])
                };
                let y = if iy == PATCH {
                    patch
                } else {
                    Simd::splat(tpl[iy])
                };
                // Copying the four words into locals is both safe and easier for LLVM
                // than `get_disjoint_mut`: the previous formulation produced materially
                // larger/slower Cerberus specializations despite these literal indices.
                let mut a = v[$a];
                let mut b = v[$b];
                let mut c = v[$c];
                let mut d = v[$d];
                g(&mut a, &mut b, &mut c, &mut d, x, y);
                v[$a] = a;
                v[$b] = b;
                v[$c] = c;
                v[$d] = d;
            }};
        }
        if round > 0 || CONSTANT_WORD_COUNT < 2 {
            mix!(0, 4, 8, 12, 0, 1);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 4 {
            mix!(1, 5, 9, 13, 2, 3);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 6 {
            mix!(2, 6, 10, 14, 4, 5);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 8 {
            mix!(3, 7, 11, 15, 6, 7);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 10 {
            mix!(0, 5, 10, 15, 8, 9);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 12 {
            mix!(1, 6, 11, 12, 10, 11);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 14 {
            mix!(2, 7, 8, 13, 12, 13);
        }
        if round > 0 || CONSTANT_WORD_COUNT < 16 {
            mix!(3, 4, 9, 14, 14, 15);
        }
    });
    repeat8!(i, {
        v[i] ^= v[i + 8];
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_compress<const LANES: usize>() {
        let mut initial = [0u32; 16];
        initial[..8].copy_from_slice(&crate::blake3::IV);
        initial[8..12].copy_from_slice(&crate::blake3::IV[..4]);
        initial[14] = 4;
        initial[15] = crate::blake3::FLAG_CHUNK_START
            | crate::blake3::FLAG_CHUNK_END
            | crate::blake3::FLAG_ROOT;

        let mut state: [Simd<u32, LANES>; 16] =
            core::array::from_fn(|word| Simd::splat(initial[word]));
        let mut block = [0u32; 16];
        block[0] = u32::from_le_bytes(*b"IETF");
        compress::<LANES, 0, 16>(&mut state, &block, Simd::splat(0));

        let expected = [
            0x1ede_a283,
            0xabe6_f4e6,
            0x2489_6868,
            0xcfc0_4e8f,
            0x9470_c54c,
            0xff82_a646,
            0xd6b4_cbd1,
            0xe281_5116,
        ];
        for lane in 0..LANES {
            let actual: [u32; 8] = core::array::from_fn(|word| state[word][lane]);
            assert_eq!(actual, expected, "lane {lane} of {LANES}");
        }
    }

    #[test]
    fn compression_matches_reference_for_8_and_16_lanes() {
        check_compress::<8>();
        check_compress::<16>();
    }
}
