// these are mainly adapted from the sha2 crate as well,
// the core logic is verbatim, but shuffling and batch message loading overhead is removed
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

use crate::sha256::K32;
use core::simd::{simd_swizzle, u32x4, u64x2};

const K32X4: [[u32; 4]; 16] = [
    [K32[3], K32[2], K32[1], K32[0]],
    [K32[7], K32[6], K32[5], K32[4]],
    [K32[11], K32[10], K32[9], K32[8]],
    [K32[15], K32[14], K32[13], K32[12]],
    [K32[19], K32[18], K32[17], K32[16]],
    [K32[23], K32[22], K32[21], K32[20]],
    [K32[27], K32[26], K32[25], K32[24]],
    [K32[31], K32[30], K32[29], K32[28]],
    [K32[35], K32[34], K32[33], K32[32]],
    [K32[39], K32[38], K32[37], K32[36]],
    [K32[43], K32[42], K32[41], K32[40]],
    [K32[47], K32[46], K32[45], K32[44]],
    [K32[51], K32[50], K32[49], K32[48]],
    [K32[55], K32[54], K32[53], K32[52]],
    [K32[59], K32[58], K32[57], K32[56]],
    [K32[63], K32[62], K32[61], K32[60]],
];

macro_rules! schedule {
    ($v0:expr, $v1:expr, $v2:expr, $v3:expr) => {{
        let t1: u32x4 = _mm_sha256msg1_epu32($v0, $v1).into();
        let v2: u32x4 = $v2.into();
        let v3: u32x4 = $v3.into();
        let t2 = simd_swizzle!(v2, v3, [1, 2, 3, 4]);
        _mm_sha256msg2_epu32((t1 + t2).into(), $v3)
    }};
}

#[inline(always)]
pub fn prepare_state(state: &[u32; 8]) -> [__m128i; 2] {
    [
        u32x4::from_array([state[5], state[4], state[1], state[0]]).into(),
        u32x4::from_array([state[7], state[6], state[3], state[2]]).into(),
    ]
}

#[inline(always)]
pub fn feedback_ab(a: u32, b: u32) -> __m128i {
    u32x4::from_array([0, 0, b, a]).into()
}

#[inline(always)]
pub fn add_words(a: __m128i, b: __m128i) -> __m128i {
    let a: u32x4 = a.into();
    let b: u32x4 = b.into();
    (a + b).into()
}

#[inline(always)]
pub fn extract_ab(state: __m128i) -> u64 {
    let state: u64x2 = state.into();
    state.to_array()[1]
}

#[inline(always)]
pub fn extract_state(state: [__m128i; 2]) -> [u32; 8] {
    let abef: u32x4 = state[0].into();
    let cdgh: u32x4 = state[1].into();
    let [f, e, b, a] = abef.to_array();
    let [h, g, d, c] = cdgh.to_array();
    [a, b, c, d, e, f, g, h]
}

#[expect(
    unused_variables,
    reason = "default trait lane helpers intentionally ignore lane indices overridden by specialized implementations"
)]
pub trait Plucker: Sized {
    #[inline(always)]
    fn pluck_qword0(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w
    }
    #[inline(always)]
    fn pluck_qword1(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w
    }
    #[inline(always)]
    fn pluck_qword2(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w
    }
    #[inline(always)]
    fn pluck_qword3(&mut self, lane: usize, w: u32x4) -> u32x4 {
        w
    }
}

impl Plucker for () {}

// Some solver families generate many distinct message-layout specializations.
// Their round network benefits from an explicit shared code boundary, whereas
// GoAway's single hot four-lane loop benefits from inlining it. Keep both
// policies over the same algorithm, rather than copying the round network.
#[inline(never)]
#[target_feature(enable = "sse4.1,sha")]
pub(crate) fn multiway_arx_abef_cdgh_outlined<
    const BEGIN_ROUND_BY_4: usize,
    const LANES: usize,
    P: Plucker,
>(
    state: &mut [&mut [__m128i; 2]; LANES],
    block_template: &[u32; 16],
    plucker: P,
) {
    multiway_arx_abef_cdgh::<BEGIN_ROUND_BY_4, LANES, P>(state, block_template, plucker);
}

// This unrolled SHA-NI network is called in solver hot loops. Allowing ThinLTO
// to outline it can reintroduce per-batch calls and prevent specialization of
// the lane state, costing throughput in type-erased as well as typed clients.
// Keep this shared helper inlined for all its monomorphized solver users.
#[inline(always)]
#[target_feature(enable = "sse4.1,sha")]
#[expect(
    clippy::too_many_lines,
    reason = "unrolled SHA-NI round network keeps the fixed message schedule and register layout visible to codegen"
)]
pub fn multiway_arx_abef_cdgh<const BEGIN_ROUND_BY_4: usize, const LANES: usize, P: Plucker>(
    state: &mut [&mut [__m128i; 2]; LANES],
    block_template: &[u32; 16],
    mut plucker: P,
) {
    macro_rules! rounds4 {
        ($abef:ident, $cdgh:ident, $rest:expr, $i:expr) => {{
            let k = K32X4[$i];
            let kv = u32x4::from_array([k[3], k[2], k[1], k[0]]);
            let t1: [__m128i; LANES] = core::array::from_fn(|i| {
                let words: u32x4 = $rest[i].into();
                (words + kv).into()
            });
            $cdgh = core::array::from_fn(|i| _mm_sha256rnds2_epu32($cdgh[i], $abef[i], t1[i]));
            let t2: [__m128i; LANES] = core::array::from_fn(|i| {
                let words: u32x4 = t1[i].into();
                simd_swizzle!(words, [2, 3, 0, 0]).into()
            });
            $abef = core::array::from_fn(|i| _mm_sha256rnds2_epu32($abef[i], $cdgh[i], t2[i]));
        }};
    }

    macro_rules! schedule_rounds4 {
        (
        $abef:ident, $cdgh:ident,
        $w0:expr, $w1:expr, $w2:expr, $w3:expr, $w4:expr,
        $i: expr
    ) => {{
            $w4 = core::array::from_fn(|i| schedule!($w0[i], $w1[i], $w2[i], $w3[i]));
            rounds4!($abef, $cdgh, $w4, $i);
        }};
    }

    let mut abef: [_; LANES] = core::array::from_fn(|i| state[i][0]);
    let mut cdgh: [_; LANES] = core::array::from_fn(|i| state[i][1]);

    let w0_t: __m128i = u32x4::from_array([
        block_template[0],
        block_template[1],
        block_template[2],
        block_template[3],
    ])
    .into();
    let w1_t: __m128i = u32x4::from_array([
        block_template[4],
        block_template[5],
        block_template[6],
        block_template[7],
    ])
    .into();
    let w2_t: __m128i = u32x4::from_array([
        block_template[8],
        block_template[9],
        block_template[10],
        block_template[11],
    ])
    .into();
    let w3_t: __m128i = u32x4::from_array([
        block_template[12],
        block_template[13],
        block_template[14],
        block_template[15],
    ])
    .into();

    let mut w0: [_; LANES] = [w0_t; LANES];
    let mut w1: [_; LANES] = [w1_t; LANES];
    let mut w2: [_; LANES] = [w2_t; LANES];
    let mut w3: [_; LANES] = [w3_t; LANES];
    for i in 0..LANES {
        w0[i] = plucker.pluck_qword0(i, w0[i].into()).into();
        w1[i] = plucker.pluck_qword1(i, w1[i].into()).into();
        w2[i] = plucker.pluck_qword2(i, w2[i].into()).into();
        w3[i] = plucker.pluck_qword3(i, w3[i].into()).into();
    }
    let mut w4: [_; LANES] = core::array::from_fn(|i| schedule!(w0[i], w1[i], w2[i], w3[i]));

    macro_rules! gate_rnds {
        ($cutoff: literal, $($body:tt)*) => {
            if $cutoff >= BEGIN_ROUND_BY_4 * 4 {
                $($body)*
            }
        };
    }

    gate_rnds!(0, {
        rounds4!(abef, cdgh, w0, 0);
    });
    gate_rnds!(4, {
        rounds4!(abef, cdgh, w1, 1);
    });
    gate_rnds!(8, {
        rounds4!(abef, cdgh, w2, 2);
    });
    gate_rnds!(12, {
        rounds4!(abef, cdgh, w3, 3);
    });
    gate_rnds!(16, {
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 4);
    });
    gate_rnds!(20, {
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 5);
    });
    gate_rnds!(24, {
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 6);
    });
    gate_rnds!(28, {
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 7);
    });
    gate_rnds!(32, {
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 8);
    });
    gate_rnds!(36, {
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 9);
    });
    gate_rnds!(40, {
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 10);
    });
    gate_rnds!(44, {
        schedule_rounds4!(abef, cdgh, w2, w3, w4, w0, w1, 11);
    });
    gate_rnds!(48, {
        schedule_rounds4!(abef, cdgh, w3, w4, w0, w1, w2, 12);
    });
    gate_rnds!(52, {
        schedule_rounds4!(abef, cdgh, w4, w0, w1, w2, w3, 13);
    });
    gate_rnds!(56, {
        schedule_rounds4!(abef, cdgh, w0, w1, w2, w3, w4, 14);
    });
    gate_rnds!(60, {
        schedule_rounds4!(abef, cdgh, w1, w2, w3, w4, w0, 15);
    });

    for i in 0..LANES {
        state[i][0] = abef[i];
        state[i][1] = cdgh[i];
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn portable_state_helpers_preserve_sha_ni_layout() {
        let prepared = prepare_state(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let prepared_abef: u32x4 = prepared[0].into();
        let prepared_cdgh: u32x4 = prepared[1].into();
        assert_eq!(prepared_abef.to_array(), [5, 4, 1, 0]);
        assert_eq!(prepared_cdgh.to_array(), [7, 6, 3, 2]);

        let feedback: u32x4 = feedback_ab(0x1122_3344, 0x5566_7788).into();
        assert_eq!(feedback.to_array(), [0, 0, 0x5566_7788, 0x1122_3344]);

        let lhs = u32x4::from_array([u32::MAX, 2, 3, 4]);
        let rhs = u32x4::from_array([1, 20, 30, 40]);
        let sum: u32x4 = add_words(lhs.into(), rhs.into()).into();
        assert_eq!(sum.to_array(), [0, 22, 33, 44]);

        let state: __m128i = u32x4::from_array([1, 2, 3, 4]).into();
        assert_eq!(extract_ab(state), (4u64 << 32) | 3);
        assert_eq!(extract_state(prepared), [0, 1, 2, 3, 4, 5, 6, 7]);
    }
}

#[cfg(all(target_feature = "sse4.1", target_feature = "sha"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiway_arx() {
        let input_block = [
            0x6162_6380,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x0000_0018,
        ];

        let state = crate::sha256::IV;

        let mut prepared_state_0 = prepare_state(&state);
        let mut prepared_state_1 = prepare_state(&state);

        multiway_arx_abef_cdgh::<0, 2, ()>(
            &mut [&mut prepared_state_0, &mut prepared_state_1],
            &input_block,
            (),
        );

        let prepared_abef: u32x4 = prepared_state_0[0].into();
        let [_, _, b, a] = prepared_abef.to_array();

        let ab = extract_ab(prepared_state_0[0]);
        let ab_b = ab as u32;
        let ab_a = (ab >> 32) as u32;

        let mut full_message_schedule = [0u32; 64];
        full_message_schedule[0..16].copy_from_slice(&input_block);
        crate::sha256::do_message_schedule(&mut full_message_schedule);

        let mut reference_state = crate::sha256::IV;
        crate::sha256::sha2_arx::<0>(&mut reference_state, &full_message_schedule);

        assert_eq!(a, reference_state[0]);
        assert_eq!(b, reference_state[1]);
        assert_eq!(ab_a, reference_state[0], "a={} b={}", a, b);
        assert_eq!(ab_b, reference_state[1], "a={} b={}", a, b);
    }
}
