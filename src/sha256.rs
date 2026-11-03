#![expect(
    clippy::inline_always,
    reason = "fixed-round SHA-256 kernels force inlining across backend wrappers to preserve codegen"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "SHA-256 round state follows the canonical a-through-h notation"
)]
#[cfg(feature = "portable-simd")]
pub mod portable;

#[cfg(feature = "portable-simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod sha_ni;

// Initial hash values for SHA-256
pub const IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

pub const K32: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// pre-compute the message schedule for a single block
///
/// The first 16 words are the input block, the rest are computed from them
#[inline(always)]
pub const fn do_message_schedule(w: &mut [u32; 64]) {
    let mut i = 16;
    while i < 64 {
        let w15 = w[i - 15];
        let s0 = (w15.rotate_right(7)) ^ (w15.rotate_right(18)) ^ (w15 >> 3);
        let w2 = w[i - 2];
        let s1 = (w2.rotate_right(17)) ^ (w2.rotate_right(19)) ^ (w2 >> 10);
        w[i] = w[i].wrapping_add(s0);
        w[i] = w[i].wrapping_add(w[i - 7]);
        w[i] = w[i].wrapping_add(s1);
        w[i] = w[i].wrapping_add(w[i - 16]);
        i += 1;
    }
}

/// pre-compute the message schedule for a single block, adding corresponding round constants
#[inline(always)]
pub const fn do_message_schedule_k_w(w: &mut [u32; 64]) {
    do_message_schedule(w);
    let mut i = 0;
    while i < 64 {
        w[i] = w[i].wrapping_add(K32[i]);
        i += 1;
    }
}

#[cfg(all(test, feature = "portable-simd"))]
pub const fn deterministic_test_word(row: u32, column: u32, domain: u32) -> u32 {
    domain
        .wrapping_add(row.wrapping_mul(0x9e37_79b9))
        .rotate_left(column & 31)
        ^ column.wrapping_mul(0x85eb_ca6b)
}

/// A reference software implementation of SHA-256 compression function from sha2 crate
#[inline(always)]
pub fn digest_block(state: &mut [u32; 8], block: &[u32; 16]) {
    let mut tmp = [0u8; 64];
    for i in 0..16 {
        tmp[i * 4..][..4].copy_from_slice(&block[i].to_be_bytes());
    }
    sha2::block_api::compress256(state, &[tmp]);
}

/// ingest a message prefix into the state
#[cfg(feature = "portable-simd")]
#[inline(always)]
pub fn ingest_message_prefix<const LEN: usize>(state: &mut [u32; 8], w: [u32; LEN]) {
    sha2_arx::<0>(state, &w);
}

/// scalar sha2 rounds for hotstart taken verbatim from sha2 crate
#[cfg(feature = "portable-simd")]
#[inline(always)]
pub fn sha2_arx<const START: usize>(state: &mut [u32; 8], w: &[u32]) {
    let [a, b, c, d, e, f, g, h] = &mut *state;

    for i in 0..w.len() {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (*e & *f) ^ ((!*e) & *g);
        let t1 = s1
            .wrapping_add(ch)
            .wrapping_add(K32[START + i])
            .wrapping_add(w[i])
            .wrapping_add(*h);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (*a & *b) ^ (*a & *c) ^ (*b & *c);
        let t2 = s0.wrapping_add(maj);

        *h = *g;
        *g = *f;
        *f = *e;
        *e = d.wrapping_add(t1);
        *d = *c;
        *c = *b;
        *b = *a;
        *a = t1.wrapping_add(t2);
    }
}

/// scalar sha2 rounds for hotstart taken verbatim from sha2 crate, but without adding constants
#[inline(always)]
pub const fn sha2_arx_without_constants<const START: usize, const LEN: usize>(
    state: &mut [u32; 8],
    ws: [u32; LEN],
) {
    let [a, b, c, d, e, f, g, h] = &mut *state;

    let mut i = 0;
    while i < LEN {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (*e & *f) ^ ((!*e) & *g);
        let t1 = s1.wrapping_add(ch).wrapping_add(ws[i]).wrapping_add(*h);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (*a & *b) ^ (*a & *c) ^ (*b & *c);
        let t2 = s0.wrapping_add(maj);

        *h = *g;
        *g = *f;
        *f = *e;
        *e = d.wrapping_add(t1);
        *d = *c;
        *c = *b;
        *b = *a;
        *a = t1.wrapping_add(t2);
        i += 1;
    }
}
