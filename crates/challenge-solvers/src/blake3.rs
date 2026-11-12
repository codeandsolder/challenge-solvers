#![expect(
    clippy::inline_always,
    reason = "fixed-round BLAKE3 kernels force inlining across backend wrappers to preserve codegen"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "BLAKE3 compression uses the canonical a/b/c/d and message-word notation"
)]
#[macro_use]
mod loop_macros;

pub mod portable;

/// Initial hash values for BLAKE3
pub const IV: [u32; 8] = crate::sha256::IV;

pub const FLAG_CHUNK_START: u32 = 0x01;
pub const FLAG_CHUNK_END: u32 = 0x02;
pub const FLAG_ROOT: u32 = 0x08;

const PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

const MESSAGE_SCHEDULE: [[usize; 16]; 7] = {
    let mut ix = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut out = [ix; 7];

    let mut i = 1;
    while i < 7 {
        let mut j = 0;
        let mut new_ix = [0; 16];
        while j < 16 {
            new_ix[j] = ix[PERMUTATION[j]];
            j += 1;
        }
        ix = new_ix;
        out[i] = new_ix;
        i += 1;
    }

    out
};

// The mixing function, G, which mixes either a column or a diagonal.
#[inline(always)]
const fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
const fn round_fixed(state: &mut [u32; 16], m: &[u32; 16], round: usize) {
    // Mix the columns.
    g(
        state,
        0,
        4,
        8,
        12,
        m[MESSAGE_SCHEDULE[round][0]],
        m[MESSAGE_SCHEDULE[round][1]],
    );
    g(
        state,
        1,
        5,
        9,
        13,
        m[MESSAGE_SCHEDULE[round][2]],
        m[MESSAGE_SCHEDULE[round][3]],
    );
    g(
        state,
        2,
        6,
        10,
        14,
        m[MESSAGE_SCHEDULE[round][4]],
        m[MESSAGE_SCHEDULE[round][5]],
    );
    g(
        state,
        3,
        7,
        11,
        15,
        m[MESSAGE_SCHEDULE[round][6]],
        m[MESSAGE_SCHEDULE[round][7]],
    );
    // Mix the diagonals.
    g(
        state,
        0,
        5,
        10,
        15,
        m[MESSAGE_SCHEDULE[round][8]],
        m[MESSAGE_SCHEDULE[round][9]],
    );
    g(
        state,
        1,
        6,
        11,
        12,
        m[MESSAGE_SCHEDULE[round][10]],
        m[MESSAGE_SCHEDULE[round][11]],
    );
    g(
        state,
        2,
        7,
        8,
        13,
        m[MESSAGE_SCHEDULE[round][12]],
        m[MESSAGE_SCHEDULE[round][13]],
    );
    g(
        state,
        3,
        4,
        9,
        14,
        m[MESSAGE_SCHEDULE[round][14]],
        m[MESSAGE_SCHEDULE[round][15]],
    );
}

#[inline(always)]
pub fn round_gated(state: &mut [u32; 16], m: &[u32]) {
    debug_assert!(m.len() <= 16);
    // Mix the columns.
    if m.len() >= 2 {
        g(state, 0, 4, 8, 12, m[0], m[1]);
    }
    if m.len() >= 4 {
        g(state, 1, 5, 9, 13, m[2], m[3]);
    }
    if m.len() >= 6 {
        g(state, 2, 6, 10, 14, m[4], m[5]);
    }
    if m.len() >= 8 {
        g(state, 3, 7, 11, 15, m[6], m[7]);
    }
    // Mix the diagonals.
    if m.len() >= 10 {
        g(state, 0, 5, 10, 15, m[8], m[9]);
    }
    if m.len() >= 12 {
        g(state, 1, 6, 11, 12, m[10], m[11]);
    }
    if m.len() >= 14 {
        g(state, 2, 7, 8, 13, m[12], m[13]);
    }
    if m.len() == 16 {
        g(state, 3, 4, 9, 14, m[14], m[15]);
    }
}

pub fn ingest_message_prefix(
    state: [u32; 8],
    m: &[u32],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut full_state = [
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
        state[6],
        state[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        crate::low_u32(counter),
        0,
        block_len,
        flags,
    ];
    round_gated(&mut full_state, m);
    full_state
}

#[inline(always)]
/// Truncated BLAKE3 compression function.
pub fn compress8(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 8] {
    let counter_low = crate::low_u32(counter);
    let counter_high = crate::high_u32(counter);
    #[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter_low,       counter_high,      block_len,         flags,
    ];
    let block = *block_words;

    repeat7!(i, {
        round_fixed(&mut state, &block, i);
    });

    for i in 0..8 {
        state[i] ^= state[i + 8];
    }
    let mut output = [0_u32; 8];
    output.copy_from_slice(&state[..8]);
    output
}

#[cfg(test)]
mod tests {
    use sha2::Digest;

    use super::*;

    #[test]
    fn test_compress_unchained() {
        for blockc in 1..=4 {
            let mut chaining_value = IV;

            let mut msg = Vec::new();
            let mut ctr = 0usize;
            while msg.len() < 64 * blockc {
                let hash = ::sha2::Sha256::digest(ctr.to_le_bytes());
                msg.extend_from_slice(&hash);
                ctr = ctr.wrapping_add(1);
            }
            assert_eq!(msg.len(), 64 * blockc);

            let mut hasher = blake3::Hasher::new();
            hasher.update(&msg);
            let hash = hasher.finalize();
            let hash = hash.as_bytes();

            let count_chunks = msg.len().div_ceil(64);
            ctr = 0;
            let (chunks, remainder) = msg.as_chunks::<64>();
            assert_eq!(remainder, &[0_u8; 0]);
            let mut output = [0u32; 8];
            for chunk in chunks {
                let block = core::array::from_fn(|i| {
                    let base = i * 4;
                    u32::from_le_bytes([
                        chunk[base],
                        chunk[base + 1],
                        chunk[base + 2],
                        chunk[base + 3],
                    ])
                });

                let this_flag = if ctr == 0 { FLAG_CHUNK_START } else { 0 }
                    | if count_chunks == ctr + 1 {
                        FLAG_CHUNK_END | FLAG_ROOT
                    } else {
                        0
                    };
                output = compress8(&chaining_value, &block, 0, 64, this_flag);
                chaining_value = output;
                ctr += 1;
            }

            let mut expected = [0u32; 8];
            for (i, word) in expected.iter_mut().enumerate() {
                let base = i * 4;
                *word = u32::from_le_bytes([
                    hash[base],
                    hash[base + 1],
                    hash[base + 2],
                    hash[base + 3],
                ]);
            }
            assert_eq!(output, expected, "output mismatch (blockc: {blockc})");
        }
    }
}
