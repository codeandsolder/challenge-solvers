// SHA-1(salt UTF-8 bytes || decimal uint32 nonce) with arbitrary salt length.
// Prefix full blocks are compressed once on CPU. Each candidate adds only
// decimal digits and SHA-1 final padding (one or two blocks).
// P: base, batch, tail_len, prefix_bytes_lo, prefix_bytes_hi,
//    initial_state[5], tail_block[16] (big-endian), target_digest[5].
// sha1_core.wgsl is prepended before this source.
struct Sha1DecimalResult {
    found: atomic<u32>,
    nonce: u32,
    spare0: u32,
    spare1: u32,
};
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read_write> R: Sha1DecimalResult;

fn write_byte(words: ptr<function, array<u32, 16>>, index: u32, value: u32) {
    let wi = index / 4u;
    (*words)[wi] = (*words)[wi] | (value << ((3u - index % 4u) * 8u));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P[1] || atomicLoad(&R.found) != 0u) {
        return;
    }
    let nonce = P[0] + gid.x;
    var divisor = 1u;
    var digits = 1u;
    loop {
        if (nonce / divisor < 10u) {
            break;
        }
        divisor = divisor * 10u;
        digits += 1u;
    }

    var first: array<u32, 16>;
    var second: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        first[i] = P[10u + i];
        second[i] = 0u;
    }
    for (var i = 0u; i < digits; i = i + 1u) {
        let byte = 48u + (nonce / divisor) % 10u;
        let pos = P[2] + i;
        if (pos < 64u) {
            write_byte(&first, pos, byte);
        } else {
            write_byte(&second, pos - 64u, byte);
        }
        divisor /= 10u;
    }
    let pad_pos = P[2] + digits;
    if (pad_pos < 64u) {
        write_byte(&first, pad_pos, 128u);
    } else {
        write_byte(&second, pad_pos - 64u, 128u);
    }

    // Full 64-bit message length in bits, implemented using u32 limbs.
    let total_lo = P[3] + digits;
    let carry = select(0u, 1u, total_lo < P[3]);
    let total_hi = P[4] + carry;
    let bit_lo = total_lo << 3u;
    let bit_hi = (total_hi << 3u) | (total_lo >> 29u);

    var state: array<u32, 5>;
    for (var i = 0u; i < 5u; i = i + 1u) {
        state[i] = P[5u + i];
    }
    if (pad_pos < 56u) {
        first[14] = bit_hi;
        first[15] = bit_lo;
        state = sha1_compress(state, first);
    } else {
        second[14] = bit_hi;
        second[15] = bit_lo;
        state = sha1_compress(state, first);
        state = sha1_compress(state, second);
    }

    var matches = true;
    for (var i = 0u; i < 5u; i = i + 1u) {
        matches = matches && (state[i] == P[26u + i]);
    }
    if (matches) {
        if (atomicExchange(&R.found, 1u) == 0u) {
            R.nonce = nonce;
        }
    }
}

