// Linear.pub: SHA1(40 ASCII hex seed || decimal uint32 nonce)
// All protocol-valid candidates fit one SHA-1 compression block.
// P: base, batch, digest_byte_offset, reserved, seed_be_words[10]
// R: 1 if found, exact matching nonce
struct Sha1Result {
    found: atomic<u32>,
    nonce: u32,
    spare0: u32,
    spare1: u32,
};
@group(0) @binding(0) var<storage, read> P: array<u32>;
@group(0) @binding(1) var<storage, read_write> R: Sha1Result;

fn rotl32(value: u32, count: u32) -> u32 {
    return (value << count) | (value >> (32u - count));
}

fn sha1_byte(words: array<u32, 5>, index: u32) -> u32 {
    return (words[index / 4u] >> ((3u - index % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P[1] || atomicLoad(&R.found) != 0u) {
        return;
    }
    let nonce = P[0] + gid.x;
    var divisor = 1u;
    var digit_count = 1u;
    loop {
        if (nonce / divisor < 10u) {
            break;
        }
        divisor = divisor * 10u;
        digit_count = digit_count + 1u;
    }

    var block: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        block[i] = 0u;
    }
    for (var i = 0u; i < 10u; i = i + 1u) {
        block[i] = P[4u + i];
    }
    for (var i = 0u; i < digit_count; i = i + 1u) {
        let byte = 48u + ((nonce / divisor) % 10u);
        let position = 40u + i;
        block[position / 4u] = block[position / 4u] |
             (byte << ((3u - position % 4u) * 8u));
        divisor = divisor / 10u;
    }
    let pad_index = 40u + digit_count;
    block[pad_index / 4u] = block[pad_index / 4u] |
        (128u << ((3u - pad_index % 4u) * 8u));
    block[15] = (40u + digit_count) * 8u;

    var w: array<u32, 80>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        w[i] = block[i];
    }
    for (var i = 16u; i < 80u; i = i + 1u) {
        w[i] = rotl32(w[i - 3u] ^ w[i - 8u] ^ w[i - 14u] ^ w[i - 16u], 1u);
    }

    var a = 0x67452301u;
    var b = 0xefcdab89u;
    var c = 0x98badcfeu;
    var d = 0x10325476u;
    var e = 0xc3d2e1f0u;
    for (var i = 0u; i < 80u; i = i + 1u) {
        var f = 0u;
        var k = 0u;
        if (i < 20u) {
            f = (b & c) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40u) {
            f = b ^ c ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60u) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ c ^ d;
            k = 0xca62c1d6u;
        }
        let temp = rotl32(a, 5u) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl32(b, 30u);
        b = a;
        a = temp;
    }
    var hash: array<u32, 5>;
    hash[0] = a + 0x67452301u;
    hash[1] = b + 0xefcdab89u;
    hash[2] = c + 0x98badcfeu;
    hash[3] = d + 0x10325476u;
    hash[4] = e + 0xc3d2e1f0u;
    let offset = P[2];
    if (sha1_byte(hash, offset) == 0xb0u && sha1_byte(hash, offset + 1u) == 0x0bu) {
        if (atomicExchange(&R.found, 1u) == 0u) {
            R.nonce = nonce;
        }
    }
}

