fn rotl32(value: u32, count: u32) -> u32 {
    return (value << count) | (value >> (32u - count));
}

fn sha1_byte(words: array<u32, 5>, index: u32) -> u32 {
    return (words[index / 4u] >> ((3u - index % 4u) * 8u)) & 255u;
}

// Standard SHA-1 compression of one 512-bit block from an arbitrary chaining state.
fn sha1_compress(initial: array<u32, 5>, block: array<u32, 16>) -> array<u32, 5> {
    var w: array<u32, 80>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        w[i] = block[i];
    }
    for (var i = 16u; i < 80u; i = i + 1u) {
        w[i] = rotl32(w[i - 3u] ^ w[i - 8u] ^ w[i - 14u] ^ w[i - 16u], 1u);
    }

    var a = initial[0];
    var b = initial[1];
    var c = initial[2];
    var d = initial[3];
    var e = initial[4];
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
    hash[0] = a + initial[0];
    hash[1] = b + initial[1];
    hash[2] = c + initial[2];
    hash[3] = d + initial[3];
    hash[4] = e + initial[4];
    return hash;
}
