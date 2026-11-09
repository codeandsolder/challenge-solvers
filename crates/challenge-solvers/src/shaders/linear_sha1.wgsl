// Linear.pub: SHA1(40 ASCII hex seed || decimal uint32 nonce)
// Prepended dependency: sha1_core.wgsl.
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

    let initial: array<u32, 5> = array<u32, 5>(
        0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u
    );
    let hash = sha1_compress(initial, block);
    let offset = P[2];
    if (sha1_byte(hash, offset) == 0xb0u && sha1_byte(hash, offset + 1u) == 0x0bu) {
        if (atomicExchange(&R.found, 1u) == 0u) {
            R.nonce = nonce;
        }
    }
}

