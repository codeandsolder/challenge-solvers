#![expect(
    clippy::inline_always,
    reason = "fixed-width digit encoders are inner-loop primitives and intentionally force inlining"
)]
#[cfg_attr(
    not(feature = "internals"),
    expect(
        clippy::redundant_pub_crate,
        reason = "the same module becomes public with the internals feature, where crate-only visibility is intentional"
    )
)]
pub(crate) static DIGIT_LUT_10000_LE_EVEN: [u32; 10000 / 2] = const {
    let mut out = [0; 10000 / 2];
    let mut i = 0;
    while i < 10000 / 2 {
        let mut copy: usize = i * 2;
        let mut ds = [b'0'; 4];
        let mut j = 0;
        while j < 4 {
            ds[j] = (copy % 10).to_le_bytes()[0] + b'0';
            copy /= 10;
            j += 1;
        }
        out[i] = u32::from_be_bytes(ds);
        i += 1;
    }
    out
};

#[expect(
    dead_code,
    reason = "compile-time assertions exist only to validate LUT layout invariants"
)]
mod static_asserts {
    use super::DIGIT_LUT_10000_LE_EVEN;

    const ASSERT_DIGIT_LUT_10000_LE_0: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN[0] == u32::from_be_bytes(*b"0000")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_1: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN[1] == u32::from_be_bytes(*b"2000")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_5: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN[5] == u32::from_be_bytes(*b"0100")) as usize];

    const ASSERT_DIGIT_LUT_10000_LE_1234: [(); 1] =
        [(); (DIGIT_LUT_10000_LE_EVEN[1234] == u32::from_be_bytes(*b"8642")) as usize];
}

/// Convert up to 8 decimal digits to ASCII.
///
/// `REGISTER_BSWAP` reverses each 32-bit half of the eight-byte output, which is
/// the layout consumed by the SHA solver backends. Values wider than `N` digits
/// intentionally keep their least-significant `N` digits, matching the historical
/// SIMD formatter.
#[inline(always)]
pub fn format_decimal_8<const N: usize, const REGISTER_BSWAP: bool, const PLACEHOLDER: u8>(
    out: &mut [u8; 8],
    mut input: u32,
) {
    debug_assert!(N <= 8);
    out.fill(PLACEHOLDER);
    for i in (0..N).rev() {
        out[i] = (input % 10).to_le_bytes()[0] + b'0';
        input /= 10;
    }
    if REGISTER_BSWAP {
        out[..4].reverse();
        out[4..].reverse();
    }
}

/// Convert up to 7 octal digits to ASCII.
///
/// `REGISTER_BSWAP` reverses each 32-bit half of the eight-byte output.
/// `ADDEND` shifts the emitted digit alphabet (1 -> digits 1-8, 0 -> 0-7).
#[inline(always)]
pub fn to_octal_7<const REGISTER_BSWAP: bool, const PLACEHOLDER: u8, const ADDEND: u8>(
    out: &mut [u8; 8],
    mut input: u32,
) {
    out.fill(PLACEHOLDER);
    for i in (0..7).rev() {
        out[i] = (input & 0b111).to_le_bytes()[0] + b'0' + ADDEND;
        input >>= 3;
    }
    if REGISTER_BSWAP {
        out[..4].reverse();
        out[4..].reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_octal_7() {
        let i = 0o1_234_567;
        let mut buf = [0u8; 8];
        to_octal_7::<false, 0x80, 0>(&mut buf, i);
        assert_eq!(buf, *b"1234567\x80");
        to_octal_7::<true, 0x80, 0>(&mut buf, i);
        assert_eq!(buf, *b"4321\x80765");
        to_octal_7::<false, 0x80, 1>(&mut buf, i);
        assert_eq!(buf, *b"2345678\x80");
        to_octal_7::<true, 0x80, 1>(&mut buf, i);
        assert_eq!(buf, *b"5432\x80876");
    }

    #[test]
    fn test_format_decimal_8() {
        let mut buf = [0u8; 8];
        format_decimal_8::<8, false, 0x80>(&mut buf, 12_345_678);
        assert_eq!(buf, *b"12345678");

        format_decimal_8::<8, false, 0x80>(&mut buf, 1_0000_0000);
        assert_eq!(buf, *b"00000000");

        format_decimal_8::<7, false, 0x80>(&mut buf, 1_234_567);
        assert_eq!(buf, *b"1234567\x80");

        format_decimal_8::<7, false, 0x80>(&mut buf, 1000_0000);
        assert_eq!(buf, *b"0000000\x80");

        format_decimal_8::<8, true, 0x80>(&mut buf, 12_345_678);
        assert_eq!(buf, *b"43218765");

        format_decimal_8::<7, true, 0x80>(&mut buf, 1_234_567);
        assert_eq!(buf, *b"4321\x80765");
    }
}
