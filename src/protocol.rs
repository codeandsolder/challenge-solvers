#[cfg(feature = "altcha")]
use core::ops::Deref;

#[cfg(feature = "altcha")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A fixed length hex string
pub struct FixedHexString<const N: usize>(pub [u8; N]);

#[cfg(feature = "altcha")]
impl<const N: usize> Deref for FixedHexString<N> {
    type Target = [u8; N];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "altcha")]
impl<'de, const N: usize> serde::Deserialize<'de> for FixedHexString<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor<const N: usize>;
        impl<const N: usize> serde::de::Visitor<'_> for Visitor<N> {
            type Value = FixedHexString<N>;

            fn expecting(&self, formatter: &mut alloc::fmt::Formatter) -> alloc::fmt::Result {
                formatter.write_str(&alloc::format!("a hex string of length {}", N * 2))
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<FixedHexString<N>, E>
            where
                E: serde::de::Error,
            {
                let mut res = [0u8; N];
                if v.len() > N * 2 {
                    return Err(serde::de::Error::custom("hex string is too long"));
                }
                for i in 0..N.min(v.len() / 2) {
                    let mut h = v[i * 2];
                    let mut l = v[i * 2 + 1];
                    if h.is_ascii_digit() {
                        h -= b'0';
                    } else if (b'a'..=b'f').contains(&h) {
                        h -= b'a' - 10;
                    } else if (b'A'..=b'F').contains(&h) {
                        h -= b'A' - 10;
                    } else {
                        return Err(serde::de::Error::custom("invalid hex character"));
                    }
                    if l.is_ascii_digit() {
                        l -= b'0';
                    } else if (b'a'..=b'f').contains(&l) {
                        l -= b'a' - 10;
                    } else if (b'A'..=b'F').contains(&l) {
                        l -= b'A' - 10;
                    } else {
                        return Err(serde::de::Error::custom("invalid hex character"));
                    }
                    res[i] = h * 16 + l;
                }
                if let Some(&b) = v.get(v.len() / 2 * 2) {
                    let b = if b.is_ascii_digit() {
                        b - b'0'
                    } else if (b'a'..=b'f').contains(&b) {
                        b - b'a' + 10
                    } else if (b'A'..=b'F').contains(&b) {
                        b - b'A' + 10
                    } else {
                        return Err(serde::de::Error::custom("invalid hex character"));
                    };
                    res[v.len() / 2] = b * 16;
                }
                Ok(FixedHexString(res))
            }
        }

        deserializer.deserialize_bytes(Visitor::<N>)
    }
}

#[cfg(all(test, feature = "altcha"))]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_hex_string() -> anyhow::Result<()> {
        let s = serde_json::from_slice::<FixedHexString<8>>(br#""0123456789abcdef""#)
            .map_err(anyhow::Error::from)?;
        assert_eq!(s.0, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]);
        let s = serde_json::from_slice::<FixedHexString<8>>(br#""0123456789abcde""#)
            .map_err(anyhow::Error::from)?;
        assert_eq!(s.0, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xe0]);
        let s = serde_json::from_slice::<FixedHexString<8>>(br#""0123456789abcd""#)
            .map_err(anyhow::Error::from)?;
        assert_eq!(s.0, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0x00]);
        let s = serde_json::from_slice::<FixedHexString<8>>(br#""0123456789abc""#)
            .map_err(anyhow::Error::from)?;
        assert_eq!(s.0, [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xc0, 0x00]);
        Ok(())
    }
}
