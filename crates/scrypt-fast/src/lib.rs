//! Reusable word-oriented Scrypt RFC 7914 derivation using a stable arena.
//! `ROMix` design follows Go x/crypto/scrypt (BSD-3-Clause).

#![no_std]
extern crate alloc;
use alloc::{vec, vec::Vec};

use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;

/// Scrypt workspace retained across arbitrary password/salt derivations.
///
/// Each independent nonce-search worker owns exactly one arena.
pub struct Workspace {
    n: usize,
    r: usize,
    b: Vec<u8>,
    v: Vec<u32>,
    x: Vec<u32>,
    y: Vec<u32>,
}

impl Workspace {
    /// Allocate workspace from Scrypt parameters already validated by `RustCrypto`.
    #[must_use]
    pub fn new(params: &scrypt::Params) -> Self {
        let n = 1_usize << params.log_n();
        let r = params.r() as usize;
        let p = params.p() as usize;
        // Params::new proves the checked products n*r*128 and p*r*128 fit.
        Self {
            n,
            r,
            b: vec![0; p * r * 128],
            v: vec![0; n * r * 32],
            x: vec![0; r * 32],
            y: vec![0; r * 32],
        }
    }
    /// Run an RFC 7914 derivation without reallocating the main `ROMix` arena.
    ///
    /// # Errors
    /// Returns the same invalid-output-length error as `RustCrypto` Scrypt.
    pub fn derive(
        &mut self,
        password: &[u8],
        salt: &[u8],
        out: &mut [u8],
    ) -> Result<(), scrypt::errors::InvalidOutputLen> {
        if out.is_empty() || out.len() / 32 > 0xffff_ffff {
            return Err(scrypt::errors::InvalidOutputLen);
        }
        pbkdf2_hmac::<Sha256>(password, salt, 1, &mut self.b);
        for chunk in self.b.chunks_exact_mut(128 * self.r) {
            smix(chunk, self.n, self.r, &mut self.v, &mut self.x, &mut self.y);
        }
        pbkdf2_hmac::<Sha256>(password, &self.b, 1, out);
        Ok(())
    }
    /// Heap bytes reserved for the workspace, excluding the caller-owned output.
    #[must_use]
    pub const fn footprint(&self) -> usize {
        self.b.len() + self.v.len() * 4 + self.x.len() * 4 + self.y.len() * 4
    }
}

#[inline(always)]
fn salsa_xor(tmp: &mut [u32; 16], src: &[u32], dst: &mut [u32]) {
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = tmp[i] ^ src[i];
    }
    let mut x = w;
    macro_rules! q {
        ($a:literal,$b:literal,$c:literal,$d:literal) => {
            x[$b] ^= x[$a].wrapping_add(x[$d]).rotate_left(7);
            x[$c] ^= x[$b].wrapping_add(x[$a]).rotate_left(9);
            x[$d] ^= x[$c].wrapping_add(x[$b]).rotate_left(13);
            x[$a] ^= x[$d].wrapping_add(x[$c]).rotate_left(18);
        };
    }
    for _ in 0..4 {
        q!(0, 4, 8, 12);
        q!(5, 9, 13, 1);
        q!(10, 14, 2, 6);
        q!(15, 3, 7, 11);
        q!(0, 1, 2, 3);
        q!(5, 6, 7, 4);
        q!(10, 11, 8, 9);
        q!(15, 12, 13, 14);
    }
    for i in 0..16 {
        let result = x[i].wrapping_add(w[i]);
        tmp[i] = result;
        dst[i] = result;
    }
}

#[inline(always)]
fn blockmix(tmp: &mut [u32; 16], src: &[u32], dst: &mut [u32], r: usize) {
    tmp.copy_from_slice(&src[(2 * r - 1) * 16..2 * r * 16]);
    for i in 0..r {
        salsa_xor(
            tmp,
            &src[(2 * i) * 16..(2 * i + 1) * 16],
            &mut dst[i * 16..(i + 1) * 16],
        );
        salsa_xor(
            tmp,
            &src[(2 * i + 1) * 16..(2 * i + 2) * 16],
            &mut dst[(i + r) * 16..(i + r + 1) * 16],
        );
    }
}

fn smix(
    bytes: &mut [u8],
    work_factor: usize,
    block_factor: usize,
    history: &mut [u32],
    state: &mut [u32],
    alternate: &mut [u32],
) {
    let row_words = 32 * block_factor;
    let mut temporary = [0u32; 16];
    let (input_words, remaining) = bytes.as_chunks::<4>();
    debug_assert_eq!(remaining, []);
    for (word, chunk) in state.iter_mut().zip(input_words.iter()) {
        *word = u32::from_le_bytes(*chunk);
    }
    for slot in (0..work_factor).step_by(2) {
        history[slot * row_words..(slot + 1) * row_words].copy_from_slice(state);
        blockmix(&mut temporary, state, alternate, block_factor);
        history[(slot + 1) * row_words..(slot + 2) * row_words].copy_from_slice(alternate);
        blockmix(&mut temporary, alternate, state, block_factor);
    }
    for _ in (0..work_factor).step_by(2) {
        let slot = (state[(2 * block_factor - 1) * 16] as usize) & (work_factor - 1);
        for (word, previous) in state
            .iter_mut()
            .zip(&history[slot * row_words..(slot + 1) * row_words])
        {
            *word ^= *previous;
        }
        blockmix(&mut temporary, state, alternate, block_factor);
        let slot = (alternate[(2 * block_factor - 1) * 16] as usize) & (work_factor - 1);
        for (word, previous) in alternate
            .iter_mut()
            .zip(&history[slot * row_words..(slot + 1) * row_words])
        {
            *word ^= *previous;
        }
        blockmix(&mut temporary, alternate, state, block_factor);
    }
    let (output_words, remaining) = bytes.as_chunks_mut::<4>();
    debug_assert_eq!(remaining, []);
    for (word, chunk) in state.iter().zip(output_words.iter_mut()) {
        *chunk = word.to_le_bytes();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn differential_scrypt_r_p_n_and_output_lengths() {
        for logn in [1, 2, 4, 8, 10] {
            for r in [1, 2, 4, 8] {
                for p in [1, 2, 3] {
                    let params = scrypt::Params::new(logn, r, p).unwrap();
                    let mut workspace = Workspace::new(&params);
                    for length in [1, 16, 32, 64, 80] {
                        let mut out = vec![0; length];
                        let mut reference = vec![0; length];
                        workspace
                            .derive(b"abc", b"16-byte-test-salt", &mut out)
                            .unwrap();
                        scrypt::scrypt(b"abc", b"16-byte-test-salt", &params, &mut reference)
                            .unwrap();
                        assert_eq!(
                            out,
                            reference,
                            "N={}, r={r}, p={p}, length={length}",
                            params.n()
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn empty_output_is_rejected() {
        let params = scrypt::Params::new(4, 1, 1).unwrap();
        let mut workspace = Workspace::new(&params);
        assert!(workspace.derive(b"password", b"salt", &mut []).is_err());
    }
    #[test]
    fn reused_workspace_handles_alternating_salts_passwords_and_empty_inputs() {
        for logn in [1, 4, 8] {
            for r in [1, 8] {
                for p in [1, 2] {
                    let params = scrypt::Params::new(logn, r, p).unwrap();
                    let mut workspace = Workspace::new(&params);
                    for index in 0..16u8 {
                        let password = vec![index.wrapping_mul(31); usize::from(index % 19)];
                        let salt = vec![index.wrapping_mul(7); usize::from(index % 23)];
                        let mut out = [0u8; 32];
                        let mut reference = [0u8; 32];
                        workspace.derive(&password, &salt, &mut out).unwrap();
                        scrypt::scrypt(&password, &salt, &params, &mut reference).unwrap();
                        assert_eq!(out, reference, "N={},r={r},p={p},index={index}", params.n());
                    }
                }
            }
        }
    }
}
