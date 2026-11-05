//! Cap.js specific protocol structures.
use crate::{
    message::{CapJSEmitter, DecimalMessage},
    solver::Solver,
};

use alloc::{string::String, vec::Vec};

const JS_MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

const fn js_safe_nonce(value: u64) -> Option<f64> {
    if value > JS_MAX_SAFE_INTEGER {
        return None;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "value is explicitly bounded to JavaScript's exact integer range before conversion"
    )]
    let value = value as f64;
    Some(value)
}

fn capjs_mask(difficulty: u8) -> Option<u64> {
    let bits = u32::from(difficulty).checked_mul(4)?;
    (bits <= 64).then(|| match bits {
        0 => 0,
        64 => u64::MAX,
        _ => u64::MAX << (64 - bits),
    })
}

fn valid_count(count: usize) -> bool {
    count <= usize::try_from(u32::MAX).unwrap_or(usize::MAX)
}

#[derive(serde::Deserialize, Debug, Clone, Copy)]
/// Cap.js `PoW` challenge rules.
pub struct ChallengeRules {
    #[serde(rename = "c")]
    /// The count. (JSON key: `c`)
    pub count: usize,
    #[serde(rename = "s")]
    /// The salt length. (JSON key: `s`)
    pub salt_length: usize,
    #[serde(rename = "d")]
    /// The difficulty. (JSON key: `d`)
    pub difficulty: u8,
}

impl ChallengeRules {
    /// Return the leading-nibble mask for this challenge.
    ///
    /// Difficulties above 16 nibbles cannot be represented by the 64-bit
    /// comparison used by Cap.js and are therefore unsupported.
    #[must_use]
    pub fn mask(self) -> Option<u64> {
        capjs_mask(self.difficulty)
    }
}

#[derive(serde::Deserialize, Debug, Clone)]
/// Cap.js `PoW` challenge descriptor.
pub struct ChallengeDescriptor {
    #[serde(rename = "challenge")]
    /// The rules. (JSON key: `challenge`)
    rules: ChallengeRules,
    /// The challenge token.
    pub token: String,
}

#[derive(serde::Serialize, Debug, Clone, Copy)]
/// Cap.js `PoW` response meta data.
pub struct SolveCapJsResponseMeta {
    #[cfg(feature = "std")]
    #[serde(rename = "elapsed_us")]
    elapsed: u64,
    attempted_nonces: u64,
    #[cfg(feature = "std")]
    hashrate: u64,
}

#[derive(serde::Serialize, Debug, Clone)]
/// Cap.js `PoW` response.
pub struct SolveCapJsResponse {
    #[serde(rename = "_meta")]
    /// The meta data.
    pub meta: SolveCapJsResponseMeta,
    /// The token.
    pub token: String,
    /// The solutions.
    pub solutions: Vec<f64>,
}

impl ChallengeDescriptor {
    /// Get the rules of a Cap.js `PoW`.
    #[must_use]
    pub const fn rules(&self) -> &ChallengeRules {
        &self.rules
    }

    /// Solve a Cap.js `PoW`.
    #[must_use]
    pub fn solve(self) -> (Option<SolveCapJsResponse>, u64) {
        self.solve_with_limit(u64::MAX)
    }

    /// Estimate the workload of a Cap.js `PoW`.
    #[must_use]
    pub fn estimated_workload(&self) -> u64 {
        16u64
            .saturating_pow(u32::from(self.rules.difficulty))
            .saturating_mul(u64::try_from(self.rules.count).unwrap_or(u64::MAX))
    }

    /// Solve a Cap.js `PoW` with a limit in parallel.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn solve_with_limit_parallel(
        self,
        pool: &rayon::ThreadPool,
        limit_per_challenge: u64,
    ) -> (Option<SolveCapJsResponse>, u64) {
        let Some(mask) = self.rules.mask() else {
            return (None, 0);
        };
        if !valid_count(self.rules.count) {
            return (None, 0);
        }
        let emitter = CapJSEmitter::new(self.token.as_bytes());
        let attempted_nonces = core::sync::atomic::AtomicU64::new(0);
        let failed = core::sync::atomic::AtomicBool::new(false);
        let mut response = SolveCapJsResponse {
            meta: SolveCapJsResponseMeta {
                elapsed: 0,
                attempted_nonces: 0,
                hashrate: 0,
            },
            token: self.token,
            solutions: vec![0f64; self.rules.count],
        };
        let elapsed = pool.install(|| {
            use rayon::prelude::*;
            let start = std::time::Instant::now();
            response.solutions.par_iter_mut().enumerate().for_each(
                |(i, solution): (usize, &mut f64)| {
                    let mut salt_buf = vec![0u8; self.rules.salt_length];
                    let mut targets = [0; 2];
                    let Ok(index) = u32::try_from(i + 1) else {
                        failed.store(true, core::sync::atomic::Ordering::Relaxed);
                        return;
                    };
                    emitter.emit(&mut salt_buf, &mut targets, index);
                    let Some((message, fixup_prefix)) = DecimalMessage::new_f64(&salt_buf, 0)
                    else {
                        failed.store(true, core::sync::atomic::Ordering::Relaxed);
                        return;
                    };
                    let mut solver = crate::DecimalSolver::from(message);
                    solver.set_limit(limit_per_challenge);
                    let Some(nonce) = solver
                        .solve_nonce_only::<{ crate::solver::SOLVE_TYPE_MASK }>(
                            u64::from(targets[0]) << 32 | u64::from(targets[1]),
                            mask,
                        )
                    else {
                        failed.store(true, core::sync::atomic::Ordering::Relaxed);
                        return;
                    };
                    attempted_nonces.fetch_add(
                        solver.attempted_nonces(),
                        core::sync::atomic::Ordering::Relaxed,
                    );
                    let nonce_f64 = fixup_prefix
                        .map_or_else(|| js_safe_nonce(nonce), |fixup| Some(fixup.fixup(nonce)));
                    let Some(nonce_f64) = nonce_f64 else {
                        failed.store(true, core::sync::atomic::Ordering::Relaxed);
                        return;
                    };
                    *solution = nonce_f64;
                },
            );
            start.elapsed()
        });
        let attempted_nonces = attempted_nonces.load(core::sync::atomic::Ordering::Relaxed);
        if failed.load(core::sync::atomic::Ordering::Relaxed) {
            return (None, attempted_nonces);
        }
        response.meta.elapsed = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        response.meta.attempted_nonces = attempted_nonces;
        response.meta.hashrate =
            attempted_nonces.saturating_mul(1_000_000) / response.meta.elapsed.max(1);
        (Some(response), attempted_nonces)
    }

    /// Solve a Cap.js `PoW` with a limit.
    #[must_use]
    pub fn solve_with_limit(self, limit: u64) -> (Option<SolveCapJsResponse>, u64) {
        let Some(mask) = self.rules.mask() else {
            return (None, 0);
        };
        if !valid_count(self.rules.count) {
            return (None, 0);
        }
        let emitter = CapJSEmitter::new(self.token.as_bytes());
        let mut attempted_nonces = 0;
        let mut response = SolveCapJsResponse {
            meta: SolveCapJsResponseMeta {
                #[cfg(feature = "std")]
                elapsed: 0,
                attempted_nonces,
                #[cfg(feature = "std")]
                hashrate: 0,
            },
            token: self.token,
            solutions: Vec::with_capacity(self.rules.count),
        };
        #[cfg(feature = "std")]
        let start = std::time::Instant::now();
        for i in 0..self.rules.count {
            if limit.saturating_sub(attempted_nonces) == 0 {
                break;
            }
            let mut salt_buf = alloc::vec![0u8; self.rules.salt_length];
            let mut targets = [0; 2];
            let Ok(index) = u32::try_from(i + 1) else {
                return (None, attempted_nonces);
            };
            emitter.emit(&mut salt_buf, &mut targets, index);
            let Some((message, fixup_prefix)) = DecimalMessage::new_f64(&salt_buf, 0) else {
                return (None, attempted_nonces);
            };
            let mut solver = crate::DecimalSolver::from(message);
            solver.set_limit(limit.saturating_sub(attempted_nonces));
            let Some(nonce) = solver.solve_nonce_only::<{ crate::solver::SOLVE_TYPE_MASK }>(
                u64::from(targets[0]) << 32 | u64::from(targets[1]),
                mask,
            ) else {
                return (None, attempted_nonces);
            };
            attempted_nonces += solver.attempted_nonces();
            let nonce_f64 =
                fixup_prefix.map_or_else(|| js_safe_nonce(nonce), |fixup| Some(fixup.fixup(nonce)));
            let Some(nonce_f64) = nonce_f64 else {
                return (None, attempted_nonces);
            };
            response.solutions.push(nonce_f64);
        }
        #[cfg(feature = "std")]
        {
            response.meta.elapsed = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
            response.meta.hashrate =
                attempted_nonces.saturating_mul(1_000_000) / response.meta.elapsed.max(1);
        }
        response.meta.attempted_nonces = attempted_nonces;
        (Some(response), attempted_nonces)
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
/// Cap.js `PoW` response.
#[serde(untagged)]
pub enum CapJsResponse {
    /// The solutions response.
    Solutions(CapJsRedeemedToken),
    /// The error response.
    Error {
        /// The error message.
        error: String,
    },
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
/// Cap.js `PoW` redeemed token.
pub struct CapJsRedeemedToken {
    /// The redeemed token.
    pub token: String,
    /// The expiration time.
    pub expires: u64,
}

#[cfg(test)]
mod tests {
    use super::capjs_mask;

    #[test]
    fn capjs_mask_covers_full_supported_difficulty_domain() {
        assert_eq!(capjs_mask(0), Some(0));
        assert_eq!(capjs_mask(1), Some(0xf000_0000_0000_0000));
        assert_eq!(capjs_mask(16), Some(u64::MAX));
        assert_eq!(capjs_mask(17), None);
    }
}
