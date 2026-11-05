use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::Context;

use crate::hashx::{Challenge, build_compiled_seed, valid_prefix};
use challenge_protocols::anubis::{Answer, BackendResult};

const SEARCH_LIMIT: u64 = u32::MAX as u64 + 1;
const CANCEL_BATCH: u64 = 256;

/// Solve a `HashX` challenge with native-JIT search over the u32 nonce range.
///
/// # Errors
/// Rejects malformed challenges, invalid worker counts, exhausted searches or invalid proofs.
pub fn solve(challenge: &Challenge, workers: usize) -> anyhow::Result<BackendResult> {
    challenge.validate()?;
    anyhow::ensure!(workers > 0, "HashX JIT needs at least one worker");
    let seed = hex::decode(&challenge.data).context("decode HashX challenge")?;
    let started = Instant::now();
    let (_usable_seed, verifier, _program) = build_compiled_seed(&seed)?;
    let verifier = Arc::new(verifier);
    let found = AtomicU64::new(u64::MAX);
    let worker_count = u64::try_from(workers).context("too many HashX JIT workers")?;

    std::thread::scope(|scope| -> anyhow::Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let found = &found;
            let hashx = Arc::clone(&verifier);
            handles.push(scope.spawn(move || -> anyhow::Result<()> {
                let mut base =
                    u64::try_from(worker).context("HashX JIT worker index exceeds u64")?;

                while base < SEARCH_LIMIT && found.load(Ordering::Relaxed) == u64::MAX {
                    let mut nonce = base;
                    for _ in 0..CANCEL_BATCH {
                        if nonce >= SEARCH_LIMIT {
                            break;
                        }
                        let word = hashx.hash(nonce);
                        if word.swap_bytes().leading_zeros() >= challenge.difficulty {
                            if found
                                .compare_exchange(
                                    u64::MAX,
                                    nonce,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                            {
                                // Validate the winning nonce through the full-digest entry point
                                // before publishing it to the service.
                                let hash = hashx.hash_bytes(nonce);
                                anyhow::ensure!(
                                    valid_prefix(&hash, challenge.difficulty),
                                    "HashX JIT returned nonce with invalid difficulty"
                                );
                            }
                            return Ok(());
                        }
                        nonce = nonce.saturating_add(worker_count);
                    }
                    base = base.saturating_add(worker_count * CANCEL_BATCH);
                }
                Ok(())
            }));
        }

        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("HashX JIT worker panicked"))??;
        }
        Ok(())
    })?;

    let nonce = found.load(Ordering::Relaxed);
    anyhow::ensure!(nonce != u64::MAX, "HashX JIT exhausted u32 nonce space");
    let hash = verifier.hash_bytes(nonce);
    anyhow::ensure!(
        valid_prefix(&hash, challenge.difficulty),
        "HashX JIT result failed final validation"
    );

    Ok(BackendResult {
        answer: Answer {
            nonce: nonce.to_string(),
            hash: hex::encode(hash),
        },
        solve_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn native_jit_is_shareable_for_validation() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<hashwx::HashX>();
    }
}
