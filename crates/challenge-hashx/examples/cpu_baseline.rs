//! Matched fixed-work and fresh-challenge CPU baseline for the native `HashX` JIT.
//! Run on the same physical host as the SM86 GPU benchmark.
use std::{hint::black_box, sync::Arc, thread, time::Instant};

use challenge_hashx::{
    hashx::{self, Challenge},
    hashx_cpu_jit,
};

const WORK: u32 = 1_048_576;
const SAMPLES: usize = 11;
const DIFFICULTIES: [u32; 3] = [12, 18, 20];
const WORKERS: [usize; 6] = [1, 2, 4, 8, 12, 16];

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn fixed_work(jit: &Arc<hashwx::HashX>, worker_count: usize) -> anyhow::Result<(f64, u64)> {
    let started = Instant::now();
    let hashes_per_worker = u64::from(WORK) / u64::try_from(worker_count)?;
    let remainder = u64::from(WORK) % u64::try_from(worker_count)?;
    let sums = thread::scope(|scope| {
        let mut jobs = Vec::with_capacity(worker_count);
        let mut first = 0_u64;
        for worker in 0..worker_count {
            let count = hashes_per_worker + u64::from(u64::try_from(worker)? < remainder);
            let last = first + count;
            let hashx = Arc::clone(jit);
            jobs.push(scope.spawn(move || {
                let mut checksum = 0_u64;
                for nonce in first..last {
                    checksum ^= hashx.hash(black_box(nonce));
                }
                checksum
            }));
            first = last;
        }
        jobs.into_iter()
            .map(|job| {
                job.join()
                    .map_err(|_| anyhow::anyhow!("CPU worker panicked"))
            })
            .try_fold(0_u64, |value, result| result.map(|hash| value ^ hash))
    })?;
    Ok((started.elapsed().as_secs_f64() * 1_000.0, sums))
}

fn main() -> anyhow::Result<()> {
    let available = thread::available_parallelism()?.get();
    println!("CPU_AVAILABLE_PARALLELISM={available}");
    let seed = std::array::from_fn::<_, 32, _>(|index| index.to_le_bytes()[0]);
    let start = Instant::now();
    let (_, jit, _) = hashx::build_compiled_seed(&seed)?;
    let initial_compile_us = start.elapsed().as_secs_f64() * 1e6;
    let jit = Arc::new(jit);
    println!("FIXED_SEED_COMPILE_US={initial_compile_us:.3}");
    for workers in WORKERS {
        if workers > available {
            continue;
        }
        let mut times = Vec::with_capacity(SAMPLES);
        let mut checksum = None;
        // Exclude the first warmup run from the timed samples.
        for iteration in 0..=SAMPLES {
            let (ms, sum) = fixed_work(&jit, workers)?;
            if let Some(previous) = checksum {
                anyhow::ensure!(
                    sum == previous,
                    "CPU checksum mismatch: {sum} != {previous}"
                );
            }
            checksum = Some(sum);
            if iteration != 0 {
                times.push(ms);
            }
        }
        let median_ms = median(&mut times);
        println!(
            "FIXED workers={workers} hashes={WORK} median_ms={median_ms:.6} mhps={:.6} checksum={:016x}",
            f64::from(WORK) / median_ms / 1_000.0,
            checksum.unwrap_or_default(),
        );
    }
    for difficulty in DIFFICULTIES {
        for workers in WORKERS {
            if workers > available {
                continue;
            }
            let mut times = Vec::with_capacity(32);
            for seed_number in 0_u32..32 {
                let challenge = Challenge {
                    data: hex::encode(seed_number.to_le_bytes()),
                    difficulty,
                };
                let result = hashx_cpu_jit::solve(&challenge, workers)?;
                anyhow::ensure!(
                    hashx::verify(&challenge, &result.answer)?,
                    "invalid CPU solution"
                );
                times.push(result.solve_ms);
            }
            let median_ms = median(&mut times);
            println!(
                "SOLVE difficulty={difficulty} workers={workers} count=32 median_ms={median_ms:.6} min_ms={:.6} max_ms={:.6}",
                times[0], times[31],
            );
        }
    }
    Ok(())
}
