use std::{env, hint::black_box, time::Instant};

use anubis_argon2::Memory;
use argon2::{Argon2, Block};

fn main() {
    let mut args = env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "portable".to_owned());
    let iterations = args
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64);
    let data = black_box(b"portable SIMD says hello".as_slice());

    match mode.as_str() {
        "portable" => {
            let mut memory = Memory::new();
            for nonce in 0_u64..3 {
                black_box(anubis_argon2::hash(data, nonce.to_le_bytes(), &mut memory));
            }
            let start = Instant::now();
            for nonce in 0_u64..u64::from(iterations) {
                black_box(anubis_argon2::hash(data, nonce.to_le_bytes(), &mut memory));
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!("portable {:.6}", f64::from(iterations) / elapsed);
        }
        "reference" => {
            let argon2 = Argon2::default();
            let mut memory = vec![Block::default(); argon2.params().block_count()];
            let mut out = [0_u8; 32];
            for nonce in 0_u64..3 {
                let _ = black_box(argon2.hash_password_into_with_memory(
                    data,
                    &nonce.to_le_bytes(),
                    &mut out,
                    &mut memory,
                ));
            }
            let start = Instant::now();
            for nonce in 0_u64..u64::from(iterations) {
                let _ = black_box(argon2.hash_password_into_with_memory(
                    data,
                    &nonce.to_le_bytes(),
                    &mut out,
                    &mut memory,
                ));
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!("reference {:.6}", f64::from(iterations) / elapsed);
        }
        _ => eprintln!("mode must be portable or reference"),
    }
}
