use challenge_hashx::{hashx, hashx_cpu_jit};

#[test]
fn native_jit_produces_publicly_verifiable_proofs_across_worker_counts() -> anyhow::Result<()> {
    let challenge = hashx::Challenge {
        data: hex::encode(b"portable hashx reusable api regression"),
        difficulty: 1,
    };
    for workers in [1, 2, 4] {
        let result = hashx_cpu_jit::solve(&challenge, workers)?;
        anyhow::ensure!(
            hashx::verify(&challenge, &result.answer)?,
            "invalid proof with {workers} workers"
        );
        let mut corrupted = result.answer;
        corrupted.hash.replace_range(..2, "ff");
        anyhow::ensure!(
            !hashx::verify(&challenge, &corrupted)?,
            "corrupted proof was accepted with {workers} workers"
        );
    }
    Ok(())
}
