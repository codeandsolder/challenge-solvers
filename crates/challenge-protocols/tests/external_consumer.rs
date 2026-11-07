//! Exercise exported entry points as a real downstream crate, not as internal modules.

use challenge_protocols::{anubis, classic_pow};
use serde_json::json;

#[test]
fn external_consumer_can_solve_and_verify_anubis() -> anyhow::Result<()> {
    let challenge = anubis::Challenge {
        data: "reusable-protocol-interface".to_owned(),
        difficulty: 1,
    };
    let result = anubis::solve_cpu(&challenge, 1_000_000)?;
    assert!(anubis::verify(&challenge, &result.answer));

    // Callers need protocol-native, serializable answers without a service dependency.
    let wire = serde_json::to_value(&result.answer)?;
    assert_eq!(wire["nonce"], result.answer.nonce);
    assert_eq!(wire["hash"], result.answer.hash);
    Ok(())
}

#[test]
fn external_consumer_can_dispatch_classic_protocol_without_service() -> anyhow::Result<()> {
    let result = classic_pow::solve(
        "haphash",
        json!({"prefix": "external-crate:", "difficulty": 0}),
        Some("safe"),
        1,
    )?;
    assert_eq!(result.implementation, "safe");
    assert!(
        result.answer["nonce"]
            .as_str()
            .is_some_and(|s| s.parse::<u64>().is_ok())
    );
    assert!(
        result.answer["attempted_nonces"]
            .as_u64()
            .is_some_and(|n| n > 0)
    );
    Ok(())
}
