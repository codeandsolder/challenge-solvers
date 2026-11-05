#![recursion_limit = "256"]
#![deny(unsafe_code)]

//! Reusable, protocol-level proof-of-work solvers.
//!
//! The caller supplies challenge data and a backend selection; this crate owns
//! search, protocol-specific validation, nonce encoding and proof formatting.
//! HTTP transport, service scheduling and telemetry live in challenge-disasterer.

/// Anubis's decimal challenge variant.
pub mod anubis;
/// Anubis WASM SHA-256/Argon2id challenge implementations.
pub mod anubis_wasm;
/// Classic proof-of-work protocols and common backend selection.
pub mod classic_pow;
/// Linear.pub's SHA-1 gate.
pub mod linear_pow;
