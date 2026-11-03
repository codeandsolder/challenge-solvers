//! x86-64 solver selection built from shared portable-SIMD implementations.
//!
//! The AVX2 and AVX-512 variants share all state machines and protocol logic.
//! Only their target-feature entry thunks and logical lane counts differ.

use crate::message::{
    AltchaMessage, BinaryMessage, CerberusMessage, DoubleBlockMessage, GoAwayMessage,
    SingleBlockMessage,
};

use super::{SolverRouter, safe, sha_ni, wide_backend, wide_backend_arch};

/// AVX2 portable-SIMD backend marker.
pub use wide_backend_arch::Avx2;
/// AVX-512 portable-SIMD backend marker.
pub use wide_backend_arch::Avx512;

impl super::CpuIDToken for Avx2 {
    #[inline]
    fn get() -> bool {
        <Self as wide_backend_arch::Backend>::available()
    }
}

impl super::CpuIDToken for Avx512 {
    #[inline]
    fn get() -> bool {
        <Self as wide_backend_arch::Backend>::available()
    }
}

/// AVX2 single-block solver.
pub type Avx2SingleBlockSolver = wide_backend::SingleBlockSolver<Avx2>;
/// AVX2 double-block solver.
pub type Avx2DoubleBlockSolver = wide_backend::DoubleBlockSolver<Avx2>;
/// AVX2 decimal solver.
pub type Avx2DecimalSolver = wide_backend::DecimalSolver<Avx2>;
/// AVX2 binary solver.
pub type Avx2BinarySolver = wide_backend::BinarySolver<Avx2>;
/// AVX2 `GoAway` solver.
pub type Avx2GoAwaySolver = wide_backend::GoAwaySolver<Avx2>;
/// AVX2 Cerberus solver.
pub type Avx2CerberusSolver = wide_backend::CerberusSolver<Avx2>;
/// AVX2 ALTCHA SHA-256 solver.
pub type Avx2AltchaSha256Solver = wide_backend::AltchaSha256Solver<Avx2>;

/// AVX-512 single-block solver.
pub type Avx512SingleBlockSolver = wide_backend::SingleBlockSolver<Avx512>;
/// AVX-512 double-block solver.
pub type Avx512DoubleBlockSolver = wide_backend::DoubleBlockSolver<Avx512>;
/// AVX-512 decimal solver.
pub type Avx512DecimalSolver = wide_backend::DecimalSolver<Avx512>;
/// AVX-512 binary solver.
pub type Avx512BinarySolver = wide_backend::BinarySolver<Avx512>;
/// AVX-512 `GoAway` solver.
pub type Avx512GoAwaySolver = wide_backend::GoAwaySolver<Avx512>;
/// AVX-512 Cerberus solver.
pub type Avx512CerberusSolver = wide_backend::CerberusSolver<Avx512>;
/// AVX-512 ALTCHA SHA-256 solver.
pub type Avx512AltchaSha256Solver = wide_backend::AltchaSha256Solver<Avx512>;

/// Automatic single-block solver: AVX-512, SHA-NI, AVX2, then scalar.
pub type SingleBlockSolver = SolverRouter<
    Avx512,
    SingleBlockMessage,
    Avx512SingleBlockSolver,
    SolverRouter<
        sha_ni::RequiredFeatures,
        SingleBlockMessage,
        sha_ni::SingleBlockSolver,
        SolverRouter<Avx2, SingleBlockMessage, Avx2SingleBlockSolver, safe::SingleBlockSolver>,
    >,
>;

/// Automatic double-block solver: AVX-512, SHA-NI, AVX2, then scalar.
pub type DoubleBlockSolver = SolverRouter<
    Avx512,
    DoubleBlockMessage,
    Avx512DoubleBlockSolver,
    SolverRouter<
        sha_ni::RequiredFeatures,
        DoubleBlockMessage,
        sha_ni::DoubleBlockSolver,
        SolverRouter<Avx2, DoubleBlockMessage, Avx2DoubleBlockSolver, safe::DoubleBlockSolver>,
    >,
>;

/// Automatic decimal solver: AVX-512, SHA-NI, AVX2, then scalar.
pub type DecimalSolver = SolverRouter<
    Avx512,
    crate::message::DecimalMessage,
    Avx512DecimalSolver,
    SolverRouter<
        sha_ni::RequiredFeatures,
        crate::message::DecimalMessage,
        sha_ni::DecimalSolver,
        SolverRouter<Avx2, crate::message::DecimalMessage, Avx2DecimalSolver, safe::DecimalSolver>,
    >,
>;

/// Automatic binary solver: AVX-512, AVX2, then scalar.
pub type BinarySolver = SolverRouter<
    Avx512,
    BinaryMessage,
    Avx512BinarySolver,
    SolverRouter<Avx2, BinaryMessage, Avx2BinarySolver, safe::BinarySolver>,
>;

/// Automatic `GoAway` solver: AVX-512, SHA-NI, AVX2, then scalar.
pub type GoAwaySolver = SolverRouter<
    Avx512,
    GoAwayMessage,
    Avx512GoAwaySolver,
    SolverRouter<
        sha_ni::RequiredFeatures,
        GoAwayMessage,
        sha_ni::GoAwaySolver,
        SolverRouter<Avx2, GoAwayMessage, Avx2GoAwaySolver, safe::GoAwaySolver>,
    >,
>;

/// Automatic Cerberus solver: AVX-512, AVX2, then scalar.
pub type CerberusSolver = SolverRouter<
    Avx512,
    CerberusMessage,
    Avx512CerberusSolver,
    SolverRouter<Avx2, CerberusMessage, Avx2CerberusSolver, safe::CerberusSolver>,
>;

/// Automatic ALTCHA SHA-256 solver: AVX-512, AVX2, then scalar.
pub type AltchaSha256Solver = SolverRouter<
    Avx512,
    AltchaMessage,
    Avx512AltchaSha256Solver,
    SolverRouter<Avx2, AltchaMessage, Avx2AltchaSha256Solver, safe::AltchaSha256Solver>,
>;

#[cfg(test)]
mod tests {
    use anyhow::Context as _;

    use super::*;
    use crate::{
        message::{CerberusBinaryMessage, CerberusDecimalMessage},
        solver::{Solver as _, wide_backend_arch::Backend},
    };

    fn lanes_u64<B: Backend>() -> anyhow::Result<u64> {
        Ok(u64::try_from(B::LANES)?)
    }

    fn run_backend_suite<B: Backend>() -> anyhow::Result<()> {
        if !B::available() {
            return Ok(());
        }

        crate::solver::tests::test_decimal_validator::<wide_backend::DecimalSolver<B>, _>(
            |prefix, search_space| {
                if let Some(solver) = SingleBlockMessage::new(prefix, search_space).map(Into::into)
                {
                    Some(wide_backend::DecimalSolver::SingleBlock(solver))
                } else {
                    DoubleBlockMessage::new(prefix, search_space).map(Into::into)
                }
            },
        )?;
        crate::solver::tests::test_decimal_validator_f64_safe::<wide_backend::DecimalSolver<B>, _>(
            |prefix, search_space| {
                if let Some((solver, p)) =
                    SingleBlockMessage::new_f64(prefix, search_space).map(|(x, p)| (x.into(), p))
                {
                    Some((wide_backend::DecimalSolver::SingleBlock(solver), p))
                } else {
                    DoubleBlockMessage::new(prefix, search_space)
                        .map(|x| (wide_backend::DecimalSolver::DoubleBlock(x.into()), None))
                }
            },
        )?;
        crate::solver::tests::test_binary_validator::<wide_backend::BinarySolver<B>, _>(
            |prefix, nonce_byte_count| {
                BinaryMessage::new(prefix, nonce_byte_count).map(wide_backend::BinarySolver::from)
            },
        )?;
        crate::solver::tests::test_goaway_validator::<wide_backend::GoAwaySolver<B>, _>(
            |prefix| {
                wide_backend::GoAwaySolver::from(GoAwayMessage::new(
                    core::array::from_fn(|i| {
                        u32::from_be_bytes([
                            prefix[i * 4],
                            prefix[i * 4 + 1],
                            prefix[i * 4 + 2],
                            prefix[i * 4 + 3],
                        ])
                    }),
                    0,
                ))
            },
        )?;
        crate::solver::tests::test_altcha_validator::<wide_backend::AltchaSha256Solver<B>, _>(
            |message| Some(wide_backend::AltchaSha256Solver::from(message)),
        )?;

        for working_set in 0..=1 {
            crate::solver::tests::test_cerberus_decimal_validator::<
                wide_backend::CerberusSolver<B>,
                _,
            >(|prefix| {
                Some(
                    CerberusMessage::Decimal(CerberusDecimalMessage::new(prefix, working_set)?)
                        .into(),
                )
            })?;
            crate::solver::tests::test_cerberus_binary_validator::<
                wide_backend::CerberusSolver<B>,
                _,
            >(|prefix| {
                Some(
                    CerberusMessage::Binary(CerberusBinaryMessage::new(prefix, working_set)).into(),
                )
            })?;
        }
        Ok(())
    }

    fn check_attempt_accounting<B: Backend>() -> anyhow::Result<()> {
        if !B::available() {
            return Ok(());
        }
        let lanes = lanes_u64::<B>()?;
        for limit in [1, lanes, lanes + 1, lanes * 2 + 1] {
            let message = SingleBlockMessage::new(&[b'a'; 64], 0)
                .context("single-block accounting fixture must construct")?;
            let mut solver = wide_backend::SingleBlockSolver::<B>::from(message);
            solver.set_limit(limit);
            assert!(
                solver
                    .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                    .is_none()
            );
            assert_eq!(solver.attempted_nonces(), limit);
        }

        let salt = [b'a'; 32];
        let nonce_width = core::num::NonZeroU8::new(4).context("four-byte nonce is nonzero")?;
        let message = BinaryMessage::new(&salt, nonce_width)
            .context("binary accounting fixture must construct")?;
        let mut binary = wide_backend::BinarySolver::<B>::from(message);
        binary.set_limit(lanes + 1);
        assert!(
            binary
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(binary.attempted_nonces(), lanes + 1);

        let mut goaway = wide_backend::GoAwaySolver::<B>::from(GoAwayMessage::new_bytes(&salt, 0));
        goaway.set_limit(lanes + 1);
        assert!(
            goaway
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(goaway.attempted_nonces(), lanes + 1);
        Ok(())
    }

    #[cfg(feature = "compare-64bit")]
    fn check_full64_lt<B: Backend>() -> anyhow::Result<()> {
        const TARGET: u64 = 1_u64 << 63;
        fn check<S: crate::solver::Solver<Output = [u32; 8]>>(mut solver: S) -> anyhow::Result<()> {
            solver.set_limit(1_000_000);
            let (_, hash) = solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(TARGET, u64::MAX)
                .context("full-width LT solver should find an easy target")?;
            let value = (u64::from(hash[0]) << 32) | u64::from(hash[1]);
            assert!(value < TARGET, "{value:#018x} is not below {TARGET:#018x}");
            Ok(())
        }
        if !B::available() {
            return Ok(());
        }
        let salt = [b'a'; 64];
        check(wide_backend::SingleBlockSolver::<B>::from(
            SingleBlockMessage::new(&salt, 0).context("single-block full64 fixture")?,
        ))?;
        let double = (0..64)
            .find_map(|len| DoubleBlockMessage::new(&salt[..len], 0))
            .context("double-block full64 fixture")?;
        check(wide_backend::DoubleBlockSolver::<B>::from(double))?;
        let nonce_width = core::num::NonZeroU8::new(4).context("four-byte nonce is nonzero")?;
        let binary = BinaryMessage::new(&salt, nonce_width).context("binary full64 fixture")?;
        check(wide_backend::BinarySolver::<B>::from(binary))?;
        let goaway_salt: &[u8; 32] = (&salt[..32]).try_into()?;
        check(wide_backend::GoAwaySolver::<B>::from(
            GoAwayMessage::new_bytes(goaway_salt, 0),
        ))?;
        Ok(())
    }

    fn check_cerberus_exact_limit<B: Backend>() -> anyhow::Result<()> {
        if !B::available() {
            return Ok(());
        }
        let lanes = lanes_u64::<B>()?;
        let message = CerberusDecimalMessage::new(b"cerberus-decimal-benchmark|", 0)
            .context("valid Cerberus decimal message")?;
        let mut solver = wide_backend::CerberusSolver::<B>::from(CerberusMessage::Decimal(message));
        solver.set_limit(lanes * 2 + 1);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_MASK }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), lanes * 2 + 1);
        Ok(())
    }

    fn check_fails_closed<B: Backend>() -> anyhow::Result<()> {
        if B::available() {
            return Ok(());
        }
        let message = SingleBlockMessage::new(b"target-feature-guard", 0)
            .context("single-block feature-guard fixture")?;
        let mut solver = wide_backend::SingleBlockSolver::<B>::from(message);
        solver.set_limit(lanes_u64::<B>()?);
        assert!(
            solver
                .solve::<{ crate::solver::SOLVE_TYPE_LT }>(0, u64::MAX)
                .is_none()
        );
        assert_eq!(solver.attempted_nonces(), 0);
        Ok(())
    }

    macro_rules! backend_tests {
        ($module:ident, $backend:ty) => {
            mod $module {
                use super::*;

                #[test]
                fn validators() -> anyhow::Result<()> {
                    run_backend_suite::<$backend>()
                }

                #[test]
                fn exact_attempt_accounting() -> anyhow::Result<()> {
                    check_attempt_accounting::<$backend>()?;
                    check_cerberus_exact_limit::<$backend>()
                }

                #[cfg(feature = "compare-64bit")]
                #[test]
                fn full_64_bit_comparisons() -> anyhow::Result<()> {
                    check_full64_lt::<$backend>()
                }

                #[test]
                fn unavailable_backend_fails_closed() -> anyhow::Result<()> {
                    check_fails_closed::<$backend>()
                }
            }
        };
    }

    backend_tests!(avx2, Avx2);
    backend_tests!(avx512, Avx512);
}
