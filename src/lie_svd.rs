//! # `lie_svd.rs` - unified dispatcher over the release SVD family
//!
//! `LieSvd::solve` is the single entry point for this exploration series'
//! current pre-release path. It picks between:
//! - `lie_svd_micro.rs`: fixed tiny rotor microkernels for `N <= 4`.
//! - `lie_svd_small.rs`: polar decomposition + dense Jacobi. This remains the
//!   fastest accurate path for small and already-nearly-diagonal matrices.
//! - `lie_svd_block4.rs`: a `4x4` macro-rotor warm start using local tiny SVD
//!   cells and power-of-two butterfly quartets, followed by digital polish.
//! - `lie_svd_coreflow.rs` + `lie_svd_topowarm.rs`: an opt-in-by-triage
//!   geometric synergy path for suspicious structured or degenerate inputs.
//! - `lie_svd_hybrid.rs`: dual-tiled Lie/Clifford geometric preconditioning
//!   followed by the exact `LieSvdSmall` polish.
//!
//! ## Release shape
//!
//! The old tiled `lie_svd_big.rs` and the standalone geometric experiments are
//! kept only as legacy benchmark targets. The active release dispatcher is now
//! adaptive: it keeps `Small` as the hot dense-Jacobi tier, but enables
//! `CoreFlow + TopoWarm + Repeller` when a cheap triage pass flags structural
//! degeneracy.
//!
//! Conceptually, the hybrid tier uses the `M = I + eA` Clifford view without
//! changing memory layout: `I` appears as a scalar numerical anchor in local
//! rotor denominators, and `eA` is represented by ordinary `f64` two-sided
//! Givens rotations.

use ndarray::{Array1, Array2};

/// Fixed tiny rotor kernels avoid the setup overhead of the general solver.
pub const MICRO_MAX_N: usize = 4;

/// Dense Jacobi is still the best measured default through this size.
///
/// `LieSvdHybrid` remains available explicitly, but the latest stress report
/// shows its geometric preconditioner is not a reliable speed win below this
/// range. Keep the dispatcher conservative.
pub const SMALL_MAX_N: usize = 512;

/// Avoid spending geometric preconditioner time on matrices already very close
/// to diagonal form.
pub const NEARLY_DIAGONAL_RATIO: f64 = 1e-5;

pub struct LieSvd;

impl LieSvd {
    /// Dispatches to the adaptive release route.
    /// Returns `(U, Σ, Vᵗ)`.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        crate::lie_svd_adaptive::LieSvdAdaptive::solve(mat)
    }
}

#[allow(dead_code)]
fn offdiag_ratio(mat: &Array2<f64>) -> f64 {
    let n = mat.nrows();
    let mut off = 0.0_f64;
    let mut all = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let v = mat[[i, j]];
            all += v * v;
            if i != j {
                off += v * v;
            }
        }
    }
    off.sqrt() / all.sqrt().max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn reconstruction_err(n: usize, a: &Array2<f64>) -> f64 {
        let (u, sigma, vt) = LieSvd::solve(a);
        let mut sigma_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(&vt);
        (&recon - a).mapv(|x| x * x).sum().sqrt()
    }

    #[test]
    fn test_dispatch_uses_small_below_threshold() {
        let n = 32;
        let mut rng = StdRng::seed_from_u64(3);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        assert!(reconstruction_err(n, &a) < 1e-8);
    }

    #[test]
    fn test_dispatch_adaptive_trace_available() {
        let n = 24;
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            let block = if i / 6 == j / 6 { 4.0 } else { 0.02 };
            block + (((i + j) * 31) as f64).sin() * 1e-3
        });
        let (_svd, trace) = crate::lie_svd_adaptive::LieSvdAdaptive::solve_with_trace(
            &a,
            crate::lie_svd_adaptive::LieSvdAdaptiveParams::default(),
        );
        assert_eq!(
            trace.route,
            crate::lie_svd_adaptive::AdaptiveRoute::CoreFlowTopo
        );
    }

    #[test]
    fn test_hybrid_large_tier_direct_smoke() {
        // Keep this below the dispatcher threshold so unit tests stay quick,
        // but exercise the new large-tier implementation directly.
        let n = 32;
        let mut rng = StdRng::seed_from_u64(5);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let (u, sigma, vt) = crate::lie_svd_hybrid::LieSvdHybrid::solve(&a);
        let mut sigma_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(&vt);
        assert!((&recon - &a).mapv(|x| x * x).sum().sqrt() < 1e-6);
    }

    #[test]
    fn test_dispatch_small_sizes() {
        for n in [1usize, 2, 3, 4, 5, 17, 63] {
            let mut rng = StdRng::seed_from_u64(100 + n as u64);
            let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
            assert!(reconstruction_err(n, &a) < 1e-6, "n={n}");
        }
    }
}
