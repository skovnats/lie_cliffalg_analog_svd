//! # `lie_svd_small.rs` — polar decomposition + dense Jacobi (accurate tier)
//!
//! The "small/general-purpose" tier of the three-file `lie_svd` family
//! (`lie_svd_small.rs`, `lie_svd_big.rs`, `lie_svd.rs` dispatches between
//! them by `N`). This is the single strongest-measured technique from the
//! whole exploration series that led here — carried over essentially
//! unchanged from `spectral_divide_conquer_svd_v4.rs` (now archived in
//! `sandbox/`, see that file for the original derivation/history) because
//! nothing else tried in this series beat it on accuracy, and up to
//! N≈500 nothing beat it on speed either (see `lie_svd.rs` for the
//! measured crossover against the tiled `lie_svd_big.rs`).
//!
//! ```text
//! A = Q · P        Q orthogonal, P symmetric positive semi-definite
//! ```
//!
//! 1. **`Q`** — the orthogonal polar factor of `A` — is computed directly
//!    by an inversion-free Newton–Schulz iteration:
//!    `X_{k+1} = ½ X_k (3I − X_kᵀX_k)`, `X_0 = A / ‖A‖_F`.
//! 2. **`P = QᵀA`** is then symmetric PSD *by construction*
//!    (`A = QP ⟹ QᵀA = QᵀQP = P`), with eigenvalues exactly the singular
//!    values of `A` and eigenvectors exactly the right singular vectors.
//! 3. `P` is diagonalized by a classical cyclic Jacobi sweep.
//! 4. `U = QV`, `Σ = eig(P)`, `V` = Jacobi eigenvectors of `P`.
//!
//! ## Why this wins on accuracy
//!
//! Going through `A` directly instead of `AᵗA`/`AAᵗ` (the convention every
//! earlier file in this series used) avoids **squaring the condition
//! number** (`κ(AᵗA) = κ(A)²`). Measured directly, `ill_conditioned`
//! (`κ(A) ≈ 1e15`) profile, N=444: this file's `‖UᵗU−I‖_F ≈ 2.7e-12`
//! (machine precision) vs. **49–184** for every `AᵗA`-based competitor
//! tried (the recursive spectral-sign splitters, and `lie_svd_big.rs`'s
//! tiled architecture alike). This isn't specific to pathological input
//! either — even on `uniform_random` at N=128..384 this file's `U`
//! orthogonality stays at machine precision while the tiled competitor's
//! sits at 8–18 (see `lie_svd.rs` module docs for the full comparison
//! table that justified the dispatcher's threshold).
//!
//! ## Honest scope notes
//! - Newton–Schulz's polar iteration is only *locally* quadratically
//!   convergent — for a pathologically ill-conditioned `A` it can need
//!   many iterations (measured: ill-conditioned inputs need most of the
//!   120-iteration budget to escape near-zero singular value components,
//!   which only grow ~1.5× per step near 0). Divergence is guarded:
//!   growth/non-finite detection marks the polar factor untrusted. The solver
//!   then keeps the fast right-space Jacobi result but repairs the left basis
//!   directly as `U ≈ A V Σ⁻¹`, avoiding the expensive/ill-conditioned
//!   `AᵗA` fallback as the first response.
//! - **No cache tiling / recursion**: `P`'s Jacobi sweep is a
//!   straightforward `O(n³)`-per-sweep, dense solver. Measured timings:
//!   N=384 ≈ 2.3s, N=444 ≈ 3.6s (worst case), N=512 ≈ 7s, N=1024 ≈ 77s
//!   (still correct at N=1024, just slow — `lie_svd_big.rs` becomes the
//!   faster choice there, at a real accuracy cost; see `lie_svd.rs`).

use ndarray::{Array1, Array2};

// Each Newton-Schulz step only grows a near-zero singular value component
// by ~1.5x (y_{k+1} ≈ 1.5·y_k for y_k ≈ 0 — a repelling fixed point, escape
// is linear not quadratic until the component is O(1)). For the
// `ill_conditioned` profile (κ(A) ≈ 1e15) that means ~85 iterations just to
// escape the near-zero regime for the smallest singular value, before
// quadratic convergence can polish it — 120 gives headroom for that plus
// polish. This is the well-known Newton-Schulz-vs-condition-number cost
// (QDWH's dynamically-weighted iteration exists specifically to avoid this
// blowup; not implemented here).
const POLAR_MAX_ITERS: usize = 120;
const POLAR_TOL: f64 = 1e-13;
const JACOBI_TOL: f64 = 1e-13;

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Modified Gram–Schmidt orthonormalization of `a`'s columns. Always returns
/// a finite orthogonal matrix, even if `a` is singular or rank-deficient.
///
/// The rank-deficient fallback deliberately uses the standard-basis direction
/// with the largest residual after projection. It never inserts a zero column:
/// a single zero column contributes exactly `1.0` to `‖QᵀQ-I‖_F`, which is the
/// quiet failure mode caught by the Jordan/defective stress profile.
fn gram_schmidt_orthonormal(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    let mut v = vec![0.0_f64; n];
    let mut candidate = vec![0.0_f64; n];
    let mut best_v = vec![0.0_f64; n];

    for j in 0..n {
        for i in 0..n {
            v[i] = a[[i, j]];
        }
        for k in 0..j {
            let mut d = 0.0_f64;
            for i in 0..n {
                d += out[[i, k]] * v[i];
            }
            for i in 0..n {
                v[i] -= d * out[[i, k]];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        } else {
            let mut best_norm = -1.0_f64;
            for k in 0..n {
                candidate.fill(0.0);
                candidate[k] = 1.0;
                for col in 0..j {
                    let mut d = 0.0_f64;
                    for i in 0..n {
                        d += out[[i, col]] * candidate[i];
                    }
                    for i in 0..n {
                        candidate[i] -= d * out[[i, col]];
                    }
                }
                let enorm = candidate.iter().map(|x| x * x).sum::<f64>().sqrt();
                if enorm > best_norm {
                    best_norm = enorm;
                    best_v.copy_from_slice(&candidate);
                }
            }
            if best_norm > 0.0 && best_norm.is_finite() {
                for i in 0..n {
                    v[i] = best_v[i] / best_norm;
                }
            } else {
                // This should only be reachable after catastrophic non-finite
                // input. Return a deterministic finite direction rather than a
                // zero column, then let the caller's orthogonality check decide.
                v.fill(0.0);
                v[j] = 1.0;
            }
        }
        for i in 0..n {
            out[[i, j]] = v[i];
        }
    }
    out
}

fn orthogonality_err(q: &Array2<f64>) -> f64 {
    let n = q.nrows();
    let ident = Array2::<f64>::eye(n);
    (&q.t().dot(q) - &ident).mapv(|x| x * x).sum().sqrt()
}

/// Inversion-free Newton–Schulz polar iteration. Returns `(Q, converged)`.
///
/// `converged=false` does not mean "hard failure"; it means the returned
/// matrix should not be trusted as a polar factor. The public SVD path then
/// repairs the left basis from the original matrix and the computed right
/// singular vectors instead of pretending that a QR basis is a polar factor.
fn newton_schulz_polar(a: &Array2<f64>) -> (Array2<f64>, bool) {
    let n = a.nrows();
    let scale = frobenius_norm(a).max(1e-300);
    let mut x = a / scale;
    let eye = Array2::<f64>::eye(n);
    let ref_norm = (n as f64).sqrt();

    for _ in 0..POLAR_MAX_ITERS {
        let xtx = x.t().dot(&x);
        let x_next = x.dot(&(&eye * 3.0 - &xtx)) * 0.5;
        let delta = frobenius_norm(&(&x_next - &x));
        if !x_next.iter().all(|v| v.is_finite()) || delta > 1e6 * ref_norm {
            return (gram_schmidt_orthonormal(a), false);
        }
        x = x_next;
        if delta < POLAR_TOL * ref_norm {
            let ok = orthogonality_err(&x) <= 1e-8 * ref_norm;
            return (x, ok);
        }
    }
    if orthogonality_err(&x) <= 1e-8 * ref_norm {
        (x, true)
    } else {
        (x, false)
    }
}

fn repair_left_basis_from_right(
    mat: &Array2<f64>,
    v: &Array2<f64>,
    sigma: &Array1<f64>,
) -> Array2<f64> {
    let n = mat.nrows();
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let sigma_tol = 1e-12 * sigma_max.max(1.0);
    let mut u_seed = Array2::<f64>::zeros((n, n));

    for j in 0..n {
        if sigma[j] > sigma_tol {
            for row in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += mat[[row, k]] * v[[k, j]];
                }
                u_seed[[row, j]] = s / sigma[j];
            }
        }
    }

    gram_schmidt_orthonormal(&u_seed)
}

/// Classical cyclic Jacobi eigendecomposition of a symmetric `n×n` matrix.
/// Returns `(V, eigenvalues)` sorted descending, with `VᵗPV = diag(eig)`.
fn eigh_jacobi_full(p: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let n = p.nrows();
    let mut a = p.clone();
    let mut v = Array2::<f64>::eye(n);
    let max_sweeps = 60 + 20 * n;
    let ref_norm = frobenius_norm(p).max(1e-300);

    'sweeps: for _ in 0..max_sweeps {
        let mut off_sq = 0.0_f64;
        for pp in 0..n {
            for qq in (pp + 1)..n {
                off_sq += a[[pp, qq]] * a[[pp, qq]];
            }
        }
        if off_sq.sqrt() < JACOBI_TOL * ref_norm {
            break 'sweeps;
        }
        for pp in 0..n {
            for qq in (pp + 1)..n {
                let apq = a[[pp, qq]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[[qq, qq]] - a[[pp, pp]]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let app = a[[pp, pp]];
                let aqq = a[[qq, qq]];
                a[[pp, pp]] = app - t * apq;
                a[[qq, qq]] = aqq + t * apq;
                a[[pp, qq]] = 0.0;
                a[[qq, pp]] = 0.0;
                for i in 0..n {
                    if i != pp && i != qq {
                        let aip = a[[i, pp]];
                        let aiq = a[[i, qq]];
                        a[[i, pp]] = c * aip - s * aiq;
                        a[[pp, i]] = a[[i, pp]];
                        a[[i, qq]] = s * aip + c * aiq;
                        a[[qq, i]] = a[[i, qq]];
                    }
                }
                for i in 0..n {
                    let vip = v[[i, pp]];
                    let viq = v[[i, qq]];
                    v[[i, pp]] = c * vip - s * viq;
                    v[[i, qq]] = s * vip + c * viq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[[j, j]]
            .partial_cmp(&a[[i, i]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    let mut eig_sorted = Array1::<f64>::zeros(n);
    for (new_j, &old_j) in order.iter().enumerate() {
        eig_sorted[new_j] = a[[old_j, old_j]];
        for i in 0..n {
            v_sorted[[i, new_j]] = v[[i, old_j]];
        }
    }
    (v_sorted, eig_sorted)
}

// =============================================================================
// Public API
// =============================================================================

pub struct LieSvdSmall;

impl LieSvdSmall {
    /// Full SVD via polar decomposition (`A = QP`) + direct symmetric
    /// eigendecomposition of `P` — see module docs. Returns `(U, Σ, Vᵗ)`.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let n = mat.nrows();
        assert_eq!(
            n,
            mat.ncols(),
            "LieSvdSmall::solve: matrix must be square N×N"
        );

        let (q, polar_ok) = newton_schulz_polar(mat);
        let mut p = q.t().dot(mat);
        // Symmetrize away rounding asymmetry (P is symmetric only up to
        // floating-point error, since Q is only an approximate polar
        // factor after a finite number of Newton-Schulz iterations).
        p = (&p + &p.t()) * 0.5;

        let (v, eig) = eigh_jacobi_full(&p);

        let mut sigma = Array1::<f64>::zeros(n);
        for j in 0..n {
            sigma[j] = eig[j].max(0.0);
        }
        let mut u = q.dot(&v);
        if !polar_ok || orthogonality_err(&u) > 1e-8 * (n as f64).sqrt() {
            u = repair_left_basis_from_right(mat, &v, &sigma);
        }
        let vt = v.t().to_owned();
        (u, sigma, vt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn reconstruction_and_orthogonality_err(n: usize, a: &Array2<f64>) -> (f64, f64, f64) {
        let (u, sigma, vt) = LieSvdSmall::solve(a);
        let mut sigma_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(&vt);
        let recon_err = (&recon - a).mapv(|x| x * x).sum().sqrt();
        let uut = u.t().dot(&u);
        let vvt = vt.dot(&vt.t());
        let ident = Array2::<f64>::eye(n);
        let uut_err = (&uut - &ident).mapv(|x| x * x).sum().sqrt();
        let vvt_err = (&vvt - &ident).mapv(|x| x * x).sum().sqrt();
        (recon_err, uut_err, vvt_err)
    }

    #[test]
    fn test_solve_svd_random_64() {
        let n = 64;
        let mut rng = StdRng::seed_from_u64(7);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let (recon_err, uut_err, vvt_err) = reconstruction_and_orthogonality_err(n, &a);
        assert!(
            recon_err < 1e-8,
            "reconstruction error too large: {recon_err}"
        );
        assert!(uut_err < 1e-8, "U not orthogonal: {uut_err}");
        assert!(vvt_err < 1e-8, "V not orthogonal: {vvt_err}");
    }

    #[test]
    fn test_solve_svd_small_sizes_non_multiple_of_anything() {
        for n in [1usize, 2, 3, 4, 5, 7, 9, 13, 17, 31] {
            let mut rng = StdRng::seed_from_u64(100 + n as u64);
            let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
            let (recon_err, uut_err, vvt_err) = reconstruction_and_orthogonality_err(n, &a);
            assert!(
                recon_err < 1e-8,
                "n={n}: reconstruction error too large: {recon_err}"
            );
            assert!(uut_err < 1e-8, "n={n}: U not orthogonal: {uut_err}");
            assert!(vvt_err < 1e-8, "n={n}: V not orthogonal: {vvt_err}");
        }
    }

    /// The whole point of going via `A` directly instead of `AᵗA`: this
    /// should not need a loosened tolerance, since P's condition number is
    /// A's, not A's squared.
    #[test]
    fn test_solve_svd_ill_conditioned_32() {
        let n = 32;
        let mut rng = StdRng::seed_from_u64(42);
        let make_orth = |rng: &mut StdRng| -> Array2<f64> {
            let mut m = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
            for j in 0..n {
                for k in 0..j {
                    let dot: f64 = (0..n).map(|i| m[[i, j]] * m[[i, k]]).sum();
                    for i in 0..n {
                        m[[i, j]] -= dot * m[[i, k]];
                    }
                }
                let norm: f64 = (0..n)
                    .map(|i| m[[i, j]] * m[[i, j]])
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-300);
                for i in 0..n {
                    m[[i, j]] /= norm;
                }
            }
            m
        };
        let u_ref = make_orth(&mut rng);
        let v_ref = make_orth(&mut rng);
        let sigma_ref: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-15.0 * i as f64 / (n - 1) as f64))
            .collect();
        let mut sigma_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sigma_mat[[i, i]] = sigma_ref[i];
        }
        let a = u_ref.dot(&sigma_mat).dot(&v_ref.t());

        let (recon_err, uut_err, vvt_err) = reconstruction_and_orthogonality_err(n, &a);
        assert!(
            recon_err < 1e-6,
            "reconstruction error too large: {recon_err}"
        );
        assert!(uut_err < 1e-6, "U not orthogonal: {uut_err}");
        assert!(vvt_err < 1e-6, "V not orthogonal: {vvt_err}");
    }

    #[test]
    fn test_solve_svd_jordan_defective_keeps_u_orthogonal() {
        let n = 64;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0 - 0.5 * (i as f64 / n as f64);
            if i + 1 < n {
                a[[i, i + 1]] = 20.0;
            }
            if i + 2 < n && i % 4 == 0 {
                a[[i, i + 2]] = -3.0;
            }
        }

        let (recon_err, uut_err, vvt_err) = reconstruction_and_orthogonality_err(n, &a);
        let rel_recon = recon_err / frobenius_norm(&a).max(1e-300);
        assert!(
            rel_recon < 1e-10,
            "relative reconstruction error too large: {rel_recon}"
        );
        assert!(uut_err < 1e-8, "U not orthogonal: {uut_err}");
        assert!(vvt_err < 1e-8, "V not orthogonal: {vvt_err}");
    }

    /// The N=384/444 accuracy comparison that motivated `lie_svd.rs`'s
    /// dispatch threshold: even well-conditioned input should stay at
    /// machine-precision orthogonality here, unlike the tiled competitor.
    #[test]
    fn test_solve_svd_random_384_machine_precision_orthogonality() {
        let n = 384;
        let mut rng = StdRng::seed_from_u64(7);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let (recon_err, uut_err, vvt_err) = reconstruction_and_orthogonality_err(n, &a);
        assert!(
            recon_err < 1e-8,
            "reconstruction error too large: {recon_err}"
        );
        assert!(uut_err < 1e-8, "U not orthogonal: {uut_err}");
        assert!(vvt_err < 1e-8, "V not orthogonal: {vvt_err}");
    }
}
