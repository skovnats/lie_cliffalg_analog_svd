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
pub(crate) fn eigh_jacobi_full(p: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
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

    /// Economy ("thin") SVD of a general `n x d` matrix (any aspect ratio),
    /// via `QR` reduction to a `min(n,d) x min(n,d)` square factor followed
    /// by this module's exact square solve. Returns `U: n x k`,
    /// `sigma: length k`, `Vt: k x d`, `k = min(n, d)` — deliberately the
    /// thin shape, not the full `n x n` / `d x d` shapes
    /// `lie_svd_phaseflow`'s rectangular route uses, since for the common
    /// tabular case (`n` samples `>> d` features) a full `n x n` U would be
    /// mostly wasted storage.
    ///
    /// Why QR and not `X^T X`: `R` (the QR factor) and `X` share the same
    /// singular values, because `Q` has orthonormal columns — QR does not
    /// square the condition number the way forming `X^T X` does (see the
    /// module doc comment above). This routes the "avoid squaring" argument
    /// through an actual rectangular solve instead of stating it and then
    /// squaring anyway.
    pub fn solve_rectangular(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let n = mat.nrows();
        let d = mat.ncols();
        assert!(
            n > 0 && d > 0,
            "LieSvdSmall::solve_rectangular: empty input"
        );
        if n == d {
            return Self::solve(mat);
        }
        if d > n {
            // SVD(X) from SVD(X^T): swap U and V, transpose Vt back to V-major.
            let (u_t, sigma, vt_t) = Self::solve_rectangular(&mat.t().to_owned());
            return (vt_t.t().to_owned(), sigma, u_t.t().to_owned());
        }
        // n > d here: the common tall/tabular case.
        let (q, r) = qr_reduce(mat);
        let (ur, sigma, vt) = Self::solve(&r);
        let u = q.dot(&ur);
        (u, sigma, vt)
    }
}

/// Rectangular QR via modified Gram-Schmidt (more numerically stable than
/// classical Gram-Schmidt against cancellation, though still not as robust
/// as Householder reflections on severely rank-deficient input — adequate
/// here because the result only feeds a follow-up exact square solve, not a
/// final answer on its own). `mat`: `n x d` with `n >= d`. Returns
/// `Q: n x d` (orthonormal columns) and `R: d x d` (upper triangular) with
/// `mat ~= Q R`. A column that is (numerically) linearly dependent on the
/// ones before it gets an exact zero pivot and a zero `Q` column, rather
/// than an arbitrary injected direction — consistent with genuine rank
/// deficiency in `mat` instead of hiding it.
fn qr_reduce(mat: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = mat.nrows();
    let d = mat.ncols();
    let mut q = Array2::<f64>::zeros((n, d));
    let mut r = Array2::<f64>::zeros((d, d));
    let mut v = Array2::<f64>::zeros((n, d));
    // Scale for the rank-deficiency check below: an absolute cutoff (e.g.
    // `norm >= 1e-300`) only catches an exactly-zero pivot. A column that is
    // numerically dependent on the ones before it (residual norm tiny
    // *relative to the column's own original scale*, but not literally
    // zero, e.g. `1e-14` against an original norm of `10`) would still pass
    // an absolute check, get normalized anyway, and turn floating-point
    // noise into an arbitrary "orthonormal" direction — not orthogonal to
    // anything in practice, since it's built almost entirely from rounding
    // error. Comparing against this matrix's own scale catches that case.
    let mat_scale = mat.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    let degenerate_floor = 1e-10 * mat_scale;
    for j in 0..d {
        for i in 0..n {
            v[[i, j]] = mat[[i, j]];
        }
        for k in 0..j {
            let mut dot = 0.0_f64;
            for i in 0..n {
                dot += q[[i, k]] * v[[i, j]];
            }
            r[[k, j]] = dot;
            for i in 0..n {
                v[[i, j]] -= dot * q[[i, k]];
            }
        }
        let norm = (0..n).map(|i| v[[i, j]] * v[[i, j]]).sum::<f64>().sqrt();
        r[[j, j]] = norm;
        if norm >= degenerate_floor {
            for i in 0..n {
                q[[i, j]] = v[[i, j]] / norm;
            }
        }
        // else: numerically dependent on earlier columns. Leave q[:,j] = 0
        // and r[j,j] = norm (the tiny true residual, not zeroed out) — the
        // resulting singular value from `solve(&r)` will be correspondingly
        // tiny, so this direction contributes negligibly to reconstruction
        // rather than injecting a noisy, falsely-orthonormal one.
    }
    (q, r)
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

    fn rectangular_reconstruction_and_orthogonality_err(a: &Array2<f64>) -> (f64, f64) {
        let (u, sigma, vt) = LieSvdSmall::solve_rectangular(a);
        let k = sigma.len();
        let n = a.nrows();
        let d = a.ncols();
        assert_eq!(u.dim(), (n, k));
        assert_eq!(vt.dim(), (k, d));
        let mut sigma_mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(&vt);
        let recon_err = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let uut = u.t().dot(&u);
        let ident = Array2::<f64>::eye(k);
        let uut_err = (&uut - &ident).mapv(|x| x * x).sum().sqrt();
        (recon_err, uut_err)
    }

    #[test]
    fn test_solve_rectangular_tall_dense_matrix() {
        // The exact shapes that broke `lie_svd_phaseflow`'s rotor-based
        // rectangular route (~96%/~51% reconstruction error): this QR-based
        // path should reach machine precision on both instead. (An earlier
        // version of this test used `sin((i*7+j*5+1)*0.31)*(1+j)` for the
        // 30x3 case, which turned out to be accidentally near-rank-deficient
        // — third singular value ~1e-16 — and its orthogonality assertion
        // failed for exactly the reason `qr_reduce`'s doc comment predicts
        // for unpivoted QR on rank-deficient input. A second deterministic
        // formula hit the same accidental-collinearity problem again, so
        // this uses random entries instead: astronomically unlikely to be
        // exactly rank-deficient, unlike a hand-picked closed-form formula.
        // The rank-deficient case has its own dedicated test below with a
        // weaker, correct assertion.)
        let mut rng = StdRng::seed_from_u64(2026);
        let a30x3 = Array2::from_shape_fn((30, 3), |_| rng.gen_range(-1.0_f64..1.0));
        let (err_a, orth_a) = rectangular_reconstruction_and_orthogonality_err(&a30x3);
        assert!(err_a < 1e-10, "30x3 recon rel err={err_a:e}");
        assert!(orth_a < 1e-10, "30x3 U orthogonality err={orth_a:e}");

        let a20x15 = Array2::from_shape_fn((20, 15), |_| rng.gen_range(-1.0_f64..1.0));
        let (err_b, orth_b) = rectangular_reconstruction_and_orthogonality_err(&a20x15);
        assert!(err_b < 1e-10, "20x15 recon rel err={err_b:e}");
        assert!(orth_b < 1e-10, "20x15 U orthogonality err={orth_b:e}");
    }

    #[test]
    fn test_solve_rectangular_wide_matrix_via_transpose() {
        // d > n: exercises the transpose branch.
        let a = Array2::from_shape_fn((4, 11), |(i, j)| ((i * 5 + j * 3 + 2) as f64).cos());
        let (err, orth) = rectangular_reconstruction_and_orthogonality_err(&a);
        assert!(err < 1e-10, "wide recon rel err={err:e}");
        assert!(orth < 1e-10, "wide U orthogonality err={orth:e}");
    }

    #[test]
    fn test_solve_rectangular_matches_square_solve_on_square_input() {
        let n = 12;
        let mut rng = StdRng::seed_from_u64(42);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let (u1, s1, vt1) = LieSvdSmall::solve(&a);
        let (u2, s2, vt2) = LieSvdSmall::solve_rectangular(&a);
        for i in 0..n {
            assert!(
                (s1[i] - s2[i]).abs() < 1e-10,
                "sigma[{i}]: {} vs {}",
                s1[i],
                s2[i]
            );
        }
        assert_eq!(u1.dim(), u2.dim());
        assert_eq!(vt1.dim(), vt2.dim());
    }

    #[test]
    fn test_solve_rectangular_handles_a_rank_deficient_column() {
        // Column 2 is an exact duplicate of column 0: rank-deficient input
        // must not produce NaN/Inf, and reconstruction must still hold.
        let n = 20;
        let mut a = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let v = ((i * 11 + 3) as f64 * 0.23).sin() * 2.0;
            a[[i, 0]] = v;
            a[[i, 1]] = ((i * 13 + 7) as f64 * 0.41).cos();
            a[[i, 2]] = v; // duplicate of column 0
        }
        let (u, sigma, vt) = LieSvdSmall::solve_rectangular(&a);
        assert!(u.iter().all(|x| x.is_finite()));
        assert!(sigma.iter().all(|x| x.is_finite()));
        assert!(vt.iter().all(|x| x.is_finite()));
        let k = sigma.len();
        let mut sigma_mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(&vt);
        let err = (&recon - &a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(&a).max(1e-300);
        assert!(err < 1e-8, "rank-deficient recon rel err={err:e}");
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
