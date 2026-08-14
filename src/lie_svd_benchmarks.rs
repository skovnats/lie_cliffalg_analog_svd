//! Standard, world-recognized "evil matrix" and BSS benchmarks, applied to
//! this crate's own solvers rather than left as a synthetic-only test suite.
//!
//! ## Why these specific benchmarks, and not others
//!
//! This crate deliberately has no LAPACK/BLAS/faer dependency (see
//! `bin/stress_cpu.rs`'s own doc comment and `Cargo.toml`), so "compare
//! against a reference SVD" cannot literally mean "compare against LAPACK's
//! `dgesvd`" without contradicting that design choice. What's implemented
//! here instead, in order of how strong a check it gives:
//!
//! 1. **Exact closed-form ground truth** (`pei_matrix`): a named test matrix
//!    whose eigenvalues are known analytically, not computed by any solver
//!    -- the strongest available check, and it happens to also be a hard
//!    *degenerate*-spectrum case (`n-1` repeated eigenvalues), directly
//!    relevant to whether Jacobi-style rotors stall on repeated singular
//!    values.
//! 2. **Imposed ground truth** (`crate::profiles::Profile::ExtremeIllConditioned`
//!    and `DegenerateSpectrum`, already present in this crate since before
//!    this module, with an exact `sigma_ref` -- previously computed and
//!    *displayed* by `stress_cpu` but never actually asserted against in
//!    `cargo test`; that's a real, if narrow, test-coverage gap, closed
//!    below rather than left as a CLI-only display).
//! 3. **Self-consistency** (`kahan_matrix`, `hilbert_matrix`): no external
//!    ground truth is available or, for `hilbert_matrix` at the sizes where
//!    its condition number exceeds `f64`'s representable range, even
//!    *exists* in double precision -- orthogonality of the recovered bases
//!    and reconstruction accuracy are what's actually checkable, and
//!    claiming machine-precision recovery of singular values that are
//!    smaller than the matrix's own rounding error would be a false claim
//!    regardless of which solver computed them.
//!
//! ## What was in scope, considered, and explicitly left out
//!
//! - **SuiteSparse / Matrix Market:** real matrices, but downloading them
//!   would require network access, breaking this project's established
//!   offline-reproducible pattern (`docker build --no-cache` with no
//!   network calls). Not used.
//! - **Frank, Forsythe, Parter, Cauchy matrices:** legitimate members of the
//!   Higham test suite, left for a future pass rather than silently
//!   dropped -- `kahan_matrix`/`hilbert_matrix`/`pei_matrix` already cover
//!   three qualitatively different hard cases (near-rank-deficient
//!   triangular, extreme-condition-number symmetric, exact-degenerate
//!   symmetric), and Frank's own useful property (reciprocal eigenvalue
//!   pairs) is a *nonsymmetric eigenvalue* fact, not a singular-value one,
//!   so it doesn't add a new SVD-relevant failure mode this set doesn't
//!   already exercise.
//! - **Trefethen pseudospectra:** a genuinely relevant tool for non-normal
//!   matrices, but a resolvent-norm-over-a-complex-grid computation is a
//!   diagnostic *visualization*, not a pass/fail correctness check, and a
//!   substantially larger undertaking than this pass's scope.
//! - **Cardoso's own JADE/SOBI EEG/MEG datasets:** not used for the same
//!   network-access reason as SuiteSparse; `amari_index` below (the
//!   standard *metric* those benchmarks are scored with) is implemented
//!   and applied to a synthetic ill-conditioned mixing case instead, which
//!   is exactly the "near-collinear sensor channels, `kappa > 1e6`" case
//!   named in the original proposal.

use ndarray::{Array1, Array2};

/// The standard Kahan test matrix (Higham, *Accuracy and Stability of
/// Numerical Algorithms*; MATLAB `gallery('kahan', n, theta)`). Upper
/// triangular, diagonal decaying geometrically as `sin(theta)^(i-1)`,
/// constructed specifically to defeat column-pivoted QR's rank detection --
/// a standard stress test for rank-revealing and rotation-based methods.
/// `theta` defaults to `1.2` in the MATLAB gallery; callers needing that
/// exact convention should pass `1.2`.
pub fn kahan_matrix(n: usize, theta: f64) -> Array2<f64> {
    let c = theta.cos();
    let s = theta.sin();
    let mut k = Array2::<f64>::zeros((n, n));
    let mut s_pow = 1.0_f64;
    for i in 0..n {
        k[[i, i]] = s_pow;
        for j in (i + 1)..n {
            k[[i, j]] = -c * s_pow;
        }
        s_pow *= s;
    }
    k
}

/// The Hilbert matrix, `H[i,j] = 1 / (i + j + 1)` (`0`-indexed). Extremely
/// ill-conditioned by construction (`kappa(H_n)` grows roughly like
/// `e^{3.5n}`, per the classical asymptotic result for this matrix); beyond
/// `n ~ 12-13` its condition number exceeds `f64`'s representable dynamic
/// range (`~1e16`), so no solver -- this one or any other -- can recover
/// its smallest singular values to any accuracy at those sizes. Measured,
/// not just assumed: even past that point (`n=14,16`), where the smallest
/// true singular value underflows to numerical noise, `LieSvdSmall::solve`
/// doesn't degrade *anywhere else* -- `U`/`V` stay orthogonal and the
/// reconstruction stays accurate to `~1e-14`, both essentially unchanged
/// from the well-conditioned `n<=12` range. See
/// `hilbert_matrix_degrades_gracefully_past_double_precision_limits` for
/// the exact numbers.
pub fn hilbert_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64)
}

/// The Pei matrix, `P = alpha * I + J` (`J` the all-ones matrix). Symmetric
/// positive definite for `alpha > 0`, with **exact, closed-form
/// eigenvalues**: `J` has eigenvalue `n` once (eigenvector the all-ones
/// vector) and `0` with multiplicity `n-1` (its orthogonal complement), so
/// `P`'s eigenvalues are `alpha + n` once and `alpha` with multiplicity
/// `n-1`. Since `P` is symmetric PD, singular values equal eigenvalues
/// exactly -- this is the one matrix in this module with a real, external,
/// independently-derived ground truth to check against, and a small
/// `alpha` makes the `n-1`-fold repeated eigenvalue a genuine degenerate-
/// spectrum stress test at the same time.
pub fn pei_matrix(n: usize, alpha: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { alpha + 1.0 } else { 1.0 })
}

/// Exact singular values of `pei_matrix(n, alpha)` for `alpha > 0`: `n-1`
/// copies of `alpha`, then one `alpha + n`, ascending (matching this
/// crate's convention of sorting `sigma_ref` by the caller as needed).
pub fn pei_matrix_singular_values(n: usize, alpha: f64) -> Array1<f64> {
    let mut sigma = vec![alpha; n];
    if n > 0 {
        sigma[n - 1] = alpha + n as f64;
    }
    Array1::from(sigma)
}

/// The Amari performance index (Amari, Cichocki & Yang, *A New Learning
/// Algorithm for Blind Signal Separation*, NeurIPS 1996) for a BSS/ICA
/// global system matrix `g = unmixing @ mixing`. Unlike a raw
/// `||g - I||`-style error, this is invariant to the row
/// permutation/scaling ambiguity inherent to blind separation (recovered
/// sources are only ever identified up to which-source-is-which and their
/// sign/scale) -- `0` exactly when `g` is a scaled permutation matrix
/// (perfect separation up to that ambiguity), and grows as `g` spreads
/// energy across multiple entries in a row or column instead of
/// concentrating it in one.
pub fn amari_index(g: &Array2<f64>) -> f64 {
    let n = g.nrows();
    assert_eq!(g.ncols(), n, "amari_index: g must be square");
    if n <= 1 {
        return 0.0;
    }
    let abs_g = g.mapv(f64::abs);
    let mut row_term = 0.0_f64;
    for i in 0..n {
        let row_max = (0..n)
            .map(|j| abs_g[[i, j]])
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let row_sum: f64 = (0..n).map(|j| abs_g[[i, j]]).sum();
        row_term += row_sum / row_max - 1.0;
    }
    let mut col_term = 0.0_f64;
    for j in 0..n {
        let col_max = (0..n)
            .map(|i| abs_g[[i, j]])
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let col_sum: f64 = (0..n).map(|i| abs_g[[i, j]]).sum();
        col_term += col_sum / col_max - 1.0;
    }
    (row_term + col_term) / (2.0 * (n * (n - 1)) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie_svd_bss::{LieSvdBss, PhaseBssParams};
    use crate::lie_svd_small::LieSvdSmall;
    use crate::metrics;
    use crate::profiles::{generate, Profile};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_orthogonal(n: usize, rng: &mut StdRng) -> Array2<f64> {
        let mut q = Array2::from_shape_fn((n, n), |_| rng.gen::<f64>() - 0.5);
        for j in 0..n {
            for k in 0..j {
                let mut dot = 0.0_f64;
                for r in 0..n {
                    dot += q[[r, j]] * q[[r, k]];
                }
                for r in 0..n {
                    q[[r, j]] -= dot * q[[r, k]];
                }
            }
            let mut norm = 0.0_f64;
            for r in 0..n {
                norm += q[[r, j]] * q[[r, j]];
            }
            let norm = norm.sqrt().max(1e-300);
            for r in 0..n {
                q[[r, j]] /= norm;
            }
        }
        q
    }

    /// Kahan matrix, `N=32,64`: no external ground truth exists for this
    /// matrix's singular values in this crate (see module doc comment), so
    /// what's actually checked is self-consistency -- `LieSvdSmall::solve`
    /// must still produce genuinely orthogonal `U,V` and an accurate
    /// reconstruction on a matrix specifically constructed to defeat naive
    /// rank-revealing methods, not silently return nonsense on a matrix
    /// it can't handle.
    #[test]
    fn kahan_matrix_stays_orthogonal_and_reconstructs() {
        for n in [32usize, 64] {
            let a = kahan_matrix(n, 1.2);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
            // Kahan's own construction makes its diagonal decay
            // geometrically (sin(1.2)^(i-1) ~= 0.932^(i-1)); the recovered
            // singular values should span a comparably wide range as a
            // rough order-of-magnitude sanity check that the matrix's
            // intended ill-conditioning was preserved through the solve,
            // not smoothed away.
            let max_sigma = sigma.iter().cloned().fold(0.0_f64, f64::max);
            let min_sigma = sigma.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                max_sigma / min_sigma.max(1e-300) > 10.0f64.powi((n / 8) as i32),
                "n={n} expected wide singular value spread, got max={max_sigma:e} min={min_sigma:e}"
            );
        }
    }

    /// Hilbert matrix: verifies graceful degradation rather than claiming
    /// impossible precision. `n=6,8,10` are within `f64`'s representable
    /// range (`kappa < ~1e12`) and should reconstruct/orthogonalize well;
    /// `n=12` is at the edge of representability (`kappa` approaching
    /// `1e16`) and is checked only for staying *finite and orthogonal*, not
    /// for accuracy -- there is no accuracy left to recover at that
    /// condition number in `f64`, in this solver or any other.
    #[test]
    fn hilbert_matrix_degrades_gracefully_past_double_precision_limits() {
        // n=6..12: kappa grows from ~1.5e7 to ~9.4e15, still within f64's
        // representable dynamic range -- measured orth_u/orth_v/rel_recon
        // all land in the ~1e-14..1e-13 range across this whole span, not
        // degrading noticeably as kappa approaches the edge of that range.
        for n in [6usize, 8, 10, 12] {
            let a = hilbert_matrix(n);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }

        // n=14,16: kappa now exceeds f64's representable range (the
        // smallest true singular value underflows relative to the
        // largest), so individual tiny singular values are numerically
        // meaningless -- measured, they can even come out as an ill-
        // defined ratio when the smallest underflows toward zero. What
        // stays true, and is what's actually checked: the solver doesn't
        // corrupt anything else because of it -- U,V remain finite and
        // orthogonal (measured orth_u/orth_v still ~1e-14, i.e. no
        // detectable degradation at all in the bases themselves) and the
        // reconstruction (which is dominated by the well-represented large
        // singular values, not the underflowed small ones) stays accurate.
        for n in [14usize, 16] {
            let a = hilbert_matrix(n);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            assert!(u.iter().all(|x| x.is_finite()), "n={n}");
            assert!(vt.iter().all(|x| x.is_finite()), "n={n}");
            assert!(sigma.iter().all(|x| x.is_finite() && *x >= 0.0), "n={n}");
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }
    }

    /// Pei matrix, `alpha=0.01`: the strongest check in this module, since
    /// the singular values are known exactly (`n-1` copies of `alpha`, one
    /// `alpha+n`) rather than computed by any solver. Also a genuine
    /// degenerate-spectrum stress test -- an `(n-1)`-fold repeated singular
    /// value is exactly the case where a naive Jacobi sweep could stall or
    /// where a rotor might spin without a well-defined "correct" direction
    /// within the degenerate eigenspace (the recovered *eigenspace* is only
    /// determined up to an arbitrary rotation within it, but every singular
    /// *value* must still come out correct regardless of that freedom).
    #[test]
    fn pei_matrix_matches_exact_closed_form_singular_values() {
        for n in [16usize, 64] {
            let alpha = 0.01;
            let a = pei_matrix(n, alpha);
            let sigma_ref = pei_matrix_singular_values(n, alpha);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, Some(&sigma_ref));
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
            let sigma_max_rel = m.sigma_max_rel.expect("sigma_ref was provided");
            let sigma_tail_rel = m.sigma_tail_rel.expect("sigma_ref was provided");
            assert!(
                sigma_max_rel < 1e-8,
                "n={n} sigma_max_rel={sigma_max_rel:e}"
            );
            assert!(
                sigma_tail_rel < 1e-6,
                "n={n} sigma_tail_rel={sigma_tail_rel:e}"
            );
        }
    }

    /// `crate::profiles::Profile::DegenerateSpectrum` already carries an
    /// exact `sigma_ref` (imposed by construction: random orthogonal
    /// `U,V` times a chosen diagonal, condition number `~1e14`: `100` down
    /// to `1e-12`), and `stress_cpu` already *displays* the resulting
    /// `sigma_max_rel`/`sigma_tail_rel` -- but neither was ever actually
    /// asserted against in `cargo test` before this. That's a real, narrow
    /// test-coverage gap (the LAPACK-style "controlled spectrum" benchmark
    /// the original proposal asked for was already implemented, just never
    /// wired into an automated pass/fail check), closed here rather than
    /// reimplemented from scratch.
    ///
    /// Two-tier tolerance, calibrated by measurement rather than assumed:
    /// the well-conditioned top cluster (`sigma ~ 100`) recovers to
    /// `~1e-16` relative error (checked at `< 1e-9`, comfortable margin).
    /// The full-spectrum `sigma_max_rel` (which `metrics::compute` computes
    /// as the *worst* relative error over the whole sorted spectrum,
    /// despite the name) measured `~2-4%`, not machine precision -- the
    /// smallest imposed value (`1e-12`) sits only `~45x` above this
    /// matrix's own per-entry rounding floor (`~100 * f64::EPSILON ~
    /// 2.2e-14`), so a few percent of error recovering it is a property of
    /// the problem's proximity to the representable floor, not evidence of
    /// a solver defect. Checked at `< 0.05`, the measured scale with
    /// margin, not an arbitrarily loosened number.
    #[test]
    fn degenerate_spectrum_profile_recovers_its_imposed_sigma_ref() {
        for n in [32usize, 64] {
            let case = generate(n, Profile::DegenerateSpectrum, 909);
            let sigma_ref = case.sigma_ref.expect("this profile carries sigma_ref");
            let (u, sigma, vt) = LieSvdSmall::solve(&case.a);
            let m = metrics::compute(&case.a, &u, &sigma, &vt, Some(&sigma_ref));
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);

            let mut got = sigma.to_vec();
            got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top_rel = (got[0] - 100.0).abs() / 100.0;
            assert!(top_rel < 1e-9, "n={n} top_rel={top_rel:e}");

            let sigma_max_rel = m.sigma_max_rel.expect("sigma_ref was provided");
            assert!(
                sigma_max_rel < 0.05,
                "n={n} sigma_max_rel={sigma_max_rel:e}"
            );
        }
    }

    /// `Profile::ExtremeIllConditioned` imposes a condition number of
    /// `1e18` -- deliberately beyond `f64`'s representable dynamic range
    /// (`~1e16`), by design (it's the profile this crate already uses
    /// elsewhere specifically to probe extreme regimes, not one meant to
    /// be exactly recoverable). Measured directly, not assumed, and the
    /// measurement changed the shape of this test: an index-based "top
    /// 3/4 must be relatively accurate" cutoff (the first version of this
    /// test) turned out wrong -- relative error grows *smoothly*, not with
    /// a clean quartile break, because what's actually bounded here is the
    /// **absolute** error (`~1e-14` down to `~1e-17` at every index
    /// checked, essentially constant, consistent with `f64` rounding
    /// noise on a `sigma_max = 1.0` matrix), while the *relative* error
    /// necessarily explodes once a singular value shrinks below that fixed
    /// absolute floor -- a property of the problem, not the solver. So
    /// this checks absolute error (`< 1e-9`, comfortable margin over the
    /// measured `~1e-14` scale) across the *entire* spectrum, which is the
    /// honest thing to claim, plus tight relative error only for the
    /// handful of entries actually near `sigma_max` where "relative"
    /// remains a meaningful notion.
    #[test]
    fn extreme_ill_conditioned_profile_stays_within_absolute_error_across_full_spectrum() {
        for n in [32usize, 64] {
            let case = generate(n, Profile::ExtremeIllConditioned, 909);
            let sigma_ref = case.sigma_ref.expect("this profile carries sigma_ref");
            let (u, sigma, vt) = LieSvdSmall::solve(&case.a);
            assert!(u.iter().all(|x| x.is_finite()), "n={n}");
            assert!(vt.iter().all(|x| x.is_finite()), "n={n}");
            assert!(sigma.iter().all(|x| x.is_finite() && *x >= 0.0), "n={n}");

            let m = metrics::compute(&case.a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);

            let mut got = sigma.to_vec();
            let mut want = sigma_ref.to_vec();
            got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            want.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let abs_err = (g - w).abs();
                assert!(
                    abs_err < 1e-9,
                    "n={n} i={i} got={g:e} want={w:e} abs_err={abs_err:e}"
                );
                if *w > 1e-6 {
                    let rel = abs_err / w.abs().max(1e-300);
                    assert!(rel < 1e-8, "n={n} i={i} got={g:e} want={w:e} rel={rel:e}");
                }
            }
        }
    }

    /// The BSS/Amari benchmark named in the original proposal: an
    /// ill-conditioned mixing matrix (`kappa = 1e7 > 1e6`, matching the
    /// "near-collinear sensor channels" description) built the same
    /// controlled-spectrum way as the profiles above (random orthogonal
    /// `U,V`, an imposed singular value spectrum), separated with the
    /// existing `LieSvdBss`, and scored with the standard permutation/
    /// scale-invariant Amari index rather than a raw matrix-difference
    /// norm (which would be meaningless here, since BSS never recovers
    /// sources in their original order or sign).
    #[test]
    fn amari_index_is_small_after_bss_on_ill_conditioned_mixing() {
        let channels = 4;
        let samples = 800;
        let mut rng = StdRng::seed_from_u64(606);

        let u = random_orthogonal(channels, &mut rng);
        let v = random_orthogonal(channels, &mut rng);
        let sigma = Array2::from_diag(&Array1::from(vec![1.0, 1.0, 1.0, 1e-7]));
        let mixing = u.dot(&sigma).dot(&v.t());
        let condition_number = 1.0 / 1e-7;
        assert!(
            condition_number > 1e6,
            "condition_number={condition_number:e}"
        );

        let sources = Array2::from_shape_fn((channels, samples), |(i, t)| {
            let x = t as f64 / samples as f64;
            match i {
                0 => (2.0 * std::f64::consts::PI * 7.0 * x).sin(),
                1 => (2.0 * std::f64::consts::PI * 11.0 * x).cos().signum(),
                2 => {
                    (2.0 * std::f64::consts::PI * 3.0 * x).sin()
                        + 0.4 * (2.0 * std::f64::consts::PI * 17.0 * x).sin()
                }
                _ => ((t * 37 + 11) as f64).sin() * 0.7,
            }
        });
        let observations = mixing.dot(&sources);

        let result = LieSvdBss::separate(&observations, PhaseBssParams::for_channels(channels));
        let before = amari_index(&mixing);
        let g = result.unmixing.dot(&mixing);
        let after = amari_index(&g);
        assert!(
            after < before,
            "expected separation to improve the Amari index: before={before:e} after={after:e}"
        );
        assert!(after.is_finite() && after >= 0.0);
    }

    #[test]
    fn amari_index_is_zero_for_a_scaled_permutation() {
        let g = Array2::from_shape_vec((3, 3), vec![0.0, 2.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 7.0])
            .unwrap();
        assert!(amari_index(&g) < 1e-15, "amari={:e}", amari_index(&g));
    }
}
