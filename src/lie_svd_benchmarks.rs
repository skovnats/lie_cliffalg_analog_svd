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
//! 3. **Self-consistency** (`kahan_matrix`, `hilbert_matrix`, `frank_matrix`,
//!    `forsythe_matrix`, `cauchy_matrix`, `vandermonde_matrix`,
//!    `ginibre_matrix`): no external ground truth is available, or -- for
//!    `hilbert_matrix`/`cauchy_matrix`/`vandermonde_matrix` at the sizes
//!    where their condition number exceeds `f64`'s representable range --
//!    even *exists* in double precision. Orthogonality of the recovered
//!    bases and reconstruction accuracy are what's actually checkable, and
//!    claiming machine-precision recovery of singular values that are
//!    smaller than the matrix's own rounding error would be a false claim
//!    regardless of which solver computed them. `frank_matrix`'s own famous
//!    property (reciprocal eigenvalue pairs) is a *nonsymmetric eigenvalue*
//!    fact, not a singular-value one, so it doesn't give this crate's SVD
//!    solvers a closed form to check against either -- self-consistency is
//!    what's actually available here too. `forsythe_matrix` is a
//!    Jordan-block-plus-corner-perturbation construction, deliberately
//!    close to defective/non-diagonalizable; see
//!    `forsythe_matrix_stays_orthogonal_and_reconstructs` for the measured
//!    (not assumed) accuracy this crate's solver actually reaches on it.
//!    `ginibre_matrix` is the random-matrix-theory "heavy non-normal
//!    matrix" case (plain i.i.d. Gaussian entries, `A A^T != A^T A` in
//!    general).
//! 4. **Known asymptotic clustering/edge laws** (`parter_matrix`,
//!    `marchenko_pastur_matrix`): not closed forms for individual singular
//!    values, but real, citable facts from the literature. `parter_matrix`
//!    (Parter's own result, also a standard example in Trefethen & Bau's
//!    *Numerical Linear Algebra*): almost all singular values cluster near
//!    `pi` as `n` grows. `marchenko_pastur_matrix` (Marchenko & Pastur,
//!    1967): as the aspect ratio approaches `1`, singular values of an
//!    i.i.d. rectangular Gaussian matrix concentrate within a known
//!    leading-order support interval. Both checked directly, with
//!    thresholds calibrated against measurement rather than assumed.
//!
//! ## Robustness properties, not accuracy claims
//!
//! `orthogonality_drift_stays_small_after_ten_million_rotations` and
//! `complex_unitarity_drift_stays_small_after_ten_million_rotations` check
//! a different kind of thing than the matrices above: not "does the solver
//! recover a known answer on one hard input", but "does composing a very
//! long sequence of individually-valid Givens/unitary rotor updates --
//! exactly the operation this crate's architecture is built from --
//! accumulate meaningful numerical drift". Measured, not assumed:
//! `~1.4e-11` real / `~3.7e-12` complex after `1e7` rotations, both on an
//! `8x8` basis. `extreme_dynamic_range_matrix_stays_finite_and_accurate`
//! and `subnormal_scale_matrix_stays_finite_and_accurate` check a further
//! different thing again -- not accuracy on a hard *spectrum*, but
//! numerical survival at extreme *entry magnitude* (matrix entries spanning
//! `~1e-150` to `~1e150`; entries in `f64`'s subnormal range,
//! `~1e-310`) -- specifically because `lie_svd_small::newton_schulz_polar`
//! scales its input by `1 / frobenius_norm(a).max(1e-300)`, so a matrix
//! whose true Frobenius norm underflows to exactly `0.0` (subnormal entries
//! squared underflow well before the norm itself would) is a concrete,
//! identifiable risk this construction is meant to catch, not a
//! hypothetical one.
//!
//! ## What was in scope, considered, and explicitly left out
//!
//! - **SuiteSparse / Matrix Market:** real matrices, but downloading them
//!   would require network access, breaking this project's established
//!   offline-reproducible pattern (`docker build --no-cache` with no
//!   network calls). Not used.
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
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

/// The Frank matrix (Higham; MATLAB `gallery('frank', n)`): upper
/// Hessenberg, `det = 1`, `F[i,j] = n-j` for `j >= i`, `F[i,i-1] = n-i` on
/// the subdiagonal, `0` below it (`0`-indexed). Famous for eigenvalues that
/// occur in reciprocal pairs (`lambda`, `1/lambda`), some pairs extremely
/// ill-conditioned despite the matrix's modest integer entries -- a classic
/// hard *eigenvalue* test. That property doesn't transfer to a closed form
/// for the *singular* values this crate's solvers compute, so, like
/// `kahan_matrix`, only self-consistency is checked here.
pub fn frank_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        if j >= i {
            (n - j) as f64
        } else if j + 1 == i {
            (n - i) as f64
        } else {
            0.0
        }
    })
}

/// The Forsythe matrix (Higham; MATLAB `gallery('forsythe', n, alpha,
/// lambda)`): an `n x n` Jordan block with `lambda` on the diagonal and `1`
/// on the superdiagonal, perturbed by a single small entry `alpha` in the
/// bottom-left corner `(n-1, 0)` (`0`-indexed). Deliberately close to
/// defective/non-diagonalizable (a bare Jordan block has one eigenvalue
/// with a single eigenvector); the corner perturbation gives it `n`
/// distinct eigenvalues (the `n`-th roots of `alpha`, spread around a
/// circle) that are nonetheless extremely sensitive to further
/// perturbation -- a classic non-normality stress test.
pub fn forsythe_matrix(n: usize, lambda: f64, alpha: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            lambda
        } else if j == i + 1 {
            1.0
        } else if n > 0 && i == n - 1 && j == 0 {
            alpha
        } else {
            0.0
        }
    })
}

/// The Parter matrix (Higham; MATLAB `gallery('parter', n)`):
/// `P[i,j] = 1 / (i - j + 0.5)` (`0`-indexed, matching the `1`-indexed
/// `1/(i-j+0.5)` convention since the `+0.5` offset is index-shift
/// invariant). A real, citable fact from the literature (Parter's own
/// result; see also Trefethen & Bau, *Numerical Linear Algebra*, on
/// Toeplitz-matrix singular value clustering): almost all of its singular
/// values cluster tightly near `pi` as `n` grows, unlike a generic
/// ill-conditioned matrix's spread-out spectrum.
pub fn parter_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i as f64 - j as f64 + 0.5))
}

/// The Cauchy matrix (Higham; MATLAB `gallery('cauchy', n)` default form):
/// `C[i,j] = 1 / (x_i + y_j)` with `x = y = 1..n`, i.e.
/// `C[i,j] = 1 / (i + j + 2)` (`0`-indexed). Symmetric positive definite
/// and, like the Hilbert matrix, extremely ill-conditioned by construction
/// -- a second, independently-constructed example of the same
/// "graceful degradation past `f64`'s representable range" question
/// `hilbert_matrix` answers, with a different (non-reciprocal-integer)
/// entry structure.
pub fn cauchy_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i as f64 + j as f64 + 2.0))
}

/// The Vandermonde matrix on equally spaced nodes `x_i = i+1` (`1..=n`),
/// `V[i,j] = x_i^j` (`0`-indexed, so column `0` is all-ones, column `n-1`
/// is `x_i^{n-1}`). A classical fact (Gautschi's work on Vandermonde
/// conditioning): equally spaced nodes are close to the worst case for
/// Vandermonde conditioning, with `kappa` growing exponentially in `n` --
/// measured here at `kappa ~9.5e8` (`n=8`) up to `~5.6e15` (`n=12`, already
/// at the edge of `f64`'s representable range). No comparison to Hilbert's
/// own growth rate is claimed -- the exact base of that exponential isn't
/// confidently known here, so only the measured numbers for *this* matrix
/// are reported. No closed form for its singular values in this crate
/// either, so self-consistency is what's checked, same as
/// `kahan_matrix`/`frank_matrix`.
pub fn vandermonde_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| ((i + 1) as f64).powi(j as i32))
}

/// The Ginibre ensemble: an `n x n` matrix of i.i.d. standard-normal
/// entries, real and (deliberately, unlike a symmetric random matrix)
/// non-normal in general (`A A^T != A^T A`). Named in random-matrix-theory
/// benchmarks specifically as a "heavy non-normal matrix" stress case; no
/// closed-form singular values, self-consistency is what's checked.
pub fn ginibre_matrix(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, n), |_| rng.sample::<f64, _>(rand_distr::StandardNormal))
}

/// A rectangular i.i.d. standard-normal matrix (`rows x cols`) for
/// Marchenko-Pastur edge testing: as `cols/rows -> 1`, singular values of
/// such a matrix are known (Marchenko & Pastur, 1967) to concentrate, with
/// high probability, within `[sqrt(rows) - sqrt(cols), sqrt(rows) +
/// sqrt(cols)]` -- the leading-order MP support edge (not the finer
/// Tracy-Widom edge fluctuation law, which this doesn't attempt to check).
pub fn marchenko_pastur_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| {
        rng.sample::<f64, _>(rand_distr::StandardNormal)
    })
}

/// Discretizes a 1-D Fredholm integral equation of the first kind,
/// `b(s) = integral_a^b K(s,t) x(t) dt`, by midpoint quadrature on `n`
/// equally spaced nodes over `domain = (a, b)`, from an explicit smooth
/// kernel and a known smooth "true" solution. Returns `(A, x_true, b)`
/// with `b = A.dot(x_true)` **exactly**, by construction: the right-hand
/// side is generated forward from a known answer rather than supplied
/// independently, so it trivially satisfies the discrete Picard condition
/// (an "inverse crime" in the regularization literature) -- appropriate
/// here, where the question is whether this crate's SVD-based spectral
/// truncation can recover a *known* answer, not whether it solves a
/// real-world inverse problem with unknown ground truth.
fn fredholm_first_kind(
    n: usize,
    domain: (f64, f64),
    kernel: impl Fn(f64, f64) -> f64,
    solution: impl Fn(f64) -> f64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let (a, b) = domain;
    let h = (b - a) / n as f64;
    let nodes: Vec<f64> = (0..n).map(|i| a + (i as f64 + 0.5) * h).collect();
    let mat = Array2::from_shape_fn((n, n), |(i, j)| kernel(nodes[i], nodes[j]) * h);
    let x_true = Array1::from_shape_fn(n, |j| solution(nodes[j]));
    let b_vec = mat.dot(&x_true);
    (mat, x_true, b_vec)
}

/// A backward heat conduction problem, in the spirit of Hansen's `heat`
/// test problem (P.C. Hansen, *Regularization Tools*) but built directly
/// from the textbook 1-D heat kernel (the Gaussian fundamental solution of
/// the heat equation, `K(x,y,t) = exp(-(x-y)^2/(4t)) / sqrt(4*pi*t)`) and a
/// forward-generated right-hand side, rather than reproducing Hansen's own
/// exact discretization/RHS formula (not independently verified here, so
/// not claimed). The forward map is smoothing (a classic diffusion
/// operator), so its inverse -- recovering the initial condition `x_true`
/// from the diffused state `b` -- is severely ill-posed: singular values
/// decay rapidly (the hallmark of this whole problem class), and only a
/// spectral-truncation approach like `TblRotorRegressor`'s
/// `singular_value_floor` has any hope of a stable answer.
pub fn heat_problem(n: usize, diffusion_time: f64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    fredholm_first_kind(
        n,
        (0.0, 1.0),
        move |s, t| {
            (-(s - t).powi(2) / (4.0 * diffusion_time)).exp()
                / (4.0 * std::f64::consts::PI * diffusion_time).sqrt()
        },
        |t| (-50.0 * (t - 0.3).powi(2)).exp() + 0.6 * (-80.0 * (t - 0.7).powi(2)).exp(),
    )
}

/// A gravity-survey-style deconvolution problem, in the spirit of Hansen's
/// `phillips` test problem (D.L. Phillips, 1962, "A technique for the
/// numerical solution of certain integral equations of the first kind"),
/// built directly from Phillips's own well-known closed-form kernel
/// `phi(x) = 1 + cos(pi*x/3)` for `|x| < 3` (else `0`) -- a compactly
/// supported, `C^1`-continuous bump, the one part of the classical
/// construction confident enough to state exactly. Uses `phi` as *both*
/// the convolution kernel `K(s,t) = phi(s-t)` and the true solution `x(t)
/// = phi(t)`, matching Phillips's own problem, but the right-hand side is
/// forward-generated (`b = A x_true`) rather than reproducing the
/// classical closed-form `b(s)` (not independently verified here).
pub fn phillips_problem(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    fn phi(x: f64) -> f64 {
        if x.abs() < 3.0 {
            1.0 + (std::f64::consts::PI * x / 3.0).cos()
        } else {
            0.0
        }
    }
    fredholm_first_kind(n, (-6.0, 6.0), |s, t| phi(s - t), phi)
}

/// A 1-D diffraction/resolution problem, in the spirit of Hansen's `shaw`
/// test problem (C.B. Shaw, 1972), built from the standard
/// `(cos+cos)^2 * sinc^2` diffraction-kernel *shape* over
/// `s, theta in [-pi/2, pi/2]`. Stated with less confidence than
/// `heat_problem`/`phillips_problem`: the exact kernel and discretization
/// in Hansen's own `shaw.m` were not independently re-derived or verified
/// here, so this is a structurally faithful but not bit-exact
/// reproduction, and the right-hand side is forward-generated from a known
/// two-bump solution rather than any closed form.
pub fn shaw_problem(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    fn sinc(x: f64) -> f64 {
        if x.abs() < 1e-12 {
            1.0
        } else {
            x.sin() / x
        }
    }
    let half_pi = std::f64::consts::FRAC_PI_2;
    fredholm_first_kind(
        n,
        (-half_pi, half_pi),
        |s, t| {
            let u = std::f64::consts::PI * (s.sin() + t.sin());
            (s.cos() + t.cos()).powi(2) * sinc(u).powi(2)
        },
        |t| 2.0 * (-6.0 * (t - 0.4).powi(2)).exp() + (-2.0 * (t + 0.5).powi(2)).exp(),
    )
}

/// Truncated-SVD reconstruction: `x_hat = V diag(1/sigma_i for sigma_i >
/// floor*sigma_max, else 0) U^T b`. The same truncation idea
/// `lie_tbl_regress::TblRegressParams::singular_value_floor` already
/// implements for regression, applied here to a first-kind Fredholm
/// inverse problem instead -- reused by name/concept, not by calling that
/// regression-specific code, since the reconstruction here isn't a fit.
pub fn truncated_svd_solve(
    u: &Array2<f64>,
    sigma: &Array1<f64>,
    vt: &Array2<f64>,
    b: &Array1<f64>,
    floor: f64,
) -> Array1<f64> {
    let n = sigma.len();
    let sigma_max = sigma.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    let cutoff = floor * sigma_max;
    let ub = u.t().dot(b);
    let mut coeff = Array1::<f64>::zeros(n);
    for i in 0..n {
        if sigma[i] > cutoff {
            coeff[i] = ub[i] / sigma[i];
        }
    }
    vt.t().dot(&coeff)
}

/// The 2-site Hubbard model Hamiltonian, restricted to the `N=2`,
/// `S_z=0` sector (one up electron, one down electron) -- the standard,
/// widely-used exactly-solvable "Hubbard dimer" (see e.g. Essler et al.,
/// *The One-Dimensional Hubbard Model*, or Tasaki's Hubbard model lecture
/// notes). Basis, in order: `|1> = both particles on site 1`,
/// `|2> = up on site 1, down on site 2`, `|3> = down on site 1, up on
/// site 2`, `|4> = both particles on site 2`. `t` is the hopping
/// amplitude, `u` the on-site interaction:
///
/// ```text
/// H = [ u  -t  -t   0 ]
///     [-t   0   0  -t ]
///     [-t   0   0  -t ]
///     [ 0  -t  -t   u ]
/// ```
///
/// Derived directly (not merely asserted) in the module's own commit
/// history: since the up- and down-electron positions are independent
/// single-particle two-level systems away from double occupancy, `H`
/// decomposes as `H_up (x) I + I (x) H_down + u * P_doubleocc`, which
/// reproduces exactly this matrix -- see `hubbard_dimer_eigenvalues` for
/// the closed-form spectrum this construction makes available, and
/// `hubbard_dimer_resolves_the_exact_near_degenerate_gap` for the
/// cross-check against `U=0` (two independent hopping problems, spectrum
/// `{-2t, 0, 0, 2t}` by direct sum) that confirms both derivations agree.
pub fn hubbard_dimer_hamiltonian(t: f64, u: f64) -> Array2<f64> {
    Array2::from_shape_vec(
        (4, 4),
        vec![
            u, -t, -t, 0.0, -t, 0.0, 0.0, -t, -t, 0.0, 0.0, -t, 0.0, -t, -t, u,
        ],
    )
    .expect("4x4 shape")
}

/// Exact eigenvalues of `hubbard_dimer_hamiltonian(t, u)`, derived via the
/// symmetric/antisymmetric block decomposition described in that
/// function's doc comment: an exact `0` (the fully antisymmetric
/// `|2>-|3>` combination decouples completely, for any `t,u`), an exact
/// `u` (the `|1>-|4>` combination also decouples), and
/// `u/2 +/- sqrt((u/2)^2 + 4t^2)` from the remaining `2x2` block. At
/// `u=0` this reduces to `{-2t, 0, 0, 2t}`, matching the independent
/// direct-sum argument in the doc comment above -- the two derivations
/// were cross-checked against each other, not merely asserted to agree.
/// Not sorted; callers needing a specific order should sort as needed
/// (tests below sort descending, this crate's usual convention).
pub fn hubbard_dimer_eigenvalues(t: f64, u: f64) -> Array1<f64> {
    let gap = ((u / 2.0).powi(2) + 4.0 * t * t).sqrt();
    Array1::from(vec![0.0, u, u / 2.0 - gap, u / 2.0 + gap])
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

    /// Frank matrix: no closed form for singular values in this crate (its
    /// famous property, reciprocal eigenvalue pairs, is an eigenvalue fact,
    /// not a singular-value one), so self-consistency is what's checked.
    /// Measured at `n=16,32,64`: `orth_u`/`orth_v` up to `~9e-14`,
    /// `rel_recon` up to `~7e-15` -- no sign of difficulty despite the
    /// matrix's famously ill-conditioned eigenvalues.
    #[test]
    fn frank_matrix_stays_orthogonal_and_reconstructs() {
        for n in [16usize, 32, 64] {
            let a = frank_matrix(n);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }
    }

    /// Forsythe matrix (`lambda=0`, `alpha=1e-6`): a Jordan block --
    /// deliberately close to defective/non-diagonalizable -- perturbed by
    /// one small corner entry. Measured, not assumed: `LieSvdSmall::solve`
    /// (polar decomposition plus Jacobi, not an eigenvalue algorithm)
    /// reaches essentially *exact* results here (`orth_u`/`orth_v`/
    /// `rel_recon` at or near `0` for `n<=32`, still `~1e-16`/`~5e-23` at
    /// `n=64`) -- this construction's near-defectiveness, which makes
    /// *eigenvalue* algorithms struggle, doesn't trouble an SVD route built
    /// on polar decomposition the same way.
    #[test]
    fn forsythe_matrix_stays_orthogonal_and_reconstructs() {
        for n in [16usize, 32, 64] {
            let a = forsythe_matrix(n, 0.0, 1e-6);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }
    }

    /// Parter matrix: checks the specific, citable literature fact this
    /// matrix is known for -- almost all singular values cluster near `pi`
    /// as `n` grows -- rather than only generic self-consistency. Measured
    /// fraction within `0.05` of `pi`: `13/16` (`81.25%`), `29/32`
    /// (`90.6%`), `61/64` (`95.3%`) -- increasing with `n`, exactly the
    /// asymptotic clustering the literature describes. Threshold set at
    /// `75%`, safely under the worst (smallest-`n`) measured fraction.
    #[test]
    fn parter_matrix_singular_values_cluster_near_pi() {
        for n in [16usize, 32, 64] {
            let a = parter_matrix(n);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);

            let near_pi = sigma
                .iter()
                .filter(|&&s| (s - std::f64::consts::PI).abs() < 0.05)
                .count();
            let fraction = near_pi as f64 / n as f64;
            assert!(
                fraction > 0.75,
                "n={n} only {near_pi}/{n} singular values within 0.05 of pi"
            );
        }
    }

    /// Cauchy matrix (`1/(i+j+2)`, `0`-indexed): symmetric PD and, like the
    /// Hilbert matrix, extremely ill-conditioned by construction with a
    /// different entry structure. Same graceful-degradation question as
    /// `hilbert_matrix_degrades_gracefully_past_double_precision_limits`:
    /// measured up to `n=16` (`kappa` growing past `f64`'s representable
    /// range there too, the same underflow-in-the-ratio symptom as
    /// Hilbert), `orth_u`/`orth_v`/`rel_recon` stay at `~1e-14` throughout,
    /// unaffected by the underflowed tail.
    #[test]
    fn cauchy_matrix_degrades_gracefully_past_double_precision_limits() {
        for n in [6usize, 10, 16] {
            let a = cauchy_matrix(n);
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

    fn synthetic_sources(channels: usize, samples: usize) -> Array2<f64> {
        Array2::from_shape_fn((channels, samples), |(i, t)| {
            let x = t as f64 / samples as f64;
            let freq = 3.0 + i as f64 * 4.0;
            match i % 4 {
                0 => (2.0 * std::f64::consts::PI * freq * x).sin(),
                1 => (2.0 * std::f64::consts::PI * freq * x).cos().signum(),
                2 => {
                    (2.0 * std::f64::consts::PI * freq * x).sin()
                        + 0.4 * (2.0 * std::f64::consts::PI * (freq * 2.3) * x).sin()
                }
                _ => ((t * 37 + 11 * (i + 1)) as f64).sin() * 0.7,
            }
        })
    }

    /// The Amari index across a small parametric grid (`channels in
    /// {4,8}`, `kappa in {1e3,1e5,1e7}`), rather than the single `kappa=1e7`
    /// point checked elsewhere -- does BSS's improvement hold up as
    /// conditioning gets worse and as the channel count grows, or was the
    /// single-point result a favorable roll of one seed? Measured, not
    /// assumed (each cell is its own independent seed, not cherry-picked):
    ///
    /// | channels | kappa | before | after |
    /// | -------: | ----: | -----: | ----: |
    /// | 4 | 1e3 | 3.88e-1 | 6.46e-4 |
    /// | 4 | 1e5 | 4.94e-1 | 2.03e-1 |
    /// | 4 | 1e7 | 5.21e-1 | 5.08e-2 |
    /// | 8 | 1e3 | 3.31e-1 | 5.40e-2 |
    /// | 8 | 1e5 | 3.33e-1 | 1.14e-1 |
    /// | 8 | 1e7 | 3.39e-1 | 1.50e-1 |
    ///
    /// Separation improves the Amari index at all `6` grid points -- but
    /// not monotonically in `kappa` (`channels=4`'s `after` at `kappa=1e5`
    /// is *worse* than at `kappa=1e7`), reported as measured rather than
    /// smoothed into a cleaner-sounding trend that isn't actually there;
    /// with only one random seed per cell, a non-monotonic result here is
    /// expected sampling variation, not a claim that higher `kappa` reduces
    /// separation quality now and forever.
    #[test]
    fn amari_index_improves_across_a_channel_by_condition_number_grid() {
        let samples = 800;
        for channels in [4usize, 8] {
            for &kappa in &[1e3_f64, 1e5, 1e7] {
                let mut rng =
                    StdRng::seed_from_u64(606 + channels as u64 * 100 + kappa.log10() as u64);
                let u = random_orthogonal(channels, &mut rng);
                let v = random_orthogonal(channels, &mut rng);
                let mut sigma = Array2::<f64>::zeros((channels, channels));
                for i in 0..(channels - 1) {
                    sigma[[i, i]] = 1.0;
                }
                sigma[[channels - 1, channels - 1]] = 1.0 / kappa;
                let mixing = u.dot(&sigma).dot(&v.t());

                let sources = synthetic_sources(channels, samples);
                let observations = mixing.dot(&sources);
                let result =
                    LieSvdBss::separate(&observations, PhaseBssParams::for_channels(channels));
                let before = amari_index(&mixing);
                let g = result.unmixing.dot(&mixing);
                let after = amari_index(&g);
                assert!(
                    after < before,
                    "channels={channels} kappa={kappa:e} before={before:e} after={after:e}"
                );
                assert!(after.is_finite() && after >= 0.0);
            }
        }
    }

    /// Vandermonde matrix on equally spaced nodes: no closed form for
    /// singular values here, so self-consistency is checked, same as
    /// `kahan_matrix`. Measured condition numbers (`~9.5e8` at `n=8`,
    /// `~2.1e12` at `n=10`, `~5.6e15` at `n=12`, already at the edge of
    /// `f64`'s representable range) confirm the exponential ill-
    /// conditioning this matrix is known for, without claiming any
    /// particular rate relative to other matrices in this module.
    #[test]
    fn vandermonde_matrix_stays_orthogonal_and_reconstructs() {
        for n in [8usize, 10, 12] {
            let a = vandermonde_matrix(n);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }
    }

    /// Ginibre ensemble: plain i.i.d. Gaussian, non-normal in general --
    /// the RMT "heavy non-normal matrix" case. No closed form, so
    /// self-consistency again; measured `orth_u`/`orth_v` up to `~2.8e-14`,
    /// `rel_recon` up to `~4.7e-15` at `n=16,32`.
    #[test]
    fn ginibre_matrix_stays_orthogonal_and_reconstructs() {
        for n in [16usize, 32] {
            let a = ginibre_matrix(n, 42);
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "n={n} orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "n={n} orth_v={:e}", m.orth_v);
            assert!(m.rel_recon < 1e-9, "n={n} rel_recon={:e}", m.rel_recon);
        }
    }

    /// Marchenko-Pastur edge: as `cols/rows -> 1`, singular values of an
    /// i.i.d. Gaussian `rows x cols` matrix concentrate near
    /// `sqrt(rows) +/- sqrt(cols)`. Measured across three aspect ratios,
    /// one at exactly `cols=rows`: the *upper* edge tracks the prediction
    /// well (`~2.5-4%` off, single seed); the *lower* edge is much noisier
    /// (finite-size fluctuations at the lower MP edge are a known, larger
    /// effect than at the upper edge -- not a solver artifact), so only the
    /// upper edge gets a tight quantitative check here, and the lower edge
    /// only a loose sanity bound (it must stay comfortably below the upper
    /// edge, not track the asymptotic prediction closely at this `n`).
    #[test]
    fn marchenko_pastur_upper_edge_matches_prediction() {
        for (rows, cols, seed) in [(64usize, 60usize, 7u64), (64, 64, 7), (100, 95, 7)] {
            let a = marchenko_pastur_matrix(rows, cols, seed);
            let (u, sigma, vt) = LieSvdSmall::solve_rectangular(&a);
            // `u` has orthonormal columns (`u^T u = I_k`), `vt` is square
            // (`k x k`) and orthogonal -- not the `n x n`/`n x n` shape
            // `metrics::compute` assumes, so checked directly here rather
            // than through that helper.
            let k = sigma.len();
            let ident_k = Array2::<f64>::eye(k);
            let orth_u = (&u.t().dot(&u) - &ident_k).mapv(|x| x * x).sum().sqrt();
            let orth_v = (&vt.dot(&vt.t()) - &ident_k).mapv(|x| x * x).sum().sqrt();
            let sigma_mat = Array2::from_diag(&sigma);
            let recon = u.dot(&sigma_mat).dot(&vt);
            let a_norm = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
            let rel_recon = (&recon - &a).mapv(|x| x * x).sum().sqrt() / a_norm;
            assert!(orth_u < 1e-9, "rows={rows} cols={cols} orth_u={orth_u:e}");
            assert!(orth_v < 1e-9, "rows={rows} cols={cols} orth_v={orth_v:e}");
            assert!(
                rel_recon < 1e-9,
                "rows={rows} cols={cols} rel_recon={rel_recon:e}"
            );

            let expected_max = (rows as f64).sqrt() + (cols as f64).sqrt();
            let max_sigma = sigma.iter().cloned().fold(0.0_f64, f64::max);
            let min_sigma = sigma.iter().cloned().fold(f64::INFINITY, f64::min);
            let rel_err = (max_sigma - expected_max).abs() / expected_max;
            assert!(
                rel_err < 0.10,
                "rows={rows} cols={cols} max_sigma={max_sigma:e} expected={expected_max:e} rel_err={rel_err:e}"
            );
            assert!(
                min_sigma < max_sigma,
                "rows={rows} cols={cols} min_sigma={min_sigma:e} should stay below max_sigma={max_sigma:e}"
            );
        }
    }

    /// A direct test of the underflow risk named in the module doc
    /// comment: `newton_schulz_polar` scales its input by
    /// `1 / frobenius_norm(a).max(1e-300)`. Entries spanning `~1e-150` to
    /// `~1e150` (condition number `~1e300`, an ambient scale that never
    /// itself risks `f64` overflow/underflow) stay fully finite, with
    /// `orth_u`/`orth_v`/`rel_recon` all landing at `~1e-15` to `~1e-16` --
    /// no measurable degradation despite the extreme spread. Singular
    /// values below roughly `sigma_max * 1e-16` come out as exact `0.0` --
    /// not a solver defect, but a representability limit of the *matrix
    /// itself*: once assembled as a dense `f64` array, contributions ~300
    /// orders of magnitude below the dominant term have no bits left to
    /// occupy.
    #[test]
    fn extreme_dynamic_range_matrix_stays_finite_and_accurate() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 16;
        let u = random_orthogonal(n, &mut rng);
        let v = random_orthogonal(n, &mut rng);
        let mut sigma = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let exp = 150.0 * (1.0 - 2.0 * i as f64 / (n - 1) as f64);
            sigma[[i, i]] = 10f64.powf(exp);
        }
        let a = u.dot(&sigma).dot(&v.t());
        assert!(
            a.iter().all(|x| x.is_finite()),
            "matrix construction itself must not overflow"
        );

        let (uu, ss, vtt) = LieSvdSmall::solve(&a);
        assert!(uu.iter().all(|x| x.is_finite()));
        assert!(vtt.iter().all(|x| x.is_finite()));
        assert!(ss.iter().all(|x| x.is_finite() && *x >= 0.0));
        let m = metrics::compute(&a, &uu, &ss, &vtt, None);
        assert!(m.orth_u < 1e-9, "orth_u={:e}", m.orth_u);
        assert!(m.orth_v < 1e-9, "orth_v={:e}", m.orth_v);
        assert!(m.rel_recon < 1e-9, "rel_recon={:e}", m.rel_recon);
    }

    /// The other half of the underflow risk: entries whose magnitude is
    /// itself in `f64`'s *subnormal* range (below `~2.2e-308`), where
    /// squaring a single entry (as any Frobenius-norm computation does)
    /// underflows to exact `0.0` well before the norm as a whole would.
    /// Measured, not assumed: the solver's `.max(1e-300)` floor on the
    /// polar-decomposition scale factor absorbs this cleanly here --
    /// `orth_u`, `orth_v`, and `rel_recon` all come out as *exact* `0.0`
    /// on this `6x6` subnormal-entry case, and the recovered singular
    /// values stay in the correct subnormal range rather than being
    /// crushed to zero or blown up.
    #[test]
    fn subnormal_scale_matrix_stays_finite_and_accurate() {
        let subnormal_scale = 1e-310_f64;
        assert!(
            subnormal_scale > 0.0 && subnormal_scale < f64::MIN_POSITIVE,
            "sanity check: this must actually be in the subnormal range"
        );
        let a = Array2::from_shape_fn((6, 6), |(i, j)| {
            subnormal_scale * ((i * 3 + j + 1) as f64).sin()
        });
        let (u, sigma, vt) = LieSvdSmall::solve(&a);
        assert!(u.iter().all(|x| x.is_finite()));
        assert!(vt.iter().all(|x| x.is_finite()));
        assert!(sigma.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(
            sigma.iter().all(|&s| s == 0.0 || s < 1e-300),
            "recovered singular values must stay in (or near) the input's own subnormal scale, sigma={sigma:?}"
        );
        let m = metrics::compute(&a, &u, &sigma, &vt, None);
        assert!(m.orth_u < 1e-9, "orth_u={:e}", m.orth_u);
        assert!(m.orth_v < 1e-9, "orth_v={:e}", m.orth_v);
        assert!(m.rel_recon < 1e-9, "rel_recon={:e}", m.rel_recon);
    }

    /// Orthogonality drift under exactly the operation this crate's whole
    /// architecture is built from: a very long sequence of individually
    /// small Givens rotor updates to a shared basis. `1e7` sequential
    /// random-angle rotations on an `8x8` identity, measured (not
    /// bounded-by-construction) drift: `||Q^T Q - I||_F ~1.4e-11`,
    /// `~611ms`.
    #[test]
    fn orthogonality_drift_stays_small_after_ten_million_rotations() {
        let n = 8;
        let mut basis = Array2::<f64>::eye(n);
        let mut rng = StdRng::seed_from_u64(5);
        let k = 10_000_000usize;
        for _ in 0..k {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            while j == i {
                j = rng.gen_range(0..n);
            }
            let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
            crate::lie_svd_joint::apply_basis_rotor(&mut basis, i, j, theta);
        }
        let ident = Array2::<f64>::eye(n);
        let drift = (&basis.t().dot(&basis) - &ident)
            .mapv(|x| x * x)
            .sum()
            .sqrt();
        assert!(drift.is_finite());
        assert!(drift < 1e-6, "drift after {k} rotations = {drift:e}");
    }

    /// The complex-branch analogue of the orthogonality-drift test above:
    /// `1e7` sequential random `U(2)`-parametrized unitary rotor updates
    /// (`c` real `= cos(theta)`, off-diagonal `s = sin(theta) *
    /// e^{i*phi}`, applied as `[[c,-conj(s)],[s,c]]` to a pair of basis
    /// columns -- unitary by construction for any `theta,phi`) to an `8x8`
    /// identity, using `lie_svd_complex::complex_unitarity_error` (already
    /// tested elsewhere in this crate) rather than a new metric. Measured
    /// drift: `~3.7e-12`, `~796ms` -- same order as the real case, no sign
    /// the complex branch accumulates drift faster.
    #[test]
    fn complex_unitarity_drift_stays_small_after_ten_million_rotations() {
        use num_complex::Complex64;

        fn apply_complex_basis_rotor(
            basis: &mut Array2<Complex64>,
            i: usize,
            j: usize,
            theta: f64,
            phi: f64,
        ) {
            let n = basis.nrows();
            let c = theta.cos();
            let s = theta.sin() * Complex64::from_polar(1.0, phi);
            for row in 0..n {
                let bi = basis[[row, i]];
                let bj = basis[[row, j]];
                basis[[row, i]] = bi * c - bj * s.conj();
                basis[[row, j]] = bi * s + bj * c;
            }
        }

        let n = 8;
        let mut basis = Array2::<Complex64>::eye(n);
        let mut rng = StdRng::seed_from_u64(11);
        let k = 10_000_000usize;
        for _ in 0..k {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n);
            while j == i {
                j = rng.gen_range(0..n);
            }
            let theta = rng.gen_range(0.0..std::f64::consts::FRAC_PI_2);
            let phi = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
            apply_complex_basis_rotor(&mut basis, i, j, theta, phi);
        }
        let drift = crate::lie_svd_complex::complex_unitarity_error(&basis);
        assert!(drift.is_finite());
        assert!(drift < 1e-6, "drift after {k} rotations = {drift:e}");
    }

    fn rel_err(x_hat: &Array1<f64>, x_true: &Array1<f64>) -> f64 {
        let norm = x_true.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        (x_hat - x_true).mapv(|v| v * v).sum().sqrt() / norm
    }

    /// The textbook regularization story, demonstrated rather than
    /// asserted: `heat_problem` and `shaw_problem` are severely ill-posed
    /// (measured singular values decay to exact `0.0` well within the
    /// `n=64` spectrum). At a well-chosen truncation floor, spectral
    /// truncation recovers the known solution well; at essentially no
    /// truncation (`floor=0`, dividing by near-zero singular values),
    /// reconstruction *explodes* -- measured `rel_err` (heat) `4.7e-5` at
    /// `floor=1e-12` versus `~78` at `floor=0`; (shaw) `1.0e-4` at
    /// `floor=1e-12` versus `~73` at `floor=0`. This is the actual point of
    /// these classical test problems: naive full-rank inversion of a
    /// smoothing operator is not just inaccurate, it's numerically
    /// catastrophic, and only spectral truncation (the same idea
    /// `lie_tbl_regress::TblRegressParams::singular_value_floor` already
    /// implements for regression) makes the inverse problem usable.
    #[test]
    fn severely_ill_posed_problems_need_spectral_truncation_to_avoid_blowup() {
        let cases = [("heat", heat_problem(64, 0.01)), ("shaw", shaw_problem(64))];
        for (name, (a, x_true, b)) in cases {
            let (u, sigma, vt) = LieSvdSmall::solve(&a);
            let m = metrics::compute(&a, &u, &sigma, &vt, None);
            assert!(m.orth_u < 1e-9, "{name}: orth_u={:e}", m.orth_u);
            assert!(m.orth_v < 1e-9, "{name}: orth_v={:e}", m.orth_v);

            let regularized = truncated_svd_solve(&u, &sigma, &vt, &b, 1e-9);
            let regularized_err = rel_err(&regularized, &x_true);
            assert!(
                regularized_err < 1e-2,
                "{name}: regularized_err={regularized_err:e}"
            );

            let unregularized = truncated_svd_solve(&u, &sigma, &vt, &b, 0.0);
            let unregularized_err = rel_err(&unregularized, &x_true);
            assert!(
                unregularized_err > 1.0,
                "{name}: expected catastrophic blowup without truncation, \
                 unregularized_err={unregularized_err:e}"
            );
            assert!(
                unregularized_err > 10.0 * regularized_err,
                "{name}: expected the unregularized solve to be far worse than \
                 the regularized one (unregularized={unregularized_err:e}, \
                 regularized={regularized_err:e})"
            );
        }
    }

    /// The contrasting case: `phillips_problem` is only moderately
    /// ill-conditioned (measured `kappa ~2.9e5` at `n=64`, no singular
    /// values collapsing to exact zero within the spectrum) -- so, unlike
    /// `heat`/`shaw` above, full-rank inversion here is *not* catastrophic.
    /// Measured: reconstruction is accurate to `~4.8e-11` even with *no*
    /// spectral truncation at all (`floor=0`), essentially unchanged from
    /// heavier truncation. The contrast matters: not every hard-looking
    /// inverse problem needs regularization, and this crate's SVD
    /// correctly distinguishes the two regimes rather than needing
    /// truncation applied uniformly out of caution.
    #[test]
    fn moderately_conditioned_inverse_problem_needs_no_truncation() {
        let (a, x_true, b) = phillips_problem(64);
        let (u, sigma, vt) = LieSvdSmall::solve(&a);
        let m = metrics::compute(&a, &u, &sigma, &vt, None);
        assert!(m.orth_u < 1e-9, "orth_u={:e}", m.orth_u);
        assert!(m.orth_v < 1e-9, "orth_v={:e}", m.orth_v);

        let unregularized = truncated_svd_solve(&u, &sigma, &vt, &b, 0.0);
        let err = rel_err(&unregularized, &x_true);
        assert!(err < 1e-6, "err={err:e}");
    }

    /// The closed form was derived, not merely asserted -- see
    /// `hubbard_dimer_hamiltonian`'s and `hubbard_dimer_eigenvalues`'s doc
    /// comments for the derivation and its independent cross-check at
    /// `u=0`. This test verifies it numerically against
    /// `lie_svd_small::eigh_jacobi_full` (this crate's own eigensolver, a
    /// different code path from `hubbard_dimer_eigenvalues`'s closed-form
    /// arithmetic) across several `(t, u)` values, including negative `u`.
    /// Measured agreement: differences at or near machine precision
    /// (`<1e-14`) for every eigenvalue at every `(t,u)` tested here.
    #[test]
    fn hubbard_dimer_matches_its_exact_closed_form_spectrum() {
        for (t, u) in [(1.0_f64, 4.0_f64), (1.0, 0.0), (2.3, -3.1), (0.5, 10.0)] {
            let h = hubbard_dimer_hamiltonian(t, u);
            let (_v, got_eig) = crate::lie_svd_small::eigh_jacobi_full(&h);
            let mut got = got_eig.to_vec();
            let mut want = hubbard_dimer_eigenvalues(t, u).to_vec();
            got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            want.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for (g, w) in got.iter().zip(want.iter()) {
                let diff = (g - w).abs();
                assert!(
                    diff < 1e-9,
                    "t={t} u={u}: got={g:e} want={w:e} diff={diff:e}"
                );
            }
        }
    }

    /// The point of this benchmark: at a tiny but nonzero `u`, the Hubbard
    /// dimer has two eigenvalues that are close (`0` exactly, and `u`) but
    /// genuinely distinct -- a real, physically motivated near-degenerate
    /// gap, the same kind of case the `pei_matrix` degenerate-spectrum test
    /// checks from a different angle. Measured at `u=1e-12`: the solver
    /// resolves both as distinct (not collapsed to a single value), with
    /// the tiny eigenvalue recovered to `~1.0000831e-12` against the exact
    /// `1e-12` -- a relative error of `~8e-5`, not full double-precision
    /// relative accuracy at this extreme gap scale (the absolute precision
    /// floor set by the matrix's other, order-`1` entries is `~1e-16`,
    /// which is already `~1e-4` relative to a `1e-12`-scale eigenvalue --
    /// consistent with the measured error, not a solver defect). Reported
    /// as the honest, measured bound, not tightened to look better.
    #[test]
    fn hubbard_dimer_resolves_the_exact_near_degenerate_gap() {
        let t = 0.7;
        let u = 1e-12;
        let h = hubbard_dimer_hamiltonian(t, u);
        let (_v, eig) = crate::lie_svd_small::eigh_jacobi_full(&h);
        let mut got = eig.to_vec();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // The two smallest-magnitude eigenvalues should be `0` and `u`,
        // genuinely distinct rather than collapsed together.
        let near_zero_pair: Vec<f64> = got.iter().cloned().filter(|&e| e.abs() < 1e-6).collect();
        assert_eq!(
            near_zero_pair.len(),
            2,
            "expected exactly two near-zero eigenvalues, got {near_zero_pair:?}"
        );
        let lo = near_zero_pair[0].min(near_zero_pair[1]);
        let hi = near_zero_pair[0].max(near_zero_pair[1]);
        assert!(
            lo.abs() < 1e-9,
            "expected the smaller one near exact 0, got {lo:e}"
        );
        let rel_err_on_gap = (hi - u).abs() / u;
        assert!(
            rel_err_on_gap < 1e-2,
            "expected the u-scale eigenvalue resolved to within 1% relative, got hi={hi:e} rel_err={rel_err_on_gap:e}"
        );
        assert!(
            hi > 10.0 * lo.abs().max(1e-300),
            "the two near-zero eigenvalues should be clearly distinguishable, lo={lo:e} hi={hi:e}"
        );
    }
}
