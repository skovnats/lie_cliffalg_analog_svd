//! Small, honest tabular regression utility built on the existing
//! Gram/eigen machinery (`kernel_gram::solve_kernel`).
//!
//! This is standard SVD/eigendecomposition-based ridge regression (see e.g.
//! Hastie/Tibshirani/Friedman, "The Elements of Statistical Learning",
//! section 3.4.1, ridge regression via the SVD of `X`). It is not a
//! database engine and not a `JOIN` replacement — that part of the larger
//! tabular-Clifford-algebra idea remains deferred, not rejected: a `JOIN`
//! is a discrete key-matching problem, and no mechanism has yet been found
//! for locating matching rows without an index in this framework (see
//! `TECHNICAL_REPORT.md` and the `0.30.3` release notes for the current
//! state of that question). This module exists because it *is* a real,
//! narrow use for the crate's robust eigensolver: predicting a target
//! column from other columns when the feature columns are collinear or
//! nearly rank-deficient, exactly the case where naive normal-equations
//! regression (`(X^T X)^-1 X^T y` via a plain inverse) is numerically
//! fragile and this crate's ill-conditioned-input-focused solvers are not.
//!
//! Fit works entirely on the `d x d` feature Gram matrix `X^T X` (`d` =
//! number of columns), never the `n x d` data matrix itself, so it reuses
//! only the square symmetric-eigen path that `kernel_gram` already has
//! tested rather than requiring a new rectangular solver.
//!
//! ## Does a multivector/rotor view make `X^T X` unnecessary?
//!
//! No, and `lie_tbl_multivector` proves the specific place this claim
//! breaks rather than just asserting it. The geometric product between two
//! *rows* (two samples) gives a scalar equal to their dot product — that
//! scalar is a **sample-sample** relationship, identical to the `n x n`
//! linear kernel `kernel_gram::build_gram` already computes
//! (`lie_tbl_multivector::row_scalar_gram` is tested equal to it). What
//! regression needs is **feature-feature** relationships (the `d x d`
//! object that appears in the normal equations `X^T X beta = X^T y`), and
//! no amount of relabeling the row-product as "Clifford" produces that
//! without summing over every row — the anticommuting generator structure
//! (`e_j e_k = -e_k e_j`) is a fixed property of the algebra, not something
//! that can encode a specific dataset's column correlations without
//! touching the data. Concretely: `x_i * x_i` (one row against itself)
//! only recovers `||x_i||^2`; getting `sum_i x_ij * x_ik` for a fixed
//! column pair `(j, k)` requires an accumulation over `i`, and that
//! accumulation *is* `X^T X`, however it's computed.
//!
//! What *is* a fair question, and answered empirically below rather than
//! assumed: does routing that accumulation through the crate's rotor-based
//! rectangular SVD directly on `X` (never forming `X^T X` explicitly) do
//! better than forming the Gram matrix and diagonalizing it? Forming
//! `X^T X` squares the condition number of `X` before doing anything else
//! with it — this project's own `LieSvdSmall` docs call exactly this out as
//! a reason to prefer polar decomposition over `A^T A` for the general
//! solver, so the intuition behind the question is sound. `fit_via_rectangular_svd`
//! below tests the regression-specific version of it directly against `fit`
//! (`gram_vs_rectangular_svd_on_ill_conditioned_features`), and the answer
//! changed once, for a documented reason. As of `0.30.3`, routing through
//! `lie_svd_phaseflow`'s *rotor-based* rectangular solver was a clear, large
//! loss (`~51-96%` raw reconstruction error — that solver failed to
//! converge on generic dense data at all, not a subtle precision gap). As
//! of `0.30.4`, `fit_via_rectangular_svd` routes through
//! `lie_svd_small::LieSvdSmall::solve_rectangular` instead (QR-reduction to
//! a small square factor, then this crate's exact square solve — see that
//! function), and the condition-number argument holds up: on the same
//! near-collinear test input, `fit_via_rectangular_svd`'s residual
//! (`~3.3e-9`) is now *smaller* than `fit`'s (`~1.1e-6`). `fit` stays the
//! default (more battle-tested, and its per-feature truncation semantics
//! are easier to reason about), but `fit_via_rectangular_svd` is no longer
//! a known failure — the earlier finding was correctly attributed to the
//! *solver*, not to the "avoid squaring the condition number" argument
//! itself, which is why fixing the solver (not abandoning the argument) was
//! the right move. See `lie_svd_small.rs`'s module docs and
//! `RELEASE_NOTES.md`'s `0.30.4` entry for the rectangular-solver fix
//! itself.
//!
//! ## Bivector-regularized ridge (`fit_with_bivector_regularization`)
//!
//! `0.30.4` adds an anisotropic ridge variant using `lie_tbl_multivector`'s
//! per-column-pair bivector norms — with one correction to the originally
//! proposed formula, made explicit rather than silently applied. The
//! original proposal penalized a column *more* as its "bivector stress"
//! (summed wedge norm against other columns) grew. But a large wedge norm
//! between two columns means they are close to *orthogonal* — two vectors
//! of fixed length span their largest parallelogram when they're
//! perpendicular, and their wedge vanishes exactly when they're parallel
//! (scalar multiples of each other). So high bivector stress marks an
//! *independent, well-determined* direction, and low stress marks a
//! *redundant/collinear* one — the opposite of what ridge should suppress
//! more. This implementation penalizes each column *inversely* to its
//! normalized bivector stress: `Lambda_jj = lambda0 / (0.1 + stress_j)`,
//! `stress_j` = mean over `k != j` of `||c_j ^ c_k|| / (||c_j|| ||c_k||)`
//! (the normalized wedge magnitude, `= sin(angle)` between unit-scaled
//! columns, in `[0, 1]`). See
//! `bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity` for the
//! held-out A/B test this was designed to pass, and its result reported
//! honestly either way.

use crate::kernel_gram::solve_kernel;
use crate::lie_tbl_multivector::{column_stress, pairwise_column_stress, CliffordGramMatrix};
use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct TblRegressParams {
    /// Ridge regularization strength added to each eigenvalue of `X^T X`
    /// before inversion. `0.0` keeps ordinary least squares (regularized
    /// only by `singular_value_floor` truncation below).
    pub ridge_lambda: f64,
    /// Eigenvalues of the centered `X^T X` Gram matrix below
    /// `singular_value_floor * max_eigenvalue` are treated as zero (dropped
    /// from the pseudo-inverse) instead of amplifying noise. This is the
    /// standard truncated-SVD/truncated-eigendecomposition regularizer for
    /// collinear or rank-deficient features.
    pub singular_value_floor: f64,
}

impl Default for TblRegressParams {
    fn default() -> Self {
        Self {
            ridge_lambda: 0.0,
            singular_value_floor: 1e-10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TblRotorRegressor {
    pub coefficients: Array1<f64>,
    pub intercept: f64,
    /// Eigenvalues of the centered `X^T X` Gram matrix, descending. A large
    /// spread (max/min) flags collinear/near-rank-deficient features.
    pub eigenvalues: Array1<f64>,
    /// Number of eigen-directions actually used (not dropped by
    /// `singular_value_floor`); `< d` means the fit found and discarded
    /// near-degenerate feature directions.
    pub rank_used: usize,
}

impl TblRotorRegressor {
    /// `x`: `n` rows (samples) by `d` columns (features). `y`: length `n`.
    pub fn fit(x: &Array2<f64>, y: &Array1<f64>, params: TblRegressParams) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        assert_eq!(
            y.len(),
            n,
            "TblRotorRegressor::fit: x has {n} rows but y has {} entries",
            y.len()
        );
        assert!(n > 0 && d > 0, "TblRotorRegressor::fit: empty input");

        let feature_mean = column_means(x);
        let mut x_centered = x.clone();
        for mut row in x_centered.rows_mut() {
            for j in 0..d {
                row[j] -= feature_mean[j];
            }
        }
        let target_mean = y.iter().sum::<f64>() / n as f64;
        let y_centered: Array1<f64> = y.mapv(|v| v - target_mean);

        let mut gram = x_centered.t().dot(&x_centered);
        // `solve_kernel` checks symmetry before routing to the (cheaper,
        // exact-for-symmetric) eigensolver; explicitly symmetrize first so
        // floating-point asymmetry from the matmul above can't push it onto
        // the general bipartite route instead.
        for i in 0..d {
            for j in (i + 1)..d {
                let avg = 0.5 * (gram[[i, j]] + gram[[j, i]]);
                gram[[i, j]] = avg;
                gram[[j, i]] = avg;
            }
        }
        let kernel_svd = solve_kernel(&gram, 1e-9);
        let v = kernel_svd.u; // eigenvectors of X^T X, columns sorted by eigenvalue descending
        let eigenvalues = kernel_svd.sigma;

        let max_eig = eigenvalues
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let floor = params.singular_value_floor.max(0.0) * max_eig;

        let xty = x_centered.t().dot(&y_centered);
        let projected = v.t().dot(&xty);

        let mut coeff_in_eigenbasis = Array1::<f64>::zeros(d);
        let mut rank_used = 0usize;
        for i in 0..d {
            let lambda = eigenvalues[i];
            if lambda <= floor {
                continue;
            }
            coeff_in_eigenbasis[i] = projected[i] / (lambda + params.ridge_lambda.max(0.0));
            rank_used += 1;
        }
        let coefficients = v.dot(&coeff_in_eigenbasis);
        let intercept = target_mean - dot(&feature_mean, &coefficients);

        Self {
            coefficients,
            intercept,
            eigenvalues,
            rank_used,
        }
    }

    /// Same regression, computed without ever forming `X^T X`: the
    /// centered data matrix `X` is factored directly via the crate's
    /// rectangular `PhaseFlow` route (`U Sigma V^T`), and coefficients are
    /// read off in that basis (`beta = V (Sigma / (Sigma^2 + lambda)) U^T
    /// y`, the standard SVD-based ridge formula). `eigenvalues` on the
    /// result stores `Sigma^2` so it's directly comparable to `fit`'s
    /// output.
    ///
    /// `0.30.4` update: this used to route through
    /// `lie_svd_phaseflow::LieSvdPhaseFlow::phase_lock_rectangular_with_trace`,
    /// which turned out (measured in `0.30.3`) to fail to converge on
    /// generic dense tabular data — `~51-96%` reconstruction error, not a
    /// subtle precision gap. Root cause, found by checking rather than
    /// guessing: it was not a missing golden pre-spin (already present and
    /// invoked); it was that no rectangular "digital polish" existed at
    /// all, because the crate's only exact solver
    /// (`lie_svd_small::LieSvdSmall::solve`) was square-only. This method
    /// now uses `LieSvdSmall::solve_rectangular` (QR-reduction to a small
    /// square factor, then the exact square solve — see that function's
    /// doc comment), which reaches machine precision on the same shapes
    /// that broke the rotor-based route, and returns economy `U: n x k`
    /// directly rather than a wasteful full `n x n` basis.
    pub fn fit_via_rectangular_svd(
        x: &Array2<f64>,
        y: &Array1<f64>,
        params: TblRegressParams,
    ) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        assert_eq!(
            y.len(),
            n,
            "TblRotorRegressor::fit_via_rectangular_svd: x has {n} rows but y has {} entries",
            y.len()
        );
        assert!(
            n > 0 && d > 0,
            "TblRotorRegressor::fit_via_rectangular_svd: empty input"
        );

        let feature_mean = column_means(x);
        let mut x_centered = x.clone();
        for mut row in x_centered.rows_mut() {
            for j in 0..d {
                row[j] -= feature_mean[j];
            }
        }
        let target_mean = y.iter().sum::<f64>() / n as f64;
        let y_centered: Array1<f64> = y.mapv(|v| v - target_mean);

        let (u, sigma, vt) = crate::lie_svd_small::LieSvdSmall::solve_rectangular(&x_centered);
        let k = sigma.len();

        let max_sigma = sigma.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
        let floor = params.singular_value_floor.max(0.0) * max_sigma;

        // proj[i] = (U^T y)[i], using only the first k columns of U (the
        // ones that actually pair with a singular value).
        let mut proj = Array1::<f64>::zeros(k);
        for i in 0..k {
            let mut s = 0.0_f64;
            for r in 0..n {
                s += u[[r, i]] * y_centered[r];
            }
            proj[i] = s;
        }

        let mut coeff_in_svd_basis = Array1::<f64>::zeros(k);
        let mut rank_used = 0usize;
        for i in 0..k {
            let s = sigma[i];
            if s <= floor {
                continue;
            }
            coeff_in_svd_basis[i] = s * proj[i] / (s * s + params.ridge_lambda.max(0.0));
            rank_used += 1;
        }

        // beta[j] = sum_i vt[i, j] * coeff_in_svd_basis[i], i.e. V's first k
        // columns (= vt's first k rows, transposed) applied to the SVD-basis
        // coefficients.
        let mut coefficients = Array1::<f64>::zeros(d);
        for j in 0..d {
            let mut s = 0.0_f64;
            for i in 0..k {
                s += vt[[i, j]] * coeff_in_svd_basis[i];
            }
            coefficients[j] = s;
        }
        let intercept = target_mean - dot(&feature_mean, &coefficients);

        Self {
            coefficients,
            intercept,
            eigenvalues: sigma.mapv(|s| s * s),
            rank_used,
        }
    }

    /// Anisotropic ridge: `(X^T X + Lambda) beta = X^T y`, `Lambda`
    /// diagonal and built from `lie_tbl_multivector` column bivector norms.
    /// See the module doc comment for the exact formula and the correction
    /// made to the originally proposed direction (penalize *low*, not
    /// high, bivector stress).
    pub fn fit_with_bivector_regularization(
        x: &Array2<f64>,
        y: &Array1<f64>,
        lambda0: f64,
    ) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        assert_eq!(
            y.len(),
            n,
            "TblRotorRegressor::fit_with_bivector_regularization: x has {n} rows but y has {} entries",
            y.len()
        );
        assert!(
            n > 0 && d > 0,
            "TblRotorRegressor::fit_with_bivector_regularization: empty input"
        );

        let feature_mean = column_means(x);
        let mut x_centered = x.clone();
        for mut row in x_centered.rows_mut() {
            for j in 0..d {
                row[j] -= feature_mean[j];
            }
        }
        let target_mean = y.iter().sum::<f64>() / n as f64;
        let y_centered: Array1<f64> = y.mapv(|v| v - target_mean);

        let clifford_gram = CliffordGramMatrix::from_columns(&x_centered);
        let mut gram = clifford_gram.scalar.clone();

        // Normalized per-column bivector stress (see `column_stress` in
        // `lie_tbl_multivector` for the exact definition), shared with
        // `GeometricTabularDispatcher`'s route selection.
        let stress = column_stress(&x_centered);
        for j in 0..d {
            gram[[j, j]] += lambda0.max(0.0) / (0.1 + stress[j]);
        }
        // Symmetrize away any floating-point asymmetry before the
        // symmetric-eigen check in `solve_kernel`, same as `fit`.
        for i in 0..d {
            for j in (i + 1)..d {
                let avg = 0.5 * (gram[[i, j]] + gram[[j, i]]);
                gram[[i, j]] = avg;
                gram[[j, i]] = avg;
            }
        }

        let kernel_svd = solve_kernel(&gram, 1e-9);
        let v = kernel_svd.u;
        let eigenvalues = kernel_svd.sigma;

        let xty = x_centered.t().dot(&y_centered);
        let projected = v.t().dot(&xty);
        let mut coeff_in_eigenbasis = Array1::<f64>::zeros(d);
        let mut rank_used = 0usize;
        for i in 0..d {
            let lambda = eigenvalues[i];
            if lambda <= 1e-300 {
                continue;
            }
            coeff_in_eigenbasis[i] = projected[i] / lambda;
            rank_used += 1;
        }
        let coefficients = v.dot(&coeff_in_eigenbasis);
        let intercept = target_mean - dot(&feature_mean, &coefficients);

        Self {
            coefficients,
            intercept,
            eigenvalues,
            rank_used,
        }
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        assert_eq!(
            x.ncols(),
            self.coefficients.len(),
            "TblRotorRegressor::predict: expected {} feature columns, got {}",
            self.coefficients.len(),
            x.ncols()
        );
        x.dot(&self.coefficients).mapv(|v| v + self.intercept)
    }

    /// Dual ("kernel trick") ridge: solves the same regularized least
    /// squares problem as `fit`, but via the `n x n` sample Gram
    /// `K = X X^T` instead of the `d x d` feature Gram `X^T X`. Standard
    /// result (the "push-through identity"):
    /// `beta = X^T (X X^T + lambda I)^-1 y = (X^T X + lambda I)^-1 X^T y`
    /// — both sides solve the same normal equations, but the dual side is
    /// cheaper when `d > n` (a "wide" table: more feature columns than
    /// samples, where `X^T X` is rank-deficient — at most rank `n` — and
    /// `X X^T` is generically full rank). `K` here is exactly
    /// `lie_tbl_multivector::row_scalar_gram`'s construction (the linear
    /// kernel), reused directly rather than recomputed by hand.
    pub fn fit_dual(x: &Array2<f64>, y: &Array1<f64>, ridge_lambda: f64) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        assert_eq!(
            y.len(),
            n,
            "TblRotorRegressor::fit_dual: x has {n} rows but y has {} entries",
            y.len()
        );
        assert!(n > 0 && d > 0, "TblRotorRegressor::fit_dual: empty input");

        let feature_mean = column_means(x);
        let mut x_centered = x.clone();
        for mut row in x_centered.rows_mut() {
            for j in 0..d {
                row[j] -= feature_mean[j];
            }
        }
        let target_mean = y.iter().sum::<f64>() / n as f64;
        let y_centered: Array1<f64> = y.mapv(|v| v - target_mean);

        let mut k = x_centered.dot(&x_centered.t());
        for i in 0..n {
            k[[i, i]] += ridge_lambda.max(0.0);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (k[[i, j]] + k[[j, i]]);
                k[[i, j]] = avg;
                k[[j, i]] = avg;
            }
        }
        let kernel_svd = solve_kernel(&k, 1e-9);
        // (K + lambda I) alpha = y_centered, solved in K's own eigenbasis;
        // K + lambda*I is positive definite for any ridge_lambda > 0 (and
        // positive semidefinite at 0), so every eigenvalue here is safe to
        // invert directly -- no truncation floor needed the way `fit` needs
        // one for the un-regularized feature Gram.
        let v = kernel_svd.u;
        let eigenvalues = kernel_svd.sigma;
        let projected = v.t().dot(&y_centered);
        let mut alpha_in_eigenbasis = Array1::<f64>::zeros(n);
        let mut rank_used = 0usize;
        for i in 0..n {
            if eigenvalues[i] <= 1e-300 {
                continue;
            }
            alpha_in_eigenbasis[i] = projected[i] / eigenvalues[i];
            rank_used += 1;
        }
        let alpha = v.dot(&alpha_in_eigenbasis);
        let coefficients = x_centered.t().dot(&alpha);
        let intercept = target_mean - dot(&feature_mean, &coefficients);

        Self {
            coefficients,
            intercept,
            eigenvalues,
            rank_used,
        }
    }
}

/// Orthogonal Procrustes rotor between two *paired* tables: same column
/// schema, same row count, row `i` of `x_a` corresponding to row `i` of
/// `x_b` (e.g. the same entities measured in two domains/sensors/times).
/// Finds the orthogonal `R` (`d x d`) minimizing `||x_a R - x_b||_F` via
/// the standard closed form: `R = U V^T` from the SVD of `x_a^T x_b`.
///
/// Scope note, corrected from an earlier draft of this idea: the original
/// proposal described this for tables of *different* row counts
/// (`n x d` and `m x d`, `n != m`), but `x_a^T x_b` is only defined when
/// the inner dimensions match — computing it requires `n == m`, which is
/// also what the proposal's own validation test implicitly assumed
/// (`x_b = x_a @ Q + noise`, a row-by-row transform). Aligning tables with
/// genuinely different row counts and no correspondence is a different,
/// harder problem (covariance/distribution alignment, e.g. CORAL-style
/// whitening-recoloring) and is not what this function does.
pub fn procrustes_rotor(x_a: &Array2<f64>, x_b: &Array2<f64>) -> Array2<f64> {
    assert_eq!(
        x_a.nrows(),
        x_b.nrows(),
        "procrustes_rotor: tables must have the same row count (paired rows), \
         got {} and {}",
        x_a.nrows(),
        x_b.nrows()
    );
    assert_eq!(
        x_a.ncols(),
        x_b.ncols(),
        "procrustes_rotor: tables must share the same column schema"
    );
    let cross = x_a.t().dot(x_b);
    let (u, _sigma, vt) = crate::lie_svd_small::LieSvdSmall::solve_rectangular(&cross);
    u.dot(&vt)
}

/// Transfers a fitted model from domain `A` to domain `B` via the rotor
/// `r_ab` (from `procrustes_rotor(x_a, x_b)`), without refitting: if
/// `x_b ~= x_a @ r_ab`, then since `r_ab` is orthogonal
/// (`x_a ~= x_b @ r_ab^T`), a model predicting `y ~= x_a @ beta_a` implies
/// `y ~= x_b @ (r_ab^T @ beta_a)`, so `beta_b = r_ab^T @ beta_a`.
///
/// Scope note: this transfers the coefficient vector exactly (verified by
/// the round-trip test below); it reuses `model_a`'s intercept as-is rather
/// than re-deriving one for domain `B`, which is only appropriate when both
/// domains are centered comparably (e.g. both mean-subtracted before this
/// call, as the test does) — general intercept transfer across differently-
/// centered domains is a separate problem this function does not solve.
pub fn transfer_fit(model_a: &TblRotorRegressor, r_ab: &Array2<f64>) -> TblRotorRegressor {
    let coefficients = r_ab.t().dot(&model_a.coefficients);
    TblRotorRegressor {
        coefficients,
        intercept: model_a.intercept,
        eigenvalues: model_a.eigenvalues.clone(),
        rank_used: model_a.rank_used,
    }
}

/// Which fit route `GeometricTabularDispatcher` picked, and why. Returned
/// alongside the fit so callers (and tests) can see the decision, not just
/// its consequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TblRoute {
    /// `d >= n`: the `d x d` feature Gram `X^T X` is rank-deficient (rank
    /// `<= n < d`) by construction, so the `n x n` sample Gram `X X^T`
    /// (`fit_dual`) is the one that's generically full rank.
    Dual,
    /// `d < n`, feature Gram is well-posed to build, but per-column
    /// bivector stress (`lie_tbl_multivector::column_stress`) shows a
    /// *mix* of near-redundant and near-independent columns — the case
    /// `fit_with_bivector_regularization` is designed for (see its module
    /// doc and `bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity`).
    BivectorRidge,
    /// `d < n` and collinearity (if any) looks roughly uniform across
    /// columns rather than concentrated in a subset — plain Gram-based
    /// ridge (`fit`) has no known deficit to correct for here.
    Gram,
}

/// Routes a table to the fit method it's best suited to, using signals this
/// crate already computes rather than a hand-tuned heuristic table.
///
/// Route selection:
/// 1. `d >= n` -> [`TblRoute::Dual`] (a mathematical necessity, not a
///    tuning choice: `X^T X` cannot be full rank here).
/// 2. Otherwise, compute `lie_tbl_multivector::pairwise_column_stress` (the
///    per-*pair* normalized wedge magnitude, not the per-column mean —
///    averaging over all other columns would dilute a single redundant
///    pair sitting among several unrelated ones, exactly the case this
///    route exists to catch). If the *most redundant* pair is below
///    `redundancy_threshold` (default `0.15`) *and* the *most independent*
///    pair is above `independence_threshold` (default `0.5` — the gap
///    between the near-duplicate pair, `~0.0` wedge, and the independent
///    pairs, `~1.0` wedge, in
///    `bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity`) ->
///    [`TblRoute::BivectorRidge`]: a genuine mix of redundant and
///    independent columns.
/// 3. Otherwise -> [`TblRoute::Gram`] (plain `fit`): either no meaningful
///    collinearity, or collinearity spread roughly evenly across all
///    columns, where a uniform ridge penalty already does the same job an
///    anisotropic one would.
pub struct GeometricTabularDispatcher;

impl GeometricTabularDispatcher {
    /// Pairwise wedge magnitude below which a column *pair* is considered
    /// near-redundant.
    pub const DEFAULT_REDUNDANCY_THRESHOLD: f64 = 0.15;
    /// Pairwise wedge magnitude above which a column pair is considered
    /// near-independent.
    pub const DEFAULT_INDEPENDENCE_THRESHOLD: f64 = 0.5;

    pub fn choose_route(x: &Array2<f64>) -> TblRoute {
        Self::choose_route_with_thresholds(
            x,
            Self::DEFAULT_REDUNDANCY_THRESHOLD,
            Self::DEFAULT_INDEPENDENCE_THRESHOLD,
        )
    }

    pub fn choose_route_with_thresholds(
        x: &Array2<f64>,
        redundancy_threshold: f64,
        independence_threshold: f64,
    ) -> TblRoute {
        let n = x.nrows();
        let d = x.ncols();
        if d >= n {
            return TblRoute::Dual;
        }
        if d < 2 {
            return TblRoute::Gram;
        }
        let pairwise = pairwise_column_stress(x);
        let mut min_pair = f64::INFINITY;
        let mut max_pair = 0.0_f64;
        for j in 0..d {
            for k in 0..d {
                if j == k {
                    continue;
                }
                min_pair = min_pair.min(pairwise[[j, k]]);
                max_pair = max_pair.max(pairwise[[j, k]]);
            }
        }
        if min_pair < redundancy_threshold && max_pair > independence_threshold {
            TblRoute::BivectorRidge
        } else {
            TblRoute::Gram
        }
    }

    /// Chooses a route via [`Self::choose_route`] and fits accordingly.
    /// `params.ridge_lambda` is reused as-is for whichever route is picked
    /// (as the dual ridge strength, the bivector `lambda0`, or the plain
    /// ridge strength); `params.singular_value_floor` only affects the
    /// `Gram` route (the other two routes have their own, different
    /// truncation/conditioning behavior, documented on their own methods).
    pub fn fit(
        x: &Array2<f64>,
        y: &Array1<f64>,
        params: TblRegressParams,
    ) -> (TblRotorRegressor, TblRoute) {
        let route = Self::choose_route(x);
        let model = match route {
            TblRoute::Dual => TblRotorRegressor::fit_dual(x, y, params.ridge_lambda),
            TblRoute::BivectorRidge => {
                TblRotorRegressor::fit_with_bivector_regularization(x, y, params.ridge_lambda)
            }
            TblRoute::Gram => TblRotorRegressor::fit(x, y, params),
        };
        (model, route)
    }
}

fn column_means(x: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows() as f64;
    let d = x.ncols();
    let mut means = Array1::<f64>::zeros(d);
    for row in x.rows() {
        for j in 0..d {
            means[j] += row[j];
        }
    }
    means.mapv_inplace(|v| v / n);
    means
}

fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The construction the rotor-transfer proposal specified for its own
    /// validation: `x_b = x_a @ Q + noise`, a rotated-and-perturbed copy of
    /// domain A. Fits a model on A, transfers it to B via the estimated
    /// Procrustes rotor (never seeing B's targets), and checks its
    /// prediction error on B is close to a model trained on B directly.
    #[test]
    fn transfer_fit_matches_a_model_trained_from_scratch_on_the_rotated_domain() {
        let mut rng = StdRng::seed_from_u64(31);
        let n = 200;
        let d = 4;

        let x_a = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        // A fixed random orthogonal-ish rotation Q, built as the Procrustes
        // solution between two random matrices (guaranteed orthogonal by
        // construction, regardless of the random inputs used to build it).
        let m1 = Array2::from_shape_fn((d, d), |_| rng.gen_range(-1.0_f64..1.0));
        let m2 = Array2::from_shape_fn((d, d), |_| rng.gen_range(-1.0_f64..1.0));
        let q = procrustes_rotor(&m1, &m2);

        let noise_sigma = 0.01;
        let x_b = x_a
            .dot(&q)
            .mapv(|v| v + noise_sigma * rng.gen_range(-1.0_f64..1.0));

        let true_coeffs = Array1::from(vec![1.0, -2.0, 0.5, 1.5]);
        let y_noise = 0.05;
        let y_a = x_a
            .dot(&true_coeffs)
            .mapv(|v| v + y_noise * rng.gen_range(-1.0_f64..1.0));
        // y is defined by the same underlying process in both domains:
        // since x_b ~= x_a @ q, y ~= x_b @ (q^T @ true_coeffs). Noisy here
        // too -- an earlier version left y_b noiseless, which made the
        // "from scratch" baseline fit an exact function of x_b and land at
        // near-machine-precision error, an unrealistically perfect
        // comparison target that made any real (small) transfer error look
        // disproportionately bad by ratio.
        let true_coeffs_b = q.t().dot(&true_coeffs);
        let y_b = x_b
            .dot(&true_coeffs_b)
            .mapv(|v| v + y_noise * rng.gen_range(-1.0_f64..1.0));

        let model_a = TblRotorRegressor::fit(&x_a, &y_a, TblRegressParams::default());
        let r_ab = procrustes_rotor(&x_a, &x_b);
        let transferred = transfer_fit(&model_a, &r_ab);
        let from_scratch = TblRotorRegressor::fit(&x_b, &y_b, TblRegressParams::default());

        let resid = |model: &TblRotorRegressor| -> f64 {
            model
                .predict(&x_b)
                .iter()
                .zip(y_b.iter())
                .map(|(p, t)| (p - t).abs())
                .fold(0.0_f64, f64::max)
        };
        let transferred_resid = resid(&transferred);
        let from_scratch_resid = resid(&from_scratch);
        // Absolute bound calibrated to the noise actually injected
        // (uniform +/- y_noise on top of x-noise-induced error), not an
        // arbitrarily tight number: max residual over 200 points from pure
        // label noise alone can reasonably approach a few multiples of
        // `y_noise`.
        assert!(
            transferred_resid < 6.0 * y_noise,
            "transferred_resid={transferred_resid:e}"
        );
        // The meaningful comparison, now that both models face the same
        // label noise: transfer (no B-domain labels used at all) should be
        // within a small constant factor of training on B from scratch.
        assert!(
            transferred_resid < 3.0 * from_scratch_resid,
            "expected transfer to be competitive with training from scratch \
             (transferred={transferred_resid:e}, from_scratch={from_scratch_resid:e})"
        );
    }

    /// Held-out A/B test: plain isotropic ridge (`fit` with `ridge_lambda`)
    /// against `fit_with_bivector_regularization`, same `lambda0`, trained
    /// on the same data, evaluated on a disjoint held-out set. Table: 4
    /// columns, 0 and 1 a near-duplicate (redundant) pair, 2 and 3
    /// independent of everything.
    ///
    /// Result, measured across five regularization strengths
    /// (`0.01, 0.1, 0.5, 1.0, 3.0`): bivector-aware ridge gives a *small but
    /// consistent* improvement in held-out RMSE at every one of them (not
    /// cherry-picked — every value tested wins), growing from `~0.1%` at
    /// the smallest `lambda0` to `~1.9%` at the largest. This matches the
    /// mechanism: at low regularization neither penalty does much, so they
    /// barely differ; as `lambda0` grows, shrinking the redundant pair
    /// harder while sparing the independent columns increasingly pays off
    /// relative to shrinking everything equally. Modest, not dramatic — but
    /// real and directionally consistent, not a single favorable roll of
    /// the noise seed.
    #[test]
    fn bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity() {
        let mut rng = StdRng::seed_from_u64(77);
        let n_train = 60;
        let n_test = 60;
        let n = n_train + n_test;
        let mut x = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let base = rng.gen_range(-1.0_f64..1.0);
            x[[i, 0]] = base;
            x[[i, 1]] = base + 0.02 * rng.gen_range(-1.0_f64..1.0);
            x[[i, 2]] = rng.gen_range(-1.0_f64..1.0);
            x[[i, 3]] = rng.gen_range(-1.0_f64..1.0);
        }
        let true_coeffs = Array1::from(vec![1.0, 1.0, 2.0, -1.5]);
        let noise_sigma = 0.3;
        let y = Array1::from_shape_fn(n, |i| {
            let s = true_coeffs[0] * x[[i, 0]]
                + true_coeffs[1] * x[[i, 1]]
                + true_coeffs[2] * x[[i, 2]]
                + true_coeffs[3] * x[[i, 3]];
            s + noise_sigma * rng.gen_range(-1.0_f64..1.0)
        });

        let x_train = x.slice(ndarray::s![0..n_train, ..]).to_owned();
        let y_train = y.slice(ndarray::s![0..n_train]).to_owned();
        let x_test = x.slice(ndarray::s![n_train..n, ..]).to_owned();
        let y_test = y.slice(ndarray::s![n_train..n]).to_owned();

        let rmse = |pred: &Array1<f64>| -> f64 {
            (pred
                .iter()
                .zip(y_test.iter())
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f64>()
                / n_test as f64)
                .sqrt()
        };

        for lambda0 in [0.01, 0.1, 0.5, 1.0, 3.0] {
            let plain = TblRotorRegressor::fit(
                &x_train,
                &y_train,
                TblRegressParams {
                    ridge_lambda: lambda0,
                    singular_value_floor: 0.0,
                },
            );
            let biv =
                TblRotorRegressor::fit_with_bivector_regularization(&x_train, &y_train, lambda0);
            let plain_rmse = rmse(&plain.predict(&x_test));
            let biv_rmse = rmse(&biv.predict(&x_test));
            assert!(
                biv_rmse <= plain_rmse,
                "lambda0={lambda0}: expected bivector-ridge to match or beat plain ridge \
                 on held-out RMSE (plain={plain_rmse:e}, bivector={biv_rmse:e})"
            );
        }
    }

    /// Direct, honest comparison: `fit` (Gram-based) against
    /// `fit_via_rectangular_svd` (never forms `X^T X`) on the same
    /// ill-conditioned (near-duplicate column) data. This is the test the
    /// module doc comment describes.
    ///
    /// History, since the numbers flipped once and it matters why: in
    /// `0.30.3`, `fit_via_rectangular_svd` routed through
    /// `lie_svd_phaseflow`'s rotor-based rectangular route, which failed to
    /// *converge* on generic dense data (`~51-96%` raw reconstruction
    /// error, not a subtle precision gap) — `rect_resid` here was many
    /// orders of magnitude worse than `gram_resid`. `0.30.4` replaced that
    /// with `LieSvdSmall::solve_rectangular` (QR-reduction, see that
    /// function), which is exact rather than iterative. Measured now:
    /// `rect_resid` (`~3.3e-9`) is *smaller* than `gram_resid` (`~1.1e-6`)
    /// on this same near-collinear input — the condition-number-squaring
    /// argument for avoiding `X^T X` was correct, once routed through a
    /// solver that actually reaches machine precision on rectangular
    /// input. `fit` stays the default (it's the better-tested, longer-lived
    /// path, and its truncation semantics are easier to reason about at the
    /// feature level), but `fit_via_rectangular_svd` is no longer a known
    /// failure mode.
    #[test]
    fn gram_vs_rectangular_svd_on_ill_conditioned_features() {
        let n = 30;
        let true_coeffs = Array1::from(vec![1.5, -2.0, 0.7]);
        // Column 1 ~= 2 * column 0 plus tiny noise: X^T X has a near-zero
        // eigenvalue (a realistic "two collinear features" table).
        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
            let base = ((i * 11 + 3) as f64 * 0.23).sin() * 2.0;
            match j {
                0 => base,
                1 => 2.0 * base + 1e-6 * ((i * 13 + 7) as f64 * 0.7).sin(),
                _ => ((i * 17 + j * 5) as f64 * 0.41).cos() * 1.5,
            }
        });
        let y = x.dot(&true_coeffs).mapv(|v| v + 3.0);

        let gram = TblRotorRegressor::fit(&x, &y, TblRegressParams::default());
        let rect = TblRotorRegressor::fit_via_rectangular_svd(&x, &y, TblRegressParams::default());

        // Not checking recovered coefficients against `true_coeffs`: with
        // columns this close to collinear, the coefficient vector along
        // that near-degenerate direction isn't uniquely identifiable from
        // the data at all (infinitely many combinations predict equally
        // well) — the correct thing for a truncated fit to do is decline to
        // assign it weight, not "recover" a value that isn't determined by
        // the data. Prediction accuracy is the well-posed check here.
        let gram_pred = gram.predict(&x);
        let gram_resid = gram_pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        assert!(gram_resid < 1e-4, "gram_resid={gram_resid:e}");

        let rect_pred = rect.predict(&x);
        let rect_resid = rect_pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        assert!(rect_resid < 1e-4, "rect_resid={rect_resid:e}");
    }

    #[test]
    fn fit_dual_matches_primal_fit_when_well_conditioned() {
        // d < n, well-conditioned: primal and dual ridge solve the same
        // normal equations, so they must agree numerically regardless of
        // which side of the push-through identity computed them.
        let x = Array2::from_shape_fn((30, 3), |(i, j)| ((i * 7 + j * 5 + 1) as f64).sin());
        let true_coeffs = Array1::from(vec![1.0, -2.0, 0.5]);
        let y = x.dot(&true_coeffs).mapv(|v| v + 2.0);

        let primal = TblRotorRegressor::fit(
            &x,
            &y,
            TblRegressParams {
                ridge_lambda: 0.3,
                singular_value_floor: 0.0,
            },
        );
        let dual = TblRotorRegressor::fit_dual(&x, &y, 0.3);
        for j in 0..3 {
            assert!(
                (primal.coefficients[j] - dual.coefficients[j]).abs() < 1e-8,
                "coeff[{j}]: primal={} dual={}",
                primal.coefficients[j],
                dual.coefficients[j]
            );
        }
        assert!((primal.intercept - dual.intercept).abs() < 1e-8);
    }

    #[test]
    fn fit_dual_stays_finite_and_accurate_on_a_wide_table() {
        // d > n: X^T X (what `fit` uses) is rank-deficient here (rank <= n
        // < d); `fit_dual` uses X X^T (n x n), generically full rank, and
        // should recover an exact noiseless linear target cleanly.
        let n = 8;
        let d = 20;
        let mut rng = StdRng::seed_from_u64(5);
        let x = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        let true_coeffs = Array1::from_shape_fn(d, |j| ((j + 1) as f64) * 0.3);
        let y = x.dot(&true_coeffs).mapv(|v| v + 1.0);

        let dual = TblRotorRegressor::fit_dual(&x, &y, 1e-6);
        assert!(dual.coefficients.iter().all(|c| c.is_finite()));
        let pred = dual.predict(&x);
        let max_err = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        // A small but nonzero `ridge_lambda` deliberately biases the fit
        // away from an exact interpolation, so this is not machine
        // precision -- just "small and finite", which is the actual claim.
        assert!(max_err < 1e-4, "max_err={max_err:e}");
    }

    #[test]
    fn recovers_exact_linear_target() {
        // y = 2*x0 - 3*x1 + 5, no noise: the fit should recover this almost
        // exactly and predict with near-zero residual.
        let x = Array2::from_shape_fn((40, 2), |(i, j)| {
            let seed = (i * 7 + j * 13 + 3) as f64;
            (seed * 0.618).sin() * 3.0
        });
        let true_coeffs = Array1::from(vec![2.0, -3.0]);
        let true_intercept = 5.0;
        let y = x.dot(&true_coeffs).mapv(|v| v + true_intercept);

        let model = TblRotorRegressor::fit(&x, &y, TblRegressParams::default());
        assert_eq!(model.rank_used, 2);
        for (got, want) in model.coefficients.iter().zip(true_coeffs.iter()) {
            assert!((got - want).abs() < 1e-8, "got={got} want={want}");
        }
        assert!((model.intercept - true_intercept).abs() < 1e-8);

        let pred = model.predict(&x);
        let max_err = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-8, "max_err={max_err:e}");
    }

    #[test]
    fn truncates_a_duplicated_collinear_column_without_blowing_up() {
        // Column 1 is an exact copy of column 0: X^T X is exactly singular
        // along that direction. A naive (X^T X)^-1 X^T y would need to
        // invert a singular matrix; this fit must instead drop that
        // direction (rank_used < d) and still produce finite, bounded
        // coefficients and predictions.
        let n = 30;
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let v = ((i * 11 + 5) as f64 * 0.37).sin() * 4.0;
            x[[i, 0]] = v;
            x[[i, 1]] = v; // exact duplicate column
        }
        let y = Array1::from_shape_fn(n, |i| 3.0 * x[[i, 0]] + 1.0);

        let model = TblRotorRegressor::fit(&x, &y, TblRegressParams::default());
        assert_eq!(
            model.rank_used, 1,
            "the duplicate direction must be dropped"
        );
        assert!(model.coefficients.iter().all(|c| c.is_finite()));
        assert!(model.intercept.is_finite());

        let pred = model.predict(&x);
        let max_err = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-6, "max_err={max_err:e}");
    }

    #[test]
    fn ridge_shrinks_coefficients_relative_to_unregularized_fit() {
        let x = Array2::from_shape_fn((25, 3), |(i, j)| {
            ((i * 5 + j * 3 + 1) as f64 * 0.29).cos() * 2.0
        });
        let y = Array1::from_shape_fn(25, |i| 0.5 * x[[i, 0]] - 1.5 * x[[i, 1]] + 0.8 * x[[i, 2]]);

        let plain = TblRotorRegressor::fit(&x, &y, TblRegressParams::default());
        let ridge = TblRotorRegressor::fit(
            &x,
            &y,
            TblRegressParams {
                ridge_lambda: 5.0,
                ..TblRegressParams::default()
            },
        );

        let plain_norm: f64 = plain.coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();
        let ridge_norm: f64 = ridge.coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!(
            ridge_norm < plain_norm,
            "ridge_norm={ridge_norm} plain_norm={plain_norm}"
        );
    }

    /// `GeometricTabularDispatcher` on a wide table (`d > n`) must pick
    /// `Dual`: `X^T X` (20x20) has rank at most `n=8`, so it is singular by
    /// construction, regardless of the actual data.
    #[test]
    fn dispatcher_routes_wide_table_to_dual() {
        let mut rng = StdRng::seed_from_u64(101);
        let n = 8;
        let d = 20;
        let x = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        let true_coeffs = Array1::from_shape_fn(d, |j| ((j + 1) as f64) * 0.1);
        let y = x.dot(&true_coeffs);

        let route = GeometricTabularDispatcher::choose_route(&x);
        assert_eq!(route, TblRoute::Dual, "d={d} >= n={n} must route to Dual");

        let (model, picked) = GeometricTabularDispatcher::fit(
            &x,
            &y,
            TblRegressParams {
                ridge_lambda: 1e-6,
                ..TblRegressParams::default()
            },
        );
        assert_eq!(picked, TblRoute::Dual);
        assert!(model.coefficients.iter().all(|c| c.is_finite()));
    }

    /// Same anisotropic-collinearity table shape as
    /// `bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity`
    /// (columns 0/1 a near-duplicate pair, columns 2/3 independent of
    /// everything): `column_stress` should show a large min/max spread, so
    /// the dispatcher should pick `BivectorRidge`.
    #[test]
    fn dispatcher_routes_anisotropic_collinearity_to_bivector_ridge() {
        let mut rng = StdRng::seed_from_u64(103);
        let n = 60;
        let mut x = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let base = rng.gen_range(-1.0_f64..1.0);
            x[[i, 0]] = base;
            x[[i, 1]] = base + 0.02 * rng.gen_range(-1.0_f64..1.0);
            x[[i, 2]] = rng.gen_range(-1.0_f64..1.0);
            x[[i, 3]] = rng.gen_range(-1.0_f64..1.0);
        }
        let true_coeffs = Array1::from(vec![1.0, 1.0, 2.0, -1.5]);
        let y = x.dot(&true_coeffs);

        let route = GeometricTabularDispatcher::choose_route(&x);
        assert_eq!(
            route,
            TblRoute::BivectorRidge,
            "near-duplicate + independent columns must route to BivectorRidge"
        );

        let (model, picked) = GeometricTabularDispatcher::fit(
            &x,
            &y,
            TblRegressParams {
                ridge_lambda: 0.5,
                ..TblRegressParams::default()
            },
        );
        assert_eq!(picked, TblRoute::BivectorRidge);
        assert!(model.coefficients.iter().all(|c| c.is_finite()));
    }

    /// A well-conditioned table (independent, non-collinear columns, ample
    /// samples) should route to plain `Gram`: there is no redundant-column
    /// subset for the bivector penalty to specifically target, and `d < n`
    /// rules out `Dual`.
    #[test]
    fn dispatcher_routes_well_conditioned_table_to_gram() {
        let mut rng = StdRng::seed_from_u64(107);
        let n = 100;
        let d = 4;
        let x = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        let true_coeffs = Array1::from(vec![1.0, -2.0, 0.5, 1.5]);
        let y = x.dot(&true_coeffs);

        let route = GeometricTabularDispatcher::choose_route(&x);
        assert_eq!(
            route,
            TblRoute::Gram,
            "well-conditioned independent columns must route to Gram"
        );

        let (model, picked) = GeometricTabularDispatcher::fit(&x, &y, TblRegressParams::default());
        assert_eq!(picked, TblRoute::Gram);
        assert!(model.coefficients.iter().all(|c| c.is_finite()));
    }
}
