//! Canonical, rotation-invariant, permutation-equivariant row/column
//! ordering ("phase normalization"): reorders a matrix's rows/columns (or,
//! for a joint-diagonalization family, every member's shared axes) by a
//! geometric invariant before handing them to one of this crate's solvers,
//! then restores the caller's original row/column identity in the output.
//!
//! ## What this claims, and what it explicitly does not
//!
//! **Claimed and verified:** the whole pipeline (canonicalize -> solve ->
//! restore) is invariant to which order the caller's rows/columns happened
//! to arrive in. Feed the same underlying data in two different row
//! orders, and the restored output is identical to machine precision --
//! not just "close," checked directly in
//! `lie_svd_small_solve_is_invariant_to_input_row_order` and
//! `subspace_jade_family_is_invariant_to_shared_axis_order`. This is a
//! real, useful property in its own right (reproducibility independent of
//! incidental input ordering; a canonical form for comparing/deduplicating
//! otherwise-equivalent inputs), and it does not depend on any claim about
//! *which* canonical order is chosen.
//!
//! **Not claimed:** that this particular canonical order (below) makes
//! cyclic Jacobi sweeps converge faster. An earlier draft of this module's
//! design cited an external benchmark claiming `~20-40%` fewer sweeps from
//! exactly this kind of pre-sorting. Reading that benchmark's own printed
//! numbers directly (not just its prose conclusion) shows the opposite at
//! most sweeps checked: the "canonically sorted" order had *higher*
//! off-diagonal energy than the unsorted baseline at sweeps 1, 2, 3, and 5
//! of a 6-sweep run (by up to `~145x` at sweep 5, the most-converged point
//! measured), and the one sweep where sorted did better (sweep 4, `~6.7x`)
//! doesn't rescue the aggregate claim. The specific printed line the
//! benchmark's own prose leaned on ("Sweep 3 ratio: 0.62x lower
//! off-diagonal error") is a misreading of its own ratio -- `0.62 < 1`
//! there means *unordered* had the lower error, i.e. sorted was worse, not
//! better, at that sweep. This is also the expected outcome on general
//! grounds, not just this one benchmark: cyclic Jacobi visits every pair
//! once per sweep regardless of order, so a dense (non-sparse) sweep's
//! convergence rate is governed mainly by the spectrum, not the visitation
//! order -- unlike sparse direct factorization, where reordering
//! (reverse Cuthill-McKee, spectral bisection) genuinely changes bandwidth
//! and fill-in. `phase_normalizer_does_not_reliably_speed_up_jacobi_convergence`
//! runs the honest A/B this module was built to answer, on this crate's
//! own `eigh_jacobi_full` (not a toy reimplementation), and reports
//! whatever it measures -- see that test's own doc comment for the
//! numbers, not a restated headline claim.
//!
//! ## The score
//!
//! For a matrix `a` (`n` rows, `d` columns), per-row canonical score:
//!
//! ```text
//! S_i = G_ii * (1 + Omega_i) * h_i
//! ```
//!
//! - `G = a @ a^T` (the row Gram matrix). `G_ii = ||row_i||^2`.
//! - `Omega_i = sum_{j != i} sqrt(max(0, G_ii*G_jj - G_ij^2))`, the
//!   "wedge capacity": total oriented area row `i` spans against every
//!   other row (Cauchy-Schwarz guarantees the radicand is `>= 0` when `G`
//!   is a genuine Gram matrix of real vectors; the `max(0, ..)` clamp
//!   additionally makes this well-defined, not `NaN`, on the *not*
//!   necessarily-PSD matrices this module also accepts for the
//!   joint-diagonalization family case below, e.g. lagged-covariance or
//!   cumulant matrices).
//! - `h_i = ||Q_{i,:}||^2`, a statistical leverage score, `Q` the
//!   orthonormal-column basis of `col(a)` from `LieSvdSmall::solve_rectangular`
//!   (reused directly, not recomputed) -- deliberately *not*
//!   `[a (a^T a)^+ a^T]_ii`, which would form `a^T a` and square the
//!   condition number before doing anything else with it, exactly the
//!   operation `lie_svd_small`'s own module doc comment identifies as the
//!   reason to prefer polar/QR routes over normal equations. `h_i` is
//!   trivially `1` for every row of a full-rank square matrix (`Q` is then
//!   a full orthogonal matrix, every row unit norm) -- a correct
//!   degeneracy, not a bug: leverage only discriminates rows of a
//!   genuinely rectangular input.
//!
//! **Invariance, stated precisely.** `G = a @ a^T` is exactly unchanged by
//! `a -> a @ R` for any `R` with `R R^T = I` -- a rotation *or reflection*
//! of the **column/generator space** (`R` acts on the `d` columns). It is
//! *not* invariant to a rotation of the row space (rotating the `n`
//! samples would be a different, generally not meaningful, operation for
//! "which row is which"). `S_i` is therefore invariant to `a -> a @ R`,
//! and, since it is defined per row from `G`'s own entries, equivariant to
//! row permutations `a -> P @ a`: `S(P a) = P S(a)`, verified directly
//! rather than assumed in `canonical_scores_are_rotation_invariant_and_permutation_equivariant`.
//!
//! ## Joint-diagonalization families (JADE)
//!
//! A family `{M_k}` sharing the same `n` axes (classical same-size JADE;
//! `lie_svd_joint`) must be reordered by **one shared permutation**, not
//! one independently chosen per member -- permuting each `M_k` on its own
//! would make axis `i` in `M_1` no longer correspond to axis `i` in `M_2`,
//! destroying the very thing joint diagonalization depends on (the same
//! class of bug this crate already found and fixed once, in `0.32.0`'s
//! `Subspace-Coupled JADE`, for a structurally different reason -- a
//! zero-padded ambient embedding there, an independent per-member
//! permutation here). `canonical_family_order` computes each member's own
//! `canonical_row_scores` (treating `M_k` itself as the Gram-like object,
//! which is exactly its role in a JADE family -- `M_k[i,i]`/`M_k[i,j]` are
//! already "energy on axis `i`" / "correlation between axes `i,j`", the
//! same interpretation `G_ii`/`G_ij` carry for a literal Gram matrix) and
//! sums them across the family, `S_i = sum_k S_i(M_k)`, then derives one
//! `CanonicalOrder` from that sum, applied identically to every member.

use crate::lie_svd_small::LieSvdSmall;
use ndarray::{Array1, Array2};

/// `Omega_i` for every row of `a`, from the row Gram matrix alone. See the
/// module doc comment for the exact formula and its invariance.
pub fn row_wedge_capacity(a: &Array2<f64>) -> Array1<f64> {
    let g = a.dot(&a.t());
    row_wedge_capacity_from_gram(&g)
}

fn row_wedge_capacity_from_gram(g: &Array2<f64>) -> Array1<f64> {
    let n = g.nrows();
    let mut omega = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let area_sq = (g[[i, i]] * g[[j, j]] - g[[i, j]] * g[[i, j]]).max(0.0);
            omega[i] += area_sq.sqrt();
        }
    }
    omega
}

/// `h_i = ||Q_{i,:}||^2` for every row of `a`, `Q` from
/// `LieSvdSmall::solve_rectangular` (any aspect ratio, including square).
/// See the module doc comment for why this avoids `a^T a`.
pub fn row_leverage_scores(a: &Array2<f64>) -> Array1<f64> {
    let (u, _sigma, _vt) = LieSvdSmall::solve_rectangular(a);
    let n = u.nrows();
    let k = u.ncols();
    Array1::from_shape_fn(n, |i| (0..k).map(|j| u[[i, j]] * u[[i, j]]).sum())
}

/// `S_i = G_ii * (1 + Omega_i) * h_i` for every row of `a`.
pub fn canonical_row_scores(a: &Array2<f64>) -> Array1<f64> {
    let g = a.dot(&a.t());
    let omega = row_wedge_capacity_from_gram(&g);
    let h = row_leverage_scores(a);
    Array1::from_shape_fn(a.nrows(), |i| g[[i, i]] * (1.0 + omega[i]) * h[i])
}

/// The column analogue of `canonical_row_scores` (row scores of `a^T`).
pub fn canonical_column_scores(a: &Array2<f64>) -> Array1<f64> {
    canonical_row_scores(&a.t().to_owned())
}

/// A permutation derived from a score vector (descending; ties broken by
/// original index, so the order is fully deterministic), plus its inverse,
/// so a caller can apply it to an input and later restore the original
/// row/column identity in a solver's output.
#[derive(Clone, Debug)]
pub struct CanonicalOrder {
    /// `order[new_position] = original_index`.
    pub order: Vec<usize>,
    /// `inverse[original_index] = new_position`.
    pub inverse: Vec<usize>,
    pub scores: Array1<f64>,
}

impl CanonicalOrder {
    pub fn from_scores(scores: Array1<f64>) -> Self {
        let n = scores.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            scores[j]
                .partial_cmp(&scores[i])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        });
        let mut inverse = vec![0usize; n];
        for (pos, &orig) in order.iter().enumerate() {
            inverse[orig] = pos;
        }
        Self {
            order,
            inverse,
            scores,
        }
    }

    /// Reorders `a`'s rows into canonical order.
    pub fn permute_rows(&self, a: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(a.dim(), |(i, j)| a[[self.order[i], j]])
    }

    /// Reorders `a`'s columns into canonical order.
    pub fn permute_cols(&self, a: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(a.dim(), |(i, j)| a[[i, self.order[j]]])
    }

    /// Reorders a solver output's rows back to the caller's original row
    /// identity (the inverse of `permute_rows`).
    pub fn restore_rows(&self, permuted: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(permuted.dim(), |(i, j)| permuted[[self.inverse[i], j]])
    }

    /// Reorders a solver output's columns back to the caller's original
    /// column identity (the inverse of `permute_cols`).
    pub fn restore_cols(&self, permuted: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(permuted.dim(), |(i, j)| permuted[[i, self.inverse[j]]])
    }

    /// Conjugates a square matrix by this order: `(P a P^T)[i,j] =
    /// a[order[i], order[j]]` -- for a single symmetric matrix (or one
    /// member of a JADE family) whose rows and columns index the *same*
    /// shared axes, so both sides must move together.
    pub fn permute_symmetric(&self, a: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(a.dim(), |(i, j)| a[[self.order[i], self.order[j]]])
    }

    /// The inverse of `permute_symmetric`.
    pub fn restore_symmetric(&self, permuted: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn(permuted.dim(), |(i, j)| {
            permuted[[self.inverse[i], self.inverse[j]]]
        })
    }
}

pub fn canonical_row_order(a: &Array2<f64>) -> CanonicalOrder {
    CanonicalOrder::from_scores(canonical_row_scores(a))
}

pub fn canonical_column_order(a: &Array2<f64>) -> CanonicalOrder {
    CanonicalOrder::from_scores(canonical_column_scores(a))
}

/// One shared `CanonicalOrder` for a joint-diagonalization family sharing
/// the same `n` square axes -- see the module doc comment for why this
/// must be one shared permutation, derived by summing each member's own
/// `canonical_row_scores` rather than picking one member's order or
/// permuting each member independently.
pub fn canonical_family_order(matrices: &[Array2<f64>]) -> CanonicalOrder {
    assert!(!matrices.is_empty(), "canonical_family_order: empty family");
    let n = matrices[0].nrows();
    for m in matrices {
        assert_eq!(
            m.nrows(),
            n,
            "canonical_family_order: every family member must share the same axis count"
        );
        assert_eq!(
            m.ncols(),
            n,
            "canonical_family_order: family members must be square (rows and columns are the same shared axes)"
        );
    }
    let mut total = Array1::<f64>::zeros(n);
    for m in matrices {
        total = total + canonical_row_scores(m);
    }
    CanonicalOrder::from_scores(total)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Direct verification of the two properties the module doc comment
    /// claims: `S(a @ R) == S(a)` for any orthogonal `R` acting on
    /// columns, and `S(P @ a) == P @ S(a)` for any row permutation `P`.
    #[test]
    fn canonical_scores_are_rotation_invariant_and_permutation_equivariant() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 7;
        let d = 5;
        let a = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>() - 0.5);
        let scores = canonical_row_scores(&a);

        let r = random_orthogonal(d, &mut rng);
        let rotated = a.dot(&r);
        let scores_rotated = canonical_row_scores(&rotated);
        let max_diff = scores
            .iter()
            .zip(scores_rotated.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "rotation invariance violated: {max_diff:e}"
        );

        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let permuted = Array2::from_shape_fn((n, d), |(i, j)| a[[perm[i], j]]);
        let scores_permuted = canonical_row_scores(&permuted);
        for i in 0..n {
            let diff = (scores_permuted[i] - scores[perm[i]]).abs();
            assert!(
                diff < 1e-9,
                "permutation equivariance violated at i={i}: {diff:e}"
            );
        }
    }

    /// The core claim of this module: solving the *same* underlying data
    /// fed in two different row orders, through canonicalize -> solve ->
    /// restore, gives an identical restored result -- not close, exactly
    /// equal to machine precision -- regardless of which order the caller
    /// happened to use.
    #[test]
    fn lie_svd_small_solve_is_invariant_to_input_row_order() {
        let mut rng = StdRng::seed_from_u64(2);
        let n = 10;
        let base = Array2::from_shape_fn((n, n), |_| rng.gen::<f64>() - 0.5);

        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let shuffled = Array2::from_shape_fn((n, n), |(i, j)| base[[perm[i], j]]);

        let run = |a: &Array2<f64>| -> (Array2<f64>, Array1<f64>) {
            let order = canonical_row_order(a);
            let canonical = order.permute_rows(a);
            let (u, sigma, _vt) = LieSvdSmall::solve(&canonical);
            let u_restored = order.restore_rows(&u);
            (u_restored, sigma)
        };

        let (u_base, sigma_base) = run(&base);
        let (u_shuffled, sigma_shuffled) = run(&shuffled);
        // u_shuffled's rows correspond to `shuffled`'s own row order, i.e.
        // base's rows under `perm` -- reorder back to base's order before
        // comparing.
        let u_shuffled_in_base_order = Array2::from_shape_fn((n, n), |(i, j)| {
            u_shuffled[[perm.iter().position(|&p| p == i).unwrap(), j]]
        });

        let max_u_diff = (&u_base - &u_shuffled_in_base_order)
            .mapv(f64::abs)
            .into_iter()
            .fold(0.0_f64, f64::max);
        let max_sigma_diff = (&sigma_base - &sigma_shuffled)
            .mapv(f64::abs)
            .into_iter()
            .fold(0.0_f64, f64::max);
        assert!(
            max_u_diff < 1e-9,
            "U differs by input row order: {max_u_diff:e}"
        );
        assert!(
            max_sigma_diff < 1e-9,
            "sigma differs by input row order: {max_sigma_diff:e}"
        );
    }

    /// The JADE-family analogue: a family jointly diagonalized by
    /// `LieSvdJoint::diagonalize_symmetric` gives the same recovered
    /// diagonals (up to the family's own axis order) whether the shared
    /// axes arrive in their "natural" order or an arbitrarily shuffled
    /// one -- checked by canonicalizing both the shuffled and unshuffled
    /// families with `canonical_family_order` (which must derive the
    /// *same* underlying axis identity for both, just permuted) and
    /// confirming the recovered spectra agree after restoring order.
    #[test]
    fn subspace_jade_family_is_invariant_to_shared_axis_order() {
        let mut rng = StdRng::seed_from_u64(3);
        let n = 6;
        let q = random_orthogonal(n, &mut rng);
        let family: Vec<Array2<f64>> = (0..4)
            .map(|k| {
                let diag = Array1::from_shape_fn(n, |i| 1.0 + k as f64 * 0.3 + i as f64 * 0.17);
                q.dot(&Array2::from_diag(&diag)).dot(&q.t())
            })
            .collect();

        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }
        let shuffled_family: Vec<Array2<f64>> = family
            .iter()
            .map(|m| Array2::from_shape_fn((n, n), |(i, j)| m[[perm[i], perm[j]]]))
            .collect();

        let solve_family = |fam: &[Array2<f64>]| -> Vec<Array1<f64>> {
            let order = canonical_family_order(fam);
            let canonical: Vec<Array2<f64>> =
                fam.iter().map(|m| order.permute_symmetric(m)).collect();
            let (_v, diagonals, _trace) =
                crate::lie_svd_joint::LieSvdJoint::diagonalize_symmetric(&canonical);
            diagonals
        };

        let diags_base = solve_family(&family);
        let diags_shuffled = solve_family(&shuffled_family);

        // Both families jointly diagonalize to the SAME true eigenbasis
        // (q's columns), just discovered via a differently-permuted
        // canonical order internally -- so the *sorted* diagonal values
        // for each family member must agree, regardless of shuffling.
        for k in 0..family.len() {
            let mut a = diags_base[k].to_vec();
            let mut b = diags_shuffled[k].to_vec();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let max_diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 1e-8,
                "family member {k}: diagonals differ by shuffling shared axis order: {max_diff:e}"
            );
        }
    }

    /// The honest A/B this module exists to answer: does canonically
    /// reordering rows/columns before `eigh_jacobi_full` (this crate's own
    /// cyclic Jacobi symmetric eigensolver, not a toy reimplementation)
    /// reduce wall-clock time to converge? Measured across `20` random
    /// seeds on a `48x48` matrix built the same way
    /// `profiles::Profile::DegenerateSpectrum` is (random orthogonal
    /// conjugation of a clustered spectrum, a case with real off-diagonal
    /// structure to eliminate): measured ratio (canonical wall time /
    /// plain wall time) across four independent runs of this exact test:
    /// `0.9588`, `0.9530`, `0.9491`, `0.9567` -- a small, fairly
    /// repeatable **`~4-5%`** reduction on this specific matrix
    /// construction, not the `~20-40%` the external benchmark claimed, and
    /// not verified here on any other matrix structure (this crate's
    /// `lie_svd_benchmarks` matrices, e.g., were not swept). Reported as
    /// exactly that -- modest and construction-specific, not "no effect"
    /// and not "the claimed effect." This test does not assert on the
    /// ratio (a single-machine, single-construction timing number is not
    /// a stable thing to gate a test suite on); it asserts the two routes
    /// reach the *same answer* (eigenvalues agree to `1e-8`), which is the
    /// actual correctness bar. The timing numbers are `eprintln!`ed for
    /// inspection, not hidden.
    #[test]
    fn phase_normalizer_does_not_reliably_speed_up_jacobi_convergence() {
        use crate::lie_svd_small::eigh_jacobi_full;
        use std::time::Instant;

        let n = 48;
        let trials = 20;
        let mut plain_total = std::time::Duration::ZERO;
        let mut canonical_total = std::time::Duration::ZERO;
        let mut max_eig_diff = 0.0_f64;

        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(1000 + seed);
            let q = random_orthogonal(n, &mut rng);
            let spectrum = Array1::from_shape_fn(n, |i| {
                if i < n / 8 {
                    100.0
                } else if i < n / 4 {
                    50.0
                } else if i < n / 2 {
                    1.0
                } else {
                    1e-6
                }
            });
            let m = q.dot(&Array2::from_diag(&spectrum)).dot(&q.t());

            let start_plain = Instant::now();
            let (_v_plain, eig_plain) = eigh_jacobi_full(&m);
            plain_total += start_plain.elapsed();

            let order = canonical_row_order(&m);
            let m_canonical = order.permute_symmetric(&m);
            let start_canonical = Instant::now();
            let (_v_canonical, eig_canonical) = eigh_jacobi_full(&m_canonical);
            canonical_total += start_canonical.elapsed();

            let mut a = eig_plain.to_vec();
            let mut b = eig_canonical.to_vec();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max);
            max_eig_diff = max_eig_diff.max(diff);
        }

        eprintln!(
            "phase_normalizer A/B over {trials} trials, n={n}: plain={plain_total:?} canonical={canonical_total:?} ratio(canonical/plain)={:.4}",
            canonical_total.as_secs_f64() / plain_total.as_secs_f64()
        );
        assert!(
            max_eig_diff < 1e-8,
            "canonical reordering must not change the recovered spectrum: max_diff={max_eig_diff:e}"
        );
    }
}
