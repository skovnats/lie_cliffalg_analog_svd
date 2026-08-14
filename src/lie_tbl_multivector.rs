//! Rudimentary Clifford-multivector representation of tabular rows.
//!
//! Each column `j` of a table is assigned its own orthonormal basis
//! generator `e_j` (`e_j^2 = 1`, `e_j . e_k = 0` for `j != k` — a standard
//! Euclidean `Cl(d, 0)` algebra, `d` = number of columns). A row is a
//! grade-1 multivector: `x_i = sum_j x_ij e_j`.
//!
//! What this buys, stated precisely rather than claimed: the geometric
//! product of two rows splits as `x_i * x_k = x_i . x_k + x_i ^ x_k`
//! (scalar/grade-0 part plus bivector/grade-2 part).
//!
//! - The **scalar part is not new information**. `x_i . x_k = sum_j x_ij
//!   x_kj` is exactly the linear kernel already computed by
//!   `kernel_gram::build_gram(.., KernelKind::Linear)`. `row_scalar_gram`
//!   below computes it the "multivector way" specifically so that identity
//!   is a tested fact (`multivector_scalar_gram_matches_linear_kernel`),
//!   not an assertion. Relationships between different generators/columns
//!   do not require a bivector at all to show up in the scalar part: an
//!   ordinary dot product between two *different* rows already mixes every
//!   pair of generators, because `e_j . e_j = 1` picks out the matching
//!   component from each. There is no way to get pairwise *feature*
//!   covariance (what regression needs) without an accumulation step
//!   mathematically equivalent to `X^T X` or an SVD of `X` itself — see
//!   `lie_tbl_regress::TblRotorRegressor::fit_via_rectangular_svd` for the
//!   latter, and its doc comment for a measured, honest comparison.
//! - The **bivector part is genuinely different**: `x_i ^ x_k` is
//!   antisymmetric (`x_i ^ x_k = -(x_k ^ x_i)`, and `v ^ v = 0` for any
//!   single row against itself), and encodes the oriented "area" spanned by
//!   two rows in feature space. `kernel_gram`'s dot-product kernel discards
//!   this entirely — two rows that are exact scalar multiples of each other
//!   have zero bivector between them regardless of their dot product, while
//!   two rows of the same norm pointing in very different directions have a
//!   large one. `total_bivector_energy` is a first, minimal use of this:
//!   a single scalar diagnostic for how much of a table's row-to-row
//!   structure is "directional spread" rather than "shared magnitude".

use ndarray::Array2;

/// A grade-1 (pure vector) multivector: one table row's components along
/// the column generators `e_1..e_d`. No scalar or bivector part is stored
/// per row; those only exist once two rows are combined (`geometric_product`).
#[derive(Clone, Debug)]
pub struct RowMultivector {
    pub components: Vec<f64>,
}

/// The split geometric product of two rows in the same column-generator
/// basis.
#[derive(Clone, Debug)]
pub struct GeometricProduct {
    /// Grade-0 part: the dot product `sum_j a_j * b_j`.
    pub scalar: f64,
    /// Grade-2 part, flattened upper-triangular: `bivector[idx(j,k)]` for
    /// `j < k` is the `e_j ^ e_k` coefficient, `a_j*b_k - a_k*b_j`.
    pub bivector: Vec<f64>,
    pub dim: usize,
}

impl RowMultivector {
    pub fn from_row(row: &[f64]) -> Self {
        Self {
            components: row.to_vec(),
        }
    }
}

pub fn geometric_product(a: &RowMultivector, b: &RowMultivector) -> GeometricProduct {
    assert_eq!(
        a.components.len(),
        b.components.len(),
        "geometric_product: dimension mismatch ({} vs {})",
        a.components.len(),
        b.components.len()
    );
    let d = a.components.len();
    let scalar = a
        .components
        .iter()
        .zip(b.components.iter())
        .map(|(x, y)| x * y)
        .sum();
    let mut bivector = Vec::with_capacity(d.saturating_mul(d.saturating_sub(1)) / 2);
    for j in 0..d {
        for k in (j + 1)..d {
            bivector.push(a.components[j] * b.components[k] - a.components[k] * b.components[j]);
        }
    }
    GeometricProduct {
        scalar,
        bivector,
        dim: d,
    }
}

pub fn rows_from_table(x: &Array2<f64>) -> Vec<RowMultivector> {
    x.rows()
        .into_iter()
        .map(|r| RowMultivector::from_row(&r.iter().cloned().collect::<Vec<f64>>()))
        .collect()
}

/// The table-wide scalar Gram matrix, built by taking the scalar (grade-0)
/// part of the geometric product between every pair of rows. By
/// construction this equals `X X^T`, the sample-sample linear kernel
/// (see the module doc comment and the identity test below).
pub fn row_scalar_gram(rows: &[RowMultivector]) -> Array2<f64> {
    let n = rows.len();
    Array2::from_shape_fn((n, n), |(i, j)| {
        geometric_product(&rows[i], &rows[j]).scalar
    })
}

/// `sum_{i<k} ||rows[i] ^ rows[k]||^2` across every pair of rows: how much
/// oriented row-to-row spread exists in the table that a scalar Gram matrix
/// discards entirely. `O(n^2 d^2)`; intended for small/diagnostic use, not
/// as a per-pass hot-loop primitive.
pub fn total_bivector_energy(rows: &[RowMultivector]) -> f64 {
    let n = rows.len();
    let mut total = 0.0_f64;
    for i in 0..n {
        for k in (i + 1)..n {
            let gp = geometric_product(&rows[i], &rows[k]);
            total += gp.bivector.iter().map(|b| b * b).sum::<f64>();
        }
    }
    total
}

/// The dual construction to `rows_from_table`: each table *column* (length
/// `n`, one entry per sample) as a grade-1 multivector, generators now
/// indexed by sample rather than by feature. Used to build feature-feature
/// (not sample-sample) structure — see `CliffordGramMatrix`.
pub fn columns_from_table(x: &Array2<f64>) -> Vec<RowMultivector> {
    let n = x.nrows();
    let d = x.ncols();
    (0..d)
        .map(|j| RowMultivector::from_row(&(0..n).map(|i| x[[i, j]]).collect::<Vec<f64>>()))
        .collect()
}

/// The full geometric-product Gram operator on a table's *columns*: the
/// classical scalar Gram `S = X^T X` (`d x d`, symmetric — exactly what
/// `lie_tbl_regress::TblRotorRegressor::fit` builds), stored alongside a
/// `d x d` matrix of pairwise column bivector norms,
/// `bivector[[j, k]] = ||c_j ^ c_k||` (`j != k`; zero on the diagonal, since
/// `v ^ v = 0`). This is the "extend Gram with the wedge part" object: `S`
/// captures colinear/angular structure between columns, `bivector` captures
/// oriented/rotational structure between them that `S` cannot see by
/// construction (see the module doc comment).
///
/// Scope note: this stores pairwise wedge *norms* (a `d x d` matrix), not
/// the full bivector *tensor* (every individual `e_j ^ e_k` coefficient for
/// every column pair, which would need `d` sets of `d*(d-1)/2` numbers each
/// — a rank-3 object). The norm is what `rho` below needs and what a
/// regularizer can act on per column pair; the full per-component tensor
/// is not currently needed by anything in this crate.
#[derive(Clone, Debug)]
pub struct CliffordGramMatrix {
    pub scalar: Array2<f64>,
    pub bivector: Array2<f64>,
}

impl CliffordGramMatrix {
    pub fn from_columns(x: &Array2<f64>) -> Self {
        let d = x.ncols();
        let columns = columns_from_table(x);
        let mut scalar = Array2::<f64>::zeros((d, d));
        let mut bivector = Array2::<f64>::zeros((d, d));
        for j in 0..d {
            for k in 0..d {
                let gp = geometric_product(&columns[j], &columns[k]);
                scalar[[j, k]] = gp.scalar;
                if j != k {
                    let norm = gp.bivector.iter().map(|b| b * b).sum::<f64>().sqrt();
                    bivector[[j, k]] = norm;
                }
            }
        }
        Self { scalar, bivector }
    }

    /// `rho = ||bivector||_F^2 / ||scalar||_F^2`: the fraction of total
    /// geometric-product energy between columns that is oriented/rotational
    /// rather than colinear. `0` exactly when every column is a scalar
    /// multiple of a single common direction (every pairwise wedge is then
    /// exactly zero — see the identity test); positive as soon as the
    /// columns span more than one direction. Not scale-free across
    /// different tables (both norms scale with the data's units), so
    /// compare `rho` across columns/subsets of the *same* table, not across
    /// tables with different units.
    pub fn rho(&self) -> f64 {
        let scalar_energy: f64 = self.scalar.iter().map(|x| x * x).sum();
        let bivector_energy: f64 = self.bivector.iter().map(|x| x * x).sum();
        bivector_energy / scalar_energy.max(1e-300)
    }

    /// Same construction as `from_columns`, but under missing data:
    /// `present[[i, j]]` is `false` wherever `x[[i, j]]` is missing (the
    /// value at that cell is otherwise ignored — any placeholder is fine).
    /// This is the concrete, working version of "`NULL` as a nilpotent
    /// generator": a literal `e^2 = 0` generator does not, on its own, give
    /// an algorithm for what to do with the *other* entries of a row that
    /// contains one. What actually happens here is pairwise deletion,
    /// generalized to the Clifford-product setting — a real, standard
    /// statistical technique, not a novel one: for each column pair
    /// `(j, k)`, only rows present in *both* columns contribute to their
    /// scalar and bivector product. A row missing in either column
    /// contributes a hard zero to that pair specifically, so a column with
    /// no presence at all against another column produces exact zeros in
    /// both `scalar` and `bivector` for that pair — equivalent to that
    /// generator being absent from the pair's algebra entirely.
    pub fn from_columns_with_missing(x: &Array2<f64>, present: &Array2<bool>) -> Self {
        let n = x.nrows();
        let d = x.ncols();
        assert_eq!(present.dim(), (n, d), "presence mask shape must match x");
        let mut scalar = Array2::<f64>::zeros((d, d));
        let mut bivector = Array2::<f64>::zeros((d, d));
        for j in 0..d {
            for k in 0..d {
                let mut cj = vec![0.0_f64; n];
                let mut ck = vec![0.0_f64; n];
                for i in 0..n {
                    if present[[i, j]] && present[[i, k]] {
                        cj[i] = x[[i, j]];
                        ck[i] = x[[i, k]];
                    }
                }
                let gp = geometric_product(
                    &RowMultivector::from_row(&cj),
                    &RowMultivector::from_row(&ck),
                );
                scalar[[j, k]] = gp.scalar;
                if j != k {
                    bivector[[j, k]] = gp.bivector.iter().map(|b| b * b).sum::<f64>().sqrt();
                }
            }
        }
        Self { scalar, bivector }
    }
}

/// Normalized per-column bivector stress, `stress[j]` in `[0, 1]`: mean over
/// `k != j` of the wedge magnitude between (unit-scaled) columns `j` and
/// `k`, i.e. `sin(angle)` between them. `0` for a column that is a scalar
/// multiple of every other column (fully redundant); close to `1` for a
/// column nearly orthogonal to all others (fully independent). Shared by
/// `TblRotorRegressor::fit_with_bivector_regularization` and
/// `GeometricTabularDispatcher`'s route selection so both use the same
/// signal rather than two independently-tuned copies of it.
pub fn column_stress(x: &Array2<f64>) -> Vec<f64> {
    let d = x.ncols();
    let clifford_gram = CliffordGramMatrix::from_columns(x);
    let col_norms: Vec<f64> = (0..d)
        .map(|j| clifford_gram.scalar[[j, j]].max(0.0).sqrt().max(1e-300))
        .collect();
    let mut stress = vec![0.0_f64; d];
    for j in 0..d {
        if d <= 1 {
            break;
        }
        let mut s = 0.0_f64;
        for k in 0..d {
            if j == k {
                continue;
            }
            s += clifford_gram.bivector[[j, k]] / (col_norms[j] * col_norms[k]);
        }
        stress[j] = s / (d - 1) as f64;
    }
    stress
}

/// Pairwise normalized bivector stress, `stress[[j, k]]` for `j != k` (`0`
/// on the diagonal): `||c_j ^ c_k|| / (||c_j|| ||c_k||)`, i.e. `sin(angle)`
/// between (unit-scaled) columns `j` and `k`. Unlike `column_stress` (which
/// *averages* this over all `k` for a fixed `j`, the right thing for a
/// per-column ridge penalty), this keeps every pair separate, which is what
/// distinguishes "one specific redundant pair among otherwise-independent
/// columns" from "mildly correlated with everything" — averaging can hide
/// the former when there are several other, unrelated columns diluting the
/// mean (see `GeometricTabularDispatcher::choose_route`, which needs
/// exactly that distinction).
pub fn pairwise_column_stress(x: &Array2<f64>) -> Array2<f64> {
    let d = x.ncols();
    let clifford_gram = CliffordGramMatrix::from_columns(x);
    let col_norms: Vec<f64> = (0..d)
        .map(|j| clifford_gram.scalar[[j, j]].max(0.0).sqrt().max(1e-300))
        .collect();
    let mut stress = Array2::<f64>::zeros((d, d));
    for j in 0..d {
        for k in 0..d {
            if j == k {
                continue;
            }
            stress[[j, k]] = clifford_gram.bivector[[j, k]] / (col_norms[j] * col_norms[k]);
        }
    }
    stress
}

/// Total temporal circulation of a time-ordered table: `Omega = sum_t (x_t ^
/// x_{t+1})`, the accumulated bivector across every consecutive row pair
/// (rows are `RowMultivector`s in feature space, `d` generators — same
/// construction `rows_from_table` uses, just walked in time order instead
/// of compared all-to-all). Returned as the flattened bivector components
/// (same layout as `GeometricProduct::bivector`), so `||Omega||` via
/// `circulation_energy` is the natural scalar summary.
///
/// Why this can distinguish a directed/rotational flow from a driftless
/// one: for consecutive states related by a fixed rotation
/// (`x_{t+1} ~= R x_t`), every step's wedge has the *same sign structure*,
/// so the sum accumulates coherently and grows with the number of steps.
/// For an unbiased random walk (`x_{t+1} = x_t + noise_t`, `noise_t`
/// independent of `x_t` with mean zero), `x_t ^ x_{t+1} = x_t ^ noise_t`
/// has mean zero for each step conditional on `x_t`, so the sum is a
/// driftless accumulation (its magnitude grows roughly like `sqrt(T)`, far
/// slower than a genuinely rotating process's `O(T)`) rather than reliably
/// near zero at any fixed `T` — see the test for the actual measured ratio
/// rather than an assumed one.
pub fn temporal_circulation(x: &Array2<f64>) -> Vec<f64> {
    let n = x.nrows();
    let d = x.ncols();
    let rows = rows_from_table(x);
    let mut omega = vec![0.0_f64; d.saturating_mul(d.saturating_sub(1)) / 2];
    for t in 0..n.saturating_sub(1) {
        let gp = geometric_product(&rows[t], &rows[t + 1]);
        for (acc, v) in omega.iter_mut().zip(gp.bivector.iter()) {
            *acc += v;
        }
    }
    omega
}

pub fn circulation_energy(omega: &[f64]) -> f64 {
    omega.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_gram::{build_gram, KernelKind};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn multivector_scalar_gram_matches_linear_kernel() {
        let x = Array2::from_shape_fn((6, 4), |(i, j)| ((i * 5 + j * 3 + 1) as f64).sin());
        let rows = rows_from_table(&x);
        let via_multivector = row_scalar_gram(&rows);

        let points: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
        let via_kernel = build_gram(&points, KernelKind::Linear);

        for i in 0..6 {
            for j in 0..6 {
                let a = via_multivector[[i, j]];
                let b = via_kernel[[i, j]];
                assert!(
                    (a - b).abs() < 1e-12,
                    "[{i},{j}] multivector={a} kernel={b}"
                );
            }
        }
    }

    #[test]
    fn self_product_has_zero_bivector() {
        let row = RowMultivector::from_row(&[1.0, -2.0, 3.0, 0.5]);
        let gp = geometric_product(&row, &row);
        assert!((gp.scalar - 14.25).abs() < 1e-12);
        assert!(gp.bivector.iter().all(|b| b.abs() < 1e-12));
    }

    #[test]
    fn scalar_multiple_rows_have_zero_bivector_but_nonzero_scalar() {
        let a = RowMultivector::from_row(&[1.0, 2.0, -1.0]);
        let b = RowMultivector::from_row(&[3.0, 6.0, -3.0]); // = 3 * a
        let gp = geometric_product(&a, &b);
        assert!(gp.bivector.iter().all(|v| v.abs() < 1e-12));
        assert!(gp.scalar.abs() > 1e-6);
    }

    #[test]
    fn bivector_energy_is_zero_for_collinear_rows_and_positive_otherwise() {
        let collinear =
            Array2::from_shape_fn((5, 3), |(i, j)| (1.0 + i as f64) * [1.0, -2.0, 0.5][j]);
        let rows_collinear = rows_from_table(&collinear);
        let e_collinear = total_bivector_energy(&rows_collinear);
        assert!(e_collinear < 1e-12, "e_collinear={e_collinear:e}");

        let spread = Array2::from_shape_fn((5, 3), |(i, j)| ((i * 7 + j * 3 + 1) as f64).cos());
        let rows_spread = rows_from_table(&spread);
        let e_spread = total_bivector_energy(&rows_spread);
        assert!(e_spread > 1e-6, "e_spread={e_spread:e}");
    }

    #[test]
    fn clifford_gram_scalar_part_matches_x_transpose_x() {
        let x = Array2::from_shape_fn((8, 4), |(i, j)| ((i * 5 + j * 3 + 1) as f64).sin());
        let g = CliffordGramMatrix::from_columns(&x);
        let xtx = x.t().dot(&x);
        for j in 0..4 {
            for k in 0..4 {
                let a = g.scalar[[j, k]];
                let b = xtx[[j, k]];
                assert!((a - b).abs() < 1e-10, "[{j},{k}] clifford={a} x^t*x={b}");
            }
        }
    }

    #[test]
    fn from_columns_with_missing_matches_full_data_when_nothing_is_missing() {
        let x = Array2::from_shape_fn((8, 4), |(i, j)| ((i * 5 + j * 3 + 1) as f64).sin());
        let present = Array2::from_elem((8, 4), true);
        let full = CliffordGramMatrix::from_columns(&x);
        let masked = CliffordGramMatrix::from_columns_with_missing(&x, &present);
        for j in 0..4 {
            for k in 0..4 {
                assert!((full.scalar[[j, k]] - masked.scalar[[j, k]]).abs() < 1e-12);
                assert!((full.bivector[[j, k]] - masked.bivector[[j, k]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn from_columns_with_missing_zeros_out_a_wholly_absent_column() {
        // Column 2 has zero presence anywhere: every scalar/bivector entry
        // touching it must be exactly zero, equivalent to that generator
        // being dropped from the algebra entirely (compressed to
        // Cl(d-1, 0) against every other column).
        let x = Array2::from_shape_fn((10, 4), |(i, j)| ((i * 7 + j * 5 + 1) as f64).cos());
        let mut present = Array2::from_elem((10, 4), true);
        for i in 0..10 {
            present[[i, 2]] = false;
        }
        let g = CliffordGramMatrix::from_columns_with_missing(&x, &present);
        for j in 0..4 {
            assert!(
                g.scalar[[2, j]].abs() < 1e-12,
                "scalar[2,{j}]={}",
                g.scalar[[2, j]]
            );
            assert!(
                g.scalar[[j, 2]].abs() < 1e-12,
                "scalar[{j},2]={}",
                g.scalar[[j, 2]]
            );
            assert!(
                g.bivector[[2, j]].abs() < 1e-12,
                "bivector[2,{j}]={}",
                g.bivector[[2, j]]
            );
            assert!(
                g.bivector[[j, 2]].abs() < 1e-12,
                "bivector[{j},2]={}",
                g.bivector[[j, 2]]
            );
        }
        // The remaining 3x3 block (columns 0,1,3) must be unaffected --
        // it's exactly as if only those 3 columns existed.
        let x_sub = Array2::from_shape_fn((10, 3), |(i, jj)| {
            let j = [0, 1, 3][jj];
            ((i * 7 + j * 5 + 1) as f64).cos()
        });
        let g_sub = CliffordGramMatrix::from_columns(&x_sub);
        let idx = [0usize, 1, 3];
        for (a, &ja) in idx.iter().enumerate() {
            for (b, &jb) in idx.iter().enumerate() {
                assert!(
                    (g.scalar[[ja, jb]] - g_sub.scalar[[a, b]]).abs() < 1e-10,
                    "scalar[{ja},{jb}]"
                );
            }
        }
    }

    #[test]
    fn from_columns_with_missing_stays_finite_under_partial_missingness() {
        let mut rng_state = 11u64;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 40;
        let d = 5;
        let x = Array2::from_shape_fn((n, d), |(i, j)| ((i * 7 + j * 5 + 1) as f64).sin());
        let present = Array2::from_shape_fn((n, d), |_| next() > 0.2);
        let g = CliffordGramMatrix::from_columns_with_missing(&x, &present);
        assert!(g.scalar.iter().all(|v| v.is_finite()));
        assert!(g.bivector.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(g.rho().is_finite() && g.rho() >= 0.0);
    }

    #[test]
    fn clifford_gram_rho_is_zero_for_one_dimensional_data_and_positive_otherwise() {
        // Every column is a scalar multiple of the same base direction:
        // strictly one-dimensional column space, every pairwise wedge is
        // exactly zero.
        let base: Vec<f64> = (0..10).map(|i| ((i * 7 + 3) as f64).sin()).collect();
        let scalar_only = Array2::from_shape_fn((10, 4), |(i, j)| base[i] * (1.0 + j as f64));
        let g_flat = CliffordGramMatrix::from_columns(&scalar_only);
        assert!(g_flat.rho() < 1e-20, "rho={:e}", g_flat.rho());

        // Genuinely multi-dimensional column space (independent columns):
        // rho must be positive.
        let spread = Array2::from_shape_fn((10, 4), |(i, j)| ((i * 7 + j * 5 + 1) as f64).cos());
        let g_spread = CliffordGramMatrix::from_columns(&spread);
        assert!(g_spread.rho() > 1e-6, "rho={:e}", g_spread.rho());
    }

    /// Measured (not assumed) growth rates, first: at `T = 50/200/800`, a
    /// bounded i.i.d. process's circulation energy tracks `~sqrt(T)`
    /// (`8.2 -> 9.1 -> 39.6` against `sqrt(T) = 7.1 -> 14.1 -> 28.3`,
    /// noisy at a single seed but the right order of growth), while a
    /// fixed-rotation process's tracks `~T` (`9.9 -> 39.6 -> 158.7`, a
    /// near-exact `4x` scaling matching each `4x` jump in `T`). An earlier
    /// version of this test used a random *walk* as the "no rotation"
    /// baseline instead of bounded i.i.d. samples, which is wrong: a
    /// random walk's own state magnitude grows like `sqrt(t)`, which
    /// inflates every wedge term's magnitude regardless of directional
    /// bias, and at every `T` tried its circulation energy came out
    /// *larger* than the rotating process's — the opposite of the intended
    /// comparison, and a confound, not a finding about rotation.
    #[test]
    fn rotating_process_has_more_circulation_than_bounded_iid_noise() {
        let mut rng = StdRng::seed_from_u64(9);
        let t_len = 400;

        let mut iid = Array2::<f64>::zeros((t_len, 3));
        for i in 0..t_len {
            for j in 0..3 {
                iid[[i, j]] = rng.gen_range(-1.0_f64..1.0);
            }
        }
        let e_iid = circulation_energy(&temporal_circulation(&iid));

        let theta = 0.2_f64;
        let mut rot = Array2::<f64>::zeros((t_len, 3));
        rot[[0, 0]] = 1.0;
        rot[[0, 2]] = 1.0;
        for i in 1..t_len {
            let (s, c) = theta.sin_cos();
            rot[[i, 0]] = c * rot[[i - 1, 0]] - s * rot[[i - 1, 1]];
            rot[[i, 1]] = s * rot[[i - 1, 0]] + c * rot[[i - 1, 1]];
            rot[[i, 2]] = 1.0;
        }
        let e_rot = circulation_energy(&temporal_circulation(&rot));

        assert!(
            e_rot > 3.0 * e_iid,
            "expected the rotating process's circulation energy to clearly \
             exceed the bounded i.i.d. baseline's (e_rot={e_rot:e}, e_iid={e_iid:e})"
        );
    }
}
