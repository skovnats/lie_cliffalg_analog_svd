//! Quad-view energy audit and 2x2 Clifford coordinates.
//!
//! This module is the 0.9.0 "what are we actually killing?" layer. It does not
//! add another SVD route. It makes the energy decomposition explicit:
//!
//! ```text
//! A = diag(A) + sym_offdiag(A) + skew(A)
//! ```
//!
//! plus upper/lower triangular flow, row/column metric views, and the exact
//! 2x2 Clifford coordinates used by a local two-sided rotor.
//!
//! There are two different "four-view" ideas here:
//!
//! - Global row/column Clifford view: every row basis vector `e_i` and column
//!   basis vector `f_j` is its own imaginary unit, so
//!   `A = sum_ij a_ij e_i tensor f_j`.
//! - Local rotor coordinates: one `2x2` block has four scalar coordinates
//!   `(E, F, G, H)` used to compute an exact pair rotor.
//!
//! They are connected, but they are not the same object. The global view is
//! captured by [`GlobalQuadView`]; the local view by [`CliffordBlock2`].

use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct QuadEnergy {
    pub n: usize,
    pub total_sq: f64,
    pub diag_sq: f64,
    pub offdiag_sq: f64,
    pub sym_offdiag_sq: f64,
    pub skew_sq: f64,
    pub upper_sq: f64,
    pub lower_sq: f64,
    pub row_metric_offdiag_sq: f64,
    pub col_metric_offdiag_sq: f64,
    pub trace_projection: f64,
    pub triangular_imbalance: f64,
    pub dual_mismatch_sq: f64,
    pub quad_spread: f64,
}

impl QuadEnergy {
    pub fn direct_balance_error(&self) -> f64 {
        (self.total_sq - self.diag_sq - self.offdiag_sq).abs()
    }

    pub fn offdiag_split_error(&self) -> f64 {
        (self.offdiag_sq - self.sym_offdiag_sq - self.skew_sq).abs()
    }

    pub fn offdiag_ratio(&self) -> f64 {
        self.offdiag_sq.sqrt() / self.total_sq.sqrt().max(1e-300)
    }

    pub fn diag_ratio(&self) -> f64 {
        self.diag_sq.sqrt() / self.total_sq.sqrt().max(1e-300)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalQuadView {
    /// Direct row-column tensor energy: `A = sum_ij a_ij e_i tensor f_j`.
    pub primal_sq: f64,
    /// Off-diagonal tension in the direct tensor view.
    pub primal_offdiag_sq: f64,
    /// Row-dual metric tension: off-diagonal energy of `A A^T`.
    pub row_dual_offdiag_sq: f64,
    /// Column-dual metric tension: off-diagonal energy of `A^T A`.
    pub col_dual_offdiag_sq: f64,
    /// Disagreement between row-dual and column-dual metric tensors.
    pub dual_mismatch_sq: f64,
    /// Root-sum global disagreement across all four views.
    pub quad_spread: f64,
}

impl GlobalQuadView {
    pub fn normalized_spread(&self) -> f64 {
        self.quad_spread / self.primal_sq.sqrt().max(1e-300)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CliffordBlock2 {
    /// Scalar part, `(a_ii + a_jj) / 2`.
    pub e_scalar: f64,
    /// Diagonal vector part, `(a_ii - a_jj) / 2`.
    pub f_diag_vector: f64,
    /// Symmetric off-diagonal vector part, `(a_ij + a_ji) / 2`.
    pub g_sym_vector: f64,
    /// Bivector/torsion part for basis `e12 -> [[0, 1], [-1, 0]]`,
    /// `(a_ij - a_ji) / 2`.
    pub h_bivector: f64,
}

impl CliffordBlock2 {
    pub fn from_entries(p: f64, q: f64, r: f64, w: f64) -> Self {
        Self {
            e_scalar: 0.5 * (p + w),
            f_diag_vector: 0.5 * (p - w),
            g_sym_vector: 0.5 * (q + r),
            h_bivector: 0.5 * (q - r),
        }
    }

    pub fn from_pair(a: &Array2<f64>, i: usize, j: usize) -> Self {
        Self::from_entries(a[[i, i]], a[[i, j]], a[[j, i]], a[[j, j]])
    }

    pub fn reconstruct_entries(self) -> (f64, f64, f64, f64) {
        (
            self.e_scalar + self.f_diag_vector,
            self.g_sym_vector + self.h_bivector,
            self.g_sym_vector - self.h_bivector,
            self.e_scalar - self.f_diag_vector,
        )
    }

    /// Exact local two-sided rotor angles for this `2x2` block.
    ///
    /// These are the same closed-form angles used by the tiny SVD kernels, but
    /// written in the four Clifford coordinates. All four coordinates matter:
    /// dropping `f_diag_vector` loses the diagonal gap and produces wrong
    /// angles for generic blocks.
    pub fn two_sided_angles(self) -> (f64, f64) {
        let alpha_shape = (-self.g_sym_vector).atan2(self.f_diag_vector);
        let alpha_torsion = self.h_bivector.atan2(self.e_scalar);
        (
            wrap_angle(0.5 * (alpha_shape + alpha_torsion)),
            wrap_angle(0.5 * (alpha_shape - alpha_torsion)),
        )
    }

    pub fn offdiag_sq(self) -> f64 {
        let (_p, q, r, _w) = self.reconstruct_entries();
        q * q + r * r
    }
}

pub fn analyze_quad_energy(a: &Array2<f64>) -> QuadEnergy {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "quad energy expects square matrix");
    let mut total_sq = 0.0_f64;
    let mut diag_sq = 0.0_f64;
    let mut offdiag_sq = 0.0_f64;
    let mut sym_offdiag_sq = 0.0_f64;
    let mut skew_sq = 0.0_f64;
    let mut upper_sq = 0.0_f64;
    let mut lower_sq = 0.0_f64;
    let mut trace_projection = 0.0_f64;

    for i in 0..n {
        let d = a[[i, i]];
        total_sq += d * d;
        diag_sq += d * d;
        trace_projection += d.abs();
        for j in (i + 1)..n {
            let upper = a[[i, j]];
            let lower = a[[j, i]];
            total_sq += upper * upper + lower * lower;
            offdiag_sq += upper * upper + lower * lower;
            upper_sq += upper * upper;
            lower_sq += lower * lower;
            let sym = 0.5 * (upper + lower);
            let skew = 0.5 * (upper - lower);
            sym_offdiag_sq += 2.0 * sym * sym;
            skew_sq += 2.0 * skew * skew;
        }
    }

    let row_metric_offdiag_sq = gram_offdiag_sq(a, false);
    let col_metric_offdiag_sq = gram_offdiag_sq(a, true);
    let dual_mismatch_sq = dual_metric_mismatch_sq(a);
    let quad_spread =
        (offdiag_sq + row_metric_offdiag_sq + col_metric_offdiag_sq + dual_mismatch_sq).sqrt();
    let triangular_imbalance = (upper_sq - lower_sq) / (upper_sq + lower_sq).sqrt().max(1e-300);

    QuadEnergy {
        n,
        total_sq,
        diag_sq,
        offdiag_sq,
        sym_offdiag_sq,
        skew_sq,
        upper_sq,
        lower_sq,
        row_metric_offdiag_sq,
        col_metric_offdiag_sq,
        trace_projection,
        triangular_imbalance,
        dual_mismatch_sq,
        quad_spread,
    }
}

pub fn analyze_global_quad_view(a: &Array2<f64>) -> GlobalQuadView {
    let q = analyze_quad_energy(a);
    GlobalQuadView {
        primal_sq: q.total_sq,
        primal_offdiag_sq: q.offdiag_sq,
        row_dual_offdiag_sq: q.row_metric_offdiag_sq,
        col_dual_offdiag_sq: q.col_metric_offdiag_sq,
        dual_mismatch_sq: q.dual_mismatch_sq,
        quad_spread: q.quad_spread,
    }
}

pub fn apply_block_angles(block: CliffordBlock2, theta_l: f64, theta_r: f64) -> CliffordBlock2 {
    let (mut p, mut q, mut r, mut w) = block.reconstruct_entries();
    let (sl, cl) = theta_l.sin_cos();
    let rp = cl * p - sl * r;
    let rq = cl * q - sl * w;
    let rr = sl * p + cl * r;
    let rw = sl * q + cl * w;
    p = rp;
    q = rq;
    r = rr;
    w = rw;

    let (sr, cr) = theta_r.sin_cos();
    let pp = cr * p - sr * q;
    let qq = sr * p + cr * q;
    let rr = cr * r - sr * w;
    let ww = sr * r + cr * w;
    CliffordBlock2::from_entries(pp, qq, rr, ww)
}

fn gram_offdiag_sq(a: &Array2<f64>, columns: bool) -> f64 {
    let n = a.nrows();
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0_f64;
            for k in 0..n {
                dot += if columns {
                    a[[k, i]] * a[[k, j]]
                } else {
                    a[[i, k]] * a[[j, k]]
                };
            }
            acc += 2.0 * dot * dot;
        }
    }
    acc
}

fn dual_metric_mismatch_sq(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut acc = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut row_metric = 0.0_f64;
            let mut col_metric = 0.0_f64;
            for k in 0..n {
                row_metric += a[[i, k]] * a[[j, k]];
                col_metric += a[[k, i]] * a[[k, j]];
            }
            let d = row_metric - col_metric;
            acc += d * d;
        }
    }
    acc
}

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::FRAC_PI_2 {
        angle -= std::f64::consts::PI;
    }
    while angle < -std::f64::consts::FRAC_PI_2 {
        angle += std::f64::consts::PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_energy_splits_frobenius_energy() {
        let a =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0])
                .expect("matrix");
        let q = analyze_quad_energy(&a);
        assert!(q.direct_balance_error() < 1e-12);
        assert!(q.offdiag_split_error() < 1e-12);
        assert!(q.quad_spread.is_finite());
        assert!((q.upper_sq - (2.0_f64 * 2.0 + 3.0_f64 * 3.0 + 6.0_f64 * 6.0)).abs() < 1e-12);
        assert!((q.lower_sq - (4.0_f64 * 4.0 + 7.0_f64 * 7.0 + 8.0_f64 * 8.0)).abs() < 1e-12);
    }

    #[test]
    fn global_quad_view_collapses_on_diagonal_matrix() {
        let a = Array2::from_diag(&ndarray::arr1(&[3.0, 2.0, 1.0]));
        let view = analyze_global_quad_view(&a);
        assert!(view.primal_offdiag_sq < 1e-24);
        assert!(view.row_dual_offdiag_sq < 1e-24);
        assert!(view.col_dual_offdiag_sq < 1e-24);
        assert!(view.dual_mismatch_sq < 1e-24);
        assert!(view.quad_spread < 1e-12);
    }

    #[test]
    fn clifford_block_reconstructs_entries() {
        let b = CliffordBlock2::from_entries(1.2, -0.7, 2.3, -4.0);
        let (p, q, r, w) = b.reconstruct_entries();
        assert!((p - 1.2).abs() < 1e-12);
        assert!((q + 0.7).abs() < 1e-12);
        assert!((r - 2.3).abs() < 1e-12);
        assert!((w + 4.0).abs() < 1e-12);
    }

    #[test]
    fn clifford_angles_annihilate_generic_2x2_offdiag() {
        let cases = [
            (1.0, 2.0, -0.5, 3.0),
            (-2.0, 0.75, 1.25, 0.4),
            (0.2, -3.0, 4.0, -1.0),
            (5.0, 1e-3, -2e-3, 2.0),
        ];
        for (p, q, r, w) in cases {
            let block = CliffordBlock2::from_entries(p, q, r, w);
            let (theta_l, theta_r) = block.two_sided_angles();
            let out = apply_block_angles(block, theta_l, theta_r);
            assert!(
                out.offdiag_sq() < 1e-24,
                "offdiag_sq={:.3e} block={:?} angles=({:.3e},{:.3e})",
                out.offdiag_sq(),
                block,
                theta_l,
                theta_r
            );
        }
    }
}
