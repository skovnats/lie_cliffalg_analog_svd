//! Fractal row/column phase-health diagnostics.
//!
//! This module is the 0.10.0 "forest, not trees" layer. It keeps the matrix in
//! ordinary `f64`, but treats each row and each column as its own local
//! Clifford-like vector:
//!
//! ```text
//! row_i = scalar mass + vector spread + phase-delay bivector proxy
//! col_j = scalar mass + vector spread + phase-delay bivector proxy
//! ```
//!
//! A single vector does not contain an intrinsic bivector by itself; a bivector
//! appears after choosing a second direction. Here the second direction is a
//! deterministic one-step cyclic phase delay. The wedge energy
//! `||x wedge delay(x)||^2 = ||x||^2 ||delay(x)||^2 - <x,delay(x)>^2`
//! is a cheap proxy for internal row/column phase twist.

use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct VectorPhaseHealth {
    pub scalar_mean: f64,
    pub scalar_sq: f64,
    pub vector_sq: f64,
    pub delay_bivector_sq: f64,
    pub gradient_sq: f64,
    pub energy_entropy: f64,
    pub twist_ratio: f64,
    /// Deterministic one-step cyclic phase-delay angle, same construction as
    /// `lie_svd_phaseflow::axis_phase`'s `phase` field: orientation from the
    /// sign of the delay dot product, magnitude from
    /// `atan2(gradient, mean + spread)`. Used to build a global, mass-weighted
    /// phase angle for the whole matrix in `global_phase_invariants`.
    pub phase: f64,
}

/// Global (whole-matrix) scalars that don't live on a single row or column.
/// These sit next to `PhaseSignature`/`PhaseHealthSummary` rather than
/// replacing them: those are per-row/per-column diagnostics, these are
/// single numbers for the whole operator.
#[derive(Clone, Copy, Debug)]
pub struct GlobalPhaseInvariants {
    /// Mass-weighted circular mean of every row's and column's phase angle.
    /// A nonzero value flags a consistent global phase drift that a Layer-0
    /// pre-spin could center in one shot, rather than many small local
    /// corrections.
    pub global_phase: f64,
    /// `H_total = ||skew(A)||_F = ||(A - A^T) / 2||_F`, the raw torsion
    /// energy locked in pure rotation (as opposed to `real_chirality` in
    /// `lie_svd_engine`, which is this same quantity normalized by
    /// `||A||_F`).
    pub torsion_energy: f64,
    /// Self-dual vs. anti-self-dual bivector balance, aggregated over every
    /// contiguous `4x4` block. This reuses
    /// `lie_svd_block4::analyze_block4_signature` rather than recomputing
    /// the `SO(4)` split; it only covers `4 * (n/4)` rows/cols, so a
    /// remainder of up to 3 axes at the edge is not included.
    pub chirality_balance: f64,
    /// Normalized Shannon entropy of `|a_ij|^2 / ||A||_F^2` over the whole
    /// flattened matrix (in `[0, 1]`), distinct from the per-row/per-column
    /// entropy already in `PhaseHealthSummary`: this is one number for how
    /// spread out the entire operator's energy is, not how spread out any
    /// single row or column is.
    pub phase_entropy: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PhaseHealthSummary {
    pub count: usize,
    pub scalar_sq_sum: f64,
    pub vector_sq_sum: f64,
    pub delay_bivector_sq_sum: f64,
    pub gradient_sq_sum: f64,
    pub max_twist_ratio: f64,
    pub mean_twist_ratio: f64,
    pub mean_entropy: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FractalPhaseHealth {
    pub rows: PhaseHealthSummary,
    pub cols: PhaseHealthSummary,
    pub row_col_twist_gap: f64,
    pub row_col_entropy_gap: f64,
    pub total_phase_stress: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PhaseSignature {
    pub mean_stress: f64,
    pub max_twist: f64,
    pub causal_disbalance: f64,
    pub entropy_gap: f64,
    /// See `GlobalPhaseInvariants`: whole-matrix phase angle, torsion
    /// energy, self-dual/anti-self-dual chirality balance, and phase
    /// entropy, folded into the compact passport since `0.27.0`.
    pub global: GlobalPhaseInvariants,
}

pub fn analyze_fractal_phase_health(a: &Array2<f64>) -> FractalPhaseHealth {
    let n = a.nrows();
    let m = a.ncols();
    let mut row_items = Vec::with_capacity(n);
    let mut col_items = Vec::with_capacity(m);

    for i in 0..n {
        row_items.push(analyze_vector_phase_by(m, |j| a[[i, j]]));
    }
    for j in 0..m {
        col_items.push(analyze_vector_phase_by(n, |i| a[[i, j]]));
    }

    let rows = summarize(&row_items);
    let cols = summarize(&col_items);
    let row_col_twist_gap = (rows.mean_twist_ratio - cols.mean_twist_ratio).abs();
    let row_col_entropy_gap = (rows.mean_entropy - cols.mean_entropy).abs();
    let total_phase_stress = (rows.delay_bivector_sq_sum + cols.delay_bivector_sq_sum).sqrt()
        + (rows.gradient_sq_sum + cols.gradient_sq_sum).sqrt();

    FractalPhaseHealth {
        rows,
        cols,
        row_col_twist_gap,
        row_col_entropy_gap,
        total_phase_stress,
    }
}

pub fn phase_signature(a: &Array2<f64>) -> PhaseSignature {
    let health = analyze_fractal_phase_health(a);
    let count = (health.rows.count + health.cols.count).max(1) as f64;
    let causal_disbalance = triangular_disbalance(a);
    PhaseSignature {
        mean_stress: health.total_phase_stress / count,
        max_twist: health.rows.max_twist_ratio.max(health.cols.max_twist_ratio),
        causal_disbalance,
        entropy_gap: health.row_col_entropy_gap,
        global: global_phase_invariants(a),
    }
}

pub fn analyze_vector_phase(x: &[f64]) -> VectorPhaseHealth {
    analyze_vector_phase_by(x.len(), |i| x[i])
}

fn analyze_vector_phase_by<F>(n: usize, at: F) -> VectorPhaseHealth
where
    F: Fn(usize) -> f64,
{
    if n == 0 {
        return VectorPhaseHealth {
            scalar_mean: 0.0,
            scalar_sq: 0.0,
            vector_sq: 0.0,
            delay_bivector_sq: 0.0,
            gradient_sq: 0.0,
            energy_entropy: 0.0,
            twist_ratio: 0.0,
            phase: 0.0,
        };
    }

    let n_f = n as f64;
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += at(i);
    }
    let scalar_mean = sum / n_f;
    let scalar_sq = n_f * scalar_mean * scalar_mean;
    let mut vector_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    let mut delay_dot = 0.0_f64;
    let mut gradient_sq = 0.0_f64;
    let mut energy_sum = 0.0_f64;

    for i in 0..n {
        let xi = at(i);
        let x_next = at((i + 1) % n);
        let centered = xi - scalar_mean;
        vector_sq += centered * centered;
        norm_sq += xi * xi;
        delay_dot += xi * x_next;
        let d = x_next - xi;
        gradient_sq += d * d;
        energy_sum += xi * xi;
    }

    let delay_bivector_sq = (norm_sq * norm_sq - delay_dot * delay_dot).max(0.0);
    let energy_entropy = normalized_energy_entropy_by(n, energy_sum, at);
    let twist_ratio = delay_bivector_sq.sqrt() / norm_sq.max(1e-300);
    let orient = if delay_dot >= 0.0 { 1.0 } else { -1.0 };
    let phase = orient
        * gradient_sq
            .sqrt()
            .atan2(scalar_mean.abs() + vector_sq.sqrt() + 1e-300);

    VectorPhaseHealth {
        scalar_mean,
        scalar_sq,
        vector_sq,
        delay_bivector_sq,
        gradient_sq,
        energy_entropy,
        twist_ratio,
        phase,
    }
}

/// Whole-matrix scalars that don't reduce to any single row or column: a
/// mass-weighted global phase angle, the raw torsion (skew) energy, the
/// aggregated self-dual/anti-self-dual chirality balance, and the global
/// phase entropy of the entire operator.
pub fn global_phase_invariants(a: &Array2<f64>) -> GlobalPhaseInvariants {
    let n = a.nrows();
    let m = a.ncols();
    let mut sin_sum = 0.0_f64;
    let mut cos_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    for i in 0..n {
        let item = analyze_vector_phase_by(m, |j| a[[i, j]]);
        let w = (item.scalar_sq + item.vector_sq).sqrt();
        sin_sum += w * item.phase.sin();
        cos_sum += w * item.phase.cos();
        weight_sum += w;
    }
    for j in 0..m {
        let item = analyze_vector_phase_by(n, |i| a[[i, j]]);
        let w = (item.scalar_sq + item.vector_sq).sqrt();
        sin_sum += w * item.phase.sin();
        cos_sum += w * item.phase.cos();
        weight_sum += w;
    }
    let global_phase = if weight_sum > 1e-300 {
        sin_sum.atan2(cos_sum)
    } else {
        0.0
    };

    let corridor = n.min(m);
    let mut torsion_sq = 0.0_f64;
    for i in 0..corridor {
        for j in (i + 1)..corridor {
            let d = (a[[i, j]] - a[[j, i]]) * 0.5;
            torsion_sq += 2.0 * d * d;
        }
    }
    let torsion_energy = torsion_sq.sqrt();

    let chirality_balance = crate::lie_svd_block4::analyze_block4_signature(a).dual_balance;

    let total_sq: f64 = a.iter().map(|x| x * x).sum();
    let count = a.len();
    let phase_entropy = if count <= 1 || total_sq <= 1e-300 {
        0.0
    } else {
        let mut entropy = 0.0_f64;
        for &x in a.iter() {
            let p = (x * x) / total_sq;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy / (count as f64).ln()
    };

    GlobalPhaseInvariants {
        global_phase,
        torsion_energy,
        chirality_balance,
        phase_entropy,
    }
}

fn summarize(items: &[VectorPhaseHealth]) -> PhaseHealthSummary {
    if items.is_empty() {
        return PhaseHealthSummary {
            count: 0,
            scalar_sq_sum: 0.0,
            vector_sq_sum: 0.0,
            delay_bivector_sq_sum: 0.0,
            gradient_sq_sum: 0.0,
            max_twist_ratio: 0.0,
            mean_twist_ratio: 0.0,
            mean_entropy: 0.0,
        };
    }

    let mut out = PhaseHealthSummary {
        count: items.len(),
        scalar_sq_sum: 0.0,
        vector_sq_sum: 0.0,
        delay_bivector_sq_sum: 0.0,
        gradient_sq_sum: 0.0,
        max_twist_ratio: 0.0,
        mean_twist_ratio: 0.0,
        mean_entropy: 0.0,
    };
    for item in items {
        out.scalar_sq_sum += item.scalar_sq;
        out.vector_sq_sum += item.vector_sq;
        out.delay_bivector_sq_sum += item.delay_bivector_sq;
        out.gradient_sq_sum += item.gradient_sq;
        out.max_twist_ratio = out.max_twist_ratio.max(item.twist_ratio);
        out.mean_twist_ratio += item.twist_ratio;
        out.mean_entropy += item.energy_entropy;
    }
    let denom = items.len() as f64;
    out.mean_twist_ratio /= denom;
    out.mean_entropy /= denom;
    out
}

fn normalized_energy_entropy_by<F>(n: usize, energy_sum: f64, at: F) -> f64
where
    F: Fn(usize) -> f64,
{
    if n <= 1 || energy_sum <= 1e-300 {
        return 0.0;
    }
    let mut entropy = 0.0_f64;
    for i in 0..n {
        let v = at(i);
        let p = (v * v) / energy_sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy / (n as f64).ln()
}

fn triangular_disbalance(a: &Array2<f64>) -> f64 {
    let n = a.nrows().min(a.ncols());
    let mut upper = 0.0_f64;
    let mut lower = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            upper += a[[i, j]] * a[[i, j]];
            lower += a[[j, i]] * a[[j, i]];
        }
    }
    (upper - lower).abs() / (upper + lower).max(1e-300)
}

#[cfg(test)]
fn upper_shift_matrix(n: usize) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        a[[i, i]] = 1.0;
        if i + 1 < n {
            a[[i, i + 1]] = 3.0;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn constant_vector_has_no_internal_phase_twist() {
        let h = analyze_vector_phase(&[3.0, 3.0, 3.0, 3.0]);
        assert!(h.vector_sq < 1e-24);
        assert!(h.delay_bivector_sq < 1e-20);
        assert!(h.gradient_sq < 1e-24);
        assert!(h.twist_ratio < 1e-12);
    }

    #[test]
    fn alternating_vector_has_phase_twist() {
        let h = analyze_vector_phase(&[1.0, -1.0, 1.0, -1.0]);
        assert!(h.vector_sq > 0.0);
        assert!(h.gradient_sq > 0.0);
        assert!(h.energy_entropy > 0.99);
    }

    #[test]
    fn diagonal_matrix_has_low_row_column_entropy() {
        let a = Array2::from_diag(&ndarray::arr1(&[3.0, 2.0, 1.0, 0.5]));
        let h = analyze_fractal_phase_health(&a);
        assert!(h.rows.mean_entropy < 0.1);
        assert!(h.cols.mean_entropy < 0.1);
        assert!(h.row_col_twist_gap < 1e-12);
    }

    #[test]
    fn phase_signature_sees_causal_flow() {
        let sig = phase_signature(&upper_shift_matrix(12));
        assert!(sig.causal_disbalance > 0.99);
        assert!(sig.max_twist > 0.9);
    }

    #[test]
    fn phase_signature_accepts_rectangular_operators() {
        let a = Array2::from_shape_fn((9, 14), |(i, j)| ((i * 7 + j * 11 + 3) as f64).sin());
        let h = analyze_fractal_phase_health(&a);
        let sig = phase_signature(&a);
        assert_eq!(h.rows.count, 9);
        assert_eq!(h.cols.count, 14);
        assert!(sig.mean_stress.is_finite());
        assert!(sig.max_twist.is_finite());
        assert!(sig.entropy_gap.is_finite());
    }

    #[test]
    fn torsion_energy_matches_direct_skew_norm_on_causal_flow() {
        let a = upper_shift_matrix(12);
        let direct: f64 = {
            let skew = (&a - &a.t()).mapv(|x| x * 0.5);
            skew.mapv(|x| x * x).sum().sqrt()
        };
        let g = global_phase_invariants(&a);
        assert!(g.torsion_energy > 0.0);
        assert!(
            (g.torsion_energy - direct).abs() < 1e-9,
            "torsion_energy={} direct={}",
            g.torsion_energy,
            direct
        );
    }

    #[test]
    fn torsion_energy_is_zero_on_a_symmetric_matrix() {
        let a = Array2::from_shape_fn((6, 6), |(i, j)| {
            if i == j {
                2.0 + i as f64
            } else {
                0.1 * (i + j) as f64
            }
        });
        let g = global_phase_invariants(&a);
        assert!(
            g.torsion_energy < 1e-12,
            "torsion_energy={}",
            g.torsion_energy
        );
    }

    #[test]
    fn phase_entropy_is_lower_on_a_concentrated_diagonal_than_a_dense_matrix() {
        let diag = Array2::from_diag(&ndarray::arr1(&[4.0, 3.0, 2.0, 1.0, 0.5, 0.25]));
        let dense =
            Array2::from_shape_fn((6, 6), |(i, j)| 1.0 + 0.3 * ((i * 5 + j * 3) as f64).sin());
        let g_diag = global_phase_invariants(&diag);
        let g_dense = global_phase_invariants(&dense);
        assert!(g_diag.phase_entropy.is_finite());
        assert!(g_dense.phase_entropy.is_finite());
        assert!(
            g_diag.phase_entropy < g_dense.phase_entropy,
            "diag={} dense={}",
            g_diag.phase_entropy,
            g_dense.phase_entropy
        );
    }

    #[test]
    fn global_invariants_stay_finite_and_in_range_on_causal_flow() {
        let g = global_phase_invariants(&upper_shift_matrix(12));
        assert!(g.global_phase.is_finite());
        assert!((0.0..=1.0).contains(&g.chirality_balance));
        assert!((0.0..=1.0).contains(&g.phase_entropy));
    }
}
