//! Adaptive dispatcher and lightweight matrix triage.
//!
//! This module is the 0.6.0 "synergy" layer: it does not replace the individual
//! solvers, it decides when their geometric views are worth paying for.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveRoute {
    Micro,
    Small,
    PhaseFlow,
    CoreFlowTopo,
    Hybrid,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveTriage {
    pub n: usize,
    pub offdiag_ratio: f64,
    pub diagonal_dominance: f64,
    pub row_cv: f64,
    pub col_cv: f64,
    pub row_col_mismatch: f64,
    pub symmetry_ratio: f64,
    pub transpose_torsion_ratio: f64,
    pub diagonal_entropy: f64,
    pub row_mass_entropy: f64,
    pub col_mass_entropy: f64,
    pub phase_mean_stress: f64,
    pub phase_max_twist: f64,
    pub phase_causal_disbalance: f64,
    pub phase_entropy_gap: f64,
    /// `H_total = ||skew(A)||_F` from `global_phase_invariants` (0.27.0).
    pub phase_torsion_energy: f64,
    /// Self-dual/anti-self-dual bivector balance from `global_phase_invariants`
    /// (0.27.0). Empirically a cleaner discriminator for directed/causal
    /// torsion than `phase_entropy` turned out to be: on the calibration set
    /// (nearly-diagonal, uniform-random, block-structured, causal-Jordan)
    /// it was ~0 on every non-causal case and clearly nonzero (`~0.38`) on
    /// the causal-Jordan case, so `should_use_phaseflow` uses it directly.
    pub phase_chirality_balance: f64,
    /// Whole-matrix phase entropy from `global_phase_invariants` (0.27.0).
    /// Exposed for visibility, but deliberately *not* used to gate routing:
    /// calibration showed the causal-Jordan test case (`~0.52`) sits almost
    /// as low as the nearly-diagonal case (`~0.49`), because both are sparse
    /// band matrices — low whole-matrix entropy does not reliably mean "safe
    /// to skip geometric routes" the way it first looked like it would.
    pub phase_entropy: f64,
    pub suspicious_score: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveTrace {
    pub route: AdaptiveRoute,
    pub triage: AdaptiveTriage,
}

#[derive(Clone, Copy, Debug)]
pub struct LieSvdAdaptiveParams {
    pub small_max_n: usize,
    pub coreflow_topo_max_n: usize,
    pub phaseflow_max_n: usize,
    pub nearly_diagonal_ratio: f64,
    pub suspicious_threshold: f64,
}

impl Default for LieSvdAdaptiveParams {
    fn default() -> Self {
        Self {
            small_max_n: 512,
            coreflow_topo_max_n: 64,
            phaseflow_max_n: 64,
            nearly_diagonal_ratio: 1e-5,
            suspicious_threshold: 1.15,
        }
    }
}

pub struct LieSvdAdaptive;

impl LieSvdAdaptive {
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        Self::solve_with_trace(mat, LieSvdAdaptiveParams::default()).0
    }

    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdAdaptiveParams,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), AdaptiveTrace) {
        let route = choose_route(mat, params);
        let result = match route.route {
            AdaptiveRoute::Micro => crate::lie_svd_micro::LieSvdMicro::solve(mat),
            AdaptiveRoute::Small => crate::lie_svd_small::LieSvdSmall::solve(mat),
            AdaptiveRoute::PhaseFlow => solve_phaseflow_route(mat, route.triage),
            AdaptiveRoute::CoreFlowTopo => solve_coreflow_topo(mat, route.triage),
            AdaptiveRoute::Hybrid => crate::lie_svd_hybrid::LieSvdHybrid::solve(mat),
        };
        (result, route)
    }
}

pub fn choose_route(mat: &Array2<f64>, params: LieSvdAdaptiveParams) -> AdaptiveTrace {
    let triage = analyze_matrix(mat);
    let route = if triage.n <= crate::lie_svd::MICRO_MAX_N {
        AdaptiveRoute::Micro
    } else if triage.offdiag_ratio <= params.nearly_diagonal_ratio {
        AdaptiveRoute::Small
    } else if triage.n <= params.phaseflow_max_n && should_use_phaseflow(triage, params) {
        AdaptiveRoute::PhaseFlow
    } else if triage.n <= params.coreflow_topo_max_n && should_use_coreflow_topo(triage, params) {
        AdaptiveRoute::CoreFlowTopo
    } else if triage.n <= params.small_max_n {
        AdaptiveRoute::Small
    } else {
        AdaptiveRoute::Hybrid
    };
    AdaptiveTrace { route, triage }
}

fn should_use_phaseflow(triage: AdaptiveTriage, params: LieSvdAdaptiveParams) -> bool {
    if is_balanced_degenerate_case(triage, params) {
        return true;
    }
    let strong_causal_flow = triage.phase_causal_disbalance > 0.88
        && triage.offdiag_ratio > 0.20
        && triage.phase_mean_stress > 1.0;
    let high_phase_locking_stress = triage.phase_mean_stress > 50.0
        && triage.phase_max_twist > 0.92
        && triage.row_cv < 0.42
        && triage.col_cv < 0.42
        && triage.row_col_mismatch < 0.30
        && triage.offdiag_ratio > 0.85
        && triage.diagonal_dominance < 0.35;
    // Self-dual/anti-self-dual chirality is a different lens on directed
    // torsion than `phase_causal_disbalance` (triangular upper/lower energy
    // split): it catches `4x4`-scale rotational asymmetry directly. The
    // `diagonal_dominance` guard matters: `sparse_structured` has real
    // structured skew energy (a banded, asymmetric-but-bidirectional
    // pattern) and tripped this trigger without it, sending an
    // already-machine-precision diagonally-dominant case through PhaseFlow
    // for a 100x slowdown with no accuracy gain. Requiring low diagonal
    // dominance keeps this trigger scoped to genuinely off-diagonal-heavy,
    // directed-torsion cases like causal/Jordan flow.
    let strong_chirality_torsion = triage.phase_chirality_balance > 0.30
        && triage.offdiag_ratio > 0.20
        && triage.phase_torsion_energy > 1.0
        && triage.diagonal_dominance < 0.5;
    strong_causal_flow || high_phase_locking_stress || strong_chirality_torsion
}

fn is_symmetric_topological_case(triage: AdaptiveTriage) -> bool {
    triage.symmetry_ratio > 0.96 && triage.offdiag_ratio > 0.35 && triage.diagonal_dominance < 0.95
}

fn should_use_coreflow_topo(triage: AdaptiveTriage, params: LieSvdAdaptiveParams) -> bool {
    if is_symmetric_topological_case(triage) {
        return true;
    }
    if is_balanced_degenerate_case(triage, params) {
        return true;
    }
    let has_cluster_mass = triage.row_mass_entropy < 0.92 || triage.col_mass_entropy < 0.92;
    let has_balanced_views = triage.row_col_mismatch < 0.22;
    let avoids_random_torsion =
        triage.transpose_torsion_ratio < 0.62 || triage.diagonal_entropy < 0.62;
    let not_rank_one_like = triage.row_cv < 0.42 && triage.col_cv < 0.42;
    triage.suspicious_score >= params.suspicious_threshold
        && has_cluster_mass
        && has_balanced_views
        && avoids_random_torsion
        && not_rank_one_like
}

fn is_balanced_degenerate_case(triage: AdaptiveTriage, params: LieSvdAdaptiveParams) -> bool {
    triage.suspicious_score >= params.suspicious_threshold + 0.25
        && triage.offdiag_ratio > 0.90
        && triage.diagonal_dominance < 0.25
        && triage.row_cv >= 0.20
        && triage.col_cv >= 0.20
        && triage.row_cv < 0.45
        && triage.col_cv < 0.45
        && triage.row_col_mismatch < 0.45
}

pub fn analyze_matrix(mat: &Array2<f64>) -> AdaptiveTriage {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "adaptive SVD expects square matrices");
    let mut all_sq = 0.0_f64;
    let mut diag_sq = 0.0_f64;
    let mut off_sq = 0.0_f64;
    let mut skew_sq = 0.0_f64;
    let mut sym_sq = 0.0_f64;
    let mut diag_abs = vec![0.0_f64; n];
    let mut row_l1 = vec![0.0_f64; n];
    let mut col_l1 = vec![0.0_f64; n];
    let mut row_l2 = vec![0.0_f64; n];
    let mut col_l2 = vec![0.0_f64; n];

    for i in 0..n {
        for j in 0..n {
            let v = mat[[i, j]];
            let av = v.abs();
            all_sq += v * v;
            row_l1[i] += av;
            col_l1[j] += av;
            row_l2[i] += v * v;
            col_l2[j] += v * v;
            if i == j {
                diag_sq += v * v;
                diag_abs[i] = av;
            } else {
                off_sq += v * v;
            }
        }
    }
    for i in 0..n {
        row_l2[i] = row_l2[i].sqrt();
        col_l2[i] = col_l2[i].sqrt();
        for j in (i + 1)..n {
            let a = mat[[i, j]];
            let b = mat[[j, i]];
            let sym = 0.5 * (a + b);
            let skew = 0.5 * (a - b);
            sym_sq += 2.0 * sym * sym;
            skew_sq += 2.0 * skew * skew;
        }
    }

    let norm = all_sq.sqrt().max(1e-300);
    let offdiag_ratio = off_sq.sqrt() / norm;
    let diagonal_dominance = diag_sq.sqrt() / norm;
    let row_cv = coeff_var(&row_l2);
    let col_cv = coeff_var(&col_l2);
    let row_col_mismatch = relative_l2_mismatch(&row_l1, &col_l1);
    let symmetry_ratio = sym_sq.sqrt() / (sym_sq + skew_sq).sqrt().max(1e-300);
    let transpose_torsion_ratio = skew_sq.sqrt() / (sym_sq + skew_sq).sqrt().max(1e-300);
    let diagonal_entropy = entropy01(&diag_abs);
    let row_mass_entropy = entropy01(&row_l1);
    let col_mass_entropy = entropy01(&col_l1);
    let phase = crate::lie_svd_phasehealth::phase_signature(mat);
    let suspicious_score = suspicious_score(
        offdiag_ratio,
        diagonal_dominance,
        row_cv,
        col_cv,
        row_col_mismatch,
        transpose_torsion_ratio,
        symmetry_ratio,
        diagonal_entropy,
    );

    AdaptiveTriage {
        n,
        offdiag_ratio,
        diagonal_dominance,
        row_cv,
        col_cv,
        row_col_mismatch,
        symmetry_ratio,
        transpose_torsion_ratio,
        diagonal_entropy,
        row_mass_entropy,
        col_mass_entropy,
        phase_mean_stress: phase.mean_stress,
        phase_max_twist: phase.max_twist,
        phase_causal_disbalance: phase.causal_disbalance,
        phase_entropy_gap: phase.entropy_gap,
        phase_torsion_energy: phase.global.torsion_energy,
        phase_chirality_balance: phase.global.chirality_balance,
        phase_entropy: phase.global.phase_entropy,
        suspicious_score,
    }
}

fn solve_phaseflow_route(
    mat: &Array2<f64>,
    triage: AdaptiveTriage,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    let mut params = crate::lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(n);
    if triage.phase_chirality_balance > 0.30 {
        // Strong self-dual/anti-self-dual imbalance is directed torsion, not
        // symmetric standing-wave noise. Lean into Causal Anti-Spin even if
        // the separate triangular `causal_bias` metric alone would have
        // sat just under its own default threshold.
        params.use_causal_antispin = true;
        params.causal_antispin_threshold = params.causal_antispin_threshold.min(0.5);
        params.causal_antispin_layers = params.causal_antispin_layers.max(3);
    }
    crate::lie_svd_phaseflow::LieSvdPhaseFlow::solve_with_digital_polish(mat, params).0
}

fn solve_coreflow_topo(
    mat: &Array2<f64>,
    triage: AdaptiveTriage,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    let mut params = crate::lie_svd_coreflow::LieSvdCoreFlowParams::for_n(n);
    params.max_sweeps = if n <= 64 { 8 } else { 4 };
    params.repel_lambda = if triage.diagonal_entropy < 0.72 || triage.row_col_mismatch > 0.08 {
        0.02
    } else {
        0.008
    };
    params.repel_eps = 1e-8;
    let mut warm = crate::lie_svd_topowarm::TopologicalWarmStartParams::for_n(n);
    warm.rank = n.min(if triage.suspicious_score > 1.8 { 12 } else { 8 });
    warm.power_steps = if triage.transpose_torsion_ratio > 0.55 {
        1
    } else {
        2
    };
    warm.graph_relax_steps = 2;
    params.warm_start = Some(warm);
    crate::lie_svd_coreflow::LieSvdCoreFlow::solve_with_trace(mat, params, 16).0
}

fn suspicious_score(
    offdiag_ratio: f64,
    diagonal_dominance: f64,
    row_cv: f64,
    col_cv: f64,
    row_col_mismatch: f64,
    torsion: f64,
    symmetry_ratio: f64,
    entropy: f64,
) -> f64 {
    let mut score = 0.0_f64;
    score += (offdiag_ratio - 0.55).max(0.0) * 0.9;
    score += (0.35 - diagonal_dominance).max(0.0) * 0.8;
    score += row_cv.max(col_cv) * 1.4;
    score += row_col_mismatch * 1.6;
    score += torsion * 0.45;
    score += offdiag_ratio * (symmetry_ratio - 0.82).max(0.0) * 1.8;
    score += (0.78 - entropy).max(0.0) * 1.2;
    score
}

#[cfg(test)]
fn structured_stress_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let block = if i / 6 == j / 6 { 4.0 } else { 0.02 };
        block + (((i + j) * 31) as f64).sin() * 1e-3
    })
}

fn coeff_var(v: &[f64]) -> f64 {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    if mean.abs() <= 1e-300 {
        return 0.0;
    }
    let var = v
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / v.len().max(1) as f64;
    var.sqrt() / mean.abs()
}

fn relative_l2_mismatch(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += x * x + y * y;
    }
    num.sqrt() / den.sqrt().max(1e-300)
}

fn entropy01(v: &[f64]) -> f64 {
    let sum = v.iter().sum::<f64>();
    if sum <= 1e-300 || v.len() <= 1 {
        return 1.0;
    }
    let mut h = 0.0_f64;
    for x in v {
        let p = *x / sum;
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h / (v.len() as f64).ln().max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn rel_recon(a: &Array2<f64>, u: &Array2<f64>, s: &Array1<f64>, vt: &Array2<f64>) -> f64 {
        let sigma = Array2::from_diag(s);
        let recon = u.dot(&sigma).dot(vt);
        let num = (&recon - a).mapv(|x| x * x).sum().sqrt();
        let den = a.mapv(|x| x * x).sum().sqrt().max(1e-300);
        num / den
    }

    #[test]
    fn test_adaptive_keeps_nearly_diagonal_on_small_path() {
        let n = 16;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 2.0 - i as f64 / n as f64;
        }
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::Small);
    }

    #[test]
    fn test_adaptive_coreflow_topo_route_for_structured_stress() {
        let n = 24;
        let a = structured_stress_matrix(n);
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::CoreFlowTopo);
    }

    #[test]
    fn test_adaptive_keeps_uniform_random_on_small_path() {
        let n = 24;
        let a = Array2::from_shape_fn((n, n), |(i, j)| ((i * 13 + j * 7 + 3) as f64).sin());
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::Small);
    }

    #[test]
    fn test_adaptive_keeps_seeded_random_on_small_path() {
        let n = 32;
        let mut state = 17u64;
        let a = Array2::from_shape_fn((n, n), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            2.0 * unit - 1.0
        });
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::Small);
    }

    #[test]
    fn test_adaptive_balanced_degenerate_trigger() {
        let triage = AdaptiveTriage {
            n: 32,
            offdiag_ratio: 0.98,
            diagonal_dominance: 0.17,
            row_cv: 0.25,
            col_cv: 0.25,
            row_col_mismatch: 0.22,
            symmetry_ratio: 0.67,
            transpose_torsion_ratio: 0.74,
            diagonal_entropy: 0.94,
            row_mass_entropy: 0.99,
            col_mass_entropy: 0.99,
            phase_mean_stress: 10.0,
            phase_max_twist: 0.98,
            phase_causal_disbalance: 0.1,
            phase_entropy_gap: 0.02,
            phase_torsion_energy: 5.0,
            phase_chirality_balance: 0.05,
            phase_entropy: 0.8,
            suspicious_score: 1.58,
        };
        assert!(super::is_balanced_degenerate_case(
            triage,
            LieSvdAdaptiveParams::default()
        ));
    }

    #[test]
    fn test_adaptive_chirality_balance_triggers_phaseflow() {
        // A triage with negligible triangular causal disbalance and mild
        // phase-locking stress, but strong self-dual/anti-self-dual
        // chirality: should_use_phaseflow must catch this via the new
        // chirality-based trigger, not the existing causal/stress ones.
        let triage = AdaptiveTriage {
            n: 32,
            offdiag_ratio: 0.40,
            diagonal_dominance: 0.30,
            row_cv: 0.30,
            col_cv: 0.30,
            row_col_mismatch: 0.10,
            symmetry_ratio: 0.60,
            transpose_torsion_ratio: 0.40,
            diagonal_entropy: 0.70,
            row_mass_entropy: 0.80,
            col_mass_entropy: 0.80,
            phase_mean_stress: 5.0,
            phase_max_twist: 0.50,
            phase_causal_disbalance: 0.10,
            phase_entropy_gap: 0.05,
            phase_torsion_energy: 4.0,
            phase_chirality_balance: 0.45,
            phase_entropy: 0.6,
            suspicious_score: 0.5,
        };
        assert!(super::should_use_phaseflow(
            triage,
            LieSvdAdaptiveParams::default()
        ));
        let low_chirality = AdaptiveTriage {
            phase_chirality_balance: 0.05,
            ..triage
        };
        assert!(!super::should_use_phaseflow(
            low_chirality,
            LieSvdAdaptiveParams::default()
        ));
    }

    #[test]
    fn test_adaptive_solve_random_smoke() {
        let n = 10;
        let a = Array2::from_shape_fn((n, n), |(i, j)| ((i * 13 + j * 7) as f64).sin());
        let ((u, s, vt), _trace) =
            LieSvdAdaptive::solve_with_trace(&a, LieSvdAdaptiveParams::default());
        assert!(rel_recon(&a, &u, &s, &vt) < 1e-10);
    }

    #[test]
    fn test_adaptive_phaseflow_route_for_causal_jordan_flow() {
        let n = 24;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0;
            if i + 1 < n {
                a[[i, i + 1]] = 5.0;
            }
        }
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::PhaseFlow);
    }

    #[test]
    fn test_adaptive_keeps_large_causal_flow_on_small_until_batch_phaseflow() {
        let n = 128;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0;
            if i + 1 < n {
                a[[i, i + 1]] = 5.0;
            }
        }
        let trace = choose_route(&a, LieSvdAdaptiveParams::default());
        assert_eq!(trace.route, AdaptiveRoute::Small);
    }
}
