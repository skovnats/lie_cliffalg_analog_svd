//! Core-flow SVD prototype.
//!
//! This module makes the two-sided SVD state explicit:
//!
//! ```text
//! core = U^T A V
//! ```
//!
//! The input matrix `A` is treated as fixed. The solver moves the two
//! orthogonal mirrors `U` and `V`, and the physical residual is the off-diagonal
//! field of `core`. This is close to what two-sided Jacobi/Kogbetliantz methods
//! do implicitly, but the state is exposed so later versions can attach
//! flow-inspired schedules, local metric probes, and analog phase constraints.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct LieSvdCoreFlowParams {
    pub max_sweeps: usize,
    pub tol: f64,
    pub min_pair_scale: f64,
    pub max_angle: f64,
    pub metric_feedback: f64,
    pub torsion_feedback: f64,
    /// Optional landmark/power warm-start for `U` and `V`.
    pub warm_start: Option<crate::lie_svd_topowarm::TopologicalWarmStartParams>,
    /// Calogero-Moser-like anti-clustering strength for nearly equal sigma estimates.
    /// Defaults to zero so existing runs are unchanged unless explicitly enabled.
    pub repel_lambda: f64,
    /// Small positive denominator stabilizer for the sigma repeller.
    pub repel_eps: f64,
    /// Relative off-diagonal residual threshold above which the repeller may act.
    pub repel_residual_threshold: f64,
    /// Backtracking attempts for monotone energy acceptance.
    pub line_search_steps: usize,
    /// Relative tolerance for accepting non-increasing energy steps.
    pub descent_tol: f64,
}

impl LieSvdCoreFlowParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_sweeps: if n <= 16 { 16 } else { 8 },
            tol: 1e-12,
            min_pair_scale: 1e-14,
            max_angle: 0.12,
            metric_feedback: 0.15,
            torsion_feedback: 0.10,
            warm_start: None,
            repel_lambda: 0.0,
            repel_eps: 1e-12,
            repel_residual_threshold: 1e-8,
            line_search_steps: 8,
            descent_tol: 1e-14,
        }
    }
}

impl Default for LieSvdCoreFlowParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdCoreFlowTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub layers: usize,
    pub rotations: usize,
    pub skipped_pairs: usize,
    pub rejected_pairs: usize,
    pub repeller_steps: usize,
    pub warm_start_used: bool,
    pub raw_offdiag: f64,
    pub warm_start_offdiag: f64,
    pub samples: Vec<f64>,
}

pub struct LieSvdCoreFlow;

impl LieSvdCoreFlow {
    /// Core-flow preconditioner plus digital polish.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let params = LieSvdCoreFlowParams::for_n(mat.nrows());
        Self::solve_with_trace(mat, params, params.max_sweeps / 4).0
    }

    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdCoreFlowParams,
        sample_every: usize,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdCoreFlowTrace) {
        let ((u0, _sigma0, vt0), trace) = Self::precondition_with_trace(mat, params, sample_every);
        let v0 = vt0.t().to_owned();
        let core = u0.t().dot(mat).dot(&v0);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&vt0);
        ((u, sigma, vt), trace)
    }

    /// Returns the approximate SVD from the core-flow rotor schedule alone.
    /// This is mainly useful for diagnostics; `solve` adds digital polish.
    pub fn precondition_with_trace(
        mat: &Array2<f64>,
        params: LieSvdCoreFlowParams,
        sample_every: usize,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdCoreFlowTrace) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdCoreFlow: matrix must be square");

        if n <= 4 {
            let (u, sigma, vt) = crate::lie_svd_micro::LieSvdMicro::solve(mat);
            return (
                (u, sigma, vt),
                LieSvdCoreFlowTrace {
                    initial_offdiag: 0.0,
                    final_offdiag: 0.0,
                    sweeps: 0,
                    layers: 0,
                    rotations: 0,
                    skipped_pairs: 0,
                    rejected_pairs: 0,
                    repeller_steps: 0,
                    warm_start_used: false,
                    raw_offdiag: 0.0,
                    warm_start_offdiag: 0.0,
                    samples: Vec::new(),
                },
            );
        }

        let raw_offdiag = offdiag_norm(mat);
        let (mut core, mut u_basis, mut v_basis, warm_start_used, warm_start_offdiag) =
            match params.warm_start {
                Some(warm_params) => {
                    let warm =
                        crate::lie_svd_topowarm::compute_topological_warm_start(mat, warm_params);
                    let accepted = warm.trace.accepted;
                    let warm_offdiag = warm.trace.warm_offdiag;
                    (warm.core, warm.u, warm.v, accepted, warm_offdiag)
                }
                None => (
                    mat.clone(),
                    Array2::<f64>::eye(n),
                    Array2::<f64>::eye(n),
                    false,
                    raw_offdiag,
                ),
            };
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let tol = params.tol * ref_norm.max(1.0);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let initial_offdiag = offdiag_norm(&core);
        let mut best_offdiag = initial_offdiag;
        let mut best_core = core.clone();
        let mut best_u = u_basis.clone();
        let mut best_v = v_basis.clone();
        let sample_every = sample_every.max(1);
        let layer_count = round_robin_layer_count(n);
        let mut samples = Vec::new();
        let mut sweeps = 0usize;
        let mut layers = 0usize;
        let mut rotations = 0usize;
        let mut skipped_pairs = 0usize;
        let mut rejected_pairs = 0usize;
        let mut repeller_steps = 0usize;

        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let energy = offdiag_norm(&core);
            if sweep % sample_every == 0 {
                samples.push(energy);
            }
            if energy <= best_offdiag {
                best_offdiag = energy;
                best_core = core.clone();
                best_u = u_basis.clone();
                best_v = v_basis.clone();
            }
            if energy <= tol || !energy.is_finite() {
                break;
            }
            let repeller_phase = params.repel_lambda > 0.0
                && energy > params.repel_residual_threshold * ref_norm.max(1.0);

            let mut changed = false;
            for layer in 0..layer_count {
                layers += 1;
                for (i, j) in layer_pairs(n, layer) {
                    if pair_offdiag(&core, i, j) <= pair_tol {
                        skipped_pairs += 1;
                        continue;
                    }
                    let target = core_pair_target(&core, i, j, &params, repeller_phase);
                    if target.theta_l.abs() + target.theta_r.abs() <= 1e-18 {
                        skipped_pairs += 1;
                        continue;
                    }
                    match accept_descent_rotor(
                        &mut core,
                        &mut u_basis,
                        &mut v_basis,
                        i,
                        j,
                        target.theta_l,
                        target.theta_r,
                        &params,
                    ) {
                        Some(accepted) => {
                            rotations += 1;
                            if target.repeller_active {
                                repeller_steps += 1;
                            }
                            if accepted < best_offdiag {
                                best_offdiag = accepted;
                                best_core = core.clone();
                                best_u = u_basis.clone();
                                best_v = v_basis.clone();
                            }
                            changed = true;
                        }
                        None => {
                            rejected_pairs += 1;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let final_energy = offdiag_norm(&core);
        if final_energy <= best_offdiag {
            best_offdiag = final_energy;
            best_core = core;
            best_u = u_basis;
            best_v = v_basis;
        }

        let (u, sigma, vt) = extract_sorted_svd(&best_core, &best_u, &best_v);
        (
            (u, sigma, vt),
            LieSvdCoreFlowTrace {
                initial_offdiag,
                final_offdiag: best_offdiag,
                sweeps,
                layers,
                rotations,
                skipped_pairs,
                rejected_pairs,
                repeller_steps,
                warm_start_used,
                raw_offdiag,
                warm_start_offdiag,
                samples,
            },
        )
    }
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn offdiag_norm(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s += a[[i, j]] * a[[i, j]];
            }
        }
    }
    s.sqrt()
}

fn pair_offdiag(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    work[[i, j]].abs() + work[[j, i]].abs()
}

fn row_dot(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    (0..work.ncols()).map(|k| work[[i, k]] * work[[j, k]]).sum()
}

fn col_dot(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    (0..work.nrows()).map(|k| work[[k, i]] * work[[k, j]]).sum()
}

fn row_norm_sq(work: &Array2<f64>, i: usize) -> f64 {
    (0..work.ncols()).map(|k| work[[i, k]] * work[[i, k]]).sum()
}

fn col_norm_sq(work: &Array2<f64>, i: usize) -> f64 {
    (0..work.nrows()).map(|k| work[[k, i]] * work[[k, i]]).sum()
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

fn local_pair_svd_angles(work: &Array2<f64>, i: usize, j: usize) -> (f64, f64) {
    let a = work[[i, i]];
    let b = work[[i, j]];
    let c = work[[j, i]];
    let d = work[[j, j]];
    let sum_angle = (-(b + c)).atan2(a - d);
    let diff_angle = (b - c).atan2(a + d);
    (
        wrap_angle(0.5 * (sum_angle + diff_angle)),
        wrap_angle(0.5 * (sum_angle - diff_angle)),
    )
}

fn metric_pair_angles(work: &Array2<f64>, i: usize, j: usize) -> (f64, f64) {
    let lij = row_dot(work, i, j);
    let rij = col_dot(work, i, j);
    let lii = row_norm_sq(work, i);
    let ljj = row_norm_sq(work, j);
    let rii = col_norm_sq(work, i);
    let rjj = col_norm_sq(work, j);
    (
        0.5 * (2.0 * lij).atan2(ljj - lii),
        0.5 * (2.0 * rij).atan2(rjj - rii),
    )
}

#[derive(Clone, Copy, Debug)]
struct CorePairTarget {
    theta_l: f64,
    theta_r: f64,
    repeller_active: bool,
}

fn core_pair_target(
    core: &Array2<f64>,
    i: usize,
    j: usize,
    params: &LieSvdCoreFlowParams,
    repeller_phase: bool,
) -> CorePairTarget {
    let (direct_l, direct_r) = local_pair_svd_angles(core, i, j);
    let (metric_l, metric_r) = metric_pair_angles(core, i, j);
    let local_scale =
        core[[i, i]].abs() + core[[j, j]].abs() + core[[i, j]].abs() + core[[j, i]].abs() + 1e-300;
    let torsion = 0.5 * (core[[i, j]] - core[[j, i]]).atan2(local_scale);
    let repeller = anti_cluster_repeller(core, i, j, params, local_scale, repeller_phase);
    let theta_l = wrap_angle(
        direct_l + params.metric_feedback * metric_l + params.torsion_feedback * torsion + repeller,
    );
    let theta_r = wrap_angle(
        direct_r + params.metric_feedback * metric_r - params.torsion_feedback * torsion - repeller,
    );
    CorePairTarget {
        theta_l: theta_l.clamp(-params.max_angle, params.max_angle),
        theta_r: theta_r.clamp(-params.max_angle, params.max_angle),
        repeller_active: repeller.abs() > 1e-18,
    }
}

fn anti_cluster_repeller(
    core: &Array2<f64>,
    i: usize,
    j: usize,
    params: &LieSvdCoreFlowParams,
    local_scale: f64,
    repeller_phase: bool,
) -> f64 {
    if !repeller_phase || params.repel_lambda <= 0.0 {
        return 0.0;
    }
    let sigma_i = core[[i, i]].abs();
    let sigma_j = core[[j, j]].abs();
    let gap = (sigma_i - sigma_j).abs();
    let eps = params.repel_eps.max(1e-300);
    let close_weight = eps / (gap * gap + eps);
    let coupling = (pair_offdiag(core, i, j) / local_scale).min(1.0);
    if close_weight * coupling <= 1e-12 {
        return 0.0;
    }
    let phase_hint = core[[i, j]] + core[[j, i]];
    let direction = if phase_hint.abs() > 1e-300 {
        phase_hint.signum()
    } else if (i + j) % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    direction * params.repel_lambda * close_weight * coupling
}

/// Calogero-Moser anti-clustering potential on singular-value estimates.
///
/// This uses the ordered-pair form from the design notes:
/// `lambda * sum_{i != j} 1 / ((sigma_i - sigma_j)^2 + eps)`.
/// Keep `lambda = 0` to disable it.
pub fn repeller_potential(sigma: &Array1<f64>, lambda: f64, eps: f64) -> f64 {
    if lambda <= 0.0 || sigma.len() < 2 {
        return 0.0;
    }
    let eps = eps.max(1e-300);
    let mut value = 0.0_f64;
    for i in 0..sigma.len() {
        for j in 0..sigma.len() {
            if i != j {
                let diff = sigma[i] - sigma[j];
                value += lambda / (diff * diff + eps);
            }
        }
    }
    value
}

/// Gradient of [`repeller_potential`] with respect to `sigma`.
pub fn repeller_gradient(sigma: &Array1<f64>, lambda: f64, eps: f64) -> Array1<f64> {
    let mut grad = Array1::<f64>::zeros(sigma.len());
    if lambda <= 0.0 || sigma.len() < 2 {
        return grad;
    }
    let eps = eps.max(1e-300);
    for i in 0..sigma.len() {
        for j in (i + 1)..sigma.len() {
            let diff = sigma[i] - sigma[j];
            let denom = diff * diff + eps;
            let g = -4.0 * lambda * diff / (denom * denom);
            grad[i] += g;
            grad[j] -= g;
        }
    }
    grad
}

fn round_robin_layer_count(n: usize) -> usize {
    if n % 2 == 0 {
        n.saturating_sub(1)
    } else {
        n
    }
}

fn layer_pairs(n: usize, layer: usize) -> Vec<(usize, usize)> {
    let m = if n % 2 == 0 { n } else { n + 1 };
    let ring = m - 1;
    let mut pairs = Vec::with_capacity(m / 2);
    for k in 0..(m / 2) {
        let (a, b) = if k == 0 {
            (m - 1, layer % ring)
        } else {
            ((layer + k) % ring, (layer + ring - k) % ring)
        };
        if a < n && b < n {
            pairs.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    pairs
}

fn accept_descent_rotor(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta_l: f64,
    theta_r: f64,
    params: &LieSvdCoreFlowParams,
) -> Option<f64> {
    let before = offdiag_norm(core);
    let accept_slack = params.descent_tol * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..params.line_search_steps.max(1) {
        let mut trial_core = core.clone();
        apply_left_rotor_to_core(&mut trial_core, i, j, theta_l * scale);
        apply_right_rotor_to_core(&mut trial_core, i, j, theta_r * scale);
        let after = offdiag_norm(&trial_core);
        if after <= before + accept_slack && after.is_finite() {
            *core = trial_core;
            apply_left_basis_rotor(u_basis, i, j, theta_l * scale);
            apply_right_basis_rotor(v_basis, i, j, theta_r * scale);
            return Some(after);
        }
        scale *= 0.5;
    }
    None
}

fn apply_left_rotor_to_core(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = work.nrows();
    let (s, c) = theta.sin_cos();
    for col in 0..n {
        let ai = work[[i, col]];
        let aj = work[[j, col]];
        work[[i, col]] = c * ai - s * aj;
        work[[j, col]] = s * ai + c * aj;
    }
}

fn apply_right_rotor_to_core(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = work.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let ai = work[[r, i]];
        let aj = work[[r, j]];
        work[[r, i]] = c * ai - s * aj;
        work[[r, j]] = s * ai + c * aj;
    }
}

fn apply_left_basis_rotor(u_basis: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = u_basis.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let ui = u_basis[[r, i]];
        let uj = u_basis[[r, j]];
        u_basis[[r, i]] = c * ui - s * uj;
        u_basis[[r, j]] = s * ui + c * uj;
    }
}

fn apply_right_basis_rotor(v_basis: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = v_basis.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let vi = v_basis[[r, i]];
        let vj = v_basis[[r, j]];
        v_basis[[r, i]] = c * vi - s * vj;
        v_basis[[r, j]] = s * vi + c * vj;
    }
}

fn extract_sorted_svd(
    work: &Array2<f64>,
    u_basis: &Array2<f64>,
    v_basis: &Array2<f64>,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = work.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        work[[b, b]]
            .abs()
            .partial_cmp(&work[[a, a]].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u_sorted = Array2::<f64>::zeros((n, n));
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    let mut sigma = Array1::<f64>::zeros(n);
    for (dst, &src) in order.iter().enumerate() {
        let d = work[[src, src]];
        sigma[dst] = d.abs();
        let sign = if d < 0.0 { -1.0 } else { 1.0 };
        for r in 0..n {
            u_sorted[[r, dst]] = sign * u_basis[[r, src]];
            v_sorted[[r, dst]] = v_basis[[r, src]];
        }
    }
    (u_sorted, sigma, v_sorted.t().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn metrics(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel_recon = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(a.nrows());
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel_recon, orth_u, orth_v)
    }

    #[test]
    fn test_coreflow_polished_random_8() {
        let n = 8;
        let mut rng = StdRng::seed_from_u64(1101);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let params = LieSvdCoreFlowParams {
            max_sweeps: 6,
            ..LieSvdCoreFlowParams::for_n(n)
        };
        let ((u, sigma, vt), trace) = LieSvdCoreFlow::solve_with_trace(&a, params, 2);
        let (rel, ou, ov) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.rotations > 0);
        assert!(rel < 1e-10, "rel={rel}");
        assert!(ou < 1e-8, "orth_u={ou}");
        assert!(ov < 1e-8, "orth_v={ov}");
    }

    #[test]
    fn test_coreflow_precondition_does_not_increase_energy_on_nearly_diagonal() {
        let n = 12;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 10.0 - i as f64 * 0.1;
            if i + 1 < n {
                a[[i, i + 1]] = 1e-5;
                a[[i + 1, i]] = -1e-5;
            }
        }
        let params = LieSvdCoreFlowParams {
            max_sweeps: 4,
            ..LieSvdCoreFlowParams::for_n(n)
        };
        let ((_u, _sigma, _vt), trace) = LieSvdCoreFlow::precondition_with_trace(&a, params, 1);
        assert!(
            trace.final_offdiag <= trace.initial_offdiag * 1.05,
            "initial={} final={}",
            trace.initial_offdiag,
            trace.final_offdiag
        );
    }

    #[test]
    fn test_soft_repeller_activates_on_coupled_near_cluster() {
        let mut core = Array2::<f64>::zeros((6, 6));
        core[[0, 0]] = 10.0;
        core[[1, 1]] = 10.0 + 1e-10;
        core[[0, 1]] = 0.2;
        core[[1, 0]] = -0.05;
        let params = LieSvdCoreFlowParams {
            repel_lambda: 0.08,
            repel_eps: 1e-4,
            repel_residual_threshold: 0.0,
            ..LieSvdCoreFlowParams::for_n(6)
        };
        let target = core_pair_target(&core, 0, 1, &params, true);
        assert!(target.repeller_active);
    }

    #[test]
    fn test_repeller_gradient_matches_finite_difference() {
        let sigma = Array1::from(vec![3.0, 1.2, 1.0]);
        let lambda = 0.02;
        let eps = 1e-5;
        let grad = repeller_gradient(&sigma, lambda, eps);
        let h = 1e-6;
        for k in 0..sigma.len() {
            let mut plus = sigma.clone();
            let mut minus = sigma.clone();
            plus[k] += h;
            minus[k] -= h;
            let numeric = (repeller_potential(&plus, lambda, eps)
                - repeller_potential(&minus, lambda, eps))
                / (2.0 * h);
            assert!(
                (numeric - grad[k]).abs() < 1e-5,
                "k={k} numeric={numeric} analytic={}",
                grad[k]
            );
        }
    }
}
