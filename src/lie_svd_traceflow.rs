//! Trace/Procrustes SVD navigator.
//!
//! This module is the 0.8.0 "inverse Rubik" view. Instead of saying "destroy
//! the off-diagonal entries of `A`", it starts from the identity bases and
//! rotates the two orthogonal mirrors so that
//!
//! ```text
//! core = U^T A V
//! ```
//!
//! gains as much signed diagonal projection as possible. With column sign
//! choices folded into `U`, the local objective is
//!
//! ```text
//! trace_projection(core) = sum_i abs(core_ii)
//! ```
//!
//! This is the computational face of the von-Neumann/Ky-Fan trace principle:
//! the global maximum over orthogonal `U,V` is `sum_i sigma_i`. The module does
//! not bypass SVD; it exposes the equivalent Procrustes viewpoint as a guarded
//! local rotor schedule and then uses the robust digital polish path.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct LieSvdTraceFlowParams {
    pub max_sweeps: usize,
    pub tol: f64,
    pub min_pair_scale: f64,
    pub line_search_steps: usize,
    pub ascent_tol: f64,
}

impl LieSvdTraceFlowParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_sweeps: if n <= 32 { 12 } else { 6 },
            tol: 1e-12,
            min_pair_scale: 1e-14,
            line_search_steps: 8,
            ascent_tol: 1e-14,
        }
    }
}

impl Default for LieSvdTraceFlowParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdTraceFlowTrace {
    pub initial_projection: f64,
    pub final_projection: f64,
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub layers: usize,
    pub rotations: usize,
    pub skipped_pairs: usize,
    pub rejected_pairs: usize,
    pub plateau_pairs: usize,
    pub samples: Vec<(f64, f64)>,
}

pub struct LieSvdTraceFlow;

impl LieSvdTraceFlow {
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let params = LieSvdTraceFlowParams::for_n(mat.nrows());
        Self::solve_with_trace(mat, params, params.max_sweeps / 4).0
    }

    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdTraceFlowParams,
        sample_every: usize,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdTraceFlowTrace,
    ) {
        let ((u0, _sigma0, vt0), trace) = Self::precondition_with_trace(mat, params, sample_every);
        let v0 = vt0.t().to_owned();
        let core = u0.t().dot(mat).dot(&v0);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&vt0);
        ((u, sigma, vt), trace)
    }

    pub fn precondition_with_trace(
        mat: &Array2<f64>,
        params: LieSvdTraceFlowParams,
        sample_every: usize,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdTraceFlowTrace,
    ) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdTraceFlow: matrix must be square");
        if n <= 4 {
            let (u, sigma, vt) = crate::lie_svd_micro::LieSvdMicro::solve(mat);
            let core = u.t().dot(mat).dot(&vt.t());
            let projection = trace_projection(&core);
            let offdiag = offdiag_norm(&core);
            return (
                (u, sigma, vt),
                LieSvdTraceFlowTrace {
                    initial_projection: projection,
                    final_projection: projection,
                    initial_offdiag: offdiag,
                    final_offdiag: offdiag,
                    sweeps: 0,
                    layers: 0,
                    rotations: 0,
                    skipped_pairs: 0,
                    rejected_pairs: 0,
                    plateau_pairs: 0,
                    samples: Vec::new(),
                },
            );
        }

        let mut core = mat.clone();
        let mut u_basis = Array2::<f64>::eye(n);
        let mut v_basis = Array2::<f64>::eye(n);
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let projection_tol = params.tol * ref_norm.max(1.0);
        let sample_every = sample_every.max(1);
        let initial_projection = trace_projection(&core);
        let initial_offdiag = offdiag_norm(&core);
        let mut best_projection = initial_projection;
        let mut best_core = core.clone();
        let mut best_u = u_basis.clone();
        let mut best_v = v_basis.clone();
        let mut samples = Vec::new();
        let mut sweeps = 0usize;
        let mut layers = 0usize;
        let mut rotations = 0usize;
        let mut skipped_pairs = 0usize;
        let mut rejected_pairs = 0usize;
        let mut plateau_pairs = 0usize;
        let layer_count = round_robin_layer_count(n);

        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let projection = trace_projection(&core);
            let offdiag = offdiag_norm(&core);
            if sweep % sample_every == 0 {
                samples.push((projection, offdiag));
            }
            if projection > best_projection {
                best_projection = projection;
                best_core = core.clone();
                best_u = u_basis.clone();
                best_v = v_basis.clone();
            }
            if offdiag <= projection_tol || !projection.is_finite() || !offdiag.is_finite() {
                break;
            }

            let mut changed = false;
            for layer in 0..layer_count {
                layers += 1;
                for (i, j) in layer_pairs(n, layer) {
                    if pair_offdiag(&core, i, j) <= pair_tol {
                        skipped_pairs += 1;
                        continue;
                    }
                    let (theta_l, theta_r) = local_pair_svd_angles(&core, i, j);
                    if theta_l.abs() + theta_r.abs() <= 1e-18 {
                        plateau_pairs += 1;
                        continue;
                    }
                    match accept_trace_rotor(
                        &mut core,
                        &mut u_basis,
                        &mut v_basis,
                        i,
                        j,
                        theta_l,
                        theta_r,
                        &params,
                    ) {
                        Some(proj) => {
                            rotations += 1;
                            if proj > best_projection {
                                best_projection = proj;
                                best_core = core.clone();
                                best_u = u_basis.clone();
                                best_v = v_basis.clone();
                            }
                            changed = true;
                        }
                        None => rejected_pairs += 1,
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let final_projection = trace_projection(&core);
        if final_projection >= best_projection {
            best_projection = final_projection;
            best_core = core;
            best_u = u_basis;
            best_v = v_basis;
        }
        let final_offdiag = offdiag_norm(&best_core);
        let (u, sigma, vt) = extract_sorted_svd(&best_core, &best_u, &best_v);
        (
            (u, sigma, vt),
            LieSvdTraceFlowTrace {
                initial_projection,
                final_projection: best_projection,
                initial_offdiag,
                final_offdiag,
                sweeps,
                layers,
                rotations,
                skipped_pairs,
                rejected_pairs,
                plateau_pairs,
                samples,
            },
        )
    }
}

pub fn trace_projection(core: &Array2<f64>) -> f64 {
    let n = core.nrows().min(core.ncols());
    (0..n).map(|i| core[[i, i]].abs()).sum()
}

pub fn offdiag_norm(a: &Array2<f64>) -> f64 {
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

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn pair_offdiag(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    work[[i, j]].abs() + work[[j, i]].abs()
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

fn accept_trace_rotor(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta_l: f64,
    theta_r: f64,
    params: &LieSvdTraceFlowParams,
) -> Option<f64> {
    let before = trace_projection(core);
    let accept_slack = params.ascent_tol * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..params.line_search_steps.max(1) {
        let mut trial_core = core.clone();
        apply_left_rotor_to_core(&mut trial_core, i, j, theta_l * scale);
        apply_right_rotor_to_core(&mut trial_core, i, j, theta_r * scale);
        let after = trace_projection(&trial_core);
        if after + accept_slack >= before && after.is_finite() {
            *core = trial_core;
            apply_basis_rotor(u_basis, i, j, theta_l * scale);
            apply_basis_rotor(v_basis, i, j, theta_r * scale);
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

fn apply_basis_rotor(basis: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = basis.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let bi = basis[[r, i]];
        let bj = basis[[r, j]];
        basis[[r, i]] = c * bi - s * bj;
        basis[[r, j]] = s * bi + c * bj;
    }
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
        let rel = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(a.nrows());
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel, orth_u, orth_v)
    }

    #[test]
    fn traceflow_polished_random_16() {
        let mut rng = StdRng::seed_from_u64(88);
        let a = Array2::from_shape_fn((16, 16), |_| rng.gen::<f64>() - 0.5);
        let ((u, sigma, vt), trace) =
            LieSvdTraceFlow::solve_with_trace(&a, LieSvdTraceFlowParams::for_n(16), 2);
        let (rel, orth_u, orth_v) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.final_projection + 1e-12 >= trace.initial_projection);
        assert!(rel < 1e-11, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn traceflow_plateau_on_identity_degenerate_spectrum() {
        let a = Array2::<f64>::eye(8);
        let ((_u, _sigma, _vt), trace) =
            LieSvdTraceFlow::precondition_with_trace(&a, LieSvdTraceFlowParams::for_n(8), 1);
        assert!((trace.final_projection - trace.initial_projection).abs() < 1e-12);
        assert!(trace.final_offdiag < 1e-12);
        assert_eq!(trace.rotations, 0);
    }
}
