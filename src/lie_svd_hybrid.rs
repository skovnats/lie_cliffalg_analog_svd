//! Hybrid release SVD solver.
//!
//! This is the cleaned-up production face of the geometric experiments:
//! a short dual-tiled Lie-rotor preconditioner followed by the accurate
//! `LieSvdSmall` polar/Jacobi polish.
//!
//! The Clifford idea is kept as an algorithmic invariant, not as a new heap
//! representation. Conceptually we view the matrix as `M = I + eA`: `I` is a
//! scalar anchor that keeps local ratios well-conditioned, while the `eA`
//! component is expressed by ordinary `f64` two-sided Givens rotors. In code
//! this means: standard dense matrices in memory, dual row/column metrics for
//! pair selection, a scalar anchor in local denominators, and an explicit
//! Manopt-style retraction of the left/right bases onto `O(n)` before solving
//! the final core problem.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct LieSvdHybridParams {
    /// Number of active axes inspected together.
    pub tile_size: usize,
    /// Conflict-free rotor pairs kept from each tile.
    pub pairs_per_tile: usize,
    /// Hard cap for one geometric preconditioner rotor.
    pub max_angle: f64,
    /// Blends direct row/column metric rotations with their dual mirrors.
    pub quad_feedback: f64,
    /// Uses the nonsymmetric `A_ij - A_ji` torsion to separate left/right moves.
    pub torsion_gain: f64,
    /// Axis lock threshold, relative to the input Frobenius norm.
    pub deflation_tol: f64,
    /// Early stop threshold, relative to the input Frobenius norm.
    pub tol: f64,
    /// Budget for the geometric preconditioner. Final accuracy comes from
    /// `LieSvdSmall`, so this should stay modest.
    pub pre_steps: usize,
}

impl LieSvdHybridParams {
    pub fn for_n(n: usize) -> Self {
        let tile_size = if n <= 128 { 32 } else { 48 };
        let pairs_per_tile = if n <= 128 { 16 } else { 24 };
        let pre_steps = if n <= 128 { 120 } else { 240 };
        Self {
            tile_size,
            pairs_per_tile,
            max_angle: 0.16,
            quad_feedback: 0.25,
            torsion_gain: 0.20,
            deflation_tol: 1e-10,
            tol: 1e-11,
            pre_steps,
        }
    }
}

impl Default for LieSvdHybridParams {
    fn default() -> Self {
        Self::for_n(128)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdHybridTrace {
    pub initial_energy: f64,
    pub best_energy: f64,
    pub core_offdiag_energy: f64,
    pub grounding_pre_orth_u: f64,
    pub grounding_pre_orth_v: f64,
    pub grounding_post_orth_u: f64,
    pub grounding_post_orth_v: f64,
    pub pre_steps: usize,
    pub locked_axes: usize,
    pub rotor_steps: usize,
    pub planned_rotors: usize,
    pub tile_passes: usize,
    pub samples: Vec<f64>,
}

pub struct LieSvdHybrid;

impl LieSvdHybrid {
    /// Runs the hybrid release solver and returns `(U, Sigma, Vt)`.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let params = LieSvdHybridParams::for_n(mat.nrows());
        Self::solve_with_trace(mat, params, params.pre_steps / 40).0
    }

    /// Same solver with a trace for benchmarks and release diagnostics.
    // Allow: the return type mirrors this crate's established (U, Sigma, Vt[, Trace]) tuple convention; a type alias would obscure the shape at call sites during this stability freeze.
    #[allow(clippy::type_complexity)]
    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdHybridParams,
        sample_every: usize,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdHybridTrace) {
        let (u0, vt0, mut trace) = hybrid_precondition(mat, params, sample_every);
        let v0 = vt0.t().to_owned();

        trace.grounding_pre_orth_u = orthogonality_err(&u0);
        trace.grounding_pre_orth_v = orthogonality_err(&v0);
        let u0 = manopt_retract_to_orthogonal(&u0);
        let v0 = manopt_retract_to_orthogonal(&v0);
        trace.grounding_post_orth_u = orthogonality_err(&u0);
        trace.grounding_post_orth_v = orthogonality_err(&v0);

        let core = u0.t().dot(mat).dot(&v0);
        trace.core_offdiag_energy = direct_offdiag_norm(&core);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&v0.t());
        ((u, sigma, vt), trace)
    }
}

#[derive(Clone, Debug)]
struct DualCache {
    row_norms: Vec<f64>,
    col_norms: Vec<f64>,
}

impl DualCache {
    fn new(work: &Array2<f64>) -> Self {
        let n = work.nrows();
        Self {
            row_norms: (0..n).map(|i| row_norm_sq(work, i)).collect(),
            col_norms: (0..n).map(|i| col_norm_sq(work, i)).collect(),
        }
    }

    fn refresh_pair(&mut self, work: &Array2<f64>, i: usize, j: usize) {
        self.row_norms[i] = row_norm_sq(work, i);
        self.row_norms[j] = row_norm_sq(work, j);
        self.col_norms[i] = col_norm_sq(work, i);
        self.col_norms[j] = col_norm_sq(work, j);
    }
}

#[derive(Clone, Copy, Debug)]
struct PlannedRotor {
    i: usize,
    j: usize,
    theta_l: f64,
    theta_r: f64,
}

fn hybrid_precondition(
    mat: &Array2<f64>,
    params: LieSvdHybridParams,
    sample_every: usize,
) -> (Array2<f64>, Array2<f64>, LieSvdHybridTrace) {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "LieSvdHybrid: matrix must be square");

    let mut work = mat.clone();
    let mut u_basis = Array2::<f64>::eye(n);
    let mut v_basis = Array2::<f64>::eye(n);
    let mut cache = DualCache::new(&work);

    let ref_norm = frobenius_norm(mat).max(1e-300);
    let scalar_anchor = ref_norm / (n.max(1) as f64).sqrt();
    let tol = params.tol * ref_norm.max(1.0);
    let lock_threshold = params.deflation_tol * ref_norm.max(1.0);
    let initial_energy = direct_offdiag_norm(&work);
    let mut best_energy = initial_energy;
    let mut best_u = u_basis.clone();
    let mut best_v = v_basis.clone();
    let mut active = vec![true; n];
    let mut best_active = active.clone();
    let mut samples = Vec::new();
    let mut rotor_steps = 0usize;
    let mut planned_rotors = 0usize;
    let mut tile_passes = 0usize;
    let mut steps = 0usize;
    let sample_every = sample_every.max(1);
    let tile_size = params.tile_size.max(2);

    for step in 0..params.pre_steps {
        steps = step + 1;
        let energy = direct_offdiag_norm(&work);
        if step % sample_every == 0 {
            samples.push(energy);
        }
        update_direct_locks(&work, &mut active, lock_threshold);
        if energy <= best_energy {
            best_energy = energy;
            best_u = u_basis.clone();
            best_v = v_basis.clone();
            best_active = active.clone();
        }
        if energy <= tol || !energy.is_finite() {
            break;
        }

        let mut axes = active_axes(&active);
        if axes.len() < 2 {
            break;
        }
        let slide = (step * (tile_size / 2).max(1)) % axes.len();
        axes.rotate_left(slide);

        for tile in axes.chunks(tile_size) {
            let pairs = build_tile_pairs(&work, tile, &params);
            if pairs.is_empty() {
                continue;
            }
            tile_passes += 1;
            let planned = plan_tile_rotors(&work, &cache, &pairs, &params, scalar_anchor);
            planned_rotors += planned.len();
            for rotor in planned {
                apply_left_rotor(&mut work, &mut u_basis, rotor.i, rotor.j, rotor.theta_l);
                apply_right_rotor(&mut work, &mut v_basis, rotor.i, rotor.j, rotor.theta_r);
                cache.refresh_pair(&work, rotor.i, rotor.j);
                rotor_steps += 1;
            }
        }
    }

    (
        best_u.clone(),
        best_v.t().to_owned(),
        LieSvdHybridTrace {
            initial_energy,
            best_energy,
            core_offdiag_energy: 0.0,
            grounding_pre_orth_u: 0.0,
            grounding_pre_orth_v: 0.0,
            grounding_post_orth_u: 0.0,
            grounding_post_orth_v: 0.0,
            pre_steps: steps,
            locked_axes: best_active.iter().filter(|&&x| !x).count(),
            rotor_steps,
            planned_rotors,
            tile_passes,
            samples,
        },
    )
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn orthogonality_err(q: &Array2<f64>) -> f64 {
    let n = q.nrows();
    let ident = Array2::<f64>::eye(n);
    (&q.t().dot(q) - &ident).mapv(|x| x * x).sum().sqrt()
}

fn manopt_retract_to_orthogonal(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    let mut v = vec![0.0_f64; n];
    let mut candidate = vec![0.0_f64; n];
    let mut best_v = vec![0.0_f64; n];

    for j in 0..n {
        for i in 0..n {
            v[i] = a[[i, j]];
        }
        for k in 0..j {
            let mut d = 0.0_f64;
            for i in 0..n {
                d += out[[i, k]] * v[i];
            }
            for i in 0..n {
                v[i] -= d * out[[i, k]];
            }
        }

        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1e-12 {
            for x in &mut v {
                *x /= norm;
            }
        } else {
            let mut best_norm = -1.0_f64;
            for k in 0..n {
                candidate.fill(0.0);
                candidate[k] = 1.0;
                for col in 0..j {
                    let mut d = 0.0_f64;
                    for i in 0..n {
                        d += out[[i, col]] * candidate[i];
                    }
                    for i in 0..n {
                        candidate[i] -= d * out[[i, col]];
                    }
                }
                let enorm = candidate.iter().map(|x| x * x).sum::<f64>().sqrt();
                if enorm > best_norm {
                    best_norm = enorm;
                    best_v.copy_from_slice(&candidate);
                }
            }
            if best_norm > 0.0 && best_norm.is_finite() {
                for i in 0..n {
                    v[i] = best_v[i] / best_norm;
                }
            } else {
                v.fill(0.0);
                v[j] = 1.0;
            }
        }

        for i in 0..n {
            out[[i, j]] = v[i];
        }
    }

    out
}

fn direct_offdiag_norm(a: &Array2<f64>) -> f64 {
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

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::FRAC_PI_2 {
        angle -= std::f64::consts::PI;
    }
    while angle < -std::f64::consts::FRAC_PI_2 {
        angle += std::f64::consts::PI;
    }
    angle
}

fn mean_rotor_angle(angles: &[f64; 4]) -> f64 {
    let mut y = 0.0_f64;
    let mut x = 0.0_f64;
    for &theta in angles {
        y += (2.0 * theta).sin();
        x += (2.0 * theta).cos();
    }
    0.5 * y.atan2(x)
}

fn row_norm_sq(work: &Array2<f64>, i: usize) -> f64 {
    (0..work.ncols()).map(|k| work[[i, k]] * work[[i, k]]).sum()
}

fn col_norm_sq(work: &Array2<f64>, i: usize) -> f64 {
    (0..work.nrows()).map(|k| work[[k, i]] * work[[k, i]]).sum()
}

fn row_dot(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    (0..work.ncols()).map(|k| work[[i, k]] * work[[j, k]]).sum()
}

fn col_dot(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    (0..work.nrows()).map(|k| work[[k, i]] * work[[k, j]]).sum()
}

fn direct_pair_offdiag(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    work[[i, j]].abs() + work[[j, i]].abs()
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

fn dual_pair_score(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    row_dot(work, i, j).abs()
        + col_dot(work, i, j).abs()
        + direct_pair_offdiag(work, i, j)
        + (work[[i, j]] - work[[j, i]]).abs()
}

fn dual_pair_target(
    work: &Array2<f64>,
    cache: &DualCache,
    i: usize,
    j: usize,
    params: &LieSvdHybridParams,
    scalar_anchor: f64,
) -> (f64, f64) {
    let lij = row_dot(work, i, j);
    let rij = col_dot(work, i, j);
    let lii = cache.row_norms[i];
    let ljj = cache.row_norms[j];
    let rii = cache.col_norms[i];
    let rjj = cache.col_norms[j];

    let left_direct = 0.5 * (2.0 * lij).atan2(ljj - lii);
    let right_direct = 0.5 * (2.0 * rij).atan2(rjj - rii);
    let axis_gap = (j as f64 - i as f64).abs().max(1.0);
    let d_delta = ((work.nrows() - j) as f64 - (work.nrows() - i) as f64) / axis_gap;
    let left_dual = 0.5 * (2.0 * (-lij * d_delta)).atan2(ljj - lii);
    let right_dual = 0.5 * (2.0 * (-rij * d_delta)).atan2(rjj - rii);
    let quad_mean = mean_rotor_angle(&[left_direct, right_direct, left_dual, right_dual]);

    let torsion_scale = work[[i, i]].abs()
        + work[[j, j]].abs()
        + work[[i, j]].abs()
        + work[[j, i]].abs()
        + scalar_anchor
        + 1e-300;
    let torsion = 0.5 * (work[[i, j]] - work[[j, i]]).atan2(torsion_scale);
    let quad_weight = params.quad_feedback / (1.0 + params.quad_feedback);
    (
        left_direct
            + quad_weight * wrap_angle(quad_mean - left_direct)
            + params.torsion_gain * torsion,
        right_direct + quad_weight * wrap_angle(quad_mean - right_direct)
            - params.torsion_gain * torsion,
    )
}

fn axis_direct_coupling(work: &Array2<f64>, axis: usize) -> f64 {
    let n = work.nrows();
    let mut s = 0.0_f64;
    for j in 0..n {
        if j != axis {
            s += work[[axis, j]] * work[[axis, j]];
            s += work[[j, axis]] * work[[j, axis]];
        }
    }
    s.sqrt()
}

fn update_direct_locks(work: &Array2<f64>, active: &mut [bool], threshold: f64) {
    for (axis, is_active) in active.iter_mut().enumerate() {
        if *is_active && axis_direct_coupling(work, axis) <= threshold {
            *is_active = false;
        }
    }
}

fn active_axes(active: &[bool]) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter_map(|(axis, &is_active)| is_active.then_some(axis))
        .collect()
}

fn build_tile_pairs(
    work: &Array2<f64>,
    axes: &[usize],
    params: &LieSvdHybridParams,
) -> Vec<(usize, usize)> {
    let pair_indices: Vec<(usize, usize)> = (0..axes.len())
        .flat_map(|a| ((a + 1)..axes.len()).map(move |b| (axes[a], axes[b])))
        .collect();

    let mut candidates: Vec<(f64, usize, usize)> = pair_indices
        .par_iter()
        .map(|&(i, j)| (dual_pair_score(work, i, j), i, j))
        .collect();
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = vec![false; work.nrows()];
    let mut pairs = Vec::with_capacity(params.pairs_per_tile);
    for &(_, i, j) in &candidates {
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            pairs.push((i, j));
            if pairs.len() >= params.pairs_per_tile {
                break;
            }
        }
    }
    pairs
}

fn plan_tile_rotors(
    work: &Array2<f64>,
    cache: &DualCache,
    pairs: &[(usize, usize)],
    params: &LieSvdHybridParams,
    scalar_anchor: f64,
) -> Vec<PlannedRotor> {
    pairs
        .par_iter()
        .map(|&(i, j)| {
            let local_scale = work[[i, i]].abs() + work[[j, j]].abs() + scalar_anchor + 1e-300;
            let local_ratio = direct_pair_offdiag(work, i, j) / local_scale;
            if local_ratio > 1e-8 {
                let (theta_l, theta_r) = local_pair_svd_angles(work, i, j);
                PlannedRotor {
                    i,
                    j,
                    theta_l,
                    theta_r,
                }
            } else {
                let (theta_l, theta_r) = dual_pair_target(work, cache, i, j, params, scalar_anchor);
                PlannedRotor {
                    i,
                    j,
                    theta_l: theta_l.clamp(-params.max_angle, params.max_angle),
                    theta_r: theta_r.clamp(-params.max_angle, params.max_angle),
                }
            }
        })
        .collect()
}

fn apply_left_rotor(
    work: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
) {
    if theta.abs() < 1e-18 {
        return;
    }
    let n = work.nrows();
    let (s, c) = theta.sin_cos();
    for col in 0..n {
        let ai = work[[i, col]];
        let aj = work[[j, col]];
        work[[i, col]] = c * ai - s * aj;
        work[[j, col]] = s * ai + c * aj;
    }
    for r in 0..n {
        let ui = u_basis[[r, i]];
        let uj = u_basis[[r, j]];
        u_basis[[r, i]] = c * ui - s * uj;
        u_basis[[r, j]] = s * ui + c * uj;
    }
}

fn apply_right_rotor(
    work: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
) {
    if theta.abs() < 1e-18 {
        return;
    }
    let n = work.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let ai = work[[r, i]];
        let aj = work[[r, j]];
        work[[r, i]] = c * ai - s * aj;
        work[[r, j]] = s * ai + c * aj;
    }
    for r in 0..n {
        let vi = v_basis[[r, i]];
        let vj = v_basis[[r, j]];
        v_basis[[r, i]] = c * vi - s * vj;
        v_basis[[r, j]] = s * vi + c * vj;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let n = a.nrows();
        let mut sigma_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel_recon = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(n);
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel_recon, orth_u, orth_v)
    }

    #[test]
    fn test_hybrid_manopt_grounding_jordan_defective() {
        let n = 64;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0 - 0.5 * (i as f64 / n as f64);
            if i + 1 < n {
                a[[i, i + 1]] = 20.0;
            }
            if i + 2 < n && i % 4 == 0 {
                a[[i, i + 2]] = -3.0;
            }
        }

        let params = LieSvdHybridParams {
            pre_steps: 48,
            ..LieSvdHybridParams::for_n(n)
        };
        let ((u, sigma, vt), trace) = LieSvdHybrid::solve_with_trace(&a, params, 12);
        let (rel_recon, orth_u, orth_v) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.grounding_post_orth_u < 1e-10);
        assert!(trace.grounding_post_orth_v < 1e-10);
        assert!(rel_recon < 1e-10, "relative reconstruction: {rel_recon}");
        assert!(orth_u < 1e-8, "U not orthogonal: {orth_u}");
        assert!(orth_v < 1e-8, "V not orthogonal: {orth_v}");
    }
}
