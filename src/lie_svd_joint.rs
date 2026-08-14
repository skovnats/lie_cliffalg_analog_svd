//! Joint diagonalization by shared Lie/Clifford phase rotors.
//!
//! This is the Phase-JADE / Joint SVD prototype.  It generalizes the
//! single-core phase-locking idea from SVD to a set of matrices.  The
//! symmetric route looks for one orthogonal basis `V` that minimizes
//!
//! `sum_k ||offdiag(V^T M_k V)||_F^2`.
//!
//! The two-sided route looks for `U,V` minimizing
//! `sum_k ||offdiag(U^T A_k V)||_F^2`.
//!
//! The implementation deliberately keeps the hot path as plain `f64`: the
//! Clifford interpretation is carried by the shared rotor field, while the
//! stored state remains ordinary dense matrices.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct JointDiagonalizationParams {
    pub max_sweeps: usize,
    pub tol: f64,
    pub min_pair_scale: f64,
    pub line_search_steps: usize,
}

impl JointDiagonalizationParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_sweeps: 12 + n.max(1),
            tol: 1e-12,
            min_pair_scale: 1e-14,
            line_search_steps: 6,
        }
    }
}

impl Default for JointDiagonalizationParams {
    fn default() -> Self {
        Self::for_n(32)
    }
}

#[derive(Clone, Debug)]
pub struct JointDiagonalizationTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub rotations: usize,
    pub rejected_rotations: usize,
}

#[derive(Clone, Debug)]
pub struct JointSvdTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub rotations: usize,
    pub rejected_rotations: usize,
}

pub struct LieSvdJoint;

impl LieSvdJoint {
    pub fn diagonalize_symmetric(
        matrices: &[Array2<f64>],
    ) -> (Array2<f64>, Vec<Array1<f64>>, JointDiagonalizationTrace) {
        let n = matrices.first().map(|m| m.nrows()).unwrap_or(0);
        Self::diagonalize_symmetric_with_params(matrices, JointDiagonalizationParams::for_n(n))
    }

    pub fn diagonalize_symmetric_with_params(
        matrices: &[Array2<f64>],
        params: JointDiagonalizationParams,
    ) -> (Array2<f64>, Vec<Array1<f64>>, JointDiagonalizationTrace) {
        assert!(!matrices.is_empty(), "LieSvdJoint: empty matrix set");
        let n = matrices[0].nrows();
        assert_eq!(
            n,
            matrices[0].ncols(),
            "LieSvdJoint: matrices must be square"
        );
        for m in matrices {
            assert_eq!(m.nrows(), n, "LieSvdJoint: row-size mismatch");
            assert_eq!(m.ncols(), n, "LieSvdJoint: col-size mismatch");
        }

        if n <= 1 {
            let basis = Array2::<f64>::eye(n);
            let diagonals = matrices.iter().map(diagonal).collect::<Vec<_>>();
            return (
                basis,
                diagonals,
                JointDiagonalizationTrace {
                    initial_offdiag: 0.0,
                    final_offdiag: 0.0,
                    sweeps: 0,
                    rotations: 0,
                    rejected_rotations: 0,
                },
            );
        }

        let mut work = matrices.to_vec();
        let mut basis = Array2::<f64>::eye(n);
        let ref_norm = joint_frobenius_norm(&work).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let tol = params.tol * ref_norm.max(1.0);
        let initial_offdiag = joint_offdiag_norm(&work);
        let mut rotations = 0usize;
        let mut rejected_rotations = 0usize;
        let mut sweeps = 0usize;

        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let before = joint_offdiag_norm(&work);
            for layer in 0..round_robin_layer_count(n) {
                for (i, j) in layer_pairs(n, layer) {
                    if joint_pair_offdiag(&work, i, j) <= pair_tol {
                        continue;
                    }
                    let theta = joint_symmetric_pair_angle(&work, i, j);
                    if theta.abs() <= 1e-18 {
                        continue;
                    }
                    if accept_joint_rotor(
                        &mut work,
                        &mut basis,
                        i,
                        j,
                        theta,
                        params.line_search_steps,
                    ) {
                        rotations += 1;
                    } else {
                        rejected_rotations += 1;
                    }
                }
            }
            let after = joint_offdiag_norm(&work);
            if after <= tol || after >= before * (1.0 - 1e-10) {
                break;
            }
        }

        let final_offdiag = joint_offdiag_norm(&work);
        let diagonals = work.iter().map(diagonal).collect::<Vec<_>>();
        (
            basis,
            diagonals,
            JointDiagonalizationTrace {
                initial_offdiag,
                final_offdiag,
                sweeps,
                rotations,
                rejected_rotations,
            },
        )
    }

    pub fn joint_svd(
        matrices: &[Array2<f64>],
    ) -> (Array2<f64>, Vec<Array1<f64>>, Array2<f64>, JointSvdTrace) {
        let n = matrices
            .first()
            .map(|m| m.nrows().min(m.ncols()))
            .unwrap_or(0);
        Self::joint_svd_with_params(matrices, JointDiagonalizationParams::for_n(n))
    }

    pub fn joint_svd_with_params(
        matrices: &[Array2<f64>],
        params: JointDiagonalizationParams,
    ) -> (Array2<f64>, Vec<Array1<f64>>, Array2<f64>, JointSvdTrace) {
        assert!(!matrices.is_empty(), "LieSvdJoint: empty matrix set");
        let rows = matrices[0].nrows();
        let cols = matrices[0].ncols();
        assert!(rows > 0 && cols > 0, "LieSvdJoint: empty matrix");
        for m in matrices {
            assert_eq!(m.nrows(), rows, "LieSvdJoint: row-size mismatch");
            assert_eq!(m.ncols(), cols, "LieSvdJoint: col-size mismatch");
        }

        let k = rows.min(cols);
        if k <= 1 {
            let u = Array2::<f64>::eye(rows);
            let vt = Array2::<f64>::eye(cols);
            let sigmas = matrices.iter().map(diagonal_corridor).collect::<Vec<_>>();
            return (
                u,
                sigmas,
                vt,
                JointSvdTrace {
                    initial_offdiag: 0.0,
                    final_offdiag: 0.0,
                    sweeps: 0,
                    rotations: 0,
                    rejected_rotations: 0,
                },
            );
        }

        let mut work = matrices.to_vec();
        let mut u_basis = Array2::<f64>::eye(rows);
        let mut v_basis = Array2::<f64>::eye(cols);
        let ref_norm = joint_frobenius_norm(&work).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let tol = params.tol * ref_norm.max(1.0);
        let initial_offdiag = joint_offdiag_norm(&work);
        let mut rotations = 0usize;
        let mut rejected_rotations = 0usize;
        let mut sweeps = 0usize;

        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let before = joint_offdiag_norm(&work);
            for layer in 0..round_robin_layer_count(k) {
                for (i, j) in layer_pairs(k, layer) {
                    if joint_pair_offdiag(&work, i, j) <= pair_tol {
                        continue;
                    }
                    let (theta_l, theta_r) = joint_svd_pair_angles(&work, i, j);
                    if theta_l.abs() + theta_r.abs() <= 1e-18 {
                        continue;
                    }
                    if accept_joint_svd_rotor(
                        &mut work,
                        &mut u_basis,
                        &mut v_basis,
                        i,
                        j,
                        theta_l,
                        theta_r,
                        params.line_search_steps,
                    ) {
                        rotations += 1;
                    } else {
                        rejected_rotations += 1;
                    }
                }
            }
            let after = joint_offdiag_norm(&work);
            if after <= tol || after >= before * (1.0 - 1e-10) {
                break;
            }
        }

        let final_offdiag = joint_offdiag_norm(&work);
        let sigmas = work.iter().map(diagonal_corridor).collect::<Vec<_>>();
        (
            u_basis,
            sigmas,
            v_basis.t().to_owned(),
            JointSvdTrace {
                initial_offdiag,
                final_offdiag,
                sweeps,
                rotations,
                rejected_rotations,
            },
        )
    }
}

fn joint_svd_pair_angles(work: &[Array2<f64>], i: usize, j: usize) -> (f64, f64) {
    let mut a = 0.0_f64;
    let mut b = 0.0_f64;
    let mut c = 0.0_f64;
    let mut d = 0.0_f64;
    for m in work {
        let weight = pair_block_norm(m, i, j).max(1e-300);
        a += m[[i, i]] / weight;
        b += m[[i, j]] / weight;
        c += m[[j, i]] / weight;
        d += m[[j, j]] / weight;
    }
    local_pair_svd_angles_from_values(a, b, c, d)
}

fn pair_block_norm(m: &Array2<f64>, i: usize, j: usize) -> f64 {
    (m[[i, i]] * m[[i, i]] + m[[i, j]] * m[[i, j]] + m[[j, i]] * m[[j, i]] + m[[j, j]] * m[[j, j]])
        .sqrt()
}

fn local_pair_svd_angles_from_values(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let sum_angle = (-(b + c)).atan2(a - d);
    let diff_angle = (b - c).atan2(a + d);
    (
        wrap_svd_angle(0.5 * (sum_angle + diff_angle)),
        wrap_svd_angle(0.5 * (sum_angle - diff_angle)),
    )
}

fn wrap_svd_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::FRAC_PI_2 {
        angle -= std::f64::consts::PI;
    }
    while angle < -std::f64::consts::FRAC_PI_2 {
        angle += std::f64::consts::PI;
    }
    angle
}

fn accept_joint_svd_rotor(
    work: &mut [Array2<f64>],
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta_l: f64,
    theta_r: f64,
    line_search_steps: usize,
) -> bool {
    let before = joint_local_offdiag_sq_for_axes(work, i, j);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let tl = theta_l * scale;
        let tr = theta_r * scale;
        for m in work.iter_mut() {
            apply_left_rotor(m, i, j, tl);
            apply_right_rotor(m, i, j, tr);
        }
        let after = joint_local_offdiag_sq_for_axes(work, i, j);
        if after <= before + slack && after.is_finite() {
            apply_basis_rotor(u_basis, i, j, tl);
            apply_basis_rotor(v_basis, i, j, tr);
            return true;
        }
        for m in work.iter_mut() {
            apply_right_rotor(m, i, j, -tr);
            apply_left_rotor(m, i, j, -tl);
        }
        scale *= 0.5;
    }
    false
}

fn joint_symmetric_pair_angle(work: &[Array2<f64>], i: usize, j: usize) -> f64 {
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for m in work {
        let x = 0.5 * (m[[i, i]] - m[[j, j]]);
        let y = 0.5 * (m[[i, j]] + m[[j, i]]);
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let theta = 0.25 * (2.0 * sxy).atan2(syy - sxx);
    let alt = wrap_jacobi_angle(theta + std::f64::consts::FRAC_PI_4);
    let theta = wrap_jacobi_angle(theta);
    if joint_pair_energy_after(work, i, j, alt) < joint_pair_energy_after(work, i, j, theta) {
        alt
    } else {
        theta
    }
}

fn accept_joint_rotor(
    work: &mut [Array2<f64>],
    basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
    line_search_steps: usize,
) -> bool {
    let before = joint_local_offdiag_sq_for_axes(work, i, j);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let angle = theta * scale;
        for m in work.iter_mut() {
            apply_symmetric_rotor(m, i, j, angle);
        }
        let after = joint_local_offdiag_sq_for_axes(work, i, j);
        if after <= before + slack && after.is_finite() {
            apply_basis_rotor(basis, i, j, angle);
            return true;
        }
        for m in work.iter_mut() {
            apply_symmetric_rotor(m, i, j, -angle);
        }
        scale *= 0.5;
    }
    false
}

fn joint_pair_energy_after(work: &[Array2<f64>], i: usize, j: usize, theta: f64) -> f64 {
    let (s, c) = theta.sin_cos();
    let mut out = 0.0_f64;
    for m in work {
        let a = m[[i, i]];
        let b = 0.5 * (m[[i, j]] + m[[j, i]]);
        let d = m[[j, j]];
        let off = 0.5 * (a - d) * (2.0 * s * c) + b * (c * c - s * s);
        out += 2.0 * off * off;
    }
    out
}

pub(crate) fn apply_symmetric_rotor(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    apply_left_rotor(work, i, j, theta);
    apply_right_rotor(work, i, j, theta);
}

pub(crate) fn apply_left_rotor(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let cols = work.ncols();
    let (s, c) = theta.sin_cos();
    for col in 0..cols {
        let ai = work[[i, col]];
        let aj = work[[j, col]];
        work[[i, col]] = c * ai - s * aj;
        work[[j, col]] = s * ai + c * aj;
    }
}

pub(crate) fn apply_right_rotor(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let rows = work.nrows();
    let (s, c) = theta.sin_cos();
    for row in 0..rows {
        let ai = work[[row, i]];
        let aj = work[[row, j]];
        work[[row, i]] = c * ai - s * aj;
        work[[row, j]] = s * ai + c * aj;
    }
}

pub(crate) fn apply_basis_rotor(basis: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = basis.nrows();
    let (s, c) = theta.sin_cos();
    for row in 0..n {
        let bi = basis[[row, i]];
        let bj = basis[[row, j]];
        basis[[row, i]] = c * bi - s * bj;
        basis[[row, j]] = s * bi + c * bj;
    }
}

fn joint_offdiag_norm(work: &[Array2<f64>]) -> f64 {
    work.iter().map(offdiag_sq).sum::<f64>().sqrt()
}

fn joint_frobenius_norm(work: &[Array2<f64>]) -> f64 {
    work.iter()
        .flat_map(|m| m.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

fn joint_pair_offdiag(work: &[Array2<f64>], i: usize, j: usize) -> f64 {
    work.iter().map(|m| m[[i, j]].abs() + m[[j, i]].abs()).sum()
}

fn joint_local_offdiag_sq_for_axes(work: &[Array2<f64>], i: usize, j: usize) -> f64 {
    work.iter()
        .map(|m| local_offdiag_sq_for_axes(m, i, j))
        .sum()
}

pub(crate) fn offdiag_sq(m: &Array2<f64>) -> f64 {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut s = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            if i != j {
                s += m[[i, j]] * m[[i, j]];
            }
        }
    }
    s
}

pub(crate) fn local_offdiag_sq_for_axes(m: &Array2<f64>, i: usize, j: usize) -> f64 {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut s = 0.0_f64;
    for col in 0..cols {
        if col != i {
            s += m[[i, col]] * m[[i, col]];
        }
        if col != j {
            s += m[[j, col]] * m[[j, col]];
        }
    }
    for row in 0..rows {
        if row != i && row != j {
            s += m[[row, i]] * m[[row, i]];
            s += m[[row, j]] * m[[row, j]];
        }
    }
    s
}

fn diagonal(m: &Array2<f64>) -> Array1<f64> {
    let n = m.nrows();
    Array1::from_shape_fn(n, |i| m[[i, i]])
}

fn diagonal_corridor(m: &Array2<f64>) -> Array1<f64> {
    let n = m.nrows().min(m.ncols());
    Array1::from_shape_fn(n, |i| m[[i, i]].abs())
}

pub(crate) fn wrap_jacobi_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::FRAC_PI_4 {
        angle -= std::f64::consts::FRAC_PI_2;
    }
    while angle < -std::f64::consts::FRAC_PI_4 {
        angle += std::f64::consts::FRAC_PI_2;
    }
    angle
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn phase_jade_jointly_diagonalizes_symmetric_family() {
        run_joint_family_case(12, 5, 130);
    }

    #[test]
    fn phase_jade_jointly_diagonalizes_two_matrices() {
        run_joint_family_case(10, 2, 131);
    }

    #[test]
    fn phase_jade_jointly_diagonalizes_larger_m_family() {
        run_joint_family_case(14, 9, 132);
    }

    fn run_joint_family_case(n: usize, k: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let q = random_orthogonal(n, &mut rng);
        let mut matrices = Vec::new();
        for family in 0..k {
            let diag = Array2::from_diag(&Array1::from_shape_fn(n, |i| {
                1.0 + family as f64 * 0.3 + i as f64 * 0.17
            }));
            matrices.push(q.dot(&diag).dot(&q.t()));
        }

        let params = JointDiagonalizationParams {
            max_sweeps: 40,
            ..JointDiagonalizationParams::for_n(n)
        };
        let (v, diagonals, trace) =
            LieSvdJoint::diagonalize_symmetric_with_params(&matrices, params);
        let ident = Array2::<f64>::eye(n);
        let orth = (&v.t().dot(&v) - &ident).mapv(|x| x * x).sum().sqrt();
        assert!(trace.final_offdiag < trace.initial_offdiag * 1e-8);
        assert!(trace.rotations > 0);
        assert_eq!(diagonals.len(), k);
        assert!(orth < 1e-10, "orth={orth:e}");
    }

    #[test]
    fn phase_jade_keeps_already_diagonal_family_stable() {
        let n = 8;
        let matrices = (0..4)
            .map(|k| Array2::from_diag(&Array1::from_shape_fn(n, |i| 1.0 + i as f64 + k as f64)))
            .collect::<Vec<_>>();
        let (_v, _diagonals, trace) = LieSvdJoint::diagonalize_symmetric(&matrices);
        assert!(trace.final_offdiag <= 1e-14);
        assert_eq!(trace.rotations, 0);
    }

    #[test]
    fn joint_svd_reduces_two_sided_offdiag_on_nonsymmetric_family() {
        let n = 10;
        let k = 4;
        let mut rng = StdRng::seed_from_u64(150);
        let u0 = random_orthogonal(n, &mut rng);
        let v0 = random_orthogonal(n, &mut rng);
        let matrices = (0..k)
            .map(|family| {
                let diag = Array2::from_diag(&Array1::from_shape_fn(n, |i| {
                    1.0 + family as f64 * 0.2 + i as f64 * 0.11
                }));
                u0.dot(&diag).dot(&v0.t())
            })
            .collect::<Vec<_>>();
        let params = JointDiagonalizationParams {
            max_sweeps: 36,
            ..JointDiagonalizationParams::for_n(n)
        };
        let (u, sigmas, vt, trace) = LieSvdJoint::joint_svd_with_params(&matrices, params);
        let ident = Array2::<f64>::eye(n);
        let orth_u = (&u.t().dot(&u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        assert!(trace.final_offdiag < trace.initial_offdiag * 1e-8);
        assert!(trace.rotations > 0);
        assert_eq!(sigmas.len(), k);
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn joint_svd_accepts_rectangular_family() {
        let rows = 9;
        let cols = 13;
        let k = rows.min(cols);
        let mut rng = StdRng::seed_from_u64(151);
        let u0 = random_orthogonal(rows, &mut rng);
        let v0 = random_orthogonal(cols, &mut rng);
        let matrices = (0..3)
            .map(|family| {
                let mut sigma = Array2::<f64>::zeros((rows, cols));
                for i in 0..k {
                    sigma[[i, i]] = 1.0 + family as f64 * 0.2 + i as f64 * 0.07;
                }
                u0.dot(&sigma).dot(&v0.t())
            })
            .collect::<Vec<_>>();
        let params = JointDiagonalizationParams {
            max_sweeps: 32,
            ..JointDiagonalizationParams::for_n(k)
        };
        let (u, sigmas, vt, trace) = LieSvdJoint::joint_svd_with_params(&matrices, params);
        let ident_u = Array2::<f64>::eye(rows);
        let ident_v = Array2::<f64>::eye(cols);
        let orth_u = (&u.t().dot(&u) - &ident_u).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident_v).mapv(|x| x * x).sum().sqrt();
        assert!(
            trace.final_offdiag <= trace.initial_offdiag,
            "initial={:.3e} final={:.3e}",
            trace.initial_offdiag,
            trace.final_offdiag
        );
        assert_eq!(sigmas.len(), 3);
        assert_eq!(sigmas[0].len(), k);
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

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
}
