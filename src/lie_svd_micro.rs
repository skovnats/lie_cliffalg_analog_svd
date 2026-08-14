//! Tiny SVD microkernels for `N <= 4`.
//!
//! The general polar/Jacobi solver is accurate, but it has unnecessary setup
//! cost for matrices that fit entirely inside one local rotor cell or one small
//! rotor mesh. This module treats `1x1..4x4` matrices as fixed schedules:
//! closed-form `2x2` two-sided rotors, a three-pair `3x3` cycle, and a
//! three-layer conflict-free `4x4` schedule.
//!
//! The implementation remains conservative: after the tiny schedule, it checks
//! the residual off-diagonal energy and escalates to `LieSvdSmall` if the local
//! rotor mesh did not finish cleanly.

use ndarray::{Array1, Array2};

const MICRO_TOL: f64 = 1e-13;

pub struct LieSvdMicro;

impl LieSvdMicro {
    /// Solves a square dense matrix with `N <= 4`.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdMicro: matrix must be square");
        assert!(n <= 4, "LieSvdMicro only supports N <= 4");

        match n {
            0 => (
                Array2::<f64>::zeros((0, 0)),
                Array1::<f64>::zeros(0),
                Array2::<f64>::zeros((0, 0)),
            ),
            1 => {
                let d = mat[[0, 0]];
                let sign = if d < 0.0 { -1.0 } else { 1.0 };
                (
                    Array2::from_elem((1, 1), sign),
                    Array1::from_elem(1, d.abs()),
                    Array2::eye(1),
                )
            }
            2 => solve_fixed_schedule(mat, &[(0, 1)], 1),
            3 => {
                let pairs = [(0, 1), (0, 2), (1, 2)];
                solve_fixed_schedule(mat, &pairs, 24)
            }
            4 => {
                let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
                solve_fixed_schedule(mat, &pairs, 24)
            }
            _ => unreachable!(),
        }
    }
}

fn solve_fixed_schedule(
    mat: &Array2<f64>,
    pairs: &[(usize, usize)],
    max_sweeps: usize,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    let mut work = mat.clone();
    let mut u_basis = Array2::<f64>::eye(n);
    let mut v_basis = Array2::<f64>::eye(n);
    let ref_norm = frobenius_norm(mat).max(1e-300);
    let tol = MICRO_TOL * ref_norm.max(1.0);

    for _ in 0..max_sweeps.max(1) {
        let energy = offdiag_norm(&work);
        if energy <= tol || !energy.is_finite() {
            break;
        }
        let mut changed = false;
        for &(i, j) in pairs {
            if pair_offdiag(&work, i, j) <= tol {
                continue;
            }
            let (theta_l, theta_r) = local_pair_svd_angles(&work, i, j);
            if theta_l.abs() + theta_r.abs() <= 1e-18 {
                continue;
            }
            apply_left_rotor(&mut work, &mut u_basis, i, j, theta_l);
            apply_right_rotor(&mut work, &mut v_basis, i, j, theta_r);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    if offdiag_norm(&work) > 1e-10 * ref_norm.max(1.0) {
        return crate::lie_svd_small::LieSvdSmall::solve(mat);
    }

    extract_sorted_svd(&work, &u_basis, &v_basis)
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

fn apply_left_rotor(
    work: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
) {
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
        let n = a.nrows();
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel_recon = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(n);
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel_recon, orth_u, orth_v)
    }

    #[test]
    fn test_micro_random_1_to_4() {
        for n in 1..=4 {
            let mut rng = StdRng::seed_from_u64(900 + n as u64);
            for _ in 0..32 {
                let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-2.0_f64..2.0));
                let (u, sigma, vt) = LieSvdMicro::solve(&a);
                let (rel, ou, ov) = metrics(&a, &u, &sigma, &vt);
                assert!(rel < 1e-10, "n={n} rel={rel}");
                assert!(ou < 1e-10, "n={n} orth_u={ou}");
                assert!(ov < 1e-10, "n={n} orth_v={ov}");
            }
        }
    }

    #[test]
    fn test_micro_degenerate_4() {
        let mut a = Array2::<f64>::zeros((4, 4));
        a[[0, 0]] = 10.0;
        a[[1, 1]] = 10.0;
        a[[2, 2]] = 1e-12;
        a[[3, 3]] = 1e-12;
        a[[0, 1]] = 0.25;
        a[[2, 3]] = -0.5;
        let (u, sigma, vt) = LieSvdMicro::solve(&a);
        let (rel, ou, ov) = metrics(&a, &u, &sigma, &vt);
        assert!(rel < 1e-10, "rel={rel}");
        assert!(ou < 1e-10, "orth_u={ou}");
        assert!(ov < 1e-10, "orth_v={ov}");
    }
}
