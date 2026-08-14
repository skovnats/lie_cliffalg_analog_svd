//! Kernel/Gram helpers for spectral SVD experiments.
//!
//! A single-domain Gram matrix is symmetric by construction. In that case the
//! left and right rotor bases coincide (`K = U Sigma U^T`), so the correct
//! route is a one-basis eigensolver/trace-maximization path. A nonsymmetric
//! square cross-kernel is a bipartite object and keeps the usual two-sided
//! `CoreFlow` route.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub enum KernelKind {
    Linear,
    Rbf { gamma: f64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelRoute {
    SymmetricEigen,
    BipartiteCoreFlow,
}

#[derive(Clone, Debug)]
pub struct KernelSvd {
    pub u: Array2<f64>,
    pub sigma: Array1<f64>,
    pub vt: Array2<f64>,
    pub route: KernelRoute,
}

pub fn build_gram(points: &[Vec<f64>], kernel: KernelKind) -> Array2<f64> {
    build_cross_kernel(points, points, kernel)
}

pub fn build_cross_kernel(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    kernel: KernelKind,
) -> Array2<f64> {
    assert!(!left.is_empty(), "kernel input must not be empty");
    assert!(!right.is_empty(), "kernel input must not be empty");
    let dim = left[0].len();
    assert!(dim > 0, "kernel points must not be empty");
    assert!(
        left.iter().all(|p| p.len() == dim) && right.iter().all(|p| p.len() == dim),
        "all kernel points must have the same dimension"
    );

    Array2::from_shape_fn((left.len(), right.len()), |(i, j)| match kernel {
        KernelKind::Linear => dot(&left[i], &right[j]),
        KernelKind::Rbf { gamma } => (-gamma * dist_sq(&left[i], &right[j])).exp(),
    })
}

pub fn is_symmetric(k: &Array2<f64>, tol: f64) -> bool {
    if k.nrows() != k.ncols() {
        return false;
    }
    let n = k.nrows();
    let scale = k.iter().map(|x| x.abs()).fold(0.0_f64, f64::max).max(1.0);
    let tol = tol.max(0.0) * scale;
    for i in 0..n {
        for j in (i + 1)..n {
            if (k[[i, j]] - k[[j, i]]).abs() > tol {
                return false;
            }
        }
    }
    true
}

pub fn solve_kernel(k: &Array2<f64>, tol: f64) -> KernelSvd {
    assert_eq!(
        k.nrows(),
        k.ncols(),
        "kernel solver expects a square matrix"
    );
    if is_symmetric(k, tol) {
        let (u, sigma) = symmetric_kernel_eigh(k, tol);
        let vt = u.t().to_owned();
        KernelSvd {
            u,
            sigma,
            vt,
            route: KernelRoute::SymmetricEigen,
        }
    } else {
        let (u, sigma, vt) = crate::lie_svd_coreflow::LieSvdCoreFlow::solve(k);
        KernelSvd {
            u,
            sigma,
            vt,
            route: KernelRoute::BipartiteCoreFlow,
        }
    }
}

pub fn trace_objective(k: &Array2<f64>, u: &Array2<f64>) -> f64 {
    let ku = k.dot(u);
    (0..u.ncols())
        .map(|i| (0..u.nrows()).map(|r| u[[r, i]] * ku[[r, i]]).sum::<f64>())
        .sum()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn symmetric_kernel_eigh(k: &Array2<f64>, tol: f64) -> (Array2<f64>, Array1<f64>) {
    let n = k.nrows();
    let mut work = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            work[[i, j]] = 0.5 * (k[[i, j]] + k[[j, i]]);
        }
    }
    let mut u = Array2::<f64>::eye(n);
    let ref_norm = work.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    let pair_tol = tol.max(1e-15) * ref_norm;

    for _ in 0..(64 * n.max(1)) {
        let mut changed = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[[p, q]];
                if apq.abs() <= pair_tol {
                    continue;
                }
                let app = work[[p, p]];
                let aqq = work[[q, q]];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = work[[r, p]];
                        let arq = work[[r, q]];
                        let new_rp = c * arp - s * arq;
                        let new_rq = s * arp + c * arq;
                        work[[r, p]] = new_rp;
                        work[[p, r]] = new_rp;
                        work[[r, q]] = new_rq;
                        work[[q, r]] = new_rq;
                    }
                }

                work[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                work[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                work[[p, q]] = 0.0;
                work[[q, p]] = 0.0;

                for r in 0..n {
                    let urp = u[[r, p]];
                    let urq = u[[r, q]];
                    u[[r, p]] = c * urp - s * urq;
                    u[[r, q]] = s * urp + c * urq;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        work[[b, b]]
            .partial_cmp(&work[[a, a]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_u = Array2::<f64>::zeros((n, n));
    let mut sigma = Array1::<f64>::zeros(n);
    for (dst, &src) in order.iter().enumerate() {
        sigma[dst] = work[[src, src]].max(0.0);
        for r in 0..n {
            sorted_u[[r, dst]] = u[[r, src]];
        }
    }
    (sorted_u, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offdiag_norm(a: &Array2<f64>) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                if i != j {
                    s += a[[i, j]] * a[[i, j]];
                }
            }
        }
        s.sqrt()
    }

    fn rel_recon(a: &Array2<f64>, u: &Array2<f64>, sigma: &Array1<f64>, vt: &Array2<f64>) -> f64 {
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let num = (&recon - a).mapv(|x| x * x).sum().sqrt();
        let den = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        num / den
    }

    fn clustered_points() -> Vec<Vec<f64>> {
        let centers = [(-3.0, 0.0), (3.0, 0.0), (0.0, 4.0)];
        let mut points = Vec::new();
        for (cx, cy) in centers {
            for k in 0..6 {
                let dx = (k as f64 - 2.5) * 0.035;
                let dy = ((k * 7 % 5) as f64 - 2.0) * 0.03;
                points.push(vec![cx + dx, cy + dy]);
            }
        }
        points
    }

    #[test]
    fn test_build_gram_linear_and_rbf_are_symmetric() {
        let points = vec![vec![1.0, 2.0], vec![3.0, -1.0], vec![0.5, 0.25]];
        let linear = build_gram(&points, KernelKind::Linear);
        let rbf = build_gram(&points, KernelKind::Rbf { gamma: 0.5 });
        assert!(is_symmetric(&linear, 1e-12));
        assert!(is_symmetric(&rbf, 1e-12));
        assert!((linear[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((rbf[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_symmetric_kernel_uses_one_basis_route() {
        let points = clustered_points();
        let k = build_gram(&points, KernelKind::Rbf { gamma: 0.7 });
        let svd = solve_kernel(&k, 1e-12);
        assert_eq!(svd.route, KernelRoute::SymmetricEigen);
        assert!((&svd.vt - &svd.u.t()).mapv(|x| x * x).sum().sqrt() < 1e-10);
        assert!(rel_recon(&k, &svd.u, &svd.sigma, &svd.vt) < 1e-10);
        let core = svd.u.t().dot(&k).dot(&svd.u);
        assert!(offdiag_norm(&core) < 1e-8);
    }

    #[test]
    fn test_square_cross_kernel_uses_bipartite_route() {
        let left = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![2.0, 1.0]];
        let right = vec![vec![0.5, 1.0], vec![2.0, -1.0], vec![1.0, 1.0]];
        let k = build_cross_kernel(&left, &right, KernelKind::Linear);
        assert!(!is_symmetric(&k, 1e-12));
        let svd = solve_kernel(&k, 1e-12);
        assert_eq!(svd.route, KernelRoute::BipartiteCoreFlow);
        assert!(rel_recon(&k, &svd.u, &svd.sigma, &svd.vt) < 1e-10);
    }

    #[test]
    fn test_rbf_cluster_kernel_starts_with_lower_core_tension_than_linear() {
        let points = clustered_points();
        let linear = build_gram(&points, KernelKind::Linear);
        let rbf = build_gram(&points, KernelKind::Rbf { gamma: 0.8 });
        let params = crate::lie_svd_coreflow::LieSvdCoreFlowParams {
            max_sweeps: 2,
            ..crate::lie_svd_coreflow::LieSvdCoreFlowParams::for_n(points.len())
        };
        let ((_u_l, _s_l, _vt_l), linear_trace) =
            crate::lie_svd_coreflow::LieSvdCoreFlow::precondition_with_trace(&linear, params, 1);
        let ((_u_r, _s_r, _vt_r), rbf_trace) =
            crate::lie_svd_coreflow::LieSvdCoreFlow::precondition_with_trace(&rbf, params, 1);

        assert!(
            rbf_trace.final_offdiag < linear_trace.final_offdiag,
            "linear_final={} rbf_final={}",
            linear_trace.final_offdiag,
            rbf_trace.final_offdiag
        );
    }
}
