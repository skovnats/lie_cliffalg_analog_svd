//! `4x4` macro-rotor warm start.
//!
//! `LieSvdMicro` already solves one `4x4` cell accurately. This module uses
//! that tiny cell as a larger building block: it applies local `4x4` SVDs to
//! selected axis quartets, including butterfly quartets that are natural for
//! power-of-two dimensions. The result is not advertised as an Abel-Ruffini
//! escape hatch. For `N >= 5`, it is a geometric warm start followed by an
//! ordinary digital polish on the much calmer core.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct LieSvdBlock4Params {
    /// Number of macro passes over contiguous and butterfly quartets.
    pub max_passes: usize,
    /// Relative off-diagonal target for the raw `4x4` macro stage.
    pub raw_tol: f64,
    /// Reject a local quartet if it increases global off-diagonal energy by
    /// more than this relative slack.
    pub accept_slack: f64,
    /// Include stride-1/2/4/... butterfly quartets. This is the power-of-two
    /// "look from the tensor tree" layer.
    pub include_butterfly: bool,
    /// Apply the phase-guided topological warm start before quartet layers.
    pub use_topological_warm_start: bool,
}

impl LieSvdBlock4Params {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_passes: (4 + n / 8).clamp(4, 24),
            raw_tol: 1e-10,
            accept_slack: 1e-12,
            include_butterfly: true,
            use_topological_warm_start: true,
        }
    }
}

impl Default for LieSvdBlock4Params {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdBlock4Trace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub passes: usize,
    pub accepted_blocks: usize,
    pub rejected_blocks: usize,
    pub butterfly_layers: usize,
    pub topo_accepted: bool,
    pub topo_offdiag: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Block4Signature {
    pub blocks: usize,
    pub skew_norm: f64,
    pub self_dual_norm: f64,
    pub anti_self_dual_norm: f64,
    pub dual_balance: f64,
}

pub struct LieSvdBlock4;

impl LieSvdBlock4 {
    /// Robust release route: `4x4` macro-relaxation followed by exact small
    /// polar/Jacobi polish on the transformed core.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        Self::solve_with_digital_polish(mat, LieSvdBlock4Params::for_n(mat.nrows())).0
    }

    pub fn solve_with_digital_polish(
        mat: &Array2<f64>,
        params: LieSvdBlock4Params,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdBlock4Trace) {
        let ((u0, _sigma0, vt0), trace) = Self::warm_start_with_trace(mat, params);
        let v0 = vt0.t().to_owned();
        let core = u0.t().dot(mat).dot(&v0);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&vt0);
        ((u, sigma, vt), trace)
    }

    /// Raw macro stage. It is useful for benchmarking the `4x4` idea itself.
    pub fn warm_start_with_trace(
        mat: &Array2<f64>,
        params: LieSvdBlock4Params,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdBlock4Trace) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdBlock4: matrix must be square");
        if n <= 4 {
            let (u, sigma, vt) = crate::lie_svd_micro::LieSvdMicro::solve(mat);
            let core = u.t().dot(mat).dot(&vt.t());
            let offdiag = offdiag_norm(&core);
            return (
                (u, sigma, vt),
                LieSvdBlock4Trace {
                    initial_offdiag: offdiag,
                    final_offdiag: offdiag,
                    passes: 0,
                    accepted_blocks: 0,
                    rejected_blocks: 0,
                    butterfly_layers: 0,
                    topo_accepted: false,
                    topo_offdiag: offdiag,
                },
            );
        }

        let mut work = mat.clone();
        let mut u_basis = Array2::<f64>::eye(n);
        let mut v_basis = Array2::<f64>::eye(n);
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let target = params.raw_tol * ref_norm.max(1.0);
        let initial_offdiag = offdiag_norm(&work);
        let mut topo_accepted = false;
        let mut topo_offdiag = initial_offdiag;
        if params.use_topological_warm_start {
            let mut warm_params = crate::lie_svd_topowarm::TopologicalWarmStartParams::for_n(n);
            warm_params.rank = n.min(12).max(1);
            warm_params.landmark_count = n.min(8).max(1);
            warm_params.phase_landmark_count = n.min(4).max(1);
            warm_params.power_steps = 1;
            let warm = crate::lie_svd_topowarm::compute_topological_warm_start(&work, warm_params);
            topo_accepted = warm.trace.accepted;
            topo_offdiag = warm.trace.warm_offdiag;
            if warm.trace.accepted {
                work = warm.core;
                u_basis = warm.u;
                v_basis = warm.v;
            }
        }
        let layers = block4_layers(n, params.include_butterfly);
        let butterfly_layers = layers
            .iter()
            .filter(|layer| layer.iter().any(|axes| !is_contiguous4(*axes)))
            .count();

        let mut accepted_blocks = 0usize;
        let mut rejected_blocks = 0usize;
        let mut passes_done = 0usize;

        for pass in 0..params.max_passes.max(1) {
            passes_done = pass + 1;
            if offdiag_norm(&work) <= target {
                break;
            }
            let mut changed = false;
            for layer in &layers {
                for &axes in layer {
                    let before = offdiag_sq(&work);
                    let block = extract_block4(&work, axes);
                    let (ub, _sigma, vtb) = crate::lie_svd_micro::LieSvdMicro::solve(&block);
                    let vb = vtb.t().to_owned();

                    apply_left_block_transform(&mut work, &mut u_basis, axes, &ub);
                    apply_right_block_transform(&mut work, &mut v_basis, axes, &vb);
                    let after = offdiag_sq(&work);

                    if after <= before * (1.0 + params.accept_slack) {
                        accepted_blocks += 1;
                        changed = true;
                    } else {
                        apply_left_block_transform(
                            &mut work,
                            &mut u_basis,
                            axes,
                            &ub.t().to_owned(),
                        );
                        apply_right_block_transform(
                            &mut work,
                            &mut v_basis,
                            axes,
                            &vb.t().to_owned(),
                        );
                        rejected_blocks += 1;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let final_offdiag = offdiag_norm(&work);
        (
            extract_sorted_svd(&work, &u_basis, &v_basis),
            LieSvdBlock4Trace {
                initial_offdiag,
                final_offdiag,
                passes: passes_done,
                accepted_blocks,
                rejected_blocks,
                butterfly_layers,
                topo_accepted,
                topo_offdiag,
            },
        )
    }
}

/// Splits contiguous `4x4` skew/torsion blocks into the two `SO(4)` halves:
/// self-dual and anti-self-dual bivectors. This is a diagnostic/triage signal,
/// not a replacement for the SVD itself.
pub fn analyze_block4_signature(mat: &Array2<f64>) -> Block4Signature {
    let n = mat.nrows().min(mat.ncols());
    let blocks = n / 4;
    if blocks == 0 {
        return Block4Signature {
            blocks: 0,
            skew_norm: 0.0,
            self_dual_norm: 0.0,
            anti_self_dual_norm: 0.0,
            dual_balance: 0.0,
        };
    }

    let mut skew_sq = 0.0_f64;
    let mut self_dual_sq = 0.0_f64;
    let mut anti_self_dual_sq = 0.0_f64;
    for block in 0..blocks {
        let start = block * 4;
        let w12 = skew_component(mat, start, 0, 1);
        let w13 = skew_component(mat, start, 0, 2);
        let w14 = skew_component(mat, start, 0, 3);
        let w23 = skew_component(mat, start, 1, 2);
        let w24 = skew_component(mat, start, 1, 3);
        let w34 = skew_component(mat, start, 2, 3);

        skew_sq += w12 * w12 + w13 * w13 + w14 * w14 + w23 * w23 + w24 * w24 + w34 * w34;

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let sd = [
            (w12 + w34) * inv_sqrt2,
            (w13 - w24) * inv_sqrt2,
            (w14 + w23) * inv_sqrt2,
        ];
        let asd = [
            (w12 - w34) * inv_sqrt2,
            (w13 + w24) * inv_sqrt2,
            (w14 - w23) * inv_sqrt2,
        ];
        self_dual_sq += sd.iter().map(|x| x * x).sum::<f64>();
        anti_self_dual_sq += asd.iter().map(|x| x * x).sum::<f64>();
    }

    let self_dual_norm = self_dual_sq.sqrt();
    let anti_self_dual_norm = anti_self_dual_sq.sqrt();
    Block4Signature {
        blocks,
        skew_norm: skew_sq.sqrt(),
        self_dual_norm,
        anti_self_dual_norm,
        dual_balance: (self_dual_norm - anti_self_dual_norm).abs()
            / (self_dual_norm + anti_self_dual_norm).max(1e-300),
    }
}

fn skew_component(mat: &Array2<f64>, start: usize, a: usize, b: usize) -> f64 {
    0.5 * (mat[[start + b, start + a]] - mat[[start + a, start + b]])
}

fn block4_layers(n: usize, include_butterfly: bool) -> Vec<Vec<[usize; 4]>> {
    let mut layers = Vec::new();
    for offset in [0usize, 2usize] {
        let mut layer = Vec::new();
        let mut start = offset;
        while start + 3 < n {
            layer.push([start, start + 1, start + 2, start + 3]);
            start += 4;
        }
        if !layer.is_empty() {
            layers.push(layer);
        }
    }

    if include_butterfly {
        let mut stride = 1usize;
        while 3 * stride < n {
            let span = 4 * stride;
            let mut layer = Vec::new();
            let mut chunk = 0usize;
            while chunk + 3 * stride < n {
                for base in 0..stride {
                    let axes = [
                        chunk + base,
                        chunk + base + stride,
                        chunk + base + 2 * stride,
                        chunk + base + 3 * stride,
                    ];
                    if axes[3] < n && !is_contiguous4(axes) {
                        layer.push(axes);
                    }
                }
                chunk += span;
            }
            if !layer.is_empty() {
                layers.push(layer);
            }
            stride *= 2;
        }
    }
    layers
}

fn is_contiguous4(axes: [usize; 4]) -> bool {
    axes[1] == axes[0] + 1 && axes[2] == axes[1] + 1 && axes[3] == axes[2] + 1
}

fn extract_block4(work: &Array2<f64>, axes: [usize; 4]) -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(i, j)| work[[axes[i], axes[j]]])
}

fn apply_left_block_transform(
    work: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    axes: [usize; 4],
    q: &Array2<f64>,
) {
    let n = work.nrows();
    let mut tmp = [0.0_f64; 4];
    for col in 0..n {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += q[[b, a]] * work[[axes[b], col]];
            }
        }
        for a in 0..4 {
            work[[axes[a], col]] = tmp[a];
        }
    }
    for row in 0..n {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += u_basis[[row, axes[b]]] * q[[b, a]];
            }
        }
        for a in 0..4 {
            u_basis[[row, axes[a]]] = tmp[a];
        }
    }
}

fn apply_right_block_transform(
    work: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    axes: [usize; 4],
    q: &Array2<f64>,
) {
    let n = work.nrows();
    let mut tmp = [0.0_f64; 4];
    for row in 0..n {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += work[[row, axes[b]]] * q[[b, a]];
            }
        }
        for a in 0..4 {
            work[[row, axes[a]]] = tmp[a];
        }
    }
    for row in 0..n {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += v_basis[[row, axes[b]]] * q[[b, a]];
            }
        }
        for a in 0..4 {
            v_basis[[row, axes[a]]] = tmp[a];
        }
    }
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn offdiag_norm(a: &Array2<f64>) -> f64 {
    offdiag_sq(a).sqrt()
}

fn offdiag_sq(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s += a[[i, j]] * a[[i, j]];
            }
        }
    }
    s
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
    fn block4_raw_reduces_block_diagonal_cells() {
        let mut rng = StdRng::seed_from_u64(1601);
        let mut a = Array2::<f64>::zeros((8, 8));
        for block in 0..2 {
            let start = block * 4;
            let cell = Array2::from_shape_fn((4, 4), |_| rng.gen_range(-2.0_f64..2.0));
            for i in 0..4 {
                for j in 0..4 {
                    a[[start + i, start + j]] = cell[[i, j]];
                }
            }
        }
        let before = offdiag_norm(&a);
        let mut params = LieSvdBlock4Params::for_n(8);
        params.use_topological_warm_start = false;
        let ((_u, _sigma, _vt), trace) = LieSvdBlock4::warm_start_with_trace(&a, params);
        assert!(trace.final_offdiag < before * 1e-8, "{trace:?}");
        assert!(trace.accepted_blocks > 0);
    }

    #[test]
    fn block4_polished_is_accurate_on_random_power_of_two() {
        let n = 8;
        let mut rng = StdRng::seed_from_u64(1602);
        for _ in 0..8 {
            let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
            let ((u, sigma, vt), trace) =
                LieSvdBlock4::solve_with_digital_polish(&a, LieSvdBlock4Params::for_n(n));
            let (rel, ou, ov) = metrics(&a, &u, &sigma, &vt);
            assert!(trace.accepted_blocks + trace.rejected_blocks > 0);
            assert!(rel < 1e-10, "rel={rel}");
            assert!(ou < 1e-10, "orth_u={ou}");
            assert!(ov < 1e-10, "orth_v={ov}");
        }
    }

    #[test]
    fn block4_layers_include_butterfly_for_power_of_two() {
        let layers = block4_layers(16, true);
        assert!(layers.iter().any(|layer| layer.contains(&[0, 4, 8, 12])));
        assert!(layers.iter().any(|layer| layer.contains(&[1, 5, 9, 13])));
    }

    #[test]
    fn block4_signature_splits_self_dual_and_anti_self_dual_parts() {
        let mut a = Array2::<f64>::zeros((4, 4));
        a[[1, 0]] = 1.0;
        a[[0, 1]] = -1.0;
        a[[3, 2]] = 1.0;
        a[[2, 3]] = -1.0;
        let sig = analyze_block4_signature(&a);
        assert_eq!(sig.blocks, 1);
        assert!(sig.self_dual_norm > 1.0);
        assert!(sig.anti_self_dual_norm < 1e-12);
        assert!(sig.dual_balance > 0.999);
    }
}
