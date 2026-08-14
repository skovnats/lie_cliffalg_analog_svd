//! Analog-oriented SVD prototype.
//!
//! This module is a CPU simulator for a future mixed-signal/photonic SVD
//! substrate. The guiding assumption is different from the ODE/flow attempts:
//! orthogonality is not something we numerically enforce after the fact; it is
//! the native invariant of a programmable rotation mesh. The matrix is routed
//! through conflict-free layers of local `2x2` SVD cells, each cell applying a
//! left and right rotor. On analog hardware those cells map naturally to
//! MZI/CORDIC-like phase shifters plus a diagonal gain/crossbar stage.
//!
//! The implementation below stays plain `f64`: it is meant to be inspectable,
//! deterministic, and benchmarkable on today's CPUs while preserving the shape
//! of a possible analog-chip schedule.

use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug)]
pub struct LieSvdAnalogParams {
    /// Number of global round-robin sweeps over all conflict-free pair layers.
    pub max_sweeps: usize,
    /// Relative off-diagonal stopping tolerance.
    pub tol: f64,
    /// Skip local cells whose `|A_ij| + |A_ji|` is below this relative scale.
    pub min_pair_scale: f64,
    /// Optional deterministic DAC phase quantization. `None` is ideal analog.
    pub angle_dac_bits: Option<u32>,
}

impl LieSvdAnalogParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_sweeps: 16 + 4 * n.max(1),
            tol: 1e-12,
            min_pair_scale: 1e-14,
            angle_dac_bits: None,
        }
    }
}

impl Default for LieSvdAnalogParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdAnalogTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub layers: usize,
    pub rotations: usize,
    pub skipped_cells: usize,
    pub quantized_rotations: usize,
    pub samples: Vec<f64>,
}

pub struct LieSvdAnalog;

impl LieSvdAnalog {
    /// Ideal analog rotor mesh simulation.
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let params = LieSvdAnalogParams::for_n(mat.nrows());
        Self::solve_with_trace(mat, params, params.max_sweeps / 32).0
    }

    /// Same solver with diagnostics for hardware/schedule experiments.
    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdAnalogParams,
        sample_every: usize,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdAnalogTrace) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdAnalog: matrix must be square");

        let mut work = mat.clone();
        let mut u_basis = Array2::<f64>::eye(n);
        let mut v_basis = Array2::<f64>::eye(n);
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let tol = params.tol * ref_norm.max(1.0);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let initial_offdiag = direct_offdiag_norm(&work);
        let mut samples = Vec::new();
        let sample_every = sample_every.max(1);
        let mut sweeps = 0usize;
        let mut layers = 0usize;
        let mut rotations = 0usize;
        let mut skipped_cells = 0usize;
        let mut quantized_rotations = 0usize;

        if n <= 1 {
            let sigma = Array1::from_shape_fn(n, |i| work[[i, i]].abs());
            return (
                (u_basis, sigma, v_basis.t().to_owned()),
                LieSvdAnalogTrace {
                    initial_offdiag,
                    final_offdiag: 0.0,
                    sweeps,
                    layers,
                    rotations,
                    skipped_cells,
                    quantized_rotations,
                    samples,
                },
            );
        }

        let layer_count = round_robin_layer_count(n);
        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let energy = direct_offdiag_norm(&work);
            if sweep % sample_every == 0 {
                samples.push(energy);
            }
            if energy <= tol || !energy.is_finite() {
                break;
            }

            let mut changed = false;
            for layer in 0..layer_count {
                layers += 1;
                for (i, j) in analog_layer_pairs(n, layer) {
                    if direct_pair_offdiag(&work, i, j) <= pair_tol {
                        skipped_cells += 1;
                        continue;
                    }
                    let (mut theta_l, mut theta_r) = local_pair_svd_angles(&work, i, j);
                    if let Some(bits) = params.angle_dac_bits {
                        theta_l = quantize_phase(theta_l, bits);
                        theta_r = quantize_phase(theta_r, bits);
                        quantized_rotations += 1;
                    }
                    if theta_l.abs() + theta_r.abs() <= 1e-18 {
                        skipped_cells += 1;
                        continue;
                    }
                    apply_left_rotor(&mut work, &mut u_basis, i, j, theta_l);
                    apply_right_rotor(&mut work, &mut v_basis, i, j, theta_r);
                    rotations += 1;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let final_offdiag = direct_offdiag_norm(&work);
        let (u, sigma, vt) = extract_sorted_svd(&work, &u_basis, &v_basis);
        (
            (u, sigma, vt),
            LieSvdAnalogTrace {
                initial_offdiag,
                final_offdiag,
                sweeps,
                layers,
                rotations,
                skipped_cells,
                quantized_rotations,
                samples,
            },
        )
    }

    /// Mixed-signal mode: use the analog mesh as a basis preconditioner, then
    /// polish the remaining digital core. This mirrors the realistic hardware
    /// caveat: analog applies very cheap rotations, while a small digital audit
    /// path can finish accuracy-critical work.
    pub fn solve_with_digital_polish(
        mat: &Array2<f64>,
        params: LieSvdAnalogParams,
        sample_every: usize,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), LieSvdAnalogTrace) {
        let ((u0, _sigma0, vt0), trace) = Self::solve_with_trace(mat, params, sample_every);
        let v0 = vt0.t().to_owned();
        let core = u0.t().dot(mat).dot(&v0);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&vt0);
        ((u, sigma, vt), trace)
    }
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
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

fn direct_pair_offdiag(work: &Array2<f64>, i: usize, j: usize) -> f64 {
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

fn quantize_phase(theta: f64, bits: u32) -> f64 {
    let bits = bits.min(30);
    let levels = (1_u64 << bits).max(2) as f64;
    let step = std::f64::consts::PI / levels;
    (theta / step).round() * step
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

fn round_robin_layer_count(n: usize) -> usize {
    if n % 2 == 0 {
        n.saturating_sub(1)
    } else {
        n
    }
}

fn analog_layer_pairs(n: usize, layer: usize) -> Vec<(usize, usize)> {
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
    fn test_analog_polished_random_16() {
        let n = 16;
        let mut rng = StdRng::seed_from_u64(77);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let params = LieSvdAnalogParams {
            max_sweeps: 24,
            ..LieSvdAnalogParams::for_n(n)
        };
        let ((u, sigma, vt), trace) = LieSvdAnalog::solve_with_digital_polish(&a, params, 4);
        let (rel, ou, ov) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.rotations > 0);
        assert!(rel < 1e-10, "rel={rel}");
        assert!(ou < 1e-8, "orth_u={ou}");
        assert!(ov < 1e-8, "orth_v={ov}");
    }

    #[test]
    fn test_analog_layer_pairs_are_conflict_free() {
        for n in 2..17 {
            for layer in 0..round_robin_layer_count(n) {
                let mut used = vec![false; n];
                for (i, j) in analog_layer_pairs(n, layer) {
                    assert!(i < n && j < n && i != j);
                    assert!(!used[i], "n={n} layer={layer} repeated i={i}");
                    assert!(!used[j], "n={n} layer={layer} repeated j={j}");
                    used[i] = true;
                    used[j] = true;
                }
            }
        }
    }
}
