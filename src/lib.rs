//! Lie/Clifford Analog SVD.
//!
//! This crate is a compact Linux/CPU release of the research SVD line:
//! a robust polar/Jacobi solver, a dual-tiled Lie/Clifford preconditioner,
//! an active row/column phase-locking solver, a Phase-JADE joint
//! diagonalizer, a complex-native phase branch, and an
//! analog-hardware-oriented rotor mesh simulator.
//!
//! The main SVD API returns `(U, sigma, Vt)` for square dense `f64` matrices;
//! phase diagnostics also include rectangular row/column-space routes. The
//! solvers intentionally keep ordinary `ndarray::Array2<f64>` storage:
//! Clifford and analog-chip ideas appear as phase passports, rotor schedules,
//! and invariants, not as a new heap representation.

pub mod kernel_gram;
pub mod lie_svd;
pub mod lie_svd_adaptive;
pub mod lie_svd_analog;
pub mod lie_svd_block4;
pub mod lie_svd_bss;
pub mod lie_svd_compiler;
pub mod lie_svd_complex;
pub mod lie_svd_coreflow;
pub mod lie_svd_engine;
pub mod lie_svd_hybrid;
pub mod lie_svd_joint;
pub mod lie_svd_micro;
pub mod lie_svd_phaseflow;
pub mod lie_svd_phasehealth;
pub mod lie_svd_quadenergy;
pub mod lie_svd_small;
pub mod lie_svd_subspace_jade;
pub mod lie_svd_tensor;
pub mod lie_svd_tensortrain;
pub mod lie_svd_topowarm;
pub mod lie_svd_traceflow;
pub mod lie_tbl_multivector;
pub mod lie_tbl_regress;

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::time::{Duration, Instant};

pub type SvdTriple = (Array2<f64>, Array1<f64>, Array2<f64>);
pub type TimedSvd = (Array2<f64>, Array1<f64>, Array2<f64>, Duration);

pub mod solvers {
    use super::*;

    pub fn run_small(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_small::LieSvdSmall::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_hybrid(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_hybrid::LieSvdHybrid::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_analog_polished(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let params = crate::lie_svd_analog::LieSvdAnalogParams::for_n(a.nrows());
        let ((u, s, vt), _trace) =
            crate::lie_svd_analog::LieSvdAnalog::solve_with_digital_polish(a, params, 16);
        (u, s, vt, start.elapsed())
    }

    pub fn run_coreflow(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_coreflow::LieSvdCoreFlow::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_block4(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_block4::LieSvdBlock4::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_coreflow_with_params(
        a: &Array2<f64>,
        params: crate::lie_svd_coreflow::LieSvdCoreFlowParams,
    ) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_coreflow::LieSvdCoreFlow::solve_with_trace(a, params, 16).0;
        (u, s, vt, start.elapsed())
    }

    pub fn run_traceflow(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_traceflow::LieSvdTraceFlow::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_phaseflow(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_phaseflow::LieSvdPhaseFlow::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_phaseflow_polished(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let params = crate::lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(a.nrows());
        let ((u, s, vt), _) =
            crate::lie_svd_phaseflow::LieSvdPhaseFlow::solve_with_digital_polish(a, params);
        (u, s, vt, start.elapsed())
    }

    pub fn run_auto(a: &Array2<f64>) -> TimedSvd {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd::LieSvd::solve(a);
        (u, s, vt, start.elapsed())
    }

    pub fn run_kron_chain(a: &Array2<f64>) -> Option<TimedSvd> {
        let start = Instant::now();
        let (u, s, vt) = crate::lie_svd_tensortrain::solve_if_kron_chain(
            a,
            crate::lie_svd_tensortrain::TensorTrainSvdParams::default(),
        )?;
        Some((u, s, vt, start.elapsed()))
    }
}

pub mod metrics {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub struct SvdMetrics {
        pub rel_recon: f64,
        pub orth_u: f64,
        pub orth_v: f64,
        pub sigma_max_rel: Option<f64>,
        pub sigma_tail_rel: Option<f64>,
    }

    pub fn frobenius_norm(a: &Array2<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn compute(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
        sigma_ref: Option<&Array1<f64>>,
    ) -> SvdMetrics {
        let n = a.nrows();
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel_recon = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(n);
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();

        let (sigma_max_rel, sigma_tail_rel) = match sigma_ref {
            Some(want) => {
                let mut got = sigma.to_vec();
                let mut want = want.to_vec();
                got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                want.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let max_rel = got
                    .iter()
                    .zip(want.iter())
                    .map(|(g, w)| (g - w).abs() / w.abs().max(1e-300))
                    .fold(0.0_f64, f64::max);
                let tail_start = n.saturating_mul(3) / 4;
                let tail_rel = got
                    .iter()
                    .zip(want.iter())
                    .skip(tail_start)
                    .map(|(g, w)| (g - w).abs() / w.abs().max(1e-300))
                    .fold(0.0_f64, f64::max);
                (Some(max_rel), Some(tail_rel))
            }
            None => (None, None),
        };

        SvdMetrics {
            rel_recon,
            orth_u,
            orth_v,
            sigma_max_rel,
            sigma_tail_rel,
        }
    }
}

pub mod profiles {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub enum Profile {
        UniformRandom,
        DegenerateSpectrum,
        ExtremeIllConditioned,
        JordanDefective,
        SparseStructured,
        NearlyDiagonal,
        KronStructured,
    }

    impl Profile {
        pub fn name(self) -> &'static str {
            match self {
                Profile::UniformRandom => "uniform_random",
                Profile::DegenerateSpectrum => "degenerate_spectrum",
                Profile::ExtremeIllConditioned => "extreme_ill_conditioned",
                Profile::JordanDefective => "jordan_defective",
                Profile::SparseStructured => "sparse_structured",
                Profile::NearlyDiagonal => "nearly_diagonal",
                Profile::KronStructured => "kron_structured",
            }
        }
    }

    pub struct MatrixCase {
        pub a: Array2<f64>,
        pub sigma_ref: Option<Array1<f64>>,
    }

    pub fn generate(n: usize, profile: Profile, seed: u64) -> MatrixCase {
        let mut rng = StdRng::seed_from_u64(seed);
        match profile {
            Profile::UniformRandom => MatrixCase {
                a: Array2::from_shape_fn((n, n), |_| StandardNormal.sample(&mut rng)),
                sigma_ref: None,
            },
            Profile::DegenerateSpectrum => {
                let u = random_orthogonal(n, &mut rng);
                let v = random_orthogonal(n, &mut rng);
                let sigma: Vec<f64> = (0..n)
                    .map(|i| {
                        if i < n / 8 {
                            100.0
                        } else if i < n / 4 {
                            50.0
                        } else if i < n / 2 {
                            1.0
                        } else {
                            1e-12
                        }
                    })
                    .collect();
                MatrixCase {
                    a: compose_from_svd(&u, &sigma, &v),
                    sigma_ref: Some(Array1::from(sigma)),
                }
            }
            Profile::ExtremeIllConditioned => {
                let u = random_orthogonal(n, &mut rng);
                let v = random_orthogonal(n, &mut rng);
                let denom = (n.saturating_sub(1)).max(1) as f64;
                let sigma: Vec<f64> = (0..n)
                    .map(|i| 10f64.powf(-18.0 * i as f64 / denom))
                    .collect();
                MatrixCase {
                    a: compose_from_svd(&u, &sigma, &v),
                    sigma_ref: Some(Array1::from(sigma)),
                }
            }
            Profile::JordanDefective => {
                let mut a = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    a[[i, i]] = 1.0 - 0.5 * (i as f64 / n.max(1) as f64);
                    if i + 1 < n {
                        a[[i, i + 1]] = 20.0;
                    }
                    if i + 2 < n && i % 4 == 0 {
                        a[[i, i + 2]] = -3.0;
                    }
                }
                MatrixCase { a, sigma_ref: None }
            }
            Profile::SparseStructured => {
                let mut a = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    a[[i, i]] = 4.0 + (i % 7) as f64 * 0.25;
                    if i + 1 < n {
                        a[[i, i + 1]] = -1.0;
                        a[[i + 1, i]] = 0.75;
                    }
                    if i + 8 < n {
                        a[[i, i + 8]] = 0.15;
                        a[[i + 8, i]] = -0.10;
                    }
                }
                MatrixCase { a, sigma_ref: None }
            }
            Profile::NearlyDiagonal => {
                let mut a = Array2::<f64>::zeros((n, n));
                let sigma: Vec<f64> = (0..n).map(|i| 1.0 + (n - i) as f64 / n as f64).collect();
                for i in 0..n {
                    a[[i, i]] = sigma[i];
                    for j in 0..n {
                        if i != j {
                            a[[i, j]] = 1e-6 * ((i * 131 + j * 17) as f64).sin();
                        }
                    }
                }
                MatrixCase {
                    a,
                    sigma_ref: Some(Array1::from(sigma)),
                }
            }
            Profile::KronStructured => MatrixCase {
                a: kron_structured_matrix(n),
                sigma_ref: None,
            },
        }
    }

    fn random_orthogonal(n: usize, rng: &mut StdRng) -> Array2<f64> {
        let mut m = Array2::from_shape_fn((n, n), |_| StandardNormal.sample(rng));
        for j in 0..n {
            for k in 0..j {
                let dot: f64 = (0..n).map(|i| m[[i, j]] * m[[i, k]]).sum();
                for i in 0..n {
                    m[[i, j]] -= dot * m[[i, k]];
                }
            }
            let norm = (0..n)
                .map(|i| m[[i, j]] * m[[i, j]])
                .sum::<f64>()
                .sqrt()
                .max(1e-300);
            for i in 0..n {
                m[[i, j]] /= norm;
            }
        }
        m
    }

    fn compose_from_svd(u: &Array2<f64>, sigma: &[f64], v: &Array2<f64>) -> Array2<f64> {
        let sigma_mat = Array2::from_diag(&Array1::from(sigma.to_vec()));
        u.dot(&sigma_mat).dot(&v.t())
    }

    fn kron_structured_matrix(n: usize) -> Array2<f64> {
        if n == 0 {
            return Array2::<f64>::zeros((0, 0));
        }
        if n == 1 {
            return Array2::from_elem((1, 1), 2.0);
        }
        if n.is_power_of_two() {
            let levels = n.trailing_zeros() as usize;
            let mut factors = Vec::with_capacity(levels);
            for k in 0..levels {
                let t = 0.17 * (k as f64 + 1.0);
                factors.push(
                    Array2::from_shape_vec(
                        (2, 2),
                        vec![
                            1.6 + 0.09 * k as f64,
                            0.25 * t.sin(),
                            -0.18 * t.cos(),
                            0.9 + 0.05 * k as f64,
                        ],
                    )
                    .expect("2x2 factor"),
                );
            }
            return crate::lie_svd_tensortrain::kron_chain_product(&factors);
        }

        let p = n.next_power_of_two() / 2;
        let mut out = Array2::<f64>::zeros((n, n));
        let core = kron_structured_matrix(p);
        for i in 0..p {
            for j in 0..p {
                out[[i, j]] = core[[i, j]];
            }
        }
        for i in p..n {
            out[[i, i]] = 1.0 + (i - p) as f64 * 0.01;
        }
        out
    }
}
