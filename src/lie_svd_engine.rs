//! Unified phase dispatcher.
//!
//! `PhaseEngine` is the 0.22.0 facade: it keeps the individual solvers
//! separate, but gives them one diagnostic language (`PhasePassport`) and one
//! route report.  That keeps the API honest while making the ecosystem easier
//! to drive from benchmarks, notebooks, and future hardware compilers.

use crate::lie_svd_bss::{LieSvdBss, PhaseBssParams, PhaseBssResult};
use crate::lie_svd_complex::{
    complex_relative_reconstruction_error, complex_unitarity_error, LieSvdComplex,
    LieSvdComplexParams,
};
use crate::lie_svd_joint::{JointDiagonalizationParams, JointDiagonalizationTrace, LieSvdJoint};
use crate::lie_svd_phaseflow::{LieSvdPhaseFlow, LieSvdPhaseFlowParams};
use crate::lie_svd_phasehealth::phase_signature;
use crate::lie_svd_small::LieSvdSmall;
use crate::lie_svd_tensor::{LieSvdTensor, TensorPhaseSvd3};
use crate::metrics;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseRoute {
    RealSmall,
    RealPhaseFlowPolished,
    ComplexPhaseFlow,
    TensorHosvd3,
    PhaseBss,
    JointPhaseJade,
}

#[derive(Clone, Debug)]
pub struct PhasePassport {
    pub rows: usize,
    pub cols: usize,
    pub family: usize,
    pub tensor_modes: Option<[usize; 3]>,
    pub mean_stress: f64,
    pub max_twist: f64,
    pub causal_disbalance: f64,
    pub entropy_gap: f64,
    pub chirality: f64,
    pub golden_resonance: f64,
    /// Mass-weighted global phase angle. See
    /// `lie_svd_phasehealth::GlobalPhaseInvariants`.
    pub global_phase: f64,
    /// `H_total = ||skew(A)||_F`, raw (unnormalized) torsion energy. Unlike
    /// `chirality` above (which is this quantity normalized by `||A||_F`),
    /// this keeps the absolute scale.
    pub torsion_energy: f64,
    /// Self-dual/anti-self-dual bivector balance aggregated over contiguous
    /// `4x4` blocks (`lie_svd_block4::analyze_block4_signature`).
    pub chirality_balance: f64,
    /// Normalized Shannon entropy of the whole matrix's energy distribution,
    /// in `[0, 1]`.
    pub phase_entropy: f64,
    pub route_hint: PhaseRoute,
}

#[derive(Clone, Debug)]
pub struct PhaseEngineReport {
    pub passport: PhasePassport,
    pub route: PhaseRoute,
    pub time_s: f64,
    pub rel_recon: Option<f64>,
    pub orth_u: Option<f64>,
    pub orth_v: Option<f64>,
    pub phase_stress: Option<(f64, f64)>,
    pub schedule_events: usize,
}

pub struct PhaseEngine;

impl PhaseEngine {
    pub fn passport_real(a: &Array2<f64>) -> PhasePassport {
        let sig = phase_signature(a);
        let route_hint = if a.nrows() == a.ncols()
            && (sig.max_twist > 0.8 || sig.causal_disbalance > 0.65 || sig.entropy_gap > 0.2)
            && a.nrows() <= 128
        {
            PhaseRoute::RealPhaseFlowPolished
        } else {
            PhaseRoute::RealSmall
        };
        PhasePassport {
            rows: a.nrows(),
            cols: a.ncols(),
            family: 1,
            tensor_modes: None,
            mean_stress: sig.mean_stress,
            max_twist: sig.max_twist,
            causal_disbalance: sig.causal_disbalance,
            entropy_gap: sig.entropy_gap,
            chirality: real_chirality(a),
            golden_resonance: golden_resonance_real(a),
            global_phase: sig.global.global_phase,
            torsion_energy: sig.global.torsion_energy,
            chirality_balance: sig.global.chirality_balance,
            phase_entropy: sig.global.phase_entropy,
            route_hint,
        }
    }

    pub fn solve_real(
        a: &Array2<f64>,
    ) -> ((Array2<f64>, Array1<f64>, Array2<f64>), PhaseEngineReport) {
        let passport = Self::passport_real(a);
        let route = passport.route_hint;
        let start = Instant::now();
        let (svd, phase_stress, events) = match route {
            PhaseRoute::RealPhaseFlowPolished => {
                let mut params = LieSvdPhaseFlowParams::for_n(a.nrows());
                params.record_mzi_phases = true;
                let (svd, trace) = LieSvdPhaseFlow::solve_with_digital_polish(a, params);
                (
                    svd,
                    Some((trace.initial_phase_stress, trace.final_phase_stress)),
                    trace.mzi_phases.len(),
                )
            }
            _ => (LieSvdSmall::solve(a), None, 0),
        };
        let metrics = metrics::compute(a, &svd.0, &svd.1, &svd.2, None);
        let report = PhaseEngineReport {
            passport,
            route,
            time_s: start.elapsed().as_secs_f64(),
            rel_recon: Some(metrics.rel_recon),
            orth_u: Some(metrics.orth_u),
            orth_v: Some(metrics.orth_v),
            phase_stress,
            schedule_events: events,
        };
        (svd, report)
    }

    pub fn passport_complex(a: &Array2<Complex64>) -> PhasePassport {
        let magnitude = a.mapv(|z| z.norm());
        let sig = phase_signature(&magnitude);
        PhasePassport {
            rows: a.nrows(),
            cols: a.ncols(),
            family: 1,
            tensor_modes: None,
            mean_stress: sig.mean_stress,
            max_twist: sig.max_twist,
            causal_disbalance: sig.causal_disbalance,
            entropy_gap: sig.entropy_gap,
            chirality: complex_chirality(a),
            golden_resonance: golden_resonance_complex(a),
            global_phase: sig.global.global_phase,
            torsion_energy: sig.global.torsion_energy,
            chirality_balance: sig.global.chirality_balance,
            phase_entropy: sig.global.phase_entropy,
            route_hint: PhaseRoute::ComplexPhaseFlow,
        }
    }

    pub fn solve_complex(
        a: &Array2<Complex64>,
    ) -> (
        (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
        PhaseEngineReport,
    ) {
        let passport = Self::passport_complex(a);
        let start = Instant::now();
        let params = LieSvdComplexParams::for_n(a.nrows());
        let (svd, trace) = LieSvdComplex::solve_with_trace(a, params);
        let v = svd.2.t().mapv(|x| x.conj());
        let report = PhaseEngineReport {
            passport,
            route: PhaseRoute::ComplexPhaseFlow,
            time_s: start.elapsed().as_secs_f64(),
            rel_recon: Some(complex_relative_reconstruction_error(
                a, &svd.0, &svd.1, &svd.2,
            )),
            orth_u: Some(complex_unitarity_error(&svd.0)),
            orth_v: Some(complex_unitarity_error(&v)),
            phase_stress: Some((trace.initial_phase_stress, trace.final_phase_stress)),
            schedule_events: trace.mzi_phases.len(),
        };
        (svd, report)
    }

    pub fn hosvd3(tensor: &Array3<f64>) -> (TensorPhaseSvd3, PhaseEngineReport) {
        let start = Instant::now();
        let fact = LieSvdTensor::hosvd3(tensor);
        let shape = tensor.dim();
        let report = PhaseEngineReport {
            passport: PhasePassport {
                rows: shape.0,
                cols: shape.1,
                family: 1,
                tensor_modes: Some([shape.0, shape.1, shape.2]),
                mean_stress: fact.trace.final_offdiag,
                max_twist: fact.trace.initial_offdiag,
                causal_disbalance: 0.0,
                entropy_gap: 0.0,
                chirality: 0.0,
                golden_resonance: 0.0,
                global_phase: 0.0,
                torsion_energy: 0.0,
                chirality_balance: 0.0,
                phase_entropy: 0.0,
                route_hint: PhaseRoute::TensorHosvd3,
            },
            route: PhaseRoute::TensorHosvd3,
            time_s: start.elapsed().as_secs_f64(),
            rel_recon: None,
            orth_u: None,
            orth_v: None,
            phase_stress: Some((fact.trace.initial_offdiag, fact.trace.final_offdiag)),
            schedule_events: 0,
        };
        (fact, report)
    }

    pub fn separate_bss(observations: &Array2<f64>) -> (PhaseBssResult, PhaseEngineReport) {
        let start = Instant::now();
        let result = LieSvdBss::separate(
            observations,
            PhaseBssParams::for_channels(observations.nrows()),
        );
        let passport = Self::passport_real(observations);
        let report = PhaseEngineReport {
            route: PhaseRoute::PhaseBss,
            time_s: start.elapsed().as_secs_f64(),
            rel_recon: None,
            orth_u: None,
            orth_v: None,
            phase_stress: Some((
                result.trace.joint_initial_offdiag,
                result.trace.joint_final_offdiag,
            )),
            schedule_events: result.trace.joint_rotations,
            passport: PhasePassport {
                route_hint: PhaseRoute::PhaseBss,
                family: observations.nrows(),
                ..passport
            },
        };
        (result, report)
    }

    pub fn diagonalize_family(
        matrices: &[Array2<f64>],
    ) -> (
        Array2<f64>,
        Vec<Array1<f64>>,
        JointDiagonalizationTrace,
        PhaseEngineReport,
    ) {
        assert!(!matrices.is_empty(), "PhaseEngine: empty matrix family");
        let start = Instant::now();
        let params = JointDiagonalizationParams::for_n(matrices[0].nrows());
        let (v, diagonals, trace) =
            LieSvdJoint::diagonalize_symmetric_with_params(matrices, params);
        let mut passport = Self::passport_real(&matrices[0]);
        passport.family = matrices.len();
        passport.route_hint = PhaseRoute::JointPhaseJade;
        let report = PhaseEngineReport {
            passport,
            route: PhaseRoute::JointPhaseJade,
            time_s: start.elapsed().as_secs_f64(),
            rel_recon: None,
            orth_u: None,
            orth_v: Some(metrics::frobenius_norm(
                &(v.t().dot(&v) - Array2::<f64>::eye(v.ncols())),
            )),
            phase_stress: Some((trace.initial_offdiag, trace.final_offdiag)),
            schedule_events: trace.rotations,
        };
        (v, diagonals, trace, report)
    }
}

fn real_chirality(a: &Array2<f64>) -> f64 {
    let n = a.nrows().min(a.ncols());
    let mut skew = 0.0;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            let x = a[[i, j]];
            total += x * x;
            if i < j {
                let d = a[[i, j]] - a[[j, i]];
                skew += d * d;
            }
        }
    }
    skew.sqrt() / total.sqrt().max(1e-300)
}

fn complex_chirality(a: &Array2<Complex64>) -> f64 {
    let n = a.nrows().min(a.ncols());
    let mut skew = 0.0;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            total += a[[i, j]].norm_sqr();
            if i < j {
                skew += (a[[i, j]] - a[[j, i]].conj()).norm_sqr();
            }
        }
    }
    skew.sqrt() / total.sqrt().max(1e-300)
}

fn golden_resonance_real(a: &Array2<f64>) -> f64 {
    let theta = 2.39996322972865332_f64;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut mass = 0.0;
    for i in 0..a.nrows() {
        let phase = theta * i as f64;
        let (s, c) = phase.sin_cos();
        let row_mass = a.row(i).iter().map(|x| x.abs()).sum::<f64>();
        re += row_mass * c;
        im += row_mass * s;
        mass += row_mass;
    }
    (re * re + im * im).sqrt() / mass.max(1e-300)
}

fn golden_resonance_complex(a: &Array2<Complex64>) -> f64 {
    let theta = 2.39996322972865332_f64;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut mass = 0.0;
    for i in 0..a.nrows() {
        let phase = theta * i as f64;
        let (s, c) = phase.sin_cos();
        let row_mass = a.row(i).iter().map(|x| x.norm()).sum::<f64>();
        re += row_mass * c;
        im += row_mass * s;
        mass += row_mass;
    }
    (re * re + im * im).sqrt() / mass.max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_engine_real_reports_a_passport() {
        let a = Array2::<f64>::eye(6);
        let ((_u, _s, _vt), report) = PhaseEngine::solve_real(&a);
        assert_eq!(report.passport.rows, 6);
        assert!(report.rel_recon.unwrap() < 1e-12);
    }

    #[test]
    fn phase_engine_real_passport_exposes_global_invariants() {
        let a = Array2::from_shape_fn((8, 8), |(i, j)| {
            if i == j {
                1.0
            } else if i + 1 == j {
                3.0
            } else {
                0.0
            }
        });
        let passport = PhaseEngine::passport_real(&a);
        assert!(passport.global_phase.is_finite());
        assert!(passport.torsion_energy > 0.0);
        assert!((0.0..=1.0).contains(&passport.chirality_balance));
        assert!((0.0..=1.0).contains(&passport.phase_entropy));
    }

    #[test]
    fn phase_engine_complex_reports_a_passport() {
        let a = Array2::from_shape_fn((4, 4), |(i, j)| {
            if i == j {
                Complex64::new(2.0 + i as f64, 0.1)
            } else {
                Complex64::new(0.01 * (i + j) as f64, -0.005)
            }
        });
        let ((_u, _s, _vh), report) = PhaseEngine::solve_complex(&a);
        assert_eq!(report.passport.cols, 4);
        assert!(report.rel_recon.unwrap() < 1e-8);
    }
}
