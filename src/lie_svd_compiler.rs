//! Unified phase-event compiler for analog, photonic, and FPGA targets.
//!
//! This module does not try to simulate hardware. It takes the rotor/phase
//! events already emitted by the real and complex PhaseFlow routes and serializes
//! them into one stable schedule shape: layers, channel pairs, phase offsets,
//! rotor angles, and optional energy status fields.

use crate::lie_svd_complex::{ComplexMziPhase, ComplexPhaseEventKind};
use crate::lie_svd_phaseflow::{MziPhase, PhaseRotorKind};
use ndarray::Array2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HardwareTarget {
    MziMesh,
    FpgaRotorMesh,
}

impl HardwareTarget {
    pub fn as_str(self) -> &'static str {
        match self {
            HardwareTarget::MziMesh => "mzi_mesh",
            HardwareTarget::FpgaRotorMesh => "fpga_rotor_mesh",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HardwarePhaseEvent {
    pub layer: usize,
    pub i: usize,
    pub j: usize,
    pub phi_l: f64,
    pub phi_r: f64,
    pub theta: f64,
    pub theta_l: f64,
    pub theta_r: f64,
    pub energy_before: Option<f64>,
    pub energy_after: Option<f64>,
    pub source: &'static str,
    pub kind: &'static str,
}

#[derive(Clone, Debug)]
pub struct HardwareSchedule {
    pub format_version: &'static str,
    pub target: HardwareTarget,
    pub channels: usize,
    pub events: Vec<HardwarePhaseEvent>,
    /// `+-1` diagonal left over after `from_orthogonal_matrix`'s Givens
    /// sweep (see that method's doc comment) — empty for schedules built
    /// from a PhaseFlow event log, which have no such leftover diagonal.
    /// Needed to reconstruct the original matrix exactly from `events`
    /// alone; a schedule missing it would silently lose information rather
    /// than just being awkward to use.
    pub diagonal_signs: Vec<f64>,
}

impl HardwareSchedule {
    pub fn from_real_phaseflow(
        events: &[MziPhase],
        channels: usize,
        target: HardwareTarget,
    ) -> Self {
        let events = events
            .iter()
            .map(|e| HardwarePhaseEvent {
                layer: e.pass,
                i: e.i,
                j: e.j,
                phi_l: 0.0,
                phi_r: 0.0,
                theta: 0.5 * (e.theta_l + e.theta_r),
                theta_l: e.theta_l,
                theta_r: e.theta_r,
                energy_before: None,
                energy_after: None,
                source: "real_phaseflow",
                kind: real_kind(e.kind),
            })
            .collect();
        Self {
            format_version: "phase-schedule-v1",
            target,
            channels,
            events,
            diagonal_signs: Vec::new(),
        }
    }

    pub fn from_complex_phaseflow(
        events: &[ComplexMziPhase],
        channels: usize,
        target: HardwareTarget,
    ) -> Self {
        let events = events
            .iter()
            .map(|e| HardwarePhaseEvent {
                layer: e.layer,
                i: e.i,
                j: e.j,
                phi_l: e.phi_l,
                phi_r: e.phi_r,
                theta: e.theta,
                theta_l: e.theta,
                theta_r: e.theta,
                energy_before: None,
                energy_after: None,
                source: "complex_phaseflow",
                kind: complex_kind(e.kind),
            })
            .collect();
        Self {
            format_version: "phase-schedule-v1",
            target,
            channels,
            events,
            diagonal_signs: Vec::new(),
        }
    }

    /// Compiles an arbitrary `d x d` orthogonal matrix (a rotor that was
    /// never accompanied by its own rotation-angle trace — e.g.
    /// `lie_tbl_regress::procrustes_rotor`'s output, or any other rotor
    /// this crate hands back only as a plain matrix) into a Givens-rotation
    /// event schedule, without needing the solver that produced it to log
    /// anything.
    ///
    /// This was needed because `lie_svd_small::eigh_jacobi_full` (the
    /// square eigensolver backing most of this crate's rotors) does not
    /// record a rotation trace as it runs — instrumenting a hot, widely
    /// shared solver path to do so was judged higher-risk than the
    /// alternative used here: decompose the *already-orthogonal result*
    /// after the fact via a standard Givens QR sweep. Since the input is
    /// already orthogonal, eliminating its strict lower triangle with
    /// Givens rotations leaves an orthogonal *and* upper-triangular
    /// matrix, which is necessarily diagonal with `+-1` entries (an
    /// upper-triangular orthogonal matrix cannot have any other off-diagonal
    /// content: each column must have unit norm using only the entries at
    /// or above its own row). So `V = G_1^T G_2^T ... G_m^T D` exactly,
    /// `D = diag(+-1)`, `G_k` the recorded rotations in elimination order —
    /// see `orthogonal_matrix_round_trips_through_givens_schedule` for the
    /// reconstruction that verifies this to machine precision rather than
    /// asserting it.
    pub fn from_orthogonal_matrix(v: &Array2<f64>, target: HardwareTarget) -> Self {
        let d = v.nrows();
        assert_eq!(
            v.ncols(),
            d,
            "HardwareSchedule::from_orthogonal_matrix: matrix must be square, got {}x{}",
            d,
            v.ncols()
        );
        let mut r = v.clone();
        let mut events = Vec::new();
        let mut layer = 0usize;
        for j in 0..d {
            for i in (j + 1)..d {
                let a = r[[j, j]];
                let b = r[[i, j]];
                if b.abs() <= 1e-300 {
                    continue;
                }
                let norm = (a * a + b * b).sqrt();
                let c = a / norm;
                let s = b / norm;
                for k in 0..d {
                    let rj = r[[j, k]];
                    let ri = r[[i, k]];
                    r[[j, k]] = c * rj + s * ri;
                    r[[i, k]] = -s * rj + c * ri;
                }
                let theta = s.atan2(c);
                events.push(HardwarePhaseEvent {
                    layer,
                    i,
                    j,
                    phi_l: 0.0,
                    phi_r: 0.0,
                    theta,
                    theta_l: theta,
                    theta_r: 0.0,
                    energy_before: None,
                    energy_after: None,
                    source: "orthogonal_givens",
                    kind: "givens_elimination",
                });
                layer += 1;
            }
        }
        let diagonal_signs: Vec<f64> = (0..d).map(|k| r[[k, k]].signum()).collect();
        Self {
            format_version: "phase-schedule-v1",
            target,
            channels: d,
            events,
            diagonal_signs,
        }
    }

    pub fn total_events(&self) -> usize {
        self.events.len()
    }

    pub fn conflict_free_layer_count(&self) -> usize {
        self.events
            .iter()
            .map(|event| event.layer)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!(
            "  \"format_version\": \"{}\",\n",
            self.format_version
        ));
        out.push_str(&format!("  \"target\": \"{}\",\n", self.target.as_str()));
        out.push_str(&format!("  \"channels\": {},\n", self.channels));
        out.push_str(&format!("  \"total_events\": {},\n", self.events.len()));
        out.push_str("  \"events\": [\n");
        for (idx, event) in self.events.iter().enumerate() {
            out.push_str("    {");
            out.push_str(&format!(
                "\"layer\": {}, \"i\": {}, \"j\": {}, \"phi_l\": {:.17e}, \"phi_r\": {:.17e}, \"theta\": {:.17e}, \"theta_l\": {:.17e}, \"theta_r\": {:.17e}, \"source\": \"{}\", \"kind\": \"{}\"",
                event.layer,
                event.i,
                event.j,
                event.phi_l,
                event.phi_r,
                event.theta,
                event.theta_l,
                event.theta_r,
                event.source,
                event.kind,
            ));
            if let Some(value) = event.energy_before {
                out.push_str(&format!(", \"energy_before\": {:.17e}", value));
            }
            if let Some(value) = event.energy_after {
                out.push_str(&format!(", \"energy_after\": {:.17e}", value));
            }
            out.push('}');
            if idx + 1 != self.events.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]");
        if !self.diagonal_signs.is_empty() {
            out.push_str(",\n  \"diagonal_signs\": [");
            for (idx, sign) in self.diagonal_signs.iter().enumerate() {
                if idx > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{sign}"));
            }
            out.push(']');
        }
        out.push('\n');
        out.push_str("}\n");
        out
    }
}

fn real_kind(kind: PhaseRotorKind) -> &'static str {
    match kind {
        PhaseRotorKind::GoldenPreSpin => "golden_prespin",
        PhaseRotorKind::CausalAntiSpin => "causal_antispin",
        PhaseRotorKind::CrossPhaseYinYang => "cross_phase_yinyang",
        PhaseRotorKind::PhaseConjugate => "phase_conjugate",
        PhaseRotorKind::Bottleneck => "bottleneck",
        PhaseRotorKind::PhaseJump => "phase_jump",
        PhaseRotorKind::Unwrap => "unwrap",
        PhaseRotorKind::Directional => "directional",
        PhaseRotorKind::Surgery => "surgery",
    }
}

fn complex_kind(kind: ComplexPhaseEventKind) -> &'static str {
    match kind {
        ComplexPhaseEventKind::GoldenPreSpin => "complex_golden_prespin",
        ComplexPhaseEventKind::DiagonalPhase => "complex_diagonal_phase",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_schedule_json_contains_layers_and_angles() {
        let events = vec![MziPhase {
            pass: 2,
            i: 0,
            j: 1,
            theta_l: 0.125,
            theta_r: -0.25,
            kind: PhaseRotorKind::Unwrap,
        }];
        let schedule = HardwareSchedule::from_real_phaseflow(&events, 4, HardwareTarget::MziMesh);
        let json = schedule.to_json_string();
        assert_eq!(schedule.total_events(), 1);
        assert_eq!(schedule.conflict_free_layer_count(), 3);
        assert!(json.contains("\"target\": \"mzi_mesh\""));
        assert!(json.contains("\"kind\": \"unwrap\""));
    }

    #[test]
    fn real_schedule_json_names_causal_antispin() {
        let events = vec![MziPhase {
            pass: 0,
            i: 2,
            j: 3,
            theta_l: 0.1,
            theta_r: -0.1,
            kind: PhaseRotorKind::CausalAntiSpin,
        }];
        let schedule = HardwareSchedule::from_real_phaseflow(&events, 4, HardwareTarget::MziMesh);
        assert!(schedule.to_json_string().contains("causal_antispin"));
    }

    #[test]
    fn real_schedule_json_names_cross_phase_yinyang() {
        let events = vec![MziPhase {
            pass: 1,
            i: 0,
            j: 2,
            theta_l: 0.03,
            theta_r: 0.0,
            kind: PhaseRotorKind::CrossPhaseYinYang,
        }];
        let schedule = HardwareSchedule::from_real_phaseflow(&events, 4, HardwareTarget::MziMesh);
        assert!(schedule.to_json_string().contains("cross_phase_yinyang"));
    }

    #[test]
    fn real_schedule_json_names_phase_conjugate_and_bottleneck() {
        let events = vec![
            MziPhase {
                pass: 0,
                i: 0,
                j: 1,
                theta_l: -0.02,
                theta_r: 0.03,
                kind: PhaseRotorKind::PhaseConjugate,
            },
            MziPhase {
                pass: 1,
                i: 2,
                j: 3,
                theta_l: 0.04,
                theta_r: -0.05,
                kind: PhaseRotorKind::Bottleneck,
            },
        ];
        let schedule = HardwareSchedule::from_real_phaseflow(&events, 4, HardwareTarget::MziMesh);
        let json = schedule.to_json_string();
        assert!(json.contains("phase_conjugate"));
        assert!(json.contains("bottleneck"));
    }

    /// Reconstructs `V` from `from_orthogonal_matrix`'s recorded Givens
    /// events and diagonal signs, and checks it matches the original to
    /// machine precision -- proving the decomposition is lossless rather
    /// than just asserting it compiles. Reconstruction applies each
    /// recorded rotation's *inverse* (transpose) in reverse order, starting
    /// from `diag(diagonal_signs)`: forward elimination built
    /// `R = G_m ... G_1 V` (`R = diag(signs)`), so
    /// `V = G_1^T G_2^T ... G_m^T R`.
    #[test]
    fn orthogonal_matrix_round_trips_through_givens_schedule() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(211);
        let d = 5;
        // Any orthogonal matrix works as a test input; build one the same
        // way other tests in this crate do (Procrustes rotor between two
        // random matrices is orthogonal by construction).
        let m1 = Array2::from_shape_fn((d, d), |_| rng.gen_range(-1.0_f64..1.0));
        let m2 = Array2::from_shape_fn((d, d), |_| rng.gen_range(-1.0_f64..1.0));
        let v = crate::lie_tbl_regress::procrustes_rotor(&m1, &m2);

        let schedule = HardwareSchedule::from_orthogonal_matrix(&v, HardwareTarget::MziMesh);
        assert_eq!(
            schedule.events.len(),
            d * (d - 1) / 2,
            "a full triangular sweep on generic (non-degenerate) input eliminates every \
             below-diagonal entry"
        );
        assert_eq!(schedule.diagonal_signs.len(), d);

        let mut reconstructed = Array2::<f64>::eye(d);
        for k in 0..d {
            reconstructed[[k, k]] = schedule.diagonal_signs[k];
        }
        for event in schedule.events.iter().rev() {
            let theta = event.theta_l;
            let c = theta.cos();
            let s = theta.sin();
            let (i, j) = (event.i, event.j);
            for k in 0..d {
                let rj = reconstructed[[j, k]];
                let ri = reconstructed[[i, k]];
                reconstructed[[j, k]] = c * rj - s * ri;
                reconstructed[[i, k]] = s * rj + c * ri;
            }
        }

        let max_err = v
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-10,
            "Givens schedule must reconstruct the original orthogonal matrix, max_err={max_err:e}"
        );
    }

    /// The concrete use case this decomposition exists for: a
    /// `TblRotorRegressor` domain-transfer rotor (`procrustes_rotor`'s
    /// output, used by `transfer_fit`) compiled to an MZI hardware
    /// schedule, end to end.
    #[test]
    fn procrustes_rotor_compiles_to_a_valid_mzi_schedule() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(223);
        let n = 40;
        let d = 3;
        let x_a = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        let x_b = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0_f64..1.0));
        let r_ab = crate::lie_tbl_regress::procrustes_rotor(&x_a, &x_b);

        let schedule = HardwareSchedule::from_orthogonal_matrix(&r_ab, HardwareTarget::MziMesh);
        assert_eq!(schedule.channels, d);
        assert!(schedule.total_events() <= d * (d - 1) / 2);
        let json = schedule.to_json_string();
        assert!(json.contains("givens_elimination"));
        assert!(json.contains("diagonal_signs"));
    }
}
