//! Unified phase-event compiler for analog, photonic, and FPGA targets.
//!
//! This module does not try to simulate hardware. It takes the rotor/phase
//! events already emitted by the real and complex PhaseFlow routes and serializes
//! them into one stable schedule shape: layers, channel pairs, phase offsets,
//! rotor angles, and optional energy status fields.

use crate::lie_svd_complex::{ComplexMziPhase, ComplexPhaseEventKind};
use crate::lie_svd_phaseflow::{MziPhase, PhaseRotorKind};

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
        out.push_str("  ]\n");
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
}
