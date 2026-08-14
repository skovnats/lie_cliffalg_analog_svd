//! Phase-BSS: blind source separation by phase-aware joint diagonalization.
//!
//! This is the first BSS/ICA bridge in the crate. It keeps the implementation
//! conservative: center and whiten the observed channels, build a small family
//! of lagged covariance matrices, then use the existing Phase-JADE joint
//! diagonalizer to find a shared rotation. The "phase" part is carried by the
//! lagged channel signatures and by a channel coherence metric; the stored data
//! remains plain dense `f64`.

use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct PhaseBssParams {
    pub lags: Vec<usize>,
    pub eps: f64,
    pub joint: crate::lie_svd_joint::JointDiagonalizationParams,
}

impl PhaseBssParams {
    pub fn for_channels(channels: usize) -> Self {
        Self {
            lags: vec![1, 2, 3, 5],
            eps: 1e-10,
            joint: crate::lie_svd_joint::JointDiagonalizationParams::for_n(channels),
        }
    }
}

impl Default for PhaseBssParams {
    fn default() -> Self {
        Self::for_channels(4)
    }
}

#[derive(Clone, Debug)]
pub struct PhaseBssTrace {
    pub channels: usize,
    pub samples: usize,
    pub covariance_floor: f64,
    pub joint_initial_offdiag: f64,
    pub joint_final_offdiag: f64,
    pub joint_rotations: usize,
    pub mean_channel_coherence: f64,
}

#[derive(Clone, Debug)]
pub struct PhaseBssResult {
    pub unmixing: Array2<f64>,
    pub separated: Array2<f64>,
    pub channel_coherence: Array1<f64>,
    pub trace: PhaseBssTrace,
}

pub struct LieSvdBss;

impl LieSvdBss {
    /// Separates `channels x samples` observations.
    pub fn separate(observations: &Array2<f64>, params: PhaseBssParams) -> PhaseBssResult {
        let channels = observations.nrows();
        let samples = observations.ncols();
        assert!(channels > 0 && samples > 1, "PhaseBSS: empty observations");

        let centered = center_channels(observations);
        let covariance = centered.dot(&centered.t()) / samples as f64;
        let (u, sigma, _vt) = crate::lie_svd_small::LieSvdSmall::solve(&covariance);
        let covariance_floor = params.eps * sigma.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let mut whitening = Array2::<f64>::zeros((channels, channels));
        for i in 0..channels {
            let inv = 1.0 / sigma[i].max(covariance_floor).sqrt();
            for j in 0..channels {
                whitening[[i, j]] = inv * u[[j, i]];
            }
        }
        let whitened = whitening.dot(&centered);
        let lagged = lagged_covariances(&whitened, &params.lags);
        let (basis, _diags, joint_trace) =
            crate::lie_svd_joint::LieSvdJoint::diagonalize_symmetric_with_params(
                &lagged,
                params.joint,
            );
        let separated = basis.t().dot(&whitened);
        let unmixing = basis.t().dot(&whitening);
        let channel_coherence = channel_phase_coherence(&separated);
        let mean_channel_coherence =
            channel_coherence.iter().sum::<f64>() / channel_coherence.len().max(1) as f64;

        PhaseBssResult {
            unmixing,
            separated,
            channel_coherence,
            trace: PhaseBssTrace {
                channels,
                samples,
                covariance_floor,
                joint_initial_offdiag: joint_trace.initial_offdiag,
                joint_final_offdiag: joint_trace.final_offdiag,
                joint_rotations: joint_trace.rotations,
                mean_channel_coherence,
            },
        }
    }
}

pub fn channel_phase_coherence(channels: &Array2<f64>) -> Array1<f64> {
    let n = channels.nrows();
    let samples = channels.ncols();
    Array1::from_shape_fn(n, |i| {
        if samples <= 1 {
            return 0.0;
        }
        let mut norm = 0.0_f64;
        let mut delay_dot = 0.0_f64;
        let mut grad = 0.0_f64;
        for t in 0..samples {
            let x = channels[[i, t]];
            let y = channels[[i, (t + 1) % samples]];
            norm += x * x;
            delay_dot += x * y;
            let d = y - x;
            grad += d * d;
        }
        let phase_lock = delay_dot.abs() / norm.max(1e-300);
        let smoothness = 1.0 / (1.0 + (grad / norm.max(1e-300)).sqrt());
        (0.5 * phase_lock + 0.5 * smoothness).clamp(0.0, 1.0)
    })
}

pub fn estimate_sir_db(reference: &Array2<f64>, separated: &Array2<f64>) -> f64 {
    let n = reference.nrows().min(separated.nrows());
    if n == 0 {
        return 0.0;
    }
    let corr = absolute_correlation_matrix(reference, separated);
    let mut used_ref = vec![false; n];
    let mut used_sep = vec![false; n];
    let mut signal = 0.0_f64;
    let mut interference = 0.0_f64;
    for _ in 0..n {
        let mut best = (0usize, 0usize, -1.0_f64);
        for i in 0..n {
            if used_ref[i] {
                continue;
            }
            for j in 0..n {
                if !used_sep[j] && corr[[i, j]] > best.2 {
                    best = (i, j, corr[[i, j]]);
                }
            }
        }
        used_ref[best.0] = true;
        used_sep[best.1] = true;
        signal += best.2 * best.2;
        for i in 0..n {
            if i != best.0 {
                interference += corr[[i, best.1]] * corr[[i, best.1]];
            }
        }
    }
    10.0 * (signal / interference.max(1e-12)).log10()
}

fn center_channels(x: &Array2<f64>) -> Array2<f64> {
    let mut out = x.clone();
    for i in 0..out.nrows() {
        let mean = (0..out.ncols()).map(|t| out[[i, t]]).sum::<f64>() / out.ncols() as f64;
        for t in 0..out.ncols() {
            out[[i, t]] -= mean;
        }
    }
    out
}

fn lagged_covariances(x: &Array2<f64>, lags: &[usize]) -> Vec<Array2<f64>> {
    let channels = x.nrows();
    let samples = x.ncols();
    let mut out = Vec::new();
    for &lag in lags {
        if lag == 0 || lag >= samples {
            continue;
        }
        let denom = (samples - lag) as f64;
        let mut c = Array2::<f64>::zeros((channels, channels));
        for t in 0..(samples - lag) {
            for i in 0..channels {
                for j in 0..channels {
                    c[[i, j]] += x[[i, t + lag]] * x[[j, t]] / denom;
                }
            }
        }
        let sym = 0.5 * (&c + &c.t());
        out.push(sym);
    }
    if out.is_empty() {
        out.push(x.dot(&x.t()) / samples as f64);
    }
    out
}

fn absolute_correlation_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows().min(b.nrows());
    let samples = a.ncols().min(b.ncols());
    Array2::from_shape_fn((n, n), |(i, j)| {
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for t in 0..samples {
            let x = a[[i, t]];
            let y = b[[j, t]];
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot.abs() / (na.sqrt() * nb.sqrt()).max(1e-300)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_bss_improves_sir_on_synthetic_sources() {
        let channels = 4;
        let samples = 512;
        let sources = Array2::from_shape_fn((channels, samples), |(i, t)| {
            let x = t as f64 / samples as f64;
            match i {
                0 => (2.0 * std::f64::consts::PI * 7.0 * x).sin(),
                1 => (2.0 * std::f64::consts::PI * 11.0 * x).cos().signum(),
                2 => {
                    (2.0 * std::f64::consts::PI * 3.0 * x).sin()
                        + 0.4 * (2.0 * std::f64::consts::PI * 17.0 * x).sin()
                }
                _ => ((t * 37 + 11) as f64).sin() * 0.7,
            }
        });
        let mixing = Array2::from_shape_fn((channels, channels), |(i, j)| {
            if i == j {
                1.0
            } else {
                0.25 * ((i * 13 + j * 7 + 3) as f64).sin()
            }
        });
        let observations = mixing.dot(&sources);
        let before = estimate_sir_db(&sources, &observations);
        let result = LieSvdBss::separate(&observations, PhaseBssParams::for_channels(channels));
        let after = estimate_sir_db(&sources, &result.separated);
        assert!(after > before + 3.0, "before={before} after={after}");
        assert!(result.trace.joint_final_offdiag <= result.trace.joint_initial_offdiag + 1e-10);
        assert!(result.trace.mean_channel_coherence.is_finite());
    }
}
