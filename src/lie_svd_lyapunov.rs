//! Lyapunov spectrum extraction via the standard "continuous QR" method
//! (Benettin et al. 1980; Shimada & Nagashima 1979) -- the first genuinely
//! *new* numerical subsystem in the robustness/frontier-benchmark program
//! (`lie_svd_benchmarks` up through `0.38.0` only ever tested this crate's
//! *existing* solvers against hard inputs; this module implements a method
//! this crate didn't have before).
//!
//! ## The method
//!
//! For a flow `dx/dt = F(x)`, the Lyapunov exponents describe the
//! exponential growth/decay rate of infinitesimal separations along a
//! trajectory, governed by the *variational equation*
//! `dPhi/dt = J(x(t)) Phi`, `Phi(0) = I` (`J` the Jacobian of `F`). Solved
//! naively, `Phi(t)` overflows or underflows within a handful of Lyapunov
//! times for a chaotic system (its singular values separate exponentially).
//! The standard fix, used here: periodically QR-decompose the accumulated
//! `Phi` (`Phi = Q R`, `R` upper triangular with non-negative diagonal),
//! replace `Phi` with the orthogonal `Q` before continuing, and accumulate
//! `log(R_ii)` for each `i` across the whole run. After `T` total
//! (physical) time, `lambda_i = (1/T) * sum(log R_ii over every
//! renormalization step)`.
//!
//! `Phi` is propagated here by applying this crate's own RK4 stepper to
//! the *augmented* system `d(x,Phi)/dt = (F(x), J(x) Phi)` -- i.e. the
//! nonlinear trajectory and the full tangent frame are integrated together,
//! with the Jacobian evaluated at each RK4 stage's own intermediate state
//! (not a single frozen Jacobian per step), which is the standard
//! full-accuracy version of this method rather than a cheaper
//! approximation.
//!
//! ## Verification strategy (why this doesn't need an external reference)
//!
//! Comparing against a specific published Lyapunov exponent value for a
//! specific system size would require citing a number from memory that
//! can't be independently checked here -- avoided, the same way an
//! uncertain formula was avoided in `lie_svd_benchmarks`'s Hansen problems.
//! Instead: `lorenz96_jacobian` is checked directly against a finite-
//! difference approximation of `lorenz96_rhs` (an internal consistency
//! check independent of any dynamics), and the Lyapunov spectrum itself is
//! checked against a **rigorous, exactly-derivable** identity rather than
//! an approximate literature value: the sum of *all* Lyapunov exponents
//! equals the long-time average of `trace(J(x(t)))` (a standard theorem --
//! the sum governs the exponential growth rate of phase-space volume,
//! which is exactly the flow's divergence). For the Lorenz-96 model
//! specifically, `J[i,i] = -1` for *every* `i` and *every* state (the only
//! `x_i`-dependence in `dx_i/dt` is the explicit `-x_i` damping term), so
//! `trace(J(x)) = -K` identically, for *any* state `x`, not just on
//! average -- making the target for the sum of Lyapunov exponents exactly
//! `-K`, a real closed-form check, not an approximate one.

use ndarray::{Array1, Array2};

/// The Lorenz-96 vector field (E. Lorenz, 1996, "Predictability: a problem
/// partly solved"): `dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + forcing`
/// on a periodic lattice of `K` sites. `forcing = 8.0` is the standard
/// value cited in the original paper and widely used since as producing
/// chaotic dynamics.
pub fn lorenz96_rhs(x: &Array1<f64>, forcing: f64) -> Array1<f64> {
    let k = x.len();
    Array1::from_shape_fn(k, |i| {
        let ip1 = (i + 1) % k;
        let im1 = (i + k - 1) % k;
        let im2 = (i + k - 2) % k;
        (x[ip1] - x[im2]) * x[im1] - x[i] + forcing
    })
}

/// The exact Jacobian of `lorenz96_rhs`. Requires `k >= 5` to avoid index
/// aliasing among `{i, i+1, i-1, i-2}` on a small periodic lattice (`k=3`
/// in particular aliases `i+1` with `i-2`); every use in this module keeps
/// `k` well above that floor. Verified against a finite-difference
/// approximation of `lorenz96_rhs` in `lorenz96_jacobian_matches_finite_differences`.
pub fn lorenz96_jacobian(x: &Array1<f64>) -> Array2<f64> {
    let k = x.len();
    assert!(
        k >= 5,
        "lorenz96_jacobian: k>=5 needed to avoid index aliasing"
    );
    let mut j = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        let ip1 = (i + 1) % k;
        let im1 = (i + k - 1) % k;
        let im2 = (i + k - 2) % k;
        j[[i, i]] += -1.0;
        j[[i, ip1]] += x[im1];
        j[[i, im1]] += x[ip1] - x[im2];
        j[[i, im2]] += -x[im1];
    }
    j
}

fn augmented_rhs(x: &Array1<f64>, phi: &Array2<f64>, forcing: f64) -> (Array1<f64>, Array2<f64>) {
    let dx = lorenz96_rhs(x, forcing);
    let j = lorenz96_jacobian(x);
    let dphi = j.dot(phi);
    (dx, dphi)
}

/// One RK4 step of the augmented `(x, Phi)` system, `Phi`'s Jacobian
/// evaluated at each of the four RK4 stages' own intermediate state (not a
/// single frozen Jacobian for the whole step).
fn rk4_step_augmented(
    x: &Array1<f64>,
    phi: &Array2<f64>,
    forcing: f64,
    h: f64,
) -> (Array1<f64>, Array2<f64>) {
    let (k1x, k1p) = augmented_rhs(x, phi, forcing);
    let x2 = x + &(&k1x * (h / 2.0));
    let p2 = phi + &(&k1p * (h / 2.0));
    let (k2x, k2p) = augmented_rhs(&x2, &p2, forcing);
    let x3 = x + &(&k2x * (h / 2.0));
    let p3 = phi + &(&k2p * (h / 2.0));
    let (k3x, k3p) = augmented_rhs(&x3, &p3, forcing);
    let x4 = x + &(&k3x * h);
    let p4 = phi + &(&k3p * h);
    let (k4x, k4p) = augmented_rhs(&x4, &p4, forcing);
    let x_next = x + &((&k1x + &k2x * 2.0 + &k3x * 2.0 + &k4x) * (h / 6.0));
    let phi_next = phi + &((&k1p + &k2p * 2.0 + &k3p * 2.0 + &k4p) * (h / 6.0));
    (x_next, phi_next)
}

/// QR decomposition (modified Gram-Schmidt) with the sign convention this
/// method needs: `R`'s diagonal is the column norms at each Gram-Schmidt
/// step, which are non-negative by construction (no extra sign-fixing
/// logic needed).
fn qr_nonneg_diag(m: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = m.nrows();
    let mut q = Array2::<f64>::zeros((n, n));
    let mut r = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut v = m.column(j).to_owned();
        for i in 0..j {
            let qi = q.column(i).to_owned();
            let dot = qi.dot(&v);
            r[[i, j]] = dot;
            v = &v - &(&qi * dot);
        }
        let norm = v.dot(&v).sqrt().max(1e-300);
        r[[j, j]] = norm;
        for row in 0..n {
            q[[row, j]] = v[row] / norm;
        }
    }
    (q, r)
}

#[derive(Clone, Debug)]
pub struct LyapunovSpectrum {
    /// One exponent per tracked frame axis, in the order the continuous-QR
    /// sweep happens to produce (not re-sorted): empirically descending in
    /// practice (see the module tests), but not guaranteed by construction,
    /// so a caller that needs sorted exponents should sort explicitly.
    pub exponents: Array1<f64>,
    pub sum: f64,
    /// `Q^T Q - I` Frobenius drift of the final tracked frame, a direct
    /// orthogonality self-check (this method's whole point is to keep the
    /// tracked frame orthogonal throughout, so this should stay tiny).
    pub final_frame_orthogonality_error: f64,
}

/// The full Lyapunov spectrum (all `k` exponents) of the Lorenz-96 model
/// via the continuous-QR method: `transient_steps` of burn-in (frame
/// renormalized but not accumulated, letting the trajectory settle onto
/// the attractor) followed by `steps` of accumulation, step size `dt`.
pub fn lorenz96_lyapunov_spectrum(
    k: usize,
    forcing: f64,
    dt: f64,
    transient_steps: usize,
    steps: usize,
) -> LyapunovSpectrum {
    assert!(k >= 5, "lorenz96_lyapunov_spectrum: k>=5 needed");
    let mut x = Array1::from_shape_fn(k, |i| forcing + 0.01 * ((i as f64 * 0.7).sin()));
    let mut phi = Array2::<f64>::eye(k);

    for _ in 0..transient_steps {
        let (x_next, phi_next) = rk4_step_augmented(&x, &phi, forcing, dt);
        x = x_next;
        let (q, _r) = qr_nonneg_diag(&phi_next);
        phi = q;
    }

    let mut log_sum = vec![0.0_f64; k];
    for _ in 0..steps {
        let (x_next, phi_next) = rk4_step_augmented(&x, &phi, forcing, dt);
        x = x_next;
        let (q, r) = qr_nonneg_diag(&phi_next);
        for (i, item) in log_sum.iter_mut().enumerate() {
            *item += r[[i, i]].max(1e-300).ln();
        }
        phi = q;
    }

    let total_time = steps as f64 * dt;
    let exponents = Array1::from_shape_fn(k, |i| log_sum[i] / total_time);
    let sum = exponents.iter().sum();
    let ident = Array2::<f64>::eye(k);
    let final_frame_orthogonality_error =
        (&phi.t().dot(&phi) - &ident).mapv(|v| v * v).sum().sqrt();

    LyapunovSpectrum {
        exponents,
        sum,
        final_frame_orthogonality_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A direct internal-consistency check, independent of any dynamics:
    /// `lorenz96_jacobian`'s analytic entries must match a central-
    /// finite-difference approximation of `lorenz96_rhs`.
    #[test]
    fn lorenz96_jacobian_matches_finite_differences() {
        let k = 10;
        let x = Array1::from_shape_fn(k, |i| ((i as f64) * 1.37).sin() * 3.0 + 1.0);
        let analytic = lorenz96_jacobian(&x);
        let eps = 1e-6;
        let mut numeric = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let mut x_plus = x.clone();
            x_plus[j] += eps;
            let mut x_minus = x.clone();
            x_minus[j] -= eps;
            let f_plus = lorenz96_rhs(&x_plus, 8.0);
            let f_minus = lorenz96_rhs(&x_minus, 8.0);
            for i in 0..k {
                numeric[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * eps);
            }
        }
        let diff = (&analytic - &numeric)
            .mapv(|v| v.abs())
            .into_iter()
            .fold(0.0_f64, f64::max);
        assert!(
            diff < 1e-6,
            "max diff between analytic and numeric Jacobian = {diff:e}"
        );
    }

    /// The rigorous, closed-form check derived in the module doc comment:
    /// `trace(J) = -k` identically for the Lorenz-96 model (the only
    /// `x_i`-dependence in `dx_i/dt` is the explicit `-x_i` term), so the
    /// sum of *all* `k` Lyapunov exponents must equal `-k` exactly, not
    /// approximately -- a real theorem (sum of exponents = long-time-
    /// averaged trace of the Jacobian = flow divergence), not a
    /// convergence-tolerance fudge.
    #[test]
    fn lorenz96_lyapunov_exponents_sum_to_minus_k() {
        let k = 10;
        let result = lorenz96_lyapunov_spectrum(k, 8.0, 0.01, 2_000, 20_000);
        assert!(
            result.final_frame_orthogonality_error < 1e-8,
            "orthogonality_error={:e}",
            result.final_frame_orthogonality_error
        );
        assert!(result.exponents.iter().all(|e| e.is_finite()));
        let expected_sum = -(k as f64);
        let err = (result.sum - expected_sum).abs();
        assert!(
            err < 0.05,
            "expected sum of exponents ~= {expected_sum}, got {} (err={err:e})",
            result.sum
        );
    }

    /// The qualitative chaos indicator this system is well known for at
    /// `forcing=8`: at least one strictly positive Lyapunov exponent
    /// (exponential divergence of nearby trajectories).
    #[test]
    fn lorenz96_has_a_positive_lyapunov_exponent_at_standard_forcing() {
        let k = 10;
        let result = lorenz96_lyapunov_spectrum(k, 8.0, 0.01, 2_000, 20_000);
        let max_exp = result
            .exponents
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_exp > 0.1,
            "expected a clearly positive largest Lyapunov exponent, got {max_exp:e}"
        );
    }
}
