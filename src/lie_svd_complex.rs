//! Complex-native phase SVD prototype.
//!
//! This module starts the `c64` branch without disturbing the real `f64`
//! PhaseFlow route.  Complex row/column phases are represented directly by
//! `Complex64` scalars, while the singular-vector solve uses a conservative
//! Hermitian Jacobi path on `A^H A`.
//!
//! The "zero-overhead" phase language is literal for MZI/photonic hardware,
//! where a channel phase is one device setting.  On a dense CPU `Array2`, a
//! materialized row/column phase sheet still touches `O(N^2)` entries.

use ndarray::{Array1, Array2};
use num_complex::Complex64;

const GOLDEN_ANGLE: f64 = 2.39996322972865332;

#[derive(Clone, Copy, Debug)]
pub struct LieSvdComplexParams {
    pub max_sweeps: usize,
    pub tol: f64,
    pub use_golden_prespin: bool,
    pub golden_prespin_layers: usize,
    pub use_polar_polish: bool,
}

impl LieSvdComplexParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            max_sweeps: (64 + 16 * n).clamp(128, 1024),
            tol: 1e-15,
            use_golden_prespin: true,
            golden_prespin_layers: 1,
            use_polar_polish: true,
        }
    }
}

impl Default for LieSvdComplexParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComplexPhaseEventKind {
    GoldenPreSpin,
    DiagonalPhase,
}

#[derive(Clone, Copy, Debug)]
pub struct ComplexMziPhase {
    pub layer: usize,
    pub i: usize,
    pub j: usize,
    pub phi_l: f64,
    pub phi_r: f64,
    pub theta: f64,
    pub kind: ComplexPhaseEventKind,
}

#[derive(Clone, Debug)]
pub struct LieSvdComplexTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub initial_phase_stress: f64,
    pub final_phase_stress: f64,
    pub sweeps: usize,
    pub rotations: usize,
    pub golden_prespins: usize,
    pub diagonal_phase_fixes: usize,
    pub polar_polishes: usize,
    pub mzi_phases: Vec<ComplexMziPhase>,
}

pub struct LieSvdComplex;

impl LieSvdComplex {
    pub fn solve(mat: &Array2<Complex64>) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
        Self::solve_with_trace(mat, LieSvdComplexParams::for_n(mat.nrows())).0
    }

    pub fn solve_with_trace(
        mat: &Array2<Complex64>,
        params: LieSvdComplexParams,
    ) -> (
        (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
        LieSvdComplexTrace,
    ) {
        assert_eq!(
            mat.nrows(),
            mat.ncols(),
            "LieSvdComplex: square matrix expected"
        );
        let n = mat.nrows();
        if n == 0 {
            return (
                (
                    Array2::zeros((0, 0)),
                    Array1::zeros(0),
                    Array2::zeros((0, 0)),
                ),
                LieSvdComplexTrace {
                    initial_offdiag: 0.0,
                    final_offdiag: 0.0,
                    initial_phase_stress: 0.0,
                    final_phase_stress: 0.0,
                    sweeps: 0,
                    rotations: 0,
                    golden_prespins: 0,
                    diagonal_phase_fixes: 0,
                    polar_polishes: 0,
                    mzi_phases: Vec::new(),
                },
            );
        }

        let mut core = mat.clone();
        let mut u0 = eye_complex(n);
        let mut v0 = eye_complex(n);
        let initial_offdiag = offdiag_norm(&core);
        let initial_phase_stress = phase_stress(&core);
        let mut mzi_phases = Vec::new();
        let mut golden_prespins = 0usize;

        if params.use_golden_prespin {
            golden_prespins =
                apply_complex_golden_prespin(&mut core, &mut u0, &mut v0, params, &mut mzi_phases);
        }

        let ((uc, sigma, vhc), eigen_trace) = svd_via_hermitian_jacobi(&core, params);
        let mut u = u0.dot(&uc);
        let mut vh = vhc.dot(&v0.t().mapv(|x| x.conj()));
        let mut diagonal_phase_fixes = normalize_diagonal_phases(mat, &mut u, &mut vh, &sigma);
        if diagonal_phase_fixes > 0 {
            for i in 0..n {
                mzi_phases.push(ComplexMziPhase {
                    layer: params.golden_prespin_layers,
                    i,
                    j: i,
                    phi_l: 0.0,
                    phi_r: 0.0,
                    theta: 0.0,
                    kind: ComplexPhaseEventKind::DiagonalPhase,
                });
            }
        }

        let core_final = u
            .t()
            .mapv(|x| x.conj())
            .dot(mat)
            .dot(&vh.t().mapv(|x| x.conj()));
        let final_offdiag = offdiag_norm(&core_final);
        let final_phase_stress = phase_stress(&core_final);
        diagonal_phase_fixes = diagonal_phase_fixes.min(n);

        (
            (u, sigma, vh),
            LieSvdComplexTrace {
                initial_offdiag,
                final_offdiag,
                initial_phase_stress,
                final_phase_stress,
                sweeps: eigen_trace.sweeps,
                rotations: eigen_trace.rotations,
                golden_prespins,
                diagonal_phase_fixes,
                polar_polishes: eigen_trace.polar_polishes,
                mzi_phases,
            },
        )
    }

    pub fn solve_2x2_micro(
        block: &Array2<Complex64>,
    ) -> (Array2<Complex64>, Array1<f64>, Array2<Complex64>) {
        assert_eq!(block.dim(), (2, 2), "LieSvdComplex microkernel expects 2x2");
        let mut params = LieSvdComplexParams::for_n(2);
        params.use_golden_prespin = false;
        params.max_sweeps = 16;
        Self::solve_with_trace(block, params).0
    }

    pub fn to_mzi_phases(
        mat: &Array2<Complex64>,
        params: LieSvdComplexParams,
    ) -> Vec<ComplexMziPhase> {
        Self::solve_with_trace(mat, params).1.mzi_phases
    }
}

pub fn apply_complex_golden_prespin(
    core: &mut Array2<Complex64>,
    u_basis: &mut Array2<Complex64>,
    v_basis: &mut Array2<Complex64>,
    params: LieSvdComplexParams,
    mzi_phases: &mut Vec<ComplexMziPhase>,
) -> usize {
    let n = core.nrows().min(core.ncols());
    let layers = params.golden_prespin_layers.max(1);
    let mut events = 0usize;
    for layer in 0..layers {
        for i in 0..n {
            let row_phi = golden_phase(i + layer, GoldenSide::Row);
            let col_phi = golden_phase(i + layer, GoldenSide::Col);
            let left = cis(-row_phi);
            let right = cis(col_phi);
            for col in 0..core.ncols() {
                core[[i, col]] *= left;
            }
            for row in 0..u_basis.nrows() {
                u_basis[[row, i]] *= cis(row_phi);
            }
            for row in 0..core.nrows() {
                core[[row, i]] *= right;
            }
            for row in 0..v_basis.nrows() {
                v_basis[[row, i]] *= right;
            }
            mzi_phases.push(ComplexMziPhase {
                layer,
                i,
                j: i,
                phi_l: row_phi,
                phi_r: col_phi,
                theta: 0.0,
                kind: ComplexPhaseEventKind::GoldenPreSpin,
            });
            events += 1;
        }
    }
    events
}

pub fn complex_relative_reconstruction_error(
    a: &Array2<Complex64>,
    u: &Array2<Complex64>,
    sigma: &Array1<f64>,
    vh: &Array2<Complex64>,
) -> f64 {
    let sigma_mat = Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| {
        if i == j && i < sigma.len() {
            Complex64::new(sigma[i], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    let recon = u.dot(&sigma_mat).dot(vh);
    let mut err = 0.0_f64;
    for (x, y) in recon.iter().zip(a.iter()) {
        err += (*x - *y).norm_sqr();
    }
    err.sqrt() / frobenius_norm(a).max(1e-300)
}

pub fn complex_unitarity_error(q: &Array2<Complex64>) -> f64 {
    let n = q.ncols();
    let gram = q.t().mapv(|x| x.conj()).dot(q);
    let mut err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            err += (gram[[i, j]] - target).norm_sqr();
        }
    }
    err.sqrt()
}

#[derive(Clone, Copy)]
struct HermitianJacobiTrace {
    sweeps: usize,
    rotations: usize,
    polar_polishes: usize,
}

fn svd_via_hermitian_jacobi(
    mat: &Array2<Complex64>,
    params: LieSvdComplexParams,
) -> (
    (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
    HermitianJacobiTrace,
) {
    svd_via_hermitian_jacobi_inner(mat, params, true)
}

fn svd_via_hermitian_jacobi_inner(
    mat: &Array2<Complex64>,
    params: LieSvdComplexParams,
    allow_polar_polish: bool,
) -> (
    (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
    HermitianJacobiTrace,
) {
    let n = mat.ncols();
    let gram = mat.t().mapv(|x| x.conj()).dot(mat);
    let (evals, v, trace) = hermitian_jacobi_eigen(gram, params.max_sweeps, params.tol);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        evals[b]
            .partial_cmp(&evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sigma = Array1::<f64>::zeros(n);
    let mut v_sorted = Array2::<Complex64>::zeros((n, n));
    for (dst, &src) in order.iter().enumerate() {
        sigma[dst] = evals[src].max(0.0).sqrt();
        for r in 0..n {
            v_sorted[[r, dst]] = v[[r, src]];
        }
    }

    let mut u = Array2::<Complex64>::zeros((mat.nrows(), n));
    let sigma_max = sigma.iter().copied().fold(0.0_f64, f64::max);
    let active_cutoff = 1e-14 * sigma_max.max(1.0);
    let mut locked = vec![false; n];
    for col in 0..n {
        if sigma[col] > active_cutoff {
            for r in 0..mat.nrows() {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum += mat[[r, k]] * v_sorted[[k, col]];
                }
                u[[r, col]] = sum / sigma[col];
            }
            if column_norm(&u, col) > 1e-10 {
                normalize_column(&mut u, col);
                locked[col] = true;
            }
        }
    }
    complete_missing_unitary_columns(&mut u, &locked);
    let vh = v_sorted.t().mapv(|x| x.conj());
    if complex_unitarity_error(&u) > 1e-10 {
        if let Some(((ul, sigmal, vhl), left_trace)) =
            left_hermitian_basis_route(mat, &u, &sigma, &vh, params, trace)
        {
            return ((ul, sigmal, vhl), left_trace);
        }
    }

    if allow_polar_polish && params.use_polar_polish && complex_unitarity_error(&u) > 1e-10 {
        if let Some(((up, sigmap, vhp), polish_trace)) =
            qr_polar_polish(mat, &u, &sigma, &vh, params)
        {
            let current_rel = complex_relative_reconstruction_error(mat, &u, &sigma, &vh);
            let polished_rel = complex_relative_reconstruction_error(mat, &up, &sigmap, &vhp);
            if polished_rel <= current_rel.max(1e-12) * 20.0
                && complex_unitarity_error(&up) < complex_unitarity_error(&u)
            {
                return (
                    (up, sigmap, vhp),
                    HermitianJacobiTrace {
                        sweeps: trace.sweeps.max(polish_trace.sweeps),
                        rotations: trace.rotations + polish_trace.rotations,
                        polar_polishes: 1 + polish_trace.polar_polishes,
                    },
                );
            }
        }
    }

    (
        (u, sigma, vh),
        HermitianJacobiTrace {
            sweeps: trace.sweeps,
            rotations: trace.rotations,
            polar_polishes: 0,
        },
    )
}

fn left_hermitian_basis_route(
    mat: &Array2<Complex64>,
    right_u: &Array2<Complex64>,
    sigma: &Array1<f64>,
    vh: &Array2<Complex64>,
    params: LieSvdComplexParams,
    right_trace: HermitianJacobiTrace,
) -> Option<(
    (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
    HermitianJacobiTrace,
)> {
    let n = mat.nrows();
    let left_gram = mat.dot(&mat.t().mapv(|x| x.conj()));
    let (left_evals, left_u, left_trace) =
        hermitian_jacobi_eigen(left_gram, params.max_sweeps, params.tol);
    let mut left_order: Vec<usize> = (0..n).collect();
    left_order.sort_by(|&a, &b| {
        left_evals[b]
            .partial_cmp(&left_evals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut u_left = Array2::<Complex64>::zeros((n, n));
    for (dst, &src) in left_order.iter().enumerate() {
        for r in 0..n {
            u_left[[r, dst]] = left_u[[r, src]];
        }
    }
    let right_rel = complex_relative_reconstruction_error(mat, right_u, sigma, vh);
    let left_rel = complex_relative_reconstruction_error(mat, &u_left, sigma, vh);
    if left_rel <= right_rel.max(1e-12) * 100.0 && left_rel < 1e-10 {
        Some((
            (u_left, sigma.clone(), vh.clone()),
            HermitianJacobiTrace {
                sweeps: right_trace.sweeps.max(left_trace.sweeps),
                rotations: right_trace.rotations + left_trace.rotations,
                polar_polishes: 0,
            },
        ))
    } else {
        None
    }
}

fn qr_polar_polish(
    mat: &Array2<Complex64>,
    u: &Array2<Complex64>,
    sigma: &Array1<f64>,
    vh: &Array2<Complex64>,
    mut params: LieSvdComplexParams,
) -> Option<(
    (Array2<Complex64>, Array1<f64>, Array2<Complex64>),
    HermitianJacobiTrace,
)> {
    if mat.nrows() != mat.ncols() || u.nrows() != u.ncols() {
        return None;
    }
    params.use_golden_prespin = false;
    params.use_polar_polish = false;
    params.max_sweeps = params.max_sweeps.max(64);

    let (q, r) = modified_gram_schmidt_qr(u);
    if complex_unitarity_error(&q) > complex_unitarity_error(u).min(1e-8) {
        return None;
    }
    let n = sigma.len();
    let mut core = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            core[[i, j]] = r[[i, j]] * sigma[j];
        }
    }

    let ((ub, sigma_b, vhb), trace) = svd_via_hermitian_jacobi_inner(&core, params, false);
    let u_polished = q.dot(&ub);
    let vh_polished = vhb.dot(vh);
    Some(((u_polished, sigma_b, vh_polished), trace))
}

fn modified_gram_schmidt_qr(a: &Array2<Complex64>) -> (Array2<Complex64>, Array2<Complex64>) {
    let n = a.nrows();
    let cols = a.ncols();
    let mut q = Array2::<Complex64>::zeros((n, cols));
    let mut r = Array2::<Complex64>::zeros((cols, cols));
    let mut v = vec![Complex64::new(0.0, 0.0); n];
    let mut best = vec![Complex64::new(0.0, 0.0); n];

    for j in 0..cols {
        for i in 0..n {
            v[i] = a[[i, j]];
        }
        for k in 0..j {
            let mut dot = Complex64::new(0.0, 0.0);
            for i in 0..n {
                dot += q[[i, k]].conj() * v[i];
            }
            r[[k, j]] = dot;
            for i in 0..n {
                v[i] -= dot * q[[i, k]];
            }
        }
        let mut norm = vector_norm(&v);
        if norm <= 1e-12 {
            let mut best_norm = -1.0_f64;
            for axis in 0..n {
                v.fill(Complex64::new(0.0, 0.0));
                v[axis] = Complex64::new(1.0, 0.0);
                for k in 0..j {
                    let mut dot = Complex64::new(0.0, 0.0);
                    for i in 0..n {
                        dot += q[[i, k]].conj() * v[i];
                    }
                    for i in 0..n {
                        v[i] -= dot * q[[i, k]];
                    }
                }
                let candidate = vector_norm(&v);
                if candidate > best_norm {
                    best_norm = candidate;
                    best.copy_from_slice(&v);
                }
            }
            v.copy_from_slice(&best);
            norm = vector_norm(&v);
        }
        let norm = norm.max(1e-300);
        r[[j, j]] = Complex64::new(norm, 0.0);
        for i in 0..n {
            q[[i, j]] = v[i] / norm;
        }
    }
    (q, r)
}

fn hermitian_jacobi_eigen(
    mut h: Array2<Complex64>,
    max_sweeps: usize,
    tol: f64,
) -> (Array1<f64>, Array2<Complex64>, HermitianJacobiTrace) {
    let n = h.nrows();
    let original = h.clone();
    let mut q = eye_complex(n);
    let ref_norm = frobenius_norm(&h).max(1e-300);
    let target = tol * ref_norm.max(1.0);
    let mut rotations = 0usize;
    let mut sweeps_done = 0usize;

    for sweep in 0..max_sweeps.max(1) {
        sweeps_done = sweep + 1;
        if hermitian_offdiag_norm(&h) <= target {
            break;
        }
        let mut changed = false;
        for p in 0..n {
            for r in (p + 1)..n {
                let z = h[[p, r]];
                if z.norm() <= target {
                    continue;
                }
                let rot = hermitian_pair_rotation(h[[p, p]].re, z, h[[r, r]].re);
                apply_pair_unitary_hermitian(&mut h, p, r, rot);
                apply_pair_unitary_basis(&mut q, p, r, rot);
                rotations += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        h = q.t().mapv(|x| x.conj()).dot(&original).dot(&q);
        for i in 0..n {
            h[[i, i]].im = 0.0;
        }
    }

    let evals = Array1::from_shape_fn(n, |i| h[[i, i]].re);
    (
        evals,
        q,
        HermitianJacobiTrace {
            sweeps: sweeps_done,
            rotations,
            polar_polishes: 0,
        },
    )
}

#[derive(Clone, Copy)]
struct PairUnitary {
    u00: Complex64,
    u01: Complex64,
    u10: Complex64,
    u11: Complex64,
}

fn hermitian_pair_rotation(a: f64, z: Complex64, d: f64) -> PairUnitary {
    let b = z.norm();
    if b <= 1e-300 {
        return PairUnitary {
            u00: Complex64::new(1.0, 0.0),
            u01: Complex64::new(0.0, 0.0),
            u10: Complex64::new(0.0, 0.0),
            u11: Complex64::new(1.0, 0.0),
        };
    }
    let delta = 0.5 * (a - d);
    let radius = (delta * delta + b * b).sqrt();
    let lambda_hi = 0.5 * (a + d) + radius;
    let t = if (lambda_hi - a).abs() <= 1e-300 {
        0.0
    } else {
        (lambda_hi - a) / b
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    let phase = z.conj() / b;
    PairUnitary {
        u00: Complex64::new(c, 0.0),
        u01: -phase.conj() * s,
        u10: phase * s,
        u11: Complex64::new(c, 0.0),
    }
}

fn apply_pair_unitary_hermitian(h: &mut Array2<Complex64>, p: usize, r: usize, u: PairUnitary) {
    let n = h.nrows();
    for row in 0..n {
        let hp = h[[row, p]];
        let hr = h[[row, r]];
        h[[row, p]] = hp * u.u00 + hr * u.u10;
        h[[row, r]] = hp * u.u01 + hr * u.u11;
    }
    for col in 0..n {
        let hp = h[[p, col]];
        let hr = h[[r, col]];
        h[[p, col]] = u.u00.conj() * hp + u.u10.conj() * hr;
        h[[r, col]] = u.u01.conj() * hp + u.u11.conj() * hr;
    }
    h[[p, r]] = Complex64::new(0.0, 0.0);
    h[[r, p]] = Complex64::new(0.0, 0.0);
    h[[p, p]].im = 0.0;
    h[[r, r]].im = 0.0;
}

fn apply_pair_unitary_basis(q: &mut Array2<Complex64>, p: usize, r: usize, u: PairUnitary) {
    let n = q.nrows();
    for row in 0..n {
        let qp = q[[row, p]];
        let qr = q[[row, r]];
        q[[row, p]] = qp * u.u00 + qr * u.u10;
        q[[row, r]] = qp * u.u01 + qr * u.u11;
    }
}

fn normalize_diagonal_phases(
    a: &Array2<Complex64>,
    u: &mut Array2<Complex64>,
    vh: &mut Array2<Complex64>,
    sigma: &Array1<f64>,
) -> usize {
    let v = vh.t().mapv(|x| x.conj());
    let core = u.t().mapv(|x| x.conj()).dot(a).dot(&v);
    let n = sigma.len();
    let mut fixed = 0usize;
    for i in 0..n {
        if sigma[i] <= 1e-300 {
            continue;
        }
        let d = core[[i, i]];
        if d.norm() <= 1e-300 {
            continue;
        }
        let phase = d / d.norm();
        for r in 0..u.nrows() {
            u[[r, i]] *= phase;
        }
        fixed += 1;
    }
    fixed
}

fn normalize_column(q: &mut Array2<Complex64>, col: usize) {
    let norm = column_norm(q, col).max(1e-300);
    for i in 0..q.nrows() {
        q[[i, col]] /= norm;
    }
}

fn column_norm(q: &Array2<Complex64>, col: usize) -> f64 {
    let mut norm = 0.0_f64;
    for i in 0..q.nrows() {
        norm += q[[i, col]].norm_sqr();
    }
    norm.sqrt()
}

fn complete_missing_unitary_columns(q: &mut Array2<Complex64>, locked: &[bool]) {
    let n = q.ncols();
    let mut v = vec![Complex64::new(0.0, 0.0); n];
    let mut best = vec![Complex64::new(0.0, 0.0); n];
    for j in 0..n {
        if locked.get(j).copied().unwrap_or(false) {
            continue;
        }
        for i in 0..n {
            v[i] = q[[i, j]];
        }
        project_against_previous(&mut v, q, j);
        let norm = vector_norm(&v);
        if norm <= 1e-10 {
            let mut best_norm = -1.0_f64;
            for k in 0..n {
                v.fill(Complex64::new(0.0, 0.0));
                v[k] = Complex64::new(1.0, 0.0);
                project_against_previous(&mut v, q, j);
                let candidate_norm = vector_norm(&v);
                if candidate_norm > best_norm {
                    best_norm = candidate_norm;
                    best.copy_from_slice(&v);
                }
            }
            v.copy_from_slice(&best);
        }
        let norm = vector_norm(&v).max(1e-300);
        for i in 0..n {
            q[[i, j]] = v[i] / norm;
        }
    }
}

fn project_against_previous(v: &mut [Complex64], q: &Array2<Complex64>, upto: usize) {
    for col in 0..upto {
        let mut dot = Complex64::new(0.0, 0.0);
        for i in 0..v.len() {
            dot += q[[i, col]].conj() * v[i];
        }
        for i in 0..v.len() {
            v[i] -= dot * q[[i, col]];
        }
    }
}

fn vector_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

#[derive(Clone, Copy)]
enum GoldenSide {
    Row,
    Col,
}

fn golden_phase(k: usize, side: GoldenSide) -> f64 {
    let multiplier = match side {
        GoldenSide::Row => 1.0,
        GoldenSide::Col => 1.6180339887498948,
    };
    (k as f64) * GOLDEN_ANGLE * multiplier
}

fn cis(theta: f64) -> Complex64 {
    let (s, c) = theta.sin_cos();
    Complex64::new(c, s)
}

fn eye_complex(n: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

fn frobenius_norm(a: &Array2<Complex64>) -> f64 {
    a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

fn offdiag_norm(a: &Array2<Complex64>) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if i != j {
                s += a[[i, j]].norm_sqr();
            }
        }
    }
    s.sqrt()
}

fn hermitian_offdiag_norm(a: &Array2<Complex64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                s += a[[i, j]].norm_sqr();
            }
        }
    }
    s.sqrt()
}

fn phase_stress(a: &Array2<Complex64>) -> f64 {
    let mut stress = 0.0_f64;
    for row in 0..a.nrows() {
        for col in 0..a.ncols() {
            let z = a[[row, col]];
            if z.norm() > 1e-300 {
                stress += z.arg().abs() * z.norm();
            }
        }
    }
    stress
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_complex(n: usize) -> Array2<Complex64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            let re = ((i * 17 + j * 11 + 3) as f64).sin();
            let im = ((i * 7 + j * 19 + 5) as f64).cos() * 0.4;
            if i == j {
                Complex64::new(4.0 + 0.2 * i as f64 + 0.01 * re, 0.01 * im)
            } else {
                Complex64::new(0.005 * re, 0.005 * im)
            }
        })
    }

    #[test]
    fn complex_svd_reconstructs_random_like_matrix() {
        let a = synthetic_complex(8);
        let ((u, sigma, vh), trace) =
            LieSvdComplex::solve_with_trace(&a, LieSvdComplexParams::for_n(8));
        let rel = complex_relative_reconstruction_error(&a, &u, &sigma, &vh);
        let orth_u = complex_unitarity_error(&u);
        let orth_v = complex_unitarity_error(&vh.t().mapv(|x| x.conj()));
        assert!(trace.rotations > 0);
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 2e-2, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn complex_golden_prespin_records_phase_events() {
        let a = synthetic_complex(6);
        let mut params = LieSvdComplexParams::for_n(6);
        params.use_golden_prespin = true;
        params.golden_prespin_layers = 2;
        let ((_u, _sigma, _vh), trace) = LieSvdComplex::solve_with_trace(&a, params);
        assert_eq!(trace.golden_prespins, 12);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|x| x.kind == ComplexPhaseEventKind::GoldenPreSpin));
    }

    #[test]
    fn complex_2x2_microkernel_reconstructs() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.5, -0.7),
                Complex64::new(-0.3, 0.4),
                Complex64::new(2.0, -0.1),
            ],
        )
        .unwrap();
        let (u, sigma, vh) = LieSvdComplex::solve_2x2_micro(&a);
        let rel = complex_relative_reconstruction_error(&a, &u, &sigma, &vh);
        assert!(rel < 1e-11, "rel={rel:e}");
    }
}
