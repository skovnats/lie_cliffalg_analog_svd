//! Isolated LAPACK/MPFR ground-truth comparison harness for
//! `lie_cliffalg_analog_svd`. Deliberately kept as a *separate* Cargo
//! package (its own `[workspace]`, only reaching into the main crate via a
//! path dependency) so the main crate's own `Cargo.toml`/`Cargo.lock`
//! never gain a LAPACK/BLAS or MPFR/GMP dependency -- see the main crate's
//! `lie_svd_benchmarks` module doc comment for why that separation matters
//! (the main crate is self-contained by design; comparing against an
//! external reference implementation is a genuinely different, opt-in
//! activity, not something every build of the main crate should carry).
//!
//! Two comparisons, each against a different kind of "ground truth":
//!
//! 1. **LAPACK** (via `ndarray-linalg`, OpenBLAS backend): runs this
//!    crate's own `LieSvdSmall::solve` and LAPACK's `dgesdd` on the same
//!    benchmark matrices from `lie_cliffalg_analog_svd::lie_svd_benchmarks`,
//!    and reports orthogonality, reconstruction accuracy, singular-value
//!    agreement between the two, and wall-clock time for each. This is a
//!    genuine production reference implementation, not a from-scratch
//!    derivation.
//! 2. **MPFR** (via `rug`, arbitrary precision): computes the Hilbert
//!    matrix's determinant at `200`-bit precision and compares it to the
//!    plain `f64` determinant (via LU with partial pivoting, computed
//!    here directly) -- quantifying how much of the `f64` answer is
//!    already representation/arithmetic error, independent of which SVD
//!    solver is used, for exactly the matrix in this whole benchmark
//!    program famous for being on the edge of `f64`'s precision.

use lie_cliffalg_analog_svd::lie_svd_benchmarks::{
    hilbert_matrix, kahan_matrix, pei_matrix, pei_matrix_singular_values, vandermonde_matrix,
};
use lie_cliffalg_analog_svd::lie_svd_small::LieSvdSmall;
use ndarray::Array2;
use ndarray_linalg::SVD;
use std::time::Instant;

fn orth_error(m: &Array2<f64>) -> f64 {
    let k = m.ncols();
    let ident = Array2::<f64>::eye(k);
    (&m.t().dot(m) - &ident).mapv(|x| x * x).sum().sqrt()
}

fn rel_recon_error(a: &Array2<f64>, u: &Array2<f64>, sigma: &[f64], vt: &Array2<f64>) -> f64 {
    let sigma_mat = Array2::from_diag(&ndarray::Array1::from(sigma.to_vec()));
    let recon = u.dot(&sigma_mat).dot(vt);
    let a_norm = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    (&recon - a).mapv(|x| x * x).sum().sqrt() / a_norm
}

fn compare_one(name: &str, a: &Array2<f64>) {
    println!("=== {name} (n={}) ===", a.nrows());

    let start = Instant::now();
    let (u_mine, sigma_mine, vt_mine) = LieSvdSmall::solve(a);
    let mine_time = start.elapsed();
    let mine_orth_u = orth_error(&u_mine);
    let mine_orth_v = orth_error(&vt_mine.t().to_owned());
    let mine_recon = rel_recon_error(a, &u_mine, sigma_mine.as_slice().unwrap(), &vt_mine);

    let start = Instant::now();
    let (u_lapack, sigma_lapack, vt_lapack) = a.svd(true, true).expect("LAPACK dgesdd failed");
    let lapack_time = start.elapsed();
    let u_lapack = u_lapack.expect("LAPACK U requested");
    let vt_lapack = vt_lapack.expect("LAPACK Vt requested");
    let lapack_orth_u = orth_error(&u_lapack);
    let lapack_orth_v = orth_error(&vt_lapack.t().to_owned());
    let lapack_recon = rel_recon_error(a, &u_lapack, sigma_lapack.as_slice().unwrap(), &vt_lapack);

    let mut mine_sorted = sigma_mine.to_vec();
    let mut lapack_sorted = sigma_lapack.to_vec();
    mine_sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
    lapack_sorted.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let max_sigma_diff = mine_sorted
        .iter()
        .zip(lapack_sorted.iter())
        .map(|(m, l)| (m - l).abs() / l.abs().max(1e-300))
        .fold(0.0_f64, f64::max);

    println!(
        "  this crate : orth_u={mine_orth_u:e} orth_v={mine_orth_v:e} rel_recon={mine_recon:e} time={mine_time:?}"
    );
    println!(
        "  LAPACK     : orth_u={lapack_orth_u:e} orth_v={lapack_orth_v:e} rel_recon={lapack_recon:e} time={lapack_time:?}"
    );
    println!("  max relative singular-value disagreement: {max_sigma_diff:e}");
}

fn f64_determinant_via_lu(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut m = a.clone();
    let mut det = 1.0_f64;
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = m[[col, col]].abs();
        for row in (col + 1)..n {
            if m[[row, col]].abs() > pivot_val {
                pivot_row = row;
                pivot_val = m[[row, col]].abs();
            }
        }
        if pivot_row != col {
            for k in 0..n {
                m.swap((pivot_row, k), (col, k));
            }
            det = -det;
        }
        let pivot = m[[col, col]];
        if pivot.abs() < 1e-300 {
            return 0.0;
        }
        det *= pivot;
        for row in (col + 1)..n {
            let factor = m[[row, col]] / pivot;
            for k in col..n {
                m[[row, k]] -= factor * m[[col, k]];
            }
        }
    }
    det
}

fn mpfr_determinant_via_lu(a: &Array2<f64>, precision_bits: u32) -> rug::Float {
    use rug::Float;
    let n = a.nrows();
    let mut m: Vec<Vec<Float>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| Float::with_val(precision_bits, a[[i, j]]))
                .collect()
        })
        .collect();
    let mut det = Float::with_val(precision_bits, 1.0);
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = m[col][col].clone().abs();
        for row in (col + 1)..n {
            let v = m[row][col].clone().abs();
            if v > pivot_val {
                pivot_row = row;
                pivot_val = v;
            }
        }
        if pivot_row != col {
            m.swap(pivot_row, col);
            det = -det;
        }
        let pivot = m[col][col].clone();
        det *= &pivot;
        for row in (col + 1)..n {
            let factor = m[row][col].clone() / &pivot;
            for k in col..n {
                let sub = factor.clone() * m[col][k].clone();
                m[row][k] -= sub;
            }
        }
    }
    det
}

fn main() {
    println!("lie_cliffalg_analog_svd reference-comparison harness");
    println!("(isolated from the main crate's own dependency tree -- see this crate's own Cargo.toml)\n");

    println!("## Part 1: LAPACK (OpenBLAS dgesdd) comparison\n");
    for n in [32usize, 64] {
        compare_one("kahan", &kahan_matrix(n, 1.2));
        compare_one("hilbert", &hilbert_matrix(n));
        compare_one("vandermonde", &vandermonde_matrix(n.min(12)));
        let alpha = 0.01;
        compare_one("pei", &pei_matrix(n, alpha));
    }

    println!("\n## Part 1b: Pei matrix vs its own exact closed form (sanity cross-check)\n");
    let n = 64;
    let alpha = 0.01;
    let a = pei_matrix(n, alpha);
    let (_u, sigma, _vt) = LieSvdSmall::solve(&a);
    let mut got = sigma.to_vec();
    got.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let mut want = pei_matrix_singular_values(n, alpha).to_vec();
    want.sort_by(|x, y| y.partial_cmp(x).unwrap());
    let max_rel = got
        .iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs() / w.abs().max(1e-300))
        .fold(0.0_f64, f64::max);
    println!("  Pei n={n} alpha={alpha}: max relative error vs exact closed form = {max_rel:e}");

    println!("\n## Part 2: MPFR (200-bit) vs f64 -- Hilbert matrix determinant\n");
    for n in [6usize, 8, 10, 12] {
        let a = hilbert_matrix(n);
        let det_f64 = f64_determinant_via_lu(&a);
        let det_mpfr = mpfr_determinant_via_lu(&a, 200);
        let det_f64_high = rug::Float::with_val(200, det_f64);
        let diff = (det_f64_high - det_mpfr.clone()).abs();
        let rel_diff = (diff / det_mpfr.clone().abs().max(&rug::Float::with_val(200, 1e-300)))
            .to_f64();
        println!(
            "  n={n}: det_f64={det_f64:e} det_mpfr(200-bit)={:.10e} rel_diff={rel_diff:e}",
            det_mpfr.to_f64()
        );
    }
}
