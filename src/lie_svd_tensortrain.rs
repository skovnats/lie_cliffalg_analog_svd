//! Kronecker-chain and tensor-train inspired diagnostics.
//!
//! This module is the 0.7.0 "dimension lift" layer. It does not claim that a
//! general dense SVD becomes trivial after a Kronecker reshape. Instead, it
//! asks a cheaper and useful question:
//!
//! ```text
//! does A look like a chain of small 2x2 tensor factors?
//! ```
//!
//! If the answer is yes, the SVD can be assembled from the SVDs of the small
//! factors:
//!
//! ```text
//! A = A0 kron A1 kron ... kron Ak
//! svd(A) = kron(svd(A0), svd(A1), ..., svd(Ak))
//! ```
//!
//! Physically, this is the matrix-product-state / Schmidt-decomposition view:
//! the full space expands as a tensor product, while low bond complexity lets
//! the solver collapse the useful information back through local rotor cells.

use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct TensorTrainSvdParams {
    pub max_local_residual: f64,
    pub max_chain_residual: f64,
    pub min_levels: usize,
}

impl Default for TensorTrainSvdParams {
    fn default() -> Self {
        Self {
            max_local_residual: 1e-10,
            max_chain_residual: 1e-10,
            min_levels: 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Kron2Approx {
    pub n: usize,
    pub child_n: usize,
    pub factor: Array2<f64>,
    pub child: Array2<f64>,
    pub relative_residual: f64,
    pub reference_block: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct KronChain {
    pub factors: Vec<Array2<f64>>,
    pub local_residuals: Vec<f64>,
    pub relative_residual: f64,
}

impl KronChain {
    pub fn levels(&self) -> usize {
        self.factors.len()
    }

    pub fn size(&self) -> usize {
        1usize << self.factors.len()
    }
}

pub fn solve_if_kron_chain(
    mat: &Array2<f64>,
    params: TensorTrainSvdParams,
) -> Option<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let chain = factor_kron2_chain(mat, params)?;
    Some(svd_from_kron_chain(&chain.factors))
}

pub fn factor_kron2_chain(mat: &Array2<f64>, params: TensorTrainSvdParams) -> Option<KronChain> {
    assert_eq!(mat.nrows(), mat.ncols(), "Kron chain expects square matrix");
    let n = mat.nrows();
    if n < 2 || !n.is_power_of_two() {
        return None;
    }

    let mut factors = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut residuals = Vec::with_capacity(n.trailing_zeros() as usize);
    let mut current = mat.clone();

    while current.nrows() > 2 {
        let approx = best_kron2_split(&current)?;
        if approx.relative_residual > params.max_local_residual {
            return None;
        }
        residuals.push(approx.relative_residual);
        factors.push(approx.factor);
        current = approx.child;
    }

    factors.push(current);
    if factors.len() < params.min_levels {
        return None;
    }

    let reconstructed = kron_chain_product(&factors);
    let relative_residual = relative_frobenius_residual(mat, &reconstructed);
    if relative_residual > params.max_chain_residual {
        return None;
    }

    Some(KronChain {
        factors,
        local_residuals: residuals,
        relative_residual,
    })
}

pub fn kron2_diagnostic(mat: &Array2<f64>) -> Option<Kron2Approx> {
    best_kron2_split(mat)
}

pub fn svd_from_kron_chain(factors: &[Array2<f64>]) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    assert!(!factors.is_empty(), "empty Kronecker chain");
    let mut u_chain = Array2::<f64>::from_elem((1, 1), 1.0);
    let mut vt_chain = Array2::<f64>::from_elem((1, 1), 1.0);
    let mut sigma_chain = Array1::<f64>::from_elem(1, 1.0);

    for factor in factors {
        assert_eq!(factor.nrows(), 2, "only 2x2 factors are supported");
        assert_eq!(factor.ncols(), 2, "only 2x2 factors are supported");
        let (u, sigma, vt) = crate::lie_svd_micro::LieSvdMicro::solve(factor);
        u_chain = kron_product(&u_chain, &u);
        vt_chain = kron_product(&vt_chain, &vt);
        sigma_chain = kron_values(&sigma_chain, &sigma);
    }

    sort_svd(u_chain, sigma_chain, vt_chain)
}

pub fn kron_chain_product(factors: &[Array2<f64>]) -> Array2<f64> {
    assert!(!factors.is_empty(), "empty Kronecker chain");
    let mut out = Array2::<f64>::from_elem((1, 1), 1.0);
    for factor in factors {
        out = kron_product(&out, factor);
    }
    out
}

pub fn kron_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    let mut out = Array2::<f64>::zeros((ar * br, ac * bc));
    for i in 0..ar {
        for j in 0..ac {
            let scale = a[[i, j]];
            for r in 0..br {
                for c in 0..bc {
                    out[[i * br + r, j * bc + c]] = scale * b[[r, c]];
                }
            }
        }
    }
    out
}

fn best_kron2_split(mat: &Array2<f64>) -> Option<Kron2Approx> {
    let n = mat.nrows();
    if n < 4 || n != mat.ncols() || n % 2 != 0 {
        return None;
    }
    let m = n / 2;
    let mut best_norm = -1.0_f64;
    let mut reference_block = (0usize, 0usize);

    for br in 0..2 {
        for bc in 0..2 {
            let norm = block_dot(mat, br, bc, mat, br, bc, m).sqrt();
            if norm > best_norm {
                best_norm = norm;
                reference_block = (br, bc);
            }
        }
    }
    if best_norm <= 0.0 || !best_norm.is_finite() {
        return None;
    }

    let ref_sq = best_norm * best_norm;
    let mut child = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            child[[i, j]] = mat[[reference_block.0 * m + i, reference_block.1 * m + j]];
        }
    }

    let mut factor = Array2::<f64>::zeros((2, 2));
    let mut residual_sq = 0.0_f64;
    for br in 0..2 {
        for bc in 0..2 {
            let coeff = block_dot(mat, br, bc, mat, reference_block.0, reference_block.1, m)
                / ref_sq.max(1e-300);
            factor[[br, bc]] = coeff;
            residual_sq += block_residual_sq(mat, br, bc, &child, coeff, m);
        }
    }

    let relative_residual = residual_sq.sqrt() / frobenius_norm(mat).max(1e-300);
    Some(Kron2Approx {
        n,
        child_n: m,
        factor,
        child,
        relative_residual,
        reference_block,
    })
}

fn block_dot(
    lhs: &Array2<f64>,
    lhs_br: usize,
    lhs_bc: usize,
    rhs: &Array2<f64>,
    rhs_br: usize,
    rhs_bc: usize,
    block_n: usize,
) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..block_n {
        for j in 0..block_n {
            acc += lhs[[lhs_br * block_n + i, lhs_bc * block_n + j]]
                * rhs[[rhs_br * block_n + i, rhs_bc * block_n + j]];
        }
    }
    acc
}

fn block_residual_sq(
    mat: &Array2<f64>,
    br: usize,
    bc: usize,
    child: &Array2<f64>,
    coeff: f64,
    block_n: usize,
) -> f64 {
    let mut acc = 0.0_f64;
    for i in 0..block_n {
        for j in 0..block_n {
            let d = mat[[br * block_n + i, bc * block_n + j]] - coeff * child[[i, j]];
            acc += d * d;
        }
    }
    acc
}

fn kron_values(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(a.len() * b.len());
    for i in 0..a.len() {
        for j in 0..b.len() {
            out[i * b.len() + j] = a[i] * b[j];
        }
    }
    out
}

fn sort_svd(
    u: Array2<f64>,
    sigma: Array1<f64>,
    vt: Array2<f64>,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = sigma.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u_sorted = Array2::<f64>::zeros((n, n));
    let mut vt_sorted = Array2::<f64>::zeros((n, n));
    let mut sigma_sorted = Array1::<f64>::zeros(n);
    for (dst, &src) in order.iter().enumerate() {
        sigma_sorted[dst] = sigma[src];
        for r in 0..n {
            u_sorted[[r, dst]] = u[[r, src]];
            vt_sorted[[dst, r]] = vt[[src, r]];
        }
    }
    (u_sorted, sigma_sorted, vt_sorted)
}

fn relative_frobenius_residual(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim(), "residual shape mismatch");
    let mut diff_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        diff_sq += d * d;
        norm_sq += x * x;
    }
    diff_sq.sqrt() / norm_sq.sqrt().max(1e-300)
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factor(a: f64, b: f64, c: f64, d: f64) -> Array2<f64> {
        Array2::from_shape_vec((2, 2), vec![a, b, c, d]).expect("factor")
    }

    fn svd_metrics(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel = relative_frobenius_residual(a, &recon);
        let ident = Array2::<f64>::eye(a.nrows());
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel, orth_u, orth_v)
    }

    #[test]
    fn detects_exact_two_level_kron_chain() {
        let factors = vec![factor(2.0, 0.3, -0.1, 1.1), factor(1.5, -0.2, 0.4, 0.8)];
        let a = kron_chain_product(&factors);
        let chain = factor_kron2_chain(&a, TensorTrainSvdParams::default()).expect("chain");
        assert_eq!(chain.levels(), 2);
        assert!(chain.relative_residual < 1e-13);
    }

    #[test]
    fn kron_chain_svd_reconstructs_exact_chain() {
        let factors = vec![
            factor(2.0, 0.3, -0.1, 1.1),
            factor(1.5, -0.2, 0.4, 0.8),
            factor(0.9, 0.15, -0.25, 1.2),
        ];
        let a = kron_chain_product(&factors);
        let (u, sigma, vt) = solve_if_kron_chain(&a, TensorTrainSvdParams::default()).expect("svd");
        let (rel, orth_u, orth_v) = svd_metrics(&a, &u, &sigma, &vt);
        assert!(rel < 1e-12, "rel={rel:e}");
        assert!(orth_u < 1e-12, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-12, "orth_v={orth_v:e}");
    }

    #[test]
    fn rejects_plain_non_chain_matrix() {
        let a = Array2::from_shape_fn((8, 8), |(i, j)| {
            ((i * 13 + j * 17 + 3) as f64).sin() + 0.1 * (i == j) as i32 as f64
        });
        let params = TensorTrainSvdParams {
            max_local_residual: 1e-8,
            max_chain_residual: 1e-8,
            min_levels: 2,
        };
        assert!(factor_kron2_chain(&a, params).is_none());
    }
}
