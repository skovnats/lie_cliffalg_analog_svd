//! Higher-order phase SVD / Tucker-style tensor factorization.
//!
//! This module is intentionally small: it implements a 3D HO-SVD route by
//! diagonalizing the Gram matrix of each tensor mode with the existing robust
//! SVD solver, then rotating the tensor into a core. The phase interpretation
//! is mode-wise: each mode owns its own Clifford-like axis family, and the core
//! measures how much mass lands on the superdiagonal.

use ndarray::{Array2, Array3};

#[derive(Clone, Debug)]
pub struct TensorPhaseTrace {
    pub shape: (usize, usize, usize),
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub superdiag_mass_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct TensorPhaseSvd3 {
    pub core: Array3<f64>,
    pub u1: Array2<f64>,
    pub u2: Array2<f64>,
    pub u3: Array2<f64>,
    pub trace: TensorPhaseTrace,
}

pub struct LieSvdTensor;

impl LieSvdTensor {
    pub fn hosvd3(tensor: &Array3<f64>) -> TensorPhaseSvd3 {
        let (n1, n2, n3) = tensor.dim();
        assert!(n1 > 0 && n2 > 0 && n3 > 0, "empty tensor");
        let g1 = mode_gram(tensor, 0);
        let g2 = mode_gram(tensor, 1);
        let g3 = mode_gram(tensor, 2);
        let (u1, _s1, _vt1) = crate::lie_svd_small::LieSvdSmall::solve(&g1);
        let (u2, _s2, _vt2) = crate::lie_svd_small::LieSvdSmall::solve(&g2);
        let (u3, _s3, _vt3) = crate::lie_svd_small::LieSvdSmall::solve(&g3);
        let initial_offdiag = tensor_offdiag_norm(tensor);
        let core = rotate_tensor3(
            tensor,
            &u1.t().to_owned(),
            &u2.t().to_owned(),
            &u3.t().to_owned(),
        );
        let final_offdiag = tensor_offdiag_norm(&core);
        let superdiag_mass_ratio = superdiag_mass_ratio(&core);
        TensorPhaseSvd3 {
            core,
            u1,
            u2,
            u3,
            trace: TensorPhaseTrace {
                shape: (n1, n2, n3),
                initial_offdiag,
                final_offdiag,
                superdiag_mass_ratio,
            },
        }
    }
}

pub fn reconstruct_hosvd3(f: &TensorPhaseSvd3) -> Array3<f64> {
    rotate_tensor3(&f.core, &f.u1, &f.u2, &f.u3)
}

pub fn tensor_relative_error(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    let diff = (a - b).mapv(|x| x * x).sum().sqrt();
    diff / a.mapv(|x| x * x).sum().sqrt().max(1e-300)
}

fn mode_gram(tensor: &Array3<f64>, mode: usize) -> Array2<f64> {
    let (n1, n2, n3) = tensor.dim();
    let n = [n1, n2, n3][mode];
    let mut g = Array2::<f64>::zeros((n, n));
    match mode {
        0 => {
            for a in 0..n1 {
                for b in 0..n1 {
                    let mut s = 0.0;
                    for j in 0..n2 {
                        for k in 0..n3 {
                            s += tensor[[a, j, k]] * tensor[[b, j, k]];
                        }
                    }
                    g[[a, b]] = s;
                }
            }
        }
        1 => {
            for a in 0..n2 {
                for b in 0..n2 {
                    let mut s = 0.0;
                    for i in 0..n1 {
                        for k in 0..n3 {
                            s += tensor[[i, a, k]] * tensor[[i, b, k]];
                        }
                    }
                    g[[a, b]] = s;
                }
            }
        }
        _ => {
            for a in 0..n3 {
                for b in 0..n3 {
                    let mut s = 0.0;
                    for i in 0..n1 {
                        for j in 0..n2 {
                            s += tensor[[i, j, a]] * tensor[[i, j, b]];
                        }
                    }
                    g[[a, b]] = s;
                }
            }
        }
    }
    g
}

fn rotate_tensor3(
    tensor: &Array3<f64>,
    q1: &Array2<f64>,
    q2: &Array2<f64>,
    q3: &Array2<f64>,
) -> Array3<f64> {
    let (n1, n2, n3) = tensor.dim();
    let mut out = Array3::<f64>::zeros((q1.nrows(), q2.nrows(), q3.nrows()));
    for a in 0..q1.nrows() {
        for b in 0..q2.nrows() {
            for c in 0..q3.nrows() {
                let mut s = 0.0_f64;
                for i in 0..n1 {
                    for j in 0..n2 {
                        for k in 0..n3 {
                            s += q1[[a, i]] * q2[[b, j]] * q3[[c, k]] * tensor[[i, j, k]];
                        }
                    }
                }
                out[[a, b, c]] = s;
            }
        }
    }
    out
}

fn tensor_offdiag_norm(tensor: &Array3<f64>) -> f64 {
    let (n1, n2, n3) = tensor.dim();
    let mut s = 0.0_f64;
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                if !(i == j && j == k) {
                    s += tensor[[i, j, k]] * tensor[[i, j, k]];
                }
            }
        }
    }
    s.sqrt()
}

fn superdiag_mass_ratio(tensor: &Array3<f64>) -> f64 {
    let (n1, n2, n3) = tensor.dim();
    let n = n1.min(n2).min(n3);
    let mut diag = 0.0_f64;
    let mut all = 0.0_f64;
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                let v = tensor[[i, j, k]];
                all += v * v;
            }
        }
    }
    for i in 0..n {
        diag += tensor[[i, i, i]] * tensor[[i, i, i]];
    }
    diag / all.max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosvd3_reconstructs_low_rank_tensor() {
        let n = 5;
        let tensor = Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            if i == j && j == k {
                5.0 - i as f64
            } else {
                1e-3 * ((i * 11 + j * 7 + k * 5) as f64).sin()
            }
        });
        let f = LieSvdTensor::hosvd3(&tensor);
        let recon = reconstruct_hosvd3(&f);
        let rel = tensor_relative_error(&tensor, &recon);
        assert!(rel < 1e-10, "rel={rel}");
        assert!(f.trace.superdiag_mass_ratio > 0.95);
    }
}
