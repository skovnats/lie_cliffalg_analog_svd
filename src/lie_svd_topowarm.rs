//! Landmark/topological warm-start for two-sided rotor flows.
//!
//! This is a deliberately modest version of the "intersecting spheres" idea:
//! choose a few far-apart row/column landmarks, build cheap distance/norm
//! features, add a cheap stationary/Fiedler-like axis from the bipartite graph
//! `|A|`, run a tiny two-sided power refinement, and retract the result onto
//! `O(n)`. It is a warm-start, not a closed-form SVD.

use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct TopologicalWarmStartParams {
    pub rank: usize,
    pub landmark_count: usize,
    pub phase_landmark_count: usize,
    pub graph_relax_steps: usize,
    pub power_steps: usize,
    pub random_probe_seed: u64,
    pub eps: f64,
}

impl TopologicalWarmStartParams {
    pub fn for_n(n: usize) -> Self {
        Self {
            rank: n.min(8).max(1),
            landmark_count: n.min(4).max(1),
            phase_landmark_count: n.min(2).max(1),
            graph_relax_steps: 2,
            power_steps: 2,
            random_probe_seed: 0x9e37_79b9_7f4a_7c15,
            eps: 1e-12,
        }
    }
}

impl Default for TopologicalWarmStartParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct TopologicalWarmStartTrace {
    pub rank: usize,
    pub landmark_count: usize,
    pub graph_relax_steps: usize,
    pub power_steps: usize,
    pub identity_offdiag: f64,
    pub warm_offdiag: f64,
    pub accepted: bool,
    pub row_landmarks: Vec<usize>,
    pub col_landmarks: Vec<usize>,
    pub row_phase_landmarks: Vec<usize>,
    pub col_phase_landmarks: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TopologicalWarmStart {
    pub u: Array2<f64>,
    pub v: Array2<f64>,
    pub core: Array2<f64>,
    pub trace: TopologicalWarmStartTrace,
}

pub fn compute_topological_warm_start(
    a: &Array2<f64>,
    params: TopologicalWarmStartParams,
) -> TopologicalWarmStart {
    let n = a.nrows();
    assert_eq!(
        n,
        a.ncols(),
        "topological warm-start expects square matrices"
    );
    let rank = params.rank.clamp(1, n);
    let landmark_count = params.landmark_count.clamp(1, n).min(rank.max(1));
    let phase_landmark_count = params.phase_landmark_count.min(landmark_count).min(n);
    let row_phase_landmarks = select_phase_landmarks(a, false, phase_landmark_count);
    let col_phase_landmarks = select_phase_landmarks(a, true, phase_landmark_count);
    let row_landmarks =
        select_phase_guided_landmarks(a, false, landmark_count, &row_phase_landmarks);
    let col_landmarks =
        select_phase_guided_landmarks(a, true, landmark_count, &col_phase_landmarks);
    let (row_degree, col_degree, row_axis, col_axis) =
        bipartite_stationary_axes(a, params.graph_relax_steps, params.eps);
    let mut u_thin = thin_orthonormalize(
        &feature_seed(
            a,
            false,
            &row_landmarks,
            &row_degree,
            &row_axis,
            rank,
            params.random_probe_seed,
        ),
        rank,
        params.eps,
    );
    let mut v_thin = thin_orthonormalize(
        &feature_seed(
            a,
            true,
            &col_landmarks,
            &col_degree,
            &col_axis,
            rank,
            params.random_probe_seed ^ 0xd1b5_4a32_d192_ed03,
        ),
        rank,
        params.eps,
    );

    for _ in 0..params.power_steps {
        let next_u = a.dot(&v_thin);
        u_thin = thin_orthonormalize(&next_u, rank, params.eps);
        let next_v = a.t().dot(&u_thin);
        v_thin = thin_orthonormalize(&next_v, rank, params.eps);
    }

    let u = complete_orthonormal(&u_thin, params.eps);
    let v = complete_orthonormal(&v_thin, params.eps);
    let core = u.t().dot(a).dot(&v);
    let identity_offdiag = offdiag_norm(a);
    let warm_offdiag = offdiag_norm(&core);
    if warm_offdiag > identity_offdiag {
        return TopologicalWarmStart {
            u: Array2::<f64>::eye(n),
            v: Array2::<f64>::eye(n),
            core: a.clone(),
            trace: TopologicalWarmStartTrace {
                rank,
                landmark_count,
                graph_relax_steps: params.graph_relax_steps,
                power_steps: params.power_steps,
                identity_offdiag,
                warm_offdiag: identity_offdiag,
                accepted: false,
                row_landmarks,
                col_landmarks,
                row_phase_landmarks,
                col_phase_landmarks,
            },
        };
    }
    TopologicalWarmStart {
        u,
        v,
        core,
        trace: TopologicalWarmStartTrace {
            rank,
            landmark_count,
            graph_relax_steps: params.graph_relax_steps,
            power_steps: params.power_steps,
            identity_offdiag,
            warm_offdiag,
            accepted: true,
            row_landmarks,
            col_landmarks,
            row_phase_landmarks,
            col_phase_landmarks,
        },
    }
}

pub fn offdiag_norm(a: &Array2<f64>) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if i != j {
                s += a[[i, j]] * a[[i, j]];
            }
        }
    }
    s.sqrt()
}

fn feature_seed(
    a: &Array2<f64>,
    columns: bool,
    landmarks: &[usize],
    degree: &[f64],
    fiedler_like: &[f64],
    rank: usize,
    random_seed: u64,
) -> Array2<f64> {
    let n = a.nrows();
    let mut seed = Array2::<f64>::zeros((n, rank));
    for i in 0..n {
        seed[[i, 0]] = 1.0;
    }
    if rank > 1 {
        for i in 0..n {
            seed[[i, 1]] = degree[i];
        }
        center_column(&mut seed, 1);
    }
    if rank > 2 {
        for i in 0..n {
            seed[[i, 2]] = fiedler_like[i];
        }
        center_column(&mut seed, 2);
    }
    for (k, &landmark) in landmarks.iter().enumerate() {
        let col = k + 3;
        if col >= rank {
            break;
        }
        for i in 0..n {
            seed[[i, col]] = point_distance_sq(a, columns, i, landmark);
        }
        center_column(&mut seed, col);
    }
    for col in (3 + landmarks.len())..rank {
        for i in 0..n {
            seed[[i, col]] = deterministic_probe(random_seed, i, col);
        }
        center_column(&mut seed, col);
    }
    seed
}

fn bipartite_stationary_axes(
    a: &Array2<f64>,
    steps: usize,
    eps: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = a.nrows();
    let mut row_degree = vec![0.0_f64; n];
    let mut col_degree = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            let w = a[[i, j]].abs();
            row_degree[i] += w;
            col_degree[j] += w;
        }
    }
    let mut row_axis = centered_copy(&row_degree);
    let mut col_axis = centered_copy(&col_degree);
    normalize_vec(&mut row_axis, eps);
    normalize_vec(&mut col_axis, eps);

    let mut next_row = vec![0.0_f64; n];
    let mut next_col = vec![0.0_f64; n];
    for _ in 0..steps {
        next_col.fill(0.0);
        for i in 0..n {
            let row_scale = row_degree[i].max(eps).sqrt();
            for j in 0..n {
                let denom = row_scale * col_degree[j].max(eps).sqrt();
                next_col[j] += a[[i, j]].abs() * row_axis[i] / denom;
            }
        }
        center_vec(&mut next_col);
        normalize_vec(&mut next_col, eps);

        next_row.fill(0.0);
        for i in 0..n {
            let row_scale = row_degree[i].max(eps).sqrt();
            for j in 0..n {
                let denom = row_scale * col_degree[j].max(eps).sqrt();
                next_row[i] += a[[i, j]].abs() * next_col[j] / denom;
            }
        }
        center_vec(&mut next_row);
        normalize_vec(&mut next_row, eps);
        row_axis.clone_from_slice(&next_row);
        col_axis.clone_from_slice(&next_col);
    }

    (row_degree, col_degree, row_axis, col_axis)
}

fn center_column(seed: &mut Array2<f64>, col: usize) {
    let n = seed.nrows();
    let mean = (0..n).map(|i| seed[[i, col]]).sum::<f64>() / n.max(1) as f64;
    for i in 0..n {
        seed[[i, col]] -= mean;
    }
}

fn thin_orthonormalize(seed: &Array2<f64>, rank: usize, eps: f64) -> Array2<f64> {
    let n = seed.nrows();
    let rank = rank.min(seed.ncols()).min(n);
    let mut q = Array2::<f64>::zeros((n, rank));
    let mut used = 0usize;
    let mut scratch = vec![0.0_f64; n];
    for col in 0..seed.ncols() {
        if used >= rank {
            break;
        }
        for i in 0..n {
            scratch[i] = seed[[i, col]];
        }
        subtract_projection(&mut scratch, &q, used);
        let norm = vec_norm(&scratch);
        if norm > eps {
            write_normalized_column(&mut q, used, &scratch, norm);
            used += 1;
        }
    }
    while used < rank {
        fill_best_basis_residual(&mut scratch, &q, used);
        let norm = vec_norm(&scratch).max(eps);
        write_normalized_column(&mut q, used, &scratch, norm);
        used += 1;
    }
    q
}

fn complete_orthonormal(thin: &Array2<f64>, eps: f64) -> Array2<f64> {
    let n = thin.nrows();
    let mut q = Array2::<f64>::zeros((n, n));
    let keep = thin.ncols().min(n);
    for col in 0..keep {
        for i in 0..n {
            q[[i, col]] = thin[[i, col]];
        }
    }
    let mut used = keep;
    let mut scratch = vec![0.0_f64; n];
    while used < n {
        fill_best_basis_residual(&mut scratch, &q, used);
        let norm = vec_norm(&scratch).max(eps);
        write_normalized_column(&mut q, used, &scratch, norm);
        used += 1;
    }
    q
}

fn write_normalized_column(q: &mut Array2<f64>, col: usize, v: &[f64], norm: f64) {
    for i in 0..v.len() {
        q[[i, col]] = v[i] / norm;
    }
}

fn fill_best_basis_residual(v: &mut [f64], q: &Array2<f64>, used: usize) {
    let mut best_basis = 0usize;
    let mut best_residual = -1.0_f64;
    for basis in 0..v.len() {
        let occupied = (0..used)
            .map(|col| q[[basis, col]] * q[[basis, col]])
            .sum::<f64>();
        let residual = (1.0 - occupied).max(0.0);
        if residual > best_residual {
            best_residual = residual;
            best_basis = basis;
        }
    }
    v.fill(0.0);
    v[best_basis] = 1.0;
    subtract_projection(v, q, used);
}

fn subtract_projection(v: &mut [f64], q: &Array2<f64>, used: usize) {
    for col in 0..used {
        let dot = v
            .iter()
            .enumerate()
            .map(|(i, x)| x * q[[i, col]])
            .sum::<f64>();
        for i in 0..v.len() {
            v[i] -= dot * q[[i, col]];
        }
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn centered_copy(v: &[f64]) -> Vec<f64> {
    let mut out = v.to_vec();
    center_vec(&mut out);
    out
}

fn center_vec(v: &mut [f64]) {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    for x in v {
        *x -= mean;
    }
}

fn normalize_vec(v: &mut [f64], eps: f64) {
    let norm = vec_norm(v);
    if norm > eps {
        for x in v {
            *x /= norm;
        }
    }
}

fn deterministic_probe(seed: u64, row: usize, col: usize) -> f64 {
    let mut x = seed
        ^ ((row as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        ^ ((col as u64 + 1).wrapping_mul(0x94d0_49bb_1331_11eb));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    let unit = ((x >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    2.0 * unit - 1.0
}

fn select_phase_guided_landmarks(
    a: &Array2<f64>,
    columns: bool,
    count: usize,
    phase_seed: &[usize],
) -> Vec<usize> {
    let n = a.nrows();
    let mut landmarks = Vec::with_capacity(count);
    for &idx in phase_seed {
        if idx < n && !landmarks.contains(&idx) && landmarks.len() < count {
            landmarks.push(idx);
        }
    }
    if landmarks.is_empty() && count > 0 {
        let first = (0..n)
            .max_by(|&i, &j| {
                point_norm_sq(a, columns, i)
                    .partial_cmp(&point_norm_sq(a, columns, j))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        landmarks.push(first);
    }
    while landmarks.len() < count {
        let next = (0..n).filter(|i| !landmarks.contains(i)).max_by(|&i, &j| {
            min_landmark_distance(a, columns, i, &landmarks)
                .partial_cmp(&min_landmark_distance(a, columns, j, &landmarks))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match next {
            Some(idx) => landmarks.push(idx),
            None => break,
        }
    }
    landmarks
}

fn select_phase_landmarks(a: &Array2<f64>, columns: bool, count: usize) -> Vec<usize> {
    let n = if columns { a.ncols() } else { a.nrows() };
    let mut axes: Vec<usize> = (0..n).collect();
    axes.sort_by(|&i, &j| {
        phase_axis_score(a, columns, j)
            .partial_cmp(&phase_axis_score(a, columns, i))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    axes.truncate(count.min(n));
    axes
}

fn phase_axis_score(a: &Array2<f64>, columns: bool, idx: usize) -> f64 {
    let len = if columns { a.nrows() } else { a.ncols() };
    if len == 0 {
        return 0.0;
    }
    let at = |k: usize| {
        if columns {
            a[[k, idx]]
        } else {
            a[[idx, k]]
        }
    };
    let mut norm_sq = 0.0_f64;
    let mut delay_dot = 0.0_f64;
    let mut gradient_sq = 0.0_f64;
    for k in 0..len {
        let x = at(k);
        let y = at((k + 1) % len);
        norm_sq += x * x;
        delay_dot += x * y;
        let d = y - x;
        gradient_sq += d * d;
    }
    let bivector_sq = (norm_sq * norm_sq - delay_dot * delay_dot).max(0.0);
    bivector_sq.sqrt() / norm_sq.max(1e-300) + gradient_sq.sqrt()
}

fn min_landmark_distance(a: &Array2<f64>, columns: bool, i: usize, landmarks: &[usize]) -> f64 {
    landmarks
        .iter()
        .map(|&j| point_distance_sq(a, columns, i, j))
        .fold(f64::INFINITY, f64::min)
}

fn point_norm_sq(a: &Array2<f64>, columns: bool, i: usize) -> f64 {
    if columns {
        (0..a.nrows()).map(|r| a[[r, i]] * a[[r, i]]).sum()
    } else {
        (0..a.ncols()).map(|c| a[[i, c]] * a[[i, c]]).sum()
    }
}

fn point_distance_sq(a: &Array2<f64>, columns: bool, i: usize, j: usize) -> f64 {
    if columns {
        (0..a.nrows())
            .map(|r| {
                let d = a[[r, i]] - a[[r, j]];
                d * d
            })
            .sum()
    } else {
        (0..a.ncols())
            .map(|c| {
                let d = a[[i, c]] - a[[j, c]];
                d * d
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orth_error(q: &Array2<f64>) -> f64 {
        let ident = Array2::<f64>::eye(q.nrows());
        (&q.t().dot(q) - &ident).mapv(|x| x * x).sum().sqrt()
    }

    #[test]
    fn test_topological_warm_start_returns_orthogonal_bases() {
        let n = 18;
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            let block = if i / 6 == j / 6 { 2.0 } else { 0.05 };
            block + ((i * 17 + j * 31) as f64).sin() * 1e-3
        });
        let warm = compute_topological_warm_start(
            &a,
            TopologicalWarmStartParams {
                rank: 6,
                landmark_count: 4,
                graph_relax_steps: 1,
                power_steps: 1,
                ..TopologicalWarmStartParams::for_n(n)
            },
        );
        assert_eq!(warm.core.dim(), (n, n));
        assert!(orth_error(&warm.u) < 1e-10);
        assert!(orth_error(&warm.v) < 1e-10);
        assert!(warm.trace.warm_offdiag.is_finite());
        assert_eq!(warm.trace.row_landmarks.len(), 4);
        assert_eq!(warm.trace.col_landmarks.len(), 4);
    }

    #[test]
    fn test_topological_warm_start_keeps_already_diagonal_stable() {
        let n = 12;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 10.0 - i as f64;
        }
        let warm = compute_topological_warm_start(&a, TopologicalWarmStartParams::for_n(n));
        assert!(
            warm.trace.warm_offdiag <= 1e-8,
            "{}",
            warm.trace.warm_offdiag
        );
    }

    #[test]
    fn test_bipartite_stationary_axis_is_centered() {
        let n = 10;
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            if i / 5 == j / 5 {
                3.0 + (i + j) as f64 * 1e-3
            } else {
                0.1
            }
        });
        let (_rd, _cd, row_axis, col_axis) = bipartite_stationary_axes(&a, 2, 1e-12);
        assert!(row_axis.iter().sum::<f64>().abs() < 1e-10);
        assert!(col_axis.iter().sum::<f64>().abs() < 1e-10);
        assert!((vec_norm(&row_axis) - 1.0).abs() < 1e-10);
        assert!((vec_norm(&col_axis) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_landmark_marks_high_twist_axis() {
        let n = 12;
        let mut a = Array2::<f64>::eye(n);
        for j in 0..n {
            a[[3, j]] = if j % 2 == 0 { 5.0 } else { -5.0 };
        }
        let warm = compute_topological_warm_start(
            &a,
            TopologicalWarmStartParams {
                landmark_count: 4,
                phase_landmark_count: 2,
                power_steps: 0,
                ..TopologicalWarmStartParams::for_n(n)
            },
        );
        assert!(warm.trace.row_phase_landmarks.contains(&3));
        assert!(warm.trace.row_landmarks.contains(&3));
    }
}
