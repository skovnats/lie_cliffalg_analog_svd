//! Subspace-coupled joint diagonalization: `lie_svd_joint`'s Phase-JADE
//! generalized to a family of matrices that do **not** all share the same
//! size, only a subset of their generator axes.
//!
//! ## Why the naive "one dense global rotor" framing is wrong
//!
//! The proposal this module implements was originally phrased as: embed
//! every matrix `M_k` (defined on generator subset `S_k`) into one shared
//! `D x D` ambient space via zero-padding, find a single global rotor
//! `R in Spin(D)`, and minimize `sum_k ||offdiag(P_k R P_k^T M_k P_k R^T
//! P_k^T)||_F^2` over the padded family, with a projector `P_k` selecting
//! `M_k`'s active rows/columns.
//!
//! That framing has a real bug, found before implementing rather than
//! after: applying one *shared* `D x D` rotation `G_{ij}` to every
//! zero-padded `tilde M_k` is only harmless for a matrix `M_k` that has
//! **both** `i` and `j` in `S_k`, or **neither**. If `M_k` has exactly one
//! of the two (say `i in S_k`, `j not in S_k`), `G_{ij}` mixes row/column
//! `i` (real data) with row/column `j` (a padded zero, since `M_k` never
//! measured that axis) — which fabricates nonzero entries on an axis the
//! matrix never observed, and the algorithm then "diagonalizes" partly
//! against data that was never there. Concretely: in the module's own test
//! scenario (a `3x3` matrix on axes `{0,1,2}` and a `4x4` matrix on axes
//! `{1,2,3,4}`), any global rotation touching axis pair `(1,3)` — jointly
//! observed only by the second matrix — would, under the naive scheme,
//! also perturb the *first* matrix's padded row/column `3`, which that
//! matrix has no data for at all.
//!
//! The fix matches what the informal write-up's own step 3 already got
//! right in words ("применяется только к тем матрицам, которые содержат
//! обе оси") but not in the formal `P_k R P_k^T` formula: never build a
//! dense padded embedding at all. Instead, each matrix `M_k` keeps its own
//! genuine `d_k x d_k` local accumulator rotor `R_k` (orthogonal by
//! construction, since it only ever receives ordinary Jacobi/Givens updates
//! restricted to its own axes). For a *global* axis pair `(i, j)`, the
//! rotation angle is computed once, jointly, from every matrix that has
//! **both** `i` and `j` (the standard multi-matrix JADE closed form,
//! `joint_symmetric_pair_angle` in `lie_svd_joint`, just restricted to that
//! data-dependent subset of the family) — and that one angle is then
//! applied to each participating matrix at *its own* local indices for `i`
//! and `j`. A matrix missing either axis is not touched at all for that
//! step: literally the identity, not an approximation of it. This is what
//! forces axes shared by two matrices to agree on a rotation while leaving
//! axis pairs no matrix jointly observes at `theta = 0` (skipped, since
//! there is no data to justify rotating a plane nothing measures both ends
//! of).
//!
//! One consequence worth stating plainly: there is in general **no single
//! dense `D x D` orthogonal matrix** whose axis-subset submatrix recovers
//! each `R_k` (a submatrix of an orthogonal matrix is not itself orthogonal
//! in general, so trying to assemble one big rotor and slice it per-matrix
//! would silently reintroduce the same padding bug in a different guise).
//! What this module returns instead is the family of per-matrix local
//! rotors `{R_k}`, which is exactly what's needed downstream: e.g. each
//! `R_k` on its own is a normal orthogonal rotor and can be compiled to an
//! MZI hardware schedule directly via
//! `lie_svd_compiler::HardwareSchedule::from_orthogonal_matrix`, the same
//! path already used for `lie_tbl_regress::procrustes_rotor`.
//!
//! ## Scale-balanced weighting (`0.33.0`)
//!
//! By default, a shared pair's angle is computed from an unweighted sum
//! over participating matrices' raw entries, matching `lie_svd_joint`'s
//! existing convention (that module also does not normalize by family
//! member). A matrix with much larger entries than its siblings then
//! dominates any pair it participates in — its own local diagonalization
//! wins the compromise, while a small-magnitude sibling barely benefits
//! from being in the family at all. `SubspaceJadeParams::weighting` set to
//! `SubspaceWeighting::InverseFrobeniusSquared` fixes this: each matrix
//! `M_k` gets weight `1 / ||M_k||_F^2` (scale-relative floor against the
//! family's mean squared norm, not an absolute one — the same fix this
//! crate already applied once before to `lie_svd_small::qr_reduce`'s
//! rank-deficiency threshold, for the identical reason: an absolute floor
//! is wrong whenever the family's own scale isn't known in advance).
//!
//! This weight is computed **once**, from the original input, not
//! recomputed per sweep — which is exact, not an approximation that goes
//! stale: orthogonal conjugation preserves Frobenius norm exactly
//! (`||R^T M R||_F = ||M||_F` for any orthogonal `R`), so `||M_k||_F`
//! never changes as the algorithm rotates `M_k`. The weighting only shapes
//! which angle gets picked and accepted at each step (`subspace_pair_angle`,
//! `pair_energy_after`, `weighted_local_offdiag_sq`,
//! `weighted_pair_offdiag`, and the sweep's own before/after stopping
//! comparison all become weighted sums when weighting is active);
//! `SubspaceJadeTrace::initial_offdiag` and
//! `final_offdiag` are always reported in **raw, unweighted** Frobenius
//! units regardless of the weighting mode, so they stay a physically
//! meaningful, comparable-across-modes number rather than an artifact of
//! whichever weighting scheme happened to be selected. One consequence
//! worth stating plainly: because the two objectives (weighted, driving
//! the algorithm; raw, reported in the trace) differ when weighting is
//! active, the raw total is not guaranteed to shrink every single sweep —
//! that is the intended trade-off (favoring the up-weighted matrix's fit
//! over the down-weighted one's), not a bug. See
//! `subspace_jade_weighting_helps_the_small_magnitude_matrix` for the
//! direct, measured A/B this was built to pass.
//!
//! ## Scope notes (stated rather than left implicit)
//!
//! - **Real-valued only**, matching `lie_svd_joint`'s symmetric route; no
//!   complex/two-sided variant here.
//! - **Connectivity is exposed, not hidden.** `axis_connected_components`
//!   reports which global axes can possibly influence each other (any two
//!   axes co-occurring in some matrix, transitively) — axes in different
//!   components never interact, by construction, and this makes that
//!   explicit rather than requiring the caller to infer it from an absence
//!   of nonzero off-diagonal entries.

use crate::lie_svd_joint::{
    apply_basis_rotor, apply_symmetric_rotor, local_offdiag_sq_for_axes, offdiag_sq,
    wrap_jacobi_angle,
};
use ndarray::Array2;

/// One matrix in a subspace-coupled family: a `d_k x d_k` symmetric matrix
/// together with the global generator index each of its local rows/columns
/// corresponds to. `axes[l]` is the global axis for local index `l`; no
/// global axis may repeat within one matrix's `axes`.
#[derive(Clone, Debug)]
pub struct SubspaceMatrix {
    pub data: Array2<f64>,
    pub axes: Vec<usize>,
}

impl SubspaceMatrix {
    pub fn new(data: Array2<f64>, axes: Vec<usize>) -> Self {
        assert_eq!(
            data.nrows(),
            data.ncols(),
            "SubspaceMatrix: matrix must be square, got {}x{}",
            data.nrows(),
            data.ncols()
        );
        assert_eq!(
            data.nrows(),
            axes.len(),
            "SubspaceMatrix: {} axes given for a {}x{} matrix",
            axes.len(),
            data.nrows(),
            data.nrows()
        );
        for a in 0..axes.len() {
            for b in (a + 1)..axes.len() {
                assert_ne!(
                    axes[a], axes[b],
                    "SubspaceMatrix: global axis {} repeated within one matrix's local axes",
                    axes[a]
                );
            }
        }
        Self { data, axes }
    }
}

/// How much each matrix's own entries count toward a shared pair's
/// rotation angle. See the module doc comment's "Scale-balanced weighting"
/// section for the exact formula and why it's safe to compute once.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SubspaceWeighting {
    /// Every matrix counts equally toward a shared pair's angle,
    /// regardless of its own scale. Matches `lie_svd_joint`'s existing
    /// convention; the default, so existing callers see no behavior
    /// change.
    #[default]
    Unweighted,
    /// `weight_k = 1 / ||M_k||_F^2`, computed once from the input (exact
    /// throughout the sweep, since orthogonal conjugation preserves
    /// Frobenius norm). Prevents a large-magnitude matrix from dominating
    /// a shared pair's angle at a small-magnitude sibling's expense.
    InverseFrobeniusSquared,
}

#[derive(Clone, Copy, Debug)]
pub struct SubspaceJadeParams {
    pub max_sweeps: usize,
    pub tol: f64,
    pub min_pair_scale: f64,
    pub line_search_steps: usize,
    pub weighting: SubspaceWeighting,
}

impl SubspaceJadeParams {
    pub fn for_axes(global_axes: usize) -> Self {
        Self {
            max_sweeps: 12 + global_axes.max(1),
            tol: 1e-12,
            min_pair_scale: 1e-14,
            line_search_steps: 6,
            weighting: SubspaceWeighting::Unweighted,
        }
    }
}

impl Default for SubspaceJadeParams {
    fn default() -> Self {
        Self::for_axes(32)
    }
}

/// Why the sweep loop stopped, so a caller doesn't have to infer it from
/// comparing `sweeps` against the `max_sweeps` it passed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubspaceJadeStopReason {
    /// The (possibly weighted) off-diagonal energy reached `tol`.
    ReachedTolerance,
    /// A full sweep produced no further improvement (within numerical
    /// slack) — a genuine local optimum for a family with no exact joint
    /// solution, not a failure. See the module doc comment and
    /// `subspace_jade_reduces_offdiag_energy_on_overlapping_3x3_and_4x4_family`.
    Plateaued,
    /// `max_sweeps` was exhausted while still improving each sweep —
    /// raising `max_sweeps` may still help.
    MaxSweepsReached,
}

#[derive(Clone, Debug)]
pub struct SubspaceJadeTrace {
    pub global_axes: usize,
    pub observed_pairs: usize,
    /// Always raw (unweighted) Frobenius off-diagonal energy, regardless
    /// of `SubspaceJadeParams::weighting` — see the module doc comment.
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub sweeps: usize,
    pub rotations: usize,
    pub rejected_rotations: usize,
    pub stop_reason: SubspaceJadeStopReason,
}

#[derive(Clone, Debug)]
pub struct SubspaceJadeResult {
    /// The input matrices after joint diagonalization, same shapes and
    /// `axes` order as the input.
    pub diagonalized: Vec<Array2<f64>>,
    /// One orthogonal `d_k x d_k` rotor per input matrix: `R_k^T M_k R_k`
    /// is `diagonalized[k]`.
    pub local_rotors: Vec<Array2<f64>>,
    /// The per-matrix weight actually used (`1.0` for every matrix under
    /// `SubspaceWeighting::Unweighted`).
    pub weights: Vec<f64>,
    pub trace: SubspaceJadeTrace,
}

pub struct LieSvdSubspaceJade;

impl LieSvdSubspaceJade {
    pub fn diagonalize(matrices: &[SubspaceMatrix]) -> SubspaceJadeResult {
        let global_axes = count_global_axes(matrices);
        Self::diagonalize_with_params(matrices, SubspaceJadeParams::for_axes(global_axes))
    }

    pub fn diagonalize_with_params(
        matrices: &[SubspaceMatrix],
        params: SubspaceJadeParams,
    ) -> SubspaceJadeResult {
        assert!(!matrices.is_empty(), "LieSvdSubspaceJade: empty family");

        let global_axes = count_global_axes(matrices);
        let owners = build_owners(matrices, global_axes);
        let observed = observed_pairs_from_owners(&owners);
        let weights = compute_weights(matrices, params.weighting);

        let mut work: Vec<Array2<f64>> = matrices.iter().map(|m| m.data.clone()).collect();
        let mut local_rotors: Vec<Array2<f64>> = matrices
            .iter()
            .map(|m| Array2::<f64>::eye(m.data.nrows()))
            .collect();

        let ref_norm = weighted_frobenius_norm(&work, &weights).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let tol = params.tol * ref_norm.max(1.0);
        let initial_offdiag = total_offdiag_norm(&work);

        let mut rotations = 0usize;
        let mut rejected_rotations = 0usize;
        let mut sweeps = 0usize;
        let mut stop_reason = SubspaceJadeStopReason::MaxSweepsReached;

        for sweep in 0..params.max_sweeps {
            sweeps = sweep + 1;
            let before = weighted_offdiag_norm(&work, &weights);
            for &(gi, gj) in &observed {
                let participants = participants_for_pair(&owners[gi], &owners[gj]);
                if participants.is_empty() {
                    continue;
                }
                if weighted_pair_offdiag(&work, &weights, &participants) <= pair_tol {
                    continue;
                }
                let theta = subspace_pair_angle(&work, &weights, &participants);
                if theta.abs() <= 1e-18 {
                    continue;
                }
                if accept_subspace_rotor(
                    &mut work,
                    &mut local_rotors,
                    &weights,
                    &participants,
                    theta,
                    params.line_search_steps,
                ) {
                    rotations += 1;
                } else {
                    rejected_rotations += 1;
                }
            }
            let after = weighted_offdiag_norm(&work, &weights);
            if after <= tol {
                stop_reason = SubspaceJadeStopReason::ReachedTolerance;
                break;
            }
            if after >= before * (1.0 - 1e-10) {
                stop_reason = SubspaceJadeStopReason::Plateaued;
                break;
            }
        }

        let final_offdiag = total_offdiag_norm(&work);
        SubspaceJadeResult {
            diagonalized: work,
            local_rotors,
            weights,
            trace: SubspaceJadeTrace {
                global_axes,
                observed_pairs: observed.len(),
                initial_offdiag,
                final_offdiag,
                sweeps,
                rotations,
                rejected_rotations,
                stop_reason,
            },
        }
    }
}

/// `weight_k` for each matrix under the given scheme. For
/// `InverseFrobeniusSquared`, floors each matrix's squared Frobenius norm
/// against a small fraction of the family's *mean* squared norm rather
/// than an absolute constant — an absolute floor would either be too loose
/// for a family of uniformly tiny matrices or too tight for a family of
/// uniformly huge ones; a scale-relative floor is correct regardless of
/// the family's own units (the same fix already applied once in this
/// crate to `lie_svd_small::qr_reduce`'s rank-deficiency threshold).
fn compute_weights(matrices: &[SubspaceMatrix], mode: SubspaceWeighting) -> Vec<f64> {
    match mode {
        SubspaceWeighting::Unweighted => vec![1.0; matrices.len()],
        SubspaceWeighting::InverseFrobeniusSquared => {
            let sq_norms: Vec<f64> = matrices
                .iter()
                .map(|m| m.data.iter().map(|x| x * x).sum::<f64>())
                .collect();
            let mean_sq_norm = sq_norms.iter().sum::<f64>() / sq_norms.len().max(1) as f64;
            let floor = 1e-12 * mean_sq_norm.max(1e-300);
            sq_norms.iter().map(|&f2| 1.0 / (f2 + floor)).collect()
        }
    }
}

/// Connected components of the global axis graph: two global axes are
/// adjacent iff some matrix in the family contains both. Axes in different
/// components never influence each other's rotation, by construction (no
/// shared matrix ever links them), which callers can otherwise only infer
/// indirectly from `theta` always coming out `0` for those pairs.
pub fn axis_connected_components(matrices: &[SubspaceMatrix]) -> Vec<Vec<usize>> {
    let global_axes = count_global_axes(matrices);
    let owners = build_owners(matrices, global_axes);
    let observed = observed_pairs_from_owners(&owners);

    let mut parent: Vec<usize> = (0..global_axes).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    for &(gi, gj) in &observed {
        let ri = find(&mut parent, gi);
        let rj = find(&mut parent, gj);
        if ri != rj {
            parent[ri] = rj;
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for axis in 0..global_axes {
        let root = find(&mut parent, axis);
        groups.entry(root).or_default().push(axis);
    }
    groups.into_values().collect()
}

fn count_global_axes(matrices: &[SubspaceMatrix]) -> usize {
    matrices
        .iter()
        .flat_map(|m| m.axes.iter().copied())
        .max()
        .map(|x| x + 1)
        .unwrap_or(0)
}

/// `owners[g]` lists every `(matrix_index, local_index)` where matrix
/// `matrix_index` has global axis `g` at its own local index `local_index`.
fn build_owners(matrices: &[SubspaceMatrix], global_axes: usize) -> Vec<Vec<(usize, usize)>> {
    let mut owners: Vec<Vec<(usize, usize)>> = vec![Vec::new(); global_axes];
    for (k, m) in matrices.iter().enumerate() {
        for (local, &g) in m.axes.iter().enumerate() {
            owners[g].push((k, local));
        }
    }
    owners
}

fn observed_pairs_from_owners(owners: &[Vec<(usize, usize)>]) -> Vec<(usize, usize)> {
    let global_axes = owners.len();
    let mut pairs = Vec::new();
    for gi in 0..global_axes {
        if owners[gi].is_empty() {
            continue;
        }
        for gj in (gi + 1)..global_axes {
            if owners[gj].is_empty() {
                continue;
            }
            if !participants_for_pair(&owners[gi], &owners[gj]).is_empty() {
                pairs.push((gi, gj));
            }
        }
    }
    pairs
}

/// Matrices that have both global axes `gi` and `gj`, as
/// `(matrix_index, local_index_of_gi, local_index_of_gj)`.
fn participants_for_pair(
    owners_i: &[(usize, usize)],
    owners_j: &[(usize, usize)],
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for &(k, li) in owners_i {
        if let Some(&(_, lj)) = owners_j.iter().find(|&&(kk, _)| kk == k) {
            out.push((k, li, lj));
        }
    }
    out
}

/// Raw (unweighted) Frobenius off-diagonal energy across the whole family
/// -- always what `SubspaceJadeTrace::initial_offdiag`/`final_offdiag`
/// report, regardless of `SubspaceJadeParams::weighting`.
fn total_offdiag_norm(work: &[Array2<f64>]) -> f64 {
    work.iter().map(offdiag_sq).sum::<f64>().sqrt()
}

fn weighted_offdiag_norm(work: &[Array2<f64>], weights: &[f64]) -> f64 {
    work.iter()
        .zip(weights)
        .map(|(m, &w)| w * offdiag_sq(m))
        .sum::<f64>()
        .sqrt()
}

fn weighted_frobenius_norm(work: &[Array2<f64>], weights: &[f64]) -> f64 {
    work.iter()
        .zip(weights)
        .map(|(m, &w)| w * m.iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        .sqrt()
}

fn weighted_pair_offdiag(
    work: &[Array2<f64>],
    weights: &[f64],
    participants: &[(usize, usize, usize)],
) -> f64 {
    participants
        .iter()
        .map(|&(k, li, lj)| weights[k] * (work[k][[li, lj]].abs() + work[k][[lj, li]].abs()))
        .sum()
}

fn weighted_local_offdiag_sq(
    work: &[Array2<f64>],
    weights: &[f64],
    participants: &[(usize, usize, usize)],
) -> f64 {
    participants
        .iter()
        .map(|&(k, li, lj)| weights[k] * local_offdiag_sq_for_axes(&work[k], li, lj))
        .sum()
}

/// The standard multi-matrix JADE closed-form Givens angle
/// (`lie_svd_joint::joint_symmetric_pair_angle`'s formula), restricted to
/// just the matrices that participate in this global axis pair, each
/// indexed by its own local row/column for the two axes, and weighted by
/// each participant's `weights[k]`.
fn subspace_pair_angle(
    work: &[Array2<f64>],
    weights: &[f64],
    participants: &[(usize, usize, usize)],
) -> f64 {
    let mut sxx = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sxy = 0.0_f64;
    for &(k, li, lj) in participants {
        let m = &work[k];
        let w = weights[k];
        let x = 0.5 * (m[[li, li]] - m[[lj, lj]]);
        let y = 0.5 * (m[[li, lj]] + m[[lj, li]]);
        sxx += w * x * x;
        syy += w * y * y;
        sxy += w * x * y;
    }
    let theta = 0.25 * (2.0 * sxy).atan2(syy - sxx);
    let alt = wrap_jacobi_angle(theta + std::f64::consts::FRAC_PI_4);
    let theta = wrap_jacobi_angle(theta);
    if pair_energy_after(work, weights, participants, alt)
        < pair_energy_after(work, weights, participants, theta)
    {
        alt
    } else {
        theta
    }
}

fn pair_energy_after(
    work: &[Array2<f64>],
    weights: &[f64],
    participants: &[(usize, usize, usize)],
    theta: f64,
) -> f64 {
    let (s, c) = theta.sin_cos();
    let mut out = 0.0_f64;
    for &(k, li, lj) in participants {
        let m = &work[k];
        let a = m[[li, li]];
        let b = 0.5 * (m[[li, lj]] + m[[lj, li]]);
        let d = m[[lj, lj]];
        let off = 0.5 * (a - d) * (2.0 * s * c) + b * (c * c - s * s);
        out += weights[k] * 2.0 * off * off;
    }
    out
}

/// Applies `theta` to every participating matrix at its own local axis
/// indices, with a line-search backoff (halving) if the combined
/// *weighted* local off-diagonal energy over the participants would
/// increase — mirrors `lie_svd_joint::accept_joint_rotor`'s robustness
/// against the closed-form angle overshooting on a family that doesn't
/// exactly share a spectrum. A non-participating matrix is never touched:
/// no embedding, no risk of perturbing axes it doesn't have.
fn accept_subspace_rotor(
    work: &mut [Array2<f64>],
    local_rotors: &mut [Array2<f64>],
    weights: &[f64],
    participants: &[(usize, usize, usize)],
    theta: f64,
    line_search_steps: usize,
) -> bool {
    let before = weighted_local_offdiag_sq(work, weights, participants);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let angle = theta * scale;
        for &(k, li, lj) in participants {
            apply_symmetric_rotor(&mut work[k], li, lj, angle);
        }
        let after = weighted_local_offdiag_sq(work, weights, participants);
        if after <= before + slack && after.is_finite() {
            for &(k, li, lj) in participants {
                apply_basis_rotor(&mut local_rotors[k], li, lj, angle);
            }
            return true;
        }
        for &(k, li, lj) in participants {
            apply_symmetric_rotor(&mut work[k], li, lj, -angle);
        }
        scale *= 0.5;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_orthogonal(n: usize, rng: &mut StdRng) -> Array2<f64> {
        let mut q = Array2::from_shape_fn((n, n), |_| rng.gen::<f64>() - 0.5);
        for j in 0..n {
            for k in 0..j {
                let mut dot = 0.0_f64;
                for r in 0..n {
                    dot += q[[r, j]] * q[[r, k]];
                }
                for r in 0..n {
                    q[[r, j]] -= dot * q[[r, k]];
                }
            }
            let mut norm = 0.0_f64;
            for r in 0..n {
                norm += q[[r, j]] * q[[r, j]];
            }
            let norm = norm.sqrt().max(1e-300);
            for r in 0..n {
                q[[r, j]] /= norm;
            }
        }
        q
    }

    fn orth_error(r: &Array2<f64>) -> f64 {
        let n = r.nrows();
        let ident = Array2::<f64>::eye(n);
        (&r.t().dot(r) - &ident).mapv(|x| x * x).sum().sqrt()
    }

    /// The exact scenario from the write-up this module implements: a
    /// `3x3` matrix on axes `{0,1,2}` and a `4x4` matrix on axes
    /// `{1,2,3,4}`, sharing axes `{1,2}` -- but with `M1` and `M2` built
    /// from two *independent* random orthogonal matrices, not a shared
    /// structure. First version of this test asserted near-zero final
    /// off-diagonal energy (matching same-size `LieSvdJoint`'s tests) and
    /// failed (`initial=6.2`, `final=1.9`, a `~69%` reduction, not the
    /// `~1e-8` relative drop asserted) -- not a bug, a wrong expectation:
    /// unlike same-size JADE, where a family built as `Q D_k Q^T` for one
    /// shared `Q` always has an exact joint solution, forcing the `(1,2)`
    /// plane to rotate by the *same* angle in both matrices only has an
    /// exact solution when the two matrices' true diagonalizing rotors
    /// happen to agree on their `{1,2}` sub-block -- which two
    /// independently-random rotors generically don't. Measured across 10
    /// seeds, the reduction ratio (`final/initial`) ranges `~0.007` to
    /// `~0.45`; the algorithm is doing a real best-effort compromise, not
    /// silently failing, and it correctly stops improving (the sweep loop's
    /// own `after >= before*(1-1e-10)` check) rather than spinning forever
    /// chasing an unreachable global optimum. See
    /// `subspace_jade_shared_axes_use_information_from_every_participant`
    /// below for a construction that *does* have an exact joint solution
    /// (a genuinely shared sub-block), where convergence to near-zero is
    /// the right expectation and is what's tested.
    ///
    /// What this test actually verifies: substantial (not total) real
    /// off-diagonal reduction on generic data (safely under the worst of
    /// the 10 measured seeds, `~0.45`), that each recovered local rotor is
    /// genuinely orthogonal, and that axis pairs no single matrix jointly
    /// observes (`(0,3)`, `(0,4)`) are never rotated at all -- not
    /// approximately close to identity, exactly identity, since
    /// `subspace_pair_angle` is never even invoked for them.
    #[test]
    fn subspace_jade_reduces_offdiag_energy_on_overlapping_3x3_and_4x4_family() {
        let mut rng = StdRng::seed_from_u64(401);
        let q1 = random_orthogonal(3, &mut rng);
        let d1 = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 5.0, 9.0]));
        let m1 = q1.dot(&d1).dot(&q1.t());

        let q2 = random_orthogonal(4, &mut rng);
        let d2 = Array2::from_diag(&ndarray::Array1::from(vec![1.0, 3.0, 7.0, 11.0]));
        let m2 = q2.dot(&d2).dot(&q2.t());

        let family = vec![
            SubspaceMatrix::new(m1.clone(), vec![0, 1, 2]),
            SubspaceMatrix::new(m2.clone(), vec![1, 2, 3, 4]),
        ];

        let result = LieSvdSubspaceJade::diagonalize(&family);
        assert!(
            result.trace.final_offdiag < result.trace.initial_offdiag * 0.6,
            "initial={:e} final={:e}",
            result.trace.initial_offdiag,
            result.trace.final_offdiag
        );
        assert_eq!(result.trace.global_axes, 5);
        // Pairs (0,3) and (0,4) are never jointly observed by any single
        // matrix -- axis 0 only exists in the 3x3 matrix, axes 3,4 only in
        // the 4x4 one -- so `observed_pairs` must exclude them.
        assert_eq!(
            result.trace.observed_pairs,
            3 + 6 - 1, // 3x3 has 3 internal pairs, 4x4 has 6, pair (1,2) is shared/counted once
            "observed_pairs={}",
            result.trace.observed_pairs
        );

        for r in &result.local_rotors {
            assert!(orth_error(r) < 1e-10, "orth_error={:e}", orth_error(r));
        }

        let components = axis_connected_components(&family);
        assert_eq!(
            components.len(),
            1,
            "all 5 axes are transitively linked through the shared {{1,2}} pair"
        );
    }

    /// Disconnected families (no shared axes at all) must behave as two
    /// completely independent single-matrix diagonalizations: each
    /// component's own axes get diagonalized, and `axis_connected_components`
    /// reports two separate groups.
    #[test]
    fn subspace_jade_keeps_disconnected_axis_groups_independent() {
        let mut rng = StdRng::seed_from_u64(402);
        let q1 = random_orthogonal(3, &mut rng);
        let d1 = Array2::from_diag(&ndarray::Array1::from(vec![1.0, 4.0, 6.0]));
        let m1 = q1.dot(&d1).dot(&q1.t());

        let q2 = random_orthogonal(3, &mut rng);
        let d2 = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 5.0, 8.0]));
        let m2 = q2.dot(&d2).dot(&q2.t());

        let family = vec![
            SubspaceMatrix::new(m1, vec![0, 1, 2]),
            SubspaceMatrix::new(m2, vec![3, 4, 5]),
        ];
        let result = LieSvdSubspaceJade::diagonalize(&family);
        assert!(
            result.trace.final_offdiag < 1e-10,
            "final={:e}",
            result.trace.final_offdiag
        );

        let mut components = axis_connected_components(&family);
        components.sort_by_key(|c| c[0]);
        assert_eq!(components, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    /// The point of joint diagonalization: does coupling through the
    /// shared axes actually recover a shared rotation that a single matrix
    /// alone could not pin down? Construct the shared `{1,2}` block with a
    /// *degenerate* (repeated) eigenvalue in the first matrix alone -- any
    /// rotation within that 2D eigenspace diagonalizes `M1`'s shared block
    /// equally well, so `M1` alone cannot identify the true shared rotor
    /// `Q_sh`. The second matrix's shared block uses a distinct spectrum,
    /// so the *family* pins `Q_sh` down uniquely. Both matrices' recovered
    /// local rotors, restricted to their own local indices for axes
    /// `{1, 2}`, must therefore agree with each other (both recover the
    /// same physical `Q_sh`, up to the axis permutation/sign freedom any
    /// diagonalization has) -- which is exactly what forces this test to
    /// fail if the shared-pair coupling weren't real.
    #[test]
    fn subspace_jade_shared_axes_use_information_from_every_participant() {
        let mut rng = StdRng::seed_from_u64(403);
        let q_sh = random_orthogonal(2, &mut rng); // true shared rotor on axes {1,2}

        // Matrix 1: axes {0,1,2}. Axis 0 is its own independent block
        // (eigenvalue 3.0); the {1,2} block has a DEGENERATE eigenvalue
        // (5.0, 5.0) under q_sh, so M1 alone cannot identify q_sh.
        let d_sh1 = Array2::from_diag(&ndarray::Array1::from(vec![5.0, 5.0]));
        let sh1 = q_sh.dot(&d_sh1).dot(&q_sh.t());
        let mut m1 = Array2::<f64>::zeros((3, 3));
        m1[[0, 0]] = 3.0;
        for a in 0..2 {
            for b in 0..2 {
                m1[[1 + a, 1 + b]] = sh1[[a, b]];
            }
        }

        // Matrix 2: axes {1,2,3}. Its {1,2} block uses the SAME q_sh but a
        // DISTINCT (non-degenerate) spectrum, so M2 alone identifies q_sh
        // uniquely (up to sign/order). Axis 3 is its own independent block.
        let d_sh2 = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 9.0]));
        let sh2 = q_sh.dot(&d_sh2).dot(&q_sh.t());
        let mut m2 = Array2::<f64>::zeros((3, 3));
        for a in 0..2 {
            for b in 0..2 {
                m2[[a, b]] = sh2[[a, b]];
            }
        }
        m2[[2, 2]] = 4.0;

        let family = vec![
            SubspaceMatrix::new(m1, vec![0, 1, 2]),
            SubspaceMatrix::new(m2, vec![1, 2, 3]),
        ];
        let result = LieSvdSubspaceJade::diagonalize(&family);
        assert!(
            result.trace.final_offdiag < 1e-8,
            "final={:e}",
            result.trace.final_offdiag
        );

        // Extract each matrix's recovered rotor restricted to its own
        // local {1,2} block (local indices 1,2 in both matrices here).
        let r1_shared = result.local_rotors[0]
            .slice(ndarray::s![1..3, 1..3])
            .to_owned();
        let r2_shared = result.local_rotors[1]
            .slice(ndarray::s![0..2, 0..2])
            .to_owned();

        // Both should recover q_sh up to a signed permutation of its two
        // columns (the usual diagonalization ambiguity). Check that
        // r1_shared^T r2_shared is itself a signed permutation matrix
        // (close to it): exactly one +-1 entry per row/column, near zero
        // elsewhere -- if the two matrices had recovered *unrelated*
        // rotations of their shared block (which is what would happen
        // without real coupling, e.g. if M1 were diagonalized alone), this
        // product would not collapse to a signed permutation.
        let cross = r1_shared.t().dot(&r2_shared);
        let mut max_off_or_nonunit = 0.0_f64;
        for i in 0..2 {
            let row_abs_max = (0..2).map(|j| cross[[i, j]].abs()).fold(0.0_f64, f64::max);
            max_off_or_nonunit = max_off_or_nonunit.max((row_abs_max - 1.0).abs());
            for j in 0..2 {
                if cross[[i, j]].abs() < row_abs_max - 1e-6 {
                    max_off_or_nonunit = max_off_or_nonunit.max(cross[[i, j]].abs());
                }
            }
        }
        assert!(
            max_off_or_nonunit < 1e-6,
            "cross={cross:?} max_off_or_nonunit={max_off_or_nonunit:e}"
        );
    }

    fn offdiag2(m: &Array2<f64>) -> f64 {
        m[[0, 1]] * m[[0, 1]] + m[[1, 0]] * m[[1, 0]]
    }

    /// The direct A/B `SubspaceWeighting::InverseFrobeniusSquared` was
    /// built to pass: two `2x2` matrices sharing both their axes, one
    /// scaled `~1000x` larger than the other, built from two genuinely
    /// *different* rotations (no exact joint solution exists, so there's a
    /// real trade-off in which angle to pick). Under `Unweighted`, the
    /// large matrix's own entries dominate the shared angle: measured, it
    /// ends up almost fully diagonalized (`big_after ~2.0e-7` from
    /// `~1.37e6` initial) while the small matrix barely improves
    /// (`small_after ~0.380` from `~1.96e0` initial, only a `~19%`
    /// reduction). Under `InverseFrobeniusSquared`, the small matrix's
    /// contribution to the shared angle is upweighted to compensate for
    /// its smaller magnitude, and it improves substantially more
    /// (`small_after ~0.0858`, a `~96%` reduction) -- at a real, honest
    /// cost to the large matrix (`big_after ~1.94e5`, barely reduced at
    /// all). This is the intended trade-off, not a free lunch: weighting
    /// redistributes whose fit the shared angle serves, it doesn't improve
    /// both simultaneously beyond what a single shared angle can do.
    #[test]
    fn subspace_jade_weighting_helps_the_small_magnitude_matrix() {
        let mut rng_a = StdRng::seed_from_u64(501);
        let q_big = random_orthogonal(2, &mut rng_a);
        let d_big = Array2::from_diag(&ndarray::Array1::from(vec![3000.0, 7000.0]));
        let m_big = q_big.dot(&d_big).dot(&q_big.t());

        let mut rng_b = StdRng::seed_from_u64(777);
        let q_small = random_orthogonal(2, &mut rng_b);
        let d_small = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 5.0]));
        let m_small = q_small.dot(&d_small).dot(&q_small.t());

        let initial_small = offdiag2(&m_small);
        let family = vec![
            SubspaceMatrix::new(m_big.clone(), vec![0, 1]),
            SubspaceMatrix::new(m_small.clone(), vec![0, 1]),
        ];

        let unweighted = LieSvdSubspaceJade::diagonalize(&family);
        let unweighted_small_after = offdiag2(&unweighted.diagonalized[1]);

        let params = SubspaceJadeParams {
            weighting: SubspaceWeighting::InverseFrobeniusSquared,
            ..SubspaceJadeParams::for_axes(2)
        };
        let weighted = LieSvdSubspaceJade::diagonalize_with_params(&family, params);
        let weighted_small_after = offdiag2(&weighted.diagonalized[1]);
        let weighted_big_after = offdiag2(&weighted.diagonalized[0]);

        assert!(
            weighted_small_after < 0.5 * unweighted_small_after,
            "weighted should cut the small matrix's residual well below the unweighted run: \
             unweighted={unweighted_small_after:e} weighted={weighted_small_after:e}"
        );
        // Honest trade-off check: the large matrix's fit genuinely gets
        // worse under weighting, not "improves for free".
        assert!(
            weighted_big_after > offdiag2(&unweighted.diagonalized[0]),
            "weighting should trade away some of the large matrix's fit, not improve everything"
        );
        assert_eq!(weighted.weights.len(), 2);
        assert!(
            weighted.weights[0] < weighted.weights[1],
            "the larger matrix must get the smaller weight"
        );
        assert!(initial_small > 0.0);
    }

    /// `stop_reason` should distinguish an exact joint solution (reaches
    /// `tol`) from a genuine best-effort plateau on a family with no exact
    /// solution -- both are legitimate outcomes, but a caller needs to be
    /// able to tell them apart rather than inferring it from raw numbers.
    #[test]
    fn subspace_jade_reports_stop_reason() {
        // Reuses the exact-solution construction from
        // `subspace_jade_shared_axes_use_information_from_every_participant`.
        let mut rng = StdRng::seed_from_u64(403);
        let q_sh = random_orthogonal(2, &mut rng);
        let d_sh1 = Array2::from_diag(&ndarray::Array1::from(vec![5.0, 5.0]));
        let sh1 = q_sh.dot(&d_sh1).dot(&q_sh.t());
        let mut m1 = Array2::<f64>::zeros((3, 3));
        m1[[0, 0]] = 3.0;
        for a in 0..2 {
            for b in 0..2 {
                m1[[1 + a, 1 + b]] = sh1[[a, b]];
            }
        }
        let d_sh2 = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 9.0]));
        let sh2 = q_sh.dot(&d_sh2).dot(&q_sh.t());
        let mut m2 = Array2::<f64>::zeros((3, 3));
        for a in 0..2 {
            for b in 0..2 {
                m2[[a, b]] = sh2[[a, b]];
            }
        }
        m2[[2, 2]] = 4.0;
        let exact_family = vec![
            SubspaceMatrix::new(m1, vec![0, 1, 2]),
            SubspaceMatrix::new(m2, vec![1, 2, 3]),
        ];
        let exact = LieSvdSubspaceJade::diagonalize(&exact_family);
        assert_eq!(
            exact.trace.stop_reason,
            SubspaceJadeStopReason::ReachedTolerance
        );

        // Reuses the no-exact-solution construction from
        // `subspace_jade_reduces_offdiag_energy_on_overlapping_3x3_and_4x4_family`.
        let mut rng2 = StdRng::seed_from_u64(401);
        let q1 = random_orthogonal(3, &mut rng2);
        let d1 = Array2::from_diag(&ndarray::Array1::from(vec![2.0, 5.0, 9.0]));
        let m1b = q1.dot(&d1).dot(&q1.t());
        let q2 = random_orthogonal(4, &mut rng2);
        let d2 = Array2::from_diag(&ndarray::Array1::from(vec![1.0, 3.0, 7.0, 11.0]));
        let m2b = q2.dot(&d2).dot(&q2.t());
        let inexact_family = vec![
            SubspaceMatrix::new(m1b, vec![0, 1, 2]),
            SubspaceMatrix::new(m2b, vec![1, 2, 3, 4]),
        ];
        let inexact = LieSvdSubspaceJade::diagonalize(&inexact_family);
        assert_eq!(inexact.trace.stop_reason, SubspaceJadeStopReason::Plateaued);
    }

    /// Ties `0.31.0`'s MZI export and `0.32.0`'s subspace-JADE together
    /// end to end: a recovered local rotor is a normal orthogonal matrix,
    /// so it compiles straight to an MZI hardware schedule via the same
    /// `from_orthogonal_matrix` path already used for
    /// `lie_tbl_regress::procrustes_rotor`, with no glue code needed.
    #[test]
    fn subspace_jade_local_rotor_compiles_to_an_mzi_schedule() {
        let mut rng = StdRng::seed_from_u64(404);
        let q1 = random_orthogonal(3, &mut rng);
        let d1 = Array2::from_diag(&ndarray::Array1::from(vec![1.0, 4.0, 9.0]));
        let m1 = q1.dot(&d1).dot(&q1.t());
        let family = vec![SubspaceMatrix::new(m1, vec![0, 1, 2])];

        let result = LieSvdSubspaceJade::diagonalize(&family);
        let schedule = crate::lie_svd_compiler::HardwareSchedule::from_orthogonal_matrix(
            &result.local_rotors[0],
            crate::lie_svd_compiler::HardwareTarget::MziMesh,
        );
        assert_eq!(schedule.channels, 3);
        assert!(schedule.total_events() <= 3);
        assert!(schedule.to_json_string().contains("givens_elimination"));
    }
}
