//! Active phase-health SVD preconditioner.
//!
//! `lie_svd_phasehealth` is the thermometer. This module is the first actuator
//! and a first-class solver route: it uses row/column phase-health to choose
//! where to apply real two-sided rotors until the phase field locks onto a
//! nearly diagonal core. The model is still plain `f64`; "phase jumps" are
//! compiled to ordinary orthogonal rotations.
//!
//! A separate `solve_with_digital_polish` path exists for final audit-quality
//! cleanup, but `solve` itself returns the phase-locked result rather than
//! delegating the actual SVD to `LieSvdSmall`.

use ndarray::{Array1, Array2};

const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_5;
const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

#[derive(Clone, Copy, Debug)]
pub struct LieSvdPhaseFlowParams {
    /// Number of active phase passes before digital polish.
    pub max_passes: usize,
    /// Maximum phase-jump rotor angle in radians.
    pub max_jump_angle: f64,
    /// Maximum unwrap rotor angle in radians.
    pub max_unwrap_angle: f64,
    /// Backtracking attempts for a proposed phase rotor.
    pub line_search_steps: usize,
    /// Skip pairs below this off-diagonal scale relative to `||A||_F`.
    pub min_pair_scale: f64,
    /// Number of high-stress axes to seed pair selection from.
    pub active_axes: usize,
    /// Above this dimension, skip full sweeps and keep only active-set rotors.
    pub active_set_only_above: usize,
    /// Relative resonance threshold for the phase-stress field.
    pub phase_resonance_tol: f64,
    /// Keep the applied rotor sequence for photonic/MZI export.
    pub record_mzi_phases: bool,
    /// Apply a guarded Layer-0 golden-angle rotor sheet before local PhaseFlow.
    pub use_golden_prespin: bool,
    /// Number of conflict-free golden pre-spin sheets.
    pub golden_prespin_layers: usize,
    /// Maximum Layer-0 pre-spin rotor angle in radians.
    pub max_prespin_angle: f64,
    /// Number of golden/causal harmonic pre-spin cascades.
    pub prespin_depth: usize,
    /// Let `PhaseFlow` raise pre-spin depth on high-stress or causal inputs.
    pub adaptive_prespin_depth: bool,
    /// Apply a guarded four-act row/column golden-antipode pre-spin cycle.
    pub use_yinyang_prespin: bool,
    /// Number of four-act Yin-Yang pre-spin cycles.
    pub yinyang_cycles: usize,
    /// Maximum angle for each Yin-Yang cycle rotor before golden annealing.
    pub max_yinyang_angle: f64,
    /// Apply state-mirrored Layer-0 rotors from the current row/column phases.
    pub use_phase_conjugate_autospin: bool,
    /// Maximum phase-conjugate rotor angle in radians.
    pub max_phase_conjugate_angle: f64,
    /// Use maximum-energy pair selection before the ordinary active-set pass.
    pub use_bottleneck_queue: bool,
    /// Keep bottleneck pair scores in a lazy max-heap and update touched axes.
    pub use_incremental_bottleneck_cache: bool,
    /// Number of top bottleneck pairs to try per pass before active sweeps.
    pub bottleneck_pairs: usize,
    /// Damping applied to exact local phase-conjugate and bottleneck rotors.
    pub phase_viscosity: f64,
    /// Optional hardware phase quantization levels. `0` keeps continuous angles.
    pub phase_quantization_levels: usize,
    /// Apply an asymmetric Layer-0 counter-flow for Jordan/causal torsion.
    pub use_causal_antispin: bool,
    /// Minimum triangular causal bias needed to replace golden pre-spin.
    pub causal_antispin_threshold: f64,
    /// Number of conflict-free causal anti-spin sheets.
    pub causal_antispin_layers: usize,
    /// Maximum causal anti-spin rotor angle in radians.
    pub max_causal_antispin_angle: f64,
    /// Modulate global phase jumps by an irrational golden-angle lattice.
    pub use_golden_jumps: bool,
    /// Try a local `4x4` block surgery when pairwise relaxation hits a plateau.
    pub enable_flow_surgery: bool,
    /// Strong-Rules-style relative screening: an axis is dropped from pair
    /// search when `row_norm + col_norm <= active_set_alpha * max(row_norm +
    /// col_norm)`, in addition to the exact `pair_tol` certificate. `0.0`
    /// (default) disables this and keeps only the exact bound. This is a
    /// heuristic, not a certificate: a legitimate pair can in principle be
    /// screened out early, so it trades a small, empirically-checked risk of
    /// extra passes for fewer candidate pairs per pass on large/structured
    /// inputs.
    pub active_set_alpha: f64,
    /// Replace the fixed `phase_viscosity` damping with a per-pair adaptive
    /// energy-ratio gain `gamma = P / (P + R)` for bottleneck rotors, where
    /// `P` is the candidate pair's own energy and `R` is the current pass's
    /// mean row/column stress (the ambient phase-field background). Loud
    /// pairs (`P >> R`) get trusted close to fully; pairs near the ambient
    /// noise floor (`P ~ R`) get damped toward half strength. This is a
    /// deliberately literal name: it is a normalized signal/background ratio,
    /// not a Kalman filter (there is no state covariance propagated across
    /// passes). Default `false` keeps the fixed `phase_viscosity` behavior.
    pub use_adaptive_viscosity: bool,
    /// Bounds the "blind window" of `BottleneckPairCache`'s lazy invalidation
    /// (0.29.0): every `bottleneck_cache_refresh_period` passes, do a full
    /// `rebuild` instead of a lazy touch-flush. Lazy invalidation only
    /// re-verifies pairs already in the heap at the last rebuild — it cannot
    /// discover a new hot pair between two axes that were both cold then.
    /// Measured on a real trace, pure-lazy-forever (`0`, i.e. only the one
    /// rebuild at cache creation) cut `BottleneckPairCache` rescoring ~294x
    /// but visibly hurt raw (pre-digital-polish) convergence quality on
    /// `uniform_random`/`degenerate_spectrum` at `N=300` (roughly 2-3x worse
    /// offdiag reduction for the same pass budget) — periodic rebuilds trade
    /// back some of that speedup to recover discovery.
    pub bottleneck_cache_refresh_period: usize,
}

impl LieSvdPhaseFlowParams {
    pub fn for_n(n: usize) -> Self {
        let active_axes = if n >= 256 {
            ((n as f64).sqrt() as usize * 4).clamp(32, 128)
        } else {
            n.max(1)
        };
        Self {
            max_passes: 24 + 2 * n.max(1),
            max_jump_angle: 0.08,
            max_unwrap_angle: 1.20,
            line_search_steps: 7,
            min_pair_scale: 1e-14,
            active_axes,
            active_set_only_above: 256,
            phase_resonance_tol: 1e-10,
            record_mzi_phases: false,
            use_golden_prespin: true,
            golden_prespin_layers: 2,
            max_prespin_angle: 0.06,
            prespin_depth: 1,
            adaptive_prespin_depth: true,
            use_yinyang_prespin: false,
            yinyang_cycles: 0,
            max_yinyang_angle: 0.08,
            use_phase_conjugate_autospin: false,
            max_phase_conjugate_angle: 0.16,
            use_bottleneck_queue: false,
            use_incremental_bottleneck_cache: true,
            bottleneck_pairs: active_axes.max(1),
            phase_viscosity: 0.85,
            phase_quantization_levels: 0,
            use_causal_antispin: true,
            causal_antispin_threshold: 0.72,
            causal_antispin_layers: 2,
            max_causal_antispin_angle: 0.10,
            use_golden_jumps: true,
            enable_flow_surgery: true,
            active_set_alpha: 0.0,
            use_adaptive_viscosity: false,
            bottleneck_cache_refresh_period: 16,
        }
    }
}

impl Default for LieSvdPhaseFlowParams {
    fn default() -> Self {
        Self::for_n(64)
    }
}

#[derive(Clone, Debug)]
pub struct LieSvdPhaseFlowTrace {
    pub initial_offdiag: f64,
    pub final_offdiag: f64,
    pub initial_phase_stress: f64,
    pub final_phase_stress: f64,
    pub passes: usize,
    pub phase_jumps: usize,
    pub golden_prespins: usize,
    pub causal_antispins: usize,
    pub yinyang_prespins: usize,
    pub phase_conjugate_prespins: usize,
    pub prespin_depth: usize,
    pub yinyang_cycles: usize,
    pub bottleneck_rotations: usize,
    pub bottleneck_cache_updates: usize,
    pub bottleneck_cache_refreshes: usize,
    pub unwrap_rotations: usize,
    pub rejected_rotations: usize,
    pub surgery_blocks: usize,
    pub samples: Vec<(f64, f64)>,
    pub mzi_phases: Vec<MziPhase>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhaseRotorKind {
    GoldenPreSpin,
    CausalAntiSpin,
    CrossPhaseYinYang,
    PhaseConjugate,
    Bottleneck,
    PhaseJump,
    Unwrap,
    Directional,
    Surgery,
}

#[derive(Clone, Copy, Debug)]
pub struct MziPhase {
    pub pass: usize,
    pub i: usize,
    pub j: usize,
    pub theta_l: f64,
    pub theta_r: f64,
    pub kind: PhaseRotorKind,
}

#[derive(Clone, Copy, Debug)]
struct AxisPhase {
    stress: f64,
    entropy: f64,
    phase: f64,
    /// L2 norm of this row/column. A trivial but exact per-entry bound:
    /// no element of the axis can exceed this in absolute value, which is
    /// what `hot_axes` uses to prune pair search without scanning `core`.
    norm: f64,
}

#[derive(Clone, Copy, Debug)]
struct CandidatePair {
    i: usize,
    j: usize,
    score: f64,
}

#[derive(Clone, Debug)]
struct PairEnergyCache {
    n: usize,
    pairs: Vec<CandidatePair>,
}

impl PairEnergyCache {
    fn square(
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
        pair_tol: f64,
        active_set_alpha: f64,
    ) -> Self {
        let n = core.nrows().min(core.ncols());
        let hot = hot_axes(rows, cols, n, pair_tol, active_set_alpha);
        let mut pairs =
            Vec::with_capacity(hot.len().saturating_mul(hot.len().saturating_sub(1)) / 2);
        for (a, &i) in hot.iter().enumerate() {
            for &j in &hot[a + 1..] {
                let entropy_gap = (rows[i].entropy - rows[j].entropy).abs()
                    + (cols[i].entropy - cols[j].entropy).abs();
                let score = pair_offdiag(core, i, j)
                    + 0.05 * (rows[i].stress + rows[j].stress + cols[i].stress + cols[j].stress)
                    + 0.01 * entropy_gap;
                pairs.push(CandidatePair { i, j, score });
            }
        }
        pairs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { n, pairs }
    }

    fn active_conflict_free(&self, active_axes: usize) -> Vec<CandidatePair> {
        let mut used = vec![false; self.n];
        let mut out = Vec::new();
        let max_pairs = active_axes.min(self.n) / 2 + active_axes.min(self.n) % 2;
        for &pair in &self.pairs {
            if !used[pair.i] && !used[pair.j] {
                used[pair.i] = true;
                used[pair.j] = true;
                out.push(pair);
                if out.len() >= max_pairs.max(1) {
                    break;
                }
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct BottleneckPairCache {
    n: usize,
    scores: Vec<f64>,
    positions: Vec<usize>,
    heap: Vec<usize>,
    /// Number of times a pair's score was actually recomputed. Under lazy
    /// invalidation this only happens at pop time for stale entries, so it
    /// is much smaller than the old eager scheme's count of every
    /// (touched_axis, other) combination.
    updates: usize,
    refreshes: usize,
    /// Per-axis touch counter. Bumping this is the entire cost of
    /// `update_axes` now: O(1) per touched axis, no O(n) rescoring loop.
    axis_gen: Vec<u64>,
    /// Per-pair "last verified at generation" stamp, indexed like `scores`.
    /// A pair is stale iff `pair_gen[idx] < max(axis_gen[i], axis_gen[j])`.
    pair_gen: Vec<u64>,
}

impl BottleneckPairCache {
    const NOT_IN_HEAP: usize = usize::MAX;

    fn new(
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
        n: usize,
        pair_tol: f64,
        active_set_alpha: f64,
    ) -> Self {
        let n = n.min(core.nrows()).min(core.ncols());
        let mut cache = Self {
            n,
            scores: vec![0.0; n.saturating_mul(n)],
            positions: vec![Self::NOT_IN_HEAP; n.saturating_mul(n)],
            heap: Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2),
            updates: 0,
            refreshes: 0,
            axis_gen: vec![0; n],
            pair_gen: vec![0; n.saturating_mul(n)],
        };
        cache.rebuild(core, rows, cols, pair_tol, active_set_alpha);
        cache
    }

    fn rebuild(
        &mut self,
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
        pair_tol: f64,
        active_set_alpha: f64,
    ) {
        self.heap.clear();
        let cap = self.n.saturating_mul(self.n);
        if self.scores.len() != cap {
            self.scores.resize(cap, 0.0);
            self.positions.resize(cap, Self::NOT_IN_HEAP);
            self.pair_gen.resize(cap, 0);
            self.axis_gen.resize(self.n, 0);
        } else {
            self.scores.fill(0.0);
            self.positions.fill(Self::NOT_IN_HEAP);
            self.pair_gen.fill(0);
        }
        self.axis_gen.fill(0);
        // Only pairs with two "hot" axes can score above pair_tol (see
        // `hot_axes`), so skip cold axes entirely instead of touching every
        // one of the n*(n-1)/2 cells and letting the heap discover it later.
        let hot = hot_axes(rows, cols, self.n, pair_tol, active_set_alpha);
        for (a, &i) in hot.iter().enumerate() {
            for &j in &hot[a + 1..] {
                let idx = self.index(i, j);
                self.scores[idx] = self.score_pair(core, rows, cols, i, j);
                self.pair_gen[idx] = 0;
                self.push_idx(idx);
            }
        }
        self.refreshes += 1;
    }

    /// O(1) per touched axis: a binary heap can't do a cheap `decrease-key`,
    /// so instead of eagerly rescoring every (touched_axis, other) pair here
    /// (the old behavior, O(touched * n) with a heap sift for each), this
    /// just bumps a per-axis generation counter. Staleness is resolved lazily
    /// in `pop_verified_root`, which only recomputes a score when that exact
    /// pair is actually about to be returned. On a real `N=300` trace this
    /// cut `cache_updates` (now: actual recomputations) from tens of millions
    /// to a small multiple of the pairs actually requested per pass.
    ///
    /// Tradeoff, stated plainly: this cannot *discover* a pair that wasn't in
    /// the heap at the last `rebuild` (a pair between two axes that were both
    /// cold then). The old eager scheme could, via `update_pair`'s
    /// insert-if-absent. In practice this matters little: rotors only
    /// redistribute energy among axes already in play, they don't manufacture
    /// mass on a pair of axes that were both quiet — but it is a real,
    /// deliberate scope reduction, not a proven-equivalent optimization.
    fn update_axes(
        &mut self,
        _core: &Array2<f64>,
        _rows: &[AxisPhase],
        _cols: &[AxisPhase],
        axes: &[usize],
    ) -> usize {
        let mut touched = 0usize;
        for &axis in axes {
            if axis < self.n {
                self.axis_gen[axis] += 1;
                touched += 1;
            }
        }
        touched
    }

    /// Pop the heap root, verifying and refreshing it if it's stale (see
    /// `update_axes`) before trusting it as the true current maximum. Loops
    /// internally: a corrected entry may still be the max (re-verified and
    /// returned) or may drop below another entry (re-pushed, loop continues).
    /// Terminates because each stale pair, once verified, cannot go stale
    /// again until one of its axes is touched further.
    fn pop_verified_root(
        &mut self,
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
    ) -> Option<usize> {
        let max_attempts = self.heap.len().saturating_add(1);
        for _ in 0..max_attempts {
            let idx = self.pop_root_idx()?;
            let (i, j) = self.axes_from_index(idx);
            if i >= self.n || j >= self.n || i == j {
                continue;
            }
            let required = self.axis_gen[i].max(self.axis_gen[j]);
            if self.pair_gen[idx] >= required {
                return Some(idx);
            }
            self.scores[idx] = self.score_pair(core, rows, cols, i, j);
            self.pair_gen[idx] = required;
            self.updates += 1;
            self.push_idx(idx);
        }
        None
    }

    fn pop_conflict_free(
        &mut self,
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
        max_pairs: usize,
    ) -> Vec<CandidatePair> {
        let mut used = vec![false; self.n];
        let mut out = Vec::new();
        let mut deferred = Vec::new();
        let max_pairs = max_pairs.max(1);
        let max_attempts = self.heap.len().min(self.n.saturating_mul(self.n));
        let mut attempts = 0usize;
        while out.len() < max_pairs && attempts < max_attempts {
            attempts += 1;
            let Some(idx) = self.pop_verified_root(core, rows, cols) else {
                break;
            };
            let (i, j) = self.axes_from_index(idx);
            let score = self.scores[idx];
            if i >= self.n || j >= self.n || i == j {
                continue;
            }
            if score <= 0.0 {
                deferred.push(idx);
                break;
            }
            if used[i] || used[j] {
                deferred.push(idx);
                continue;
            }
            used[i] = true;
            used[j] = true;
            deferred.push(idx);
            out.push(CandidatePair { i, j, score });
        }
        for idx in deferred {
            self.push_idx(idx);
        }
        out
    }

    fn push_idx(&mut self, idx: usize) {
        if self.positions[idx] != Self::NOT_IN_HEAP {
            return;
        }
        self.positions[idx] = self.heap.len();
        self.heap.push(idx);
        self.sift_up(self.heap.len() - 1);
    }

    fn pop_root_idx(&mut self) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let root = self.heap[0];
        self.positions[root] = Self::NOT_IN_HEAP;
        let last = self.heap.pop().expect("heap root exists");
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.positions[last] = 0;
            self.sift_down(0);
        }
        Some(root)
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.score_cmp(self.heap[pos], self.heap[parent]) <= 0 {
                break;
            }
            self.swap_heap(pos, parent);
            pos = parent;
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        loop {
            let left = 2 * pos + 1;
            let right = left + 1;
            let mut best = pos;
            if left < self.heap.len() && self.score_cmp(self.heap[left], self.heap[best]) > 0 {
                best = left;
            }
            if right < self.heap.len() && self.score_cmp(self.heap[right], self.heap[best]) > 0 {
                best = right;
            }
            if best == pos {
                break;
            }
            self.swap_heap(pos, best);
            pos = best;
        }
    }

    fn swap_heap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a]] = a;
        self.positions[self.heap[b]] = b;
    }

    fn score_cmp(&self, a: usize, b: usize) -> i8 {
        match self.scores[a]
            .partial_cmp(&self.scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => {
                if a > b {
                    1
                } else if a < b {
                    -1
                } else {
                    0
                }
            }
        }
    }

    fn score_pair(
        &self,
        core: &Array2<f64>,
        rows: &[AxisPhase],
        cols: &[AxisPhase],
        i: usize,
        j: usize,
    ) -> f64 {
        let pair_energy = pair_offdiag(core, i, j);
        let phase_gap = rows
            .get(i)
            .zip(rows.get(j))
            .map(|(a, b)| (a.phase - b.phase).abs())
            .unwrap_or(0.0)
            + cols
                .get(i)
                .zip(cols.get(j))
                .map(|(a, b)| (a.phase - b.phase).abs())
                .unwrap_or(0.0);
        let stress = rows.get(i).map(|x| x.stress).unwrap_or(0.0)
            + rows.get(j).map(|x| x.stress).unwrap_or(0.0)
            + cols.get(i).map(|x| x.stress).unwrap_or(0.0)
            + cols.get(j).map(|x| x.stress).unwrap_or(0.0);
        let score = pair_energy * (1.0 + 0.01 * phase_gap) + 1e-12 * stress;
        if score.is_finite() {
            score
        } else {
            0.0
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < j);
        i * self.n + j
    }

    fn axes_from_index(&self, idx: usize) -> (usize, usize) {
        (idx / self.n, idx % self.n)
    }
}

fn mark_touched_axes(touched: &mut [bool], axes: &[usize]) {
    for &axis in axes {
        if let Some(slot) = touched.get_mut(axis) {
            *slot = true;
        }
    }
}

fn flush_bottleneck_cache_updates(
    cache: Option<&mut BottleneckPairCache>,
    core: &Array2<f64>,
    rows: &[AxisPhase],
    cols: &[AxisPhase],
    touched: &mut [bool],
) {
    let Some(cache) = cache else {
        touched.fill(false);
        return;
    };
    if touched.iter().all(|&is_touched| !is_touched) {
        return;
    }
    let axes: Vec<usize> = touched
        .iter()
        .enumerate()
        .filter_map(|(axis, &is_touched)| is_touched.then_some(axis))
        .collect();
    cache.update_axes(core, rows, cols, &axes);
    touched.fill(false);
}

pub struct LieSvdPhaseFlow;

impl LieSvdPhaseFlow {
    pub fn solve(mat: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let params = LieSvdPhaseFlowParams::for_n(mat.nrows());
        Self::solve_with_trace(mat, params).0
    }

    // Allow: the return type mirrors this crate's established (U, Sigma, Vt[, Trace]) tuple convention; a type alias would obscure the shape at call sites during this stability freeze.
    #[allow(clippy::type_complexity)]
    pub fn solve_with_trace(
        mat: &Array2<f64>,
        params: LieSvdPhaseFlowParams,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdPhaseFlowTrace,
    ) {
        Self::phase_lock_with_trace(mat, params)
    }

    // Allow: the return type mirrors this crate's established (U, Sigma, Vt[, Trace]) tuple convention; a type alias would obscure the shape at call sites during this stability freeze.
    #[allow(clippy::type_complexity)]
    pub fn solve_with_digital_polish(
        mat: &Array2<f64>,
        params: LieSvdPhaseFlowParams,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdPhaseFlowTrace,
    ) {
        let ((u0, _sigma0, vt0), trace) = Self::phase_lock_with_trace(mat, params);
        let v0 = vt0.t().to_owned();
        let core = u0.t().dot(mat).dot(&v0);
        let (ur, sigma, vrt) = crate::lie_svd_small::LieSvdSmall::solve(&core);
        let u = u0.dot(&ur);
        let vt = vrt.dot(&vt0);
        ((u, sigma, vt), trace)
    }

    pub fn to_mzi_phases(mat: &Array2<f64>, mut params: LieSvdPhaseFlowParams) -> Vec<MziPhase> {
        params.record_mzi_phases = true;
        Self::phase_lock_with_trace(mat, params).1.mzi_phases
    }

    // Allow: the return type mirrors this crate's established (U, Sigma, Vt[, Trace]) tuple convention; a type alias would obscure the shape at call sites during this stability freeze.
    #[allow(clippy::type_complexity)]
    pub fn phase_lock_with_trace(
        mat: &Array2<f64>,
        params: LieSvdPhaseFlowParams,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdPhaseFlowTrace,
    ) {
        let n = mat.nrows();
        assert_eq!(n, mat.ncols(), "LieSvdPhaseFlow: matrix must be square");
        if n <= 4 {
            let (u, sigma, vt) = crate::lie_svd_micro::LieSvdMicro::solve(mat);
            let core = u.t().dot(mat).dot(&vt.t());
            let offdiag = offdiag_norm(&core);
            let stress = phase_stress(&core);
            return (
                (u, sigma, vt),
                LieSvdPhaseFlowTrace {
                    initial_offdiag: offdiag,
                    final_offdiag: offdiag,
                    initial_phase_stress: stress,
                    final_phase_stress: stress,
                    passes: 0,
                    phase_jumps: 0,
                    unwrap_rotations: 0,
                    golden_prespins: 0,
                    causal_antispins: 0,
                    yinyang_prespins: 0,
                    phase_conjugate_prespins: 0,
                    prespin_depth: 0,
                    yinyang_cycles: 0,
                    bottleneck_rotations: 0,
                    bottleneck_cache_updates: 0,
                    bottleneck_cache_refreshes: 0,
                    rejected_rotations: 0,
                    surgery_blocks: 0,
                    samples: Vec::new(),
                    mzi_phases: Vec::new(),
                },
            );
        }

        let mut core = mat.clone();
        let mut u_basis = Array2::<f64>::eye(n);
        let mut v_basis = Array2::<f64>::eye(n);
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let initial_offdiag = offdiag_norm(&core);
        let initial_phase_stress = phase_stress(&core);
        let resonance_tol = params.phase_resonance_tol * initial_phase_stress.max(1.0);
        let mut samples = Vec::new();
        let mut mzi_phases = Vec::new();
        let mut golden_prespins = 0usize;
        let mut causal_antispins = 0usize;
        let mut yinyang_prespins = 0usize;
        let mut phase_conjugate_prespins = 0usize;
        let mut bottleneck_rotations = 0usize;
        let mut phase_jumps = 0usize;
        let mut unwrap_rotations = 0usize;
        let mut rejected_rotations = 0usize;
        let mut surgery_blocks = 0usize;
        let mut passes = 0usize;
        let causal_bias = triangular_causal_bias(&core);
        let prespin_depth =
            effective_prespin_depth(&params, n, causal_bias, initial_phase_stress, ref_norm);
        let yinyang_cycles =
            effective_yinyang_cycles(&params, n, causal_bias, initial_phase_stress, ref_norm);
        let directional_causal = causal_bias.abs() > 0.72;
        let layer0_causal =
            params.use_causal_antispin && causal_bias.abs() >= params.causal_antispin_threshold;
        if params.use_phase_conjugate_autospin {
            let (accepted, rejected) = apply_phase_conjugate_autospin_square(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                &params,
                &mut mzi_phases,
                pair_tol,
            );
            phase_conjugate_prespins += accepted;
            rejected_rotations += rejected;
        }
        if params.use_yinyang_prespin && yinyang_cycles > 0 {
            let (accepted, rejected) = apply_yinyang_prespin_square(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                yinyang_cycles,
                &params,
                &mut mzi_phases,
            );
            yinyang_prespins += accepted;
            rejected_rotations += rejected;
        } else if layer0_causal {
            let (accepted, rejected) = apply_causal_antispin_square(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                causal_bias,
                prespin_depth,
                &params,
                &mut mzi_phases,
            );
            causal_antispins += accepted;
            rejected_rotations += rejected;
        } else if params.use_golden_prespin {
            let (accepted, rejected) = apply_golden_prespin_square(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                prespin_depth,
                &params,
                &mut mzi_phases,
            );
            golden_prespins += accepted;
            rejected_rotations += rejected;
        }

        let mut bottleneck_cache =
            if params.use_bottleneck_queue && params.use_incremental_bottleneck_cache {
                let rows = row_phases(&core);
                let cols = col_phases(&core);
                Some(BottleneckPairCache::new(
                    &core,
                    &rows,
                    &cols,
                    n,
                    pair_tol,
                    params.active_set_alpha,
                ))
            } else {
                None
            };
        let mut bottleneck_touched = vec![false; n];
        // Reused across every pass instead of allocating a fresh Vec<AxisPhase>
        // each time `row_phases`/`col_phases` would otherwise be called; each
        // pass recomputes this buffer's contents twice (top of pass, and
        // again after Layer-0 phase-jump layers mutate `core`).
        let mut row_phase_buf: Vec<AxisPhase> = Vec::with_capacity(n);
        let mut col_phase_buf: Vec<AxisPhase> = Vec::with_capacity(n);

        for pass in 0..params.max_passes {
            passes = pass + 1;
            let before_pass = offdiag_norm(&core);
            row_phases_into(&core, &mut row_phase_buf);
            col_phases_into(&core, &mut col_phase_buf);
            let row_phase = &row_phase_buf;
            let col_phase = &col_phase_buf;
            let before_stress = summarize_stress(row_phase, col_phase);
            samples.push((before_pass, before_stress));
            if before_stress <= resonance_tol {
                break;
            }

            if directional_causal {
                for layer in 0..2 {
                    let mut i = layer;
                    while i + 1 < n {
                        let j = i + 1;
                        if pair_offdiag(&core, i, j) <= pair_tol {
                            i += 2;
                            continue;
                        }
                        let (theta_l, theta_r) = directional_causal_rotor(
                            &core,
                            i,
                            j,
                            causal_bias,
                            params.max_unwrap_angle,
                        );
                        if theta_l.abs() + theta_r.abs() > 1e-18 {
                            if accept_offdiag_rotor(
                                &mut core,
                                &mut u_basis,
                                &mut v_basis,
                                i,
                                j,
                                theta_l,
                                theta_r,
                                params.line_search_steps,
                            ) {
                                unwrap_rotations += 1;
                                mark_touched_axes(&mut bottleneck_touched, &[i, j]);
                                if params.record_mzi_phases {
                                    mzi_phases.push(MziPhase {
                                        pass,
                                        i,
                                        j,
                                        theta_l,
                                        theta_r,
                                        kind: PhaseRotorKind::Directional,
                                    });
                                }
                            } else {
                                rejected_rotations += 1;
                            }
                        }
                        i += 2;
                    }
                }
            }

            for layer in 0..2 {
                let mut i = layer;
                while i + 1 < n {
                    let j = i + 1;
                    let theta_l = clamp_angle(
                        golden_phase_jump(
                            -0.5 * (row_phase[j].phase - row_phase[i].phase),
                            pass,
                            i,
                            j,
                            params.use_golden_jumps,
                        ),
                        params.max_jump_angle,
                    );
                    let theta_r = clamp_angle(
                        golden_phase_jump(
                            -0.5 * (col_phase[j].phase - col_phase[i].phase),
                            pass,
                            j,
                            i,
                            params.use_golden_jumps,
                        ),
                        params.max_jump_angle,
                    );
                    if theta_l.abs() + theta_r.abs() > 1e-18 {
                        if accept_offdiag_rotor(
                            &mut core,
                            &mut u_basis,
                            &mut v_basis,
                            i,
                            j,
                            theta_l,
                            theta_r,
                            params.line_search_steps,
                        ) {
                            phase_jumps += 1;
                            mark_touched_axes(&mut bottleneck_touched, &[i, j]);
                            if params.record_mzi_phases {
                                mzi_phases.push(MziPhase {
                                    pass,
                                    i,
                                    j,
                                    theta_l,
                                    theta_r,
                                    kind: PhaseRotorKind::PhaseJump,
                                });
                            }
                        } else {
                            rejected_rotations += 1;
                        }
                    }
                    i += 2;
                }
            }

            row_phases_into(&core, &mut row_phase_buf);
            col_phases_into(&core, &mut col_phase_buf);
            let row_phase = &row_phase_buf;
            let col_phase = &col_phase_buf;
            if params.use_bottleneck_queue {
                let due_for_rebuild = params.bottleneck_cache_refresh_period > 0
                    && pass > 0
                    && pass % params.bottleneck_cache_refresh_period == 0;
                if due_for_rebuild {
                    if let Some(cache) = bottleneck_cache.as_mut() {
                        cache.rebuild(
                            &core,
                            row_phase,
                            col_phase,
                            pair_tol,
                            params.active_set_alpha,
                        );
                    }
                    bottleneck_touched.fill(false);
                } else {
                    flush_bottleneck_cache_updates(
                        bottleneck_cache.as_mut(),
                        &core,
                        row_phase,
                        col_phase,
                        &mut bottleneck_touched,
                    );
                }
                let candidates = if let Some(cache) = bottleneck_cache.as_mut() {
                    cache.pop_conflict_free(&core, row_phase, col_phase, params.bottleneck_pairs)
                } else {
                    bottleneck_pairs(
                        &core,
                        row_phase,
                        col_phase,
                        params.bottleneck_pairs,
                        pair_tol,
                        params.active_set_alpha,
                    )
                };
                let background_energy = mean_axis_stress(row_phase, col_phase);
                for pair in candidates {
                    if pair_offdiag(&core, pair.i, pair.j) <= pair_tol {
                        continue;
                    }
                    let viscosity = if params.use_adaptive_viscosity {
                        adaptive_energy_ratio_viscosity(pair.score, background_energy)
                    } else {
                        params.phase_viscosity
                    };
                    let (theta_l, theta_r) = local_pair_svd_angles(&core, pair.i, pair.j);
                    let theta_l = prepare_phase_angle(
                        theta_l * viscosity,
                        params.max_unwrap_angle,
                        params.phase_quantization_levels,
                    );
                    let theta_r = prepare_phase_angle(
                        theta_r * viscosity,
                        params.max_unwrap_angle,
                        params.phase_quantization_levels,
                    );
                    if theta_l.abs() + theta_r.abs() <= 1e-18 {
                        continue;
                    }
                    if accept_offdiag_rotor(
                        &mut core,
                        &mut u_basis,
                        &mut v_basis,
                        pair.i,
                        pair.j,
                        theta_l,
                        theta_r,
                        params.line_search_steps,
                    ) {
                        bottleneck_rotations += 1;
                        mark_touched_axes(&mut bottleneck_touched, &[pair.i, pair.j]);
                        if params.record_mzi_phases {
                            mzi_phases.push(MziPhase {
                                pass,
                                i: pair.i,
                                j: pair.j,
                                theta_l,
                                theta_r,
                                kind: PhaseRotorKind::Bottleneck,
                            });
                        }
                    } else {
                        rejected_rotations += 1;
                    }
                }
            }
            let candidates = active_phase_pairs(
                &core,
                row_phase,
                col_phase,
                params.active_axes,
                pair_tol,
                params.active_set_alpha,
            );
            for pair in candidates {
                if pair_offdiag(&core, pair.i, pair.j) <= pair_tol {
                    continue;
                }
                let (theta_l, theta_r) = local_pair_svd_angles(&core, pair.i, pair.j);
                let stress_gain = (pair.score / ref_norm.max(1.0)).tanh();
                let theta_l = clamp_angle(
                    theta_l * (1.0 + 0.25 * stress_gain),
                    params.max_unwrap_angle,
                );
                let theta_r = clamp_angle(
                    theta_r * (1.0 + 0.25 * stress_gain),
                    params.max_unwrap_angle,
                );
                if theta_l.abs() + theta_r.abs() <= 1e-18 {
                    continue;
                }
                if accept_offdiag_rotor(
                    &mut core,
                    &mut u_basis,
                    &mut v_basis,
                    pair.i,
                    pair.j,
                    theta_l,
                    theta_r,
                    params.line_search_steps,
                ) {
                    unwrap_rotations += 1;
                    mark_touched_axes(&mut bottleneck_touched, &[pair.i, pair.j]);
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i: pair.i,
                            j: pair.j,
                            theta_l,
                            theta_r,
                            kind: PhaseRotorKind::Unwrap,
                        });
                    }
                } else {
                    rejected_rotations += 1;
                }
            }

            if n < params.active_set_only_above {
                for layer in 0..round_robin_layer_count(n) {
                    for (i, j) in layer_pairs(n, layer) {
                        if pair_offdiag(&core, i, j) <= pair_tol {
                            continue;
                        }
                        let (theta_l, theta_r) = local_pair_svd_angles(&core, i, j);
                        let theta_l = clamp_angle(theta_l, params.max_unwrap_angle);
                        let theta_r = clamp_angle(theta_r, params.max_unwrap_angle);
                        if theta_l.abs() + theta_r.abs() <= 1e-18 {
                            continue;
                        }
                        if accept_offdiag_rotor(
                            &mut core,
                            &mut u_basis,
                            &mut v_basis,
                            i,
                            j,
                            theta_l,
                            theta_r,
                            params.line_search_steps,
                        ) {
                            unwrap_rotations += 1;
                            mark_touched_axes(&mut bottleneck_touched, &[i, j]);
                            if params.record_mzi_phases {
                                mzi_phases.push(MziPhase {
                                    pass,
                                    i,
                                    j,
                                    theta_l,
                                    theta_r,
                                    kind: PhaseRotorKind::Unwrap,
                                });
                            }
                        } else {
                            rejected_rotations += 1;
                        }
                    }
                }
            }

            let mut after_pass = offdiag_norm(&core);
            let mut after_stress = phase_stress(&core);
            let plateau = after_pass >= before_pass * (1.0 - 1e-10);
            let periodic_high_stress = (pass + 1) % 16 == 0 && before_stress > 10.0 * resonance_tol;
            if params.enable_flow_surgery && n >= 4 && (plateau || periodic_high_stress) {
                let row_phase = row_phases(&core);
                let col_phase = col_phases(&core);
                if accept_block4_surgery(
                    &mut core,
                    &mut u_basis,
                    &mut v_basis,
                    &row_phase,
                    &col_phase,
                ) {
                    surgery_blocks += 1;
                    unwrap_rotations += 1;
                    let axes = select_surgery_axes(&row_phase, &col_phase);
                    mark_touched_axes(&mut bottleneck_touched, &axes);
                    after_pass = offdiag_norm(&core);
                    after_stress = phase_stress(&core);
                    if params.record_mzi_phases {
                        let axes = select_surgery_axes(&row_phase, &col_phase);
                        mzi_phases.push(MziPhase {
                            pass,
                            i: axes[0],
                            j: axes[3],
                            theta_l: 0.0,
                            theta_r: 0.0,
                            kind: PhaseRotorKind::Surgery,
                        });
                    }
                } else {
                    rejected_rotations += 1;
                }
            }
            if after_stress <= resonance_tol
                || after_pass <= pair_tol
                || after_pass >= before_pass * (1.0 - 1e-10)
            {
                break;
            }
            let after_row_phase = row_phases(&core);
            let after_col_phase = col_phases(&core);
            flush_bottleneck_cache_updates(
                bottleneck_cache.as_mut(),
                &core,
                &after_row_phase,
                &after_col_phase,
                &mut bottleneck_touched,
            );
        }

        let final_rows = row_phases(&core);
        let final_cols = col_phases(&core);
        flush_bottleneck_cache_updates(
            bottleneck_cache.as_mut(),
            &core,
            &final_rows,
            &final_cols,
            &mut bottleneck_touched,
        );

        let final_offdiag = offdiag_norm(&core);
        let final_phase_stress = phase_stress(&core);
        let bottleneck_cache_updates = bottleneck_cache
            .as_ref()
            .map(|cache| cache.updates)
            .unwrap_or(0);
        let bottleneck_cache_refreshes = bottleneck_cache
            .as_ref()
            .map(|cache| cache.refreshes)
            .unwrap_or(0);
        let (u, sigma, vt) = extract_sorted_svd(&core, &u_basis, &v_basis);
        (
            (u, sigma, vt),
            LieSvdPhaseFlowTrace {
                initial_offdiag,
                final_offdiag,
                initial_phase_stress,
                final_phase_stress,
                passes,
                phase_jumps,
                golden_prespins,
                causal_antispins,
                yinyang_prespins,
                phase_conjugate_prespins,
                prespin_depth,
                yinyang_cycles,
                bottleneck_rotations,
                bottleneck_cache_updates,
                bottleneck_cache_refreshes,
                unwrap_rotations,
                rejected_rotations,
                surgery_blocks,
                samples,
                mzi_phases,
            },
        )
    }

    // Allow: the return type mirrors this crate's established (U, Sigma, Vt[, Trace]) tuple convention; a type alias would obscure the shape at call sites during this stability freeze.
    #[allow(clippy::type_complexity)]
    pub fn phase_lock_rectangular_with_trace(
        mat: &Array2<f64>,
        params: LieSvdPhaseFlowParams,
    ) -> (
        (Array2<f64>, Array1<f64>, Array2<f64>),
        LieSvdPhaseFlowTrace,
    ) {
        let rows = mat.nrows();
        let cols = mat.ncols();
        if rows == cols {
            return Self::phase_lock_with_trace(mat, params);
        }
        assert!(
            rows > 0 && cols > 0,
            "LieSvdPhaseFlow: empty rectangular matrix"
        );

        let mut core = mat.clone();
        let mut u_basis = Array2::<f64>::eye(rows);
        let mut v_basis = Array2::<f64>::eye(cols);
        let ref_norm = frobenius_norm(mat).max(1e-300);
        let pair_tol = params.min_pair_scale * ref_norm.max(1.0);
        let initial_offdiag = offdiag_norm(&core);
        let initial_phase_stress = phase_stress(&core);
        let resonance_tol = params.phase_resonance_tol * initial_phase_stress.max(1.0);
        let mut samples = Vec::new();
        let mut mzi_phases = Vec::new();
        let mut golden_prespins = 0usize;
        let mut causal_antispins = 0usize;
        let mut yinyang_prespins = 0usize;
        let mut phase_conjugate_prespins = 0usize;
        let mut bottleneck_rotations = 0usize;
        let mut phase_jumps = 0usize;
        let mut unwrap_rotations = 0usize;
        let mut rejected_rotations = 0usize;
        let surgery_blocks = 0usize;
        let mut passes = 0usize;
        let corridor = rows.min(cols);
        let causal_bias = triangular_causal_bias(&core);
        let prespin_depth = effective_prespin_depth(
            &params,
            rows.max(cols),
            causal_bias,
            initial_phase_stress,
            ref_norm,
        );
        let yinyang_cycles = effective_yinyang_cycles(
            &params,
            rows.max(cols),
            causal_bias,
            initial_phase_stress,
            ref_norm,
        );
        let layer0_causal =
            params.use_causal_antispin && causal_bias.abs() >= params.causal_antispin_threshold;
        if params.use_phase_conjugate_autospin {
            let (accepted, rejected) = apply_phase_conjugate_autospin_rectangular(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                &params,
                &mut mzi_phases,
                pair_tol,
            );
            phase_conjugate_prespins += accepted;
            rejected_rotations += rejected;
        }
        if params.use_yinyang_prespin && yinyang_cycles > 0 {
            let (accepted, rejected) = apply_yinyang_prespin_rectangular(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                yinyang_cycles,
                &params,
                &mut mzi_phases,
            );
            yinyang_prespins += accepted;
            rejected_rotations += rejected;
        } else if layer0_causal {
            let (accepted, rejected) = apply_causal_antispin_rectangular(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                causal_bias,
                prespin_depth,
                &params,
                &mut mzi_phases,
            );
            causal_antispins += accepted;
            rejected_rotations += rejected;
        } else if params.use_golden_prespin {
            let (accepted, rejected) = apply_golden_prespin_rectangular(
                &mut core,
                &mut u_basis,
                &mut v_basis,
                prespin_depth,
                &params,
                &mut mzi_phases,
            );
            golden_prespins += accepted;
            rejected_rotations += rejected;
        }

        let mut bottleneck_cache =
            if params.use_bottleneck_queue && params.use_incremental_bottleneck_cache {
                let rows = row_phases(&core);
                let cols = col_phases(&core);
                Some(BottleneckPairCache::new(
                    &core,
                    &rows,
                    &cols,
                    corridor,
                    pair_tol,
                    params.active_set_alpha,
                ))
            } else {
                None
            };
        let mut bottleneck_touched = vec![false; corridor];
        // See the square route above: reused across every pass instead of
        // allocating a fresh Vec<AxisPhase> each time.
        let mut row_phase_buf: Vec<AxisPhase> = Vec::with_capacity(rows);
        let mut col_phase_buf: Vec<AxisPhase> = Vec::with_capacity(cols);

        for pass in 0..params.max_passes {
            passes = pass + 1;
            let before_pass = offdiag_norm(&core);
            row_phases_into(&core, &mut row_phase_buf);
            col_phases_into(&core, &mut col_phase_buf);
            let row_phase = &row_phase_buf;
            let col_phase = &col_phase_buf;
            let before_stress = summarize_stress(row_phase, col_phase);
            samples.push((before_pass, before_stress));
            if before_stress <= resonance_tol {
                break;
            }

            for layer in 0..2 {
                let mut i = layer;
                while i + 1 < rows {
                    let j = i + 1;
                    let theta = clamp_angle(
                        golden_phase_jump(
                            -0.5 * (row_phase[j].phase - row_phase[i].phase),
                            pass,
                            i,
                            j,
                            params.use_golden_jumps,
                        ),
                        params.max_jump_angle,
                    );
                    if theta.abs() > 1e-18 {
                        if accept_left_phase_rotor(
                            &mut core,
                            &mut u_basis,
                            i,
                            j,
                            theta,
                            params.line_search_steps,
                        ) {
                            phase_jumps += 1;
                            mark_touched_axes(&mut bottleneck_touched, &[i, j]);
                        } else {
                            rejected_rotations += 1;
                        }
                    }
                    i += 2;
                }
            }

            for layer in 0..2 {
                let mut i = layer;
                while i + 1 < cols {
                    let j = i + 1;
                    let theta = clamp_angle(
                        golden_phase_jump(
                            -0.5 * (col_phase[j].phase - col_phase[i].phase),
                            pass,
                            j,
                            i,
                            params.use_golden_jumps,
                        ),
                        params.max_jump_angle,
                    );
                    if theta.abs() > 1e-18 {
                        if accept_right_phase_rotor(
                            &mut core,
                            &mut v_basis,
                            i,
                            j,
                            theta,
                            params.line_search_steps,
                        ) {
                            phase_jumps += 1;
                            mark_touched_axes(&mut bottleneck_touched, &[i, j]);
                        } else {
                            rejected_rotations += 1;
                        }
                    }
                    i += 2;
                }
            }

            if params.use_bottleneck_queue {
                let due_for_rebuild = params.bottleneck_cache_refresh_period > 0
                    && pass > 0
                    && pass % params.bottleneck_cache_refresh_period == 0;
                if due_for_rebuild {
                    if let Some(cache) = bottleneck_cache.as_mut() {
                        cache.rebuild(
                            &core,
                            row_phase,
                            col_phase,
                            pair_tol,
                            params.active_set_alpha,
                        );
                    }
                    bottleneck_touched.fill(false);
                } else {
                    flush_bottleneck_cache_updates(
                        bottleneck_cache.as_mut(),
                        &core,
                        row_phase,
                        col_phase,
                        &mut bottleneck_touched,
                    );
                }
                let candidates = if let Some(cache) = bottleneck_cache.as_mut() {
                    cache.pop_conflict_free(
                        &core,
                        row_phase,
                        col_phase,
                        params.bottleneck_pairs.min(corridor),
                    )
                } else {
                    bottleneck_pairs(
                        &core,
                        row_phase,
                        col_phase,
                        params.bottleneck_pairs.min(corridor),
                        pair_tol,
                        params.active_set_alpha,
                    )
                };
                let background_energy = mean_axis_stress(row_phase, col_phase);
                for pair in candidates {
                    if pair_offdiag(&core, pair.i, pair.j) <= pair_tol {
                        continue;
                    }
                    let viscosity = if params.use_adaptive_viscosity {
                        adaptive_energy_ratio_viscosity(pair.score, background_energy)
                    } else {
                        params.phase_viscosity
                    };
                    let (theta_l, theta_r) = local_pair_svd_angles(&core, pair.i, pair.j);
                    let theta_l = prepare_phase_angle(
                        theta_l * viscosity,
                        params.max_unwrap_angle,
                        params.phase_quantization_levels,
                    );
                    let theta_r = prepare_phase_angle(
                        theta_r * viscosity,
                        params.max_unwrap_angle,
                        params.phase_quantization_levels,
                    );
                    if theta_l.abs() + theta_r.abs() <= 1e-18 {
                        continue;
                    }
                    if accept_offdiag_rotor(
                        &mut core,
                        &mut u_basis,
                        &mut v_basis,
                        pair.i,
                        pair.j,
                        theta_l,
                        theta_r,
                        params.line_search_steps,
                    ) {
                        bottleneck_rotations += 1;
                        mark_touched_axes(&mut bottleneck_touched, &[pair.i, pair.j]);
                        if params.record_mzi_phases {
                            mzi_phases.push(MziPhase {
                                pass,
                                i: pair.i,
                                j: pair.j,
                                theta_l,
                                theta_r,
                                kind: PhaseRotorKind::Bottleneck,
                            });
                        }
                    } else {
                        rejected_rotations += 1;
                    }
                }
            }

            for pair in active_rectangular_corridor_pairs(
                &core,
                row_phase,
                col_phase,
                params.active_axes.min(corridor),
                pair_tol,
                params.active_set_alpha,
            ) {
                if pair_offdiag(&core, pair.i, pair.j) <= pair_tol {
                    continue;
                }
                let (theta_l, theta_r) = local_pair_svd_angles(&core, pair.i, pair.j);
                let stress_gain = (pair.score / ref_norm.max(1.0)).tanh();
                let theta_l = clamp_angle(
                    theta_l * (1.0 + 0.25 * stress_gain),
                    params.max_unwrap_angle,
                );
                let theta_r = clamp_angle(
                    theta_r * (1.0 + 0.25 * stress_gain),
                    params.max_unwrap_angle,
                );
                if theta_l.abs() + theta_r.abs() <= 1e-18 {
                    continue;
                }
                if accept_offdiag_rotor(
                    &mut core,
                    &mut u_basis,
                    &mut v_basis,
                    pair.i,
                    pair.j,
                    theta_l,
                    theta_r,
                    params.line_search_steps,
                ) {
                    unwrap_rotations += 1;
                    mark_touched_axes(&mut bottleneck_touched, &[pair.i, pair.j]);
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i: pair.i,
                            j: pair.j,
                            theta_l,
                            theta_r,
                            kind: PhaseRotorKind::Unwrap,
                        });
                    }
                } else {
                    rejected_rotations += 1;
                }
            }

            let after_pass = offdiag_norm(&core);
            let after_stress = phase_stress(&core);
            if after_stress <= resonance_tol
                || after_pass <= pair_tol
                || after_pass >= before_pass * (1.0 - 1e-10)
            {
                break;
            }
            let after_row_phase = row_phases(&core);
            let after_col_phase = col_phases(&core);
            flush_bottleneck_cache_updates(
                bottleneck_cache.as_mut(),
                &core,
                &after_row_phase,
                &after_col_phase,
                &mut bottleneck_touched,
            );
        }

        let final_rows = row_phases(&core);
        let final_cols = col_phases(&core);
        flush_bottleneck_cache_updates(
            bottleneck_cache.as_mut(),
            &core,
            &final_rows,
            &final_cols,
            &mut bottleneck_touched,
        );

        let final_offdiag = offdiag_norm(&core);
        let final_phase_stress = phase_stress(&core);
        let bottleneck_cache_updates = bottleneck_cache
            .as_ref()
            .map(|cache| cache.updates)
            .unwrap_or(0);
        let bottleneck_cache_refreshes = bottleneck_cache
            .as_ref()
            .map(|cache| cache.refreshes)
            .unwrap_or(0);
        let (u, sigma, vt) = extract_sorted_rectangular_svd(&core, &u_basis, &v_basis);
        (
            (u, sigma, vt),
            LieSvdPhaseFlowTrace {
                initial_offdiag,
                final_offdiag,
                initial_phase_stress,
                final_phase_stress,
                passes,
                phase_jumps,
                golden_prespins,
                causal_antispins,
                yinyang_prespins,
                phase_conjugate_prespins,
                prespin_depth,
                yinyang_cycles,
                bottleneck_rotations,
                bottleneck_cache_updates,
                bottleneck_cache_refreshes,
                unwrap_rotations,
                rejected_rotations,
                surgery_blocks,
                samples,
                mzi_phases,
            },
        )
    }
}

fn apply_phase_conjugate_autospin_square(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
    pair_tol: f64,
) -> (usize, usize) {
    let rows = row_phases(core);
    let cols = col_phases(core);
    let pairs = bottleneck_pairs(
        core,
        &rows,
        &cols,
        params.bottleneck_pairs.max(params.active_axes),
        pair_tol,
        params.active_set_alpha,
    );
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for (idx, pair) in pairs.into_iter().enumerate() {
        let theta_l = prepare_phase_angle(
            -0.5 * wrap_two_pi(rows[pair.j].phase - rows[pair.i].phase) * params.phase_viscosity,
            params.max_phase_conjugate_angle,
            params.phase_quantization_levels,
        );
        let theta_r = prepare_phase_angle(
            -0.5 * wrap_two_pi(cols[pair.j].phase - cols[pair.i].phase) * params.phase_viscosity,
            params.max_phase_conjugate_angle,
            params.phase_quantization_levels,
        );
        if theta_l.abs() + theta_r.abs() <= 1e-18 {
            continue;
        }
        if accept_offdiag_rotor(
            core,
            u_basis,
            v_basis,
            pair.i,
            pair.j,
            theta_l,
            theta_r,
            params.line_search_steps,
        ) {
            accepted += 1;
            if params.record_mzi_phases {
                mzi_phases.push(MziPhase {
                    pass: idx,
                    i: pair.i,
                    j: pair.j,
                    theta_l,
                    theta_r,
                    kind: PhaseRotorKind::PhaseConjugate,
                });
            }
        } else {
            rejected += 1;
        }
    }
    (accepted, rejected)
}

fn apply_phase_conjugate_autospin_rectangular(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
    pair_tol: f64,
) -> (usize, usize) {
    let rows = row_phases(core);
    let cols = col_phases(core);
    let corridor = core.nrows().min(core.ncols());
    let pairs = bottleneck_pairs(
        core,
        &rows,
        &cols,
        params.bottleneck_pairs.min(corridor),
        pair_tol,
        params.active_set_alpha,
    );
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for (idx, pair) in pairs.into_iter().enumerate() {
        let theta_l = prepare_phase_angle(
            -0.5 * wrap_two_pi(rows[pair.j].phase - rows[pair.i].phase) * params.phase_viscosity,
            params.max_phase_conjugate_angle,
            params.phase_quantization_levels,
        );
        let theta_r = prepare_phase_angle(
            -0.5 * wrap_two_pi(cols[pair.j].phase - cols[pair.i].phase) * params.phase_viscosity,
            params.max_phase_conjugate_angle,
            params.phase_quantization_levels,
        );
        if theta_l.abs() + theta_r.abs() <= 1e-18 {
            continue;
        }
        if accept_offdiag_rotor(
            core,
            u_basis,
            v_basis,
            pair.i,
            pair.j,
            theta_l,
            theta_r,
            params.line_search_steps,
        ) {
            accepted += 1;
            if params.record_mzi_phases {
                mzi_phases.push(MziPhase {
                    pass: idx,
                    i: pair.i,
                    j: pair.j,
                    theta_l,
                    theta_r,
                    kind: PhaseRotorKind::PhaseConjugate,
                });
            }
        } else {
            rejected += 1;
        }
    }
    (accepted, rejected)
}

fn apply_yinyang_prespin_square(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    cycles: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let n = core.nrows().min(core.ncols());
    let layers = params
        .golden_prespin_layers
        .max(params.causal_antispin_layers)
        .min(round_robin_layer_count(n))
        .max(usize::from(n > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for cycle in 0..cycles.clamp(1, 4) {
        for act in 0..4 {
            for layer in 0..layers {
                let pass = cycle * 4 * layers + act * layers + layer;
                for (i, j) in layer_pairs(n, layer) {
                    let theta = yinyang_pair_angle(i, j, act, params.max_yinyang_angle, cycle);
                    let ok = match act {
                        0 | 2 => {
                            if theta.abs() <= 1e-18 {
                                continue;
                            }
                            accept_left_phase_rotor(
                                core,
                                u_basis,
                                i,
                                j,
                                theta,
                                params.line_search_steps,
                            )
                        }
                        _ => {
                            if theta.abs() <= 1e-18 {
                                continue;
                            }
                            accept_right_phase_rotor(
                                core,
                                v_basis,
                                i,
                                j,
                                theta,
                                params.line_search_steps,
                            )
                        }
                    };
                    if ok {
                        accepted += 1;
                        if params.record_mzi_phases {
                            mzi_phases.push(MziPhase {
                                pass,
                                i,
                                j,
                                theta_l: if act == 0 || act == 2 { theta } else { 0.0 },
                                theta_r: if act == 1 || act == 3 { theta } else { 0.0 },
                                kind: PhaseRotorKind::CrossPhaseYinYang,
                            });
                        }
                    } else {
                        rejected += 1;
                    }
                }
            }
        }
    }
    (accepted, rejected)
}

fn apply_yinyang_prespin_rectangular(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    cycles: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let rows = core.nrows();
    let cols = core.ncols();
    let row_layers = params
        .golden_prespin_layers
        .max(params.causal_antispin_layers)
        .min(round_robin_layer_count(rows))
        .max(usize::from(rows > 1));
    let col_layers = params
        .golden_prespin_layers
        .max(params.causal_antispin_layers)
        .min(round_robin_layer_count(cols))
        .max(usize::from(cols > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for cycle in 0..cycles.clamp(1, 4) {
        for act in 0..4 {
            let layers = if act == 0 || act == 2 {
                row_layers
            } else {
                col_layers
            };
            for layer in 0..layers {
                let pass = cycle * 4 * layers + act * layers + layer;
                let dimension = if act == 0 || act == 2 { rows } else { cols };
                for (i, j) in layer_pairs(dimension, layer) {
                    let theta = yinyang_pair_angle(i, j, act, params.max_yinyang_angle, cycle);
                    if theta.abs() <= 1e-18 {
                        continue;
                    }
                    let ok = if act == 0 || act == 2 {
                        accept_left_phase_rotor(
                            core,
                            u_basis,
                            i,
                            j,
                            theta,
                            params.line_search_steps,
                        )
                    } else {
                        accept_right_phase_rotor(
                            core,
                            v_basis,
                            i,
                            j,
                            theta,
                            params.line_search_steps,
                        )
                    };
                    if ok {
                        accepted += 1;
                        if params.record_mzi_phases {
                            mzi_phases.push(MziPhase {
                                pass,
                                i,
                                j,
                                theta_l: if act == 0 || act == 2 { theta } else { 0.0 },
                                theta_r: if act == 1 || act == 3 { theta } else { 0.0 },
                                kind: PhaseRotorKind::CrossPhaseYinYang,
                            });
                        }
                    } else {
                        rejected += 1;
                    }
                }
            }
        }
    }
    (accepted, rejected)
}

fn yinyang_pair_angle(i: usize, j: usize, act: usize, limit: f64, cycle: usize) -> f64 {
    let side = if act == 0 || act == 2 {
        GoldenSide::Row
    } else {
        GoldenSide::Col
    };
    let sign = match act {
        0 | 3 => 1.0,
        _ => -1.0,
    };
    sign * golden_prespin_pair_angle(i, j, side, limit, cycle)
}

fn apply_causal_antispin_square(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    causal_bias: f64,
    prespin_depth: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let n = core.nrows().min(core.ncols());
    let layers = params
        .causal_antispin_layers
        .min(round_robin_layer_count(n))
        .max(usize::from(n > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for depth in 0..prespin_depth.max(1) {
        for layer in 0..layers {
            let pass = depth * layers + layer;
            for (i, j) in layer_pairs(n, layer) {
                let (theta_l, theta_r) = causal_antispin_pair_angles(
                    i,
                    j,
                    causal_bias,
                    params.max_causal_antispin_angle,
                    depth,
                );
                if theta_l.abs() + theta_r.abs() <= 1e-18 {
                    continue;
                }
                if accept_offdiag_rotor(
                    core,
                    u_basis,
                    v_basis,
                    i,
                    j,
                    theta_l,
                    theta_r,
                    params.line_search_steps,
                ) {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l,
                            theta_r,
                            kind: PhaseRotorKind::CausalAntiSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }
    (accepted, rejected)
}

fn apply_causal_antispin_rectangular(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    causal_bias: f64,
    prespin_depth: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let rows = core.nrows();
    let cols = core.ncols();
    let row_layers = params
        .causal_antispin_layers
        .min(round_robin_layer_count(rows))
        .max(usize::from(rows > 1));
    let col_layers = params
        .causal_antispin_layers
        .min(round_robin_layer_count(cols))
        .max(usize::from(cols > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for depth in 0..prespin_depth.max(1) {
        for layer in 0..row_layers {
            let pass = depth * row_layers + layer;
            for (i, j) in layer_pairs(rows, layer) {
                let (theta_l, _) = causal_antispin_pair_angles(
                    i,
                    j,
                    causal_bias,
                    params.max_causal_antispin_angle,
                    depth,
                );
                if theta_l.abs() <= 1e-18 {
                    continue;
                }
                if accept_left_phase_rotor(core, u_basis, i, j, theta_l, params.line_search_steps) {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l,
                            theta_r: 0.0,
                            kind: PhaseRotorKind::CausalAntiSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }

    for depth in 0..prespin_depth.max(1) {
        for layer in 0..col_layers {
            let pass = depth * col_layers + layer;
            for (i, j) in layer_pairs(cols, layer) {
                let (_, theta_r) = causal_antispin_pair_angles(
                    i,
                    j,
                    causal_bias,
                    params.max_causal_antispin_angle,
                    depth,
                );
                if theta_r.abs() <= 1e-18 {
                    continue;
                }
                if accept_right_phase_rotor(core, v_basis, i, j, theta_r, params.line_search_steps)
                {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l: 0.0,
                            theta_r,
                            kind: PhaseRotorKind::CausalAntiSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }
    (accepted, rejected)
}

fn causal_antispin_pair_angles(
    i: usize,
    j: usize,
    causal_bias: f64,
    limit: f64,
    depth: usize,
) -> (f64, f64) {
    let harmonic = golden_harmonic(depth);
    let layer_limit = annealed_limit(limit, depth);
    let diff = wrap_two_pi((j as f64 - i as f64) * GOLDEN_ANGLE * harmonic);
    let sign = if causal_bias >= 0.0 { 1.0 } else { -1.0 };
    let chirality = if depth.is_multiple_of(2) { 1.0 } else { -1.0 };
    let theta = clamp_angle(
        layer_limit * causal_bias.abs().min(1.0) * diff.sin(),
        layer_limit,
    );
    let signed = sign * chirality * theta;
    (signed, -signed)
}

fn apply_golden_prespin_square(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    prespin_depth: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let n = core.nrows().min(core.ncols());
    let layers = params
        .golden_prespin_layers
        .min(round_robin_layer_count(n))
        .max(usize::from(n > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    for depth in 0..prespin_depth.max(1) {
        for layer in 0..layers {
            let pass = depth * layers + layer;
            for (i, j) in layer_pairs(n, layer) {
                let theta_l = golden_prespin_pair_angle(
                    i,
                    j,
                    GoldenSide::Row,
                    params.max_prespin_angle,
                    depth,
                );
                let theta_r = golden_prespin_pair_angle(
                    i,
                    j,
                    GoldenSide::Col,
                    params.max_prespin_angle,
                    depth,
                );
                if theta_l.abs() + theta_r.abs() <= 1e-18 {
                    continue;
                }
                if accept_offdiag_rotor(
                    core,
                    u_basis,
                    v_basis,
                    i,
                    j,
                    theta_l,
                    theta_r,
                    params.line_search_steps,
                ) {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l,
                            theta_r,
                            kind: PhaseRotorKind::GoldenPreSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }
    (accepted, rejected)
}

fn apply_golden_prespin_rectangular(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    prespin_depth: usize,
    params: &LieSvdPhaseFlowParams,
    mzi_phases: &mut Vec<MziPhase>,
) -> (usize, usize) {
    let rows = core.nrows();
    let cols = core.ncols();
    let row_layers = params
        .golden_prespin_layers
        .min(round_robin_layer_count(rows))
        .max(usize::from(rows > 1));
    let col_layers = params
        .golden_prespin_layers
        .min(round_robin_layer_count(cols))
        .max(usize::from(cols > 1));
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for depth in 0..prespin_depth.max(1) {
        for layer in 0..row_layers {
            let pass = depth * row_layers + layer;
            for (i, j) in layer_pairs(rows, layer) {
                let theta = golden_prespin_pair_angle(
                    i,
                    j,
                    GoldenSide::Row,
                    params.max_prespin_angle,
                    depth,
                );
                if theta.abs() <= 1e-18 {
                    continue;
                }
                if accept_left_phase_rotor(core, u_basis, i, j, theta, params.line_search_steps) {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l: theta,
                            theta_r: 0.0,
                            kind: PhaseRotorKind::GoldenPreSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }

    for depth in 0..prespin_depth.max(1) {
        for layer in 0..col_layers {
            let pass = depth * col_layers + layer;
            for (i, j) in layer_pairs(cols, layer) {
                let theta = golden_prespin_pair_angle(
                    i,
                    j,
                    GoldenSide::Col,
                    params.max_prespin_angle,
                    depth,
                );
                if theta.abs() <= 1e-18 {
                    continue;
                }
                if accept_right_phase_rotor(core, v_basis, i, j, theta, params.line_search_steps) {
                    accepted += 1;
                    if params.record_mzi_phases {
                        mzi_phases.push(MziPhase {
                            pass,
                            i,
                            j,
                            theta_l: 0.0,
                            theta_r: theta,
                            kind: PhaseRotorKind::GoldenPreSpin,
                        });
                    }
                } else {
                    rejected += 1;
                }
            }
        }
    }
    (accepted, rejected)
}

#[derive(Clone, Copy)]
enum GoldenSide {
    Row,
    Col,
}

fn golden_prespin_pair_angle(
    i: usize,
    j: usize,
    side: GoldenSide,
    limit: f64,
    depth: usize,
) -> f64 {
    let layer_limit = annealed_limit(limit, depth);
    let phase_i = golden_axis_phase(i, side, depth);
    let phase_j = golden_axis_phase(j, side, depth);
    let diff = wrap_two_pi(phase_j - phase_i);
    clamp_angle(layer_limit * diff.sin(), layer_limit)
}

fn golden_axis_phase(k: usize, side: GoldenSide, depth: usize) -> f64 {
    let multiplier = match side {
        GoldenSide::Row => 1.0,
        GoldenSide::Col => GOLDEN_RATIO,
    };
    wrap_two_pi((k as f64) * GOLDEN_ANGLE * multiplier * golden_harmonic(depth))
}

fn effective_prespin_depth(
    params: &LieSvdPhaseFlowParams,
    dimension: usize,
    causal_bias: f64,
    initial_phase_stress: f64,
    ref_norm: f64,
) -> usize {
    let base = params.prespin_depth.clamp(1, 4);
    if !params.adaptive_prespin_depth {
        return base;
    }
    let stress_ratio = initial_phase_stress / ref_norm.max(1.0);
    // The two branches below currently share the same depth-boost body by
    // design, not oversight: a strong causal bias and a high stress ratio
    // are different triggers that both warrant the same extra pre-spin
    // depth once either fires. Not merged with `||` to keep each trigger's
    // condition independently readable/tunable.
    #[allow(clippy::if_same_then_else)]
    let suggested = if causal_bias.abs() >= params.causal_antispin_threshold {
        if dimension >= 64 {
            3
        } else {
            2
        }
    } else if stress_ratio > 8.0 {
        if dimension >= 64 {
            3
        } else {
            2
        }
    } else {
        base
    };
    base.max(suggested).clamp(1, 4)
}

fn effective_yinyang_cycles(
    params: &LieSvdPhaseFlowParams,
    dimension: usize,
    causal_bias: f64,
    initial_phase_stress: f64,
    ref_norm: f64,
) -> usize {
    if !params.use_yinyang_prespin {
        return 0;
    }
    let base = params.yinyang_cycles.min(4);
    if base > 0 || !params.adaptive_prespin_depth {
        return base;
    }
    let stress_ratio = initial_phase_stress / ref_norm.max(1.0);
    if causal_bias.abs() >= params.causal_antispin_threshold || stress_ratio > 8.0 {
        if dimension >= 64 {
            3
        } else {
            2
        }
    } else {
        1
    }
}

fn golden_harmonic(depth: usize) -> f64 {
    GOLDEN_RATIO.powi(depth.min(8) as i32)
}

fn annealed_limit(limit: f64, depth: usize) -> f64 {
    limit / golden_harmonic(depth).max(1.0)
}

fn wrap_two_pi(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    while angle < -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

fn row_phases(a: &Array2<f64>) -> Vec<AxisPhase> {
    let rows = a.nrows();
    let cols = a.ncols();
    (0..rows).map(|i| axis_phase(cols, |j| a[[i, j]])).collect()
}

fn col_phases(a: &Array2<f64>) -> Vec<AxisPhase> {
    let rows = a.nrows();
    let cols = a.ncols();
    (0..cols).map(|j| axis_phase(rows, |i| a[[i, j]])).collect()
}

/// Same as `row_phases`, but writes into an existing buffer instead of
/// allocating a fresh `Vec` every call. `buf.clear()` drops the old elements
/// without releasing the backing allocation, so once `buf` has grown to `n`
/// capacity, repeated calls across passes are allocation-free. Used in the
/// main pass loop, which recomputes row/column phase twice per pass.
fn row_phases_into(a: &Array2<f64>, buf: &mut Vec<AxisPhase>) {
    let rows = a.nrows();
    let cols = a.ncols();
    buf.clear();
    buf.extend((0..rows).map(|i| axis_phase(cols, |j| a[[i, j]])));
}

/// See `row_phases_into`.
fn col_phases_into(a: &Array2<f64>, buf: &mut Vec<AxisPhase>) {
    let rows = a.nrows();
    let cols = a.ncols();
    buf.clear();
    buf.extend((0..cols).map(|j| axis_phase(rows, |i| a[[i, j]])));
}

fn axis_phase<F>(n: usize, at: F) -> AxisPhase
where
    F: Fn(usize) -> f64,
{
    if n == 0 {
        return AxisPhase {
            stress: 0.0,
            entropy: 0.0,
            phase: 0.0,
            norm: 0.0,
        };
    }

    let mut sum = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    let mut delay_dot = 0.0_f64;
    let mut gradient_sq = 0.0_f64;
    for i in 0..n {
        let x = at(i);
        let y = at((i + 1) % n);
        sum += x;
        norm_sq += x * x;
        delay_dot += x * y;
        let d = y - x;
        gradient_sq += d * d;
    }

    let mean = sum / n as f64;
    let mut vector_sq = 0.0_f64;
    for i in 0..n {
        let centered = at(i) - mean;
        vector_sq += centered * centered;
    }

    let bivector_sq = (norm_sq * norm_sq - delay_dot * delay_dot).max(0.0);
    let entropy = energy_entropy_by(n, norm_sq, at);
    let twist = bivector_sq.sqrt() / norm_sq.max(1e-300);
    let stress = bivector_sq.sqrt() + gradient_sq.sqrt() + entropy * norm_sq.sqrt();
    let orient = if delay_dot >= 0.0 { 1.0 } else { -1.0 };
    let phase = orient
        * gradient_sq
            .sqrt()
            .atan2(mean.abs() + vector_sq.sqrt() + 1e-300);

    AxisPhase {
        stress: stress * (0.5 + 0.5 * twist),
        entropy,
        phase,
        norm: norm_sq.sqrt(),
    }
}

/// Axes whose row/col norm bound cannot rule them out of any pair above
/// `pair_tol`. For any `k`, every entry of row `k` and column `k` is bounded
/// by `rows[k].norm` and `cols[k].norm` respectively (an entry can't exceed
/// the norm of the vector it belongs to), so
/// `pair_offdiag(i, j) = |core[i,j]| + |core[j,i]|`
///   `<= (col_norm_j + row_norm_j) = axis_energy_j`, and symmetrically
///   `<= axis_energy_i`.
/// So `pair_offdiag(i, j) <= min(axis_energy_i, axis_energy_j)`, meaning any
/// pair with a "cold" axis (`axis_energy <= pair_tol`) on either side is
/// provably at or below `pair_tol` and can be skipped without touching
/// `core`. This part is an exact bound, not a heuristic: it never discards a
/// pair that could matter, only ones that provably can't.
///
/// `active_set_alpha > 0.0` additionally applies a Strong-Rules-style
/// relative floor `alpha * max(axis_energy)`, on top of the exact
/// `pair_tol` floor. That part is a heuristic (same idea as LASSO/glmnet
/// active-set screening): on inputs where every axis carries some energy
/// above the machine-noise floor `pair_tol`, the exact bound alone rarely
/// prunes anything, while the relative floor drops axes that are merely
/// small relative to the current hottest one. `0.0` keeps the exact-only
/// behavior.
fn hot_axes(
    rows: &[AxisPhase],
    cols: &[AxisPhase],
    n: usize,
    pair_tol: f64,
    active_set_alpha: f64,
) -> Vec<usize> {
    let n = n.min(rows.len()).min(cols.len());
    let floor = if active_set_alpha > 0.0 {
        let max_energy = (0..n)
            .map(|k| rows[k].norm + cols[k].norm)
            .fold(0.0_f64, f64::max);
        pair_tol.max(active_set_alpha * max_energy)
    } else {
        pair_tol
    };
    (0..n)
        .filter(|&k| rows[k].norm + cols[k].norm > floor)
        .collect()
}

/// Mean row/column `stress` for the current pass: the ambient phase-field
/// background `R` that `adaptive_energy_ratio_viscosity` compares a
/// candidate pair's own energy against.
fn mean_axis_stress(rows: &[AxisPhase], cols: &[AxisPhase]) -> f64 {
    let count = rows.len() + cols.len();
    if count == 0 {
        return 0.0;
    }
    let sum: f64 =
        rows.iter().map(|x| x.stress).sum::<f64>() + cols.iter().map(|x| x.stress).sum::<f64>();
    sum / count as f64
}

/// `gamma = P / (P + R)`: a normalized signal/background damping ratio, not
/// a Kalman gain (no covariance is propagated across passes). `P` is the
/// candidate pair's own energy, `R` is `mean_axis_stress` for the current
/// pass. A pair much louder than the ambient field is trusted near-fully; a
/// pair near the noise floor is damped toward half strength.
fn adaptive_energy_ratio_viscosity(pair_energy: f64, background_energy: f64) -> f64 {
    let p = pair_energy.max(0.0);
    let r = background_energy.max(0.0);
    if p + r <= 1e-300 {
        0.5
    } else {
        p / (p + r)
    }
}

fn energy_entropy_by<F>(n: usize, energy_sum: f64, at: F) -> f64
where
    F: Fn(usize) -> f64,
{
    if n <= 1 || energy_sum <= 1e-300 {
        return 0.0;
    }
    let mut entropy = 0.0_f64;
    for i in 0..n {
        let v = at(i);
        let p = (v * v) / energy_sum;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy / (n as f64).ln()
}

fn summarize_stress(rows: &[AxisPhase], cols: &[AxisPhase]) -> f64 {
    rows.iter().map(|x| x.stress).sum::<f64>() + cols.iter().map(|x| x.stress).sum::<f64>()
}

fn golden_phase_jump(theta: f64, pass: usize, i: usize, j: usize, enabled: bool) -> f64 {
    if !enabled || theta.abs() <= 1e-18 {
        return theta;
    }
    let phase = GOLDEN_ANGLE * ((pass + 1) as f64)
        + (i as f64 + 1.0) * 0.6180339887498948
        + (j as f64 + 1.0) * 0.3819660112501051;
    let scale = 0.6180339887498948 + 0.3819660112501051 * phase.sin().abs();
    theta * scale
}

fn phase_stress(a: &Array2<f64>) -> f64 {
    let rows = row_phases(a);
    let cols = col_phases(a);
    summarize_stress(&rows, &cols)
}

fn active_phase_pairs(
    core: &Array2<f64>,
    rows: &[AxisPhase],
    cols: &[AxisPhase],
    active_axes: usize,
    pair_tol: f64,
    active_set_alpha: f64,
) -> Vec<CandidatePair> {
    let n = core.nrows();
    if active_axes >= n {
        return PairEnergyCache::square(core, rows, cols, pair_tol, active_set_alpha)
            .active_conflict_free(active_axes);
    }
    let mut axes: Vec<usize> = (0..n).collect();
    axes.sort_by(|&a, &b| {
        let sa = rows[a].stress + cols[a].stress;
        let sb = rows[b].stress + cols[b].stress;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    axes.truncate(active_axes.min(n));

    // Restrict the partner search to axes that can't be ruled out by the
    // exact row/col norm bound (see `hot_axes`); a cold column can't hold a
    // pair above `pair_tol` with any row.
    let hot_partners = hot_axes(rows, cols, n.min(core.ncols()), pair_tol, active_set_alpha);

    let mut candidates = Vec::new();
    for &i in &axes {
        for &j in &hot_partners {
            if i == j {
                continue;
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            let entropy_gap = (rows[a].entropy - rows[b].entropy).abs()
                + (cols[a].entropy - cols[b].entropy).abs();
            let score = pair_offdiag(core, a, b)
                + 0.05 * (rows[a].stress + rows[b].stress + cols[a].stress + cols[b].stress)
                + 0.01 * entropy_gap;
            candidates.push(CandidatePair { i: a, j: b, score });
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used = vec![false; n];
    let mut out = Vec::new();
    for pair in candidates {
        if !used[pair.i] && !used[pair.j] {
            used[pair.i] = true;
            used[pair.j] = true;
            out.push(pair);
        }
    }
    out
}

fn active_rectangular_corridor_pairs(
    core: &Array2<f64>,
    rows: &[AxisPhase],
    cols: &[AxisPhase],
    active_axes: usize,
    pair_tol: f64,
    active_set_alpha: f64,
) -> Vec<CandidatePair> {
    let k = core.nrows().min(core.ncols());
    if active_axes >= k {
        return PairEnergyCache::square(core, rows, cols, pair_tol, active_set_alpha)
            .active_conflict_free(active_axes);
    }
    let mut axes: Vec<usize> = (0..k).collect();
    axes.sort_by(|&a, &b| {
        let sa = rows[a].stress + cols[a].stress;
        let sb = rows[b].stress + cols[b].stress;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    axes.truncate(active_axes.min(k));

    let hot_partners = hot_axes(rows, cols, k, pair_tol, active_set_alpha);
    let mut candidates = Vec::new();
    for &i in &axes {
        for &j in &hot_partners {
            if i == j {
                continue;
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            let entropy_gap = (rows[a].entropy - rows[b].entropy).abs()
                + (cols[a].entropy - cols[b].entropy).abs();
            let score = pair_offdiag(core, a, b)
                + 0.05 * (rows[a].stress + rows[b].stress + cols[a].stress + cols[b].stress)
                + 0.01 * entropy_gap;
            candidates.push(CandidatePair { i: a, j: b, score });
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used = vec![false; k];
    let mut out = Vec::new();
    for pair in candidates {
        if !used[pair.i] && !used[pair.j] {
            used[pair.i] = true;
            used[pair.j] = true;
            out.push(pair);
        }
    }
    out
}

fn bottleneck_pairs(
    core: &Array2<f64>,
    rows: &[AxisPhase],
    cols: &[AxisPhase],
    max_pairs: usize,
    pair_tol: f64,
    active_set_alpha: f64,
) -> Vec<CandidatePair> {
    let n = core.nrows().min(core.ncols());
    let hot = hot_axes(rows, cols, n, pair_tol, active_set_alpha);
    let mut candidates =
        Vec::with_capacity(hot.len().saturating_mul(hot.len().saturating_sub(1)) / 2);
    for (a, &i) in hot.iter().enumerate() {
        for &j in &hot[a + 1..] {
            let pair_energy = core[[i, j]] * core[[i, j]] + core[[j, i]] * core[[j, i]];
            let phase_gap =
                (rows[i].phase - rows[j].phase).abs() + (cols[i].phase - cols[j].phase).abs();
            let stress = rows[i].stress + rows[j].stress + cols[i].stress + cols[j].stress;
            let score = pair_energy * (1.0 + 0.01 * phase_gap) + 1e-12 * stress;
            candidates.push(CandidatePair { i, j, score });
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used = vec![false; n];
    let mut out = Vec::new();
    for pair in candidates {
        if !used[pair.i] && !used[pair.j] {
            used[pair.i] = true;
            used[pair.j] = true;
            out.push(pair);
            if out.len() >= max_pairs.max(1) {
                break;
            }
        }
    }
    out
}

fn prepare_phase_angle(theta: f64, limit: f64, quantization_levels: usize) -> f64 {
    let theta = clamp_angle(theta, limit);
    if quantization_levels >= 4 {
        quantize_phase_angle(theta, quantization_levels)
    } else {
        theta
    }
}

fn quantize_phase_angle(theta: f64, levels: usize) -> f64 {
    let step = 2.0 * std::f64::consts::PI / levels.max(4) as f64;
    (theta / step).round() * step
}

// Allow: internal hot-path/state-threading signature; restructuring risks introducing bugs in already-verified numerical code during this stability freeze.
#[allow(clippy::too_many_arguments)]
fn accept_offdiag_rotor(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta_l: f64,
    theta_r: f64,
    line_search_steps: usize,
) -> bool {
    let before = local_offdiag_sq_for_axes(core, i, j);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let tl = theta_l * scale;
        let tr = theta_r * scale;
        apply_left_rotor_to_core(core, i, j, tl);
        apply_right_rotor_to_core(core, i, j, tr);
        let after = local_offdiag_sq_for_axes(core, i, j);
        if after <= before + slack && after.is_finite() {
            apply_basis_rotor(u_basis, i, j, tl);
            apply_basis_rotor(v_basis, i, j, tr);
            return true;
        }
        apply_right_rotor_to_core(core, i, j, -tr);
        apply_left_rotor_to_core(core, i, j, -tl);
        scale *= 0.5;
    }
    false
}

fn accept_block4_surgery(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    rows: &[AxisPhase],
    cols: &[AxisPhase],
) -> bool {
    let n = core.nrows().min(core.ncols());
    if n < 4 {
        return false;
    }
    let axes = select_surgery_axes(rows, cols);
    let before = offdiag_norm(core);
    let block = extract_block4(core, axes);
    let (ub, _sigma, vtb) = crate::lie_svd_micro::LieSvdMicro::solve(&block);
    let vb = vtb.t().to_owned();

    apply_left_block_to_core(core, axes, &ub);
    apply_right_block_to_core(core, axes, &vb);
    let after = offdiag_norm(core);
    if after.is_finite() && after < before * (1.0 - 1e-12) {
        apply_basis_block(u_basis, axes, &ub);
        apply_basis_block(v_basis, axes, &vb);
        true
    } else {
        apply_right_block_to_core(core, axes, &vb.t().to_owned());
        apply_left_block_to_core(core, axes, &ub.t().to_owned());
        false
    }
}

fn select_surgery_axes(rows: &[AxisPhase], cols: &[AxisPhase]) -> [usize; 4] {
    let n = rows.len().min(cols.len());
    let mut axes: Vec<usize> = (0..n).collect();
    axes.sort_by(|&a, &b| {
        let sb = rows[b].stress + cols[b].stress;
        let sa = rows[a].stress + cols[a].stress;
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = [0usize; 4];
    for (k, out_k) in out.iter_mut().enumerate() {
        *out_k = axes.get(k).copied().unwrap_or(k);
    }
    out.sort_unstable();
    out
}

fn extract_block4(core: &Array2<f64>, axes: [usize; 4]) -> Array2<f64> {
    Array2::from_shape_fn((4, 4), |(i, j)| core[[axes[i], axes[j]]])
}

fn apply_left_block_to_core(core: &mut Array2<f64>, axes: [usize; 4], q: &Array2<f64>) {
    let cols = core.ncols();
    let mut tmp = [0.0_f64; 4];
    for col in 0..cols {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += q[[b, a]] * core[[axes[b], col]];
            }
        }
        for a in 0..4 {
            core[[axes[a], col]] = tmp[a];
        }
    }
}

fn apply_right_block_to_core(core: &mut Array2<f64>, axes: [usize; 4], q: &Array2<f64>) {
    let rows = core.nrows();
    let mut tmp = [0.0_f64; 4];
    for row in 0..rows {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += core[[row, axes[b]]] * q[[b, a]];
            }
        }
        for a in 0..4 {
            core[[row, axes[a]]] = tmp[a];
        }
    }
}

fn apply_basis_block(basis: &mut Array2<f64>, axes: [usize; 4], q: &Array2<f64>) {
    let rows = basis.nrows();
    let mut tmp = [0.0_f64; 4];
    for row in 0..rows {
        for a in 0..4 {
            tmp[a] = 0.0;
            for b in 0..4 {
                tmp[a] += basis[[row, axes[b]]] * q[[b, a]];
            }
        }
        for a in 0..4 {
            basis[[row, axes[a]]] = tmp[a];
        }
    }
}

fn local_offdiag_sq_for_axes(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    let rows = work.nrows();
    let cols = work.ncols();
    let mut s = 0.0_f64;
    for col in 0..cols {
        if col != i {
            s += work[[i, col]] * work[[i, col]];
        }
        if col != j {
            s += work[[j, col]] * work[[j, col]];
        }
    }
    for row in 0..rows {
        if row != i && row != j {
            s += work[[row, i]] * work[[row, i]];
            s += work[[row, j]] * work[[row, j]];
        }
    }
    s
}

fn accept_left_phase_rotor(
    core: &mut Array2<f64>,
    u_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
    line_search_steps: usize,
) -> bool {
    let before = local_row_offdiag_sq(core, i, j);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let t = theta * scale;
        apply_left_rotor_to_core(core, i, j, t);
        let after = local_row_offdiag_sq(core, i, j);
        if after <= before + slack && after.is_finite() {
            apply_basis_rotor(u_basis, i, j, t);
            return true;
        }
        apply_left_rotor_to_core(core, i, j, -t);
        scale *= 0.5;
    }
    false
}

fn accept_right_phase_rotor(
    core: &mut Array2<f64>,
    v_basis: &mut Array2<f64>,
    i: usize,
    j: usize,
    theta: f64,
    line_search_steps: usize,
) -> bool {
    let before = local_col_offdiag_sq(core, i, j);
    let slack = 1e-14 * before.max(1.0);
    let mut scale = 1.0_f64;
    for _ in 0..line_search_steps.max(1) {
        let t = theta * scale;
        apply_right_rotor_to_core(core, i, j, t);
        let after = local_col_offdiag_sq(core, i, j);
        if after <= before + slack && after.is_finite() {
            apply_basis_rotor(v_basis, i, j, t);
            return true;
        }
        apply_right_rotor_to_core(core, i, j, -t);
        scale *= 0.5;
    }
    false
}

fn local_row_offdiag_sq(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    let cols = work.ncols();
    let mut s = 0.0_f64;
    for col in 0..cols {
        if col != i {
            s += work[[i, col]] * work[[i, col]];
        }
        if col != j {
            s += work[[j, col]] * work[[j, col]];
        }
    }
    s
}

fn local_col_offdiag_sq(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    let rows = work.nrows();
    let mut s = 0.0_f64;
    for row in 0..rows {
        if row != i {
            s += work[[row, i]] * work[[row, i]];
        }
        if row != j {
            s += work[[row, j]] * work[[row, j]];
        }
    }
    s
}

fn triangular_causal_bias(a: &Array2<f64>) -> f64 {
    let n = a.nrows().min(a.ncols());
    let mut upper = 0.0_f64;
    let mut lower = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            upper += a[[i, j]] * a[[i, j]];
            lower += a[[j, i]] * a[[j, i]];
        }
    }
    (upper - lower) / (upper + lower).max(1e-300)
}

fn directional_causal_rotor(
    core: &Array2<f64>,
    i: usize,
    j: usize,
    causal_bias: f64,
    limit: f64,
) -> (f64, f64) {
    let diag = 0.5 * (core[[i, i]].abs() + core[[j, j]].abs()).max(1e-300);
    if causal_bias >= 0.0 {
        let theta = -0.5 * core[[i, j]].atan2(diag);
        (0.0, clamp_angle(theta, limit))
    } else {
        let theta = -0.5 * core[[j, i]].atan2(diag);
        (clamp_angle(theta, limit), 0.0)
    }
}

fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn offdiag_norm(a: &Array2<f64>) -> f64 {
    let rows = a.nrows();
    let cols = a.ncols();
    let mut s = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            if i != j {
                s += a[[i, j]] * a[[i, j]];
            }
        }
    }
    s.sqrt()
}

fn pair_offdiag(work: &Array2<f64>, i: usize, j: usize) -> f64 {
    work[[i, j]].abs() + work[[j, i]].abs()
}

fn clamp_angle(theta: f64, limit: f64) -> f64 {
    theta.clamp(-limit.abs(), limit.abs())
}

fn wrap_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::FRAC_PI_2 {
        angle -= std::f64::consts::PI;
    }
    while angle < -std::f64::consts::FRAC_PI_2 {
        angle += std::f64::consts::PI;
    }
    angle
}

fn local_pair_svd_angles(work: &Array2<f64>, i: usize, j: usize) -> (f64, f64) {
    let a = work[[i, i]];
    let b = work[[i, j]];
    let c = work[[j, i]];
    let d = work[[j, j]];
    let sum_angle = (-(b + c)).atan2(a - d);
    let diff_angle = (b - c).atan2(a + d);
    (
        wrap_angle(0.5 * (sum_angle + diff_angle)),
        wrap_angle(0.5 * (sum_angle - diff_angle)),
    )
}

fn round_robin_layer_count(n: usize) -> usize {
    if n.is_multiple_of(2) {
        n.saturating_sub(1)
    } else {
        n
    }
}

fn layer_pairs(n: usize, layer: usize) -> Vec<(usize, usize)> {
    let m = if n.is_multiple_of(2) { n } else { n + 1 };
    let ring = m - 1;
    let mut pairs = Vec::with_capacity(m / 2);
    for k in 0..(m / 2) {
        let (a, b) = if k == 0 {
            (m - 1, layer % ring)
        } else {
            ((layer + k) % ring, (layer + ring - k) % ring)
        };
        if a < n && b < n {
            pairs.push(if a < b { (a, b) } else { (b, a) });
        }
    }
    pairs
}

fn apply_left_rotor_to_core(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let cols = work.ncols();
    let (s, c) = theta.sin_cos();
    for col in 0..cols {
        let ai = work[[i, col]];
        let aj = work[[j, col]];
        work[[i, col]] = c * ai - s * aj;
        work[[j, col]] = s * ai + c * aj;
    }
}

fn apply_right_rotor_to_core(work: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let rows = work.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..rows {
        let ai = work[[r, i]];
        let aj = work[[r, j]];
        work[[r, i]] = c * ai - s * aj;
        work[[r, j]] = s * ai + c * aj;
    }
}

fn apply_basis_rotor(basis: &mut Array2<f64>, i: usize, j: usize, theta: f64) {
    let n = basis.nrows();
    let (s, c) = theta.sin_cos();
    for r in 0..n {
        let bi = basis[[r, i]];
        let bj = basis[[r, j]];
        basis[[r, i]] = c * bi - s * bj;
        basis[[r, j]] = s * bi + c * bj;
    }
}

fn extract_sorted_svd(
    work: &Array2<f64>,
    u_basis: &Array2<f64>,
    v_basis: &Array2<f64>,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = work.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        work[[b, b]]
            .abs()
            .partial_cmp(&work[[a, a]].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u_sorted = Array2::<f64>::zeros((n, n));
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    let mut sigma = Array1::<f64>::zeros(n);
    for (dst, &src) in order.iter().enumerate() {
        let d = work[[src, src]];
        sigma[dst] = d.abs();
        let sign = if d < 0.0 { -1.0 } else { 1.0 };
        for r in 0..n {
            u_sorted[[r, dst]] = sign * u_basis[[r, src]];
            v_sorted[[r, dst]] = v_basis[[r, src]];
        }
    }
    (u_sorted, sigma, v_sorted.t().to_owned())
}

fn extract_sorted_rectangular_svd(
    work: &Array2<f64>,
    u_basis: &Array2<f64>,
    v_basis: &Array2<f64>,
) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let rows = work.nrows();
    let cols = work.ncols();
    let k = rows.min(cols);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        work[[b, b]]
            .abs()
            .partial_cmp(&work[[a, a]].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u_sorted = Array2::<f64>::zeros((rows, rows));
    let mut v_sorted = Array2::<f64>::zeros((cols, cols));
    let mut sigma = Array1::<f64>::zeros(k);
    let mut used_u = vec![false; rows];
    let mut used_v = vec![false; cols];

    for (dst, &src) in order.iter().enumerate() {
        let d = work[[src, src]];
        sigma[dst] = d.abs();
        let sign = if d < 0.0 { -1.0 } else { 1.0 };
        for r in 0..rows {
            u_sorted[[r, dst]] = sign * u_basis[[r, src]];
        }
        for r in 0..cols {
            v_sorted[[r, dst]] = v_basis[[r, src]];
        }
        used_u[src] = true;
        used_v[src] = true;
    }

    let mut dst = k;
    for src in 0..rows {
        if !used_u[src] {
            for r in 0..rows {
                u_sorted[[r, dst]] = u_basis[[r, src]];
            }
            dst += 1;
        }
    }

    let mut dst = k;
    for src in 0..cols {
        if !used_v[src] {
            for r in 0..cols {
                v_sorted[[r, dst]] = v_basis[[r, src]];
            }
            dst += 1;
        }
    }

    (u_sorted, sigma, v_sorted.t().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn metrics(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let sigma_mat = Array2::from_diag(sigma);
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident = Array2::<f64>::eye(a.nrows());
        let orth_u = (&u.t().dot(u) - &ident).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident).mapv(|x| x * x).sum().sqrt();
        (rel, orth_u, orth_v)
    }

    fn rectangular_metrics(
        a: &Array2<f64>,
        u: &Array2<f64>,
        sigma: &Array1<f64>,
        vt: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let mut sigma_mat = Array2::<f64>::zeros((a.nrows(), a.ncols()));
        for i in 0..sigma.len() {
            sigma_mat[[i, i]] = sigma[i];
        }
        let recon = u.dot(&sigma_mat).dot(vt);
        let rel = (&recon - a).mapv(|x| x * x).sum().sqrt() / frobenius_norm(a).max(1e-300);
        let ident_u = Array2::<f64>::eye(a.nrows());
        let ident_v = Array2::<f64>::eye(a.ncols());
        let orth_u = (&u.t().dot(u) - &ident_u).mapv(|x| x * x).sum().sqrt();
        let orth_v = (&vt.dot(&vt.t()) - &ident_v).mapv(|x| x * x).sum().sqrt();
        (rel, orth_u, orth_v)
    }

    #[test]
    fn phaseflow_raw_random_16_reduces_phase_field() {
        let mut rng = StdRng::seed_from_u64(111);
        let a = Array2::from_shape_fn((16, 16), |_| rng.gen::<f64>() - 0.5);
        let ((u, sigma, vt), trace) =
            LieSvdPhaseFlow::solve_with_trace(&a, LieSvdPhaseFlowParams::for_n(16));
        let (rel, orth_u, orth_v) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(rel < 2e-1, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phaseflow_polished_random_16() {
        let mut rng = StdRng::seed_from_u64(112);
        let a = Array2::from_shape_fn((16, 16), |_| rng.gen::<f64>() - 0.5);
        let ((u, sigma, vt), trace) =
            LieSvdPhaseFlow::solve_with_digital_polish(&a, LieSvdPhaseFlowParams::for_n(16));
        let (rel, orth_u, orth_v) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(rel < 1e-11, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phaseflow_preconditioner_is_monotone_on_structured_shift() {
        let n = 16;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0 + i as f64 * 0.01;
            if i + 1 < n {
                a[[i, i + 1]] = 4.0;
            }
        }
        let ((_u, _sigma, _vt), trace) =
            LieSvdPhaseFlow::phase_lock_with_trace(&a, LieSvdPhaseFlowParams::for_n(n));
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(trace.phase_jumps + trace.unwrap_rotations > 0);
    }

    #[test]
    fn phaseflow_rectangular_route_tracks_row_and_column_spaces() {
        let rows = 12;
        let cols = 20;
        let mut a = Array2::<f64>::zeros((rows, cols));
        for i in 0..rows.min(cols) {
            a[[i, i]] = 2.0 + i as f64 * 0.1;
        }
        for i in 0..rows {
            for j in 0..cols {
                if i != j {
                    a[[i, j]] = 1e-4 * ((i * 7 + j * 11 + 5) as f64).sin();
                }
            }
        }
        let mut params = LieSvdPhaseFlowParams::for_n(rows.max(cols));
        params.max_passes = 24;
        let ((u, sigma, vt), trace) =
            LieSvdPhaseFlow::phase_lock_rectangular_with_trace(&a, params);
        let (rel, orth_u, orth_v) = rectangular_metrics(&a, &u, &sigma, &vt);
        assert_eq!(u.dim(), (rows, rows));
        assert_eq!(vt.dim(), (cols, cols));
        assert_eq!(sigma.len(), rows.min(cols));
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(rel < 1e-3, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phaseflow_polished_degenerate_32() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::DegenerateSpectrum, 17);
        let ((u, sigma, vt), trace) =
            LieSvdPhaseFlow::solve_with_digital_polish(&case.a, LieSvdPhaseFlowParams::for_n(32));
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(rel < 1e-11, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phaseflow_polished_causal_jordan_64() {
        let case = crate::profiles::generate(64, crate::profiles::Profile::JordanDefective, 17);
        let mut params = LieSvdPhaseFlowParams::for_n(64);
        params.max_passes = 80;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.final_offdiag <= trace.initial_offdiag + 1e-10);
        assert!(trace.phase_jumps + trace.unwrap_rotations > 0);
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-9, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-9, "orth_v={orth_v:e}");
    }

    #[test]
    fn golden_jumps_polished_route_stays_accurate() {
        let n = 32;
        let mut rng = StdRng::seed_from_u64(1717);
        let a = Array2::from_shape_fn((n, n), |_| rng.gen_range(-1.0_f64..1.0));
        let mut params = LieSvdPhaseFlowParams::for_n(n);
        params.use_golden_jumps = true;
        params.use_golden_prespin = false;
        params.enable_flow_surgery = true;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&a, params);
        let (rel, orth_u, orth_v) = metrics(&a, &u, &sigma, &vt);
        assert!(trace.phase_jumps > 0);
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn golden_prespin_layer_records_and_polishes() {
        let n = 32;
        let case = crate::profiles::generate(n, crate::profiles::Profile::DegenerateSpectrum, 19);
        let mut params = LieSvdPhaseFlowParams::for_n(n);
        params.use_golden_prespin = true;
        params.golden_prespin_layers = 2;
        params.record_mzi_phases = true;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.golden_prespins > 0);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|p| p.kind == PhaseRotorKind::GoldenPreSpin));
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn causal_antispin_records_on_jordan_flow() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::JordanDefective, 17);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.record_mzi_phases = true;
        params.max_passes = 32;
        params.use_causal_antispin = true;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.causal_antispins > 0);
        assert_eq!(trace.golden_prespins, 0);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|p| p.kind == PhaseRotorKind::CausalAntiSpin));
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn yinyang_prespin_records_and_polishes() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::JordanDefective, 23);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.record_mzi_phases = true;
        params.use_yinyang_prespin = true;
        params.yinyang_cycles = 2;
        params.max_passes = 48;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert_eq!(trace.yinyang_cycles, 2);
        assert!(trace.yinyang_prespins > 0);
        assert!(trace.golden_prespins == 0 && trace.causal_antispins == 0);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|p| p.kind == PhaseRotorKind::CrossPhaseYinYang));
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phase_conjugate_autospin_records_and_polishes() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::DegenerateSpectrum, 29);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.record_mzi_phases = true;
        params.use_phase_conjugate_autospin = true;
        params.use_golden_prespin = false;
        params.use_golden_jumps = false;
        params.max_passes = 48;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.phase_conjugate_prespins > 0);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|p| p.kind == PhaseRotorKind::PhaseConjugate));
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn bottleneck_queue_records_damped_quantized_rotors() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::JordanDefective, 31);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.record_mzi_phases = true;
        params.use_bottleneck_queue = true;
        params.use_incremental_bottleneck_cache = false;
        params.bottleneck_pairs = 16;
        params.phase_viscosity = 0.75;
        params.phase_quantization_levels = 256;
        params.max_passes = 48;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.bottleneck_rotations > 0);
        assert!(trace
            .mzi_phases
            .iter()
            .any(|p| p.kind == PhaseRotorKind::Bottleneck));
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn incremental_bottleneck_cache_updates_touched_axes() {
        let case = crate::profiles::generate(32, crate::profiles::Profile::JordanDefective, 41);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.use_bottleneck_queue = true;
        params.use_incremental_bottleneck_cache = true;
        params.bottleneck_pairs = 16;
        params.phase_viscosity = 0.8;
        params.max_passes = 40;
        // This test exercises the pure lazy-incremental path specifically
        // (no periodic full rebuild), so it can assert exactly one refresh.
        params.bottleneck_cache_refresh_period = 0;
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(trace.bottleneck_rotations > 0);
        assert_eq!(trace.bottleneck_cache_refreshes, 1);
        assert!(trace.bottleneck_cache_updates > 0);
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn bottleneck_cache_periodic_rebuild_bounds_the_lazy_staleness_window() {
        // Same case/params as the pure-lazy test above, but with the default
        // period (16): over max_passes=40 that must trigger periodic
        // rebuilds at pass 16 and 32, i.e. refreshes == 1 (initial) + 2.
        let case = crate::profiles::generate(32, crate::profiles::Profile::JordanDefective, 41);
        let mut params = LieSvdPhaseFlowParams::for_n(32);
        params.use_bottleneck_queue = true;
        params.use_incremental_bottleneck_cache = true;
        params.bottleneck_pairs = 16;
        params.phase_viscosity = 0.8;
        params.max_passes = 40;
        assert_eq!(params.bottleneck_cache_refresh_period, 16);
        let ((u, sigma, vt), trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert_eq!(
            trace.bottleneck_cache_refreshes, 3,
            "expected initial + 2 periodic rebuilds"
        );
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn phaseflow_exports_mzi_phases() {
        let case = crate::profiles::generate(16, crate::profiles::Profile::JordanDefective, 17);
        let mut params = LieSvdPhaseFlowParams::for_n(16);
        params.max_passes = 4;
        let phases = LieSvdPhaseFlow::to_mzi_phases(&case.a, params);
        assert!(!phases.is_empty());
        assert!(phases
            .iter()
            .all(|p| p.i < 16 && p.j < 16 && p.i != p.j && p.theta_l.is_finite()));
    }

    #[test]
    fn row_col_phases_into_matches_allocating_version() {
        let a = Array2::from_shape_fn((11, 7), |(i, j)| ((i * 5 + j * 3 + 1) as f64).sin());
        let expected_rows = row_phases(&a);
        let expected_cols = col_phases(&a);

        let mut row_buf = Vec::new();
        let mut col_buf = Vec::new();
        // Call twice with different prior contents to check `.clear()` reuse
        // (not just first-call correctness) matches a fresh allocation.
        row_phases_into(&a, &mut row_buf);
        col_phases_into(&a, &mut col_buf);
        row_phases_into(&a, &mut row_buf);
        col_phases_into(&a, &mut col_buf);

        assert_eq!(row_buf.len(), expected_rows.len());
        assert_eq!(col_buf.len(), expected_cols.len());
        for (got, want) in row_buf.iter().zip(expected_rows.iter()) {
            assert!((got.stress - want.stress).abs() < 1e-15);
            assert!((got.phase - want.phase).abs() < 1e-15);
            assert!((got.norm - want.norm).abs() < 1e-15);
        }
        for (got, want) in col_buf.iter().zip(expected_cols.iter()) {
            assert!((got.stress - want.stress).abs() < 1e-15);
            assert!((got.phase - want.phase).abs() < 1e-15);
            assert!((got.norm - want.norm).abs() < 1e-15);
        }
    }

    #[test]
    fn hot_axes_exact_bound_drops_only_provably_cold_axes() {
        // Row/col 0 is all zeros: its energy is exactly 0, so it must be
        // dropped by any positive pair_tol. Every other axis carries real
        // mass and must survive.
        let mut a = Array2::<f64>::zeros((5, 5));
        for i in 1..5 {
            for j in 1..5 {
                a[[i, j]] = if i == j { 2.0 } else { 0.3 };
            }
        }
        let rows = row_phases(&a);
        let cols = col_phases(&a);
        let hot = hot_axes(&rows, &cols, 5, 1e-9, 0.0);
        assert_eq!(hot, vec![1, 2, 3, 4]);

        // pair_tol == 0.0 still excludes an exactly-zero-energy axis (its
        // energy is not *greater than* 0.0), but nothing else: the filter
        // is strict, so a genuinely nonzero axis of any size survives.
        let all = hot_axes(&rows, &cols, 5, 0.0, 0.0);
        assert_eq!(all, vec![1, 2, 3, 4]);
    }

    #[test]
    fn hot_axes_relative_alpha_screens_low_energy_axes() {
        // One dominant axis (index 0) and four much smaller ones. A relative
        // active_set_alpha should drop the small axes even though their
        // energy is well above the machine-noise pair_tol floor.
        let mut a = Array2::<f64>::zeros((5, 5));
        a[[0, 1]] = 10.0;
        a[[1, 0]] = 10.0;
        for i in 1..5 {
            for j in 1..5 {
                if i != j {
                    a[[i, j]] = 0.01;
                }
            }
        }
        let rows = row_phases(&a);
        let cols = col_phases(&a);
        let exact_only = hot_axes(&rows, &cols, 5, 1e-14, 0.0);
        assert_eq!(
            exact_only.len(),
            5,
            "exact bound alone keeps every axis here"
        );

        let screened = hot_axes(&rows, &cols, 5, 1e-14, 0.3);
        assert!(screened.contains(&0));
        assert!(
            screened.len() < exact_only.len(),
            "alpha screening should drop at least one low-energy axis, got {screened:?}"
        );
    }

    #[test]
    fn active_set_alpha_still_converges_to_machine_precision() {
        // The relative screen is a heuristic, not a certificate, so verify it
        // doesn't quietly break accuracy on a real stress case.
        let case = crate::profiles::generate(64, crate::profiles::Profile::UniformRandom, 23);
        let mut params = LieSvdPhaseFlowParams::for_n(64);
        params.use_bottleneck_queue = true;
        params.active_set_alpha = 0.2;
        let ((u, sigma, vt), _trace) = LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
        let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
        assert!(rel < 1e-10, "rel={rel:e}");
        assert!(orth_u < 1e-10, "orth_u={orth_u:e}");
        assert!(orth_v < 1e-10, "orth_v={orth_v:e}");
    }

    #[test]
    fn adaptive_energy_ratio_viscosity_matches_definition() {
        // Loud pair against a quiet background: gamma close to 1.
        let loud = adaptive_energy_ratio_viscosity(9.0, 1.0);
        assert!((loud - 0.9).abs() < 1e-12, "loud={loud}");
        // Pair exactly at the background level: gamma == 0.5.
        let even = adaptive_energy_ratio_viscosity(2.0, 2.0);
        assert!((even - 0.5).abs() < 1e-12, "even={even}");
        // Both exactly zero: defined as 0.5, not NaN.
        let zero = adaptive_energy_ratio_viscosity(0.0, 0.0);
        assert!((zero - 0.5).abs() < 1e-12, "zero={zero}");
    }

    #[test]
    fn adaptive_viscosity_still_converges_to_machine_precision() {
        // Adaptive viscosity replaces a fixed damping constant with a
        // per-pair heuristic, so verify it doesn't quietly break accuracy on
        // real stress cases where the fixed bottleneck path is already used.
        for profile in [
            crate::profiles::Profile::JordanDefective,
            crate::profiles::Profile::SparseStructured,
        ] {
            let case = crate::profiles::generate(64, profile, 29);
            let mut params = LieSvdPhaseFlowParams::for_n(64);
            params.use_bottleneck_queue = true;
            params.use_adaptive_viscosity = true;
            let ((u, sigma, vt), _trace) =
                LieSvdPhaseFlow::solve_with_digital_polish(&case.a, params);
            let (rel, orth_u, orth_v) = metrics(&case.a, &u, &sigma, &vt);
            assert!(rel < 1e-10, "{profile:?} rel={rel:e}");
            assert!(orth_u < 1e-10, "{profile:?} orth_u={orth_u:e}");
            assert!(orth_v < 1e-10, "{profile:?} orth_v={orth_v:e}");
        }
    }
}
