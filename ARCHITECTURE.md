# Architecture Notes

This document explains what is in this release, why the pieces exist, and what
may be interesting to readers who want to study or extend the code.

The project is deliberately modest in its claims. It does not replace LAPACK,
and it does not claim that analog hardware alone solves high-precision SVD.
The useful idea is narrower and more concrete: keep normal dense `f64` matrices
in memory, but organize the computation as orthogonal rotor schedules that are
friendly to CPU caches today and to analog or photonic rotation meshes later.

The strongest practical motivation is not the easy dense random case. It is the
uncomfortable edge of SVD: degenerate spectra, very ill-conditioned matrices,
non-normal/Jordan-like structure, nearly rank-deficient tails, and cases where
reconstruction error alone does not reveal a damaged `U` or `V` basis.

## What Is New Here

Most dense SVD implementations are presented as either:

- a classical numerical linear algebra routine, such as bidiagonalization plus
  QR/divide-and-conquer polish;
- an eigensolver on `A^T A` or `A A^T`;
- a direct Jacobi/Kogbetliantz-style sequence of local rotations.

This crate explores a different packaging of those ideas:

- `LieSvdMicro` treats `N <= 4` as tiny rotor schedules rather than invoking
  the full polar/Jacobi machinery.
- `LieSvdSmall` avoids `A^T A` as the primary route and instead uses a polar
  factor `A = QP`, followed by Jacobi on the symmetric factor `P`.
- `LieSvdHybrid` treats the matrix through four coupled views: row metric,
  column metric, and simple dual/torsion mirrors. In code this remains ordinary
  `f64`, but pair selection and preconditioning use Lie/Clifford-style rotor
  invariants.
- `LieSvdAnalog` turns the computation into conflict-free layers of local
  `2x2` rotor cells. That gives a concrete schedule resembling an analog
  rotation mesh: many independent local rotations, optional phase quantization,
  and a final digital polish.
- `LieSvdCoreFlow` makes the usual two-sided Jacobi state explicit:
  `core = U^T A V`, with `A` fixed and only the two bases moving. In
  `0.3.0..0.6.0`, this path also includes monotone line-search acceptance,
  optional anti-clustering repellers, and optional guarded warm-starts.
- `LieSvdAdaptive` is the current release dispatcher brain. It uses a cheap
  triage pass to decide when to combine `CoreFlow`, `TopoWarm`, and repellers,
  while keeping `LieSvdSmall` as the normal fast path.
- `kernel_gram` separates symmetric single-domain kernels from nonsymmetric
  bipartite kernels. If `K = K^T`, the solver uses the one-basis
  `K = U Sigma U^T` route; if not, it keeps the two-sided `CoreFlow` route.
- `lie_svd_topowarm` implements a guarded topological warm-start: landmark
  sphere features, stationary/Fiedler-like bipartite graph features, a small
  two-sided power refinement, and Manopt-style orthogonal completion before
  `CoreFlow`.
- `lie_svd_tensortrain` adds the 0.7.0 dimension-lift view: detect whether a
  matrix is close to a chain of `2x2` Kronecker factors, and if so assemble the
  full SVD from the tiny factor SVDs.
- `lie_svd_traceflow` adds the 0.8.0 inverse-Rubik view: start from identity
  bases and rotate `U,V` to maximize the signed diagonal trace projection of
  `core = U^T A V`.
- `lie_svd_quadenergy` adds the 0.9.0 global row/column Clifford audit plus
  local `2x2` rotor coordinates. It separates the user's global four views
  from the local four coordinates of a pair block.
- `lie_svd_phasehealth` adds the 0.10.0 fractal row/column audit. Each row and
  each column is treated as its own local Clifford-like signal with scalar
  mass, vector spread, deterministic phase-delay twist, entropy, and
  row/column disagreement.
- `lie_svd_phaseflow` adds the active phase-locking route. The same
  row/column phase portrait now chooses global phase jumps and targeted unwrap
  rotors, producing a standalone phase-flow SVD result before any optional
  digital polish. In `0.19.0`, PhaseFlow also gets a Layer-0 Golden Global
  Phase Dispersion stage: a deterministic Fibonacci/golden-angle rotor sheet
  applied to row and column axes before local phase relaxation begins.
- In `0.23.0`, PhaseFlow gets the directed counterpart: Causal Anti-Spin.
  When triangular causal bias is high, the layer-0 sheet uses opposite-sign
  row/column rotations instead of isotropic golden dispersion. This targets
  Jordan-like one-way flow rather than balanced standing-wave resonance.
- In `0.24.0`, PhaseFlow gets a multi-layer Cross-Phase Yin-Yang pre-spin.
  It explicitly alternates row and column acts with opposite signs:
  row golden, column antipod, row antipod, column golden. The cycle is annealed
  by the golden ratio and exported as a first-class hardware schedule event.
- In `0.25.0`, PhaseFlow adds state-driven cancellation. Phase-Conjugate
  Auto-Spin mirrors the measured row/column phase delays, while Bottleneck
  Phase Alignment applies damped local rotors to maximum-energy pairs before
  the ordinary active-set pass.
- `lie_svd_complex` adds the `0.20.0` complex-native phase branch. In complex
  storage, a scalar already carries phase, so Layer-0 golden dispersion becomes
  direct U(1) row/column multiplication instead of a real `2x2` surrogate
  rotor. In `0.22.0`, its Hermitian Jacobi route is made stricter by
  recomputing the tracked Gram matrix from the accumulated basis and by adding
  a guarded QR/polar-style polish attempt.
- `lie_svd_engine` adds the 0.22.0 unified dispatcher facade. It does not hide
  the specialist routes; it gives them one `PhasePassport` and one report
  shape.
- `lie_svd_compiler` adds the 0.22.0 hardware schedule compiler. It turns
  real and complex phase events into one MZI/FPGA-friendly layer/channel/angle
  format with JSON export.
- `PhaseSignature` adds the 0.12.0 compact phase passport used by
  `LieSvdAdaptive`: mean stress, max twist, causal disbalance, and entropy gap.

The important engineering theme is separation of concerns:

- geometry chooses rotations;
- `f64` arrays store data;
- orthogonality is preserved by rotor updates;
- high precision is finished by a conservative digital polish path.

## What Has Been Done

By `0.6.0`, the project contains six connected layers:

- a robust baseline SVD (`LieSvdSmall`) based on polar decomposition and
  symmetric Jacobi;
- tiny rotor kernels (`LieSvdMicro`) for `N <= 4`;
- analog/photonic rotation-mesh scheduling (`LieSvdAnalog`);
- explicit two-sided core dynamics (`LieSvdCoreFlow`);
- kernel/topological helpers (`kernel_gram`, `lie_svd_topowarm`);
- adaptive route selection (`LieSvdAdaptive`) that combines the expensive
  pieces only when the matrix triage supports it.

By `0.7.0`, there is a seventh optional layer:

- tensor/Kronecker-chain diagnostics (`lie_svd_tensortrain`) for matrices with
  low tensor-network bond complexity.

By `0.8.0`, there is an eighth diagnostic layer:

- trace/Procrustes navigation (`lie_svd_traceflow`) that frames each local
  rotor as a move increasing `sum(abs(diag(U^T A V)))`.

By `0.9.0`, there is a ninth audit layer:

- quad-energy decomposition (`lie_svd_quadenergy`) for making the global
  row/column/dual Clifford views measurable instead of metaphorical.

By `0.10.0`, there is a tenth diagnostic layer:

- fractal phase-health analysis (`lie_svd_phasehealth`) for measuring internal
  row/column phase stress without changing the matrix storage representation.

By `0.11.0`, there is an eleventh active layer:

- phase-locking SVD (`lie_svd_phaseflow`) that uses phase-health as an actuator
  rather than only a detector.

By `0.12.0`, that active layer is wired into dispatch:

- phase passport triage (`PhaseSignature`) routes repeated/clustered spectra
  and causal/Jordan-like flow through `PhaseFlow` automatically.

By `0.19.0`, the phase route starts one level earlier:

- Golden Global Phase Dispersion (`use_golden_prespin`) applies an initial
  conflict-free sheet of real row/column rotors whose angles come from an
  irrational golden lattice. Conceptually this is a Clifford/phase
  anti-resonance step over all row and column generators; in code it remains a
  list of ordinary `f64` Givens rotations suitable for MZI/photonic schedules.
  The pre-spin is guarded and measurable through `golden_prespins` in the
  `LieSvdPhaseFlowTrace`.

By `0.20.0`, the same phase idea enters complex storage:

- `LieSvdComplex` works on `ndarray::Array2<num_complex::Complex64>`.
- Complex golden pre-spin applies row and column phase factors directly:
  `exp(i k theta_phi)` for row generators and a golden-ratio shifted lattice
  for column generators.
- The complex route exports `ComplexMziPhase` events with `phi_l`, `phi_r`,
  and `theta`, matching the language of photonic phase shifters more closely
  than the real surrogate rotor mesh.
- The original complex branch was intentionally honest about precision:
  reconstruction was strong, but `U` unitarity on dense complex tails was still
  prototype-grade.

By `0.22.0`, the phase ecosystem has one integration layer:

- `PhaseEngine` accepts the main research object families through explicit
  methods: real matrices, complex matrices, 3D tensors, BSS observations, and
  symmetric matrix families.
- `PhasePassport` gives every route a common diagnostic vocabulary: shape,
  family/tensor metadata, mean stress, max twist, causal disbalance, entropy
  gap, chirality, golden resonance, and route hint.
- `HardwareSchedule` compiles real and complex phase events into a targetable
  MZI/FPGA-style format.
- Complex stability is improved by stricter Hermitian tracking and guarded
  QR/polar polish. Current smoke reaches machine reconstruction and `U`
  unitarity around `1e-8`; fully production-grade complex SVD still needs a
  dedicated QDWH or bidiagonal route.

By `0.23.0`, PhaseFlow distinguishes two different layer-0 physics:

```text
balanced/standing-wave stress -> Golden Pre-Spin
one-way triangular/Jordan flow -> Causal Anti-Spin
```

Golden Pre-Spin distributes phase isotropically over row/column generators.
Causal Anti-Spin applies an asymmetric counter-flow: row and column pair
rotors get opposite signs, and the sheet is accepted only when it does not
increase local off-diagonal energy.

By `0.24.0`, this becomes a controllable multi-layer cycle:

```text
cycle m:
  act 1: row golden      +theta_phi * phi^(-m)
  act 2: column antipod  -theta_phi * phi^(-m)
  act 3: row antipod     -theta_phi * phi^(-m)
  act 4: column golden   +theta_phi * phi^(-m)
```

This is the "two families of imaginary units" view made executable. Rows and
columns are not collapsed into one basis; row generators and column generators
receive their own phase acts from opposite sides. The implementation stays
conservative: every act is an ordinary real rotor, every accepted event
preserves orthogonality, and local off-diagonal energy guards rollback.

By `0.25.0`, the layer becomes adaptive rather than only prescribed:

```text
state scan -> phase-conjugate counter-rotor -> bottleneck pair -> damped local rotor
```

Phase-Conjugate Auto-Spin reads the current row/column phases and tries the
opposite phase difference as a Layer-0 rotor. Bottleneck Phase Alignment uses
the strongest pair energy `a_ij^2 + a_ji^2` as the first target in each pass,
which is the phase-flow version of a Gauss-Southwell rule. `phase_viscosity`
keeps exact local angles from overdriving neighboring axes, and optional phase
quantization snaps accepted trial angles to a hardware-like phase grid.

The main engineering result is that the exploratory pieces are no longer only
standalone experiments. They now form an inspectable dispatcher:

```text
matrix -> cheap triage -> Micro | Small | PhaseFlow | CoreFlowTopo | Hybrid
```

`CoreFlowTopo` means:

```text
TopologicalWarmStart + CoreFlow + Repeller + LieSvdSmall polish
```

This is the current "1 + 1 > 2" path. It is intentionally not used everywhere:
on random dense or extreme ill-conditioned matrices it can cost more than it
returns, while on balanced-degenerate smoke cases it improves the residual.

The new tensor route is even more selective:

```text
matrix -> Kron2 chain detector -> KronChain SVD | normal dispatcher
```

It is not enabled as a broad default. Its job is to recognize the special but
important case where the large matrix is really a product of small local
spaces. In that case SVD becomes a chain of tiny rotor SVDs. In the generic
dense case, the detector rejects the path.

The trace route is a different kind of layer:

```text
I, I -> local trace-maximizing rotors -> core = U^T A V -> diagonal readout
```

It is conceptually close to solving the Rubik cube from the clean state toward
the observed matrix. The diagonal entries are not guessed; after the bases have
aligned, they are read from `core_ii` and their signs are absorbed into `U`.
This is the von-Neumann/Ky-Fan trace principle in code form, not a claim that
SVD has been bypassed.

The quad-energy route is not a solver route. The global view is:

```text
A = sum_ij a_ij e_i tensor f_j

view 1: primal A
view 2: row-dual metric A A^T
view 3: column-dual metric A^T A
view 4: dual mismatch / quad spread
```

The bookkeeping view additionally reports:

```text
diag + sym_offdiag + skew + upper/lower flow
```

Together they answer which physical component dominates before we choose a
rotor policy. This is where triangular/Jordan flow, symmetric strain, torsion,
metric degeneracy, and tensor-chain structure become visible as different
signatures.

## Problem-Matrix Focus

This release is intentionally stress-test oriented. The design grew out of
cases where a solver can report a good reconstruction while silently returning
poor singular vectors. That usually happens when errors are hidden in tiny or
repeated singular directions.

The package therefore treats these as first-class diagnostics:

- `degenerate_spectrum`: repeated singular values and rank-like plateaus;
- `extreme_ill_conditioned`: spectra spanning many orders of magnitude;
- `jordan_defective`: non-normal, strongly coupled upper-shift structure;
- `sparse_structured`: local FEM-like coupling patterns;
- `nearly_diagonal`: matrices where a good solver should avoid unnecessary
  disruption of an almost solved basis.

The research claim is deliberately cautious: these methods are interesting for
robustness studies and hardware-aware schedules on difficult matrices. They are
not presented as a blanket replacement for mature production SVD libraries.

## Main Observations

From the current smoke tests:

- `LieSvdSmall` is still the best default for ordinary dense and many difficult
  matrices.
- `AnalogPolished` often improves orthogonality while staying close to the
  baseline timing in small smoke runs.
- `CoreFlow + TopoWarm + Repeller` can significantly improve residuals on
  balanced-degenerate cases, but is allocation-heavy and too slow to enable
  broadly.
- `KronChain` is extremely accurate and light on exact tensor-product inputs,
  but ordinary random, degenerate, sparse, and nearly diagonal matrices should
  not be treated as Kronecker chains unless the residual test says so.
- `TraceFlow` makes the Procrustes/trace objective visible and can strongly
  reduce off-diagonal core energy, but it remains a sweep of local rotations
  and is not a default speed path.
- `QuadEnergy` shows why one universal shortcut is unlikely for generic dense
  matrices: if energy is spread evenly across diagonal, symmetric, skew, and
  metric views, there may be no cheap hidden surface to ride. Its value is in
  detecting special structure early.
- `PhaseHealth` adds the missing row/column internal view: it can distinguish
  low-entropy structured stress from high-entropy dense mixing before the
  solver commits to a geometric route.
- `LieSvdAdaptive` is therefore deliberately conservative: it keeps the fast
  path except when the matrix looks structurally/topologically suitable.
- Reconstruction error alone is not enough; orthogonality and spectrum-tail
  diagnostics remain essential.

## File Guide

### `Cargo.toml`

Defines the public crate `lie_cliffalg_analog_svd`.

Notable choices:

- no LAPACK/OpenBLAS/faer dependency in the release crate;
- `ndarray` plus `rayon` are the only heavy numerical/runtime dependencies;
- release profile enables `opt-level = 3`, fat LTO, one codegen unit, and
  stripped symbols;
- `license-file = "License"` is used to match the included custom license text.

Why it matters:

The crate should build on a normal Linux Rust toolchain without native BLAS
setup. That makes the release easier to test, containerize, and share.

### `Cargo.lock`

Pins the exact dependency graph used for the release.

Why it matters:

This makes Docker, CI, and local checks reproducible with `--locked`. For a
research numerical project, reproducibility is not decoration; small dependency
changes can affect performance, threading behavior, and benchmark output.

### `src/lib.rs`

The public library surface.

It exports:

- `lie_svd`
- `lie_svd_micro`
- `lie_svd_small`
- `lie_svd_hybrid`
- `lie_svd_analog`
- `lie_svd_coreflow`
- `kernel_gram`
- `lie_svd_topowarm`
- `lie_svd_tensortrain`
- `lie_svd_traceflow`
- `lie_svd_quadenergy`
- `lie_svd_phasehealth`
- `lie_svd_phaseflow`
- `lie_svd_complex`
- small helper modules for solvers, metrics, and matrix profiles.

Why it matters:

This file keeps examples and benchmarks from depending on private internals. It
also gives outside users a clean place to start:

```rust
use lie_cliffalg_analog_svd::lie_svd::LieSvd;
```

### `src/lie_svd.rs`

The dispatcher.

Current behavior:

- delegate to `LieSvdAdaptive`;
- keep `LieSvdMicro` for `N <= 4`;
- keep `LieSvdSmall` for ordinary dense, nearly diagonal, Jordan-like, and
  extreme ill-conditioned cases where the geometric stack is not a measured
  win;
- enable `CoreFlow + TopoWarm + Repeller` for balanced-degenerate or
  graph/topological cases;
- use `LieSvdHybrid` as the large fallback above the small/adaptive tier.

Why it matters:

The dispatcher is conservative. Earlier experiments showed that the geometric
preconditioner is interesting, but not automatically faster on small and medium
dense matrices. So the default route favors the measured reliable path, while
the research solvers remain explicitly available.

### `src/lie_svd_adaptive.rs`

Adaptive solver triage.

It computes cheap `O(n^2)` diagnostics:

- off-diagonal ratio;
- diagonal dominance;
- row/column norm coefficient of variation;
- row/column mass mismatch;
- symmetric/skew split, reported as symmetry and transpose torsion;
- diagonal and row/column mass entropy;
- a combined suspicious score.

Routes:

- `Micro`: tiny matrices;
- `Small`: default fast path;
- `CoreFlowTopo`: full synergy path using `CoreFlow + TopoWarm + Repeller`;
- `Hybrid`: large fallback.

The key rule is restraint: high suspicious score alone is not enough. The
dispatcher avoids the expensive geometry on random dense and extreme
ill-conditioned cases unless the row/column/topological views also line up.

`0.28.0` adds `phase_torsion_energy`, `phase_chirality_balance`, and
`phase_entropy` (from `lie_svd_phasehealth::global_phase_invariants`) to the
triage, and a `strong_chirality_torsion` trigger to `should_use_phaseflow`
gated by `phase_chirality_balance > 0.30 && offdiag_ratio > 0.20 &&
phase_torsion_energy > 1.0 && diagonal_dominance < 0.5`. The
`diagonal_dominance` guard is load-bearing, not decorative: an earlier
version without it fired on `sparse_structured` at `N=64`, which has real
structured (asymmetric but bidirectional) skew energy despite being
diagonally dominant and already at machine precision under `Small`, and sent
it through `PhaseFlow` for a 100x wall-clock regression with zero accuracy
gain. `phase_entropy` is exposed on the triage but intentionally *not* used
to gate routing: calibration on `nearly_diagonal`/`uniform_random`/
`structured_stress`/causal-Jordan matrices showed the causal case's
whole-matrix entropy (`~0.52`) sits almost as low as the nearly-diagonal
case's (`~0.49`), because both are sparse band matrices — a blanket
"low entropy means fast-path" rule would have risked misrouting genuine
causal/Jordan flow. When `strong_chirality_torsion` fires, `solve_phaseflow_route`
now takes the triage as a parameter and leans into Causal Anti-Spin
(lowering `causal_antispin_threshold`, raising `causal_antispin_layers`)
rather than relying solely on the separate triangular `causal_bias` metric
that `LieSvdPhaseFlowParams::for_n` gates it on by default.

### `src/lie_svd_micro.rs`

Tiny SVD microkernels for `N <= 4`.

Algorithm shape:

1. `1x1`: direct absolute value plus sign into `U`.
2. `2x2`: one closed-form two-sided rotor cell.
3. `3x3`: fixed `(0,1), (0,2), (1,2)` rotor schedule with residual check.
4. `4x4`: three conflict-free pair layers:
   `(0,1)/(2,3)`, `(0,2)/(1,3)`, `(0,3)/(1,2)`.
5. If the tiny schedule does not finish cleanly, escalate to `LieSvdSmall`.

Why it is interesting:

For very small matrices, a full general solver has more setup cost than useful
work. These kernels model the computation as a local rotor element, which is
also the right primitive for block methods and analog mesh cells.

The important caveat:

The `3x3` and `4x4` paths are fixed schedules with a correctness check, not a
claim that all small SVDs converge in a fixed number of rotations without
fallback.

### `src/lie_svd_block4.rs`

`0.16.0` promotes the `4x4` cell from "tiny special case" to a reusable
macro-rotor warm start.

Algorithm shape:

1. Keep the matrix in ordinary `f64` storage.
2. Select axis quartets:
   - contiguous blocks `[k, k+1, k+2, k+3]`;
   - shifted contiguous blocks;
   - optional power-of-two butterfly quartets such as `[0,4,8,12]`.
3. Extract the local `4x4` core.
4. Solve that cell with `LieSvdMicro`.
5. Apply the resulting left/right `4x4` transforms to the full matrix and to
   the accumulated bases.
6. Accept the block only if global off-diagonal energy does not increase beyond
   a small numerical slack.
7. Finish with the robust digital polish when `solve` is requested.

This is the implementable version of the `SO(4)` / powers-of-two idea. A `4x4`
cell contains six local rotation planes, so it is a richer phase object than a
single `2x2` Givens rotor. For `N >= 5`, however, it remains a warm-start and
block-relaxation mechanism. It does not claim a closed-form formula for the
general SVD problem.

The same module exposes `analyze_block4_signature`. It takes the local skew
part of each contiguous `4x4` block and splits its six bivector components into
self-dual and anti-self-dual triples. This is the concrete diagnostic form of
the `SO(4) ~= SU(2) x SU(2)` intuition: not a new storage representation, but a
measurable phase passport for quartet-level torsion.

### `src/lie_svd_small.rs`

The main robust CPU solver.

Algorithm:

1. Compute an approximate polar factor `A = QP` using inversion-free
   Newton-Schulz iteration.
2. Form `P = Q^T A`.
3. Symmetrize `P`.
4. Diagonalize `P` with classical cyclic Jacobi.
5. Build `U = QV`, `Sigma`, and `Vt`.
6. If the polar factor was not trusted, repair the left basis directly from
   `A V Sigma^{-1}` and a rank-safe Gram-Schmidt fallback.

Why it is interesting:

The solver avoids using `A^T A` as the main computational object. That matters
because `A^T A` squares the condition number. The implementation still uses
plain dense operations, but it is less fragile on several stress profiles than
the earlier normal-equation-based experiments.

Engineering details worth noticing:

- `newton_schulz_polar` reports whether the polar factor should be trusted;
- the Gram-Schmidt fallback never inserts a zero column;
- the Jordan/defective test exists specifically to catch the quiet failure
  where reconstruction looks good but `U` is not orthogonal.

### `src/lie_svd_hybrid.rs`

The Lie/Clifford-inspired preconditioner plus digital polish.

Algorithm shape:

1. Keep a working copy of `A`.
2. Maintain left and right orthogonal bases.
3. Build local row/column metric information.
4. Select hot axis pairs inside active tiles.
5. Apply two-sided Givens rotors.
6. Use a Manopt-style retraction to re-ground the bases on `O(n)`.
7. Solve the smaller cleaned-up core with `LieSvdSmall`.

What "Clifford" means here:

The code does not allocate Clifford multivectors. Instead, the Clifford view is
used as a rule for choosing and coupling rotations. Rows, columns, and simple
dual/torsion mirrors are treated as different views of the same operator. The
actual update is still a fast `f64` rotor.

Why it is interesting:

It explores whether a matrix can be preconditioned by local geometric tension
rather than by a fully global dense sweep. It is not always faster today, but it
is useful as a bridge between numerical linear algebra, Lie group updates, and
hardware-friendly local rotation schedules.

### `src/lie_svd_analog.rs`

The analog/photonic hardware schedule simulator.

Algorithm shape:

1. Arrange all axis pairs into round-robin conflict-free layers.
2. For each pair `(i, j)`, compute a local `2x2` SVD-style left/right rotor.
3. Apply left and right rotations to the working matrix and bases.
4. Optionally quantize rotor angles with `angle_dac_bits`.
5. Extract a sorted approximate SVD.
6. In `solve_with_digital_polish`, solve the remaining core with `LieSvdSmall`.

Why it is interesting:

Analog or photonic chips are naturally good at applying rotations and gains.
This module models SVD as a mesh of local rotor cells, which is closer to how
future hardware might execute the operation. The CPU code is a simulator for
that schedule.

The honest limitation:

The analog mesh is not presented as a standalone high-precision replacement.
The realistic mode is mixed-signal: analog-style rotation preconditioning plus
digital precision audit/polish.

### `src/lie_svd_coreflow.rs`

The explicit core-flow prototype.

Instead of thinking of the algorithm as "changing `A`", this module holds the
input fixed and moves the two orthogonal bases:

```text
core = U^T A V
residual = offdiag(core)
```

Algorithm shape:

1. Start with `U = I`, `V = I`, `core = A`, or optionally with a guarded
   `lie_svd_topowarm` basis.
2. Sweep conflict-free pair layers.
3. For each pair, compute a direct local two-sided rotor plus small row/column
   metric and torsion feedback terms.
4. Add a small soft repeller when two diagonal/sigma estimates are close and
   still coupled by off-diagonal energy.
5. Clamp rotor angles so feedback cannot become a large uncontrolled move.
6. Accept the rotor only through a backtracking line-search that keeps the
   global `offdiag(core)` energy non-increasing.
7. Track the best off-diagonal energy seen so far and roll back to it if the
   final sweep is worse.
8. Finish with `LieSvdSmall` digital polish on the final core.

Why it is interesting:

This is the cleanest expression of the "move the mirrors, not the object"
viewpoint. It also gives a natural place for future double-bracket flow,
annealing schedules, metric probes, or phase-continuous analog constraints.

The important caveat:

`CoreFlow` is a prototype. It is intentionally explicit and inspectable, not
yet a default speed path. The monotone acceptance makes it more physically
disciplined than a free heuristic rotor, but it currently costs extra memory
traffic because trial cores are evaluated during backtracking.

Soft vs hard repellers:

- Soft repellers are potential terms. The public helper
  `repeller_potential(sigma, lambda, eps)` implements the ordered-pair
  Calogero-Moser form
  `lambda * sum_{i != j} 1 / ((sigma_i - sigma_j)^2 + eps)`, with a matching
  gradient helper. The `CoreFlow` pair scheduler uses this idea only when
  `lambda > 0` and the current off-diagonal residual is still above the
  clustered-phase threshold, so the final polish is not fighting an artificial
  separation force.
- Hard repellers are invariants and guards: orthogonal rotor updates, residual
  checks, no zero-column fallback, angle clamps, and digital polish. They
  prevent impossible or numerically unsafe states from silently becoming output.

### `src/kernel_gram.rs`

Kernel and Gram-matrix helpers.

Included kernels:

- `Linear`: `K_ij = x_i dot x_j`;
- `Rbf { gamma }`: `K_ij = exp(-gamma ||x_i - x_j||^2)`.

The important mathematical condition:

- for a symmetric single-domain Gram matrix, `K = K^T`, the left and right
  bases are identical by construction. The correct objective is the spectral
  trace form `max tr(U^T K U)`, and the code routes through a one-basis
  symmetric Jacobi eigensolver;
- for a nonsymmetric square cross-kernel, the row and column domains are
  genuinely different, so the code routes through the two-sided
  `LieSvdCoreFlow` path.

This directly matches the row-angle/column-angle interpretation: a single
domain has one rotor basis seen from both sides; a bipartite domain has two.

### `src/lie_svd_topowarm.rs`

Landmark/topological warm-start for `CoreFlow`.

What it does:

1. Treat rows and columns as two point clouds.
2. Compute stationary masses from the bipartite graph `|A|` using row/column
   absolute sums.
3. Build a cheap Fiedler-like split axis by alternating a normalized
   row-to-column and column-to-row relaxation. This is not an exact Fiedler
   vector; it is a low-cost stress-axis proxy.
4. Pick a few high-stress phase landmarks, then fill the remaining landmark
   budget by a deterministic farthest-point rule.
5. Build cheap sphere-like features: constant mass, stationary mass,
   Fiedler-like axis, distances to landmarks, and fixed pseudo-random probes.
6. Run a tiny two-sided power refinement:
   `U_seed <- A V_seed`, `V_seed <- A^T U_seed`.
7. Retract/complete both thin seeds into full orthogonal bases.
8. Accept the warm-start only if it lowers `offdiag(U^T A V)` versus the
   identity start.

Why it exists:

It is the engineering version of the "diffusion center / intersecting spheres"
intuition. The code does not compute a Laplace-Beltrami eigensystem and does
not claim a closed-form SVD for `N >= 5`. It uses a cheap approximate center of
mass in landmark coordinates to give `CoreFlow` a better starting frame when
the structure is visible enough to help.

Allocation note:

`0.5.0` rewrites the orthogonal completion hot path to reuse scratch buffers
and choose completion basis vectors by residual occupancy instead of allocating
a temporary vector for every candidate basis direction.

### `src/lie_svd_tensortrain.rs`

Kronecker-chain and tensor-train inspired diagnostics.

What it does:

1. Try to split a square matrix as `A ~= B kron C`, where `B` is `2x2`.
2. Estimate the best split by comparing the four half-size blocks of `A`.
3. Recursively continue the split while every local residual stays below the
   configured threshold.
4. If the whole chain reconstructs the original matrix accurately, solve each
   `2x2` factor with `LieSvdMicro`.
5. Assemble the full `U`, `Sigma`, and `Vt` by Kronecker products of the small
   SVD factors.
6. Sort the product singular values and permute the corresponding basis
   columns/rows.

Why it is interesting:

This is the concrete engineering version of the "dimension blow-up, then
phase collapse" intuition. A tensor product expands a few local spaces into a
large matrix. If the matrix still has low tensor bond complexity, the SVD is
not a global dense fight; it is a synchronized chain of small Schmidt/SVD
cuts.

What it does not claim:

It does not make general dense SVD trivial. A random dense matrix will usually
have a large Kronecker residual, and the detector rejects it. The module is a
fast-path and diagnostic for structured tensor-network-like matrices.

### `src/lie_svd_traceflow.rs`

Trace/Procrustes SVD navigator.

What it does:

1. Start with `U = I`, `V = I`, and `core = A`.
2. Sweep conflict-free axis pairs, exactly like local Rubik moves.
3. For pair `(i,j)`, compute the local `2x2` two-sided rotor that maximizes
   the diagonal projection of that block.
4. Accept the move only when
   `sum(abs(diag(core)))` does not decrease.
5. Track the best trace projection and corresponding bases.
6. Read local singular-value estimates from the diagonal of the best core.
7. Finish with `LieSvdSmall` on that core for machine-precision polish.

Why it is interesting:

It gives a clean variational name to the local rotor: a Procrustes/von-Neumann
trace move. Instead of saying only "remove off-diagonal elements", the code can
say "increase the visible diagonal projection until the singular values appear
on the diagonal".

How this maps to the Clifford language:

- scalar view: total diagonal/nuclear projection;
- left vector view: row-basis rotor;
- right vector view: column-basis rotor;
- bivector/torsion view: disagreement between the two off-diagonal entries of
  the local `2x2` core block.

The caveat:

The global trace maximum is an equivalent formulation of SVD, not an easier
closed-form problem. On repeated singular values, the maximum is not a single
point; it is a flat manifold of valid bases inside the degenerate subspace.
That is exactly why repellers, locks, polish, or deterministic schedules remain
relevant.

### `src/lie_svd_quadenergy.rs`

Global quad-view energy audit and local `2x2` Clifford coordinates.

Terminology note:

The phrase "four Clifford views" has two levels in this project:

- **Global views**: row basis units `e_i`, column basis units `f_j`, and their
  dual metric contractions. This is the user's original meaning.
- **Local coordinates**: four scalar coordinates `(E,F,G,H)` of one selected
  `2x2` rotor block.

The module exposes both, but keeps them separate.

What it does:

1. Treat the matrix as a row/column tensor:
   `A = sum_ij a_ij e_i tensor f_j`.
2. Measure global quad-view energies:
   primal offdiag, row-dual metric offdiag, column-dual metric offdiag, and
   dual mismatch.
3. Split the ordinary matrix energy into:
   `diag`, `offdiag`, `sym_offdiag`, `skew`, `upper`, and `lower`.
4. Compute row and column Gram off-diagonal energies:
   `offdiag(A A^T)` and `offdiag(A^T A)` in squared form.
5. Report triangular imbalance:
   whether energy mostly lives above or below the diagonal.
6. For any local `2x2` block, expose the exact Clifford coordinates:

```text
[[p, q], [r, w]]

E = (p + w) / 2      scalar
F = (p - w) / 2      diagonal vector / spectral gap
G = (q + r) / 2      symmetric off-diagonal vector / strain
H = (q - r) / 2      bivector / torsion
```

7. Compute the exact local two-sided rotor angles from all four coordinates.

Why it is important:

Earlier three-coordinate formulas using only trace, symmetric off-diagonal,
and torsion drop `F = (p-w)/2`. That loses the diagonal gap and can increase
off-diagonal energy. The correct local rotor needs all four coordinates.

How this maps to the local rotor:

- `E`: scalar mass / trace center;
- `F`: diagonal-vector gap / local ordering force;
- `G`: symmetric strain / elastic deformation;
- `H`: bivector torsion / phase twist;
- upper/lower split: one-way triangular flow, useful for Jordan-like profiles;

How this maps to the global architecture:

- primal view: the direct tensor `A`;
- row-dual view: `A A^T`, contraction over column units;
- column-dual view: `A^T A`, contraction over row units;
- dual mismatch / quad spread: disagreement between those metric views.

What it says about hoped-for `N` or `N log N` routes:

For a fully generic dense matrix, the audit usually shows energy spread across
all views. That is evidence against a universal cheap route. But for special
classes it can expose the right shortcut: triangular sweep for Jordan-like
operators, torsion-first rotors for skew-dominated sparse systems, Kronecker
chain detection for tensor-product inputs, or symmetric one-basis flow for
single-domain Gram kernels.

### `src/lie_svd_phasehealth.rs`

Fractal row/column phase-health diagnostics.

Why this exists:

The global Clifford tensor picture says:

```text
A = sum_ij a_ij e_i tensor f_j
```

where each row direction `e_i` and each column direction `f_j` is its own basis
unit. The dual views contract over one side of this tensor. That is the
global, four-view picture.

`lie_svd_phasehealth` adds a second question:

```text
what is the internal phase health of row_i and col_j themselves?
```

For each individual row or column, the module reports:

- scalar mass: the mean component;
- vector spread: centered energy around that mean;
- phase-delay bivector proxy: wedge energy between the vector and a one-step
  cyclic delay of itself;
- cyclic gradient energy;
- normalized energy entropy;
- row/column disagreement summaries.

The important caveat:

A single vector does not contain a canonical bivector by itself. A bivector
requires two directions. This module chooses a deterministic one-step cyclic
phase delay as the second direction:

```text
||x wedge delay(x)||^2 = ||x||^2 ||delay(x)||^2 - <x, delay(x)>^2
```

Because `delay(x)` has the same norm as `x`, this is cheap to compute and needs
no heap copy of each row/column. It is a reproducible diagnostic proxy for
internal phase twist, not a coordinate-invariant theorem about every possible
Clifford representation.

Why it is useful:

- high stress with high entropy often marks random or degenerate dense mixing;
- high stress with low entropy often marks structured flow, such as
  Jordan-like one-way coupling;
- near-zero entropy marks nearly diagonal/sparse one-hot rows that should not
  be disturbed by an expensive geometric route;
- row/column gaps can become future dispatcher features.

This module complements `lie_svd_quadenergy`. `QuadEnergy` measures the global
row/column/dual tensor views; `PhaseHealth` measures the internal row and
column signal health before selecting a solver route.

`0.27.0` adds `global_phase_invariants`, four whole-matrix scalars that don't
reduce to any single row or column, distinct from the per-row/per-column
summaries above:

- `global_phase`: a mass-weighted circular mean of every row's and column's
  `phase` (the same deterministic one-step cyclic phase-delay angle used
  internally, now surfaced as a `VectorPhaseHealth` field rather than
  discarded after the twist/entropy computation). A nonzero value means the
  matrix has a consistent directional phase drift that one global pre-spin
  rotor could correct, instead of many local passes discovering it
  piecemeal.
- `torsion_energy`: `H_total = ||skew(A)||_F`, computed directly from `A`.
  This is the *raw* antisymmetric energy; `lie_svd_engine::real_chirality`
  already computed the same quantity normalized by `||A||_F`, so
  `torsion_energy` is the absolute-scale sibling of that ratio, not a
  duplicate.
- `chirality_balance`: reuses `lie_svd_block4::analyze_block4_signature(a).dual_balance`
  directly rather than recomputing the self-dual/anti-self-dual `SO(4)` split
  at a different granularity. It only covers `4 * (n/4)` contiguous rows/cols
  — up to 3 axes at the trailing edge are not included, same caveat as the
  block-4 module itself.
- `phase_entropy`: normalized Shannon entropy of `|a_ij|^2 / ||A||_F^2` over
  the *whole flattened matrix*. This is a new quantity, not a duplicate of
  `PhaseHealthSummary::mean_entropy` above, which is the entropy of one row
  or one column at a time, never the entropy of the entire 2D energy field.

These four are folded into `PhaseSignature::global` and, from there, into
`lie_svd_engine::PhasePassport` as flat fields, so both the compact passport
used by `LieSvdAdaptive` and the unified `PhaseEngine` facade carry them.

### `src/lie_svd_phaseflow.rs`

Active phase-health SVD route.

What changed in `0.11.0..0.13.0`:

`PhaseHealth` is no longer only a detector. `PhaseFlow` uses the same
row/column phase portrait to drive rotations:

```text
A
-> row/column phase scan
-> global phase-jump rotors
-> targeted unwrap rotors on high-stress axes
-> full conflict-free phase-locking sweeps
-> diagonal readout
```

The primary method is:

```rust
LieSvdPhaseFlow::solve(a)
```

This returns the phase-locked result directly. The separate method
`solve_with_digital_polish` exists for final precision cleanup and audit
comparisons, but it is not the definition of the phase-flow solver.

The 0.12.0 dispatcher passport is:

```text
PhaseSignature = (mean_stress, max_twist, causal_disbalance, entropy_gap)
```

It is computed in `O(n^2)` without materializing `A^T A`. `causal_disbalance`
selects Jordan-like one-way flow; high `mean_stress + max_twist` with balanced
row/column mass selects repeated or clustered spectra.

The default adaptive cap for geometric routes is currently `N <= 64`. Above
that, explicit `--phaseflow`/`--coreflow` remain available for research runs,
but `Auto` returns to the conservative `Small`/hybrid routes until
conflict-free batch apply and cached pair energies are implemented.

How rotations are chosen:

- row and column phase profiles produce per-axis stress and phase estimates;
- adjacent axes receive guarded global phase-jump rotors;
- in `0.17.0`, those jumps can be modulated by a golden-angle lattice. This is
  deterministic anti-resonance, not randomness: it avoids replaying the same
  rational phase pattern on every pass;
- high-stress axes are paired with the strongest off-diagonal/phase-coupled
  partners;
- every pass also performs a complete conflict-free sweep so the solver does
  not remain only a local "hot spot" preconditioner;
- proposed rotations are accepted only if `offdiag(U^T A V)` does not
  increase.

`0.17.0` also adds guarded flow surgery. When the phase route reaches a
high-stress plateau or scheduled high-stress checkpoint, it extracts the four
most stressed axes, solves that local `4x4` cell with `LieSvdMicro`, and keeps
the block transform only if global off-diagonal energy decreases. This is the
engineering version of the project's "Perelman-like surgery" vocabulary:
relax a stuck quartet as a richer `SO(4)` cell, then splice it back only if the
whole matrix becomes calmer.

Why the acceptance guard remains:

The guard is not there to make `Small` the real solver. It is the numerical
version of an energy conservation law: a real finite-precision phase jump is
allowed only if it lowers the observable torsion/off-diagonal field. Without
that guard, an aggressive global phase operator can inject energy into the
wrong pair on adversarial matrices.

Current behavior:

- on `N=16 degenerate_spectrum`, raw `PhaseFlow` reaches `rel_recon ~2e-14`
  without digital polish, and `Auto` selects `PhaseFlow`;
- on `N=64 degenerate_spectrum`, `Auto` selects `PhaseFlow` and the polished
  route reaches `rel_recon ~5e-15`;
- on `N=16` and `N=64 jordan_defective`, `Auto` selects `PhaseFlow` because the
  phase passport reports `causal_disbalance ~1`;
- on random, extreme ill-conditioned, sparse, nearly diagonal, and generic
  Kronecker smoke profiles, `Auto` keeps the conservative fast path.

This is the first active implementation of the row/column-as-imaginary-units
idea. It is now a central dispatcher route for phase-passport cases. `0.13.0`
removed the allocation-heavy trial core from the phase-flow hot path.
Rotor acceptance now uses in-place trial application, local two-axis
off-diagonal delta evaluation, and inverse-rotor rollback on rejection. The
matrix is still stored once as ordinary `f64`; the extra Clifford/dual "views"
live in the phase passport, pair schedule, and acceptance law.

`0.14.0` adds a rectangular phase-locking route:

```rust
LieSvdPhaseFlow::phase_lock_rectangular_with_trace(a, params)
```

For an `N x M` operator, rows and columns are treated as genuinely different
Clifford basis families:

```text
row space:    e_0 ... e_{N-1}
column space: f_0 ... f_{M-1}
```

The output uses full rectangular SVD shapes:

```text
U:     N x N
sigma: min(N, M)
Vt:    M x M
```

The current rectangular route is an active phase diagnostic and pre-locking
kernel. It proves that the phase passport and row/column rotor layers do not
require `N == M`; final audit-quality rectangular SVD polish remains future
work.

For large dimensions, `LieSvdPhaseFlowParams::for_n` now caps the active axes
to a high-stress subset and skips full sweeps at `N >= 256`. This is the first
hierarchical active-set form: large matrices pay for `O(N k)` candidate
selection rather than forcing every pair through the phase-flow route.

`0.15.0` adds a compact `PairEnergyCache` inside this active-set path. The
cache is rebuilt once per pass from local pair coupling, row/column stress,
and entropy gap; it then emits a conflict-free active layer. This is not yet a
persistent dependency graph across passes, but it is the first cached rotor
planner and the right interface for future batch/SIMD application.

`0.26.0` adds `hot_axes`, an exact pre-filter that runs before any of the
above pair-candidate builders (`PairEnergyCache::square`, `bottleneck_pairs`,
`BottleneckPairCache::rebuild`, `active_phase_pairs`,
`active_rectangular_corridor_pairs`). The bound is trivial but load-bearing:
no entry of row `k` can exceed `‖row_k‖₂`, and no entry of column `k` can
exceed `‖col_k‖₂`, so

```text
pair_offdiag(i, j) = |core[i,j]| + |core[j,i]|
                   <= min(axis_energy_i, axis_energy_j),
    axis_energy_k = row_norm_k + col_norm_k
```

Any axis with `axis_energy_k <= pair_tol` can be dropped from pair search
without reading `core` at all, and this is exact: it never discards a pair
that could matter. Measured on the stock stress profiles at `N=300`, this
alone changes no accuracy numbers (expected: it is a strict no-op when the
certificate wasn't proving anything) and no measurable wall-clock time either,
because none of the seven synthetic profiles have axes with energy near the
`pair_tol` machine-noise floor. It is a correctness-neutral floor, not a
demonstrated speedup on the current benchmark suite; real payoff is expected
on inputs with actually cold rows/columns (e.g. genuinely sparse or
block-structured operators), which the stress harness does not yet generate.

`LieSvdPhaseFlowParams::active_set_alpha` (default `0.0`) is a second,
explicitly opt-in layer on top of the same `hot_axes` filter: a relative
Strong-Rules-style floor `alpha * max(axis_energy)`, the LASSO/glmnet active-set
screening idea applied to axis energy instead of gradient magnitude. Unlike
the exact bound, this is a heuristic — it can in principle screen out an axis
that a later pass would have wanted, trading a small, empirically-checked risk
of extra passes for fewer candidate pairs. `0.26.0` ships it disabled by
default and validates it with a dedicated accuracy test
(`active_set_alpha_still_converges_to_machine_precision`) rather than claiming
it as a default speedup, matching the restraint policy every other opt-in
mode in this module already follows.

`0.27.0` adds `use_adaptive_viscosity`, an opt-in per-pair replacement for the
fixed `phase_viscosity` scalar in the bottleneck rotor path (both the square
and rectangular routes). Instead of damping every accepted bottleneck rotor by
the same constant, `adaptive_energy_ratio_viscosity(pair_energy,
background_energy)` computes `gamma = P / (P + R)` per pair, where `P` is
that candidate's own `score` and `R` is `mean_axis_stress` for the current
pass (the ambient row/column phase field, already computed every pass). The
name is deliberate: an earlier draft called this idea a "Kalman gain", but
there is no state estimate or covariance carried between passes here, only a
per-pass signal/background ratio — so it is named for what it actually is,
an energy-ratio viscosity, not borrowed terminology from a different
algorithm family. Measured on `N=64 --phaseflow --bottleneck --no-golden-jumps`,
it roughly doubles bottleneck rotation attempts and gives a mixed raw-accuracy
result (worse on `jordan_defective`, slightly better on `sparse_structured`),
so it ships disabled by default; the accuracy test only confirms the final
digitally-polished result still reaches machine precision either way.

`0.28.0` adds `row_phases_into`/`col_phases_into`, buffer-reusing siblings of
`row_phases`/`col_phases` that `.clear()` and rewrite an existing
`Vec<AxisPhase>` rather than allocating a fresh one, wired into the square and
rectangular main pass loops (which recompute row/column phase once or twice
per pass). This was scoped as a targeted fix for a specific claim — that
`PhaseFlow`'s allocation count explains most of its time cost versus `Small`
— and the measurement did not support that claim: on `uniform_random` at
`N=300 --phaseflow --bottleneck`, allocations dropped `43301 -> 40807`
(`~6%`) with no measurable wall-clock change. The same trace ran the full
`624`-pass budget without converging and logged
`cache_updates=53,375,387` from `BottleneckPairCache::update_axes` alone —
tens of millions of `O(n)` rescore operations, on top of hundreds of
thousands of `O(n)` rotor applications and line-search evaluations in
`accept_offdiag_rotor`. At roughly 50-100ns per allocation, 40k allocations
account for low-single-digit milliseconds; the real cost is that `O(n)`
compute repeated far more often than the allocation count. The remaining
per-pass candidate-vector allocations in `active_phase_pairs`,
`bottleneck_pairs`, and `BottleneckPairCache` are still real and un-reused,
but a full buffer-reuse refactor there is deferred rather than rushed, since
this measurement shows it would not be expected to move wall-clock time much.

`0.29.0` fixes the bottleneck the measurement above actually pointed at:
`BottleneckPairCache::update_axes` used to be `O(touched_axes * n)` per
accepted rotor — every touched axis triggered a rescore-and-reheapify against
every other axis, whether or not that pair would ever be examined again. A
binary heap has no cheap `decrease-key`, so the fix is the standard technique
for that situation: lazy invalidation with verify-on-pop. `update_axes` now
just increments `axis_gen[axis]` per touched axis (`O(1)`, no inner loop, no
heap operations). A pair `(i, j)` is stale iff its stored `pair_gen[idx] <
axis_gen[i].max(axis_gen[j])`. Staleness is only resolved in the new
`pop_verified_root`: pop the heap root, and if it's stale, recompute its
score fresh, stamp `pair_gen[idx]` with the current required generation,
re-push it, and pop again — looping until a popped entry is already fresh
(provably the true current max, since a verified pair can't go stale again
until one of its axes is touched further). `pop_conflict_free` now calls this
instead of the raw heap pop.

This is not a free lunch, and the release notes measure the tradeoff
honestly rather than only reporting the win. Pure lazy invalidation cannot
*discover* a pair between two axes that were both cold at the last `rebuild`
— the old eager `update_pair` inserted newly-touched combinations into the
heap on the fly; lazy invalidation only re-verifies pairs already present.
On a real `N=300` trace this cut `BottleneckPairCache` rescoring `294x`
(`53,375,387 -> 181,721`) and wall time `~15%`, but raw (pre-digital-polish)
reconstruction error got measurably worse across all three profiles tested
(`uniform_random`, `degenerate_spectrum`, `jordan_defective`), roughly `2-3x`
in each case — the discovery gap, not a bug.
`LieSvdPhaseFlowParams::bottleneck_cache_refresh_period` (default `16`)
bounds that gap: every N passes, do a full `rebuild` instead of a lazy flush.
With it, rescoring is still cut `153x` (`53M -> 349K`), and two of the three
profiles came out *more* accurate than the original eager cache, not less.
The net wall-clock win with periodic rebuild is real but modest (`7-16%`,
not what a `150x+` rescoring drop alone would suggest): a large share of
`PhaseFlow`'s remaining per-pass cost is the `O(n)` rotor application and
line-search work in `accept_offdiag_rotor`, which this release does not
touch, and profiles with few bottleneck rotations to begin with
(`jordan_defective`: `2284` vs `uniform_random`'s `39965`) see almost no
wall-clock change since the cache was never their dominant cost.

### `src/lie_svd_joint.rs`

Phase-JADE / joint diagonalization prototype.

This module extends the phase-flow idea from one matrix to a family of square
matrices. For symmetric joint diagonalization it searches for a shared
orthogonal basis `V`:

```text
min_V sum_k ||offdiag(V^T M_k V)||_F^2
```

The local step is a Cardoso/JADE-style shared Jacobi rotor, but implemented
with the same engineering rule as `PhaseFlow`:

- no materialized Gram matrix;
- no clone-based trial matrix in the inner acceptance loop;
- local `O(K n)` two-axis delta evaluation for a family of `K` matrices;
- inverse rotor rollback if the ensemble off-diagonal energy does not fall.

In the user's language, this is the first concrete "ensemble phase-locking"
layer: each matrix has its own row/column phase field, but one shared rotor
field tries to bring the whole family into simultaneous diagonal resonance.

`0.15.0` adds the two-sided ensemble route:

```rust
LieSvdJoint::joint_svd(matrices)
```

Its objective is the nonsymmetric family analogue:

```text
min_{U,V} sum_k ||offdiag(U^T A_k V)||_F^2
```

Scope and restraint:

- this prototype targets symmetric `V^T M_k V` joint diagonalization;
- square two-sided nonsymmetric joint SVD is now implemented and tested;
- rectangular two-sided joint SVD currently preserves shapes and orthogonality
  through the diagonal corridor, but extra row/column scheduling remains future
  work;
- the module is useful as a tested foundation for BSS/ICA/JADE-style work, not
  as a replacement for every specialized joint diagonalization algorithm.

### `src/lie_svd_bss.rs`

Phase-BSS / blind source separation prototype.

Pipeline:

1. Input is `channels x samples`.
2. Center each observed channel.
3. Whiten with the robust `LieSvdSmall` path on the channel covariance.
4. Build a small family of lagged covariance matrices.
5. Jointly diagonalize that family with `LieSvdJoint::diagonalize_symmetric`.
6. Return the unmixing matrix, separated channels, and per-channel phase
   coherence.

The current method is closest to a second-order / SOBI-style bridge into
Phase-JADE. It does not yet implement a full fourth-order cumulant tensor ICA
engine. That restraint is intentional: it gives the crate a working BSS route
using already-tested phase rotors before adding heavier cumulant machinery.

The BSS metric layer includes:

- `channel_phase_coherence`: a phase-lock/smoothness score per separated
  channel;
- `estimate_sir_db`: a synthetic benchmark helper that matches separated
  channels to reference sources by absolute correlation.

### `src/lie_svd_tensor.rs`

Higher-order phase SVD / Tucker-style tensor prototype.

The first tensor route targets 3D tensors:

```text
T(i,j,k) -> Core(a,b,c)
Core = T x1 U1^T x2 U2^T x3 U3^T
```

Each mode builds its own Gram matrix and is diagonalized with the robust SVD
path. In the row/column Clifford vocabulary, this means each tensor mode gets
its own axis family; the rotated core measures whether mass concentrates near
the superdiagonal.

This is HO-SVD/Tucker-like, not a full CP/PARAFAC optimizer. The key guarantees
for this release are stable orthogonal mode factors and reconstruction. Future
versions can add iterative CP-style phase locking on top of this core.

### `src/lie_svd_engine.rs`

Unified phase dispatcher facade.

The engine is intentionally a coordinator, not a new numerical monolith. It
keeps the specialist solvers separate and gives them a common diagnostic
surface:

- `PhaseEngine::solve_real`;
- `PhaseEngine::solve_complex`;
- `PhaseEngine::hosvd3`;
- `PhaseEngine::separate_bss`;
- `PhaseEngine::diagonalize_family`.

The shared `PhasePassport` reports stress, twist, causality, chirality, golden
resonance, and route hint. This is the "eagle-eye" layer: one cheap diagnostic
view over the real, complex, BSS, tensor, and joint-diagonalization branches.
In `0.23.0`, the causal part of this passport becomes executable through
PhaseFlow's Causal Anti-Spin layer.

### `src/lie_svd_compiler.rs`

Hardware schedule compiler.

This module converts software phase events into a stable execution-target
format:

```text
layer, channel i, channel j, phi_l, phi_r, theta, theta_l, theta_r, source, kind
```

It supports real `MziPhase` events and complex `ComplexMziPhase` events. The
current export format is JSON via `HardwareSchedule::to_json_string`; the data
model is deliberately simple enough to map later to CBOR, FlatBuffers, FPGA
tables, or photonic MZI control frames.

### `src/lie_svd_complex.rs`

Complex-native phase algebra prototype.

This module is the first branch that stores the matrix itself as
`Array2<Complex64>`. That changes the phase model:

- in the real solver, a phase shift must be represented by a real `2x2`
  Givens rotor over a pair of axes;
- in the complex solver, a single scalar already has a phase, so row and
  column U(1) shifts can be applied directly.

Implemented pieces:

- `LieSvdComplex::solve`;
- `LieSvdComplex::solve_2x2_micro`;
- `apply_complex_golden_prespin`;
- `complex_relative_reconstruction_error`;
- `complex_unitarity_error`;
- `ComplexMziPhase` export events.

The numerical route uses Hermitian Jacobi on the right metric, reconstructs
the left basis as `U = A V Sigma^-1`, and now recomputes the tracked Hermitian
matrix from the accumulated basis after sweeps to avoid false convergence from
manually zeroed pair entries. A guarded QR/polar-style polish can further
factor the provisional `U` and push the residual into a small complex core.

This is now a credible complex research route, but still not a claim of
LAPACK-grade complex SVD. The remaining hard target is uniform machine-tight
`U^H U` on dense I/Q tails, best addressed by true complex QDWH or
Householder/bidiagonal preconditioning.

### `src/bin/stress_cpu.rs`

The self-contained benchmark and stress-test CLI.

It tests:

- `Small`
- `Hybrid`
- `Auto`
- optionally `AnalogPolished`
- optionally `CoreFlow`
- optional `CoreFlow` repeller flags: `--repel-lambda` and `--repel-eps`
- optional Phase-JADE smoke via `--joint`
- optional two-sided Joint SVD smoke via `--joint-svd`
- optional `4x4` macro-rotor route via `--block4`
- optional rectangular phase smoke via `--rect` and `--rect-cols`
- optional Phase-BSS synthetic demo via `--bss-demo`
- optional 3D tensor HO-SVD demo via `--tensor-hosvd`
- optional complex-native SVD smoke via `--complex-svd`
- `--diagnostics-only` to run only `--joint`/`--rect` smoke without the full
  square SVD stress table
- optional topological warm-start flags: `--topowarm`, `--topowarm-rank`,
  `--topowarm-power-steps`, `--topowarm-graph-steps`, and `--topowarm-seed`
- adaptive route diagnostics via `--auto-trace`
- Kronecker diagnostics via `--kron-trace`
- optional `KronChain` solver row via `--kron-chain`
- trace navigator diagnostics via `--trace-nav`
- optional `TraceFlow` solver row via `--traceflow`
- quad-view energy diagnostics via `--quad-energy`
- row/column phase-health diagnostics via `--phase-health`
- active phase-flow solver via `--phaseflow`
- optional phase-flow plus digital cleanup via `--phaseflow-polish`

Profiles:

- uniform random dense matrices;
- degenerate spectra;
- extreme ill-conditioned spectra;
- Jordan/defective-like non-normal matrices;
- sparse structured matrices;
- nearly diagonal matrices.
- exact Kronecker-chain structured matrices.

Metrics:

- relative reconstruction error;
- `U` orthogonality;
- `V` orthogonality;
- spectrum error where the reference spectrum is known;
- allocation count and memory pressure sample.

Why it matters:

Reconstruction error alone is not enough. Some failed SVD paths reconstruct
well because small singular directions hide errors, while `U` or `V` is badly
non-orthogonal. This benchmark keeps those failure modes visible.

### `Dockerfile`

Linux smoke-test and runnable artifact.

Structure:

- builder stage: `rust:1-bookworm`;
- runs `cargo test --release --lib --locked`;
- builds `stress_cpu`;
- runtime stage: `debian:bookworm-slim`;
- copies only the final binary into the runtime image.

Why it matters:

This proves the release does not depend on local macOS setup or Homebrew
OpenBLAS. It also gives contributors a quick reproducible test:

```bash
docker build -t lie-cliffalg-analog-svd .
docker run --rm lie-cliffalg-analog-svd
```

### `.github/workflows/ci.yml`

Basic Linux CI.

It runs:

- `cargo fmt --check`
- `cargo check --release --locked`
- `cargo test --release --lib --locked`
- `cargo run --release --bin stress_cpu --locked -- 32 --analog`

Why it matters:

The CI is intentionally small. It catches formatting, compilation, unit tests,
and a real solver smoke-test without creating a heavy benchmark dependency.

### `.gitignore` and `.dockerignore`

Packaging hygiene.

They prevent local build products and platform files from entering the public
repository or Docker build context.

Why it matters:

Numerical projects can produce large `target/` directories quickly. Keeping the
repository clean makes cloning, reviewing, and container builds faster.

### `README.md`

The public landing page.

It explains:

- what the crate is;
- which solvers are included;
- how to build;
- how to run stress tests;
- how to run Docker;
- what the analog solver does and does not claim.

Why it matters:

README is intentionally shorter than this architecture note. It should help a
new reader decide whether to continue, without forcing them through the whole
research history.

### `RELEASE_NOTES.md`

A short snapshot of the first release.

Why it matters:

This gives Codeberg/GitHub visitors and future contributors a clear boundary:
what is included in `0.1.0`, what is known scope, and what is still research.

### `License`

The included project license text.

Why it matters:

The `Cargo.toml` points to this file directly via `license-file`, avoiding a
mismatch between the manifest and the actual repository terms.

## Why This May Be Useful

For numerical linear algebra readers:

- it is a compact experimental comparison between polar/Jacobi, geometric
  rotor preconditioning, and local analog-style rotor schedules;
- it includes stress profiles that expose orthogonality bugs, not just
  reconstruction error.

For hardware people:

- the analog module turns SVD into independent local pair layers;
- angle quantization gives a hook for DAC/phase-shifter studies;
- the mixed-signal path gives a practical split between cheap physical
  transformations and digital precision cleanup.

For researchers:

- the code makes the "Clifford in the architecture, `f64` in memory" principle
  explicit;
- it is small enough to modify without learning a full production LAPACK stack;
- it preserves failed-edge-case lessons, especially around degenerate spectra
  and non-normal matrices.

## Known Limits

- Square dense matrices only.
- `LieSvdSmall` is still cubic and can become slow at large `N`.
- `LieSvdHybrid` is a research path, not always a speed win.
- `LieSvdAnalog` is a schedule simulator and preconditioner, not a pure analog
  machine-precision SVD.
- `LieSvdCoreFlow` is a prototype route with best-state rollback and digital
  polish, not a proven replacement for the dispatcher default. Its monotone
  line-search is safer but still allocation-heavy compared with the default
  path.
- `lie_svd_tensortrain` only supports chains of `2x2` Kronecker factors in
  power-of-two dimensions. That limitation is deliberate for `0.7.0`: it keeps
  the physical rotor-cell interpretation simple and the residual check clear.
- `lie_svd_traceflow` is currently a diagnostic/prototype route. Its objective
  is mathematically clean, but it still performs many local rotor moves and is
  not wired into the default adaptive dispatcher.
- `lie_svd_quadenergy` is an audit module, not a solver. It currently reports
  structure; future versions may use these signatures for route selection.
- `lie_svd_phasehealth` is also an audit module. Its phase-delay bivector is a
  deterministic coordinate-order proxy, useful for diagnostics and future
  dispatch features, but not a canonical invariant of a lone row vector.
- Explicit hand-written SIMD kernels are not included yet. Release builds rely
  on Rust/LLVM optimization and `target-cpu=native` in Docker.
- No unsafe SIMD kernels are included yet.
- No comparison against system LAPACK is included in this release crate, by
  design; external comparisons should be run separately.
