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

`0.30.4` adds `solve_rectangular` (any `n x d`): a real fix, root-caused
before being written, for a genuine gap found while investigating
`lie_tbl_regress`'s `fit_via_rectangular_svd`. That function used to route
through `lie_svd_phaseflow`'s rotor-based rectangular SVD, which measurably
failed to converge on generic dense data (`~51-96%` reconstruction error).
The cause was not a missing golden pre-spin — `apply_golden_prespin_rectangular`
was already present and invoked there — it was that no rectangular "digital
polish" existed anywhere in the crate: `solve_with_digital_polish` asserts
its input square, because the only exact solver (`solve` above) is
square-only. `solve_rectangular` fixes this the standard textbook way:
QR-reduce `X` (`n x d`) to `R` (`min(n,d) x min(n,d)`) via modified
Gram-Schmidt, run the existing exact square `solve` on `R`, then compose
`U = Q U_r`. `R` and `X` share the same singular values (`Q` has orthonormal
columns), so this doesn't square the condition number the way forming
`X^T X` would — the same principle this module's own doc comment states for
avoiding `A^T A` in the square case, now actually available for rectangular
input too. Returns economy shapes (`U: n x k`, `k = min(n,d)`), not the full
`n x n` the rotor route returns, since for the realistic tabular case
(`n >> d`) a full `n x n` U is mostly wasted storage.

A real bug surfaced and was fixed during this work, worth recording:
`qr_reduce`'s rank-deficiency check initially used an absolute threshold
(`norm >= 1e-300`), which only catches an *exactly* zero pivot. A column
that is numerically dependent on earlier ones — residual norm tiny relative
to the matrix's own scale, but not literally zero (e.g. `1e-14` against
column norms of order `10`) — still passed that check and got normalized,
turning floating-point rounding noise into a direction the code then
treated as orthonormal. This was caught by a test that initially used
deterministic `sin(...)`-based matrices for "generic dense" data — twice,
independently, those formulas turned out to be *accidentally* near-rank-
deficient for the specific index patterns tried (third singular value
`~1e-16`), which is exactly the scenario the bug needed to manifest. Fixed
the check to a scale-relative threshold (`1e-10 * frobenius_norm(mat)`,
rather than an absolute one), and switched the test data to random matrices
(astronomically unlikely to be exactly rank-deficient by accident, unlike a
hand-picked closed form) so a passing test result cannot as easily hide
this class of issue again.

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

### `src/lie_tbl_regress.rs`

Small SVD/eigendecomposition-based ridge regression utility, added `0.30.0`
as the scoped-down real part of a much larger "geometric relational
database" proposal (tables as Clifford `k`-blades, columns as basis
generators, `JOIN` as a geometric contraction, `NULL` as a nilpotent
generator). That idea is **deferred, not rejected**: see
`TECHNICAL_REPORT.md` section 6 and `RELEASE_NOTES.md`'s `0.30.3` entry.
The specific, narrower gap is `JOIN`: it's a discrete key-matching problem,
and "collapsing a shared generator" does not propose a mechanism for
finding matching rows without an index, so it restates the problem rather
than solving it. That gap does not, on its own, rule out representing
tables as Clifford multivectors at all — see `lie_tbl_multivector.rs` below
for the part of the idea that was picked back up and tested directly.

What actually reduces to a real algorithm is standard ridge regression via
the SVD/eigendecomposition of the design matrix (Hastie/Tibshirani/Friedman,
*The Elements of Statistical Learning*, section 3.4.1). `TblRotorRegressor`
implements exactly that, and deliberately does not implement anything else:

1. Center `X` (features) and `y` (target).
2. Build the `d x d` feature Gram matrix `X^T X` (`d` = column count),
   never the `n x d` data matrix itself. This is why the fit only needs a
   square, symmetric solve regardless of how many rows (`n`) the input has.
3. Route through `kernel_gram::solve_kernel`, which already has a tested
   symmetric-eigen path for exactly this shape of matrix — no new numerical
   code was written for the eigendecomposition itself.
4. In the resulting eigenbasis, invert each eigenvalue to get the
   regression coefficients, except: eigenvalues below
   `singular_value_floor * max_eigenvalue` are treated as zero (dropped)
   rather than inverted, and `ridge_lambda` is added to every eigenvalue
   before inversion if set. Both are standard, well-known regularizers for
   the case this module targets: collinear or rank-deficient feature
   columns, where the naive `(X^T X)^-1 X^T y` needs to invert a singular
   or near-singular matrix.

`TblRotorRegressor::eigenvalues` and `rank_used` are exposed directly on the
fitted model rather than hidden, so a caller can tell whether the fit
silently dropped a near-duplicate/collinear feature direction.

`0.30.3` added `fit_via_rectangular_svd`: the same regression, but factoring
the centered `X` directly instead of forming `X^T X`. Originally (`0.30.3`)
it routed through the crate's rotor-based rectangular `PhaseFlow` route,
which turned out not to converge reliably on generic dense data (`~51-96%`
raw reconstruction error, measured on the same near-collinear feature table
the comparison test still uses). `0.30.4` root-caused that (see
`lie_svd_small.rs` above) and switched this function to
`LieSvdSmall::solve_rectangular`. The comparison test
(`gram_vs_rectangular_svd_on_ill_conditioned_features`) result flipped as a
result: `fit_via_rectangular_svd`'s residual (`~3.3e-9`) is now smaller than
`fit`'s (`~1.1e-6`) on the same input — the condition-number-squaring
argument for avoiding `X^T X` holds up once routed through a solver that
actually reaches machine precision on rectangular input. `fit` (Gram-based)
stays the default (more battle-tested, simpler per-feature truncation
semantics), but `fit_via_rectangular_svd` is no longer a known failure.

`0.30.4` also adds `fit_with_bivector_regularization`: anisotropic ridge,
`(X^T X + Lambda) beta = X^T y` with `Lambda` diagonal, built from
`lie_tbl_multivector::CliffordGramMatrix`'s pairwise column bivector norms.
The direction was corrected before implementing: the original proposal
penalized a column *more* as its bivector energy against other columns
grew, but a large wedge norm between two columns means they're close to
*orthogonal* (maximal between perpendicular vectors, zero between
parallel/collinear ones) — high bivector "stress" marks an independent,
well-determined direction, the opposite of what ridge should suppress more.
Implemented with the inverse relationship instead:
`Lambda_jj = lambda0 / (0.1 + stress_j)`, `stress_j` the mean normalized
wedge magnitude (`sin` of the angle between unit-scaled columns) against
every other column. Validated with a held-out train/test A/B against plain
isotropic ridge (`bivector_ridge_beats_plain_ridge_on_anisotropic_collinearity`):
a table with one near-duplicate column pair and two independent columns,
noisy target, 60 training / 60 held-out rows. Bivector-aware ridge won on
held-out RMSE at every one of five regularization strengths tested (`0.01`
to `3.0`), by a margin that grew from `~0.1%` to `~1.9%` as regularization
strength increased — consistent with the mechanism (at low regularization
neither penalty does much; the anisotropic shape matters more as shrinkage
grows). Modest, not dramatic, but a real, direction-consistent, non-cherry-
picked result.

`0.31.0` adds three more pieces, synthesized from a batch of AI-drafted
brainstorming the user handed over directly rather than followed literally.

`fit_dual` is the dual/kernel-trick side of the same ridge problem `fit`
solves: `(X^T X + lambda I) beta = X^T y` and
`beta = X^T (X X^T + lambda I)^-1 y` are the same normal equations solved
in two different bases (the "push-through identity"), and the second is
cheaper when `d > n` — `fit`'s `d x d` feature Gram is then rank-deficient
(`<= n`) by construction, while the `n x n` sample Gram `K = X X^T` is
generically full rank. `K` is exactly
`lie_tbl_multivector::row_scalar_gram`'s construction, reused rather than
recomputed. Because `K + lambda I` is positive definite for any
`lambda > 0`, every eigenvalue is safe to invert directly — no truncation
floor is needed the way `fit`'s un-regularized feature Gram needs one.
Tested against `fit` on a well-conditioned `n=30,d=3` table
(`ridge_lambda=0.3`): coefficients agree to `~1.5e-10`. Tested on a
genuinely underdetermined `n=8,d=20` wide table (`ridge_lambda=1e-6`):
predictions stay finite with `max_err ~2.5e-6` — the test bound (`< 1e-4`)
was set to that measured scale rather than an arbitrary tight number,
because a nonzero ridge deliberately trades exact interpolation for
stability, so machine precision was never the right target here.

`GeometricTabularDispatcher::choose_route` picks among `fit_dual`,
`fit_with_bivector_regularization`, and plain `fit` using signals this
crate already computes. `d >= n` routes to `fit_dual` unconditionally (a
mathematical necessity: `X^T X` cannot be full rank there). Otherwise it
needs to detect "a genuine mix of redundant and independent columns" — the
case `fit_with_bivector_regularization` was built for — and the existing
`column_stress` (mean wedge magnitude against every other column) turned
out to be the wrong signal for this: on `0.30.4`'s own near-duplicate-pair
test table (columns `0`/`1` near-identical, columns `2`/`3` independent of
everything), the redundant pair's *averaged* stress comes out `~0.67`, not
a near-zero value, because it's diluted by the two unrelated independent
columns. A new function, `lie_tbl_multivector::pairwise_column_stress`,
keeps every column *pair* separate instead of averaging: the same
redundant pair's *pairwise* stress is `~0.0196`, cleanly separated from the
independent pairs' `~0.997-0.9998`. `choose_route` routes to
`BivectorRidge` when the most-redundant pair is below
`DEFAULT_REDUNDANCY_THRESHOLD` (`0.15`) and the most-independent pair is
above `DEFAULT_INDEPENDENCE_THRESHOLD` (`0.5`); otherwise `Gram`. Verified
on three synthetic table shapes (wide `d=20,n=8`; the anisotropic-collinear
table above; a well-conditioned `d=4,n=100` table of independent random
columns), each routed to the intended method by both `choose_route` and the
end-to-end `fit`.

`procrustes_rotor`/`transfer_fit` implement orthogonal-Procrustes domain
transfer: `R = UV^T` from the SVD of `X_A^T X_B`
(`LieSvdSmall::solve_rectangular`), then, since `R` is orthogonal
(`X_A ~= X_B R^T`), a model `y ~= X_A beta_A` transfers as
`beta_B = R^T beta_A` without refitting or seeing any target-domain labels.
Scope correction made *before* implementing: the original proposal
described this for tables with *different* row counts, but `X_A^T X_B` is
only defined when the inner dimensions match, which requires `n = m` — a
requirement the proposal's own validation construction
(`X_B = X_A Q + noise`, a row-by-row transform) already implicitly assumed.
Aligning genuinely different-row-count, uncorresponded tables is a
different, harder problem (distribution/covariance alignment, e.g.
CORAL-style whitening-recoloring) and is out of scope here. Tested on a
`200`-row, `4`-column rotated-and-noisy domain, with realistic label noise
(`sigma=0.05`) injected in *both* domains — an earlier draft of this test
left the target domain's `y` noiseless, which let the "from scratch"
baseline hit near-machine-precision error and made any real transfer error
look disproportionately bad by ratio; fixed by calibrating both sides to
the same noise level. Measured: the transferred model's held-out max
residual (`~0.0578`) is close to a model trained from scratch on the target
domain (`~0.0520`, about `1.11x`) — competitive, using zero target-domain
labels.

### `src/lie_tbl_multivector.rs`

Rudimentary Clifford-multivector table representation, `0.30.3`: each
column `j` is assigned its own orthonormal basis generator `e_j`
(`e_j^2 = 1`, `e_j . e_k = 0` for `j != k`, standard Euclidean `Cl(d, 0)`),
and a row is a grade-1 multivector `x_i = sum_j x_ij e_j`
(`RowMultivector`). This is the part of the larger tabular-Clifford-algebra
idea that was deferred (not rejected) in `0.30.0`'s writeup, picked back up
directly here — separately from the `JOIN`-specific gap, which remains
unresolved.

The module's entire content is answering one question precisely: what does
this framing actually add over the `n x n` linear kernel already in
`kernel_gram`? The geometric product of two rows splits into a scalar
(grade-0) and bivector (grade-2) part:

```text
x_i * x_k = x_i . x_k + x_i ^ x_k
```

- The scalar part, `x_i . x_k = sum_j x_ij x_kj`, is **not new** — it is
  exactly `kernel_gram::build_gram(.., KernelKind::Linear)`, and
  `row_scalar_gram` is tested equal to it to `1e-12`
  (`multivector_scalar_gram_matches_linear_kernel`). This is the precise
  place a plausible-sounding claim ("Clifford multiplication already
  encodes relationships between generators/columns, so no accumulation is
  needed") breaks: the geometric product between two *rows* produces a
  **sample-sample** relationship (one number per pair of rows, `n^2` of
  them), while regression needs **feature-feature** relationships (one
  number per pair of columns, `d^2` of them, each summed over every row).
  Those are different index pairs. A single row's self-product,
  `x_i * x_i`, only recovers `||x_i||^2` — getting `sum_i x_ij x_ik` for a
  fixed column pair `(j, k)` requires summing over every row `i`, and that
  sum is `X^T X`, independent of what algebra computes it. The
  anticommuting generator relation `e_j e_k = -e_k e_j` is a fixed property
  of the algebra's signature; it cannot encode one specific dataset's
  column correlations without the data being accumulated into it somehow.
- The bivector part, `x_i ^ x_k`, **is new**: antisymmetric
  (`x_i ^ x_k = -(x_k ^ x_i)`), zero for a row against itself
  (`v ^ v = 0`), and — the useful property — zero whenever two rows are
  exact scalar multiples of each other, regardless of their dot product.
  `total_bivector_energy` sums `||x_i ^ x_k||^2` over every row pair as a
  first, minimal diagnostic: tested at zero on exactly collinear rows and
  positive on rows with real directional spread
  (`bivector_energy_is_zero_for_collinear_rows_and_positive_otherwise`).
  `kernel_gram`'s symmetric dot-product kernel cannot see this by
  construction — two rows related by a positive scalar always look
  "maximally similar" to it.

`0.30.4` adds the dual construction on *columns* rather than rows
(`columns_from_table`) and `CliffordGramMatrix`, which pairs the two column
views: `scalar` is the classical `X^T X` (tested equal to it directly,
`clifford_gram_scalar_part_matches_x_transpose_x`), and `bivector` is a
`d x d` matrix of pairwise column wedge norms,
`bivector[[j,k]] = ||c_j ^ c_k||`. Scope decision, stated rather than left
implicit: this stores wedge *norms*, not the full per-component bivector
*tensor* (every individual `e_j ^ e_k` coefficient for every column pair,
`d` sets of `d*(d-1)/2` numbers) — the norm is what `rho()` and
`lie_tbl_regress`'s bivector regularizer both actually need.
`rho = ||bivector||_F^2 / ||scalar||_F^2` measures what fraction of a
table's column-to-column geometric-product energy is oriented/rotational
rather than colinear: exactly `0` when every column is a scalar multiple of
one common direction (every pairwise wedge vanishes identically — tested),
positive as soon as the columns span more than one direction.

`0.31.0` adds three more pieces to this module.

`from_columns_with_missing` is the concrete, working version of "`NULL` as
a nilpotent generator" that kept coming up in discussion: a literal
`e^2 = 0` generator does not, on its own, give an algorithm for what to do
with the *other* entries of a row that contains one. What actually happens
here is pairwise deletion, generalized to the Clifford-product setting — a
real, standard statistical technique, not a novel one: for each column pair
`(j, k)`, only rows present in *both* columns contribute to their scalar
and bivector product; a row missing in either column contributes a hard
zero to that pair specifically. Tested exact: matches `from_columns` to
`< 1e-12` when nothing is missing
(`from_columns_with_missing_matches_full_data_when_nothing_is_missing`);
zeros every scalar/bivector entry touching a wholly-absent column to
`< 1e-12`, equivalent to that generator being dropped from the algebra
entirely
(`from_columns_with_missing_zeros_out_a_wholly_absent_column`); stays
finite (bivector norms `>= 0`, `rho` finite) under 20% random missingness
on a `40x5` table
(`from_columns_with_missing_stays_finite_under_partial_missingness`).

`pairwise_column_stress` keeps the per-*pair* normalized wedge magnitude
(`stress[[j,k]] = ||c_j ^ c_k|| / (||c_j|| ||c_k||)`) separate, rather than
averaging it over every other column the way `column_stress` (used by
`fit_with_bivector_regularization`'s per-column ridge penalty) does. The
two give materially different answers on the same table: on `0.30.4`'s
near-duplicate-pair test table, the redundant pair's *averaged*
`column_stress` is `~0.67` (diluted by two unrelated independent columns),
while its *pairwise* stress is `~0.0196` — the signal
`GeometricTabularDispatcher` (`lie_tbl_regress.rs` above) actually needs to
detect "one specific redundant pair among otherwise-independent columns"
rather than "mildly correlated with everything".

`temporal_circulation`/`circulation_energy` implement
`Omega = sum_t (x_t ^ x_{t+1})`, the accumulated bivector across every
consecutive row pair in a time-ordered table (same `RowMultivector`
construction as `rows_from_table`, walked in time order instead of
compared all-to-all). For consecutive states related by a fixed rotation
(`x_{t+1} ~= R x_t`), every step's wedge shares the same sign structure, so
the sum accumulates coherently and grows with the number of steps; for an
unbiased random walk (`x_{t+1} = x_t + noise_t`, `noise_t` independent of
`x_t`, mean zero), `x_t ^ x_{t+1} = x_t ^ noise_t` has mean zero
conditional on `x_t`, so the sum stays driftless. The first version of the
validating test used exactly that random-walk construction as the "no
circulation" baseline and got confounded: a cumulative random walk's state
magnitude *grows* over time regardless of rotation, which inflates every
wedge term and produced the *opposite* of the intended result at every step
count tried. Documented as a corrected mistake rather than silently
patched: the baseline was switched to bounded i.i.d. samples instead
(state magnitude stays fixed, so growth in circulation energy can only come
from coherent rotation, not from growing state norm). Measured on a
`400`-step, `3`-column table
(`rotating_process_has_more_circulation_than_bounded_iid_noise`): a
fixed-rotation process's circulation energy (`~79.3`) is `~4.7x` the
bounded-noise baseline's (`~16.9`), comfortably past the `3x` margin the
test asserts.

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

### `src/lie_svd_subspace_jade.rs`

`Subspace-Coupled JADE`, `0.32.0`: generalizes `lie_svd_joint`'s symmetric
route to a family of matrices that do not all share one `n x n` size, only a
subset of their generator axes — e.g. a `3x3` matrix on axes `{0,1,2}` and a
`4x4` matrix on axes `{1,2,3,4}`, sharing `{1,2}`.

**The correction made before implementing.** The originating proposal
described this as: embed every matrix `M_k` (defined on generator subset
`S_k`) into one shared `D x D` ambient space via zero-padding, and find a
single global rotor `R in Spin(D)` there, using per-matrix projectors `P_k`
to select each matrix's active rows/columns
(`min_R sum_k ||offdiag(P_k R P_k^T M_k P_k R^T P_k^T)||_F^2`). That has a
real bug: applying one *shared* `D x D` Givens rotation `G_{ij}` to every
zero-padded `tilde M_k` is only harmless for a matrix that has *both* axes
`i,j` in its support, or *neither*. If a matrix has exactly one of the two
(say `i in S_k`, `j not in S_k`), `G_{ij}` mixes row/column `i` (real data)
with row/column `j` (a padded zero, since that matrix never measured axis
`j`) — fabricating nonzero entries on an axis the matrix has no data for,
which the algorithm then partly "diagonalizes away" against data that was
never there. Concretely, in the module's own test scenario, any rotation
touching pair `(1,3)` — jointly observed only by the second (`4x4`) matrix —
would, under the naive scheme, also perturb the *first* (`3x3`) matrix's
padded row/column `3`, which that matrix has no axis for at all.

The fix matches what the proposal's own informal description already said
in words ("применяется только к тем матрицам, которые содержат обе оси")
but not in its formal `P_k R P_k^T` version: never build a dense padded
embedding. Each matrix `M_k` keeps its own genuine `d_k x d_k` local
accumulator rotor `R_k` (orthogonal by construction — it only ever receives
ordinary Jacobi/Givens updates restricted to its own axes). For a global
axis pair `(i, j)`, the rotation angle is computed once, jointly, from
every matrix that has *both* `i` and `j` — reusing `lie_svd_joint`'s exact
closed-form multi-matrix angle formula
(`joint_symmetric_pair_angle`'s `x,y` accumulation and `0.25*atan2(2sxy,
syy-sxx)` construction, promoted to `pub(crate)` alongside its Givens
helpers — `apply_symmetric_rotor`, `apply_basis_rotor`,
`local_offdiag_sq_for_axes`, `offdiag_sq`, `wrap_jacobi_angle` — and reused
rather than duplicated), just restricted to that data-dependent subset of
the family. That one angle is then applied to each participating matrix at
*its own* local indices for `i` and `j` (via `accept_subspace_rotor`, which
mirrors `lie_svd_joint::accept_joint_rotor`'s line-search/backoff
robustness). A matrix missing either axis is not touched at all for that
step: literally the identity, not an approximation of it — this is what
forces axes shared by two matrices to agree on a rotation while leaving
axis pairs no matrix jointly observes at `theta = 0` (the pair is skipped
before an angle is ever computed, via `observed_pairs_from_owners`, since
there's no data to justify rotating a plane nothing measures both ends of).

One consequence stated explicitly rather than glossed over: there is in
general **no single dense `D x D` orthogonal matrix** whose axis-subset
submatrix recovers each `R_k` (a submatrix of an orthogonal matrix is not
itself orthogonal in general, so assembling one big rotor and slicing it
per-matrix would silently reintroduce the same padding bug in a different
guise). `SubspaceJadeResult` therefore returns the family of per-matrix
local rotors directly — which is also exactly what's useful downstream:
each `R_k` is a normal orthogonal matrix and compiles straight to an MZI
schedule via `lie_svd_compiler::HardwareSchedule::from_orthogonal_matrix`
(the same `0.31.0` path already used for
`lie_tbl_regress::procrustes_rotor`), so the two pieces of generator-coupled
work connect end to end without any new glue code.

**Verification, three tests.**

1. `subspace_jade_reduces_offdiag_energy_on_overlapping_3x3_and_4x4_family`:
   the write-up's own scenario. Confirms `observed_pairs = 8` (`3` internal
   to the `3x3` matrix, `6` to the `4x4`, minus `1` for the shared `(1,2)`
   pair counted once) and that pairs no single matrix jointly observes
   (`(0,3)`, `(0,4)`) are excluded; every recovered local rotor is
   orthogonal to `< 1e-10`; `axis_connected_components` reports one
   component (all five axes transitively linked through the shared pair).
   Its first version asserted near-zero final off-diagonal energy (matching
   `lie_svd_joint`'s same-size test conventions) and failed:
   `initial=6.196`, `final=1.914`, a `~69%` reduction, not the `~1e-8`
   relative drop asserted. Not a bug — a wrong expectation, corrected
   rather than patched around: unlike same-size JADE (where a family built
   as `Q D_k Q^T` for one shared `Q` always has an *exact* joint solution),
   forcing the `(1,2)` plane to rotate by the *same* angle in both matrices
   only has an exact solution when the two matrices' true diagonalizing
   rotors happen to agree on their `{1,2}` sub-block — which two
   independently-random rotors generically don't. Measured across 10 seeds,
   the reduction ratio (`final/initial`) ranges `~0.007` to `~0.45`; the
   test now asserts a real, safely-bounded reduction (`< 0.6x`, comfortably
   above the worst measured seed) instead of an unreachable near-zero
   target.
2. `subspace_jade_keeps_disconnected_axis_groups_independent`: two matrices
   with no shared axes at all diagonalize independently to `< 1e-10` each,
   and `axis_connected_components` reports two separate groups — the
   algorithm degrades to ordinary single-matrix diagonalization when there
   is nothing to couple, rather than doing something spurious with the
   disconnected axes.
3. `subspace_jade_shared_axes_use_information_from_every_participant`: the
   test built specifically to check the coupling is *load-bearing*, not
   just present. Constructs a shared `2x2` block with one true rotor
   `Q_sh`, but a *degenerate* (repeated) eigenvalue `(5, 5)` in the first
   matrix's copy of that block — `M1` alone cannot identify `Q_sh` (any
   rotation within a 2D degenerate eigenspace diagonalizes it equally
   well) — while the second matrix's copy uses a distinct spectrum
   `(2, 9)` and pins `Q_sh` down uniquely on its own. The family converges
   to `~1.0e-15` off-diagonal energy in a single sweep (one rotation — the
   shared pair is the only one with any energy to move, both matrices'
   unique blocks are already diagonal by construction), and the two
   matrices' recovered local rotors, restricted to their own local indices
   for the shared axes, agree with each other to `~3.7e-20` (their product
   is the identity to that precision — no sign ambiguity in this instance).
   That agreement is only possible if the degenerate matrix's rotor was
   actually resolved using the second matrix's data, not left arbitrary
   within its own degenerate eigenspace.

**Scope notes, stated rather than left implicit.** Real-valued symmetric
only, matching `lie_svd_joint`'s symmetric route — no complex/two-sided
variant.

**`0.33.0` stabilization.** Two follow-ups were requested; one was already
done. `axis_connected_components` was already `pub` and tested in `0.32.0`
(see above and its own tests) — no new code was written for it, since
reimplementing an already-shipped, already-tested function would falsely
imply it hadn't existed. The genuinely new piece is scale-balanced
weighting, closing the "no inter-matrix weighting" scope note `0.32.0` had
left explicit: a shared pair's angle previously summed participating
matrices' raw entries unweighted (matching `lie_svd_joint`'s own
convention), so a matrix with much larger entries than its siblings
dominated any pair it participated in.

A new `SubspaceWeighting` enum (`Unweighted`, the default — preserves
every `0.32.0` test's behavior exactly, since the weight is `1.0`
everywhere — and `InverseFrobeniusSquared`) is threaded through
`SubspaceJadeParams::weighting`. Each matrix's weight,
`1 / (||M_k||_F^2 + floor)`, is computed **once**, in `compute_weights`,
from the *original* input, never recomputed mid-sweep — which is exact
rather than an approximation that goes stale, because orthogonal
conjugation preserves Frobenius norm exactly
(`||R^T M R||_F = ||M||_F` for any orthogonal `R`), so `||M_k||_F` cannot
change as the algorithm rotates `M_k`. `floor` is `1e-12` times the
family's own *mean* squared Frobenius norm, not an absolute constant —
the identical fix already made once in this project to
`lie_svd_small::qr_reduce`'s rank-deficiency threshold (see that section
above), for the identical reason: an absolute floor is wrong whenever the
family's own scale isn't known in advance (too loose for a family of
uniformly tiny matrices, too tight for a uniformly huge one).

The weight reshapes every internal energy comparison consistently: the new
`weighted_offdiag_norm`/`weighted_frobenius_norm`/`weighted_pair_offdiag`/
`weighted_local_offdiag_sq` replace the previously-unweighted equivalents
inside `subspace_pair_angle`, `pair_energy_after`,
`accept_subspace_rotor`'s line-search acceptance, and the sweep loop's own
before/after stopping comparison — deliberately, since accepting a
rotation under one objective while checking convergence under a different
one would be internally inconsistent. `SubspaceJadeTrace::initial_offdiag`
and `final_offdiag`, by contrast, always call the original (unweighted)
`total_offdiag_norm`, regardless of `weighting` — kept that way on purpose
so the trace reports a physically meaningful, mode-independent number
rather than an artifact of whichever weighting scheme happened to be
active. Documented consequence: because the algorithm's internal objective
and the trace's reported objective can now differ, the raw
`final_offdiag` is not guaranteed to be `<=` every prior sweep's raw value
once weighting is active — the algorithm is legitimately trading the
down-weighted matrix's fit for the up-weighted one's, and the raw total can
reflect that trade honestly rather than hiding it.

Verified with a direct, measured A/B
(`subspace_jade_weighting_helps_the_small_magnitude_matrix`), not assumed:
two `2x2` matrices sharing both axes (so this isolates the weighting
mechanism itself from the multi-axis machinery), one built with entries
`~1000x` larger than the other, from two independently-random rotations
(no exact joint solution, so there's a genuine trade-off in which single
shared angle to pick). Measured: `Unweighted` drives the large matrix
nearly to machine-precision diagonal (`offdiag^2`: `~1.37e6 -> ~2.0e-7`)
while the small matrix barely moves (`~1.96 -> ~0.380`, `~19%` reduction)
— the shared angle serves the large matrix almost exclusively.
`InverseFrobeniusSquared` cuts the small matrix's residual far more
(`~1.96 -> ~0.0858`, `~96%` reduction) at a real, honestly-measured cost to
the large matrix (`~1.37e6 -> ~1.94e5`, barely reduced) — the test also
asserts this cost directly (`weighted_big_after > unweighted_big_after`),
not just the small matrix's improvement, so a future change that
"improved" the small matrix by accidentally also improving the large one
(i.e. broke the weighting mechanism into a no-op) would be caught.

Two small, low-risk additions rounded out the stabilization pass.
`SubspaceJadeStopReason` (`ReachedTolerance`, `Plateaued`,
`MaxSweepsReached`) is now set explicitly at the sweep loop's two break
points (previously a single combined condition with no way for a caller to
tell which branch fired), tested against `0.32.0`'s own two reference
constructions: the degenerate-eigenvalue case (an exact joint solution
exists) reaches `ReachedTolerance`; the independent-random `3x3`/`4x4` case
(no exact solution) reaches `Plateaued`. And
`subspace_jade_local_rotor_compiles_to_an_mzi_schedule` directly chains a
recovered local rotor through `lie_svd_compiler`'s
`HardwareSchedule::from_orthogonal_matrix` — no new integration code was
needed, since a local rotor is already a normal orthogonal matrix, but the
test makes the connection between the two most recent releases concrete
and checked rather than merely claimed in prose.

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

### `src/lie_svd_benchmarks.rs`

`0.34.0`, extended in `0.35.0`: standard, world-recognized "evil matrix" and
BSS benchmarks, applied to this crate's own solvers rather than left as a
synthetic-only test suite. Prompted directly by the question of whether
standard benchmarks for bad matrices/signals exist in the wider field (they
do) and whether this crate holds up against them (checked here, honestly,
rather than assumed).

**Why not just compare against LAPACK.** This crate deliberately has no
LAPACK/BLAS/faer dependency (see `bin/stress_cpu.rs`'s own doc comment and
`Cargo.toml`), so "compare against a reference SVD" cannot literally mean
"compare against `dgesvd`" without contradicting that design choice. The
module's own doc comment lays out what's used instead, ranked by how
strong a check it gives:

1. **Exact closed-form ground truth** (`pei_matrix`): `P = alpha*I + J`
   (`J` the all-ones matrix). `J`'s eigenvalues are `n` once (eigenvector
   the all-ones vector) and `0` with multiplicity `n-1` (its orthogonal
   complement), so `P`'s eigenvalues are exactly `alpha+n` once and `alpha`
   with multiplicity `n-1` -- independently derivable, not computed by any
   solver here, the strongest available check. A small `alpha` makes this
   simultaneously a genuine degenerate-spectrum stress test: an `(n-1)`-fold
   repeated singular value is exactly where a Jacobi sweep could stall, or
   a rotor might wander freely within the degenerate eigenspace without
   individual singular values coming out wrong. Measured
   (`pei_matrix_matches_exact_closed_form_singular_values`, `n=16,64`,
   `alpha=0.01`): `sigma_max_rel ~8.8e-14` and `~7.8e-13` respectively --
   essentially machine precision on the repeated eigenvalue, direct
   evidence against stalling.
2. **Imposed ground truth**
   (`crate::profiles::Profile::ExtremeIllConditioned`/`DegenerateSpectrum`,
   built from a random orthogonal `U,V` times a chosen diagonal spectrum,
   already present in this crate with an exact `sigma_ref` before this
   module existed). `stress_cpu` already computed and *displayed* the
   resulting relative error, but nothing in `cargo test` ever asserted on
   it -- a real, narrow test-coverage gap, closed here
   (`degenerate_spectrum_profile_recovers_its_imposed_sigma_ref`,
   `extreme_ill_conditioned_profile_stays_within_absolute_error_across_full_spectrum`)
   rather than reimplementing the LAPACK-style "controlled spectrum"
   generator that already existed.
3. **Self-consistency** (`kahan_matrix`, `hilbert_matrix`, and, added in
   `0.35.0`, `frank_matrix`/`forsythe_matrix`/`cauchy_matrix`): no external
   ground truth is available, or -- for `hilbert_matrix`/`cauchy_matrix`
   past the size where their condition number exceeds `f64`'s representable
   range -- no ground truth is even representable in double precision.
   Orthogonality of the recovered bases and reconstruction accuracy are
   what's actually checkable; claiming machine-precision recovery of
   singular values smaller than the matrix's own rounding error would be
   false regardless of which solver computed them.
4. **Known asymptotic clustering** (`parter_matrix`, added in `0.35.0`):
   not a closed form for individual singular values, but a real, citable
   fact from the literature (Parter's own result; also a standard example
   in Trefethen & Bau's *Numerical Linear Algebra*) -- almost all singular
   values cluster near `pi` as `n` grows. See the `0.35.0` notes below for
   the measured fractions.

**A test that was corrected after measuring, not before.** The first
version of the `ExtremeIllConditioned` check (`kappa=1e18`, deliberately
beyond `f64`'s `~1e16` representable range) asserted that the top `3/4` of
the sorted spectrum -- the same quartile split `metrics::compute` itself
uses for `sigma_tail_rel` -- would be accurate to a tight relative
tolerance, assuming a clean break between "recoverable" and
"unrecoverable" singular values. Measured directly: no such break exists.
Relative error grows *smoothly* as singular values shrink, because what's
actually bounded is the **absolute** error -- `~1e-14` down to `~1e-17`,
essentially constant across the entire spectrum at every index checked,
consistent with `f64` rounding noise on a `sigma_max=1` matrix -- and
relative error is just that near-constant absolute error divided by an
ever-shrinking denominator, so it necessarily explodes once a singular
value drops below the noise floor, regardless of where in the sorted list
that happens to fall. The test was rewritten
(`extreme_ill_conditioned_profile_stays_within_absolute_error_across_full_spectrum`)
to check absolute error (`< 1e-9`, ample margin over the measured `~1e-14`
scale) across the *entire* spectrum -- the honest thing to claim -- plus
tight relative error only for entries where `want > 1e-6`, i.e. still well
above the noise floor, where "relative error" remains a meaningful notion
at all.

**Hilbert matrix: measuring where "no ground truth" becomes "no ground
truth exists".** `hilbert_matrix`'s condition number grows roughly like
`e^{3.5n}` (the classical asymptotic result for this matrix) and exceeds
`f64`'s representable dynamic range beyond `n~13`; no solver can recover
its smallest singular values there, a fact about the problem, not this
crate's implementation. What was actually measured, not assumed: does the
solver degrade gracefully elsewhere once that happens? At `n=14,16`, where
the true smallest singular value underflows toward numerical noise (the
computed `sigma_max/sigma_min` ratio itself becomes an ill-defined,
near-`f64::MAX` number at these sizes -- a symptom of the underflow, not a
new failure mode), `orth_u`/`orth_v`/`rel_recon` all stay at `~1e-14`,
statistically indistinguishable from the well-conditioned `n<=12` range
(also measured at `~1e-14` to `~1e-15`). The reconstruction stays accurate
because it's dominated by the well-represented large singular values, not
the underflowed small ones.

**Amari performance index** (`amari_index`; Amari, Cichocki & Yang, *A New
Learning Algorithm for Blind Signal Separation*, NeurIPS 1996): the
standard permutation/scale-invariant metric for scoring a BSS/ICA global
system matrix `g = unmixing @ mixing`. Complements, rather than replaces,
`lie_svd_bss`'s existing `estimate_sir_db` -- a raw `||g - I||`-style error
would be meaningless here, since recovered sources are only ever
identified up to which-source-is-which and their sign/scale, exactly the
ambiguity this index is invariant to (`0` exactly for a scaled permutation
matrix, tested directly in `amari_index_is_zero_for_a_scaled_permutation`).
Applied (`amari_index_is_small_after_bss_on_ill_conditioned_mixing`) to a
synthetic mixing matrix built the same controlled-spectrum way as the
profiles above (`sigma = [1,1,1,1e-7]`, condition number `1e7`, matching
the "near-collinear sensor channels, `kappa > 1e6`" case from the
originating question) and separated with the existing `LieSvdBss`: the
Amari index measurably improves (`~0.295 -> ~0.192`) -- a real,
non-trivial improvement, reported as the moderate result it actually is,
not inflated into a claim of near-perfect separation at a condition number
this challenging for a covariance/lagged-statistics method.

**Explicitly scoped out, stated rather than silently dropped:**
SuiteSparse/Matrix Market and Cardoso's own JADE/SOBI EEG/MEG benchmark
datasets (both would require network downloads, breaking this project's
established offline-reproducible `docker build --no-cache` pattern);
Trefethen pseudospectra (a genuinely relevant diagnostic for non-normal
matrices, but a resolvent-norm-over-a-complex-grid computation is a
visualization tool, not a pass/fail correctness check, and a substantially
larger undertaking than this pass's scope, still in the backlog).
Frank/Forsythe/Parter/Cauchy, originally scoped out in `0.34.0` for the
same "left for a future pass" reason, were added in `0.35.0` (below).

**`0.35.0` additions: the rest of the Higham set, plus a parametric BSS
grid.**

`frank_matrix` (upper Hessenberg, `det=1`, `F[i,j]=n-j` for `j>=i`,
`F[i,i-1]=n-i` on the subdiagonal, `0`-indexed): famous for eigenvalues in
ill-conditioned reciprocal pairs, but that's a *nonsymmetric eigenvalue*
fact, not a singular-value one, so it gives this crate's SVD solvers no
closed form to check against either -- self-consistency is what's actually
available, same as `kahan_matrix`. Measured at `n=16,32,64`:
`orth_u`/`orth_v` up to `~9e-14`, `rel_recon` up to `~7e-15` -- no sign of
difficulty despite the matrix's famously ill-conditioned eigenvalues.

`forsythe_matrix(n, lambda, alpha)` (a Jordan block with `lambda` on the
diagonal, `1` on the superdiagonal, perturbed by one small entry `alpha` in
the bottom-left corner): deliberately close to defective/non-diagonalizable
-- a bare Jordan block has one eigenvalue with a single eigenvector, and
the corner perturbation only barely fixes that (`n` distinct eigenvalues,
the `n`-th roots of `alpha`, but still extremely sensitive to further
perturbation). Measured (`lambda=0`, `alpha=1e-6`) rather than assumed:
`LieSvdSmall::solve` -- polar decomposition plus Jacobi, not an eigenvalue
algorithm -- reaches essentially *exact* results (`orth_u`/`orth_v`/
`rel_recon` at literal `0` for `n<=32`, still `~1e-16`/`~5e-23` at `n=64`).
Worth stating plainly: this construction's near-defectiveness is exactly
what makes *eigenvalue* algorithms struggle, and a polar-decomposition-based
SVD route simply isn't exposed to that particular failure mode the same
way.

`parter_matrix` (`P[i,j] = 1/(i-j+0.5)`): the one addition checked against
an actual citable fact from the literature rather than only
self-consistency -- Parter's own result (also a standard Toeplitz-matrix
example in Trefethen & Bau, *Numerical Linear Algebra*) that almost all
singular values cluster tightly near `pi` as `n` grows.
`parter_matrix_singular_values_cluster_near_pi` measures the fraction
within `0.05` of `pi` directly: `13/16` (`81.25%`), `29/32` (`90.6%`),
`61/64` (`95.3%`) at `n=16,32,64` -- increasing with `n`, exactly the
asymptotic clustering the literature describes, not a coincidence at one
size. The test's threshold (`>75%`) sits safely under the worst
(smallest-`n`) measured fraction.

`cauchy_matrix` (`C[i,j] = 1/(i+j+2)`, the default `x=y=1..n` case of
`gallery('cauchy', x, y)`): symmetric PD and, like the Hilbert matrix,
extremely ill-conditioned by construction, but with a different entry
structure -- a second, independently-constructed instance of the same
graceful-degradation question `hilbert_matrix` answers.
`cauchy_matrix_degrades_gracefully_past_double_precision_limits` measures
`n=6,10,16`: `kappa` grows past `f64`'s representable range by `n=16`
(the same underflow-in-the-ratio symptom seen with Hilbert), while
`orth_u`/`orth_v`/`rel_recon` stay at `~1e-14` throughout, unaffected.

`amari_index_improves_across_a_channel_by_condition_number_grid` extends
the single `kappa=1e7` BSS check to a `channels in {4,8} x kappa in
{1e3,1e5,1e7}` grid (`6` cells, each its own independent random seed --
`synthetic_sources` generalizes the single hand-written `4`-channel source
set from `amari_index_is_small_after_bss_on_ill_conditioned_mixing` to
arbitrary channel counts via a `i % 4` pattern cycle). Separation improves
the Amari index at all `6` points, but measured, not smoothed into a
cleaner story than is actually there: the improvement is *not* monotonic in
`kappa` (`channels=4` lands worse at `kappa=1e5` than at `kappa=1e7`) --
expected sampling variation with one seed per cell, reported as such rather
than glossed over, and not asserted as a trend the test would need to
enforce (the test only asserts `after < before` at each cell, which is what
was actually measured to hold everywhere).

**`0.36.0`: first cycle of a multi-release robustness/frontier-benchmark
program.** Scope decided explicitly before writing code: no LAPACK/MPFR/Arb
dependency added to this crate (a separate, isolated comparison harness
with its own `Dockerfile`, depending on those, is planned as its own later
cycle in the same program — kept out of the main crate's dependency tree on
purpose); "canonical symplectic drift" (`Sp(2n)`) was replaced with a
*unitary* drift test, since this crate has no symplectic-group structure
anywhere to test, but does have a genuine unitary (`U(n)`) complex branch
(`lie_svd_complex.rs`).

- `vandermonde_matrix` (equally spaced nodes `x_i=i+1`) and `ginibre_matrix`
  (plain i.i.d. Gaussian, deliberately non-normal): self-consistency only,
  same tier as `kahan_matrix`. Vandermonde's condition number was measured
  (`~9.5e8` at `n=8` to `~5.6e15` at `n=12`) rather than compared to
  Hilbert's own growth rate by name -- an earlier draft of the doc comment
  claimed Vandermonde was "worse than Hilbert", caught as unverified before
  merging (that specific rate comparison isn't reliably known here) and
  replaced with only the measured numbers for this matrix.
- `marchenko_pastur_matrix`: a rectangular i.i.d. Gaussian matrix for
  testing the Marchenko-Pastur edge law (1967) -- as `cols/rows -> 1`,
  singular values concentrate near `sqrt(rows) +/- sqrt(cols)`.
  `marchenko_pastur_upper_edge_matches_prediction` computes orthogonality
  and reconstruction directly (`u` has orthonormal columns, `vt` is square
  orthogonal -- the shape `metrics::compute` assumes for square input
  doesn't apply to `LieSvdSmall::solve_rectangular`'s output, so this test
  doesn't route through that helper) and checks the *upper* edge tightly
  (`<10%`, measured `~2.5-4%` off across three aspect ratios). The *lower*
  edge only gets a loose sanity bound (`min_sigma < max_sigma`) --
  finite-size fluctuations at the lower MP edge are a known, substantially
  larger effect than at the upper edge, a real RMT fact rather than a
  solver artifact, so a tight quantitative claim there wouldn't be honest
  at these sizes with one seed per case.
- `extreme_dynamic_range_matrix_stays_finite_and_accurate` and
  `subnormal_scale_matrix_stays_finite_and_accurate` test a specific,
  named numerical risk rather than a generic "big/small numbers" worry:
  `lie_svd_small::newton_schulz_polar` scales its input by
  `1 / frobenius_norm(a).max(1e-300)`. For a matrix with entries in `f64`'s
  *subnormal* range (below `~2.2e-308`), squaring a single entry (any
  Frobenius-norm computation) underflows to exact `0.0` well before the
  norm as a whole would -- a concrete, identifiable failure mode, not a
  hypothetical one. Measured: the `.max(1e-300)` floor absorbs it cleanly
  -- `orth_u`/`orth_v`/`rel_recon` all come out as *exact* `0.0` on the
  tested `6x6` subnormal case, and recovered singular values stay correctly
  in the subnormal range. Separately, a `~1e-150` to `~1e150` dynamic-range
  matrix (`kappa~1e300`, an ambient scale that never itself risks
  overflow/underflow the way subnormal entries do) stays fully finite with
  no measurable accuracy loss (`orth`/`rel_recon` at `~1e-15` to `~1e-16`);
  singular values below `~sigma_max * 1e-16` collapse to exact `0.0`, which
  is a representability limit of the *assembled dense matrix itself* (no
  bits left to hold a contribution 300 orders of magnitude below the
  dominant term), not something attributable to the solver.
- `orthogonality_drift_stays_small_after_ten_million_rotations` and
  `complex_unitarity_drift_stays_small_after_ten_million_rotations`: `1e7`
  sequential random-angle Givens (real) or `U(2)`-parametrized unitary
  (complex, `c=cos(theta)` real, `s=sin(theta)*e^{i*phi}`, applied as
  `[[c,-conj(s)],[s,c]]`, unitary by construction for any `theta,phi`)
  rotor updates to an `8x8` identity basis. Directly relevant rather than a
  generic stress test, since composing long sequences of individually
  small rotor updates is the operation this crate's whole architecture
  (PhaseFlow, Phase-JADE, Subspace-JADE, the MZI compiler) is built from.
  Measured drift: `~1.4e-11` real, `~3.7e-12` complex (reusing
  `lie_svd_complex::complex_unitarity_error`, already tested elsewhere,
  rather than a new metric), both in well under a second -- no sign either
  branch accumulates meaningful drift at this scale, and no sign the
  complex branch is worse than the real one.

**`0.37.0`: cycle 2 -- Hansen's ill-posed inverse-problem benchmarks.**
Scope decided explicitly, not left implicit: `heat_problem`,
`phillips_problem`, and `shaw_problem` are **not** bit-exact reproductions
of P.C. Hansen's `Regularization Tools` MATLAB source
(`heat.m`/`phillips.m`/`shaw.m`), which wasn't available to verify against
here. Each is instead built from `fredholm_first_kind` (a general 1-D
first-kind Fredholm discretizer: midpoint quadrature on `n` nodes from an
explicit kernel and a known "true" solution, returning `(A, x_true, b)`
with `b = A.dot(x_true)` **exactly** -- a deliberate "inverse crime": the
question these benchmarks ask is whether spectral truncation recovers a
*known* answer, not whether the crate solves a real-world inverse problem
with unknown ground truth) using only the part of each classical
construction confident enough to state exactly:

- `heat_problem`: the textbook 1-D heat kernel (Gaussian fundamental
  solution, `K(x,y,t) = exp(-(x-y)^2/(4t)) / sqrt(4*pi*t)`) -- stated with
  high confidence, a standard PDE fact independent of Hansen's own
  discretization choices.
- `phillips_problem`: Phillips's own well-known closed-form kernel
  `phi(x) = 1+cos(pi*x/3)` for `|x|<3` (D.L. Phillips, 1962) -- also high
  confidence, a widely-cited textbook formula.
- `shaw_problem`: the `(cos+cos)^2 * sinc^2` diffraction-kernel *shape* --
  stated with the least confidence of the three; the doc comment says so
  directly rather than presenting it with the same certainty as the other
  two.

In every case the right-hand side is forward-generated from a known smooth
solution rather than reproducing any memorized closed-form RHS formula --
avoiding a specific, checkable risk (getting an unverified formula subtly
wrong) rather than an abstract one.

`truncated_svd_solve` (`x_hat = V diag(1/sigma_i for sigma_i >
floor*sigma_max, else 0) U^T b`) is the same truncation idea
`lie_tbl_regress::TblRegressParams::singular_value_floor` already
implements for regression, reused by concept (not by calling that
regression-specific code) for a genuinely different domain. Two tests
demonstrate the textbook regularization story rather than merely asserting
a threshold: `severely_ill_posed_problems_need_spectral_truncation_to_avoid_blowup`
measures `heat`/`shaw` (singular values decaying to exact `0.0` well within
the `n=64` spectrum) at a well-chosen truncation floor (`1e-9`: `heat`
`~4.7e-5` error, `shaw` `~1.0e-4`) against essentially no truncation
(`floor=0`: `heat` explodes to `~78x` the true solution's own norm, `shaw`
to `~73x`) -- naive full-rank inversion of a smoothing operator is
numerically catastrophic, not just imprecise, which is the actual point of
this problem class. `moderately_conditioned_inverse_problem_needs_no_truncation`
is the deliberate contrast: `phillips` (`kappa~2.9e5`, no singular values
collapsing to exact zero) recovers the true solution to `~4.8e-11` even
with *no* truncation at all -- not every hard-looking inverse problem
needs regularization, and the test confirms this crate's SVD correctly
tells the two regimes apart rather than needing truncation applied
uniformly out of caution.

**`0.38.0`: cycle 3 -- a quantum many-body benchmark with a genuine
closed-form ground truth.** Scope decision, made before writing any code:
the originally proposed multi-site Hubbard model, built in second-
quantized Fock space, carries real risk of subtle fermionic-sign bugs that
would be hard to catch without an external reference -- so this uses the
2-site Hubbard dimer instead, restricted to the `N=2, S_z=0` sector (one up
electron, one down electron), a small (`4x4`), standard, exactly-solvable
textbook model.

`hubbard_dimer_hamiltonian(t, u)` and its closed-form spectrum
(`hubbard_dimer_eigenvalues`) were **derived from scratch in this session**,
not recalled from a possibly-misremembered source, via two independent
arguments that were then cross-checked against each other:

1. At `u=0`, the up- and down-electron positions are independent
   single-particle two-level (hopping) problems, so
   `H = H_up (x) I + I (x) H_down`, giving eigenvalues `{-2t, 0, 0, 2t}`
   directly (sums of `+-t` from each independent factor).
2. For general `u`, a symmetric/antisymmetric block decomposition of the
   full `4x4` matrix: the fully antisymmetric combination of the two
   singly-occupied basis states decouples completely (exact eigenvalue
   `0`, any `t,u`); the antisymmetric combination of the two
   doubly-occupied basis states also decouples (exact eigenvalue `u`); the
   remaining `2x2` block gives `u/2 +/- sqrt((u/2)^2 + 4t^2)`.

Both derivations agree at `u=0` (`{-2t,0,2t}` matches `{0,u,u/2-gap,u/2+gap}`
at `u=0`, where `gap=2t`), which is itself the first check that the
general-`u` closed form is self-consistent, not merely convenient.
`hubbard_dimer_matches_its_exact_closed_form_spectrum` then verifies the
closed form numerically against `lie_svd_small::eigh_jacobi_full` (promoted
from a private `fn` to `pub(crate)` specifically for this -- a genuinely
different code path from the closed-form arithmetic, classical cyclic
Jacobi rather than algebra) across four `(t,u)` pairs including negative
`u`: measured differences at or near machine precision (`<1e-14`)
throughout, checked at `<1e-9` margin.

The actual point of the benchmark is
`hubbard_dimer_resolves_the_exact_near_degenerate_gap`: at `u=1e-12`, the
`0` and `u` eigenvalues are close but genuinely distinct -- a real,
physically motivated near-degenerate gap (the frontier-benchmark proposal's
own "energy gaps `Delta E ~ 1e-12`" case, but on a matrix with an
independently verified exact answer rather than an opaque one). Measured:
both resolved as distinct, not collapsed into a single value; the tiny
eigenvalue comes out `~1.0000831e-12` against the exact `1e-12`, a `~8e-5`
relative error. Reported honestly rather than tightened to look better:
this is *not* full double-precision relative accuracy at this extreme gap
scale, because the absolute-precision floor set by the matrix's other,
order-`1` entries is `~1e-16`, which is already `~1e-4` relative to a
`1e-12`-scale eigenvalue -- consistent with what was measured, not a
solver defect.

### `src/lie_svd_lyapunov.rs`

`0.39.0`: Lyapunov spectrum extraction via the standard "continuous QR"
method (Benettin, Galgani, Giorgilli & Strelcyn, 1980; Shimada & Nagashima,
1979) -- the first cycle of the robustness/frontier-benchmark program that
adds a genuinely *new* numerical subsystem, rather than testing an existing
solver against a hard input (`lie_svd_benchmarks`'s whole scope through
`0.38.0`).

**The method.** For a flow `dx/dt = F(x)`, Lyapunov exponents describe the
exponential growth/decay of infinitesimal separations, governed by the
variational equation `dPhi/dt = J(x(t)) Phi`, `Phi(0) = I`. Solved naively,
`Phi(t)`'s singular values separate exponentially and the matrix over/
underflows within a handful of Lyapunov times for any genuinely chaotic
system -- the standard fix, implemented here: periodically QR-decompose
`Phi` (`Phi = Q R`, `R` upper triangular with non-negative diagonal by
construction of the Gram-Schmidt process used, `qr_nonneg_diag`), replace
`Phi` with the orthogonal `Q`, and accumulate `log(R_ii)` for each `i`
across the whole run; after total time `T`, `lambda_i = (1/T) sum(log
R_ii)`. `Phi` is propagated by applying this crate's own RK4 stepper to the
*augmented* system `d(x,Phi)/dt = (F(x), J(x) Phi)` -- state and full
tangent frame integrated together, Jacobian evaluated at each of the 4 RK4
stages' own intermediate state rather than a single frozen Jacobian per
step (the standard, full-accuracy version of this method, not a cheaper
approximation).

**Scope decision, made before writing code.** The originating proposal
named both Lorenz-96 and the Kuramoto-Sivashinsky (KS) equation. KS is a
4th-order stiff nonlinear PDE: naive explicit time-stepping on a 4th
spatial derivative is numerically unstable, so a correct implementation
needs a proper spectral (Fourier) discretization *and* a stable implicit-
explicit integrator (e.g. ETDRK4) -- substantially more numerical-methods
risk than could be adequately verified in this cycle's budget. Scoped down
to Lorenz-96 only (a finite-dimensional chaotic ODE system, where RK4 is
directly adequate); KS is deferred, stated explicitly, not silently
dropped.

`lorenz96_rhs` (E. Lorenz, 1996): `dx_i/dt = (x_{i+1}-x_{i-2}) x_{i-1} -
x_i + forcing` on a periodic `K`-site lattice, `forcing=8.0` the standard
value cited in the original paper as producing chaos. `lorenz96_jacobian`
is its exact analytic Jacobian (`k>=5` required to avoid index aliasing
among `{i,i+1,i-1,i-2}` on a small periodic lattice), verified directly
against a central-finite-difference approximation of `lorenz96_rhs`
(`lorenz96_jacobian_matches_finite_differences`, `<1e-6` max difference) --
an internal-consistency check independent of any dynamics, catching a
transcription error in the analytic formula regardless of whether the
downstream Lyapunov computation would have looked "plausible" anyway.

**Verification strategy for the spectrum itself.** Comparing against a
specific published Lyapunov exponent value for a specific `K` would mean
citing a number from memory that can't be independently checked here --
avoided, the same reasoning applied to `0.37.0`'s Hansen problems and
`0.38.0`'s Hubbard dimer. Instead, a **rigorous, exactly-derivable**
identity: the sum of *all* Lyapunov exponents equals the long-time average
of `trace(J(x(t)))` (a standard theorem -- the sum governs the exponential
growth rate of phase-space volume, which is exactly the flow's divergence).
For Lorenz-96 specifically, `J[i,i] = -1` for *every* `i` and *every*
state (the only `x_i`-dependence in `dx_i/dt` is the explicit `-x_i`
damping term), so `trace(J(x)) = -K` identically, for any state, not just
on average -- making the target for the sum of all `K` exponents exactly
`-K`, a real closed-form check rather than an approximate one.
`lorenz96_lyapunov_exponents_sum_to_minus_k` measures this at `K=10`: sum
`= -9.999993414457066` against the exact target `-10`
(`diff = 6.6e-6`), and `LyapunovSpectrum::final_frame_orthogonality_error`
(the tracked frame's own `Q^T Q - I` drift, a direct check that the method
kept the frame genuinely orthogonal throughout, not just plausible-looking)
measured at `5.5e-16`. `lorenz96_has_a_positive_lyapunov_exponent_at_standard_forcing`
confirms the qualitative chaos indicator this system is well known for:
the measured spectrum at `K=10, forcing=8` has four positive exponents
(`~1.18, ~0.70, ~0.065, ~0.021`), comfortably past the `>0.1` threshold
checked.

### `src/lie_svd_streaming.rs`

`0.40.0`: streaming/incremental low-rank tracking with rank adaptation --
the second genuinely new numerical subsystem in the robustness/frontier-
benchmark program (after `lie_svd_lyapunov`), processing a data stream one
column at a time instead of recomputing a full SVD on every arrival.

**Scope decision, made before writing code.** The classical reference is
Brand's rank-1 SVD update (Brand, 2006, "Fast low-rank modifications of the
thin singular value decomposition"), a specific closed-form update via a
small `(r+1)x(r+1)` block-matrix SVD. Reproducing that exact formula from
memory carried the same risk category already flagged and avoided twice in
this program (`0.37.0`'s Hansen problems, `0.38.0`'s preference for a
from-scratch-derived Hamiltonian over a memorized one): a subtly wrong sign
or index that wouldn't look obviously wrong just from running it.

**What's implemented instead**, simpler and lower-risk while still a
genuine, correct streaming/rank-adaptive tracker (`StreamingTracker`):
maintain an orthonormal basis `Q` (`n x r`) and a small `r x r` "core"
matrix representing `Q^T (sum of c c^T over columns seen) Q` in `Q`'s
current basis. On each new column `c`:

1. Project `c` onto `Q`; the residual norm `rho` measures how much of `c`
   lies outside the current tracked subspace.
2. If `rho` is large relative to `c`'s own norm (a real new direction, not
   noise) and the tracked rank is below `max_rank`, **extend** `Q` by one
   column (the normalized residual) -- the rank-jump mechanism named in the
   original proposal.
3. Accumulate this column's outer-product contribution into the core.
4. Re-diagonalize the (small) core with this crate's own
   `lie_svd_small::eigh_jacobi_full` (promoted from private to `pub(crate)`
   in `0.38.0` for the Hubbard dimer verification, reused again here rather
   than re-derived), rotate `Q` by the resulting small orthogonal matrix,
   and truncate to `max_rank` if the extend step pushed past it -- dropping
   the *smallest*-eigenvalue direction, keeping the dominant ones.

Re-diagonalizing every step (not just periodically, as a more
efficiency-optimized design might) trades a little of the amortized speed
a production streaming SVD would have for a simpler, more obviously correct
algorithm -- `max_rank` stays small (single digits) in every use in this
crate, so the extra `O(r^3)` work per step is cheap regardless, and
correctness was judged more valuable than that speed here.

**Verification strategy.** When `max_rank` is set at or above the stream's
true total rank, no energy is ever discarded by truncation, so the core
exactly equals `Q^T (C C^T) Q` for the full accumulated data `C` with no
approximation -- meaning the tracker's final singular values and (rotated)
basis should match a **direct batch SVD of the same accumulated data**
(`lie_svd_small::LieSvdSmall::solve_rectangular`, already tested elsewhere)
to near machine precision, not an external reference.
`streaming_tracker_matches_batch_svd_when_rank_is_not_truncated` measures
this (`20`-dim ambient space, `40` streamed columns, true rank `3`, two
independent seeds): singular-value relative error `~1-2e-15`, tracked-basis
orthogonality error `~8e-15`, subspace-residual agreement against the batch
left singular vectors (checked via `||v - Q Q^T v||` for each batch left
singular vector `v`) `~4-6e-15` -- essentially exact, confirming the
"no truncation means no approximation" argument numerically, not just
algebraically.

The actual point of the module,
`streaming_tracker_grows_rank_when_a_new_direction_appears`: a stream whose
true rank grows partway through (first `40` columns confined to a random
rank-2 subspace, next `40` columns drawn from a rank-3 subspace containing
the original two directions plus a genuine third) makes the tracked rank
grow from `2` to `3` in response, and the final tracked subspace captures
the new third direction specifically (checked directly against the known
true third basis vector, residual `<1e-6`), not just the original two --
the rank-adaptation mechanism actually does what it's meant to, not just
"doesn't crash when a new direction appears."

### `compare/`

`0.41.0`, cycle 6 (final) of the robustness/frontier-benchmark program: an
isolated LAPACK/MPFR ground-truth comparison harness, closing the
program's last open item without touching the main crate's own dependency
tree.

**Isolation is structural, not just documented.** `compare/` is a
completely separate Cargo package with its own `[workspace]` marker (so
`cargo build`/`cargo test` from the repository root never pulls it in) and
its own `Dockerfile`. It depends on the main crate via `path = ".."` --
one direction only, main crate source as a read-only reference, never the
reverse. Verified directly rather than assumed: after building and running
the comparison image, `grep`-ing the *main* crate's own `Cargo.lock` for
`ndarray-linalg`, `rug`, `openblas-src`, `lapack-sys`, or `gmp-mpfr-sys`
returns nothing, and `cargo build`/`cargo test --release --lib --locked`
from the main crate's own directory pass unchanged (157/157) with
`compare/` sitting right next to it on disk.

**Two real build obstacles, solved rather than routed around.**
`lapack-sys` 0.14.0 has a genuine FFI bug on `aarch64` (this development
machine's own architecture): its generated bindings declare some OpenBLAS
function parameters as `*const u8` where the actual C signature uses
`*const i8` (`char` signedness is platform-defined, and the bindings were
generated assuming the `x86_64` convention), which fails to link natively
on ARM. Building for `linux/amd64` via Docker's QEMU emulation sidesteps
this without patching the crate. Separately, `openblas-src`'s own
build-dependency `openblas-build` unconditionally requires either the
`rustls` or `native-tls` feature just to *compile* its build script -- even
when using the `system` (link against an already-installed OpenBLAS, no
download) backend, which doesn't need networking at all. Requesting
`rustls` explicitly on that dependency satisfies the compile-time
requirement without changing which OpenBLAS actually gets linked at
runtime (the system one, installed via `apt` in the `Dockerfile`).

**What it runs**, reusing the main crate's own benchmark matrix generators
from `lie_svd_benchmarks` rather than duplicating them:

1. `LieSvdSmall::solve` vs. LAPACK's `dgesdd` (via `ndarray-linalg`) on
   Kahan, Hilbert, Vandermonde, and Pei (`n=32,64`, `n=12` for Vandermonde):
   orthogonality, reconstruction accuracy, wall-clock time, and the maximum
   relative disagreement between the two solvers' sorted singular values.
2. `pei_matrix`/`pei_matrix_singular_values` cross-checked independently
   here too (`max relative error = 1.0e-12` at `n=64`), confirming the main
   crate's own closed-form test rather than just trusting it.
3. The Hilbert matrix's determinant via plain `f64` LU vs. 200-bit MPFR LU
   (via `rug`) -- a solver-*independent* measurement of how much of `f64`'s
   answer on this matrix is representation/arithmetic error alone, nothing
   to do with which SVD algorithm is used.

**Findings, measured on real production LAPACK and 200-bit MPFR, not
assumed:**

- Well-conditioned cases (Kahan, Pei): this crate agrees with LAPACK to
  `~1e-12`-`~1e-13` relative singular-value accuracy, both near
  machine-precision orthogonality/reconstruction. LAPACK is consistently
  faster (`~10-40x` in these runs) -- expected, and never claimed
  otherwise; this crate's own README has stated "not universally faster
  than classical dense solvers" since before this comparison existed.
- Hilbert and Vandermonde (condition numbers exceeding `f64`'s
  representable range, per `0.34.0`'s and `0.36.0`'s own findings): this
  crate and LAPACK **disagree substantially** on the smallest singular
  values (`~16x` relative disagreement on Hilbert `n=32`, `~2.5x` at
  `n=64`, `~27%` on Vandermonde `n=12`). This is not a bug in either
  solver -- it is the specific, concrete, externally-confirmed consequence
  of the claim those earlier cycles already made: past the representable
  condition number, the smallest singular values are numerical noise, and
  two independently competent solvers have no reason to agree on what
  that noise looks like. Getting *this* result on Hilbert against real
  LAPACK is what that claim predicts, not a surprise it failed to
  anticipate.
- MPFR vs. `f64` on the Hilbert determinant: `rel_diff` from the 200-bit
  reference grows from `~2.4e-11` (`n=6`) to `~5.4e-2` (`n=12`) -- by
  `n=12`, plain `f64`'s answer is already off by `~5%`, a clean, absolute,
  solver-independent quantification of representation error on exactly the
  matrix this whole program has repeatedly flagged as sitting on the edge
  of `f64`'s precision.

Full numbers, native-build instructions, and the exact Docker commands are
in `compare/README.md`.

**Program summary, `0.34.0`-`0.41.0`.** Eight cycles: standard "evil
matrix" benchmarks (`0.34.0`); the rest of the Higham set plus an Amari
parametric grid (`0.35.0`); self-contained robustness properties -- dynamic
range, subnormals, rotor drift (`0.36.0`); classical ill-posed inverse
problems (`0.37.0`); a from-scratch-derived quantum many-body benchmark
(`0.38.0`); Lyapunov spectrum extraction as a new numerical subsystem
(`0.39.0`); streaming low-rank tracking as a second new subsystem
(`0.40.0`); and this isolated external-reference comparison (`0.41.0`).
Each cycle made an explicit scope decision about what to build and what to
defer (Kuramoto-Sivashinsky, Trefethen pseudospectra, SuiteSparse/
Cardoso's datasets, Frank/Forsythe/Parter/Cauchy until `0.35.0`) and
recorded the reason, rather than silently dropping scope or silently
overreaching into unverified territory.

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

`0.31.0` adds `HardwareSchedule::from_orthogonal_matrix`: compiling an
arbitrary `d x d` orthogonal matrix — not a solver's own recorded event
log, but a plain rotor handed back as a matrix, e.g.
`lie_tbl_regress::procrustes_rotor`'s output — into the same schedule
shape. Feasibility was checked before building anything:
`lie_svd_small::eigh_jacobi_full`, the square eigensolver backing most of
this crate's rotors, does not record a rotation trace as it runs, and
instrumenting that hot, widely shared solver path was judged higher-risk
than the alternative implemented here: decompose the *already-orthogonal
result* after the fact via a standard Givens QR sweep, one elimination per
below-diagonal entry. Since the input is already orthogonal, eliminating
its strict lower triangle with Givens rotations leaves a matrix that is
both orthogonal *and* upper-triangular, which is necessarily diagonal with
`+-1` entries (an upper-triangular orthogonal matrix cannot have any other
off-diagonal content — each column must already have unit norm using only
entries at or above its own row). So
`V = G_1^T G_2^T ... G_m^T D` exactly, `D = diag(+-1)`, `G_k` the recorded
rotations in elimination order. `D` is stored on the new
`HardwareSchedule::diagonal_signs` field (empty for schedules built from a
PhaseFlow event log) — leaving it out would have silently dropped
information needed to reconstruct the matrix from the exported schedule
alone. Verified rather than asserted
(`orthogonal_matrix_round_trips_through_givens_schedule`): reconstructing a
`5x5` orthogonal test matrix (a Procrustes rotor between two random
matrices) from its recorded events and diagonal, by applying each
rotation's inverse in reverse order starting from `diag(diagonal_signs)`,
reproduces the original to `max_err ~1.19e-15`. A second test
(`procrustes_rotor_compiles_to_a_valid_mzi_schedule`) exercises the
concrete intended use case end to end: a `TblRotorRegressor` domain-transfer
rotor compiled straight to an MZI hardware schedule.

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
