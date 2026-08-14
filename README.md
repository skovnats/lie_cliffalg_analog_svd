# lie_cliffalg_analog_svd

Research-grade SVD solvers and phase-rotor diagnostics for dense `f64`
matrices, packaged as a small Linux/CPU crate.

The project keeps the useful part of the Lie/Clifford/analog-chip exploration:
ordinary CPU arrays in memory, but rotor-based update rules that mirror future
analog or photonic hardware where orthogonal rotations are native operations.

The main practical focus is **difficult SVD inputs**: degenerate spectra,
extreme ill-conditioning, non-normal/Jordan-like structure, and cases where a
small reconstruction error can hide a broken singular-vector basis. The package
therefore reports orthogonality and spectrum-tail diagnostics, not only
`||A - U Sigma Vt||`.

## What We Built

This release is the result of a sequence of geometry-inspired SVD experiments
that were gradually made more conservative and testable:

- **Robust baseline:** `LieSvdSmall` uses polar decomposition plus symmetric
  Jacobi polish, avoiding `A^T A` as the primary route so the condition number
  is not squared up front.
- **Tiny rotor kernels:** `LieSvdMicro` handles `N <= 4` as local rotor cells,
  with residual checks and fallback to the robust baseline.
- **Block-4 macro rotors:** `LieSvdBlock4` promotes the `4x4` cell into a
  larger warm-start primitive, using contiguous quartets and power-of-two
  butterfly quartets before a final digital polish.
- **Analog/photonic schedule:** `LieSvdAnalog` models SVD as conflict-free
  layers of `2x2` rotation cells, then finishes with digital polish.
- **Core-flow state:** `LieSvdCoreFlow` exposes `core = U^T A V`, keeps `A`
  fixed, and moves only the two orthogonal bases.
- **Repellers:** optional Calogero-Moser-style anti-clustering terms act during
  degenerate/off-diagonal phases, then stay out of final polish.
- **Kernel route:** `kernel_gram` separates symmetric single-domain Gram/RBF
  kernels (`K = U Sigma U^T`) from nonsymmetric bipartite kernels.
- **Topological warm-start:** `lie_svd_topowarm` uses stationary masses,
  Fiedler-like graph relaxation, phase-stress landmarks, pseudo-random probes,
  and orthogonal retraction to seed `CoreFlow` and `Block4`.
- **Adaptive synergy:** `LieSvdAdaptive` decides when the full geometric stack
  is worth using and otherwise preserves the `Small` fast path.
- **Tensor/Kronecker view:** `lie_svd_tensortrain` detects matrices that are
  close to chains of `2x2` Kronecker factors and, when that structure is real,
  assembles the SVD from tiny local rotor SVDs.
- **Trace/Procrustes navigator:** `lie_svd_traceflow` starts from identity
  bases and rotates them like an inverse Rubik path, maximizing
  `sum(abs(diag(U^T A V)))` before digital polish.
- **Global quad-view audit:** `lie_svd_quadenergy` treats rows and columns as
  separate Clifford basis families (`e_i`, `f_j`) and measures the primal
  row-column tensor, row-dual metric, column-dual metric, and full dual
  mismatch.
- **Fractal phase-health audit:** `lie_svd_phasehealth` treats every row and
  every column as its own local Clifford-like signal and reports scalar mass,
  vector spread, deterministic phase-delay twist, entropy, and row/column
  disagreement.
- **Active phase-flow solver:** `lie_svd_phaseflow` turns that phase portrait
  into an actuator. It applies global phase-jump rotors and targeted unwrap
  rotors as a first-class SVD route, with golden-angle anti-resonance jumps and
  a guarded `4x4` surgery fallback for high-stress plateaus. `0.19.0` adds a
  Layer-0 **Golden Global Phase Dispersion** sheet: all row/column axes receive
  a deterministic Fibonacci/golden pre-spin through real conflict-free rotors
  before the local `4x4` and `2x2` phase cells begin. `0.23.0` adds the
  causal/Jordan antipode: a **Causal Anti-Spin** layer with opposite-sign
  row/column rotors for one-sided triangular flow. `0.24.0` adds the
  **Cross-Phase Yin-Yang Cycle**: a multi-layer four-act row/column
  golden-antipode cascade with golden-ratio annealing. `0.25.0` adds
  **Phase-Conjugate Auto-Spin** and **Bottleneck Phase Alignment**, so the
  solver can mirror the phase state it sees and attack maximum-energy pairs
  first.
- **Phase passport dispatch:** `PhaseSignature` compresses the row/column
  phase field into mean stress, max twist, causal disbalance, and entropy gap.
  `LieSvdAdaptive` now uses this passport to route degenerate and causal/Jordan
  stress through `PhaseFlow` up to the current geometric auto cap (`N <= 64`
  by default). Larger matrices can still run `PhaseFlow`/`CoreFlow` explicitly
  from `stress_cpu` while the batch/cache kernel is developed.
- **Phase-JADE joint diagonalization:** `lie_svd_joint` applies the same shared
  rotor logic to matrix families, minimizing
  `sum_k ||offdiag(V^T M_k V)||^2` with in-place joint pair updates.
- **Phase-BSS:** `lie_svd_bss` uses whitening plus lagged-covariance
  Phase-JADE to separate mixed channels and report channel phase coherence.
- **Tensor phase factorization:** `lie_svd_tensor` adds a first 3D HO-SVD /
  Tucker-style route, rotating each tensor mode into an orthogonal core.
- **Complex-native phase engine:** `lie_svd_complex` starts the `Complex64`
  branch. It supports complex SVD, direct U(1) golden pre-spin, a `2x2`
  complex microkernel, Hermitian Jacobi phase alignment, and MZI-native phase
  event export.
- **Unified phase engine:** `lie_svd_engine` adds `PhaseEngine` and
  `PhasePassport`, giving real SVD, complex SVD, Phase-BSS, tensor HO-SVD, and
  Phase-JADE a shared diagnostic/report interface.
- **Hardware schedule compiler:** `lie_svd_compiler` converts real and complex
  phase events into a unified MZI/FPGA-style schedule with layers, channels,
  phase angles, and JSON export.
- **Two-sided Joint SVD:** `LieSvdJoint::joint_svd` extends the ensemble route
  to nonsymmetric families `U^T A_k V`; square families now have a tested
  two-sided phase-locking path.
- **Rectangular phase diagnostics:** `LieSvdPhaseFlow` now has a rectangular
  phase-locking route for `N x M` operators. Row generators `e_i` and column
  generators `f_j` are treated as different spaces, with full rectangular
  output shapes `U: N x N`, `sigma: min(N,M)`, and `Vt: M x M`.
- **Exact axis-energy pruning:** `0.26.0` adds `hot_axes`, a certificate-based
  pre-filter used by every pair-candidate builder in `PhaseFlow`:
  `pair_offdiag(i,j) <= min(axis_energy_i, axis_energy_j)` where
  `axis_energy_k = row_norm_k + col_norm_k`, so any axis at or below `pair_tol`
  can be dropped from search without reading the matrix. It's an exact, free
  floor, not a demonstrated speedup on the current dense/random benchmark
  suite (those profiles don't have genuinely cold axes); `active_set_alpha`
  adds an opt-in relative "strong rules" screen on top for structured/sparse
  inputs where axis energy is actually skewed.
- **Global phase invariants:** `0.27.0` adds
  `lie_svd_phasehealth::global_phase_invariants`, four whole-matrix scalars
  that sit next to the existing per-row/per-column diagnostics rather than
  replacing them: `global_phase` (mass-weighted circular mean phase angle),
  `torsion_energy` (`H_total = ||skew(A)||_F`), `chirality_balance`
  (self-dual/anti-self-dual bivector balance, reusing
  `lie_svd_block4::analyze_block4_signature`), and `phase_entropy`
  (normalized Shannon entropy of the whole matrix's energy distribution).
  They're exported through `PhaseSignature` and `lie_svd_engine::PhasePassport`.
  Also adds an opt-in `use_adaptive_viscosity` (`--adaptive-viscosity`)
  bottleneck-rotor damping heuristic, `gamma = P / (P + R)`, named literally
  as an energy-ratio gain rather than a Kalman filter (no state covariance is
  propagated across passes). Measured mixed on `N=64`: it roughly doubles
  bottleneck rotation attempts and gives a worse raw result on
  `jordan_defective` but a slightly better one on `sparse_structured` — see
  the 0.27.0 section below for the numbers.
- **Allocation reduction and chirality-driven dispatch:** `0.28.0` reuses
  `Vec<AxisPhase>` buffers across `PhaseFlow` passes instead of allocating
  fresh ones (`row_phases_into`/`col_phases_into`); measured a real but
  partial `~6%` allocation drop with no measurable time change, which itself
  shows allocations were not the dominant cost (millions of `O(n)` rescore
  operations are). `Auto` also gains a `phase_chirality_balance`-driven
  `PhaseFlow` trigger, calibrated on real matrices and guarded by
  `diagonal_dominance` after an unguarded first version caused a 100x
  slowdown on `sparse_structured` — see the 0.28.0 section below.
- **Lazy `BottleneckPairCache` invalidation:** `0.29.0` found and fixed the
  actual bottleneck `0.28.0`'s profiling pointed at: `update_axes` no longer
  eagerly rescores every `(touched_axis, other)` pair (`O(touched * n)` per
  rotor); it bumps a per-axis generation counter and defers rescoring to pop
  time, verified before trusting an entry as the true max. Cut rescoring
  `53M -> 349K` (`~153x`) on a real `N=300` trace with a periodic full
  rebuild (default every `16` passes) needed to avoid a real, measured
  accuracy regression from pure lazy invalidation — see the 0.29.0 section
  below for the three-stage honest measurement.

The important practical lesson so far is restraint: the geometric methods are
most useful on structured, balanced-degenerate, and causal/Jordan-like cases.
On ordinary random, extreme ill-conditioned, sparse structured, nearly
diagonal, and generic tensor-chain smoke profiles, the adaptive dispatcher
keeps the simpler fast path.

## What Is Included

- `LieSvdSmall`: polar decomposition plus dense Jacobi polish. This is the
  conservative CPU default and the best current path for small and medium
  dense matrices.
- `LieSvdMicro`: tiny fixed-schedule rotor microkernels for `N <= 4`, avoiding
  the setup overhead of the general solver on very small hot blocks.
- `LieSvdBlock4`: macro-rotor warm start built from local `4x4` SVD cells.
  It applies contiguous quartet layers and stride-1/2/4/... butterfly quartets
  that match powers-of-two tensor layouts. For `N >= 5`, this is deliberately a
  warm start followed by robust polish, not a claim of closed-form SVD.
  `analyze_block4_signature` additionally splits each contiguous `4x4`
  torsion block into self-dual and anti-self-dual `SO(4)` halves.
- `LieSvdHybrid`: dual-tiled Lie/Clifford preconditioner plus `LieSvdSmall`
  polish. It is kept as the large/research path.
- `LieSvdAnalog`: analog-chip-oriented rotor mesh simulator. It uses
  conflict-free local `2x2` cells, optional phase quantization, and a digital
  polish path.
- `LieSvdCoreFlow`: prototype that keeps `A` fixed while moving `U` and `V`,
  explicitly minimizing the off-diagonal field of `core = U^T A V`. Versions
  `0.3.0..0.6.0` add monotone line-search acceptance plus optional
  Calogero-Moser-style anti-clustering repellers.
- `kernel_gram`: Linear/RBF Gram builders and a mathematically explicit kernel
  route. Symmetric single-domain kernels use one basis (`K = U Sigma U^T`);
  nonsymmetric square cross-kernels use the two-sided `CoreFlow` path.
- `lie_svd_topowarm`: landmark/sphere warm-start for `CoreFlow`. It uses
  row/column stationary masses, a cheap Fiedler-like bipartite relaxation,
  phase-guided landmarks, pseudo-random probes, tiny two-sided power
  refinement, and an orthogonal retraction to produce a guarded initial basis.
- `LieSvdAdaptive` / `LieSvd`: adaptive dispatcher. It keeps `Small` as the
  default fast path, but can enable `CoreFlow + TopoWarm + Repeller` on
  balanced degenerate or graph/topological inputs.
- `lie_svd_tensortrain`: Kronecker-chain diagnostics and an exact fast-path for
  matrices that factor as a chain of `2x2` tensor products. This is the
  tensor-network/Schmidt-decomposition angle: useful when the matrix has low
  tensor bond complexity, intentionally bypassed otherwise.
- `lie_svd_traceflow`: trace/Procrustes diagnostic solver. It exposes the
  mathematically equivalent "build the matrix from identity bases" view:
  maximize diagonal trace projection, then read `sigma_i` from the resulting
  diagonal and finish with the robust polish path.
- `lie_svd_quadenergy`: energy microscope for two related but distinct
  pictures. The global picture is the user's row/column Clifford tensor view:
  `A = sum_ij a_ij e_i tensor f_j`, plus row and column dual metric views.
  The local picture fixes the precise `2x2` coordinates:
  scalar `(p+w)/2`, diagonal-vector `(p-w)/2`, symmetric-vector `(q+r)/2`,
  and bivector/torsion `(q-r)/2`.
- `lie_svd_phasehealth`: fractal row/column phase-health diagnostics. It keeps
  the global `e_i tensor f_j` view, but also asks what each individual row and
  column looks like as a local signal: how concentrated its energy is, how much
  scalar mass it has, and how much deterministic phase-delay twist it carries.
  Its `PhaseSignature` is the compact `O(n^2)` passport used by the dispatcher:
  mean stress, max twist, causal disbalance, and entropy gap.
- `lie_svd_phaseflow`: active phase-locking SVD route. It uses the row/column
  phase portrait to choose stress pairs, applies global phase jumps and
  targeted unwrap rotors, and returns the phase-locked SVD directly. The
  separate `PhaseFlowPolished` route adds `LieSvdSmall` only as a final
  audit-quality cleanup. `phase_lock_rectangular_with_trace` exposes the
  rectangular row/column-space version. When `PhaseSignature` sees strong
  triangular causal disbalance, the route uses Causal Anti-Spin instead of
  the isotropic golden pre-spin sheet. For explicit experiments,
  `--yinyang-cycles N` applies a four-act row/column cross-phase cascade before
  local phase locking. `--phase-conjugate` and `--bottleneck` turn on the
  state-driven 0.25.0 actuator path.
- `lie_svd_joint`: Phase-JADE prototype for symmetric joint diagonalization.
  It searches for one shared orthogonal basis `V` for a family `{M_k}` by
  driving down `sum_k ||offdiag(V^T M_k V)||_F^2`. This is the natural
  Cardoso/JADE-style extension of the phase-flow view: one rotor field acts on
  an ensemble instead of a single matrix. `joint_svd` adds the two-sided
  nonsymmetric family route `U^T A_k V`.
- `lie_svd_bss`: Phase-BSS prototype. It centers and whitens observed
  `channels x samples` signals, builds lagged covariance matrices, applies the
  shared Phase-JADE rotor field, and returns an unmixing matrix plus separated
  channels. It also reports `Channel Phase Coherence` and a simple SIR estimate
  helper for synthetic benchmarks.
- `lie_svd_tensor`: Higher-order phase SVD prototype for 3D tensors. It builds
  Gram matrices per mode, diagonalizes each mode with the robust SVD path, and
  rotates the tensor into a Tucker-style core. The current target is stable
  orthogonal mode factors and reconstruction, not full CP/PARAFAC optimization.
- `lie_svd_complex`: complex-native `Complex64` phase engine. It adds direct
  U(1) golden pre-spin, complex SVD, a complex `2x2` microkernel, Hermitian
  Jacobi phase alignment, and MZI-native phase event export. This is a
  foundation module with improved Hermitian/QR polish in `0.22.0`; strict
  machine-tight `U` unitarity on all complex dense tails remains future
  QDWH/bidiagonal work.
- `lie_svd_engine`: unified facade for the current ecosystem. It emits a
  `PhasePassport` with stress, twist, causality, chirality, golden resonance,
  and route hints, then calls the appropriate specialist route.
- `lie_svd_compiler`: hardware schedule export layer. It compiles real
  `MziPhase` and complex `ComplexMziPhase` events into a common schedule shape
  for MZI meshes or FPGA rotor meshes and can serialize that schedule to JSON.
- `stress_cpu`: self-contained benchmark binary with no LAPACK/OpenBLAS/faer
  dependency.

For a calmer file-by-file explanation of the design, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Install

```bash
cargo build --release
```

For best CPU performance on the local machine:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

For reproducible release checks:

```bash
cargo fmt --check
cargo test --release --lib --locked
cargo run --release --bin stress_cpu --locked -- 64 --analog
```

## Quick Use

```rust
use lie_cliffalg_analog_svd::lie_svd::LieSvd;
use ndarray::Array2;

let a = Array2::<f64>::eye(8);
let (u, sigma, vt) = LieSvd::solve(&a);
```

The API returns `(U, sigma, Vt)`, so reconstruction is:

```rust
let sigma_mat = ndarray::Array2::from_diag(&sigma);
let reconstructed = u.dot(&sigma_mat).dot(&vt);
```

## Benchmark

```bash
cargo run --release --bin stress_cpu -- 64
cargo run --release --bin stress_cpu -- 64 --analog
cargo run --release --bin stress_cpu -- 64 --coreflow
cargo run --release --bin stress_cpu -- 64 --coreflow --repel-lambda 0.02 --repel-eps 1e-8
cargo run --release --bin stress_cpu -- 64 --coreflow --topowarm --topowarm-rank 8 --topowarm-graph-steps 2
cargo run --release --bin stress_cpu -- 64 --auto-trace
cargo run --release --bin stress_cpu -- 64 --kron-trace --kron-chain
cargo run --release --bin stress_cpu -- 32 --trace-nav --traceflow
cargo run --release --bin stress_cpu -- 32 --quad-energy
cargo run --release --bin stress_cpu -- 32 --phase-health
cargo run --release --bin stress_cpu -- 32 --phaseflow --phaseflow-polish
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --no-causal-antispin
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 2
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --phase-conjugate --bottleneck
cargo run --release --bin stress_cpu -- 16 --full-suite --diagnostics-only
cargo run --release --bin stress_cpu -- 128
```

Profiles tested:

- `uniform_random`
- `degenerate_spectrum`
- `extreme_ill_conditioned`
- `jordan_defective`
- `sparse_structured`
- `nearly_diagonal`
- `kron_structured`

Metrics:

- relative reconstruction error
- `U` and `V` orthogonality error
- known-spectrum recovery error where a known spectrum is available
- allocation count and peak memory sample

These stress tests are part of the design, not just examples. The project is
most interesting on matrices where ordinary happy-path checks can be
misleading: repeated singular values, near-rank-deficiency, very large condition
numbers, and non-normal structured operators.

## Docker Linux Smoke Test

```bash
docker build -t lie-cliffalg-analog-svd .
docker run --rm lie-cliffalg-analog-svd
```

The Docker image builds in release mode with `RUSTFLAGS="-C target-cpu=native"`
and runs the `N=64` analog-inclusive stress benchmark.

## License

This release folder follows the license text in `License` (SVE Meta-License
v5.0). If you plan to publish this crate to crates.io, consider whether you want
that exact custom license there or a separate permissive/public research
license; the current `Cargo.toml` intentionally points to the included file to
avoid mismatched license metadata.

## Design Notes

The analog solver is not pretending that an analog chip magically computes SVD
at arbitrary precision. It models a realistic split:

1. A cheap analog/photonic mesh applies many local orthogonal rotations.
2. Diagonal gains/crossbars represent the singular-value scaling stage.
3. A small digital polish/audit path finishes high-precision work.

That makes the module useful today as a CPU schedule simulator and useful later
as a hardware mapping target.

`CoreFlow` follows the same cautious style. Repellers are off by default. When
enabled with `--repel-lambda`, the Calogero-Moser-style potential acts only
during the active clustered/off-diagonal phase and every proposed rotor is still
accepted through a backtracking check that requires the off-diagonal `core`
energy to not increase.

For kernel experiments, the single-domain Gram case is not treated as a generic
two-basis SVD. If `K = K^T`, the correct relaxation is the one-basis
trace/eigen route (`max tr(U^T K U)`), so left and right rotors are the same
object. Only nonsymmetric/bipartite kernels have genuinely separate row and
column bases.

The topological warm-start is intentionally guarded. It does not claim an exact
diffusion center, exact Fiedler vector, or non-iterative SVD. It builds a cheap
stationary/Fiedler-like landmark approximation, retracts it to orthogonal
bases, and accepts it only when the initial `offdiag(U^T A V)` is lower than the
identity start.

The adaptive dispatcher uses a cheap triage pass before choosing a solver:
off-diagonal ratio, diagonal dominance, row/column norm spread, row/column mass
mismatch, transpose torsion, symmetry, and entropy. The geometric stack is only
enabled for structured cases where those views agree that the extra work may
pay off.

The tensor/Kronecker route is similarly guarded. It first asks whether the
matrix looks like `A0 kron A1 kron ...` with `2x2` factors. If the residual of
that chain is low, the SVD is assembled from the tiny factor SVDs. If not, it
does nothing. This is a useful "eagle-eye" diagnostic for tensor-network-like
inputs, not a replacement for the dense default solver.

The trace navigator is another equivalent view of the same SVD target. It uses
the Procrustes/von-Neumann trace principle: the best orthogonal `U,V` maximize
`tr(U^T A V)` up to sign choices, so the implementation tracks
`sum(abs(diag(U^T A V)))`. This is useful for explaining and auditing the local
rotors: each accepted pair rotation is a small "Rubik move" that increases the
visible diagonal projection. It is not a shortcut around SVD, because repeated
singular values still create flat maximizing manifolds.

The quad-energy audit is the most useful debugging view for the global
row/column Clifford idea. In that global picture, each row and column is its
own basis direction:

```text
A = sum_ij a_ij e_i tensor f_j
```

The four global views are:

```text
1. Primal row-column tensor:      A
2. Row-dual metric view:          A A^T
3. Column-dual metric view:       A^T A
4. Dual mismatch / quad spread:   disagreement between those views
```

Separately, for ordinary matrix energy bookkeeping it uses the split:

```text
A = diag(A) + sym_offdiag(A) + skew(A)
```

not `diag(A) + (A + A^T)/2`, which would double-count the diagonal. For every
local `2x2` block `[[p, q], [r, w]]`, all four coordinates are necessary:

```text
E = (p + w) / 2     scalar
F = (p - w) / 2     diagonal vector / gap
G = (q + r) / 2     symmetric off-diagonal strain
H = (q - r) / 2     bivector torsion
```

Dropping `F` loses the diagonal gap and gives wrong rotor angles. This local
`E,F,G,H` block algebra is not the same as the global four views above; it is
the exact rotor microkernel used inside one selected row/column plane.

## Release Status

This is an experimental research release. Use it for exploration, benchmarking,
and hardware-oriented algorithm design. For production numerical software, keep
comparing against LAPACK-class solvers.

## Reproducible Smoke Results

The tables below are short Linux/Docker sanity checks, not full benchmark
claims. The main Docker table was produced on the `0.6.0` release path and
remains a baseline snapshot for the current package:

```bash
docker build -t lie-cliffalg-analog-svd .
docker run --rm lie-cliffalg-analog-svd
```

Default Docker command:

```bash
stress_cpu 64 --analog
```

### Docker `N=64` Smoke

The Docker command runs `Small`, `Hybrid`, `Auto`, and `AnalogPolished`.
Selected rows:

| Profile | Solver | Time (s) | Rel. Recon | Orth U | Orth V |
| --- | --- | ---: | ---: | ---: | ---: |
| `uniform_random` | `Small` | 0.006 | 1.154e-14 | 8.784e-14 | 8.806e-14 |
| `uniform_random` | `Auto` | 0.004 | 1.154e-14 | 8.784e-14 | 8.806e-14 |
| `uniform_random` | `AnalogPolished` | 0.005 | 1.279e-13 | 2.437e-14 | 2.366e-14 |
| `degenerate_spectrum` | `Small` | 0.009 | 8.082e-14 | 1.551e-13 | 1.547e-13 |
| `degenerate_spectrum` | `Auto` | 0.009 | 8.082e-14 | 1.551e-13 | 1.547e-13 |
| `degenerate_spectrum` | `AnalogPolished` | 0.008 | 6.989e-14 | 2.627e-14 | 2.516e-14 |
| `extreme_ill_conditioned` | `Small` | 0.009 | 9.223e-14 | 9.235e-14 | 9.143e-14 |
| `extreme_ill_conditioned` | `Auto` | 0.008 | 9.223e-14 | 9.235e-14 | 9.143e-14 |
| `extreme_ill_conditioned` | `AnalogPolished` | 0.011 | 3.305e-14 | 3.814e-14 | 3.507e-14 |
| `jordan_defective` | `Small` | 0.005 | 1.029e-14 | 2.913e-15 | 1.147e-13 |
| `jordan_defective` | `Auto` | 0.005 | 1.029e-14 | 2.913e-15 | 1.147e-13 |
| `jordan_defective` | `AnalogPolished` | 0.007 | 5.284e-15 | 2.983e-14 | 3.041e-14 |
| `sparse_structured` | `Small` | 0.004 | 1.256e-14 | 1.014e-13 | 1.014e-13 |
| `sparse_structured` | `Auto` | 0.004 | 1.256e-14 | 1.014e-13 | 1.014e-13 |
| `sparse_structured` | `AnalogPolished` | 0.004 | 4.079e-15 | 2.313e-14 | 2.293e-14 |
| `nearly_diagonal` | `Small` | 0.001 | 7.301e-15 | 5.726e-14 | 5.794e-14 |
| `nearly_diagonal` | `Auto` | 0.001 | 7.301e-15 | 5.726e-14 | 5.794e-14 |
| `nearly_diagonal` | `AnalogPolished` | 0.001 | 7.059e-15 | 7.663e-15 | 6.473e-15 |

The full command also prints `Hybrid` and `Auto` rows, known-spectrum tail
errors where available, allocation counts, and peak memory samples.

### Adaptive `N=32` Smoke

To see the adaptive dispatcher decisions:

```bash
cargo run --release --bin stress_cpu --locked -- 32 --auto-trace
```

Observed route behavior:

| Profile | Auto Route | Auto Rel. Recon | Why It Matters |
| --- | --- | ---: | --- |
| `uniform_random` | `Small` | 5.982e-15 | avoids false-positive geometry |
| `degenerate_spectrum` | `CoreFlowTopo` | 4.663e-15 | full synergy path improves residual |
| `extreme_ill_conditioned` | `Small` | 1.215e-14 | avoids a route that hurt precision |
| `jordan_defective` | `Small` | 3.383e-15 | robust polar/Jacobi path is enough |
| `sparse_structured` | `Small` | 5.218e-15 | no unnecessary preconditioner cost |
| `nearly_diagonal` | `Small` | 3.311e-15 | preserves an already solved basis |

The key result is not that `CoreFlowTopo` is always faster. It is that `Auto`
now composes the expensive geometric views only when the triage says they are
likely to help.

### Experimental Paths

For the explicit prototype paths, use:

```bash
cargo run --release --bin stress_cpu --locked -- 4 --analog --coreflow
cargo run --release --bin stress_cpu --locked -- 32 --analog --coreflow
cargo run --release --bin stress_cpu --locked -- 32 --coreflow --repel-lambda 0.02 --repel-eps 1e-8
cargo run --release --bin stress_cpu --locked -- 32 --coreflow --topowarm --topowarm-rank 8 --topowarm-power-steps 2 --topowarm-graph-steps 2
```

In a local `N=32` smoke comparison, `CoreFlow + repeller` on
`degenerate_spectrum` gave `rel_recon ~ 1.39e-13`; enabling `--topowarm` with
the same repeller settings gave `rel_recon ~ 2.20e-15`. This is encouraging for
structured/degenerate cases, but it is not a universal speedup on random dense
matrices.

In `0.6.0`, `Auto` enables the full `CoreFlow + TopoWarm + Repeller` path on
the synthetic `degenerate_spectrum` profile and keeps the `Small` fast path on
`uniform_random`, `extreme_ill_conditioned`, `jordan_defective`,
`sparse_structured`, and `nearly_diagonal` in the `N=32` smoke run.

At `N=64`, manual `CoreFlow + TopoWarm` improves some residuals but is much
more expensive in the current implementation, so `Auto` intentionally keeps the
fast path there. This is a design choice, not a missed trigger.

### Tensor/Kronecker `0.7.0` Smoke

To inspect the new tensor-chain view:

```bash
cargo run --release --bin stress_cpu --locked --offline -- 16 --kron-trace --kron-chain
```

Observed selected rows:

| Profile | Kron First Residual | Chain Levels | Solver | Rel. Recon | Orth U | Orth V |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `uniform_random` | 8.430e-1 | 0 | rejected | n/a | n/a | n/a |
| `degenerate_spectrum` | 8.079e-1 | 0 | rejected | n/a | n/a | n/a |
| `nearly_diagonal` | 1.818e-2 | 0 | rejected | n/a | n/a | n/a |
| `kron_structured` | 8.601e-17 | 4 | `KronChain` | 2.889e-16 | 6.500e-16 | 8.007e-16 |

This is the intended behavior: the new path is excellent when the matrix
really is a tensor-product chain, and it stays out of the way on ordinary dense
profiles.

### Trace/Procrustes `0.8.0` Smoke

To inspect the inverse-Rubik trace navigator:

```bash
cargo run --release --bin stress_cpu --locked --offline -- 16 --trace-nav --traceflow
```

Selected observations:

| Profile | Trace Projection | Offdiag Core | TraceFlow Rel. Recon | Orth U | Orth V |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform_random` | 1.367e1 -> 5.459e1 | 1.523e1 -> 1.343e-7 | 1.816e-15 | 5.370e-15 | 5.080e-15 |
| `degenerate_spectrum` | 1.193e2 -> 3.040e2 | 1.540e2 -> 3.987e-12 | 1.907e-14 | 5.097e-15 | 6.002e-15 |
| `nearly_diagonal` | 2.450e1 -> 2.450e1 | 1.093e-5 -> 4.383e-16 | 5.490e-16 | 1.088e-15 | 1.369e-15 |

This confirms the interpretation: the trace objective is a clean navigator for
local rotors and makes the diagonal projection visible. It is still a rotor
sweep, so it is a diagnostic/prototype route rather than the default speed
path.

### Quad-Energy `0.9.0` Smoke

To inspect the four-view energy split:

```bash
cargo run --release --bin stress_cpu --locked --offline -- 16 --quad-energy
```

Selected observations:

| Profile | Diag | Offdiag | Sym Strain | Skew/Torsion | Upper | Lower | Row Metric | Col Metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_random` | 4.190e0 | 1.523e1 | 1.174e1 | 9.694e0 | 1.043e1 | 1.110e1 | 5.421e1 | 5.452e1 |
| `degenerate_spectrum` | 3.589e1 | 1.540e2 | 1.076e2 | 1.102e2 | 8.851e1 | 1.260e2 | 1.207e4 | 1.212e4 |
| `jordan_defective` | 3.116e0 | 7.769e1 | 5.494e1 | 5.494e1 | 7.769e1 | 0.000e0 | 1.443e2 | 1.463e2 |
| `sparse_structured` | 1.880e1 | 4.868e0 | 6.919e-1 | 4.819e0 | 3.896e0 | 2.918e0 | 8.310e0 | 8.198e0 |

This gives a practical answer to "what angle are we not seeing?" For dense
generic matrices there is no obvious `N log N` shortcut in this audit; the
energy is spread across many views. But for structured cases, the split exposes
low-complexity directions: Jordan is a one-way triangular flow, sparse
structured is mostly torsion, and exact tensor chains are visible through
separate Kronecker diagnostics.

### Phase-Health `0.10.0` Smoke

To inspect row/column internal phase health:

```bash
cargo run --release --bin stress_cpu --locked --offline -- 16 --phase-health
```

Selected observations:

| Profile | Row Twist | Col Twist | Row Entropy | Col Entropy | Phase Stress |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uniform_random` | 9.708e-1 | 9.654e-1 | 7.631e-1 | 7.663e-1 | 1.205e2 |
| `degenerate_spectrum` | 9.644e-1 | 9.470e-1 | 7.292e-1 | 7.164e-1 | 1.157e4 |
| `jordan_defective` | 9.980e-1 | 9.980e-1 | 1.342e-2 | 1.317e-2 | 2.358e3 |
| `nearly_diagonal` | 1.000e0 | 1.000e0 | 3.797e-11 | 3.784e-11 | 2.708e1 |
| `kron_structured` | 9.944e-1 | 9.929e-1 | 1.070e-1 | 8.004e-2 | 1.740e2 |

Interpretation:

- `degenerate_spectrum` has very high total phase stress, matching the earlier
  observation that repeated spectra need special handling.
- `jordan_defective` has low entropy but high stress: the energy is structured
  and one-way rather than random.
- `nearly_diagonal` has near-zero entropy, so the dispatcher should avoid
  disturbing it even though the cyclic-delay twist proxy is high for sparse
  one-hot rows.

The phase-delay bivector is a deterministic diagnostic proxy. It is not a
canonical intrinsic bivector of a single row or column; a lone vector only gets
a bivector after choosing a second direction. Here that second direction is a
one-step cyclic phase delay, chosen because it is cheap, reproducible, and
useful for seeing internal row/column phase stress.

### Active PhaseFlow / Phase-JADE / Block-4 `0.17.0` Smoke

To run the active phase-locking solver:

```bash
cargo run --release --bin stress_cpu --locked --offline -- 16 --phaseflow --phaseflow-polish
cargo run --release --bin stress_cpu --locked --offline -- 64 --auto-trace --phaseflow --phaseflow-polish --golden-jumps
cargo run --release --bin stress_cpu --locked --offline -- 64 --phaseflow --golden-prespin --golden-jumps
cargo run --release --bin stress_cpu --locked --offline -- 64 --phaseflow --no-golden-jumps
cargo run --release --bin stress_cpu --locked --offline -- 64 --phaseflow --no-golden-prespin
cargo run --release --bin stress_cpu --locked --offline -- 32 --joint
cargo run --release --bin stress_cpu --locked --offline -- 64 --joint-svd --diagnostics-only
cargo run --release --bin stress_cpu --locked --offline -- 64 --block4 --topo-warm
cargo run --release --bin stress_cpu --locked --offline -- 16 --bss-demo --tensor-hosvd --diagnostics-only
cargo run --release --bin stress_cpu --locked --offline -- 16 --complex-svd --diagnostics-only
cargo run --release --bin stress_cpu --locked --offline -- 512 --joint --rect --rect-cols 768 --diagnostics-only
```

Selected `N=16` observations:

| Profile | PhaseFlow Offdiag | PhaseFlow Rel. Recon | PhaseFlowPolished Rel. Recon | Orth U/V |
| --- | ---: | ---: | ---: | ---: |
| `uniform_random` | 1.523e1 -> 5.468e-3 | 3.462e-4 | 1.921e-15 | ~5e-15 |
| `degenerate_spectrum` | 1.540e2 -> 3.402e-12 | 2.170e-14 | 1.316e-14 | ~6e-15 |
| `extreme_ill_conditioned` | 9.721e-1 -> 2.011e-5 | 2.007e-5 | 2.442e-14 | ~6e-15 |
| `jordan_defective` | 7.769e1 -> 1.653e-1 | 2.126e-3 | 2.718e-15 | ~5e-15 |
| `nearly_diagonal` | 1.093e-5 -> 4.549e-14 | 7.484e-15 | 3.611e-15 | ~1e-15 |

Selected `N=64` observations:

| Profile | Auto Route | Phase Stress | PhaseFlow Offdiag | PhaseFlow Rel. Recon | PhaseFlowPolished Rel. Recon |
| --- | --- | ---: | ---: | ---: | ---: |
| `uniform_random` | `Small` | 1.031e4 | 6.340e1 -> 5.521e-2 | 8.646e-4 | 3.904e-15 |
| `degenerate_spectrum` | `PhaseFlow` | 2.084e5 | 3.139e2 -> 2.270e0 | 7.177e-3 | 5.718e-15 |
| `extreme_ill_conditioned` | `Small` | 3.945e1 | 1.161e0 -> 2.263e-3 | 1.936e-3 | 1.366e-13 |
| `jordan_defective` | `PhaseFlow` | 5.420e4 | 1.592e2 -> 4.815e-1 | 3.023e-3 | 9.175e-15 |
| `nearly_diagonal` | `Small` | 5.746e2 | 4.490e-5 -> 1.327e-13 | 1.093e-14 | 8.000e-15 |

This is the first version where phase-health becomes an adaptive control
surface. `Auto` now chooses `PhaseFlow` for the two profiles that match the
phase-passport idea most clearly: repeated/clustered spectra and causal
Jordan-like flow. `PhaseFlowPolished` is kept separate so the final digital
cleanup is not confused with the phase-flow mechanism itself.

`0.13.0` removes the clone-based trial core from `PhaseFlow` acceptance.
Candidate rotors are now applied in-place, evaluated through the local
two-axis off-diagonal delta, and rolled back by the inverse rotor if rejected.
This keeps the "1 writes, 2-4 in mind" principle literal: the stored matrix is
plain `f64`, while the phase/dual geometry stays in the rotor schedule.

`stress_cpu --joint` runs a small Phase-JADE smoke test. It builds a synthetic
family of jointly diagonalizable symmetric matrices and reports the drop in
joint off-diagonal energy, sweep count, accepted rotors, and rejected rotors.
Observed smoke:

| N | Family | Joint Offdiag | Sweeps | Rotations | Time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6 | 2.537e0 -> 9.804e-14 | 6 | 681 | 0.002s |
| 64 | 6 | 5.835e0 -> 1.398e-12 | 8 | 14917 | 0.078s |
| 128 | 6 | 8.353e0 -> 2.695e-12 | 10 | 63520 | 0.66s |
| 256 | 6 | 1.198e1 -> 8.459e-12 | 11 | 270435 | 11.24s |

`0.14.0` also adds `--rect` for rectangular phase diagnostics:

| Shape | Rect Offdiag | Rect Stress | Passes | Time |
| ---: | ---: | ---: | ---: | ---: |
| 256x384 | 2.214e-1 -> 2.213e-1 | 2.411e4 -> 2.411e4 | 46 | 1.14s |
| 512x768 | 4.431e-1 -> 4.430e-1 | 1.298e5 -> 1.298e5 | 44 | 2.95s |

These rectangular runs are stability/shape tests, not claims that the current
raw rectangular phase route replaces a production rectangular SVD. They prove
that the row and column Clifford views can have different dimensions and still
produce orthogonal full bases.

`0.15.0` adds a two-sided Joint SVD smoke:

| N | Family | Joint SVD Offdiag | Sweeps | Rotations | Time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 4.688e1 -> 4.582e-12 | 12 | 23384 | 0.071s |
| 128 | 4 | 1.076e2 -> 6.987e-12 | 16 | 118627 | 1.06s |

It also introduces a compact per-pass `PairEnergyCache` for PhaseFlow active
pair selection. This is not yet the full persistent invalidation scheduler,
but it is the first concrete cached rotor planner.

`0.16.0` adds a `4x4` macro-rotor route:

| N | Profile | Raw Block-4 Offdiag | Block4Polished Rel. Recon | Time |
| ---: | --- | ---: | ---: | ---: |
| 64 | `uniform_random` | 6.340e1 -> 1.823e1 | 9.950e-15 | 0.012s |
| 64 | `degenerate_spectrum` | 3.139e2 -> 1.435e1 | 7.575e-15 | 0.013s |
| 64 | `jordan_defective` | 1.592e2 -> 6.329e0 | 6.794e-15 | 0.013s |
| 64 | `kron_structured` | 2.009e1 -> 8.470e0 | 2.106e-15 | 0.011s |

The important point is architectural: `4x4` is now a reusable local phase cell,
not only a special tiny-matrix case. The block route uses the exact `N <= 4`
microkernel inside larger contiguous and butterfly quartet schedules. The
companion `Block4Signature` reports self-dual versus anti-self-dual torsion in
contiguous quartets, giving a concrete `SO(4)` phase passport without claiming
a closed-form SVD for general large matrices.

On this CPU smoke, `Block4Polished` is not a replacement for `Small` on ordinary
dense inputs. Its value is as a testable macro-cell for larger phase schedules,
tensor/Kronecker layouts, and future analog/photonic compilation.

`0.17.0` deepens the phase route:

| N | Profile | Mode | Raw PhaseFlow Offdiag | Passes | Jumps | Unwrap | Surgery |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | `uniform_random` | golden | 6.340e1 -> 2.863e-12 | 10 | 244 | 17303 | 0 |
| 64 | `uniform_random` | no-golden | 6.340e1 -> 5.521e-2 | 152 | 969 | 17107 | 0 |
| 64 | `degenerate_spectrum` | golden | 3.139e2 -> 2.931e-11 | 23 | 793 | 29998 | 0 |
| 64 | `degenerate_spectrum` | no-golden | 3.139e2 -> 2.270e0 | 14 | 475 | 26107 | 0 |
| 64 | `jordan_defective` | golden | 1.592e2 -> 6.758e-1 | 152 | 1109 | 43922 | 1 |
| 64 | `kron_structured` | golden | 2.009e1 -> 2.493e0 | 152 | 887 | 230109 | 9 |

The golden lattice is not universally faster on every profile, but on the
`uniform_random` and `degenerate_spectrum` raw traces above it breaks the old
standing-wave pattern dramatically. The `4x4` surgery path is intentionally
rare: it only accepts a quartet if global off-diagonal energy drops.

`0.18.0` adds Phase-BSS and tensor HO-SVD demos:

| Demo | Size | Result |
| --- | ---: | --- |
| `--bss-demo` | 4 channels x 1024 samples | SIR `7.997 -> 24.230 dB`, coherence `8.757e-1` |
| `--tensor-hosvd` | 16x16x16 | rel. recon `3.123e-15`, superdiag mass `9.999e-1` |

These are research demos. Phase-BSS currently uses lagged second-order
statistics plus Phase-JADE, not a complete fourth-order ICA cumulant engine.
The tensor route is HO-SVD/Tucker-like; full CP/PARAFAC phase locking is a
future layer.

`0.19.0` adds Layer-0 Golden Global Phase Dispersion to PhaseFlow. This is the
"all axes at once" version of the golden-angle idea: instead of waiting for
local pair sweeps to discover every phase lock, the solver first lays down a
deterministic irrational phase sheet over rows and columns. In `f64` this is
implemented as conflict-free real Givens rotors, not as complex storage.

| Flag | Meaning |
| --- | --- |
| `--golden-prespin` | Explicitly enable Layer-0 golden pre-spin. It is on by default for `LieSvdPhaseFlowParams`. |
| `--no-golden-prespin` | Disable the Layer-0 sheet for A/B tests. |
| `--golden-jumps` | Keep golden modulation inside regular PhaseFlow passes. |
| `--no-golden-jumps` | Disable in-pass golden modulation. |
| `--prespin-depth N` | Fix golden/causal harmonic depth instead of adaptive depth. |
| `--yinyang-cycles N` | Enable the 0.24.0 four-act Cross-Phase Yin-Yang pre-spin for `N` cycles. |
| `--phase-conjugate` | Enable the 0.25.0 state-mirrored Layer-0 phase-conjugate auto-spin. |
| `--bottleneck` | Enable the 0.25.0 maximum-energy pair queue before the ordinary active-set pass. |
| `--phase-viscosity X` | Dampen phase-conjugate/bottleneck angles; useful range `0.6..0.95`. |
| `--phase-quantization-levels N` | Snap phase-conjugate/bottleneck angles to a hardware-like phase grid. |
| `--active-set-alpha X` | 0.26.0: opt-in relative axis-energy screen (`0.0` default = exact bound only); see the 0.26.0 section below. |
| `--adaptive-viscosity` | 0.27.0: opt-in per-pair `gamma = P/(P+R)` damping for bottleneck rotors, replacing the fixed `--phase-viscosity`; see the 0.27.0 section below. |

Observed `N=64` A/B with in-pass golden jumps disabled:

| Profile | Mode | Raw PhaseFlow Offdiag | Passes | Pre-spin Rotors | Time |
| --- | --- | ---: | ---: | ---: | ---: |
| `uniform_random` | pre-spin | 6.340e1 -> 5.395e-2 | 10 | 32 | 0.014s |
| `uniform_random` | no pre-spin | 6.340e1 -> 5.521e-2 | 152 | 0 | 0.052s |
| `degenerate_spectrum` | pre-spin | 3.139e2 -> 2.767e-11 | 24 | 30 | 0.023s |
| `degenerate_spectrum` | no pre-spin | 3.139e2 -> 2.270e0 | 14 | 0 | 0.018s |
| `jordan_defective` | pre-spin | 1.592e2 -> 1.228e0 | 152 | 4 | 0.134s |
| `jordan_defective` | no pre-spin | 1.592e2 -> 4.815e-1 | 152 | 0 | 0.085s |

Rectangular smoke with `64x96 --rect --golden-prespin`:

| Shape | Rect Offdiag | Passes | Pre-spin Rotors | Time |
| ---: | ---: | ---: | ---: | ---: |
| 64x96 | 5.514e-2 -> 5.494e-2 | 28 | 24 | 0.021s |

Design note: golden pre-spin is an anti-resonance initialization, not a
closed-form SVD. It is meant to break phase standing waves before `4x4`
butterfly cells, targeted unwrap rotors, and optional digital polish.

`0.23.0` adds the causal/Jordan counterpart to this idea. If triangular
causal disbalance is high, PhaseFlow applies **Causal Anti-Spin** instead of
the isotropic golden sheet:

```text
row pair rotor:  +theta
col pair rotor:  -theta
```

Observed local A/B on `N=64 --phaseflow --no-golden-jumps`:

| Mode | Jordan Raw Offdiag | Raw Recon | Layer-0 Events |
| --- | ---: | ---: | ---: |
| causal anti-spin | `1.592e2 -> 2.485e-1` | `1.560e-3` | `causal=4` |
| no causal anti-spin | `1.592e2 -> 1.228e0` | `7.707e-3` | `prespin=4` |

The interpretation is deliberately narrow: golden dispersion breaks symmetric
standing waves; causal anti-spin is the directed counter-flow for one-way
Jordan-like torsion. The conservative `Small`/polished route remains the
accuracy path.

`0.24.0` adds a multi-layer version of this phase idea: **Cross-Phase
Yin-Yang**. Instead of choosing only golden dispersion or only causal
anti-spin, it cycles through row and column spaces with alternating signs:

```text
1. row golden      +theta
2. column antipod  -theta
3. row antipod     -theta
4. column golden   +theta
```

Each cycle is annealed by the golden ratio, so deeper cycles are gentler. In
the user's row/column Clifford language, this is a direct act on the two basis
families: row generators `e_i` and column generators `f_j` are phase-shifted
from opposite sides, then local `4x4`/`2x2` phase cells finish the lock. In
code it is still ordinary real `f64` Givens rotations with monotone acceptance.

Example A/B:

```bash
cargo run --release --bin stress_cpu --locked -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 1
cargo run --release --bin stress_cpu --locked -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 2
cargo run --release --bin stress_cpu --locked -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 3
```

The hardware compiler exports these acts as `"cross_phase_yinyang"` events,
so the same schedule can be inspected as an MZI/FPGA layer sequence. Current
`N=64` raw PhaseFlow measurements are intentionally mixed:

| Profile | Default Layer-0 | Yin-Yang 1 Cycle | Yin-Yang 2 Cycles |
| --- | ---: | ---: | ---: |
| `degenerate_spectrum` | `3.139e2 -> 2.952e-11` | `3.139e2 -> 2.930e-11` | `3.139e2 -> 2.947e-11` |
| `jordan_defective` | `1.592e2 -> 4.382e-1` | `1.592e2 -> 1.259e0` | `1.592e2 -> 1.118e0` |
| `nearly_diagonal` | `4.490e-5 -> 1.327e-13` | `4.490e-5 -> 1.326e-13` | `4.490e-5 -> 1.326e-13` |

So this is a new experimental actuator and schedule primitive, not a new
default for every hard matrix. The directed causal antipode still wins on the
current Jordan generator; the cross cycle is useful for probing combined
standing-wave plus row/column counter-flow hypotheses.

`0.25.0` moves from prescribed harmonics to **state-driven cancellation**:
Phase-Conjugate Auto-Spin reads the current row/column phase portrait and
applies a mirrored counter-phase rotor, while Bottleneck Phase Alignment uses a
maximum-energy rule to try the most resonant pair before ordinary active-set
sweeps. The angles can be damped with `--phase-viscosity` and quantized with
`--phase-quantization-levels`, which makes the exported schedule closer to a
real MZI/FPGA phase grid. These modes remain opt-in because they are research
actuators, not yet the conservative default route.

Observed `N=64 --phaseflow --no-golden-jumps`:

| Profile | Default Layer-0 | Phase-Conjugate | Bottleneck `0.8` | Conjugate + Bottleneck `0.8` |
| --- | ---: | ---: | ---: | ---: |
| `uniform_random` | `6.340e1 -> 3.775e-2` | `2.504e-12` | `2.853e-1` | `5.298e-2` |
| `degenerate_spectrum` | `3.139e2 -> 2.952e-11` | `1.225e-7` | `2.929e-11` | `3.002e-11` |
| `jordan_defective` | `1.592e2 -> 4.382e-1` | `1.636e0` | `3.717e-1` | `3.336e-2` |
| `sparse_structured` | `1.001e1 -> 2.965e0` | `3.056e0` | `1.080e0` | `8.969e-1` |

The most interesting 0.25.0 signal is the Jordan case: the combined
phase-conjugate plus bottleneck route reduces raw off-diagonal energy to
`3.336e-2` in `20` passes, where the default layer took `152` passes to reach
`4.382e-1`.

`0.26.0` adds `hot_axes`, an exact per-axis pruning certificate used by every
pair-candidate builder in `PhaseFlow`. The bound is simple: no entry of row
`k` can exceed `‖row_k‖₂`, and no entry of column `k` can exceed `‖col_k‖₂`,
so for `axis_energy_k = row_norm_k + col_norm_k`:

```text
pair_offdiag(i, j) = |core[i,j]| + |core[j,i]| <= min(axis_energy_i, axis_energy_j)
```

Any axis at or below `pair_tol` is provably below the acceptance threshold
for every pair touching it and can be dropped from search without reading
`core`. This corrects an earlier draft of the idea that proposed a
Cauchy-Schwarz-style bound `|a_ij| <= sqrt(row_energy_i * col_energy_j)`,
which does not hold in general (`a_ij` is one matrix entry, not an inner
product of the full row and column vectors).

An opt-in `active_set_alpha` (default `0.0`) layers a second, heuristic
Strong-Rules-style relative floor `alpha * max(axis_energy)` on top, the same
active-set screening idea LASSO/glmnet use for gradient magnitude, applied
here to axis energy.

Measured honestly on `N=300 --phaseflow --bottleneck` across all seven stock
profiles: the exact bound changes zero accuracy numbers (expected — it's a
no-op whenever the certificate can't prove anything) and produces no
measurable wall-clock change either. Sweeping `active_set_alpha` up to `0.6`
on `uniform_random` likewise showed no measurable time change, though total
allocated bytes dropped (`1301 MB -> 441 MB`) as candidate buffers shrank.
The reason is structural, not a bug: these synthetic profiles have fairly
uniform row/column energy (concentration of measure on dense random-like
inputs), so there are no genuinely cold axes to prune. Both bounds are
shipped as a free, provably-safe floor for inputs that do have real energy
imbalance — sparse, block-structured, or power-law-degree operators — which
the current stress harness does not generate, rather than as a claimed
speedup on this release's own benchmark suite.

`0.27.0` adds four whole-matrix "global phase invariant" scalars
(`lie_svd_phasehealth::global_phase_invariants`) that sit next to the
existing per-row/per-column diagnostics: `global_phase` (mass-weighted
circular mean of every row's and column's phase-delay angle), `torsion_energy`
(`H_total = ||skew(A)||_F`, the raw antisymmetric energy), `chirality_balance`
(self-dual/anti-self-dual bivector balance, reusing
`lie_svd_block4::analyze_block4_signature` rather than recomputing the `SO(4)`
split), and `phase_entropy` (normalized Shannon entropy of the whole matrix's
energy distribution). They're exported through `PhaseSignature` and
`lie_svd_engine::PhasePassport`. Several other ideas floated for this release
were not implemented because they already exist under different names:
trace-as-scalar-mass and torsion-as-skew-part are the `E,F,G,H` split in
`lie_svd_quadenergy`; per-row/per-column entropy and twist are already
`PhaseHealthSummary`.

`0.27.0` also adds `LieSvdPhaseFlowParams::use_adaptive_viscosity`
(`--adaptive-viscosity`, default `false`): a per-pair damping gain
`gamma = P / (P + R)` for the bottleneck rotor path, where `P` is the
candidate pair's own energy and `R` is the current pass's mean row/column
stress. This corrects an earlier draft of the idea that called it a "Kalman
filter" — it isn't one; there is no state estimate or covariance propagated
across passes, just a per-pass signal/background ratio, so it's named
literally as an energy-ratio gain instead.

Measured honestly on `N=64 --phaseflow --bottleneck --no-golden-jumps` (raw,
pre-digital-polish route):

| Profile | Fixed viscosity `0.8` | Adaptive `gamma` | Bottleneck rotations (fixed -> adaptive) |
| --- | ---: | ---: | ---: |
| `jordan_defective` | `8.451e-4` | `3.512e-3` | `455 -> 998` |
| `sparse_structured` | `2.377e-2` | `1.918e-2` | `437 -> 868` |

Adaptive viscosity roughly doubles bottleneck rotation attempts on both
profiles and gives a mixed result: markedly worse raw reconstruction on
`jordan_defective`, slightly better on `sparse_structured`. This is not a
demonstrated win, so it ships disabled by default; the accuracy test only
confirms that the final digitally-polished result still reaches machine
precision regardless of which viscosity mode ran the raw phase-locking stage.

A quartic (`degree-4`) matrix-polynomial "Galois collapse" idea was also
discussed for `0.27.0` and intentionally dropped rather than implemented: it
conflated Abel-Ruffini solvability of scalar polynomial equations (a
statement about radicals) with Jordan-block structure of matrices, and
`A^4 + I` does not by itself yield an orthogonal factor.

`0.28.0` is two focused fixes. First, a partial, honestly-measured allocation
reduction: `row_phases_into`/`col_phases_into` reuse an existing
`Vec<AxisPhase>` buffer instead of allocating fresh ones in the `PhaseFlow`
pass loop. Measured on `uniform_random` at `N=300 --phaseflow --bottleneck`:
allocations dropped `43301 -> 40807` (`~6%`) with **no measurable wall-clock
change**. That near-zero time delta despite a real allocation drop is the
actual finding: allocations were never the dominant cost. The same trace ran
the full `624`-pass budget without converging and logged over 53 million
`BottleneckPairCache::update_axes` rescore operations — tens of millions of
`O(n)` events dwarf the ~50-100ns cost of 40k allocations. The original "90%
allocations" read of the benchmark's `allocs` column correlated with the
slowdown but wasn't its cause; the remaining per-pass candidate-vector
allocations are real but not expected to move wall-clock time much, so
fixing them is deferred rather than rushed on a now-corrected premise.

Second, a chirality-driven `Auto` dispatch trigger, calibrated on real
matrices rather than guessed: `phase_chirality_balance` (0.27.0) cleanly
separated a synthetic causal-Jordan test case (`~0.38`) from
`nearly_diagonal`/`uniform_random`/block-structured cases (`~0.0-0.03`),
while the originally proposed `phase_entropy`-based fast-path did not — the
causal case's whole-matrix entropy (`~0.52`) sat almost as low as the
nearly-diagonal case's (`~0.49`), since both are sparse band matrices, so a
blanket "low entropy means skip geometric routes" rule would have risked
misrouting real causal/Jordan inputs. `phase_entropy` is exposed on the
triage for visibility but deliberately not used to gate routing. The first
version of the chirality trigger (no `diagonal_dominance` guard) caught a
real regression before it shipped: it fired on `sparse_structured` at `N=64`,
routing an already machine-precision, diagonally-dominant case through
`PhaseFlow` for a **100x slowdown** (`0.007s -> 0.767s`) with no accuracy
gain. Adding `diagonal_dominance < 0.5` fixed it, re-verified against all
seven stock profiles at `N=32` and `N=64`.

`0.29.0` fixes the actual bottleneck `0.28.0`'s profiling found:
`BottleneckPairCache`'s eager `O(touched_axes * n)` rescoring on every
accepted rotor, not allocations. `update_axes` now bumps a per-axis
generation counter (`O(1)` per touched axis) instead of eagerly rescoring and
re-heapifying every `(touched_axis, other)` pair; staleness is resolved
lazily, only when a pair is actually about to be popped as a candidate
(`pop_verified_root`, which verifies and corrects before trusting an entry as
the true max, looping until it returns a genuinely fresh one).

Measured in three honest stages on `uniform_random` at
`N=300 --phaseflow --bottleneck`:

| Stage | Rescore ops | Wall time | Raw rel. recon |
| --- | ---: | ---: | ---: |
| Original (eager) | `53,375,387` | `4.944s` | `1.272e-1` |
| Pure lazy, no rebuild | `181,721` (`294x`) | `4.199s` (`-15%`) | `4.025e-1` (`3.2x` worse) |
| + periodic rebuild (`period=16`, default) | `349,101` (`153x`) | `4.592s` (`-7%`) | `1.522e-1` (roughly on par) |

Pure lazy invalidation can't *discover* a pair between two axes that were
both cold at the last rebuild — the old eager scheme's `update_pair` inserted
newly-touched combinations on the fly; lazy invalidation only re-verifies
pairs already in the heap. That gap showed up directly as worse raw
(pre-digital-polish) convergence on `uniform_random`, `degenerate_spectrum`
(`5.881e-2 -> 1.316e-1`), and `jordan_defective` (`2.010e-1 -> 4.988e-1`).
`LieSvdPhaseFlowParams::bottleneck_cache_refresh_period` (default `16`) fixes
this by doing a full `rebuild` every N passes instead of a lazy flush,
bounding the discovery gap to a small window. With it, two of the three
profiles ended up *more* accurate than the original eager cache
(`degenerate_spectrum`: `5.881e-2 -> 1.923e-2`; `jordan_defective`:
`2.010e-1 -> 7.148e-2`), not less, while still cutting the identified
bottleneck two orders of magnitude. The honest net conclusion: the
wall-clock win is real but modest (`7-16%`, not what a `150x+` rescoring
drop alone would suggest) — `0.28.0`'s finding holds here too, since a large
share of `PhaseFlow`'s remaining per-pass cost is the `O(n)` rotor
application and line-search work in `accept_offdiag_rotor`, untouched by this
release. `jordan_defective`'s wall time barely moved (`6.654s -> 6.673s`)
because its bottleneck-rotation count is small (`2284` vs `uniform_random`'s
`39965`) — the cache was never its dominant cost.

`0.22.0` adds the unified phase engine, hardware schedule compiler, `--full-suite`
CLI smoke, and a stricter complex Hermitian/QR polish path:

| Demo | Rel. Recon | Unitary U | Unitary V | Pre-spin | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex_iq`, N=16 | 3.704e-14 | 1.935e-8 | 1.409e-13 | 16 | Hermitian recompute + guarded polar polish |
| `complex_degenerate`, N=16 | 2.251e-15 | 4.808e-8 | 1.004e-14 | 16 | repeated complex phase spectrum |

The complex branch is now much more stable than `0.20.0`, where the same smoke
showed `U` unitarity around `1e-3..1e-2`. It is still not advertised as a
production complex SVD: a dedicated complex QDWH or Householder/bidiagonal
route is the remaining path to uniformly machine-tight `U^H U` on hard dense
I/Q tails.

`--full-suite` runs an integrated smoke of the ecosystem:

| Component | N=16 Smoke Result |
| --- | --- |
| Phase-JADE | offdiag `2.537e0 -> 9.848e-14` |
| Joint SVD | offdiag `1.267e1 -> 9.231e-14` |
| Phase-BSS | SIR `7.997 -> 24.230 dB` |
| Tensor HO-SVD | rel. reconstruction `3.123e-15` on `16x16x16` |
| PhaseEngine real | `RealPhaseFlowPolished`, rel. reconstruction `1.921e-15` |
| Hardware compiler | `989` events, `10` layers, JSON schedule `247287` bytes |

Reproduce:

```bash
cargo run --release --bin stress_cpu --locked -- 16 --complex-svd --diagnostics-only
cargo run --release --bin stress_cpu --locked -- 16 --full-suite --diagnostics-only
```

Remaining performance work: the in-place line search removes the worst
allocation pressure, but the next speed step is a true conflict-free batch
apply with cached pair energies and explicit SIMD-friendly row/column kernels.

SIMD note: this release still relies on Rust/LLVM and the release profile
(`target-cpu=native` in Docker) rather than hand-written AVX2/NEON kernels.
Explicit SIMD micro-rotors are a separate future step.

On `N=4`, `Auto` exercises the `LieSvdMicro` dispatcher path. On larger
matrices, `CoreFlow` is explicit and remains a research/prototype route rather
than the default solver.
