# Technical Report: PhaseFlow Pair-Selection Engineering (0.26.0-0.29.0)

**Author:** Dr. Artiom A. Kovnatsky, Lie/Clifford Analog SVD contributors
**Software:** `lie_cliffalg_analog_svd` v0.29.0
**Repository:** <https://codeberg.org/skovnats/lie_cliffalg_analog_svd>
**License:** SVE Meta-License v5.0 (see `License` in the repository)
**Date:** 2026-08-14

**Abstract.** Four consecutive point releases (0.26.0-0.29.0) of a
research SVD solver's `PhaseFlow` pair-selection engine are documented with
measured before/after results only. Findings: (1) an exact axis-energy
pruning certificate changes no accuracy and no measured time on the
project's stock benchmark suite, because that suite lacks genuinely
low-energy axes; (2) an initial diagnosis attributing most of the
`PhaseFlow`-vs-baseline runtime gap to heap allocations was tested directly
and found incorrect — allocation count dropped 6% for no measurable time
change; (3) the actual dominant cost was identified as `O(n)` eager
rescoring in a candidate cache on every accepted rotor (53 million such
operations in one traced run); (4) a lazy-invalidation fix cut that
rescoring 150-300x, but a pure version of it measurably degraded
convergence quality, requiring a periodic-rebuild correction to recover it.
Two proposed directions (a quartic matrix-polynomial preconditioner framed
via Galois theory; a "Kalman filter" framing for an adaptive damping term)
were evaluated and not implemented, with the specific mathematical or
terminological error identified in each case.

Internal engineering report. Every number below is a measured result from
this repository's own test suite or `stress_cpu` benchmark, reproducible
with the commands given. This report makes no claims beyond what was run
and observed. It does not use "paradigm shift" framing, does not claim
novelty relative to the broader numerical linear algebra literature, and
explicitly documents what was tried and did not work alongside what did.

Scope: the `PhaseFlow` pair-selection and candidate-caching machinery in
`src/lie_svd_phaseflow.rs`, and the `Auto` dispatcher in
`src/lie_svd_adaptive.rs`. Not in scope: the tabular/relational-algebra,
Galois-polynomial, and Kalman-filter-branded ideas discussed earlier in this
line of work — none of that was implemented, for reasons given inline where
relevant.

## 1. Baseline correctness

Across all four releases in this arc, the full test suite passed at every
step (83 -> 86 -> 91 -> 94 -> 96 tests as features were added), and
`cargo fmt --check` stayed clean. Reproduce:

```bash
cargo fmt --check
cargo test --release --lib --locked
```

Orthogonality on the stock `N=300` stress profiles (`uniform_random`,
`degenerate_spectrum`, `jordan_defective`, `sparse_structured`,
`nearly_diagonal`, `extreme_ill_conditioned`, `kron_structured`) stays at
`orth_u`, `orth_v` in the `1e-13` to `1e-14` range under
`solve_with_digital_polish` throughout this work — machine precision for
`f64`, unaffected by any of the changes below. Reproduce:

```bash
cargo run --release --bin stress_cpu -- 300 --phaseflow --bottleneck
```

## 2. 0.26.0: exact axis-energy pruning (`hot_axes`)

**Claim, proven:** for row `k` and column `k`, no entry of row `k` exceeds
`rows[k].norm` (`= ||row_k||_2`) and no entry of column `k` exceeds
`cols[k].norm`. Therefore for any pair `(i, j)`:

```
pair_offdiag(i, j) = |core[i,j]| + |core[j,i]|
                   <= min(axis_energy_i, axis_energy_j)
    where axis_energy_k = row_norm_k + col_norm_k
```

Any axis with `axis_energy_k <= pair_tol` can be dropped from every pair
candidate builder without reading `core`. This is exact, not a heuristic: it
never discards a pair that could score above `pair_tol`. Implemented in
`hot_axes` (`src/lie_svd_phaseflow.rs`), used by `PairEnergyCache::square`,
`bottleneck_pairs`, `BottleneckPairCache::rebuild`, `active_phase_pairs`,
`active_rectangular_corridor_pairs`.

**Correction of an earlier draft:** a prior version of this idea proposed
`|a_ij| <= sqrt(row_energy_i * col_energy_j)` (a claimed Cauchy-Schwarz
bound). That bound does not hold in general: `a_ij` is a single matrix
entry, not an inner product of the full row and column vectors. The bound
actually implemented and used is the one above.

**Measured impact:** on the seven stock `N=300` profiles, this pruning
changed zero accuracy numbers (expected — a correct exact bound is a no-op
whenever it can't prove anything) and produced no measurable wall-clock
change, because none of those profiles have axes with energy near the
`pair_tol` machine-noise floor. It ships as a free, provably-safe floor for
inputs that do have real energy imbalance (sparse, block-structured,
power-law-degree operators), which the current stress harness does not
generate — not as a demonstrated speedup on this suite.

An opt-in relative extension, `LieSvdPhaseFlowParams::active_set_alpha`
(Strong-Rules-style, `alpha * max(axis_energy)`), is a heuristic layered on
top, disabled by default (`0.0`). It is not used to gate any behavior
change in this codebase by default.

## 3. 0.27.0: global phase invariants and adaptive viscosity

`lie_svd_phasehealth::global_phase_invariants(a)` computes four whole-matrix
scalars: `global_phase` (mass-weighted circular mean phase angle),
`torsion_energy` (`H_total = ||skew(A)||_F`), `chirality_balance`
(self-dual/anti-self-dual bivector balance, reusing
`lie_svd_block4::analyze_block4_signature`), `phase_entropy` (normalized
Shannon entropy of the whole matrix's energy distribution). Exported through
`PhaseSignature` and `lie_svd_engine::PhasePassport`.

`LieSvdPhaseFlowParams::use_adaptive_viscosity` replaces the fixed
`phase_viscosity` damping constant with a per-pair gain
`gamma = P / (P + R)` (`P` = candidate pair's own energy, `R` = current
pass's mean row/column stress) for bottleneck rotors. Named literally as an
energy-ratio gain — it is not a Kalman filter; no state covariance is
propagated across passes.

**Measured, `N=64 --phaseflow --bottleneck --no-golden-jumps`, raw
(pre-digital-polish) route:**

| Profile | Fixed viscosity `0.8` | Adaptive `gamma` | Bottleneck rotations |
| --- | ---: | ---: | ---: |
| `jordan_defective` | `8.451e-4` | `3.512e-3` (worse) | `455 -> 998` |
| `sparse_structured` | `2.377e-2` | `1.918e-2` (better) | `437 -> 868` |

**What did not work:** adaptive viscosity roughly doubles bottleneck
rotation attempts and gives a mixed result — worse on one profile, better on
the other. Not a demonstrated win. Ships disabled by default
(`use_adaptive_viscosity: false`). The accuracy test for this feature only
confirms the final digitally-polished result still reaches machine precision
regardless of which mode ran the raw phase-locking stage — it does not claim
the raw route itself improved.

## 4. 0.28.0: allocation reduction (partial) and dispatch fix

### 4.1 Allocation reduction

`row_phases_into`/`col_phases_into` reuse an existing `Vec<AxisPhase>`
buffer instead of allocating fresh ones, wired into the square and
rectangular `PhaseFlow` main pass loops (which recompute row/column phase
once or twice per pass).

**Measured, `uniform_random` at `N=300 --phaseflow --bottleneck`:**
allocations `43,301 -> 40,807` (`~6%` reduction), wall time `4.944s` vs.
`4.7-4.9s` range before — **no measurable change**.

**What this proved:** the near-zero time delta despite a real allocation
drop is the actual finding. Allocations were never the dominant cost. The
same trace ran the full `624`-pass budget without converging and logged
`cache_updates=53,375,387` from `BottleneckPairCache::update_axes` alone —
tens of millions of `O(n)` rescore operations, on top of hundreds of
thousands of `O(n)` rotor applications and line-search evaluations in
`accept_offdiag_rotor`. At an estimated 50-100ns per allocation, 40k
allocations account for low-single-digit milliseconds of the multi-second
runtime; the real cost is `O(n)` compute repeated far more often than the
allocation count. The original "allocations explain ~90% of the
`PhaseFlow`-vs-`Small` time gap" hypothesis, drawn from the benchmark's
`allocs`/`alloc_mb` columns, correlated with the slowdown but was not its
cause.

### 4.2 `Auto` dispatch: `phase_chirality_balance` trigger

`AdaptiveTriage` gained `phase_torsion_energy`, `phase_chirality_balance`,
`phase_entropy` from `global_phase_invariants`. `should_use_phaseflow` gained
a `strong_chirality_torsion` trigger.

**Calibration (not guessed) on four matrices** — `nearly_diagonal` (N=16
diagonal), `uniform_random` (N=24), `structured_stress_matrix` (N=24
block-diagonal), and a synthetic causal-Jordan matrix (N=24,
`a[i,i]=1, a[i,i+1]=5`):

| Matrix | `phase_entropy` | `chirality_balance` |
| --- | ---: | ---: |
| `nearly_diagonal` | `0.4878` | `0.0000` |
| `uniform_random` | `0.9515` | `0.0348` |
| `structured_stress` | `0.7820` | `0.0000` |
| causal-Jordan | `0.5200` | `0.3820` |

`chirality_balance` cleanly separates the causal case (`0.382`) from every
other case (`<= 0.035`). `phase_entropy` does not: the causal case's entropy
(`0.52`) sits almost as low as the nearly-diagonal case's (`0.49`), because
both are sparse band matrices. **The originally proposed
`phase_entropy < threshold` fast-path rule was not implemented** for this
reason — it would have risked misrouting genuine causal/Jordan inputs to the
fast (non-geometric) path. `phase_entropy` is exposed on `AdaptiveTriage` for
visibility only.

**A real regression, caught before shipping:** the first version of the
chirality trigger had no `diagonal_dominance` guard. It fired on
`sparse_structured` at `N=64` (which has real structured, asymmetric-but-
bidirectional skew energy despite being diagonally dominant and already at
machine precision under `Small`), routing it through `PhaseFlow`:

```
sparse_structured  Auto (before fix)  0.767s  rel_recon=4.896e-14
sparse_structured  Auto (after fix)   0.007s  rel_recon=1.256e-14
```

A 100x wall-clock regression with no accuracy gain. Adding
`diagonal_dominance < 0.5` to the trigger's guard fixed it. Re-verified
`Auto`'s route choice against all seven stock profiles at `N=32` and `N=64`
before closing this out. Reproduce:

```bash
cargo run --release --bin stress_cpu -- 64 --auto-trace
cargo run --release --bin stress_cpu -- 64
```

## 5. 0.29.0: `BottleneckPairCache` lazy invalidation

Section 4.1 identified the real cost driver: `update_axes` did
`O(touched_axes * n)` eager rescoring (score recompute + binary-heap
sift) on every accepted rotor, because a binary heap has no cheap
`decrease-key`. The fix is the standard technique for this situation: lazy
invalidation with verify-on-pop.

- `update_axes` now bumps a per-axis generation counter, `O(1)` per touched
  axis, no inner loop, no heap operations.
- A pair `(i, j)`'s cached score is stale iff its stored generation is below
  `max(axis_gen[i], axis_gen[j])`.
- `pop_verified_root` pops the heap root; if stale, recomputes its score,
  restamps its generation, re-pushes it, and pops again — looping until a
  popped entry is already fresh (provably the current true max at that
  point).

**Measured in three stages, `uniform_random` at
`N=300 --phaseflow --bottleneck`:**

| Stage | `BottleneckPairCache` rescore ops | Wall time | Raw rel. recon |
| --- | ---: | ---: | ---: |
| 0.28.0 baseline (eager) | `53,375,387` | `4.944s` | `1.272e-1` |
| Pure lazy, no periodic rebuild | `181,721` (`294x` fewer) | `4.199s` (`-15%`) | `4.025e-1` (`3.2x` worse) |
| + periodic rebuild, `period=16` (shipped default) | `349,101` (`153x` fewer) | `4.592s` (`-7%`) | `1.522e-1` (roughly on par) |

**What did not work as a standalone fix, and why:** pure lazy invalidation
cannot discover a pair between two axes that were both cold at the last
`rebuild` — the old eager `update_pair` inserted newly-touched combinations
into the heap on the fly; lazy invalidation only re-verifies pairs already
present. This showed up directly as worse raw convergence:
`degenerate_spectrum` `5.881e-2 -> 1.316e-1`, `jordan_defective`
`2.010e-1 -> 4.988e-1` (both roughly `2-3x` worse), matching
`uniform_random`'s degradation above.

**The fix:** `LieSvdPhaseFlowParams::bottleneck_cache_refresh_period`
(default `16`) does a full `rebuild` every N passes, bounding the discovery
gap. With it, two of three profiles tested came out *more* accurate than
the original eager cache, not less (`degenerate_spectrum`
`5.881e-2 -> 1.923e-2`; `jordan_defective` `2.010e-1 -> 7.148e-2`), while
still cutting rescore operations `153x`.

**Honest net conclusion:** the wall-clock win (`7-16%`) is real but far more
modest than the `150x+` reduction in rescore operations alone would
suggest. A large share of `PhaseFlow`'s remaining per-pass cost is `O(n)`
rotor application and line-search work in `accept_offdiag_rotor`, untouched
by this change. Profiles with few bottleneck rotations to begin with
(`jordan_defective`: `2,284` accepted bottleneck rotations vs.
`uniform_random`'s `39,965`) see almost no wall-clock change, because the
cache was never their dominant cost. Reproduce:

```bash
cargo run --release --bin stress_cpu -- 300 --phaseflow --bottleneck
```

(the `phaseflow_trace` line reports `cache_updates` and `passes`; compare
against the table above.)

## 6. What was discussed and explicitly not implemented

- **Clifford relational algebra / tabular database engine** (`JOIN` as
  geometric contraction, tables as k-blades, `NULL` as nilpotent
  generators). Not implemented: no mechanism was proposed for how a
  contraction over "the same generator" would locate matching rows in large
  tables without an index — the vocabulary is evocative but does not reduce
  to a working algorithm for the actual computational problem (discrete key
  matching, one-to-many joins, non-numeric keys).
- **Quartic ("Galois collapse") matrix polynomial preconditioning**
  (`P_4(A) = A^4 + I`). Not implemented: the proposal conflated
  Abel-Ruffini solvability of scalar polynomial equations (a statement about
  radicals) with Jordan-block structure of matrices; `A^4 + I` does not by
  itself yield an orthogonal factor.
- **"Kalman filter" framing.** The one part of this family of ideas that
  corresponded to real, implementable behavior (an adaptive per-pair damping
  ratio) was built and measured (Section 3) under an accurate name,
  Adaptive Energy-Ratio Viscosity — not framed as a Kalman filter, since no
  state covariance is propagated between passes.
- **Academic preprint.** Not drafted. A document combining verified results
  (this report) with the unproven claims above, under academic/journal
  framing, would read as more authoritative than warranted.
