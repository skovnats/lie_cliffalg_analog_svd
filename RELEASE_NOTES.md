# Release Notes

## 0.29.0

Fixes the actual bottleneck 0.28.0's profiling found: `BottleneckPairCache`'s
eager `O(touched_axes * n)` rescoring on every rotor, not allocations.

Included:

- `BottleneckPairCache` switches to lazy invalidation: `update_axes` now
  bumps a per-axis generation counter (`O(1)` per touched axis, no inner
  loop, no heap operations) instead of eagerly rescoring and re-heapifying
  every `(touched_axis, other)` pair. Staleness is resolved lazily in a new
  `pop_verified_root`, which only recomputes a pair's score when that exact
  pair is about to be returned from the heap, verifying it's still the true
  current maximum before trusting it (loops internally on corrections;
  terminates because a verified pair can't go stale again until one of its
  axes is touched further).
- `LieSvdPhaseFlowParams::bottleneck_cache_refresh_period` (default `16`):
  every N passes, do a full `rebuild` instead of a lazy flush. This bounds a
  real, measured tradeoff of pure lazy invalidation: it can't discover a pair
  between two axes that were both cold at the last rebuild, only re-verify
  pairs already in the heap, so raw (pre-digital-polish) convergence quality
  measurably degraded without periodic rebuilds.
- `BottleneckPairCache::update_pair` removed (dead code once eager rescoring
  was gone).
- Two new tests: pure-lazy behavior (`bottleneck_cache_refresh_period = 0`)
  keeps exactly one refresh over 40 passes, matching the old eager cache's
  observable refresh count; periodic rebuild at the default period produces
  exactly `1 + 2` refreshes over 40 passes (initial plus periods 16 and 32).
  The existing `active_set_alpha`/`adaptive_viscosity` accuracy tests (both
  exercise the bottleneck path) continue to pass unmodified.

Design notes — measured honestly, in three stages:

1. **Pure lazy, no periodic rebuild.** On `uniform_random` at
   `N=300 --phaseflow --bottleneck`: `BottleneckPairCache` rescoring dropped
   from `53,375,387` to `181,721` (`~294x`) and wall time dropped from
   `4.944s` to `4.199s` (`~15%`). But raw reconstruction error got
   measurably *worse*: `1.272e-1 -> 4.025e-1` on `uniform_random`,
   `5.881e-2 -> 1.316e-1` on `degenerate_spectrum`, `2.010e-1 -> 4.988e-1` on
   `jordan_defective` (roughly `2-3x` worse in each case). This is the
   discovery-completeness gap stated in the design, not a bug: the eager
   scheme's `update_pair` inserted previously-absent pairs into the heap on
   touch; pure lazy invalidation never does, so a pair between two axes both
   cold at the last rebuild is invisible to it for the rest of the solve.
2. **With periodic rebuild (default period `16`).** Same benchmark:
   rescoring `53,375,387 -> 349,101` (`~153x`, still enormous), wall time
   `4.944s -> 4.592s` (`~7%`). Raw accuracy: `uniform_random`
   `1.272e-1 -> 1.522e-1` (roughly on par), `degenerate_spectrum`
   `5.881e-2 -> 1.923e-2` (*better*), `jordan_defective`
   `2.010e-1 -> 7.148e-2` (*better*, `~2.8x`). Two of three profiles ended up
   *more* accurate than the original eager cache, not less, while still
   cutting the identified bottleneck two orders of magnitude.
3. **Net honest conclusion.** The wall-clock win is real but modest
   (`7-16%`, not the dramatic number the `294x`/`153x` rescoring drops alone
   would suggest) — 0.28.0's own finding holds up under this change too: a
   large chunk of `PhaseFlow`'s remaining per-pass cost is the `O(n)` rotor
   application and line-search work in `accept_offdiag_rotor`, which this
   release does not touch. `jordan_defective`'s wall time barely moved
   (`6.654s -> 6.673s`) because its bottleneck-rotation count is small
   (`2284` vs `uniform_random`'s `39965`) — the cache was never its dominant
   cost to begin with.

## 0.28.0

Two focused fixes: a partial, honestly-measured allocation reduction in the
`PhaseFlow` pass loop, and a chirality-driven `Auto` dispatch trigger that
caught and fixed a real regression before shipping.

Included:

- `row_phases_into` / `col_phases_into`: buffer-reusing variants of
  `row_phases`/`col_phases` that `.clear()` and rewrite an existing
  `Vec<AxisPhase>` instead of allocating a fresh one. Wired into both the
  square and rectangular `PhaseFlow` main pass loops, which recompute
  row/column phase once or twice every pass.
- `lie_svd_adaptive::AdaptiveTriage` gains `phase_torsion_energy`,
  `phase_chirality_balance`, and `phase_entropy` (from 0.27.0's
  `global_phase_invariants`). `should_use_phaseflow` adds a
  `strong_chirality_torsion` trigger gated by `phase_chirality_balance`,
  `offdiag_ratio`, `phase_torsion_energy`, *and* `diagonal_dominance`; when it
  fires, `solve_phaseflow_route` now leans into Causal Anti-Spin
  (`causal_antispin_threshold` lowered, `causal_antispin_layers` raised)
  instead of leaving it to the separate triangular `causal_bias` metric alone.
- Two new tests: buffer-reuse output matches the allocating original
  (including on a second call, to check `.clear()` reuse and not just
  first-call correctness), and a synthetic-triage test confirming the new
  chirality trigger fires when expected and doesn't when `chirality_balance`
  alone drops back to baseline.

Design notes — allocation reduction, measured honestly:

- The stated goal was "eliminate the ~40k allocations that make `PhaseFlow`
  4.7ms vs `Small`'s 0.45ms at `N=300`". After wiring in the buffer reuse:
  `uniform_random` at `N=300 --phaseflow --bottleneck` went from `43301` to
  `40807` allocations (about `6%`) with **no measurable wall-clock change**
  (`4.7-4.9s` either way, within run-to-run noise). This is real but partial,
  and the near-zero time delta despite a real allocation drop is itself the
  useful finding: **allocations were never the dominant cost here.** A
  `uniform_random` `N=300` trace ran the full `624`-pass budget without
  converging and logged `cache_updates=53,375,387` from
  `BottleneckPairCache::update_axes` alone — that's tens of millions of
  `O(n)` rescore operations, on top of hundreds of thousands of `O(n)` rotor
  applications and line-search evaluations. At ~50-100ns per allocation,
  40k allocations cost low-single-digit milliseconds; the actual time is
  going into that `O(n)`-per-event compute, repeated far more often than the
  allocation count. The original "90% allocations" diagnosis, drawn from the
  `allocs`/`alloc_mb` benchmark columns alone, correlated with the slowdown
  but wasn't its cause. Remaining allocation sites (the per-pass candidate
  vectors in `active_phase_pairs`/`bottleneck_pairs`/`BottleneckPairCache`)
  are still real and still un-reused, but fixing them would not be expected
  to move wall-clock time much given this finding — that work is deferred
  rather than rushed on a false premise.

Design notes — chirality-driven dispatch:

- This was calibrated on real matrices, not guessed: a scratch check across
  `nearly_diag`, `uniform_random`, `structured_stress_matrix`, and a causal-
  Jordan test matrix showed `phase_chirality_balance` cleanly separating the
  causal case (`~0.38`) from every other case (`~0.0-0.03`), while the
  originally-proposed `phase_entropy`-based fast-path did not: the causal case
  (`~0.52`) sat almost as low as the nearly-diagonal case (`~0.49`), both
  being sparse band matrices, so a blanket "low entropy -> skip geometric
  routes" rule would have risked misrouting real causal/Jordan inputs. It's
  exposed on `AdaptiveTriage` for visibility but intentionally not used to
  gate routing.
- The first version of the chirality trigger (no `diagonal_dominance` guard)
  shipped a real regression to this test session, not to a release: it fired
  on `sparse_structured` at `N=64`, sending an already machine-precision,
  diagonally-dominant case (`diag_dom=0.967`) through `PhaseFlow` for a
  **100x wall-clock slowdown** (`0.007s -> 0.767s`) with no accuracy gain.
  Adding `diagonal_dominance < 0.5` fixed it; re-verified `Auto`'s route
  choice against all seven stock profiles at both `N=32` and `N=64` before
  and after.

## 0.27.0

Adds whole-matrix global phase invariants and an opt-in adaptive-viscosity
heuristic for the bottleneck rotor path.

Included:

- `lie_svd_phasehealth::GlobalPhaseInvariants` and
  `global_phase_invariants(a)`: four whole-matrix scalars that don't reduce
  to any single row or column:
  - `global_phase`: mass-weighted circular mean of every row's and column's
    phase-delay angle.
  - `torsion_energy`: `H_total = ||skew(A)||_F`, raw (unnormalized) torsion
    energy.
  - `chirality_balance`: self-dual/anti-self-dual bivector balance, reusing
    `lie_svd_block4::analyze_block4_signature` rather than recomputing the
    `SO(4)` split (so it only covers `4 * (n/4)` rows/cols).
  - `phase_entropy`: normalized Shannon entropy of the whole matrix's energy
    distribution, in `[0, 1]`.
- `VectorPhaseHealth` gains a `phase` field (the same deterministic one-step
  cyclic phase-delay angle already used by `lie_svd_phaseflow::axis_phase`),
  computed for free in the existing per-row/per-column scan and used to build
  `global_phase`.
- `PhaseSignature` gains a `global: GlobalPhaseInvariants` field;
  `lie_svd_engine::PhasePassport` gains matching flat
  `global_phase`/`torsion_energy`/`chirality_balance`/`phase_entropy` fields
  for both the real and complex routes (the tensor HO-SVD route, which has no
  single 2D operator to diagnose, reports `0.0` placeholders like its other
  unavailable diagnostics).
- `LieSvdPhaseFlowParams::use_adaptive_viscosity` (default `false`): replaces
  the fixed `phase_viscosity` damping constant with a per-pair adaptive gain
  `gamma = P / (P + R)` in the bottleneck rotor path (both square and
  rectangular), where `P` is the candidate pair's own energy and `R` is the
  current pass's mean row/column stress. `stress_cpu` adds
  `--adaptive-viscosity`.
- Eight new tests: four for the global invariants (torsion energy matches a
  direct `skew(A)` computation, is zero on a symmetric matrix, phase entropy
  is lower on a concentrated diagonal than a dense matrix, and all four
  invariants stay finite/in-range on a causal-flow matrix), one passport
  smoke test, one unit test for the `gamma = P/(P+R)` formula itself, and one
  accuracy test confirming `use_adaptive_viscosity` still converges to
  machine precision on `jordan_defective` and `sparse_structured`.

Design notes:

- Naming correction from an earlier draft of this idea: the adaptive gain is
  named literally as what it is, an "Adaptive Energy-Ratio Viscosity". It is
  **not** a Kalman filter — there is no state estimate or covariance
  propagated across passes, just a per-pass, per-pair signal/background
  ratio. Calling it "Kalman gain" would overclaim what the code does.
- Several ideas floated for this release turned out to already exist under
  different names and were not reimplemented: trace-as-scalar-mass and
  torsion-as-skew-part are already the `E,F,G,H` split in
  `lie_svd_quadenergy`; per-row/per-column entropy and twist are already
  `PhaseHealthSummary`; a self-dual/anti-self-dual chirality index already
  exists per-`4x4`-block in `lie_svd_block4::Block4Signature` (`chirality_balance`
  above reuses it directly rather than recomputing the `SO(4)` split at
  global scope).
- Measured honestly on `N=64 --phaseflow --bottleneck --no-golden-jumps`
  (raw, pre-digital-polish route): adaptive viscosity roughly doubles
  bottleneck rotation attempts on both profiles tested (`jordan_defective`:
  `455 -> 998`; `sparse_structured`: `437 -> 868`) and gives a **mixed**
  accuracy result — worse raw reconstruction on `jordan_defective`
  (`8.451e-4 -> 3.512e-3`, about `4x`) and slightly better on
  `sparse_structured` (`2.377e-2 -> 1.918e-2`). This is not a demonstrated
  win; it's shipped disabled by default, with the accuracy test only
  confirming that the final digitally-polished result still reaches machine
  precision regardless of which viscosity mode ran the raw phase-locking
  stage.
- A quartic (`degree-4`) matrix-polynomial "Galois collapse" idea was also
  discussed for this release and intentionally dropped: it conflated
  Abel-Ruffini solvability of scalar polynomial equations (a statement about
  radicals) with Jordan-block structure of matrices, and `A^4 + I` does not
  by itself yield an orthogonal factor. No code was written for it.

CLI examples:

```bash
cargo run --release --bin stress_cpu -- 64 --phaseflow --bottleneck --adaptive-viscosity
cargo run --release --bin stress_cpu -- 64 --full-suite --diagnostics-only
```

## 0.26.0

Adds an exact norm-bound pre-filter for pair search, plus an opt-in relative
active-set screen on top of it.

Included:

- `hot_axes`: an exact, certificate-based pre-filter used by every pair
  candidate builder in `lie_svd_phaseflow`
  (`PairEnergyCache::square`, `bottleneck_pairs`, `BottleneckPairCache::rebuild`,
  `active_phase_pairs`, `active_rectangular_corridor_pairs`). For axis `k`,
  `axis_energy_k = row_norm_k + col_norm_k` bounds `pair_offdiag(i, j)` for
  every pair touching `k`; axes at or below `pair_tol` are dropped from search
  without reading `core`.
- `AxisPhase` gains a `norm` field (the row/column L2 norm), computed for free
  alongside the existing phase-health pass.
- `LieSvdPhaseFlowParams::active_set_alpha` (default `0.0`): an opt-in
  relative floor `alpha * max(axis_energy)` layered on top of the exact bound,
  in the spirit of LASSO/glmnet active-set "strong rules". `0.0` keeps the
  exact-only behavior; the default did not change and all 83 pre-existing
  tests pass unmodified.
- `stress_cpu` adds `--active-set-alpha X`.
- Three new tests: the exact bound drops only provably-cold axes (and nothing
  else, even at `pair_tol == 0.0`), the relative screen actually drops
  low-energy axes that the exact bound alone would keep, and enabling
  `active_set_alpha` on a real stress case still converges to machine
  precision.

Design notes:

- The math behind `hot_axes` corrects an earlier draft of this idea that
  proposed a Cauchy-Schwarz-style bound `|a_ij| <= sqrt(row_energy_i *
  col_energy_j)`. That bound does not hold in general — `a_ij` is a single
  matrix entry, not an inner product of the full row and column vectors. The
  bound actually used here is simpler and tighter: an entry can't exceed the
  norm of the vector it belongs to, so `|a_ij| <= min(row_norm_i, col_norm_j)`,
  which gives the per-axis `axis_energy` certificate above.
- Measured honestly: on the seven synthetic `stress_cpu` profiles at `N=300`
  with `--phaseflow --bottleneck`, the exact bound changes zero accuracy
  numbers (by design — it's a no-op when the certificate isn't provable) and
  produces no measurable wall-clock change either, because none of those
  profiles have axes with energy near the machine-noise `pair_tol` floor.
  Sweeping `active_set_alpha` up to `0.6` on `uniform_random` at `N=300` also
  showed no measurable time change, though total allocated bytes per run
  dropped (`1301 MB -> 441 MB`) as candidate-pair buffers shrank — allocation
  *count* barely moved, so this did not translate into wall-clock savings on
  these balanced-energy inputs.
- This is a deliberately narrow, honest result: dense random/structured test
  matrices have fairly uniform row/column energy (concentration of measure),
  so there is little for either the exact or relative bound to prune. Both are
  expected to pay off on inputs with real energy imbalance — genuinely sparse,
  block-structured, or power-law-degree operators — which the current stress
  harness does not generate. `hot_axes` is shipped as a free, provably-safe
  floor for that future case rather than a claimed speedup on this release's
  benchmark suite.

CLI examples:

```bash
cargo run --release --bin stress_cpu -- 300 --phaseflow --bottleneck
cargo run --release --bin stress_cpu -- 300 --phaseflow --bottleneck --active-set-alpha 0.3
```

## 0.25.0

Adds adaptive Phase-Conjugate Auto-Spin and Bottleneck Phase Alignment.

Included:

- Version bump to `0.25.0`.
- `LieSvdPhaseFlowParams` adds:
  - `use_phase_conjugate_autospin`;
  - `max_phase_conjugate_angle`;
  - `use_bottleneck_queue`;
  - `bottleneck_pairs`;
  - `phase_viscosity`;
  - `phase_quantization_levels`.
- `LieSvdPhaseFlowTrace` adds:
  - `phase_conjugate_prespins`;
  - `bottleneck_rotations`.
- `PhaseRotorKind` adds:
  - `PhaseConjugate`;
  - `Bottleneck`.
- `stress_cpu` adds:
  - `--phase-conjugate`;
  - `--bottleneck`;
  - `--phase-viscosity X`;
  - `--phase-quantization-levels N`.
- `lie_svd_compiler` exports real phase-conjugate and bottleneck events as
  `"phase_conjugate"` and `"bottleneck"` in hardware schedules.
- New tests for:
  - phase-conjugate pre-spin event recording plus polished SVD accuracy;
  - damped/quantized bottleneck event recording plus polished SVD accuracy;
  - hardware schedule JSON naming for the new event types.

Design notes:

- Golden, causal, and Yin-Yang pre-spins are prescribed anti-resonance
  patterns. Phase-Conjugate Auto-Spin is state-driven: it reads the current
  row/column phase portrait and applies a mirrored counter-phase rotor.
- Bottleneck alignment is the Gauss-Southwell/maximum-energy rule in the
  phase-flow language: try the strongest off-diagonal pair first, then update
  only the affected row/column field through normal rotor application.
- `phase_viscosity` damps exact local angles to reduce hot-spot ping-pong.
- `phase_quantization_levels` lets experiments snap rotor angles to a
  hardware-like phase grid before acceptance.
- These modes are opt-in. They are intended for A/B exploration and hardware
  schedule research while the conservative dispatcher keeps its established
  defaults.

CLI examples:

```bash
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --phase-conjugate
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --bottleneck --phase-viscosity 0.8
cargo run --release --bin stress_cpu -- 64 --phaseflow --phaseflow-polish --phase-conjugate --bottleneck --phase-quantization-levels 256
```

Observed local A/B, `N=64 --phaseflow --no-golden-jumps`:

| Profile | Default Layer-0 | Phase-Conjugate | Bottleneck `0.8` | Conjugate + Bottleneck `0.8` |
| --- | ---: | ---: | ---: | ---: |
| `uniform_random` | `6.340e1 -> 3.775e-2` | `6.340e1 -> 2.504e-12` | `6.340e1 -> 2.853e-1` | `6.340e1 -> 5.298e-2` |
| `degenerate_spectrum` | `3.139e2 -> 2.952e-11` | `3.139e2 -> 1.225e-7` | `3.139e2 -> 2.929e-11` | `3.139e2 -> 3.002e-11` |
| `jordan_defective` | `1.592e2 -> 4.382e-1` | `1.592e2 -> 1.636e0` | `1.592e2 -> 3.717e-1` | `1.592e2 -> 3.336e-2` |
| `sparse_structured` | `1.001e1 -> 2.965e0` | `1.001e1 -> 3.056e0` | `1.001e1 -> 1.080e0` | `1.001e1 -> 8.969e-1` |

Interpretation:

- Phase-Conjugate alone is excellent on the seeded random phase field, but can
  over-mirror directed Jordan flow.
- Bottleneck alone improves directed/sparse stress because it stops wasting
  early effort on low-energy pairs.
- The combined state mirror plus bottleneck rule gives the strongest observed
  Jordan raw reduction in this release: `4.382e-1` default raw offdiag becomes
  `3.336e-2`, with passes dropping from `152` to `20`.
- Balanced degenerate spectra are already well handled by existing layer-0
  behavior, so 0.25.0 is mainly a Jordan/sparse active-set improvement.

## 0.24.0

Adds the Cross-Phase Yin-Yang Cycle to `LieSvdPhaseFlow`.

Included:

- Version bump to `0.24.0`.
- `LieSvdPhaseFlowParams` adds:
  - `use_yinyang_prespin`;
  - `yinyang_cycles`;
  - `max_yinyang_angle`.
- `LieSvdPhaseFlowTrace` adds:
  - `yinyang_prespins`;
  - `yinyang_cycles`.
- `PhaseRotorKind` adds `CrossPhaseYinYang`.
- `stress_cpu` adds:
  - `--prespin-depth N`;
  - `--yinyang-cycles N`.
- `lie_svd_compiler` exports real Yin-Yang phase events as
  `"cross_phase_yinyang"` in hardware schedules.
- New tests for:
  - Yin-Yang pre-spin event recording plus polished SVD accuracy;
  - hardware schedule JSON naming for cross-phase events.

Design notes:

- `0.19.0` introduced isotropic Golden Pre-Spin for standing-wave phase traps.
- `0.23.0` introduced directed Causal Anti-Spin for Jordan-like one-way flow.
- `0.24.0` combines both ideas into an explicit four-act cycle:

```text
1. Row Golden:     +theta on row generators
2. Column Antipod: -theta on column generators
3. Row Antipod:   -theta on row generators
4. Column Golden: +theta on column generators
```

- Each cycle is annealed by the golden ratio, so cycle `m` uses a smaller
  angle scale than cycle `m - 1`.
- The implementation remains real-valued: every phase act compiles to ordinary
  conflict-free Givens rotors over `f64` arrays.
- The layer is guarded by local off-diagonal acceptance. It is a phase actuator
  and hardware schedule primitive, not a closed-form SVD shortcut.

CLI examples:

```bash
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 1
cargo run --release --bin stress_cpu -- 64 --phaseflow --no-golden-jumps --yinyang-cycles 2
cargo run --release --bin stress_cpu -- 64 --phaseflow --phaseflow-polish --prespin-depth 3
```

Observed local A/B, `N=64 --phaseflow --no-golden-jumps`:

| Profile | Default Layer-0 | Yin-Yang 1 Cycle | Yin-Yang 2 Cycles |
| --- | ---: | ---: | ---: |
| `degenerate_spectrum` | `3.139e2 -> 2.952e-11` | `3.139e2 -> 2.930e-11` | `3.139e2 -> 2.947e-11` |
| `jordan_defective` | `1.592e2 -> 4.382e-1` | `1.592e2 -> 1.259e0` | `1.592e2 -> 1.118e0` |
| `nearly_diagonal` | `4.490e-5 -> 1.327e-13` | `4.490e-5 -> 1.326e-13` | `4.490e-5 -> 1.326e-13` |

Interpretation:

- The new cycle is stable and exportable, and it preserves the polished
  accuracy route.
- On balanced degenerate spectra it matches the existing layer-0 behavior.
- On the current Jordan stress generator, the simpler directed Causal
  Anti-Spin remains better than the four-act cross cycle. That is useful
  information: Yin-Yang should stay opt-in while the dispatcher keeps the
  causal route as the default for strongly one-way triangular flow.

## 0.23.0

Adds the causal/Jordan antipode to Golden Pre-Spin.

Included:

- Version bump to `0.23.0`.
- `LieSvdPhaseFlowParams` adds:
  - `use_causal_antispin`;
  - `causal_antispin_threshold`;
  - `causal_antispin_layers`;
  - `max_causal_antispin_angle`.
- `LieSvdPhaseFlowTrace` adds `causal_antispins`.
- `PhaseRotorKind` adds `CausalAntiSpin`.
- `stress_cpu` adds:
  - `--causal-antispin`;
  - `--no-causal-antispin`.
- `lie_svd_compiler` exports real causal anti-spin events as
  `"causal_antispin"` in hardware schedules.
- New tests for:
  - causal anti-spin recording on Jordan-like flow;
  - hardware schedule JSON naming for causal anti-spin.

Design notes:

- Golden Pre-Spin is a symmetric/irrational anti-resonance sheet, useful for
  standing-wave or balanced-degenerate phase traps.
- Jordan-like matrices are different: they carry one-sided triangular flow.
  For these, `PhaseFlow` now detects strong triangular causal bias and applies
  an asymmetric Layer-0 counter-flow instead of the isotropic golden sheet.
- The counter-flow uses opposite-sign row/column rotors and is still guarded
  by monotone off-diagonal acceptance.

Observed local A/B, `N=64 --phaseflow --no-golden-jumps`:

- default causal anti-spin on `jordan_defective`:
  raw offdiag `1.592e2 -> 2.485e-1`, `causal=4`,
  raw reconstruction `1.560e-3`;
- with `--no-causal-antispin`:
  raw offdiag `1.592e2 -> 1.228e0`, `prespin=4`,
  raw reconstruction `7.707e-3`.

This is not a claim that raw PhaseFlow beats the conservative polished SVD on
Jordan matrices. It is a targeted result: the causal layer breaks the one-way
triangular flow more effectively than isotropic golden dispersion.

Observed Docker A/B, `N=32 --phaseflow --no-golden-jumps`:

- default causal anti-spin on `jordan_defective`:
  raw offdiag `1.117e2 -> 3.803e-12`, `causal=4`;
- with `--no-causal-antispin`:
  raw offdiag `1.117e2 -> 9.736e-1`, `prespin=4`.

Observed Docker full-suite smoke, `N=16 --full-suite --diagnostics-only`:

- `PhaseEngine` real route: rel. reconstruction `1.921e-15`,
  orthogonality around `5e-15`, route `RealPhaseFlowPolished`;
- Hardware compiler smoke: `989` real phase events, `10` layers,
  JSON schedule size `247287` bytes.

## 0.22.0

Final unified phase-compiler release for the current research line.

Included:

- Version bump to `0.22.0`.
- New `src/lie_svd_engine.rs`.
- New `src/lie_svd_compiler.rs`.
- New public modules:
  - `lie_svd_engine`;
  - `lie_svd_compiler`.
- `PhaseEngine` facade:
  - `solve_real`;
  - `solve_complex`;
  - `hosvd3`;
  - `separate_bss`;
  - `diagonalize_family`.
- Unified `PhasePassport`:
  - matrix shape and family/tensor metadata;
  - mean phase stress;
  - max twist;
  - causal disbalance;
  - entropy gap;
  - chirality;
  - golden resonance;
  - route hint.
- Hardware schedule compiler:
  - `HardwareSchedule`;
  - `HardwarePhaseEvent`;
  - `HardwareTarget`;
  - real PhaseFlow event import;
  - complex PhaseFlow event import;
  - JSON export with conflict-free layers, channel indices, phase angles, and
    source/kind tags.
- `stress_cpu` adds `--full-suite`.
- Complex Hermitian polish now recomputes the tracked Gram matrix from
  `Q^H H Q` after Jacobi sweeps, preventing premature convergence from
  manually zeroed pair entries.
- Complex route adds a guarded QR/polar-style polish attempt that factors the
  provisional `U` basis and pushes the residual into a small complex core.
- New tests for:
  - hardware schedule JSON export;
  - real `PhaseEngine` passport/solve path;
  - complex `PhaseEngine` passport/solve path.

Observed Docker smoke, `N=16 --full-suite --diagnostics-only`:

- `complex_iq`: rel. reconstruction `3.704e-14`, `unitary_u 1.935e-8`,
  `unitary_v 1.409e-13`, `polar=1`.
- `complex_degenerate`: rel. reconstruction `2.251e-15`,
  `unitary_u 4.808e-8`, `unitary_v 1.004e-14`, `polar=1`.
- Phase-JADE joint diagonalization: offdiag `2.537e0 -> 9.848e-14`.
- Two-sided joint SVD: offdiag `1.267e1 -> 9.231e-14`.
- Phase-BSS: SIR `7.997 -> 24.230 dB`.
- Tensor HO-SVD, `16^3`: rel. reconstruction `3.123e-15`.
- `PhaseEngine` real route on Jordan stress: rel. reconstruction `2.964e-15`,
  orthogonality around `1e-14`, route `RealPhaseFlowPolished`.
- Hardware compiler smoke: `1465` real phase events, `16` layers,
  JSON schedule size `367009` bytes.

Design notes:

- `0.22.0` does not collapse all methods into one opaque mega-solver. It gives
  the ecosystem one dispatcher/passport/report layer while preserving separate
  specialist routes for real SVD, complex SVD, BSS, tensor HO-SVD, and joint
  diagonalization.
- Complex stability is substantially improved from `0.20.0`, but the strict
  `U^H U <= 1e-10` target is not claimed as universally closed. The remaining
  production-grade complex work is a real QDWH or bidiagonal/Householder
  polish, especially for dense I/Q tails.

## 0.20.0

Adds the first Complex-Native Phase Algebra engine.

Included:

- Version bump to `0.20.0`.
- New direct dependency: `num-complex`.
- New `src/lie_svd_complex.rs`.
- Public complex API:
  - `LieSvdComplex::solve`;
  - `LieSvdComplex::solve_with_trace`;
  - `LieSvdComplex::solve_2x2_micro`;
  - `LieSvdComplex::to_mzi_phases`;
  - `apply_complex_golden_prespin`;
  - `complex_relative_reconstruction_error`;
  - `complex_unitarity_error`.
- New complex trace/export structures:
  - `LieSvdComplexParams`;
  - `LieSvdComplexTrace`;
  - `ComplexMziPhase`;
  - `ComplexPhaseEventKind`.
- `stress_cpu` adds `--complex-svd`.
- New tests for:
  - complex SVD reconstruction;
  - complex `2x2` microkernel reconstruction;
  - complex golden pre-spin phase-event export.

Observed smoke, `N=16 --complex-svd --diagnostics-only`:

- `complex_iq`: rel. reconstruction `3.852e-14`, `unitary_v 6.168e-15`,
  `unitary_u 1.401e-3`, `prespin=16`.
- `complex_degenerate`: rel. reconstruction `4.238e-10`,
  `unitary_v 5.751e-15`, `unitary_u 1.094e-2`, `prespin=16`.

Design notes:

- In the real route, a phase shift is compiled into real `2x2` rotors. In the
  complex route, U(1) phase shifts are native scalar multiplications, closer to
  I/Q signals and photonic phase shifters.
- Complex golden pre-spin is direct row/column multiplication by
  `exp(i k theta_phi)` and its column-side golden-ratio companion.
- The current complex solve path is a research foundation, not yet a
  production-grade complex SVD. It gives machine-scale reconstruction in the
  included smoke tests, but dense complex tails still need a dedicated complex
  QDWH, polar, or bidiagonal polish to make `U` unitarity uniformly
  machine-tight.

## 0.19.0

Adds Layer-0 Golden Global Phase Dispersion to `LieSvdPhaseFlow`.

Included:

- Version bump to `0.19.0`.
- `LieSvdPhaseFlowParams` adds:
  - `use_golden_prespin`;
  - `golden_prespin_layers`;
  - `max_prespin_angle`.
- `LieSvdPhaseFlowTrace` adds `golden_prespins`.
- `PhaseRotorKind` adds `GoldenPreSpin` for MZI/photonic schedule export.
- Square PhaseFlow now optionally applies a guarded global golden pre-spin
  sheet before the usual directional, golden-jump, active-set, and `4x4`
  surgery stages.
- Rectangular PhaseFlow applies independent row-space and column-space golden
  pre-spin sheets, respecting different `N x M` dimensions.
- `stress_cpu` adds:
  - `--golden-prespin`;
  - `--no-golden-prespin`.
- New test for golden pre-spin trace recording and polished reconstruction.

Design notes:

- This is the "all axes first" form of the golden-angle idea. The solver lays
  an irrational Fibonacci/golden phase lattice over rows and columns before
  local phase relaxation.
- The implementation stays real-valued: Clifford/phase language compiles to
  ordinary conflict-free Givens rotors over `f64` arrays.
- Golden pre-spin is an anti-resonance initialization, not a closed-form SVD.
  It is intended to break standing-wave phase traps before `4x4` macro cells
  and active `2x2` unwrap rotors take over.

Observed smoke:

- `N=64 degenerate_spectrum`, with in-pass golden jumps disabled:
  pre-spin raw offdiag `3.139e2 -> 2.767e-11` with `30` accepted pre-spin
  rotors; no-pre-spin raw offdiag `3.139e2 -> 2.270e0`.
- `N=64 uniform_random`, with in-pass golden jumps disabled:
  pre-spin needed `10` passes versus `152` without pre-spin for a comparable
  raw offdiag floor.
- `64x96 --rect --golden-prespin`: rectangular route accepted `24` pre-spin
  rotors and preserved full row/column output shape.

## 0.18.0

Adds Phase-BSS and the first higher-order tensor phase factorization route.

Included:

- Version bump to `0.18.0`.
- New `src/lie_svd_bss.rs`.
- New `src/lie_svd_tensor.rs`.
- `LieSvdBss::separate`:
  - centers and whitens observed channels;
  - builds lagged covariance matrices;
  - applies the existing Phase-JADE joint diagonalizer;
  - returns an unmixing matrix and separated channels.
- New BSS metrics:
  - `channel_phase_coherence`;
  - `estimate_sir_db`.
- `LieSvdTensor::hosvd3`:
  - builds mode-wise Gram matrices;
  - solves each mode with the robust SVD path;
  - rotates a 3D tensor into a Tucker-style core.
- New tensor helpers:
  - `reconstruct_hosvd3`;
  - `tensor_relative_error`.
- `stress_cpu` adds:
  - `--bss-demo`;
  - `--tensor-hosvd`.
- New tests for:
  - synthetic BSS SIR improvement;
  - 3D HO-SVD reconstruction.

Observed smoke:

- `--bss-demo`, 4 channels x 1024 samples:
  SIR `7.997 -> 24.230 dB`, channel coherence `8.757e-1`,
  joint offdiag `1.823e0 -> 3.271e-3`.
- `--tensor-hosvd`, 16x16x16:
  relative reconstruction `3.123e-15`, superdiagonal mass `9.999e-1`.

Design notes:

- Phase-BSS currently uses lagged second-order covariance families plus
  Phase-JADE. It is a working BSS bridge, not yet a full fourth-order cumulant
  ICA engine.
- Tensor phase factorization is currently HO-SVD/Tucker-like. Full
  CP/PARAFAC phase locking remains future work.

## 0.17.0

Deepens PhaseFlow with phase-guided landmarks, golden-angle anti-resonance
jumps, and guarded `4x4` flow surgery.

Included:

- Version bump to `0.17.0`.
- `TopologicalWarmStartParams` adds `phase_landmark_count`.
- `LieSvdTopoWarm` now seeds landmarks from high local phase stress before
  filling the rest by farthest-point landmarks.
- `LieSvdBlock4` can apply the phase-guided TopoWarm before contiguous and
  butterfly quartet layers.
- `LieSvdPhaseFlowParams` adds:
  - `use_golden_jumps`;
  - `enable_flow_surgery`.
- `LieSvdPhaseFlowTrace` adds `surgery_blocks`.
- Global phase jumps are modulated by a deterministic golden-angle lattice.
- High-stress plateaus can trigger a guarded local `4x4` surgery cell.
- `stress_cpu` adds:
  - `--topo-warm` alias for `--topowarm`;
  - `--golden-jumps`;
  - `--no-golden-jumps`.
- New tests for:
  - phase-guided landmark selection;
  - golden-jump polished accuracy.

Observed `N=64 --phaseflow` smoke:

- `uniform_random`: golden raw offdiag `6.340e1 -> 2.863e-12` in `10` passes;
  no-golden raw offdiag `6.340e1 -> 5.521e-2` in `152` passes.
- `degenerate_spectrum`: golden raw offdiag `3.139e2 -> 2.931e-11` in `23`
  passes; no-golden raw offdiag `3.139e2 -> 2.270e0`.
- `jordan_defective`: golden raw offdiag `1.592e2 -> 6.758e-1`, with one
  accepted `4x4` surgery block.
- `kron_structured`: golden raw offdiag `2.009e1 -> 2.493e0`, with nine
  accepted `4x4` surgery blocks.

Design notes:

- Golden jumps are deterministic anti-resonance modulation, not random search.
- Flow surgery is deliberately guarded: a quartet is kept only if it lowers
  global off-diagonal energy.
- The raw phase route still is not the universal fastest CPU path. The result
  is most interesting as a deeper phase-locking mechanism and as a future
  analog/photonic schedule primitive.

## 0.16.0

Adds the first `4x4` macro-rotor route.

Included:

- Version bump to `0.16.0`.
- New `src/lie_svd_block4.rs`.
- New `LieSvdBlock4Params` and `LieSvdBlock4Trace`.
- `LieSvdBlock4::warm_start_with_trace` applies local `4x4` SVD cells to:
  - contiguous quartets;
  - shifted quartets;
  - power-of-two butterfly quartets.
- `LieSvdBlock4::solve` and `solve_with_digital_polish` finish the warmed core
  with the robust `LieSvdSmall` polish.
- New `analyze_block4_signature` splits local `4x4` skew/torsion into
  self-dual and anti-self-dual `SO(4)` components.
- `stress_cpu` adds `--block4`.
- New tests for:
  - raw block-diagonal `4x4` energy reduction;
  - polished accuracy on random `8x8` inputs;
  - butterfly layer generation.
  - self-dual/anti-self-dual signature splitting.

Design notes:

- This is the practical version of the `4x4` / `SO(4)` idea: the `4x4` cell is
  now a reusable macro-rotor primitive for larger matrices.
- It is not presented as a closed-form SVD for general `N >= 5`. The block
  stage is a geometric warm start; final exactness still comes from digital
  polish.
- The butterfly quartets are the first explicit powers-of-two block schedule in
  the crate and connect naturally to tensor/Kronecker layouts.

Observed `N=64 --block4` smoke:

- `uniform_random`: raw offdiag `6.340e1 -> 2.325e1`,
  polished rel. recon `2.724e-14`.
- `degenerate_spectrum`: raw offdiag `3.139e2 -> 9.121e1`,
  polished rel. recon `1.069e-14`.
- `jordan_defective`: raw offdiag `1.592e2 -> 2.033e0`,
  polished rel. recon `1.435e-14`.
- `kron_structured`: raw offdiag `2.009e1 -> 8.470e0`,
  polished rel. recon `2.106e-15`.

The CPU result is deliberately described as architectural rather than a speed
claim: `Block4Polished` is currently slower than `Small` on ordinary dense
inputs, but it makes the `4x4` phase cell explicit and measurable.

## 0.15.0

Adds a compact cached pair-energy scheduler for PhaseFlow and the first
two-sided Joint SVD route for matrix families.

Included:

- Version bump to `0.15.0`.
- `LieSvdPhaseFlow` now builds a per-pass `PairEnergyCache` for active
  conflict-free pair selection.
- The cache stores pair scores derived from off-diagonal coupling, row/column
  stress, and entropy gap, then emits a non-overlapping active layer.
- `LieSvdJoint::joint_svd` and `joint_svd_with_params` implement a two-sided
  family route for `U^T A_k V`.
- New `JointSvdTrace`.
- New tests for:
  - square nonsymmetric two-sided Joint SVD;
  - rectangular Joint SVD shape/orthogonality acceptance.
- `stress_cpu` adds `--joint-svd`.

Observed smoke result:

- `N=64, family=4` two-sided Joint SVD:
  `4.688e1 -> 4.582e-12` joint offdiag in `~0.07s`.
- `N=128, family=4` two-sided Joint SVD:
  `1.076e2 -> 6.987e-12` joint offdiag in `~1.06s`.

Design notes:

- The cached scheduler is intentionally conservative: it is per-pass rather
  than a full invalidation graph across passes. It is the first safe step
  toward the larger cached/batched rotor planner.
- Square two-sided Joint SVD is the strongest new result in this release.
- Rectangular Joint SVD currently rotates the diagonal corridor and preserves
  full row/column orthogonality, but it does not yet solve the extra-column
  scheduling problem. That requires a dedicated right-extra active scheduler.

## 0.14.0

Adds rectangular phase-flow support, larger diagnostic smoke routes, and
explicit Joint/Phase-JADE coverage for both two matrices and arbitrary
matrix-family sizes.

Included:

- Version bump to `0.14.0`.
- `PhaseSignature` is now explicitly regression-tested on rectangular
  operators.
- `LieSvdPhaseFlow::phase_lock_rectangular_with_trace` supports `N x M`
  phase-locking with distinct row and column spaces.
- Rectangular output follows full SVD shapes:
  `U: N x N`, `sigma: min(N,M)`, `Vt: M x M`.
- `LieSvdPhaseFlowParams::for_n` now switches large matrices (`N >= 256`) to a
  hierarchical active-set budget instead of selecting every axis.
- Full conflict-free sweeps are skipped for large phase-flow runs; high-stress
  active axes remain enabled.
- `LieSvdJoint` now has explicit tests for:
  - `m = 2` matrices;
  - the previous medium family case;
  - a larger arbitrary `m` family.
- `stress_cpu` adds:
  - `--rect`;
  - `--rect-cols M`;
  - `--diagnostics-only` for quick large Joint/Rect smoke without the full SVD
    profile table.

Observed smoke result:

- `Phase-JADE N=256, family=6`: joint offdiag
  `1.198e1 -> 8.459e-12` in `~11.24s`.
- Rectangular `256x384`: phase route completed in `~1.14s`.
- Rectangular `512x768`: phase route completed in `~2.95s`.
- Full release unit tests: `58 passed`.

Design notes:

- Rectangular phase-flow is currently an active phase diagnostic and
  pre-locking route, not yet a production replacement for a polished
  rectangular SVD.
- The key conceptual step is now in code: row generators `e_i` and column
  generators `f_j` no longer need the same cardinality.
- The next real performance step is cached pair-energy scheduling across
  passes plus actual parallel batch apply. The present active-set layer is the
  first conservative version of that direction.

## 0.13.0

Turns the `PhaseFlow` hot path from clone-based probing into in-place
phase-rotor updates, and adds the first tested Phase-JADE joint
diagonalization prototype.

Included:

- Version bump to `0.13.0`.
- `LieSvdPhaseFlow` now accepts trial rotors in-place:
  apply the left/right rotor, evaluate the local two-axis off-diagonal delta,
  and roll back with the inverse rotor on rejection.
- The old `core.clone()` inside PhaseFlow line-search has been removed from
  the acceptance loop.
- High `causal_disbalance` cases now get a directed asymmetric rotor pass
  before the regular phase-jump and unwrap sweeps. This is aimed at
  Jordan-like one-way flow.
- New `lie_svd_joint` module:
  `LieSvdJoint::diagonalize_symmetric` and
  `diagonalize_symmetric_with_params`.
- New `JointDiagonalizationParams` and `JointDiagonalizationTrace`.
- New `stress_cpu --joint` smoke benchmark for a synthetic jointly
  diagonalizable symmetric matrix family.
- New unit tests for Phase-JADE convergence and already-diagonal stability.
- `LieSvdAdaptive` keeps automatic geometric routes (`PhaseFlow` and
  `CoreFlowTopo`) capped at `N <= 64` until the batch/cache kernel lands;
  larger runs can still request `--phaseflow`/`--coreflow` explicitly.

Design notes:

- This release keeps the user's "1 writes, 2-4+ in mind" rule literal:
  matrix storage remains one ordinary `f64` state, while the row/column/dual
  phase views live in diagnostics, rotor schedules, and local energy deltas.
- The Phase-JADE path is a Cardoso/JADE-style generalization of the phase
  rotor idea: one shared basis `V` minimizes
  `sum_k ||offdiag(V^T M_k V)||_F^2` for a family of symmetric matrices.
- Nonsymmetric two-sided joint SVD (`U^T A_k V`) is the next natural extension,
  not claimed in this version.
- Remaining speed work: conflict-free batch apply, cached pair energies, and
  SIMD-friendly row/column kernels.

## 0.12.0

Promotes `LieSvdPhaseFlow` into the adaptive dispatcher and benchmark center.

Included:

- Version bump to `0.12.0`.
- New `PhaseSignature` in `lie_svd_phasehealth`:
  `mean_stress`, `max_twist`, `causal_disbalance`, and `entropy_gap`.
- `stress_cpu --phase-health` now prints the compact phase passport fields.
- `stress_cpu --auto-trace` now prints phase passport fields used by the
  dispatcher.
- New adaptive route: `AdaptiveRoute::PhaseFlow`.
- `LieSvdAdaptive` now routes phase-passport cases through `PhaseFlow`:
  repeated/clustered spectra and causal/Jordan-like flow.
- `LieSvdPhaseFlowParams` now includes `phase_resonance_tol` and
  `record_mzi_phases`.
- `LieSvdPhaseFlow::to_mzi_phases` exports the accepted rotor sequence as
  MZI/photonic phase events.
- Additional unit tests for `PhaseSignature`, `PhaseFlow` on `N=32`
  degenerate inputs, `N=64` causal/Jordan inputs, and MZI phase export.

Observed smoke result:

- `N=16 degenerate_spectrum`: `Auto` selects `PhaseFlow`; raw `PhaseFlow`
  reaches `rel_recon ~2.17e-14`.
- `N=16 jordan_defective`: `Auto` selects `PhaseFlow`; polished route reaches
  `rel_recon ~2.05e-15`.
- `N=64 degenerate_spectrum`: `Auto` selects `PhaseFlow`; polished route
  reaches `rel_recon ~4.89e-15`.
- `N=64 jordan_defective`: `Auto` selects `PhaseFlow`; polished route reaches
  `rel_recon ~1.56e-14`.
- `N=64 uniform_random`, `extreme_ill_conditioned`, `sparse_structured`,
  `nearly_diagonal`, and `kron_structured`: `Auto` stays on `Small`.

Design notes:

- `PhaseFlow` is now a central solver route, not only an opt-in experiment.
- The raw phase-lock route is strongest on degenerate and nearly diagonal
  cases; polished mode remains necessary for machine precision on random,
  Jordan, sparse, and generic tensor profiles.
- The biggest performance bottleneck is still clone-based line-search trial
  cores. The next serious optimization is in-place pair-energy delta
  evaluation plus conflict-free batch apply.
- `to_mzi_phases` is the first explicit bridge from the Rust solver to a
  possible photonic/MZI phase compiler.

## 0.11.0

Turns phase-health from a detector into an active SVD route.

Included:

- New `lie_svd_phaseflow` module.
- New public `LieSvdPhaseFlow::solve(a)` primary route.
- New `LieSvdPhaseFlow::phase_lock_with_trace(a, params)` diagnostic route.
- New `LieSvdPhaseFlow::solve_with_digital_polish(a, params)` audit-quality
  cleanup route.
- Phase-health-driven global phase jumps across adjacent axes.
- Targeted unwrap rotors for high-stress row/column axes.
- Full conflict-free phase-locking sweeps so the method is not merely a hot
  spot preconditioner.
- Monotone off-diagonal energy acceptance for proposed phase rotors.
- New CLI flags: `--phaseflow` and `--phaseflow-polish`.
- Unit tests for raw phase-flow monotonicity and polished reconstruction.

Observed smoke result:

- `N=16 degenerate_spectrum`: raw `PhaseFlow` reduced offdiag
  `1.540e2 -> 3.424e-12` and reached `rel_recon ~2.17e-14`.
- `N=16 nearly_diagonal`: raw `PhaseFlow` reduced offdiag
  `1.093e-5 -> 4.548e-14` and reached `rel_recon ~7.33e-15`.
- `N=16 jordan_defective`: raw `PhaseFlow` reduced offdiag
  `7.769e1 -> 2.527e-2`, with orthogonality around `1e-15`; the polished route
  reached `rel_recon ~2.05e-15`.
- `N=16 uniform_random`: raw `PhaseFlow` reduced offdiag
  `1.523e1 -> 5.468e-3`, reaching `rel_recon ~3.46e-4`; the polished route
  reached `~1.89e-15`.

Design notes:

- `PhaseFlow` is intentionally a first-class solver route, not just a
  preconditioner for `LieSvdSmall`.
- `PhaseFlowPolished` is kept separate. Its job is final floating-point cleanup
  and comparison, not redefining the phase-flow mechanism.
- The acceptance guard remains because finite-precision global phase jumps can
  inject off-diagonal energy on adversarial pairings. The guard is an energy
  law, not a retreat from the geometric model.
- The next high-impact optimization is to remove clone-based trial cores from
  the line search and evaluate pair energy deltas in-place.

## 0.10.0

Adds fractal row/column phase-health diagnostics.

Included:

- New `lie_svd_phasehealth` module.
- Public `analyze_fractal_phase_health(a)` helper.
- Public `analyze_vector_phase(x)` helper for inspecting one row/column-like
  signal.
- Row and column summaries for scalar mass, centered vector spread,
  deterministic phase-delay bivector proxy, cyclic gradient energy, energy
  entropy, twist ratio, and row/column disagreement.
- New CLI flag: `--phase-health`.
- Allocation-conscious implementation: matrix rows and columns are analyzed
  through indexed access instead of copying every row/column into temporary
  vectors.
- Unit tests for constant vectors, alternating vectors, and diagonal-matrix
  low-entropy behavior.

Observed smoke result:

- `N=16 degenerate_spectrum`: high total phase stress, `~1.16e4`.
- `N=16 jordan_defective`: very low row/column entropy, `~1.3e-2`, but high
  phase stress, `~2.36e3`, matching the one-way structured-flow diagnosis from
  `QuadEnergy`.
- `N=16 nearly_diagonal`: near-zero row/column entropy, `~3.8e-11`, confirming
  that this profile should stay on the conservative fast path.

Design notes:

- This release clarifies the user's "four angles of view" as a global tensor
  picture:
  `A = sum_ij a_ij e_i tensor f_j`, where rows and columns have their own
  basis-unit families, plus dual row/column contractions.
- `PhaseHealth` is the local/fractal complement: every row and every column is
  inspected as its own Clifford-like signal.
- A single row vector has no canonical bivector by itself. The module therefore
  uses a deterministic one-step cyclic phase delay as the second direction for
  a cheap wedge-energy proxy. This is a diagnostic, not a new exact SVD formula.
- The next promising use is dispatcher triage: low-entropy structured stress,
  high-entropy dense mixing, and row/column phase gaps should eventually select
  different rotor schedules.

## 0.9.0

Adds the global quad-view audit and correct local `2x2` Clifford coordinates.

Included:

- New `lie_svd_quadenergy` module.
- Global row/column Clifford view:
  `A = sum_ij a_ij e_i tensor f_j`.
- Four global views: primal `A`, row-dual metric `A A^T`, column-dual metric
  `A^T A`, and dual mismatch / quad spread.
- Matrix energy split:
  `diag`, `offdiag`, `sym_offdiag`, `skew`, `upper`, and `lower`.
- Row/column metric off-diagonal energies for the dual Gram views.
- Triangular imbalance diagnostic for upper-vs-lower flow.
- Correct local `2x2` Clifford coordinates:
  `E=(p+w)/2`, `F=(p-w)/2`, `G=(q+r)/2`, `H=(q-r)/2`.
- Exact two-sided local rotor angles using all four coordinates.
- New CLI flag: `--quad-energy`.
- Unit tests proving the Frobenius energy split and verifying that the
  four-coordinate local angles annihilate generic `2x2` off-diagonal entries.

Observed smoke result:

- `N=16 jordan_defective`: almost pure upper-triangular flow:
  `upper ~7.77e1`, `lower ~0`.
- `N=16 sparse_structured`: mostly torsion/skew:
  `sym ~6.92e-1`, `skew ~4.82e0`.
- `N=16 degenerate_spectrum`: very large row/column metric energies:
  `row_metric ~1.21e4`, `col_metric ~1.21e4`.

Design notes:

- The correct decomposition is
  `A = diag(A) + sym_offdiag(A) + skew(A)`.
  Writing `diag(A) + (A + A^T)/2` double-counts the diagonal.
- The diagonal itself has two local Clifford components: scalar trace
  `E=(p+w)/2` and diagonal vector/gap `F=(p-w)/2`.
- Dropping `F` is the concrete reason the earlier three-component angle formula
  can increase off-diagonal energy.
- This module answers "what view are we missing?" by making the views
  measurable. It does not prove a universal `N log N` dense SVD route, but it
  can identify special low-complexity cases where a faster route may exist.
- The local `E,F,G,H` block coordinates are not the same as the global four
  row/column/dual views. They are the exact coordinates used inside one
  selected row/column rotor plane.

## 0.8.0

Adds the trace/Procrustes "inverse Rubik" navigator.

Included:

- New `lie_svd_traceflow` module.
- Trace objective helper:
  `trace_projection(core) = sum_i abs(core_ii)`.
- Local `2x2` trace-maximizing two-sided rotors starting from `U = I`,
  `V = I`, `core = A`.
- Monotone trace-ascent acceptance for each local move.
- Diagnostic trace: initial/final projection, initial/final off-diagonal core
  norm, rotations, rejected pairs, and plateau pairs.
- New CLI flags: `--trace-nav` and `--traceflow`.
- Unit tests for polished random reconstruction and the identity/degenerate
  plateau case.

Observed smoke result:

- `N=16 uniform_random`: trace projection improved from `~1.37e1` to
  `~5.46e1`; core offdiag fell from `~1.52e1` to `~1.34e-7`;
  `TraceFlow rel_recon ~1.82e-15`.
- `N=16 degenerate_spectrum`: trace projection improved from `~1.19e2` to
  `~3.04e2`; core offdiag fell from `~1.54e2` to `~3.99e-12`;
  `TraceFlow rel_recon ~1.91e-14`.
- `N=16 nearly_diagonal`: trace projection was already saturated while offdiag
  fell from `~1.09e-5` to `~4.38e-16`; `TraceFlow rel_recon ~5.49e-16`.

Design notes:

- This formalizes the "start from identity and rotate toward A" idea through
  the Procrustes/von-Neumann trace viewpoint.
- It does not replace the default solver: the route is clear and physically
  meaningful, but it still uses many local rotor moves.
- Repeated singular values still form a flat manifold of equivalent maximizers,
  so the trace view confirms why degenerate cases need deterministic schedules,
  polish, and sometimes repeller/lock logic.

## 0.7.0

Adds the tensor/Kronecker-chain view.

Included:

- New `lie_svd_tensortrain` module.
- `Kron2` split diagnostic for checking whether `A ~= B kron C` with a `2x2`
  outer factor.
- Recursive `2x2` Kronecker-chain detector with local and global residual
  thresholds.
- Exact `KronChain` SVD assembly for accepted chains: solve each `2x2` factor
  with `LieSvdMicro`, combine `U`, `Sigma`, and `Vt` by Kronecker products,
  then sort the product singular values.
- New synthetic `kron_structured` stress profile.
- New CLI flags: `--kron-trace` and `--kron-chain`.
- Unit tests for exact chain detection, exact chain SVD reconstruction, and
  rejection of a plain non-chain matrix.

Observed smoke result:

- `N=16 kron_structured`: first Kronecker residual `~8.60e-17`, four accepted
  chain levels, `KronChain rel_recon ~2.89e-16`, `orth_u ~6.50e-16`,
  `orth_v ~8.01e-16`.
- Ordinary `uniform_random` and `degenerate_spectrum` at `N=16` are rejected
  by the detector with first residuals around `8e-1`, as intended.

Design notes:

- This is a tensor-network/Schmidt-decomposition fast-path, not a closed-form
  SVD for arbitrary dense matrices.
- The module currently targets power-of-two dimensions and chains of `2x2`
  factors, matching the local rotor-cell interpretation.
- The most promising next step is a true Tensor Train rank profile for
  non-exact chains, where the solver could use low TT rank as a preconditioner
  rather than requiring an exact Kronecker chain.

## 0.6.0

Adds the adaptive synergy dispatcher.

Included:

- New `lie_svd_adaptive` module with route tracing.
- `LieSvd::solve` now delegates to the adaptive dispatcher.
- Cheap matrix triage: off-diagonal ratio, diagonal dominance, row/column
  spread, row/column mismatch, symmetry, transpose torsion, and entropy.
- Automatic `CoreFlow + TopoWarm + Repeller` route for balanced-degenerate and
  graph/topological cases.
- Conservative fast-path guards so random dense, extreme ill-conditioned,
  Jordan-like, sparse structured, and nearly diagonal profiles remain on
  `LieSvdSmall` in the `N=32` smoke run.
- `stress_cpu --auto-trace` prints adaptive route diagnostics.
- Regression tests for adaptive route decisions.

Observed smoke result:

- `N=32 degenerate_spectrum`: `Auto` now selects `CoreFlowTopo` and improves
  `rel_recon` from the `Small` row's `~8.57e-14` to `~4.66e-15`.
- `N=32 uniform_random` and `extreme_ill_conditioned`: `Auto` stays on
  `Small`, avoiding the earlier false-positive geometric route.

Design notes:

- This is the first version where the different "angles of view" are composed
  automatically rather than only by CLI experiments.
- Hand-written AVX2/NEON SIMD micro-rotors are still future work; this release
  focuses on dispatch correctness and solver synergy.

## 0.5.0

Refines topological warm-start into a more useful and cheaper preconditioner.

Included:

- Stationary row/column masses from the bipartite graph `|A|`.
- Fiedler-like low-frequency split axis from a few normalized row/column graph
  relaxations. This is an approximation, not an eigensolve.
- Fixed pseudo-random probes in the warm-start feature seed, replacing the
  previous sin/cos filler columns.
- Optimized orthogonal completion in `lie_svd_topowarm`: scratch buffers are
  reused and candidate basis residuals are scored without allocating a `Vec`
  for every candidate direction.
- New stress flags: `--topowarm-graph-steps` and `--topowarm-seed`.
- Regression test for centered/normalized stationary graph axes.

Observed smoke result:

- On local `N=32 degenerate_spectrum`, `CoreFlow + repeller` improved from
  `rel_recon ~ 1.39e-13` to `~2.20e-15` when `--topowarm` was enabled.

Design notes:

- The method still does not compute an exact Fiedler vector or exact
  Laplace-Beltrami center. It is a guarded `O(n^2 * k)` warm-start.
- It remains opt-in because random dense matrices may not repay the warm-start
  cost.

## 0.4.0

Adds the kernel/Gram route and makes repeller dynamics explicit and opt-in.

Included:

- `kernel_gram` module with `Linear` and `Rbf { gamma }` kernels.
- Symmetry detection for kernel matrices.
- Symmetric single-domain route: `K = U Sigma U^T`, with one rotor/eigen basis
  instead of unrelated left/right bases.
- Nonsymmetric square cross-kernel route through the existing two-sided
  `CoreFlow`.
- Public `repeller_potential` and `repeller_gradient` helpers implementing the
  Calogero-Moser-style singular-value anti-clustering potential.
- `stress_cpu --repel-lambda ... --repel-eps ...` flags. Defaults keep
  repellers disabled, so baseline behavior does not silently change.
- `lie_svd_topowarm` module: guarded landmark/sphere warm-start for `CoreFlow`
  using row/column landmarks, tiny two-sided power refinement, and orthogonal
  retraction.
- `stress_cpu --topowarm --topowarm-rank ... --topowarm-power-steps ...` flags.
- Tests for kernel symmetry, symmetric route reconstruction, bipartite routing,
  repeller finite-difference gradient, topological warm-start invariants, and
  an RBF clustered-data smoke check.

Design notes:

- For a single-domain Gram/RBF kernel, `U=V` is not an optimization shortcut; it
  is the mathematical structure of the problem.
- Repellers act on singular-value estimates during clustered/off-diagonal
  phases. They should be disabled during final precision polish.
- The topological warm-start is accepted only when it lowers the starting
  off-diagonal core energy. It is a preconditioner, not a non-iterative exact
  SVD.

## 0.3.0

Adds explicit repeller dynamics to the `CoreFlow` prototype.

Included:

- Soft anti-clustering repeller for coupled near-clusters in the diagonal/core
  estimates.
- Monotone backtracking line-search for `CoreFlow` rotor acceptance: proposed
  pair rotations must keep `offdiag(core)` non-increasing.
- Trace counters for rejected pairs and repeller-assisted accepted steps.
- Unit test confirming that the soft repeller activates on a coupled
  near-cluster.
- `CoreFlow` line-search optimized to clone only the trial core during
  backtracking, then update `U/V` once after accepting a step.

Design notes:

- This makes `CoreFlow` closer to a real energy-controlled flow, but it remains
  a prototype route and is not the default dispatcher path.
- The repeller is local and conservative. It biases basis orientation inside
  near-clusters; it does not change the true singular values of `A`.
- The line-search improves stability at the cost of additional memory traffic.

## 0.2.0

Adds the first explicit "small rotor + core-flow" prototype layer.

Included:

- `LieSvdMicro`: tiny fixed-schedule SVD microkernels for `N <= 4`.
- Dispatcher update: `LieSvd::solve` now routes `N <= 4` through `Micro`,
  then `Small`, then `Hybrid`.
- `LieSvdCoreFlow`: prototype state model around `core = U^T A V`, where `A`
  is fixed and the two orthogonal bases move.
- `stress_cpu --coreflow`: optional benchmark row for the new prototype route.
- Additional unit tests for microkernels and core-flow stability.

Design notes:

- `E_metric` remains local/proxy-based in the prototype; the release still
  avoids global `A^T A` / `A A^T` as a default control path.
- `CoreFlow` tracks the best off-diagonal core energy and rolls back to it if a
  later local sweep is worse.
- The microkernels keep a residual check and escalate to `LieSvdSmall` if a
  tiny fixed schedule does not finish cleanly.

## 0.1.0

Initial public research release of `lie_cliffalg_analog_svd`.

Included:

- `LieSvdSmall`: robust polar/Jacobi SVD path for dense square matrices.
- `LieSvdHybrid`: dual-tiled Lie/Clifford rotor preconditioner with digital
  polish.
- `LieSvdAnalog`: analog/photonic-chip-oriented rotor mesh simulator with
  conflict-free pair layers and optional phase quantization hooks.
- `stress_cpu`: Linux/CPU benchmark and stress-test binary.
- Docker smoke-test image and GitHub Actions CI.

Known scope:

- Square dense `f64` matrices only.
- Research-quality algorithms, not a drop-in LAPACK replacement.
- The analog module is a hardware schedule simulator plus digital polish, not
  a claim that analog hardware alone computes machine-precision SVD.




> “Храним обычные f64, но проектируем алгоритм как расписание локальных ортогональных роторов, пригодное для CPU сегодня и для аналоговой/фотонной ткани завтра.”

> “We store standard f64s but design the algorithm as a schedule of local orthogonal rotors—suitable for today’s CPUs and for the analog/photonic fabrics of tomorrow.”

> A CPU-simulated SVD rotor schedule designed as a bridge toward analog and photonic orthogonal meshes.
