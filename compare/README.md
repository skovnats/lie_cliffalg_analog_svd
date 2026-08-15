# lie_svd_reference_compare

An isolated LAPACK/MPFR ground-truth comparison harness for
[`lie_cliffalg_analog_svd`](..), the main crate this directory sits next to.

**This is a separate Cargo package with its own `[workspace]` marker and its
own `Dockerfile`, on purpose.** The main crate is deliberately
zero-heavy-dependency (no LAPACK/BLAS/faer, no arbitrary-precision library —
see its own `lie_svd_benchmarks` module doc comment). Comparing against a
real production LAPACK implementation and an arbitrary-precision (MPFR)
reference is a genuinely useful, but genuinely *different*, activity from
building and testing that crate — so it lives here, reachable only via a
`path = ".."` dependency, never the other way around. Building or running
`lie_cliffalg_analog_svd` itself never touches this directory or its
dependencies.

## What it does

1. **LAPACK comparison** (`ndarray-linalg`, OpenBLAS backend): runs both
   `lie_cliffalg_analog_svd::lie_svd_small::LieSvdSmall::solve` and LAPACK's
   `dgesdd` on the same benchmark matrices from the main crate's own
   `lie_svd_benchmarks` module (Kahan, Hilbert, Vandermonde, Pei), and
   reports orthogonality, reconstruction accuracy, singular-value agreement
   between the two, and wall-clock time for each.
2. **MPFR comparison** (`rug`, 200-bit arbitrary precision): computes the
   Hilbert matrix's determinant via plain `f64` LU and via 200-bit MPFR LU,
   quantifying how much of the `f64` answer is already representation/
   arithmetic error — independent of which SVD solver is used, for exactly
   the matrix this whole benchmark program has repeatedly flagged as
   sitting on the edge of `f64`'s precision.

## Measured results (x86_64, `--platform linux/amd64`)

This crate's own solver is not claimed to be faster than LAPACK, and isn't:
LAPACK is consistently faster (a highly optimized production library,
`~10-40x` in these runs) and, on well-conditioned matrices, both solvers
agree closely.

| Matrix | n | orth_u (this crate / LAPACK) | rel_recon (this crate / LAPACK) | max singular-value disagreement |
| --- | ---: | --- | --- | ---: |
| Kahan | 32 | 2.7e-14 / 6.7e-15 | 5.1e-15 / 2.4e-15 | 8.6e-12 |
| Pei (`alpha=0.01`) | 32 | 6.0e-15 / 3.4e-15 | 1.6e-15 / 4.1e-16 | 3.0e-13 |
| Pei (`alpha=0.01`) | 64 | 1.2e-14 / 5.4e-15 | 2.9e-15 / 4.7e-15 | 5.1e-13 |
| Vandermonde | 12 | 7.2e-15 / 1.8e-15 | 5.2e-14 / 3.5e-16 | **2.7e-1** |
| Hilbert | 32 | 2.9e-14 / 4.8e-15 | 1.2e-14 / 4.2e-16 | **1.6e1** |
| Hilbert | 64 | 7.7e-14 / 1.1e-14 | 1.5e-14 / 6.1e-16 | **2.5e0** |

The large disagreements on Hilbert and Vandermonde are not a bug in either
solver — they're the expected, honest consequence of what
`lie_svd_benchmarks`'s own module doc comment already establishes: past the
point where a matrix's condition number exceeds `f64`'s representable
range, the smallest singular values are numerical noise, not a meaningful
answer, and *two independently competent solvers have no reason to agree on
what that noise looks like*. Getting a `~16x` relative "disagreement" on
Hilbert `n=32`'s smallest singular value against production LAPACK is
exactly the kind of concrete, external confirmation that claim predicts,
not a surprise it overlooked.

The Pei matrix's exact closed-form eigenvalues (`hubbard_dimer`-style cross
-check, but against `pei_matrix_singular_values`) are also confirmed
independently here: `max relative error vs exact closed form = 1.0e-12` at
`n=64`, consistent with what the main crate's own test suite already
measures.

MPFR vs `f64` on the Hilbert determinant (a solver-independent
representation-error measurement):

| n | rel. difference from 200-bit MPFR |
| ---: | ---: |
| 6 | 2.4e-11 |
| 8 | 7.8e-9 |
| 10 | 1.3e-5 |
| 12 | **5.4e-2** |

By `n=12`, the plain `f64` determinant of the Hilbert matrix is already
off by `~5%` from the 200-bit reference — a clean, absolute, solver-
independent quantification of exactly how much of `f64`'s answer on this
matrix is representation error, not something attributable to any
particular SVD algorithm's choices.

## Building

Both a native build (for local iteration on a machine with the LAPACK/MPFR
system libraries installed) and the isolated Docker image are supported.

### Native (needs OpenBLAS, LAPACK, GMP, MPFR, MPC installed)

```bash
# macOS: brew install openblas gmp mpfr libmpc
# Debian/Ubuntu: apt install libopenblas-dev liblapack-dev libgmp-dev libmpfr-dev libmpc-dev
cd compare
cargo run --release
```

Note: on Apple Silicon (arm64), `lapack-sys` 0.14.0 has a known FFI bug
(`*const u8` vs `*const i8`, a char-signedness mismatch in its generated
bindings) that breaks the native build. The Docker image below builds for
`linux/amd64` specifically to sidestep this.

### Docker (isolated, does not touch the main crate's image or dependencies)

Build context must be the repository root, not this directory, since this
crate reaches the main one via `path = ".."`:

```bash
# from the repository root, not from compare/
docker build --platform linux/amd64 -f compare/Dockerfile -t lie_svd_reference_compare .
docker run --rm --platform linux/amd64 lie_svd_reference_compare
```
