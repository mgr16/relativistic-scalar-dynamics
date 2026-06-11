# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] — 2026-06-11

### Added
- **Live visualization** (`psyop run --live`): interactive PyVista window showing
  the scalar field φ on a z=0 slice, updated during the evolution. Optional
  `--live-every N` controls the refresh cadence. Degrades gracefully (warning,
  no crash) when PyVista is missing, the environment is headless, or the run
  is parallel; zero overhead when disabled. New module `psyop.utils.live_view`,
  optional dependency extra `viz` (`pip install -e .[viz]`).
- **Sponge layer** (`solver.sponge {enabled, width, strength}`): damps the
  dispersive tails of massive fields that the characteristic BC cannot absorb.
- **4th-order Kreiss–Oliger dissipation** (`solver.ko_order = 4`): biharmonic
  filter normalized by λmax; same stability as 2nd order with far less impact
  on smooth modes.
- **Multipole extraction** (`analysis.extraction {enabled, radius, lmax}`):
  projects φ onto real spherical harmonics on an extraction sphere
  (`series/multipoles.csv`).
- **Graded meshes** (`mesh.lc_inner < mesh.lc`): radial refinement near the
  horizon/excision boundary.
- **Consistent initial momentum** (`initial_conditions.direction =
  "ingoing" | "outgoing" | "static"`): pure spherical pulse with
  Π = ±(∂ᵣφ + φ/r).
- **Energy balance diagnostics** (`series/balance.csv`): tracks
  E(t) + ∫F dt − E(0); the residual converges ~h².
- `CHANGELOG.md` and package `__version__`.
- `scripts/record_live_demo.py`: regenerates the README demo GIF
  (`docs/media/live_demo.gif`) with an off-screen run.

### Changed
- README rewritten in English and restructured for v3.0.
- `scripts/run_simulation.py` now calls the packaged CLI (`psyop.cli`)
  directly instead of the removed root `main.py`.

### Removed
- Root `main.py` launcher (use `psyop`, `python -m psyop`, or `psyop-run`).
- `docs/reviews/` historical review documents (superseded by this changelog;
  available in git history).

## [2.1.0] — 2026-02-14

### Added
- First-order formulation (φ, Π) with SSP-RK3 time integration and adaptive
  CFL timestep.
- Characteristic (Sommerfeld) absorbing boundary conditions with consistent
  metric weights.
- Black-hole support: horizon excision (`mesh.r_inner > 0`) with inner
  "do-nothing" boundary, Schwarzschild (isotropic) and Kerr (Kerr-Schild)
  backgrounds, symbolic extrinsic curvature K.
- QNM analysis via FFT and Prony; CLI `psyop run` / `psyop postprocess`.

### Changed
- Complete migration to DOLFINx (FEniCSx); `src/` packaging layout with
  console entry points.
