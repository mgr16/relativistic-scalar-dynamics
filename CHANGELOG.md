# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **Package renamed `psyop` → `rsd`** (Relativistic Scalar Dynamics). The
  import package (`src/rsd/`, `import rsd`), the distribution name, and the
  console entry points (`rsd`, `rsd-run`, `rsd-postprocess`) all move from
  the previous `psyop` identifier; the repository/folder is now
  `relativistic-scalar-dynamics`. No behavioural change — a pure identifier
  rename validated by the full fast suite (134 passed). Downstream imports
  and any `psyop …` CLI invocations must be updated to `rsd …`.
- **Dissipation honesty** (`docs/math/dissipation.md`): the 3D solver's
  artificial dissipation is an FEM Laplacian/biharmonic spectral filter
  (M⁻¹K, λmax-normalized), *not* finite-difference Kreiss–Oliger — the
  genuine FD-KO lives only in the 1D oracle. The knob is renamed to the
  canonical `solver.filter_strength` / `solver.filter_order` (aliases
  `ko_eps` / `ko_order` kept, canonical wins if both are set); the internal
  step is `_apply_spectral_filter`. New doc derives the operator, its
  spectral damping factors, the `ε·dt·λmax < 2` stability bound, and states
  the observable-contamination case (reference interior runs use ε = 0).
- **The spectral filter now enforces its stability bound.** λmax(M⁻¹K) is
  power-iterated for *both* orders (previously only order 4) and the first
  step checks `ε·dt·λmax < 2`: crossing it raises `RuntimeError` reporting
  the mesh's concrete ε_max instead of silently amplifying — the bound is
  mesh-dependent (ε_max ∝ h_min under CFL), and a fine-mesh sweep at
  ε = 0.05 had diverged to ~10¹⁴⁸ over t = 20M with no warning. The damping
  number `ε·dt·λmax` is logged when the filter is active.
  (`tests/test_filter_stability_guard.py`.)

### Added
- **Quasi-stationary line estimator** `fit_tail_lines`
  (`rsd.analysis.ringdown`): joint two-sinusoid frequency regression (2D
  scan + re-centering refinement) with profile-likelihood uncertainties.
  Born from a real failure: FFT "peaks" of a ~30M tail sit on the bin grid
  (Δω = 2π/T ≈ 0.21) and manufactured a fake harmonic ladder
  "0.209/0.419"; greedy sequential fitting is also biased for sub-Rayleigh
  line pairs (synthetic tests demonstrate both). Cavity-doublet regression
  canary: `tests/test_cavity_mode_slow.py` pins the R=20 domain artifact
  (l=2 doublet 0.351±0.003 / 0.560±0.007); if R/sponge/BCs change, it
  fails and spectroscopy calibrations must be revisited.
  (`tests/test_tail_lines.py`.)
- **Mass-lumping option** (`optimization.mass_lumping`, P1 only): replaces
  the consistent-mass CG solve of every RK stage with a pointwise scale by
  the row-sum diagonal M_L⁻¹ (strictly positive for P1; `degree > 1`
  raises). 251 → 7.2 ms/step (×35) at 262k cells — mass solves dominated
  the step after the fast path. For mesh-resolved fields the lumped state
  differs from the consistent one by the expected O(hᵖ) mass-discretization
  gap (2.8 % rel L2 at lc = 0.8, shrinking with h); under-resolved fronts
  differ O(1) in L∞ by spectral design. Default OFF — A/B once on a
  production config before adopting for extraction-quality runs.
  (`tests/test_mass_lumping.py`, `benchmarks/benchmark_mass_lumping.py`.)
- **Killing-energy diagnostic** (`docs/math/killing_energy.md`): the energy
  E_K = ∫√γ(αρ + Π β·∇φ) associated with the stationary Killing vector
  ξ = ∂_t obeys an exact surface-flux balance on Kerr–Schild backgrounds —
  no β/K volume terms. New `energy_killing()` / `killing_flux()` /
  `killing_inner_flux()` on the solver, `energy_killing()` /
  `killing_boundary_flux()` on the 1D oracle, and a `series/killing.csv`
  balance series in the CLI. Validated: 1D horizon-crossing pulse closes to
  3×10⁻⁴·E₀ (2nd-order convergent) with ∫F_inner = 1.000·E₀; on the Fase-0
  probe-B mesh the Killing residual is 11% vs 290% for the Eulerian balance
  (`tests/test_killing_energy.py`).

### Fixed
- `scripts/convergence_study.py`: the ringdown fit selected the mode with
  the largest |frequency| over windows far longer than the usable signal
  (the extracted waveform hits a non-decaying discretization floor ~×4
  below the ring peak), producing garbage fits at fine resolutions. Now
  uses short anchored windows and amplitude-dominant oscillatory mode
  selection (`fit_ringdown_modes`), plus a `--refit` flag to re-fit saved
  waveforms without re-evolving.

## [3.2.0] — 2026-07-07

Research infrastructure batch (Fase 0 of the interior-dynamics program):
a validated 1D reference oracle, a ×4.2 solver fast path, derived
excision-safety windows, curved-background initial data, and the Fase 0
feasibility report with its GO decision
([docs/research/phase0/report.md](docs/research/phase0/report.md)).

### Added
- **1D spherical reference oracle** (`rsd.reference.spherical1d`): exact
  l-mode spherical reduction on Schwarzschild–Kerr-Schild in (φ, Π) form,
  log/uniform radial grids, RK4 + KO dissipation. Cross-validates the 3D
  pipeline: reproduces Leaver QNMs (l=1 within 0.1%/1.9%, l=2 within
  1%/2.2% in Re/Im ω) with 19 fast tests.
- **Preassembled linear fast path** (`solvers.first_order`): the full linear
  RHS operator is assembled once and applied per step, with the nonlinear
  remainder handled exactly (potentials expose `linear_coefficient` /
  `cubic_coefficient`). ×4.2 (linear) / ×2.1 (Higgs) at 262k cells; exact
  A/B agreement with the generic path (`tests/test_operator_fastpath.py`,
  `benchmarks/benchmark_fastpath.py`).
- **Kerr excision window** (`physics.metrics.kerr_excision_window`, enforced
  by `validate_config`): admissible `mesh.r_inner` range (√(r₋² + a²), r₊)
  for Cartesian excision spheres in Kerr–Schild; the window closes at
  |a| > 0.9718 M. Derivation: `docs/math/excision_window.md`.
- **Curved-background initial data**
  (`initial_conditions.direction = "ingoing_curved"`): ingoing momentum
  consistent with the Kerr–Schild background; suppresses the spurious
  outgoing transient ×1.67 and delays its front ~3M (1D pilot).
- **Second-order meshes** (`mesh.geom_order = 2`): curved cells in
  `build_ball_mesh` so the geometric error does not dominate at high
  resolution.
- **Inner absorption flux**: `series/flux.csv` gains a `flux_inner` column
  (conormal + advective −(β·n)ρ terms) and the energy balance includes it.
  Currently a qualitative diagnostic — exact closure needs a Killing-energy
  diagnostic (report §5).
- **Math notes**: `docs/math/excision_window.md` and
  `docs/math/energy_stability.md` (semi-discrete energy stability of the
  FEM scheme).
- **Fase 0 research report** (`docs/research/phase0/report.md`): 1D pilot +
  3D deep-excision probes → **GO**. Support scripts:
  `scripts/pilot_phase0_oracle.py`, `scripts/phase0_probes/`,
  `scripts/convergence_study.py` (Fase 1 ladder).

### Changed
- `analysis.ringdown`: the default excision radius is now the midpoint of
  the admissible window for the requested spin (a=0.9 → r_inner ≈ 1.249)
  instead of a fixed 1.0 M, which at a=0.9 lay outside the admissible
  window (ingoing characteristics re-entered through the inner boundary).
  Slow Kerr spectroscopy tolerances will be recalibrated in Fase 1.

## [3.1.0] — 2026-06-11

Physics realism batch: reference-quality QNM targets, validity monitoring,
and astrophysical units.

### Added
- **Leaver reference QNM solver** (`rsd.analysis.leaver`): scalar (s=0)
  Kerr quasinormal frequencies for any (l, m, n, spin) via Leaver (1985)
  continued fractions with spin continuation. Reproduces published
  Schwarzschild values (Berti et al. 2009) to 1e-6 and agrees with the
  `qnm` package (Stein 2019) to machine precision up to a = 0.95.
- **Kerr ringdown pipeline** (`rsd.analysis.ringdown`): evolve Y_lm
  shell pulses on Kerr, Prony-fit the ringdown. With |m| > 0 a real field
  contains both prograde and retrograde modes — frame dragging splits them
  (~80% at a = 0.9), measurable in a single run with discretization
  systematics cancelling in the ratio.
- **Spin sweep script** (`scripts/qnm_kerr_sweep.py`): FEM vs Leaver
  comparison table across spins.
- **Cowling validity monitor** (`rsd.analysis.cowling`): dimensionless
  backreaction measure ζ = 8πρ/√(Kretschmann) and E_field/M logged each
  output step (`series/cowling.csv`); warns when the test-field
  approximation becomes marginal (ζ > 1e-2).
- **Astrophysical units** (`rsd.utils.units`): set
  `output.physical_units.M_solar` to get QNM frequencies in Hz and damping
  times in ms (`series/qnm_physical.json`) — e.g. the l=2 fundamental of a
  30 M_sun black hole lands at ~521 Hz, in the LIGO band.
- **Price tail analysis** (`rsd.analysis.tails`): log-log power-law fit
  with R² quality measure and theoretical exponents t^-(2l+3).

### Fixed
- `.gitignore` had an unanchored `rsd/` pattern that silently ignored
  new files under `src/rsd/` — `analysis/extraction.py` and
  `utils/live_view.py` were missing from every previous commit (broken
  imports on fresh clones). Pattern is now root-anchored (`/rsd/`).

## [3.0.0] — 2026-06-11

### Added
- **Live visualization** (`rsd run --live`): interactive PyVista window showing
  the scalar field φ on a z=0 slice, updated during the evolution. Optional
  `--live-every N` controls the refresh cadence. Degrades gracefully (warning,
  no crash) when PyVista is missing, the environment is headless, or the run
  is parallel; zero overhead when disabled. New module `rsd.utils.live_view`,
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
- `scripts/run_simulation.py` now calls the packaged CLI (`rsd.cli`)
  directly instead of the removed root `main.py`.

### Removed
- Root `main.py` launcher (use `rsd`, `python -m rsd`, or `rsd-run`).
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
- QNM analysis via FFT and Prony; CLI `rsd run` / `rsd postprocess`.

### Changed
- Complete migration to DOLFINx (FEniCSx); `src/` packaging layout with
  console entry points.
