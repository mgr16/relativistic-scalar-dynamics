# Validation Summary

This repository includes lightweight and environment-safe validation layers:

- **Unit/logic tests** (always runnable): config loading/validation, QNM transforms, packaging/import layout.
- **Physics tests** guarded by environment markers:
  - `requires_dolfinx`
  - `mpi`
  - `slow`

## Key checks implemented

1. **3+1 convention consistency**
   - Solver/docs use \(\Pi=(\partial_t\phi-\beta^i\partial_i\phi)/\alpha\).

2. **Radiative BC via facet tags**
   - Solver requires mesh facet tags for Sommerfeld/characteristic BC activation.

3. **MPI diagnostics**
   - Energy/flux use MPI global reductions.
   - `tests/test_mpi_diagnostics.py` compares 1-rank vs 2-rank diagnostics (when MPI+DOLFINx are available).

4. **QNM robustness**
   - Detrend + windowed FFT path.
   - Prony mode extraction with multi-order stability filtering and quality score.

5. **Convergence scaffold**
   - `tests/test_convergence_slow.py` provides temporal-refinement sanity check (`@pytest.mark.slow`).

6. **Background-metric checks**
   - `tests/test_metrics.py` compares the symbolic Kerr-Schild trace `K`
     against the analytic Schwarzschild-KS value.
   - `validate_config` enforces excision (`mesh.r_inner > 0`) for black hole
     metrics and `|a| <= M` for Kerr.

7. **Sommerfeld A/B test**
   - `tests/test_sommerfeld_reflection.py` is assert-based: the absorbing BC
     must lose strictly more energy than the reflecting (natural) case and
     report positive outgoing flux.

8. **QNM amplitude/phase**
   - Prony modes carry least-squares amplitudes/phases, with dominant-mode
     ordering and conjugate-pair clustering by |f| (`tests/test_config_and_qnm.py`).

9. **Consistent initial momentum**
   - `initial_conditions.direction = ingoing|outgoing` sets Π = ±(∂_rφ + φ/r);
     `tests/test_physics.py` verifies an outgoing pulse empties the domain.

10. **Energy balance**
    - `boundary_flux()` reports the exact discrete drain of the weak radiative
      BC, so `E(t) + ∫F dt − E(0)` closes (residual converges ~h²); checked in
      `tests/test_physics.py` and written to `series/balance.csv` by the CLI.

11. **Sponge layer & 4th-order filter**
    - The sponge absorbs dispersive massive-field tails; the biharmonic filter
      (`ko_order=4`, normalized by power-iteration λmax) damps grid modes with
      the same stability condition as the 2nd-order one but barely touches
      smooth modes. Both validated in `tests/test_physics.py`.

12. **Multipole extraction**
    - Real-Y_lm projection on extraction spheres (Gauss-Legendre × uniform
      quadrature); validated against an analytic Y_10 field to <1% and
      orthonormality to 1e-10 (`tests/test_extraction.py`).

13. **Measured convergence order**
    - `tests/test_convergence_slow.py` measures the observed SSP-RK3 temporal
      order (calibrated p ≈ 3.1) instead of a weak monotonicity check.

14. **Leaver QNM benchmark** (slow)
    - `tests/test_qnm_leaver_slow.py` evolves an ingoing Y_10 pulse on
      Kerr-Schild Schwarzschild with excision and compares the dominant
      ringdown mode against Mω(l=1,n=0) = 0.292936 − 0.097660i.
    - Measured at CI resolution (lc=1.5 graded to 0.4, P1, 109k cells):
      Mω = 0.2505 − 0.0709i (errors 15%/27%, tolerances 30%/50%).
    - Honest scope: at this resolution the discretization error dominates
      over the αKΠ term (a K=0 control also lands within tolerance), so the
      benchmark validates the full pipeline and catches gross errors, but
      discriminating the extrinsic-curvature term requires finer meshes.

15. **Leaver reference solver** (fast)
    - `psyop.analysis.leaver` computes scalar Kerr QNMs by continued
      fractions for any (l, m, n, a): `tests/test_leaver.py` checks the
      published Schwarzschild values (≤2e-6), the l=2 first overtone, and
      Kerr values cross-validated against the `qnm` package (Stein 2019)
      to ~1e-15 at a ≤ 0.95 (hardcoded; live cross-check is `slow`).

16. **Frame-dragging splitting on Kerr** (slow)
    - `tests/test_kerr_splitting_slow.py` evolves a real Y_1|1| pulse on
      a = 0.9: the signal must contain BOTH the prograde and retrograde
      modes (Leaver: 0.4372 and 0.2434, ratio 1.797). Calibration
      2026-06-11 (CI mesh): modes at 0.6037/0.3137, ratio 1.924.
    - The splitting is the robust observable — discretization shifts both
      modes (+30-40% at this resolution, partly spheroidal-spherical
      mixing of the Y_lm extraction) but cannot split a degenerate pair;
      in Schwarzschild the test must fail by construction.

17. **Cowling validity monitor**
    - `psyop.analysis.cowling` logs ζ = 8πρ/√(Kretschmann) and E/M per
      output step (`series/cowling.csv`), warning above ζ = 1e-2;
      `tests/test_cowling.py` verifies the exact A² scaling and the
      one-shot warning.

18. **Price tails** (slow)
    - `psyop.analysis.tails` fits the late-time power law; the slow test
      evolves a massless l=1 pulse on a causally clean domain (R = 60,
      boundary echo arrives after the fit window) and checks the Price
      exponent t^-(2l+3) = t^-5.

## Quadrature note

UFL's automatic quadrature-degree estimation diverges for non-polynomial
metric coefficients (Kerr-Schild). The solver fixes
`solver.quadrature_degree` (default `2·degree + 2`) on all measures.

## Reproducibility

Runtime outputs are written under:

```
results/run_YYYYmmdd_HHMMSS/
  config.json
  manifest.json
  fields/
  series/
  plots/
```

`manifest.json` records runtime metadata (timestamp, commit when available, Python/library versions, MPI size).
