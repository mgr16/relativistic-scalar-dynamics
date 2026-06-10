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
