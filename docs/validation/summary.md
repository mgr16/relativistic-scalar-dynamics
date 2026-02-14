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

## Reproducibility

Runtime outputs are written under:

```
output/<run_id>/
  config.json
  manifest.json
  fields/
  series/
  plots/
```

`manifest.json` records runtime metadata (timestamp, commit when available, Python/library versions, MPI size).
