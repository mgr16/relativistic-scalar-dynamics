# C3 figure data

This directory contains the two versioned 3D profile banks needed by
`scripts/paper_figures.py` for the interior-profile figure. They are exact
copies of the canonical F2 smoke outputs; no values were selected, resampled,
or recomputed during promotion.

| Versioned file | Local F2 source | SHA-256 |
|---|---|---|
| `ab_smoke_3d_linear_l0_lc0.040.npz` | `results/phase2_interior_ab/run_linear/run_20260710_170635/series/interior_profiles.npz` | `46ceca926ac7235e8a4f8ac2bda2aedb232af43d2ef9054285a1b6dd88b5c160` |
| `ab_smoke_3d_mexhat_l0_lc0.040.npz` | `results/phase2_interior_ab/run_mexhat/run_20260710_171321/series/interior_profiles.npz` | `e555cb31f994aa70308653f65fd886315fdc32ad03d4676b0dcad2363a3ab446` |

Both files have keys `t` (128), `radii` (32), `modes` (9, 2), and `u`
(128, 32, 9). The matching dense 1D references remain in
`docs/research/phase2/interior/data/ab_smoke_ref_{linear,mexhat}_A0.1_n1600.npz`.
The figure script validates these hashes before rendering and never reads the
gitignored source paths.
