#!/usr/bin/env bash
# Fase 0: sondas de viabilidad 3D del interior profundo (r_inner -> 0).
# Mide: celdas, dt CFL, costo de pared por paso, estabilidad.
set -uo pipefail
cd "$(dirname "$0")/../.."

export CC=/usr/bin/clang   # macOS: FFCx JIT contra el linker de Xcode

for probe in A B C; do
    echo "=== PROBE ${probe} ==="
    /usr/bin/time conda run -n psyop-dolfinx psyop run \
        --config "scripts/phase0_probes/probe_${probe}.json" \
        --output "results/phase0_probes/probe_${probe}" \
        2>&1 | tail -40
    echo "=== PROBE ${probe} exit: $? ==="
done
