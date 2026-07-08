# Fase 1 — Escalera de convergencia l=1 (Schwarzschild-KS): interpretación

**Fecha:** 2026-07-07. Datos: `ladder.json`, `waveform_lc*.npz`, `convergence.png`
(re-ajustados con el fitter corregido, commit `fe9d8c0`).

## Resultado principal

**El esquema converge al orden de diseño en la métrica robusta.** La
auto-convergencia de waveforms (‖W_i − W_{i+1}‖ sobre malla temporal común,
sin ajuste de modos de por medio) da:

| triplete | orden observado |
|---|---|
| lc = 2.0 → 1.4 → 1.0 | **2.10** |
| lc = 1.4 → 1.0 → 0.7 | 0.97 |

La caída a ~1 en el triplete fino es consistente con que el error de
**geometría facetada** de las esferas (excisión y borde, `geom_order=1`)
pase a dominar: exactamente el régimen para el que se implementó
`mesh.geom_order=2` (v3.2.0). Verificación en curso:
`convergence_p2/` (misma escalera con celdas curvas).

## Los errores de frecuencia NO son una medida de convergencia aquí

Los errores ω vs Leaver tras el refit (4–19 % en Re ω, dispersión entre
ventanas ±0.01–0.05) rebotan sin patrón monótono. Causa raíz, diagnosticada
sobre los waveforms guardados:

1. **Señal útil corta.** El ring l=1 (período 21M, τ=10.2M) emerge a
   |c₁₀|≈9×10⁻⁴ y toca un suelo a ~2×10⁻⁴ tras ~1.5 e-folds: quedan
   **<1 ciclo utilizable**. Ningún estimador espectral da sub-1 % con eso.
2. **El suelo es un modo de cavidad, no error de malla.** Es independiente
   de la resolución (2.1–2.5×10⁻⁴ en todas las mallas), no decae
   (pendiente log-log ≈ 0; la cola de Price l=1 daría t⁻⁵) y oscila a
   **Mω = 0.209 con armónico 0.419** (FFT de t>40, idéntico en lc=1.0 y
   0.7): una onda cuasi-estacionaria débilmente amortiguada entre la
   barrera de potencial (~3M) y la esponja/borde exterior (R=20, esponja
   en r>15).
3. La primera pasada además usaba ventanas de 42M (ajustaban suelo) y
   selección por max|f| (elegía espurios) — corregido en `fe9d8c0`; el
   "0.61 %" de lc=2.0 de esa pasada fue azar (el error sistemático P1 real
   a lc≈1.5 es ~15 %/27 %, documentado en `test_qnm_leaver_slow.py`).

## Recomendaciones (F1/F2)

- **Convergencia de esquema:** reportar auto-convergencia de waveforms
  (hecho); confirmar recuperación de orden ~2 con `geom_order=2` (en curso).
- **Espectroscopía cuantitativa:** subir la razón ring/cavidad — dominio
  mayor (R≥40) con esponja más ancha, extracción más cerca del pico de la
  barrera, pulso más angosto (excita más QNM), y/o **l=2** (período 13M ⇒
  ~2× más ciclos útiles). Para sub-1 %: ajuste conjunto ring+fondo
  (modelo QNM + modo de cavidad) en vez de Prony puro.
- El modo de cavidad Mω≈0.209 merece un test de regresión propio cuando se
  ataque (depende de R y la esponja, no de la malla).

## Datos

Refit (ventanas cortas ancladas + modo oscilatorio dominante):

| lc | Re Mω (err) | −Im Mω (err) | pared |
|---|---|---|---|
| 2.0 | 0.3051 (4.2 %) | 0.0860 (12.0 %) | 41 s |
| 1.4 | 0.2443 (16.6 %) | 0.0790 (19.1 %) | 142 s |
| 1.0 | 0.2676 (8.7 %) | 0.0999 (2.2 %) | 464 s |
| 0.7 | 0.2377 (18.9 %) | 0.0514 (47.4 %) | 2221 s |

Referencia Leaver: Mω = 0.29294 − 0.09766i. Celdas: 1.08M a lc=0.7.
