# Fase 2 — Espectroscopía exterior de producción (l=2): escalera, dominio R=40 y el número QNM

**Fecha:** 2026-07-12. **Estado:** capítulo CERRADO — escalera R=20 completa,
dos validaciones de dominio R=40 con graduación apareada, los dos
sistemáticos de −Im identificados y medidos, números de producción con
presupuesto honesto.
**Datos:** [`spectroscopy.json`](spectroscopy.json),
[`figures/spectroscopy.png`](figures/spectroscopy.png), `data/wf_*.npz`
(waveforms citables, ~8–50 KB). Script:
`scripts/exterior_spectroscopy.py` (idempotente: waveforms existentes se
reusan; `--refit` re-analiza sin evolucionar; corridas en subprocesos).
**Contexto:** diseño del capítulo de cavidad
([`../../phase1/cavity/note.md`](../../phase1/cavity/note.md) §4) sin
cambios: l=2, r_ext=6, R=20, pulso r0=8/w=2/A=10⁻³, ventanas ancladas al
pico (abanico 26M, offsets 0–8M), Prony modes=4. Referencia Leaver l=2
n=0: Mω = 0.48364 − 0.09676i (overtone n=1: 0.46394 − 0.29560i).

## 1. Matriz y proveniencia

- **Escalera R=20:** lc ∈ {1.4, 1.0, 0.7}, lc_inner = lc/3.75. Los rungs
  1.4 y 1.0 **reusan los waveforms del capítulo de cavidad** (2026-07-09;
  config idéntica, camino de código verificado sin cambios desde entonces,
  y el fit compartido `fit_anchored_windows` reproduce su `summary.json`
  bit a bit). El rung 0.7 es nuevo (1 779 s).
- **Validación de dominio R=40 con graduación APAREADA:** la ley de malla
  es lc(r) = lc_inner + (lc_out − lc_inner)·r/R, así que con
  lc_out = lc_inner + (lc − lc_inner)·(R/20) el perfil lc(r) en r < 20 es
  idéntico al del rung base: el par aísla el efecto de dominio del de
  resolución, y cuesta ~×2 (no ×8 como R=40 a lc uniforme). Esponja ancha
  (width 10, r > 30): el pozo barrera↔esponja se alarga (round-trip ~74M
  vs ~24M) y el doblete de cavidad no llena a t_end = 70. Corridas:
  apareada a lc=1.0 (1 025 s) y a lc=0.7 (3 177 s).
- Todo con t_end = 70, extracción c₂₀ en r_ext = 6, muestreo cada 4 pasos.

## 2. Re Mω: converge como el diseño predijo

Protocolo de diseño (abanico temprano), err firmado vs Leaver:

| rung R=20 | Re Mω ± abanico | err |
|---|---|---|
| lc=1.4 | 0.5002 ± 0.0295 | +3.4 % |
| lc=1.0 | 0.4666 ± 0.0058 | −3.5 % |
| lc=0.7 | 0.4746 ± 0.0172 | **−1.9 %** |

- **p(1.0→0.7) ≈ 1.8** — la predicción del diseño (~2.º orden ⇒ 1.5–2 % a
  lc=0.7) se cumple. El rung 1.4 está fuera de régimen (signo opuesto,
  abanico ±6 %): Richardson de 3 puntos no aplicable (diferencias no
  monótonas) — el contenido citable es la escalera 1.0→0.7 contra la
  verdad externa.
- **Dominio acotado:** Δω_re(R40−R20, par apareado lc=1.0) = −0.0022,
  dentro del scatter del abanico (±0.0058) ⇒ sistemático de dominio
  ≲ 0.5 % en Re.
- Chequeo independiente: el pooled tardío R=40 (§3) da Re = 0.4877 ±
  0.0124 (+0.8 %) — consistente.

## 3. −Im Mω: dos sistemáticos descubiertos, una retracción y el número

**(a) El suelo de cavidad hace in-medible −Im en R=20 (RETRACCIÓN).** Con
pico/suelo ≈ 4 (1.4 e-folds útiles), las ventanas tardías del abanico caen
al suelo cuasi-estacionario y el −Im por ventana deriva monótonamente
(lc=1.0: 0.073 → 0.118 a lo largo del abanico; lc=0.7: 0.047 → 0.074). Las
medias de abanico R=20 (−28 %, +2 %, −38 % en la escalera) son ruido de esa
deriva: **el "−Im err 2.1 %" del capítulo de cavidad era suerte de la media
del abanico y queda retractado como punto-estimado** (su scatter ±0.019 ya
lo advertía; la afirmación de diseño "l=2 esquiva la cavidad" sigue en pie
para Re, que es lo que el capítulo midió de forma robusta).

**(b) El overtone n=1 no es separable y sesga las ventanas tempranas en
CUALQUIER dominio.** n=1 difiere del fundamental 4 % en ω y ×3 en
decaimiento; el Prony devuelve como "segundo modo" el conjugado del
dominante (no separa el par casi-degenerado en ω) y el modo dominante
absorbe la mezcla: en R=40 (sin suelo) el abanico temprano da −Im = 0.110 ±
0.003 (+14 %) a lc=1.0-eq y 0.112 ± 0.007 (+16 %) a 0.7-eq —
**independiente de la resolución (p ≈ −0.3): no es malla, es protocolo.**
La firma es que el −Im por ventana decae hacia Leaver al correr el offset
(e^(−off/3.4) del overtone).

**El número de producción** sale del barrido declarado (no de un abanico
elegido a posteriori): ventana 16M, offsets 0–22M (paso 2), pooled sobre
off ≥ 10 (overtone < e^(−10/3.4) ≈ 5 %), solo posible en R=40 (en R=20
esas ventanas están en el suelo). 14 ventanas × 2 rungs:

> **−Im Mω = 0.1016 ± 0.0055 (+5.0 % ± 5.6 % vs Leaver)** —
> **Re Mω = 0.4877 ± 0.0124 (+0.8 % ± 2.6 %)** en el mismo pooled.

La dispersión restante es de protocolo, no de discretización: la malla
fina preserva estructura débil tardía que la gruesa disipa (en off ≥ 10 el
rung 0.7-eq deriva Re → 0.50 mientras el 1.0-eq no; ambos anteriores al
eco de esponja, que llega a r_ext recién a t ≈ 60). Por eso se citan los
dos rungs juntos con su spread completo.

## 4. El experimento de dominio confirma el modo atrapado

- Suelo de cola R40/R20 = 0.02–0.04 (×25–50 menor); pico/suelo 4 →
  196/109 (4.7–5.3 e-folds útiles).
- El doblete l=2 de R=20 es estable al refinar: lc=0.7 da (0.3475 ±
  0.0015, 0.5495 ± 0.0045) — dentro de las bandas del canario
  (`tests/test_cavity_mode_slow.py`), coherencia rms2/rms0 = 0.02.
- En R=40 **no hay doblete a t=70**: el ajuste de cola es degenerado
  (w₁ ≈ w₂ ≈ 0.464 ± 0.021, sobre el remanente del propio ring;
  rms2/rms0 = 0.13) — el pozo alargado aún no llena, como exige la
  interpretación de modo atrapado (round-trip ~74M). La cavidad queda
  confirmada por **manipulación experimental del pozo**, no solo por
  ajuste.
- El par apareado a 0.7 da Δω_im = −0.052, "fuera" del scatter — como
  debe ser: ese Δ **es** el sesgo de suelo de R=20 medido directamente.

## 5. Números de producción (resumen para el paper)

| observable | valor | err vs Leaver | protocolo |
|---|---|---|---|
| Re Mω | 0.4746 ± 0.0172 | −1.9 % | diseño, R=20 lc=0.7 (escalera p≈1.8) |
| Re Mω | 0.4877 ± 0.0124 | +0.8 % | pooled tardío R=40 (2 rungs) |
| −Im Mω | 0.1016 ± 0.0055 | +5.0 % | pooled tardío R=40 (2 rungs) |
| dominio (Re) | Δ = −0.0022 | ≲ 0.5 % | par apareado R40↔R20 @ lc=1.0 |

Presupuesto: el fit (abanico/barrido de ventanas) domina; la malla es
subdominante en Re (converge p≈1.8) y no limitante en −Im; el dominio está
acotado por el par apareado. La validación exterior del solver 3D queda al
**~2 % en Re y ~5 % en −Im** con sistemáticos identificados — suficiente
para el rol del capítulo (validación cuantitativa del exterior; la física
de H2 vive en el interior).

## 6. Límites y qué sigue

- **Sub-1 % en Re** (si el paper lo exige): un rung lc=0.5 (~2.4 h) debería
  dar ~−1 % por p≈1.8; no lo exige H2.
- **−Im mejor que ~5 %:** haría falta más señal limpia — R≥60 con
  t_end ≳ 100M (eco de esponja más tardío ⇒ más ventanas off-grandes) o un
  estimador que imponga el par n=0+n=1 con priors de Leaver; opcional.
- El barrido tardío es EXCLUSIVO de dominios grandes: R=20 no puede
  (suelo). Si alguna corrida futura cita −Im desde R=20, está mal por
  construcción.
- m≠0 no aporta en a=0 (degenerado); l=1 queda cualitativo (diseño F1).
- El fit de 2 líneas de cola en R=40/t≤70 es degenerado por diseño (no hay
  cola de cavidad todavía): no citar sus w como líneas físicas.

## 7. Reproducción

```bash
# matriz completa (reusa waveforms existentes; ~80 min si falta todo)
python scripts/exterior_spectroscopy.py --r40-lcs 1.0 0.7
# re-análisis sin evolucionar (5 s)
python scripts/exterior_spectroscopy.py --refit --r40-lcs 1.0 0.7
# humo del pipeline (no toca docs/)
python scripts/exterior_spectroscopy.py --fast
```
