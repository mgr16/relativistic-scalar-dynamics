# Fase 1 — Mitigación del modo de cavidad: interpretación y diseño para F2

**Fecha:** 2026-07-09. Datos: `summary*.json`, `waveform_l2*.npz`,
`l2_probe*.png`; script `scripts/cavity_l2_probe.py`; estimador de líneas
`rsd.analysis.ringdown.fit_tail_lines`. Contexto: el capítulo de
convergencia estableció que la espectroscopía l=1 a R=20 es fit-limited
por un modo de cavidad atrapado entre la barrera de potencial y la esponja
(r>15). Este capítulo prueba las mitigaciones, de barata a cara.

**Referencias Leaver:** l=1: Mω = 0.29294 − 0.09766i; l=2: Mω = 0.48364 −
0.09676i.

## 1. Corrección previa: las "líneas 0.209 + armónico 0.419" eran un artefacto

La nota de convergencia identificó la cavidad l=1 en "Mω = 0.209 con
armónico 0.419" por FFT de la cola t>40M. Con una cola de T≈30M la
resolución de la FFT es Δω = 2π/T ≈ 0.21 y esos "picos" son exactamente
k·2π/T — **cuantización de bins, no física**. El estimador honesto
(`fit_tail_lines`: regresión conjunta de 2 sinusoides, barrido 2D +
refinamiento re-centrado + incertidumbre por perfil de verosimilitud;
guardián `tests/test_tail_lines.py`) da:

- **Cola l=2: un doblete coherente limpio** — w₁ = 0.351 ± 0.003 y
  w₂ = 0.560 ± 0.007 (consistente en lc=1.4/1.0, en pulso w=2/w=1, y en
  r_ext=6/4 con corrimientos leves), que explica ≥96 % del rms de la cola.
  NO es armónico (w₂ ≠ 2w₁): son dos modos atrapados del pozo
  barrera(l)↔esponja.
- **Cola l=1: NI SIQUIERA es un doblete limpio** — mezcla una componente
  secular lenta (en lc=1.4 domina un término w→0.05 con amplitud 10⁻³) con
  líneas que vagan entre resoluciones (w₁ ∈ 0.15–0.25). La afirmación
  cualitativa del capítulo de convergencia (potencia cuasi-estacionaria
  resolución-independiente que no es cola de Price) sobrevive; el
  inventario de frecuencias específico queda retractado.
- Nota de método: dos líneas separadas Δω < 2π/T son identificables por
  regresión conjunta (no por FFT ni por ajuste greedy secuencial — ambos
  quedan sesgados; el test lo demuestra con sintéticos), pero cerca del
  límite la verosimilitud es una **trinchera diagonal** y la incertidumbre
  honesta es la del perfil (dw₁/dw₂ del estimador).

## 2. Experimentos de mitigación (todos sobre la config de la escalera)

| # | experimento | costo | resultado |
|---|---|---|---|
| a | ajuste conjunto QNM+cavidad, l=1 (9 parámetros) | 0 | **degenerado**: <1 ciclo útil no fija 9 parámetros; scatter ±0.03–0.06 entre ventanas, sin mejora |
| b | sustracción de cavidad ajustada en cola, l=1 | 0 | **inválida**: la cavidad aún se está llenando durante la ventana del ring (solo es estacionaria en t>40M, rms de doblete 8×10⁻⁶); extrapolarla hacia atrás mejora unos lc y empeora otros |
| c | **l=2** (lc=1.4, 1.0; w=2, r_ext=6) | 158 s / 540 s | **la palanca principal**: ver §3 |
| d | pulso angosto w=1 (l=2, lc=1.0) | 583 s | **nulo**: pico/suelo 4.0 (vs 4.1); reduce ring y cavidad por igual; líneas de cola idénticas (0.3545 vs 0.3535) — la razón ring/cavidad es invariante de la excitación |
| e | extracción cerca de la barrera r_ext=4 (l=2, lc=1.0) | 623 s | **trade-off**: pico/suelo sube a **6.4** (+56 %), y el scatter de −Im mejora ×4 (±0.005), pero aparece sesgo de zona cercana: Re 4.6 % (vs 3.5 %), Im 5.7 % (vs 2.1 %) |

## 3. Resultado principal: l=2 convierte el problema de fit-limited en resolución-limited

En la misma malla (lc=1.0), mismas ventanas ancladas al pico:

| modo | Re Mω (err) | scatter | −Im Mω (err) | scatter | pico/suelo |
|---|---|---|---|---|---|
| l=1 | 0.2676 (8.7 %) | ±0.052 | 0.0999 (2.2 %) | ±0.017 | ~4 |
| **l=2** | 0.4666 (3.5 %) | **±0.006** | 0.0988 (2.1 %) | ±0.019 | 4.1 |

- El scatter de ventanas en Re cae **×9**: con período 13M el ring l=2
  mete ~2 ciclos sobre el suelo donde l=1 metía <1 — el Prony por fin está
  condicionado. El error restante (3.5 %/2 %) es sistemático de
  discretización P1, el régimen donde la convergencia demostrada (~2.º
  orden en la ventana física) sí ayuda: extrapolando, lc=0.7 debería dar
  ~1.5–2 %.
- La contaminación relativa de cavidad es la MISMA (pico/suelo ≈ 4 en
  ambos l): l=2 no mitiga la cavidad, la **esquiva** dando más ciclos
  antes de tocarla.

## 4. Diseño para F2 (espectroscopía cuantitativa)

1. **Config de trabajo: l=2, r_ext=6, R=20, ventanas ancladas** — 3.5 %/2 %
   a lc=1.0 (540 s), mejorable con malla por convergencia. l=1 queda para
   física cualitativa (el interior H2 no depende de espectroscopía l=1).
2. **Si el paper exige sub-1 %:** una validación con **R≥40 + esponja
   ancha** (costo ~×8/corrida ≈ 70–80 min a lc=1.0): el pozo
   barrera↔esponja se alarga (round-trip ~74M vs ~24M), el doblete baja de
   frecuencia y llega tarde — la ventana del ring queda limpia. No hace
   falta migrar toda la producción: basta demostrar que el valor R=20/l=2
   no se mueve.
3. **No gastar en:** dar forma al pulso (d: nulo), post-proceso l=1
   (a, b: degenerado/inválido), ni r_ext<6 para MEDIR frecuencias (e:
   sesgo near-zone) — aunque r_ext=4 es legítimo para DETECTAR el ring
   (ratio 6.4).
4. **Canario:** `tests/test_cavity_mode_slow.py` fija el doblete l=2
   (0.32<w₁<0.38, 0.52<w₂<0.60, coherencia ≥85 %) — si cambia R/esponja/BC,
   falla y obliga a recalibrar ventanas y suelos.

## 5. Reproducción

`python scripts/cavity_l2_probe.py` (l=2 base), `--w 1.0 --tag _w1`
(pulso angosto), `--r-ext 4.0 --tag _rext4` (extracción cerca de la
barrera); `--refit` re-analiza npz guardados sin re-evolucionar. Los
prototipos de (a) y (b) fueron exploratorios (no versionados); su
resultado está documentado arriba y el estimador que sobrevivió es
`fit_tail_lines`.
