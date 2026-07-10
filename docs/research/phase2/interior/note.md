# Fase 2 — Diagnóstico interior: estimador de a(t) y calibración de ventana

**Fecha:** 2026-07-09. **Estado:** capítulo EN CURSO — §1 (estimador) y §2
(calibración 1D de ventana) listos; queda el estimador 3D (§4) y su humo A/B.
**Datos:** `window_calibration.json`, `data/*.npz`,
`figures/window_calibration.png`; script
`scripts/interior_window_calibration.py`; estimador
`rsd.analysis.interior` (tests `tests/test_interior_fit.py`).
**Contexto:** el pase de literatura ([`../literature.md`](../literature.md))
verificó que el observable de H2, a(t), es exactamente el coeficiente
A(t,ω) del teorema de Fournodavlos–Sbierski, con jerarquía completa
u = Σₙ [ζₙ rⁿ ln r + ηₙ rⁿ] (ζ₀ ≡ a, η₀ ≡ b). Este capítulo construye el
estimador que la producción 3D usará y calibra su sesgo de truncamiento
por ventana radial con el oráculo 1D, para decidir el r_inner de
producción.

## 1. El estimador (rsd.analysis.interior)

`fit_log_profile(r, u, r_window, order=N)`: regresión OLS de u contra la
base truncada {rⁿ·ln r, rⁿ : n ≤ N} (N ≤ 2) sobre la ventana radial, con
columnas normalizadas. Devuelve coeficientes (a ≡ ζ₀ el observable),
errores 1σ OLS —declarados **indicativos**: el residuo sobre perfiles FD
suaves es el resto correlacionado de la jerarquía, no ruido blanco—, rms
del residuo y número de condición (aviso de colinealidad en ventanas
angostas). `fit_log_profile_series` lo aplica a pilas de snapshots.

Hechos del estimador fijados por sintéticos (tests):

- Recuperación exacta a precisión de máquina cuando el modelo es completo
  (órdenes 0, 1 y 2).
- **Linealidad del sesgo:** el sesgo inducido por un término omitido g(r)
  es exactamente el coeficiente `a` de ajustar g a solas — la aritmética
  con que se interpreta toda la calibración.
- Término omitido η₁·r sobre ventanas auto-similares [w, 2w]: sesgo ∝ w
  (ratio 2.0 por halving). Término ζ₁·r·ln r: su pendiente logarítmica
  r·(ln r + 1) **se anula en r = 1/e ≈ 0.37, dentro de [0.25, 0.5]** — su
  contribución al sesgo está ahí suprimida por coincidencia de posición y
  no es monótona ventana a ventana; hacia adentro decae más lento que ∝ r
  (el factor |ln r| crece). Moraleja: no extrapolar el sesgo de una
  ventana a otra "a ojo"; se calibra por ventana (§2).
- El orden 1 elimina exactamente el sesgo de (ζ₁, η₁); el precio es
  varianza y colinealidad (cond ~10³–10⁴ en ventanas angostas).

## 2. Calibración 1D del sesgo por ventana

**Protocolo.** Verdad = fit orden 1 sobre la ventana profunda
[0.02, 0.2]M (inalcanzable en 3D), con su piso propio estimado moviendo
la ventana de verdad ([0.015,0.15] / [0.03,0.3]). Candidatas 3D:
[0.25,0.5], [0.15,0.5], [0.1,0.5], [0.1,0.3] × orden {0,1,2}. Fase
temporal: t ≥ 4M; niveles FUERTE (|a| ≥ 0.3·max — la fase activa
t ≈ 4–8M donde vive la medición de H2) y MEDIBLE (≥ 0.05·max). Datasets:
lineal l=0 y l=1 del piloto F0 (n=2600, r_min=0.01), l=2 lineal nuevo,
**sombrero mexicano con u∞ = v = 1, A=0.1** (la configuración H2; el
piloto de F0 no guardó sus snapshots), y réplica l=0 a n=1300 para la
estabilidad del sesgo en resolución.

**Estructura temporal de a(t)** (verdad profunda, lineal): fase activa
t ≈ 4–8M con el pico del observable; decaimiento rápido hasta t ~ 11M; y
una **era tardía cuasi-estacionaria al 1–4 % del pico** (t ≈ 13–30M,
suave y de variación lenta) — cualitativamente el régimen de tasas
tardías |A₀| ~ t^{−(q+1)} de F–S/AAG; medirla exigiría más señal y
tiempos más largos (queda anotado como opcional en el plan).

### Resultados

Sesgo relativo de a(t) vs verdad profunda, **máx/mediana sobre la fase
FUERTE** (tabla completa en `window_calibration.json`; piso de la verdad
por dataset: l0 0.96 %, l1 2.28 %, l2 7.07 %, mexhat 0.56 %):

| ventana/orden | lineal l=0 | lineal l=1 | lineal l=2 † | mexican hat u∞=v |
|---|---|---|---|---|
| [0.25,0.5] o0 | 107 % / 59 % | 116 % / 50 % | 212 % / 103 % | 5.9 % ‡ |
| [0.25,0.5] o2 | 13.4 % / 8.1 % | 10.9 % / 2.8 % | 22 % / 14 % | 4.7 % ‡ |
| [0.1,0.5] o1 | 17.8 % / 6.3 % | 21 % / 12 % | 44 % / 29 % | 1.7 % ‡ |
| **[0.1,0.5] o2** | **2.5 % / 1.3 %** | **2.6 % / 1.6 %** | 9.4 % / 6.4 % | **0.5 % ‡** |
| [0.1,0.3] o1 | 5.7 % / 1.2 % | 8.3 % / 7.8 % | 26 % / 17 % | 0.9 % ‡ |

† la verdad de l=2 (n=1600) tiene piso 7 % — su calibración está limitada
por la propia verdad; si el interior l=2 llegara a necesitar <7 %,
recalibrar con n ≥ 2600 (~17 min).
‡ el mexhat tiene 1 solo snapshot FUERTE (su a(t) pica angosto en t≈6M);
las medianas MEDIBLES lo respaldan (o2 [0.1,0.5]: 0.13 %).

Lecturas principales:

1. **La ventana [0.25, 0.5] es inutilizable para a(t) cuantitativo**, con
   cualquier orden: en la fase activa produce errores O(1) e incluso de
   SIGNO (figura: l=0 en t=5–6 da −1.7 donde la verdad es +1.0). El
   "r ~ 0.5" de F0 era una lectura visual de planitud; la jerarquía ζ/η
   contamina fuerte por encima de r ≈ 0.3. Esto REVISA el r_inner = 0.25M
   que el plan traía para las corridas interiores.
2. **[0.1, 0.5] con orden 2 alcanza ≤ 2.6 % (l=0,1) y 0.5 % (mexhat)**
   durante la fase fuerte — subdominante frente al error de malla 3D
   esperado (~2.º orden, unos %). Los máximos citados incluyen los
   snapshots t = 4–5M donde el pulso aún transita la ventana
   (conservador); anclar la ventana temporal a la llegada del pulso (como
   las ventanas de ring de F1) los reduce.
3. **Estabilidad en resolución** (l=0, n=1300 vs 2600): o2 estable
   (2.42 % vs 2.48 % en [0.1,0.5]; 0.38 % vs 0.29 % en [0.15,0.5]); el o1
   fluctúa más con el muestreo temporal de snapshots (9 % vs 18 % máx) —
   otra razón para preferir o2 + scan de ventanas.
4. Advertencia para el 3D: cond(o2) ~ 6·10⁴–3·10⁶. En 1D con miles de
   puntos y sin ruido, σ_a es despreciable (≤0.2 %); con K ~ 12–16 radios
   de extracción y error de malla correlacionado NO lo será — usar
   K ≥ 16 radios y validar la varianza en el humo A/B.
5. Preview física (no es el A/B aún): la config H2 (mexican hat con
   u∞ = v, A = 0.1) desarrolla el perfil logarítmico nítidamente — escala
   a = 0.263 con pulso A = 0.1, y es el dataset donde el estimador rinde
   MEJOR. La medición del "borrado" (a_Higgs vs a_lineal con dato
   idéntico) queda para el A/B.

### Decisión

- **r_inner de producción para corridas interiores: 0.1M** (las sondas F0
  fueron estables ahí; el plan §3.2 queda corregido — 0.25M rechazado por
  el punto 1). Las corridas de espectroscopía exterior no cambian.
- **Estimador de producción: orden 2 sobre [0.1, 0.5]** (primario) con
  **orden 1 sobre [0.1, 0.3]** de contraste; el sistemático de ventana se
  cita de esta calibración y del acuerdo entre ambas configuraciones.
- El diagnóstico 3D usa K ≥ 16 radios de extracción para soportar los 6
  parámetros del o2 (§4).

## 3. Límites honestos de esta calibración

- Es 1D (modo a modo). El 3D agrega error de malla, interpolación de
  extracción y mezcla de modos — la calibración fija SOLO el término de
  sesgo de ventana; el presupuesto total de error del diagnóstico 3D se
  medirá con el humo A/B (§4) y la escalera de convergencia si hace falta.
- La "verdad" es a su vez un fit (orden 1 profundo); su piso medido
  (spread por ventana de verdad) acota lo que esta calibración puede
  resolver.
- Los errores OLS del estimador son indicativos (residuo correlacionado);
  la incertidumbre de programa del a(t) 3D vendrá del scan de
  ventanas/resoluciones, como en el resto del proyecto.

## 4. Diseño del estimador 3D (siguiente paso del capítulo)

Reusar `MultipoleExtractor` (rsd.analysis.extraction) en K ~ 12–16 radios
de extracción dentro de [r_inner, 0.5M] → series u_lm(r_k, t) por modo →
`fit_log_profile_series` sobre los K radios da a_lm(t) con la misma
maquinaria calibrada aquí (la OLS es idéntica con K puntos). Humo A/B:
par de corridas 3D lineal vs sombrero-mexicano-en-vacío con dato
idéntico, r_inner según §2, resolución moderada — valida el pipeline y da
la primera medición 3D del discriminador de H2.
