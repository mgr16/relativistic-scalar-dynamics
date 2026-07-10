# Fase 2 — Diagnóstico interior: estimador de a(t) y calibración de ventana

**Fecha:** 2026-07-09 (§1–2), 2026-07-10 (§4–5). **Estado:** capítulo
CERRADO — estimador (§1), calibración 1D de ventana (§2), estimador 3D
multipolar (§4) y humo A/B lineal-vs-mexhat (§5) hechos; la decisión de
producción 3D queda en §5 (revisa la de §2 para el caso 3D).
**Datos:** `window_calibration.json`, `ab_smoke.json`, `data/*.npz`,
`figures/window_calibration.png`, `figures/ab_smoke.png`; scripts
`scripts/interior_window_calibration.py`, `scripts/interior_ab_smoke.py`;
estimadores `rsd.analysis.interior` + `rsd.analysis.extraction`
(tests `tests/test_interior_fit.py`, `tests/test_extraction.py`).
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
- Estimador por SESGO (válido en 1D/grids densos): orden 2 sobre
  [0.1, 0.5] primario, orden 1 sobre [0.1, 0.3] de contraste.
  **REVISADO PARA 3D por el humo A/B (§5): en 3D el error de malla
  correlacionado × cond invierte la jerarquía — primario 3D = o1 sobre
  [0.1, 0.5], con o0 de ancla de fase; o2 y ventana angosta quedan
  inusables a resolución práctica.**
- El diagnóstico 3D usa K ≥ 16 radios de extracción (el humo corrió K=32;
  el costo de extracción es despreciable).

## 3. Límites honestos de esta calibración

- Es 1D (modo a modo). El 3D agrega error de malla, interpolación de
  extracción y mezcla de modos — la calibración fija SOLO el término de
  sesgo de ventana; el presupuesto total de error del diagnóstico 3D quedó
  medido en el humo A/B (§5): término de malla ≈ 10–15 % mediana a
  lc_inner=0.04 con el primario 3D (escalera de convergencia a producción).
- La "verdad" es a su vez un fit (orden 1 profundo); su piso medido
  (spread por ventana de verdad) acota lo que esta calibración puede
  resolver.
- Los errores OLS del estimador son indicativos (residuo correlacionado);
  la incertidumbre de programa del a(t) 3D vendrá del scan de
  ventanas/resoluciones, como en el resto del proyecto.

## 4. El estimador 3D multipolar (HECHO 2026-07-10)

`rsd.analysis.extraction.MultiRadiusExtractor`: banco de K radios de
extracción con cuadratura angular compartida, una sola pasada de
localización y una reducción MPI por llamada; **valida cobertura completa
en la construcción** (un punto de cuadratura fuera de malla — p.ej. el
radio interno rozando la excisión facetada — aborta con la lista de
radios afectados, en vez de sesgar c_lm en silencio). extract() →
(K, n_modes) con c_lm(r_k) = ∮ φ Y_lm dΩ.
`rsd.analysis.interior.fit_log_profile_multipole` aplica la OLS calibrada
en §2 a la pila (t, K, modos) → a_lm(t) por modo.

Cableado de corrida: `analysis.interior_profile` en la config
(r_lo/r_hi/n_radii/lmax/spacing/fit_order, validada en `rsd.config`);
el CLI guarda `series/interior_profiles.npz` (crudo) +
`series/interior_alm.csv` (fit primario). Radios log-uniformes por
defecto — la calibración §2 se hizo sobre el grid log del oráculo y el
sesgo de ventana depende de la distribución de muestras.

Convención: c_00 = √4π·u para campos esféricos (Y_00 = 1/√4π); los
cocientes A/B son independientes de la norma.

## 5. Humo A/B 3D: lineal vs mexhat con dato idéntico (2026-07-10)

**Protocolo** (`scripts/interior_ab_smoke.py`; resumen `ab_smoke.json`,
figura `figures/ab_smoke.png`): dos corridas 3D con la MISMA perturbación
(gaussiana A=0.1, r0=5, w=1, `ingoing_curved`) sobre su vacío respectivo —
lineal V=0 con φ∞=0 y sombrero mexicano λ=0.1, v=1 con φ∞=v (la config
H2) —, malla de la sonda C de F0 (R=15, lc=1.2, r_inner=0.1,
lc_inner=0.04), t_end=12M, banco K=32 log en [0.1, 0.5], lmax=2. BC
característica en ambas (es exactamente compatible con el vacío constante:
con φ=cte, Π=0 el término de borde es cero; la variante
`sommerfeld_spherical` NO lo es — su término φ/r advectaría el vacío).
Verdad: oráculo 1D con el mismo dato, muestreo denso (80 snapshots/15M;
la referencia de §2, cada ~1M, submuestrea el pico angosto del mexhat),
verdad profunda o1 [0.02, 0.2], comparada como √4π·a_1D. Fase fuerte
anclada en el fit o0 (cond ~6: el timing no debe definirlo un estimador
de varianza alta). Costo: 406 s (lineal) / 1272 s (mexhat) serial.

**Presupuesto de error medido** (dev = |a00_3D − √4π·a_1D| máx/mediana
sobre la fase fuerte, en unidades de la escala 1D; "sesgo 1D" = el mismo
estimador sobre el grid denso del oráculo — separa sesgo de ventana del
término de malla):

| estimador | dev 3D lin | dev 3D hat | sesgo 1D lin | σ_a med | cond |
|---|---|---|---|---|---|
| o0 [0.1,0.5] | 45 % / 19 % | 45 % / 17 % | 60 % / 26 % | 0.5 % | 6 |
| **o1 [0.1,0.5]** | **33 % / 15 %** | **25 % / 10 %** | 13 % / 2.3 % | 12–15 % | 573 |
| o1 [0.1,0.3] | 134 % / 106 % | 120 % / 82 % | 6.7 % / 0.7 % | 41–45 % | 1946 |
| o2 [0.1,0.5] | 884 % / 651 % | 728 % / 521 % | 7.7 % / 1.6 % | ~3× la señal | 5·10⁴ |

Lecturas:

1. **La jerarquía calibrada por sesgo se INVIERTE en 3D por varianza**: el
   error de malla correlacionado entre radios (~pocos % de u00) se
   amplifica con el cond de la base; el o2, óptimo en 1D, produce series
   sin sentido (su "pico" cae en tiempos tardíos de señal chica — puro
   junk amplificado), y la ventana angosta [0.1,0.3] (14 de los 32 radios)
   tampoco sobrevive. Esto materializa la advertencia del §2 punto 4.
2. **Decisión de producción 3D: primario o1 [0.1, 0.5]** (sesgo 1D ≤ 2.3 %
   mediana + término de malla ≈ 10–15 % mediana a lc_inner=0.04, con σ_a
   OLS honesta de esa magnitud) **con o0 [0.1, 0.5] de ancla de fase y
   consistencia** (dev mediana 17–19 % dominado por su sesgo de
   truncamiento, varianza despreciable). Mejorar el ~10–15 % es cuestión
   de resolución (escalera de convergencia en producción), no de ventana.
3. **El pipeline 3D reproduce la física del oráculo**: a00(t) sigue a
   √4π·a_1D en toda la fase activa para ambos potenciales (figura, panel
   a); el vacío del mexhat se proyecta exacto (u00(t=0) = √4π·v a 4
   decimales). Fuga de modos l>0: 0.7–1.6 % de la escala en fase fuerte
   (crece a ~3–6 % en tiempos tardíos de señal chica).
4. **Primer número 3D del discriminador de H2** (a00_hat/a00_lin, dato
   idéntico, fase fuerte común; el cociente puntual es mal condicionado en
   el cruce por cero de a_lin — se citan resúmenes robustos):
   cociente de picos 0.86 (3D) vs 1.15 (1D); **cociente L2 0.94 (3D) vs
   1.03 (1D)**. Ambos mundos dan O(1) dentro del presupuesto: con dato
   idéntico, la estructura de vacío Higgs NO altera cualitativamente el
   perfil a(t) — consistente con H2 (la dominación cinética manda). Es un
   humo (una resolución, un pulso), no la medición.

**Caveats de salud (honestos):**

- El residual del balance de Killing es 40–41 % en ambas corridas, y salta
  exactamente durante la absorción del pulso por el borde excisado
  (t≈3.5–7): a r_inner=0.1 con lc_inner=0.04 la esfera de excisión tiene
  facetas de ~23° y la cuadratura del flujo interior no captura la energía
  absorbida (el benchmark de F1 con residual 11 % era la malla sonda-B,
  geometría mucho más benigna). Es una limitación DEL DIAGNÓSTICO de
  balance en esta geometría, no del campo (validado directo contra el
  oráculo, punto 3); si producción quiere citar el balance como métrica de
  calidad a r_inner=0.1, necesita su propio estudio de resolución.
- El ζ_max de Cowling reportado por el monitor es GLOBAL y está dominado
  por el exterior de curvatura débil (T/curvatura → grande donde la
  curvatura → 0): 5.4–5.9 aquí. NO es comparable con el ζ ≤ 1.2·10⁻³
  interior de F0; el monitor interior-restringido queda como mejora
  pendiente si el paper lo cita.

**Límites del humo:** una resolución (el término de malla del punto 2 se
midió, no se extrapoló — la escalera va en producción); t_end=12M (no
cubre la era tardía); comparación 3D↔1D interpolada en tiempo (offsets de
timing inflan los máximos cerca del pico/cruce; las medianas son lo
robusto); disipación apagada en 3D (ε=0, la regla de producción de F1)
mientras el oráculo usa KO ε=0.02.
