# Fase 2 — Producción interior: escalera, modos l = 0, 1, 2 y el número de H2

**Fecha:** 2026-07-11. **Estado:** capítulo CERRADO — matriz completa (12
corridas, 0 fallos, ~88 min de pared con pool de 3), A/B de mass lumping
resuelto (RECHAZADO), presupuesto de error medido con escalera, y el
discriminador de H2 medido por modo con dato idéntico.
**Datos:** `production.json`, `figures/production.png`,
`data/prod_ref_linear_l{1,2}_*.npz` (refs 1D densas nuevas; las de l=0 se
reusan del humo); corridas crudas en `results/phase2_production/`
(gitignorado). Script: `scripts/interior_production.py` (idempotente:
re-ejecutar reusa lo corrido; `--skip-runs` re-analiza).
**Contexto:** protocolo y decisiones del humo A/B en
[`../interior/note.md`](../interior/note.md) §4–5; aquí no se cambia
ninguna decisión de estimador (primario o1 [0.1, 0.5] + ancla o0), solo se
ejecuta la medición a escala de producción.

## 1. Protocolo

Pulso idéntico en TODA la matriz (el de la config H2): gaussiana A = 0.1,
r0 = 5, w = 1, `ingoing_curved`, BC característica, ε = 0, masa
consistente (§2). Modos por dato inicial gaussiana × Y_l0 (implementado
2026-07-11; c_l0(r, 0) = A·g(r) = modo u_l del oráculo). Matriz:

- **Escalera l=0:** lc_inner ∈ {0.056, 0.04, 0.028} × {lineal, mexhat
  u∞=v}; el rung 0.04 reusa las corridas del humo (misma config exacta).
- **l=1 @ 0.04; l=2 @ {0.04, 0.028}**, banco lmax=4 en l=2 (canales de
  acoplamiento 2⊗2 → {0, 4} visibles).
- Banco K=32 radios log [0.1, 0.5]; muestreo ≈ 0.093M en todos los rungs.
- Verdad 1D: oráculo denso con el mismo dato por modo — lineal l=0/1/2 y
  mexhat l=0. **Mexhat l>0 no tiene oráculo** (la reducción 1D no es
  exacta con potencial no lineal): su validación es la escalera + el par.
- Normalización por modo: c_00 = √4π·u (dato esférico sin Y_00);
  c_l0 = u_l para l ≥ 1.
- **Fase fuerte = t ∈ [4, 10] M** ∧ |a_o0| ≥ 0.3·máx. El tope superior es
  nuevo respecto del humo: en los canales l>0 el junk tardío del dominio
  (R=15 sin esponja) crece hasta ~la escala de la señal después de
  t ≈ 10M, y sin tope contaminaba dev y discriminador (la regla de F1:
  colas no convergentes fuera de toda métrica). Con el tope, el "exceso de
  acoplamiento" ×3.3 que se leía en l=2 fino resultó ser junk tardío —
  retractado en §5.

## 2. Etapa 0 — mass lumping: RECHAZADO en producción

A/B sobre la config de producción exacta (lineal l=0 @ 0.04, 12M):
**Δa00/escala mediana 27.7 %, máx 38.6 %** vs masa consistente (criterio
de adopción: < 2 % / < 5 %). La aceleración era real (×3.1 de pared) pero
el desplazamiento del observable supera al propio término de malla. Es el
caveat de F1 materializado: el lumping desvía O(1) en estructuras
sub-resueltas, y el perfil logarítmico en [0.1, 0.5] con lc_inner=0.04
(facetas ~23° en la excisión) es exactamente eso — además el fit o1
amplifica el error correlacionado entre radios. **Producción interior =
masa consistente**; el lumping queda para observables exteriores
resueltos, con su propio A/B si se usa.

## 3. Presupuesto de error (corridas lineales vs oráculo denso)

dev = |a_l0^3D − a_1D|/escala, máx/mediana sobre fase fuerte, primario o1:

| corrida | 0.056 | 0.04 | 0.028 |
|---|---|---|---|
| lineal l=0 | 109 % / 65 % | 33 % / 15 % | 31 % / 13 % |
| mexhat l=0 | 79 % / 66 % | 25 % / 10 % | 24 % / 8.6 % |
| lineal l=1 | — | 67 % / 28 % | — |
| lineal l=2 | — | 63 % / 31 % | 65 % / 30 % |

Lecturas:

1. **El rung 0.056 está fuera de régimen** (dev ~65 % mediana; diferencia
   rms a 0.04 de 61 %): no participa de ningún número citable. El "orden
   ~5" formal de la escalera es artefacto de ese rung — lo citable es la
   diferencia 0.04→0.028: **rms 9.3 % (lineal) / 11.8 % (mexhat) de la
   escala en l=0**, y 19–21 % en l=2.
2. **El piso de dev en el rung fino (13 %/8.6 % mediana en l=0) ya no baja
   como h²**: la mejora 0.04→0.028 es ~15 %, no ×2. A esta resolución el
   dev vs oráculo está dominado por sistemáticas de comparación (offsets
   de interpolación temporal 3D↔1D cerca del pico/cruce por cero — el
   humo ya lo señalaba) y no es una medida limpia del error de malla; la
   σ_a OLS sí se contrae con la resolución (0.118 → 0.062 absoluta en
   l=0, 7.6 % de la escala en el fino).
3. **La inflación relativa de l>0 es de escala de señal, no de pipeline:**
   los errores ABSOLUTOS en norma u son casi idénticos entre modos
   (~0.030 l=0, 0.038 l=1, 0.042 l=2 en el rung correspondiente) — pero
   la respuesta interior cae con l por la barrera centrífuga (escala
   0.23 en u para l=0 vs 0.134/0.142 para l=1/2), así que el mismo error
   pesa el doble. Para llevar l>0 al nivel relativo de l=0 haría falta
   otra ×√2–2 de resolución interior (≈ 3–4 h/corrida mexhat) — no
   necesario para el veredicto de H2 (§4).

## 4. El número de H2: a_hat/a_lin por modo (dato idéntico)

Cocientes sobre fase fuerte común, primario o1 (los puntuales son mal
condicionados en el cruce por cero — se citan L2 y picos):

| par | L2 | picos | mediana [IQR] |
|---|---|---|---|
| l=0 @ 0.028 | **0.92** | 0.94 | 0.85 [0.57, 1.18] |
| l=0 @ 0.040 | 0.94 | 0.86 | 1.12 [0.66, 1.38] |
| l=0 @ 0.056 | 0.94 | 0.99 | 0.88 [0.69, 1.20] |
| l=1 @ 0.040 | **0.87** | 0.95 | 0.64 [0.33, 0.88] |
| l=2 @ 0.028 | **0.88** | 1.01 | 0.81 [0.49, 0.98] |
| l=2 @ 0.040 | 0.90 | 1.06 | 0.80 [0.39, 0.99] |
| oráculo 1D l=0 | 1.01 | 1.15 | 0.86 |

**Resultado de producción: a_hat/a_lin = O(1) en TODOS los modos medidos
(L2 0.87–0.94; picos 0.86–1.06), estable en la escalera al 2 % (l=0) /
3 % (l=2).** Con dato idéntico, la estructura de vacío del sombrero
mexicano NO altera cualitativamente el perfil de pendiente logarítmica
a_l(t) en ningún modo angular: la dominación cinética manda —
**consistente con H2, ahora a calidad de producción y por modo.**

Sistemática honesta: el L2 3D queda 6–13 % por debajo de 1 mientras el
oráculo l=0 da 1.01, y esa diferencia NO decrece al refinar (0.92/0.94/
0.94 en la escalera). Como el cociente es entre corridas con malla y
estimador idénticos (los errores se correlacionan y cancelan al primer
orden — por eso su dispersión de escalera es 2 % cuando el dev individual
es 10–15 %), el déficit residual es una sistemática del pipeline no
resuelta por esta escalera (candidatos: sesgo de ventana dependiente del
perfil — el o1 sobre [0.1, 0.5] no sesga igual al perfil hat que al lin —
y junk correlacionado distinto entre potenciales, el mexhat corre ×3 más
pasos de reloj no lineal). Cota honesta: **el "borrado" de H2 se confirma
al nivel de ~10–15 %**; distinguir un efecto físico de vacío por debajo de
ese nivel exigiría o4 de sesgo de ventana por perfil (calibrable con el
oráculo mexhat denso) o una ventana más profunda (r_inner < 0.1).

## 5. Acoplamiento de modos no lineal: COTA, no detección

Con la ventana honesta (t ≤ 10M), los canales ajenos del mexhat son
indistinguibles del junk lineal en la fase fuerte:

- l=2: canal →0: 1.9 % (mexhat) vs 2.0 % (lineal) @ 0.028; canal →4:
  3.4 % vs 3.4 %. Nada.
- l=1: canal →0: 3.5 % vs 1.6 % — único indicio (×2.2), débil.

**Retracción metodológica:** sin el tope temporal, el canal →0 del mexhat
l=2 fino mostraba 6.5 % vs 2.0 % (×3.3), que leído ingenuamente era "la
primera detección de acoplamiento 2⊗2→0". Era junk tardío (t > 10M) del
dominio sin esponja. La medición honesta da una **cota: el acoplamiento
de modos interior con A = 0.1 queda por debajo de ~4–5 % de la señal del
modo durante la fase fuerte** — consistente con la imagen F–S de
asintóticas lineales modo a modo bajo dominación cinética. (Detectarlo
exigiría A mayor — con cuidado de no salir del régimen de invariancia de
amplitud de F0 — o esponja + t_end más largo para limpiar la ventana.)

## 6. Salud y costos

- Killing residual 0.34–0.49 en toda la matriz: la limitación conocida de
  cuadratura del flujo interior a r_inner=0.1 (facetas ~23°) — NO es
  métrica de calidad del campo aquí (validación = oráculo + escalera).
- Cowling ζ_max global 5.4–11.8 (l=0) / 1.1–3.2 (l>0): monitor global
  dominado por el exterior débil, estatus sin cambios (no comparable al
  ζ interior de F0).
- cond(primario) = 573 estable en toda la matriz; σ_a mediana 0.02–0.16.
- Costos de pared (serial equivalente): lineal 8–17 min, mexhat 22–54 min
  por corrida según rung; matriz completa 88 min con pool de 3. El rung
  0.028 mexhat (54 min) marca el techo práctico de esta máquina sin
  checkpoint/restart.

## 7. Límites y qué sigue

- t_end = 12M: la era tardía cuasi-estacionaria (1–4 % del pico, t≈13–30M)
  sigue sin medirse — corrida larga ℓ=0 opcional (necesitaría esponja o
  R mayor para el junk tardío, y probablemente checkpoint/restart).
- El déficit sistemático ~10 % del cociente L2 (§4) queda abierto; el
  siguiente paso barato es calibrar el sesgo de ventana o1 sobre el
  oráculo mexhat denso ya cacheado y restarlo por perfil.
- Espectroscopía exterior de producción (l=2, r_ext=6, R=20, ventanas
  ancladas — diseño de F1 sin cambios) es el capítulo siguiente de F2;
  después, F3 (pipeline de paper).
