# Fase 1 — Sesgo del filtro espectral sobre observables interiores: interpretación

**Fecha:** 2026-07-09 (barrido v1 divergente: 2026-07-07, preservado en
`v1_diverged/`). Datos: `summary.json`, `series.npz`, `dissipation.png`;
script `scripts/dissipation_study.py`. Guard de estabilidad: commit
`6d5ec58`; mass lumping (no usado aquí, masa consistente en todo el
barrido): commit `1c5ac2b`.

**Configuración:** Schwarzschild-KS (a=0), bola R=15M con excisión en
r_inner=0.25M, lc=1.2 / lc_inner=0.08, P1, CFL 0.3, potencial cero, pulso
gaussiano entrante A=0.01 (r₀=5M, w=1, `ingoing_curved`), t=20M, muestreo
interior en r*=0.4M (dentro del horizonte). Métrica de sesgo: desplazamiento
relativo vs la corrida sin filtro sobre la malla temporal común (L∞
normalizado al pico del baseline; L2 relativo global).

## 1. Qué pasó con el barrido original (v1, 2026-07-07)

La v1 corrió (off, o2 ε=0.02, o2 ε=0.05, o4 ε=0.05) y **divergió a ~10¹⁴⁸
en ambas corridas con ε=0.05**, sin ningún aviso. No fue un bug del
operador: es la cota de estabilidad documentada en
`docs/math/dissipation.md` — `ε·dt·λmax < 2`, idéntica para ambos órdenes
por diseño de la normalización — que **depende de la malla**
(λmax(M⁻¹K) ~ 1/h_min²) y nadie verificaba. En esta malla:

    λmax ≈ 5.382·10³,  dt = 1.853·10⁻²  ⇒  ε_max = 2/(dt·λmax) = 0.02005

- ε = 0.05 ⇒ ε·dt·λmax = 4.99 ≥ 2: el filtro amplifica el modo de malla
  más rápido en vez de disiparlo. Divergencia garantizada.
- ε = 0.02 ⇒ ε·dt·λmax = **1.995 = 99.7 % de la cota**: la v1 "funcionó"
  ahí por margen cero — el modo más rápido alternaba signo con
  amortiguación casi nula (factor −0.995). Una malla un 0.3 % más fina
  habría divergido igual.

Desde el commit `6d5ec58` el solver estima λmax para ambos órdenes y
**rechaza** la configuración inestable con `RuntimeError` que reporta el
ε_max concreto (la demo del barrido v2 lo verifica: ε=0.05 rechazado en
13 s en lugar de 354 s de basura exponencial).

## 2. Resultados v2 (todos los runs estables)

| corrida | ε·dt·λmax | φ interior L∞/L2 | E(t) L∞/L2 | E_Killing L∞/L2 | E_fin/E_off |
|---|---|---|---|---|---|
| o2 ε=0.005 | 0.499 | 3.7 % / 4.0 % | 8.1 % / 7.3 % | 7.8 % / 5.5 % | 0.055 |
| o2 ε=0.02 | 1.995 | 12.8 % / 11.2 % | 25.4 % / 18.8 % | 25.2 % / 17.6 % | 0.050 |
| **o4 ε=0.02** | 1.995 | **1.5 % / 2.2 %** | **2.0 % / 3.0 %** | **0.44 % / 0.70 %** | 0.19 |
| o2 ε=0.05 | 4.99 | — rechazado por el guard — | | | |

Costo: el filtro añade ~25–38 % de pared (332 s → 410–457 s).

## 3. Lectura

1. **El orden 2 mueve la física.** Su factor de daño por paso es ∝ ε·λ_k
   para *todo* modo: integrado sobre t=20M el pulso interior se desplaza
   4–25 % en el rango práctico de ε. La dependencia en ε es
   aproximadamente lineal (ε×4 ⇒ sesgo ×2.8–3.2; sublineal por saturación
   exponencial). Ningún ε de orden 2 útil es compatible con observables
   al nivel del 1 %.
2. **El orden 4 es transparente, y ahora está medido.** Al mismo
   ε=0.02 (y misma agresividad sobre el modo de malla, 99.7 % de la
   cota), el sesgo cae un orden de magnitud: φ interior ≤ 2.2 %, energía
   ≤ 3 %, y el **balance de Killing — la referencia bajo excisión — queda
   en 0.4–0.7 %**. La transparencia λ_k/λmax predicha en
   `docs/math/dissipation.md` es real.
3. **`E_fin/E_off` no es sesgo físico.** A t=20M la energía del baseline
   está en un suelo dominado por junk de malla; cualquier filtro lo
   colapsa (0.05–0.19). El sesgo honesto es el desplazamiento sobre la
   ventana completa (columnas L∞/L2), no ese cociente.
4. **Margen obligatorio.** ε=0.02 opera a 99.7 % de la cota de esta
   malla; el guard convierte cruzarla en error, pero operar pegado a ella
   deja el modo de malla sin amortiguar (zigzag −0.995). Trabajar con
   ε ≤ 0.5·ε_max.

## 4. Recomendación de producción (F2)

- **Corridas interiores de referencia: filtro APAGADO** (ε=0), como en
  Fase 0 y en la escalera de convergencia — el esquema es estable sin él
  y sin filtro no hay contaminación posible.
- Si el caso no lineal (Higgs) exige control de junk: **orden 4
  únicamente**, con ε ≤ 0.5·ε_max de la malla (aquí ≤ 0.01), verificando
  en el log el número de amortiguación que ahora emite el solver
  (`filtro espectral orden N: ε·dt·λmax = …`). Sesgo esperable a ese
  nivel: ≲ 1 % en φ/E y ≲ 0.4 % en E_Killing (escala ~lineal desde la
  fila o4 de la tabla).
- Frase para el paper: *el filtro biarmónico a ε=0.02 (99.7 % de su cota
  de estabilidad) desplaza el campo interior ≤ 2.2 % y el balance de
  Killing ≤ 0.7 %; las corridas de referencia usan ε=0.*

## 5. Reproducción

`python scripts/dissipation_study.py` (~28 min: demo del guard fail-fast
primero, luego 4 corridas t=20M; `--smoke` para validación en ~2 min).
Las series crudas de cada corrida van a un directorio temporal
(`--keep DIR` para conservarlas); las series relevantes quedan en
`series.npz`.
