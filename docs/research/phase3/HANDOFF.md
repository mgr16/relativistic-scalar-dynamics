# F3 — HANDOFF de orquestación (C2)

**Fecha:** 2026-07-12 · **Orquestador/revisor:** Claude Fable 5 (créditos
reservados; revisa cuando vuelva el weekly de Marco, ~2026-07-15) ·
**Implementador de C2:** GPT-5.6 Sol · **Coordinación y commits:** Marco.

**Baseline de atribución:** el commit que introduce este archivo, con
árbol limpio. Todo cambio posterior en el árbol/historia es del
implementador (mismo patrón que el handoff de 2026-07-09).

Este documento es el contrato de trabajo de C2. El plan canónico del
programa sigue siendo [`../plan.md`](../plan.md) — leerlo PRIMERO
(§3.3, §4 "hechos técnicos", §6 "reglas de trabajo"); después
[`../phase2/production/note.md`](../phase2/production/note.md) y
[`../phase2/interior/note.md`](../phase2/interior/note.md). Nada de lo
de abajo tiene sentido sin ese contexto.

---

## 1. Entorno y convenciones del repo

- Python del entorno: `~/miniforge3/envs/rsd-dolfinx/bin/python`
  (conda env `rsd-dolfinx`, DOLFINx 0.10). En macOS exportar
  `CC=/usr/bin/clang` antes de correr nada que JIT-ee formas (el 3D).
  Para instalar paquetes: **mamba**, nunca conda clásico (se cuelga).
- Suite rápida: `python -m pytest -m "not slow"` desde la raíz —
  **172 en verde** al momento de este handoff. Debe seguir verde al
  cierre (más los tests que agregues).
- Los scripts de estudio corren desde la raíz, son **idempotentes**
  (`--force` re-corre, variantes `--fast` existen donde aplica), dejan
  crudo en `results/` (**gitignored** — local de esta máquina) y curado
  en `docs/research/` (versionado). Mantener ese patrón.
- El post-proceso de la calibración es numpy puro (no necesita DOLFINx),
  pero regenerar refs 1D usa `rsd.reference` (`SphericalOracle1D`) y
  re-correr 3D usa el stack completo.

## 2. Estado del pipeline F3 (C1–C5)

| Cap. | Qué | Estado |
|---|---|---|
| C1 | Related work + declaración de novedad | **CERRADO** (commit `b3e8df9`; [`related_work.md`](related_work.md)) |
| C2 | Congelado de números (calibración o1 + tabla canónica) | **ESTE HANDOFF — lo ejecuta Sol** |
| C3 | Pipeline de figuras (`scripts/paper_figures.py`) | NO ARRANCAR sin revisión del orquestador |
| C4 | Manuscrito revtex (`paper/`) | NO ARRANCAR |
| C5 | Empaquetado reproducible | NO ARRANCAR |

## 3. C2 — alcance exacto

Dos entregables, en este orden.

### 3a. Calibración o1 por perfil contra el oráculo denso

**Qué ataca.** El dev mediano de las series 3D vs el oráculo 1D en el
rung fino es 13 % (linear) / 8.6 % (mexhat) en l=0, estable bajo
refinamiento, y el L2 del discriminador arrastra un déficit sistemático
~10 % (`production.json::runs::*::dev_vs_1d_primary`). Ese dev se mide
contra la **verdad profunda** `deep_truth` = `fit_log_profile_series`
con ventana **(0.02, 0.2), orden 1** sobre las refs densas
(`scripts/interior_production.py::deep_truth`, `TRUTH_WINDOW` línea
~78). El estimador 3D primario en cambio ajusta **o1 en [0.1, 0.5]**
(`production.json::protocol::primary`). ⇒ El dev 3D INCLUYE el sesgo de
ventana o1-[0.1,0.5]-vs-verdad-profunda. **Hipótesis medible de C2:**
ese sesgo, computado por perfil (t a t) sobre el propio oráculo denso,
explica una parte cuantificable del déficit; corrigiéndolo, el dev y el
déficit L2 deben bajar. Lo que no baje queda como sistemática 3D real
(malla/extracción) y así se declara.

**Insumos (verificados hoy, salvo donde se indica):**

- Refs densas versionadas (l=0, n=1600, 80 snaps/15M):
  `docs/research/phase2/interior/data/ab_smoke_ref_linear_A0.1_n1600.npz`
  y `ab_smoke_ref_mexhat_A0.1_n1600.npz`. Claves esperadas: `r`, `ts`,
  `snapshots_u` (confirmá con `np.load(...).files` y anotalo en el log).
- Refs densas de producción (n=800, locales, gitignored):
  `results/phase2_production/data/prod_ref_{pot}_l{l}_A0.1_n800.npz` —
  **listá ese directorio al empezar** y anotá qué modos hay (esperado:
  linear l0/l1/l2, mexhat l0; el espejo `_fast` tiene menos).
- Series 3D por corrida:
  `results/phase2_production/run_*/series/interior_profiles.npz`
  (radios del banco K=32 incluidos — los radios se toman DE AHÍ, no se
  re-derivan). **Si el directorio no existe o está incompleto: NO
  re-corras la matriz por tu cuenta** — son ~88 min pool-3
  (`scripts/interior_production.py` es idempotente y `--skip-runs`
  re-analiza sin re-correr); proponelo en el log y que Marco dé el OK.
- Protocolo congelado: `production.json::protocol` (pulso A=0.1 r0=5
  w=1 ingoing_curved; primary o1 [0.1,0.5]; anchor o0 [0.1,0.5];
  truth (0.02,0.2) o1; mexhat λ=0.1 v=1 u∞=v).

**Pasos:**

1. Script nuevo `scripts/o1_profile_calibration.py` (idempotente,
   mismo estilo que los estudios existentes). Por cada ref densa X
   disponible (mínimo citable: linear l0 y mexhat l0; extendé a l1/l2
   lineal si las refs n800 están):
   a. `a_truth(t)` = `fit_log_profile_series(r, snaps, (0.02, 0.2),
      order=1)` — **la misma verdad que producción; NO inventes otra.**
   b. `a_win(t)` = fit o1 [0.1, 0.5] sobre el MISMO snapshot, en dos
      variantes: (i) ventana continua densa (todos los puntos r de la
      malla 1D en [0.1,0.5]) y (ii) muestreada SOLO en los K=32 radios
      del banco 3D (leídos de un `interior_profiles.npz`) — la
      diferencia (i)↔(ii) separa sesgo-de-ventana de
      sesgo-de-muestreo-K.
   c. Cociente por perfil `c_X(t) = a_win(t)/a_truth(t)` SOLO donde
      `|a_truth| ≥ 0.3·max|a_truth|` (fase fuerte, misma convención
      `STRONG_FRACTION` que `interior_window_calibration.py`); fuera de
      ahí el cociente está mal condicionado — se excluye, no se rellena.
   d. Estabilidad de c_X: (n1600 vs n800 donde existan ambas) y verdad
      movida al `TRUTH_SCAN` de siempre ((0.015,0.15), (0.03,0.3)) ⇒
      piso de la calibración.
2. Aplicación a las series 3D: interpolá `c_X(t)` a los tiempos 3D
   (misma convención de interpolación que usa
   `interior_production.py::analyze_run` al comparar contra truth) y
   corregí `a_3D_corr(t) = a_3D(t)/c_X(t)` para **ambos** miembros de
   cada par idéntico. Recomputá con las series corregidas: dev_vs_1d
   (contra `a_truth`, que ya es ventana-independiente) y el
   discriminador (ratio_median / peak_ratio / l2_ratio, mismas
   definiciones que `production.json::discriminator`).
3. La nota debe responder, con números: (i) ¿qué fracción del dev
   mediano fino (13 %/8.6 % l=0) explica c(t)? (ii) ¿se mueve el
   discriminador corregido? (esperado: poco — los sesgos del par
   correlacionan; si se mueve >2–3 % absoluto es flag de revisión, no
   lo publiques como corrección sin más); (iii) ¿c(t) es estable en
   resolución y bajo TRUTH_SCAN? (iv) veredicto: déficit explicado
   total / parcial / no, y el presupuesto de error actualizado del
   discriminador.
4. Entregables: `docs/research/phase3/o1_calibration.json` (curvas
   c_X(t) por variante, números antes/después, piso) +
   `docs/research/phase3/figures/` (c(t) por potencial; dev
   antes/después) + sección interpretativa en `numbers.md` (abajo).
   `production.json` y las notas de F2 quedan INTACTOS — los números
   corregidos viven en los artefactos nuevos de C2 con su procedencia.

### 3b. Tabla canónica de números citables

**Qué es.** LA fuente única de la que el manuscrito (C4) tomará cada
número. Script nuevo `scripts/paper_numbers.py` que **lee** los JSON
versionados y **emite** `docs/research/phase3/numbers.json` (máquina) +
`docs/research/phase3/numbers.md` (humano). Regla dura: **ningún número
re-tipeado a mano** — cada entrada lleva `source` = `archivo::ruta.de.clave`
y el script lo extrae de verdad. Si un número citable hoy solo existe
en una nota .md y no en un JSON (pasa con varios de F0/F1), la entrada
lo declara `source: "note-only", status: "pendiente-de-promoción"` — NO
lo copies al JSON a mano; listalo en el log para que el orquestador
decida (promover con script o degradar a texto).

**Formato de cada entrada (JSON) / fila (md):**
`id`, `value`, `uncertainty` (o presupuesto desglosado), `units`,
`source`, `status` ∈ {citable, citable-con-caveat, no-citable},
`caveat` (1 línea, obligatorio si status≠citable),
`paper_section` (destino previsto: intro/methods/interior/exterior/
discusión).

**Contenido mínimo (fuentes):**
- De `phase2/production/production.json`: discriminador por modo y rung
  (l2_ratio, peak_ratio, ratio_median con IQR), devs primario/ancla por
  corrida, σ_a, cond, junk off-channel, balance Killing (status:
  citable-con-caveat — ver plan §4), ζ Cowling global (ídem), A/B de
  mass lumping (rechazo), walls.
- De `phase2/exterior/spectroscopy.json`: referencia Leaver l=2, Re por
  escalera R=20 (−1.9 %, p≈1.8), pooled tardío R=40 (Re y
  −Im = 0.1016±0.0055), sesgo de overtone temprano (+14–16 %), checks
  de dominio (suelo ×25–50, pico/suelo), costos.
- De `phase3/o1_calibration.json` (3a): c(t) resumen, dev corregido,
  veredicto del déficit.
- De `phase1/cavity/summary.json` + canario: doblete 0.351/0.560.
- F0/F1 que el paper citará (ζ interior de F0, ×35 lumping, órdenes de
  convergencia F1, ε_max de disipación): mayormente note-only ⇒
  entradas `pendiente-de-promoción` con puntero exacto a nota y sección.

**Criterio de cierre de C2:** o1_calibration.json + numbers.{json,md} +
figuras + suite verde + log completo (abajo). **No marques C2 como
CERRADO en plan.md** — eso lo hace el orquestador tras revisar.

## 4. Límites duros (qué NO tocar)

1. **`related_work.md` — NADA**, en particular §5 (declaración de
   novedad). Solo lectura.
2. **C3–C5 no arrancan**: ni `paper/`, ni `scripts/paper_figures.py`,
   ni empaquetado, hasta revisión del orquestador.
3. **`plan.md` no se edita** (ni siquiera para marcar avances — el log
   de abajo es tu registro; el orquestador sincroniza el plan).
4. Artefactos congelados de F0–F2: `production.json`,
   `spectroscopy.json`, `ab_smoke.json`, `window_calibration.json`,
   notas y npz versionados — solo lectura. Error encontrado ⇒ se
   reporta en el log, no se corrige in situ.
5. `src/rsd/`: evitá tocarlo — C2 es post-proceso. Si algo es
   imprescindible (p. ej. exponer una utilidad de muestreo), cambio
   mínimo + tests nuevos + entrada `REVIEW` en el log.
6. **Ni commit ni push sin OK explícito de Marco** (plan §6.5). Dejá el
   árbol de trabajo con los archivos nuevos y el log al día.
7. Corridas caras: nada > ~15 min de pared sin proponerlo antes en el
   log y recibir OK de Marco (la matriz 3D completa son ~88 min).
8. No borres ni regeneres in-place refs/series existentes; toda ref
   nueva va a un archivo NUEVO con su resolución en el nombre.

## 5. Qué documentar y dónde

Todo avance se registra AL FINAL de este archivo, sección §7, formato:

```
### AAAA-MM-DD HH:MM — <hito corto>
- Hecho: ...
- Decisiones tomadas (cada una con justificación de 1 línea): ...
- Ambigüedades / preguntas para el orquestador (numeradas): ...
- Artefactos producidos/modificados: paths
- Suite: N pasando / detalle si algo rompió
- [REVIEW] items que exigen revisión de Fable antes de usarse
```

Distinguí siempre **decisión tomada** (seguiste adelante con criterio
declarado) de **ambigüedad abierta** (elegiste un default pero el
orquestador debe validarlo). Las segundas, numeradas, para que la
revisión las conteste una a una.

## 6. Checklist de revisión del orquestador (Fable, ~07-15)

- [ ] Log leído entero; ambigüedades numeradas respondidas.
- [ ] `deep_truth` reutilizada tal cual (ventana (0.02,0.2) o1) — no
      hay "verdad" nueva inventada.
- [ ] c(t) solo en fase fuerte; sin rellenos fuera de ella; piso de
      calibración (TRUTH_SCAN + n800↔n1600) reportado.
- [ ] Corrección aplicada a ambos miembros del par antes de recomputar
      el discriminador; movimiento del discriminador ≤2–3 % o
      flaggeado.
- [ ] Spot-check de procedencia: ≥5 entradas de `numbers.json` contra
      sus JSON fuente a mano.
- [ ] Entradas `pendiente-de-promoción` resueltas (promover/degradar).
- [ ] Suite verde local; diff de `src/rsd/` (si existe) revisado línea
      a línea.
- [ ] Decidir cierre de C2 ⇒ actualizar `plan.md` §3.3 + memoria; si el
      déficit quedó explicado, evaluar si el presupuesto de la
      declaración de novedad (§5 de related_work.md) se ajusta (solo el
      orquestador).

---

## 7. Log de ejecución

### 2026-07-12 — Handoff creado (orquestador)
- Hecho: C1 cerrado y commiteado (`b3e8df9`); este handoff define C2.
- Siguiente acción (Sol): §3a paso 1 — listar
  `results/phase2_production/{data,run_*}` y confirmar claves de los
  npz de refs; anotar hallazgos aquí antes de escribir código.
