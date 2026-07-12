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
| C2 | Congelado de números (calibración o1 + tabla canónica) | **CERRADO 2026-07-12** (implementó Sol, revisó y cerró Fable — log §7) |
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

**Insumos (CORREGIDOS por el orquestador 2026-07-12 tras el inventario de
Sol — ver log §7; todos verificados en disco/git):**

- Refs densas de la verdad, TODAS versionadas (claves `r`, `snapshot_ts`,
  `snapshots_u`; 81 snapshots sobre [0, 15]M):
  - l=0: `docs/research/phase2/interior/data/ab_smoke_ref_{linear,mexhat}_A0.1_n1600.npz`
    (las mismas que producción reusa vía `SMOKE_DATA`);
  - l=1: `docs/research/phase2/production/data/prod_ref_linear_l1_A0.1_n1600.npz`;
  - l=2: `docs/research/phase2/production/data/prod_ref_linear_l2_A0.1_n2600.npz`;
  - mexhat l>0 NO tiene ref por diseño (reducción 1D no exacta con
    potencial no lineal) — regla de transferencia en el log §7.
- Series 3D de la matriz (COMPLETA en disco):
  `results/phase2_production/run_*/run_*/series/interior_profiles.npz`
  (11 dirs; `run_linear_l0_lc0.040_ml` = A/B de lumping RECHAZADO,
  excluido del análisis) + rung l=0 @ 0.040 de masa consistente =
  `results/phase2_interior_ab/run_{linear,mexhat}/run_*/series/interior_profiles.npz`
  (fallback de diseño del script, ~línea 580). Radios del banco: clave
  `radii`, K=32 log [0.1, 0.5], verificados idénticos en las 13 series —
  se leen de ahí, no se re-derivan.
- Espejo `results/phase2_production_fast/`: NO es insumo canónico (rungs
  lc 0.08/0.12 fuera de producción); sus `prod_ref_*_n800` sirven solo
  como check de resolución de c(t) en l=0.
- **PROHIBIDO invocar `scripts/interior_production.py` en cualquier
  modo** (incluso `--skip-runs`): su `out_dir` de modo completo es
  `docs/research/phase2/production/` y re-escribiría `production.json` y
  `figures/` congelados. C2 lee los npz directamente (numpy puro).
- Protocolo congelado: `production.json::protocol` (pulso A=0.1 r0=5
  w=1 ingoing_curved; primary o1 [0.1,0.5]; anchor o0 [0.1,0.5];
  truth (0.02,0.2) o1; mexhat λ=0.1 v=1 u∞=v).

**Pasos:**

1. Script nuevo `scripts/o1_profile_calibration.py` (idempotente,
   mismo estilo que los estudios existentes). Por cada ref densa X de la
   lista de insumos (mínimo citable: linear l0 y mexhat l0; extendé a
   linear l1 y l2, cuyas refs versionadas existen):
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
   d. Estabilidad de c_X: en l=0, n1600 versionada vs n800 del espejo
      `_fast`; en l>0, `TRUTH_SCAN` (y si hace falta segunda resolución,
      ref 1D nueva a archivo NUEVO — oráculo puro, minutos, regla §4.8);
      más verdad movida al `TRUTH_SCAN` de siempre ((0.015,0.15),
      (0.03,0.3)) ⇒ piso de la calibración.
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

### 2026-07-12 17:35 — Inventario inicial de insumos C2; bloqueo por caché ausente
- Hecho: se listó `results/phase2_production/` antes de tocar código. Hay
  11 directorios `run_*` y 11 archivos `series/interior_profiles.npz`,
  todos con `radii` de longitud K=32; no existe el subdirectorio `data/`.
- Hecho: los rungs l=0 @ 0.040 de masa consistente no están bajo
  `results/phase2_production/`; sí existen las dos series del humo en
  `results/phase2_interior_ab/run_{linear,mexhat}/...`, y
  `scripts/interior_production.py` contiene un fallback explícito a esas
  corridas. Bajo producción solo está el A/B rechazado
  `run_linear_l0_lc0.040_ml`.
- Hecho: las refs profundas versionadas linear/mexhat l=0 n=1600 existen.
  Sus claves reales son `r`, `snapshot_ts`, `snapshots_u` (no `ts` como
  dice §3a), con shapes `(1600,)`, `(81,)`, `(81,1600)`.
- Hecho: las refs lineales versionadas adicionales son l=1 n=1600 y l=2
  n=2600, con las mismas claves. El espejo
  `results/phase2_production_fast/data/` contiene n=800 para linear l=0,
  linear l=2 y mexhat l=0, pero no linear l=1; no se promovió ni se tomó
  como sustituto de la caché canónica ausente.
- Decisiones tomadas: (1) no escribir código ni regenerar/reanalizar datos,
  en cumplimiento del stop obligatorio ante insumos ausentes; (2) no
  asumir que el espejo `_fast` es intercambiable con
  `results/phase2_production/data/`; (3) mantener todos los artefactos
  congelados F0--F2 en solo lectura.
- Ambigüedades / preguntas para el orquestador: (1) ¿Marco autoriza la
  regeneración de la matriz/caché de producción, con presupuesto informado
  de ~88 min de cómputo? No fue lanzada. (2) El script vigente escribe las
  refs no-fast curadas bajo `docs/research/phase2/production/data/` y reusa
  las l=0 del humo; no produciría por sí solo la ruta n=800 que §3a declara
  canónica. ¿Debe aceptarse el espejo `_fast` existente o se requiere un
  procedimiento específico para reconstruir/promover esa caché? (3) ¿Se
  acepta para C2 el fallback versionado por el propio script a las corridas
  l=0 @ 0.040 del humo? (4) ¿La mención de clave `ts` en §3a es un typo y
  C2 debe leer la clave real `snapshot_ts`?
- Artefactos producidos/modificados:
  `docs/research/phase3/HANDOFF.md` (solo esta entrada de log).
- Suite: no ejecutada; no hay código C2 escrito.
- [REVIEW] Bloqueo previo a implementación: resolver preguntas 1--4 y dar
  aprobación explícita antes de cualquier corrida/regeneración.

### 2026-07-12 — Respuesta del orquestador (Fable): bloqueo RESUELTO, sin regeneración
Verifiqué en disco, código y git cada hallazgo del inventario. El stop fue
correcto y el diagnóstico fino del log también; la causa raíz era un
**error del contrato (§3a, mío)**: declaré canónica una ruta de refs que el
script nunca produce. La matriz 3D está COMPLETA en disco. §3a queda
corregido en el cuerpo (edición del orquestador, esta fecha).

**Respuestas 1–4:**
1. **Regeneración de 88 min: DENEGADA — innecesaria y peligrosa.** Las 12
   series de la matriz existen: 11 `run_*` bajo `results/phase2_production/`
   + las 2 del humo (`results/phase2_interior_ab/run_{linear,mexhat}`) que
   cubren el rung l=0 @ 0.040 por el fallback de diseño del propio script
   (~línea 580); `run_linear_l0_lc0.040_ml` es el A/B de lumping RECHAZADO
   (excluido del análisis). Verifiqué además el banco: `radii` K=32
   log [0.1, 0.5] **idéntico en las 13 series** (np.allclose). Peligro
   descubierto en la revisión: el `out_dir` del modo completo de
   `interior_production.py` es `docs/research/phase2/production/` (líneas
   546/696) ⇒ cualquier invocación, incluso `--skip-runs`, re-escribiría
   `production.json` y `figures/` CONGELADOS. **Queda PROHIBIDO invocar
   `interior_production.py` en cualquier modo durante C2**; tu script lee
   los npz directamente.
2. **Caché canónica de refs = la versionada en docs, no `results/`:**
   l=0 → `docs/research/phase2/interior/data/ab_smoke_ref_{pot}_A0.1_n1600.npz`
   (exactamente las que `oracle_reference()` reusa vía `SMOKE_DATA`, línea
   224); l=1 → `docs/research/phase2/production/data/prod_ref_linear_l1_A0.1_n1600.npz`;
   l=2 → `prod_ref_linear_l2_A0.1_n2600.npz` (ambas git-tracked; claves
   `r`/`snapshot_ts`/`snapshots_u`, 81 snaps en [0,15]M — cobertura sobrada
   de la fase fuerte). El espejo `_fast` NO se promueve: sus corridas
   (lc 0.08/0.12) no son rungs de producción; sus `prod_ref_*_n800` valen
   SOLO como check de resolución de c(t) en l=0. No existe ref mexhat l>0
   **por diseño** (la reducción 1D no es exacta con potencial no lineal) —
   ver decisión de transferencia abajo.
3. **Fallback del humo para l=0 @ 0.040: ACEPTADO.** Es el mecanismo con el
   que `production.json` se construyó; C2 debe usar esas mismas series
   (misma procedencia que el capítulo congelado).
4. **Sí, `ts` era typo mío: la clave real es `snapshot_ts`.** Corregido en
   §3a.

**Decisión nueva del orquestador (pares l>0, paso 2):** c_X(t) existe para
linear l=0/1/2 y mexhat l=0. Para corregir los pares l>0 (sin ref mexhat):
en el paso 1 reportá `c_linear_l0(t)` vs `c_mexhat_l0(t)`; si su diferencia
en fase fuerte es chica (≲2–3 %), aplicá **c_linear_l a AMBOS miembros** del
par en l=1,2, declarando la transferencia con esa diferencia l=0 como cota
de su sistemática. Si difieren más, los pares l>0 quedan SIN corregir y se
documenta el porqué. (Racional: el sesgo de ventana depende de la forma F–S
del perfil, y los perfiles del par son proporcionales a ~0.9; la diferencia
l=0 mide directamente la validez de esa transferencia.)

**Arranque autorizado:** con esto podés escribir
`scripts/o1_profile_calibration.py` (§3a paso 1, refs de la lista de
arriba; mínimo citable linear+mexhat l=0, extendé a linear l=1/l=2 que SÍ
están). Para estabilidad en resolución en l>0: TRUTH_SCAN; si querés una
segunda n, generá una ref 1D nueva a archivo NUEVO (oráculo puro, minutos,
regla §4.8 — no toca nada 3D). Sigue vigente: sin commits (Marco), corridas
>15 min no deberían hacer falta — si aparece una, se propone aquí primero.

### 2026-07-12 18:26 — Calibración ejecutada; alarma del discriminador y STOP
- Hecho: se volvió a listar `results/phase2_production/` antes de tocar
  código (11 `run_*` visibles) y se releyeron completos §3a corregido y §7.
  No se invocó `interior_production.py` ni se regeneró ningún dato.
- Hecho: se implementó `scripts/o1_profile_calibration.py`, postproceso
  numpy puro que (a) valida las 12 series canónicas + el A/B `_ml` excluido,
  (b) lee las cuatro refs versionadas y los dos checks n=800 l=0, (c) usa
  deep truth o1 `(0.02,0.2)`, (d) calcula c(t) continuo y en los K=32 radios
  leídos, solo en fase fuerte y sin extrapolar/rellenar sus huecos, (e)
  corrige ambos miembros antes de recomputar, y (f) emite JSON estricto y
  figuras. Durante el primer pase se corrigieron dos bugs internos de
  implementación (resolución de globs y comparación del delta contra
  soporte desigual); no afectaron artefactos congelados.
- Hecho: estabilidad l=0 del c K=32 n=1600↔n=800: máx 0.50 % linear y
  0.37 % mexhat. Piso TRUTH_SCAN: 1.83 % linear l=0, 2.43 % mexhat l=0,
  10.09 % linear l=1 y 6.58 % linear l=2. Efecto continuo↔K=32: máx
  0.24 % / 0.44 % en linear/mexhat l=0 (subdominante).
- Hecho: c_l0 K=32 tiene mediana 1.0202 (linear) / 1.0145 (mexhat). En los
  13 snapshots fuertes comunes, `|c_mexhat/c_linear-1|` tiene mediana
  0.97 %, p95 5.75 % y máximo 10.06 %. Con el criterio conservador de
  máximo ≤3 %, la transferencia a l>0 FALLA y los pares l=1,2 quedaron sin
  corregir, como ordena §7.
- Hecho: en el rung fino l=0 y sobre el mismo soporte válido de calibración,
  dev mediano baja 10.31 %→7.11 % (linear; 31.0 % explicado) y
  4.83 %→3.89 % (mexhat; 19.3 % explicado). El soporte corregible es un
  subconjunto del congelado porque c(t) no se rellena fuera de la fase
  fuerte del oráculo.
- Hecho: ALARMA del discriminador en l=0.028 K=32, comparando antes/después
  sobre los mismos 22 samples: ratio_median 1.07755→1.07031 (−0.72 puntos
  absolutos), peak_ratio 0.94125→1.05105 (**+10.98 puntos**) y l2_ratio
  0.94515→0.96389 (+1.87 puntos). La variante continua da el mismo
  diagnóstico (movimiento máximo 11.00 %). El script escribió el artefacto
  con `status=review-required`, `stop_required=true` y terminó rc=2. El
  veredicto “partial” (provisional, NO promovido) atribuye 31 % del déficit
  L2 sobre soporte común y deja 4.16 puntos absolutos respecto del oráculo.
- Decisiones tomadas: (1) definición de fase fuerte de c idéntica a
  `interior_window_calibration.py` (`t≥4`, umbral sobre deep truth); dev y
  discriminador conservan la máscara 3D congelada o0, `4≤t≤10`; (2) la
  interpolación temporal se limita a segmentos fuertes contiguos, sin la
  extrapolación constante de `np.interp`; (3) el gate l>0 usa el máximo
  relativo común, no la mediana, por ser la lectura conservadora de
  “difieren ≲2–3 %”; (4) la alarma se evalúa antes/después sobre soporte
  idéntico y el delta contra el resumen congelado se guarda aparte; (5) al
  superar 3 %, se detuvo C2 antes de `paper_numbers.py` y de toda promoción.
- Ambigüedades / preguntas para el orquestador: (1) ¿confirma el máximo
  relativo como estadístico del gate l>0? La mediana (0.97 %) pasaría pero
  máximo/p95 (10.06 %/5.75 %) no. (2) ¿El soporte corregido debe seguir
  siendo estrictamente la intersección de las fases fuertes, aun cuando
  reduce l=0 fino de 31 a 22 samples, o se aprueba otro tratamiento que no
  viole “c solo en fase fuerte / sin relleno”? (3) El peak_ratio cambia
  9–11 % en todos los rungs l=0 aun sobre soporte idéntico, mientras L2 se
  mueve 1.9–2.3 %: ¿se rechaza la corrección para el discriminador, se
  degrada peak_ratio, o se requiere otra revisión antes de continuar C2?
- Artefactos producidos/modificados:
  `scripts/o1_profile_calibration.py`,
  `docs/research/phase3/o1_calibration.json`,
  `docs/research/phase3/figures/o1_calibration_{c,dev}.png`, y esta entrada.
- Suite: sintaxis Python verificada; postproceso completo en ~30 s (la
  mayor parte fue caché inicial de fuentes), rc=2 intencional por alarma.
  Pytest no ejecutado porque el contrato exige STOP antes de seguir.
- [REVIEW] No usar/promover números corregidos ni iniciar `paper_numbers.py`
  hasta que Fable/Marco resuelvan preguntas 1–3. `related_work.md`,
  `plan.md` y todos los artefactos F0–F2 permanecen intactos. No hubo commit.

### 2026-07-12 — Revisión del orquestador (Fable): alarma RESUELTA con mecanismo; 3b autorizado
Revisé script, JSON y figuras; repliqué el discriminador del rung fino
importando las funciones de `o1_profile_calibration.py` sobre los npz de
producción: **reproduce exacto** (0.94125→1.05105). Las definiciones son
fieles a producción (`_fit_run` = multipole o1/o0 + máscara congelada;
`raw_discriminator` = definición de `production.py`). El trabajo se acepta.

**Mecanismo de la alarma (verificado, no especulado):** el pico del par
vive en tiempos DISTINTOS por miembro — pico lineal en t≈4.45 (entrada del
pulso, donde el sesgo de ventana o1 vale c≈1.11) y pico mexhat en t=5.88
(plena fase fuerte, c≈0.99). El `peak_ratio` congelado 0.9413 incorpora ese
diferencial de sesgo (~11 %); **la corrección lo EXPONE, no lo crea**. Por
eso peak se mueve +9.0/+9.6/+11.0 pts en los tres rungs mientras L2
(integral, mismo soporte ambos miembros) solo +1.9/+2.3 pts y ratio_median
−0.6/−0.8 pts. Complemento: el máx del gate (10.06 %) vive en t=4.87,
pegado al borde interno del hueco del cruce por cero (t≈5.1–5.6), donde
|a_truth| se hunde y el cociente c degenera; en el corazón de la fase
(t=5.6–6.9) las c's difieren ≤1.1 %.

**Respuestas 1–3:**
1. **Gate l>0: transferencia DENEGADA en forma definitiva** — pero la razón
   que manda no es el estadístico: el piso `TRUTH_SCAN` de las c_l>0
   propias (10.09 % l=1, 6.58 % l=2) es ≥ el efecto a corregir, y c_l>0
   llega a ±40–80 % (figura): una "corrección" así puede inyectar más error
   del que quita. NINGUNA corrección l>0 es citable — ni transferida ni
   propia. El discriminador l>0 citable = el congelado, tal cual. Para
   futuros usos del gate queda fijado: mediana ≤2–3 % Y p95 ≤5 % sobre
   soporte común, excluyendo los 2 puntos adyacentes a cada borde de
   segmento (ahí el cociente está mal condicionado); acá es moot.
   **Ajuste pedido (1 línea, sin recomputar):** el `reason` de los runs
   linear l1/l2 en el JSON debe decir "own-truth TRUTH_SCAN floor ≥ effect"
   — su c propia existe; lo que los mata no es el gate de transferencia.
2. **Soporte 31→22: intersección estricta CONFIRMADA** (sin relleno). Los
   números corregidos NUNCA reemplazan a los congelados: el canónico del
   discriminador sigue siendo `production.json` sobre su máscara completa;
   la calibración entra en `numbers.md` como **análisis de sistemática**
   con soporte declarado (n=22 disc / n=24–27 dev) y con el before
   recomputado en el mismo soporte al lado (el JSON ya lo guarda así —
   correcto). El hueco (cruce por cero de a_truth) y el techo t≈7.1 (fin
   de fase fuerte 1D < tope 3D t=10) se explican en la nota.
3. **peak_ratio: DEGRADADO a no-citable como número de H2** (compara
   argmax en tiempos distintos ⇒ hereda el diferencial del sesgo de
   ventana ~11 %; su versión corregida tampoco se promueve — la corrección
   puntual en la entrada del pulso hereda el gradiente fuerte de c). El
   discriminador citable = **l2_ratio (primario) + ratio_median/IQR
   (secundario)**. Nota: esto REFUERZA H2 — el l2 corregido se acerca al
   oráculo (1.0055) en los tres rungs: 0.964/0.978/1.010.

**Veredicto de 3a (cerrado):** "partial" de Sol CONFIRMADO — el sesgo de
ventana o1 explica ~19–31 % del dev l=0 en los rungs en régimen (fino:
10.31→7.11 lin / 4.83→3.89 hat) y ~31 % del déficit L2 en soporte común
(6.04→4.16 pts vs oráculo); rung 0.056 casi no mejora (0.7–8 % — fuera de
régimen, domina malla: valida la interpretación); floor de calibración
2.43 pts; estabilidad en n excelente (≤0.5 %) y muestreo K subdominante
(≤0.44 %). El residual 4.16 pts queda declarado como sistemática 3D real.

**Autorización 3b (`paper_numbers.py`) con reglas de promoción:**
- Discriminador por modo: números CONGELADOS. `l2_ratio` status=citable
  (primario), `ratio_median`+IQR citable (secundario), `peak_ratio`
  status=no-citable con caveat del mecanismo (puntero a esta entrada).
- Fila nueva de presupuesto por rung l=0: fracción del dev explicada por
  ventana, dev residual, l2 corregido (diagnóstico, soporte declarado),
  floor de calibración. Fuente: `o1_calibration.json::(runs|verdict)`.
- l>0: congelados sin corrección; caveat "piso de verdad 6.6–10.1 %".
- Tras el ajuste del `reason`, re-emitir el JSON con
  `status=reviewed-diagnostic`, `stop_required=false` y un campo
  `review: {by: "Fable", date: 2026-07-12, log: "§7"}`.
- Al cierre de 3b: pytest completo (`-m "not slow"`) y entrada §7 con el
  formato de siempre. Sigue vigente: sin commits (Marco), C3–C5 cerrados.

### 2026-07-12 19:26 — 3b ejecutado; tabla canónica emitida y suite verde
- Hecho: se aplicó la revisión de Fable a la calibración sin alterar sus
  números ni soporte: `o1_calibration.json` quedó
  `status=reviewed-diagnostic`, `stop_required=false`, con
  `review={by:Fable,date:2026-07-12,log:§7}`. Los flags de umbral y deltas
  peak/L2 se conservan como evidencia diagnóstica revisada. Las correcciones
  l>0 están desactivadas en forma definitiva; los runs lineales declaran
  `own-truth TRUTH_SCAN floor ≥ effect` y los mexhat, ausencia de oráculo
  exacto + piso lineal. Re-emisión idempotente verificada byte a byte
  (SHA-256 sin cambios al segundo pase).
- Hecho: se implementó `scripts/paper_numbers.py`. Usa procedencia
  `archivo::/JSON/Pointer` (RFC 6901, necesaria porque labels como
  `l0_lc0.028` contienen puntos), resuelve y revalida cada valor, cada
  incertidumbre y cada transformación (`%`, `abs(-Im)`, recíproco de piso),
  y escribe atómicamente `numbers.json` + `numbers.md`. `--check` confirma
  que ambos outputs están al día.
- Hecho: la tabla contiene 222 entradas: 34 `citable`, 156
  `citable-con-caveat`, 22 `no-citable` y 10
  `pendiente-de-promoción`. Promoción de discriminador aplicada como ordenó
  Fable: l2_ratio congelado primario, ratio_median/IQR congelado secundario,
  peak_ratio no-citable; l>0 congelado con caveat del piso 6.6–10.1 %.
  Los l2 corregidos l=0 viven solo como diagnóstico con soporte declarado,
  nunca sustituyen a `production.json`.
- Hecho: se agregaron 12 tests numpy/stdlib puros para C2: deep truth,
  máscara fuerte sin relleno, radios K=32 provistos, corrección previa de
  ambos miembros, soporte común/delta, decisión l>0 revisada, JSON Pointer,
  procedencia de valor+incertidumbre+transformación, note-only nulo,
  estados/caveats, idempotencia y cinco spot-checks reales. Los spot-checks
  (`disc_l2` fino, dev residual o1, QNM −Im, cavidad w1, ζ Cowling F0)
  reproducen exactamente las fuentes.
- Decisiones tomadas: (1) los note-only usan `value=null`,
  `source=note-only`, `status=pendiente-de-promoción` y puntero de nota
  exacto; no se copió ningún número de prosa; (2) incertidumbres compuestas
  guardan procedencia por componente, no solo la del valor central; (3) el
  `peak_ratio` corregido y congelado permanece en la tabla como
  `no-citable`, para conservar trazabilidad del mecanismo sin promoverlo;
  (4) `numbers.md` interpreta la calibración como análisis de sistemática
  parcial y mantiene explícita la jerarquía canónico vs diagnóstico.
- Ambigüedades / preguntas para el orquestador: (1) quedan pendientes de
  promoción los 10 ids note-only:
  `mass_lumping_f1_speedup`, `convergence_physical_p1_coarse`,
  `convergence_physical_p1_fine`, `convergence_physical_p2_fine`,
  `dissipation_epsilon_max`, `cavity_doublet_headline_w1`,
  `cavity_doublet_headline_w2`, `production_matrix_wall_pool`,
  `production_linear_run_wall_range`,
  `production_mexhat_run_wall_range`. Fable debe decidir promover con
  script o degradar a texto; no se resolvieron aquí. (2) El cierre formal de
  C2 y cualquier cambio de `plan.md` siguen reservados al orquestador.
- Artefactos producidos/modificados:
  `scripts/o1_profile_calibration.py`, `scripts/paper_numbers.py`,
  `tests/test_o1_profile_calibration.py`, `tests/test_paper_numbers.py`,
  `docs/research/phase3/o1_calibration.json`,
  `docs/research/phase3/numbers.{json,md}`,
  `docs/research/phase3/figures/o1_calibration_{c,dev}.png`, y esta entrada.
- Suite: 184 pasando / 7 slow deseleccionados en 95.44 s con
  `python -m pytest -m "not slow"`. La primera corrida dentro del sandbox
  tuvo un único fallo de launcher MPI (`bind Operation not permitted`);
  el test aislado y luego la suite completa pasaron fuera del sandbox.
  `git diff --check` limpio; `paper_numbers.py --check` verde.
- [REVIEW] Resolver los 10 `pendiente-de-promoción` y decidir cierre de C2.
  `related_work.md`, `plan.md`, `src/rsd/`, `paper/` y todos los artefactos
  F0–F2 permanecen intactos. No hubo commit ni push; C3–C5 no se iniciaron.

### 2026-07-12 — Cierre de C2 (orquestador, Fable): checklist §6 completo
Revisión de cierre sobre los entregables de 3b:

- **Árbol verificado:** congelados F0–F2, `plan.md`, `related_work.md` y
  `src/rsd/` intactos (git status limpio sobre esos paths); los únicos
  cambios son los artefactos C2 declarados + este log.
- **JSON de calibración re-emitido conforme:** `reviewed-diagnostic`,
  `stop_required=false`, `review` con resolución, `reason` l>0 corregido
  (own-truth floor; mexhat además sin oráculo exacto).
- **Procedencia spot-checkeada por el orquestador con resolutor PROPIO
  (independiente del de Sol): 7/7 MATCH** — l2_ratio congelado 0.040,
  Killing residual fino (transform fraction-to-percent), dev residual o1
  0.040 (+ componente de incertidumbre), fracción del déficit 31.05 %,
  sesgo de overtone rung 0.7-eq (15.82 %), −Im pooled 0.10161±0.00545
  (valor + incertidumbre por puntero), Leaver −Im (transform absolute).
- **Suite verificada por el orquestador:** 184 passed / 7 slow
  deseleccionados (96 s, fuera de sandbox por el launcher MPI — mismo
  comportamiento que reportó Sol).
- **Criterio de Sol validado con fuente:** `walls_s`/`total_wall_seconds`
  de `production.json` quedaron en 0.0/51.6 s (re-análisis con caché) —
  apuntar ahí habría citado un costo falso; note-only era lo correcto.
- **Resolución de los 10 `pendiente-de-promoción`: TODOS degradados a
  prosa** (edición del orquestador en `scripts/paper_numbers.py`,
  firmada aquí: nuevo status terminal `degradado-a-prosa` + campo
  `resolution` en `note_only()`; los números viven en sus notas y el
  manuscrito los cita como prosa; ninguno amerita crear fuentes JSON
  nuevas). Casos con matiz, registrado en cada `resolution`:
  `mass_lumping_f1_speedup` (el número de DECISIÓN Δa00 28 %/39 % ya es
  canónico vía `production.json::mass_lumping_ab`; el ×35 es contexto),
  `dissipation_epsilon_max` (el paper cita la regla + guard, no el
  ε_max de una malla), `cavity_doublet_headline_w1/w2` (camino de
  promoción documentado si C4 lo tabula: `fit_tail_lines` sobre las
  waveforms versionadas), wall-times (histórico no regenerable).
  Re-emisión + `--check` verdes; counts finales: 34 citable /
  156 citable-con-caveat / 22 no-citable / 10 degradado-a-prosa /
  0 pendiente. Los 12 tests C2 pasan tras el cambio.
- **Declaración de novedad (§5 de `related_work.md`): SIN CAMBIO** — el
  "at the 10–15 % level" sigue siendo el presupuesto congelado citable;
  la calibración es diagnóstico que lo explica parcialmente (~31 % del
  déficit L2) sin sustituirlo.

**C2 CERRADO.** `plan.md` §3.3 actualizado por el orquestador. Queda para
Marco: commit (sugerido: un solo commit de todo C2). Próximo capítulo:
C3 (pipeline de figuras) — NO arrancar sin contrato del orquestador.
