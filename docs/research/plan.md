# Plan de investigación — Relativistic Scalar Dynamics (RSD)

**Estado a 2026-07-09.** Este archivo es la **copia canónica y versionada** del
plan del programa (antes vivía solo en la memoria local de Claude Code).
Cualquier persona, herramienta o modelo que continúe el trabajo debe **leer
este archivo primero** y **actualizar la sección correspondiente al cerrar cada
capítulo**. El plan original se estableció el 2026-06-12/13.

**Estado en una línea:** Fase 0 cerrada (GO). Fase 1: todo lo que bloquea la
física está CERRADO (convergencia, disipación, cavidad, mass lumping); quedan
extras de §3.1 que no bloquean H2. **Fase 2 ABIERTA**: literatura (F–S)
CERRADA; **diagnóstico interior CERRADO (2026-07-10)** — estimador 1D+3D,
calibración de ventana y humo A/B hechos (decisión 3D: r_inner = 0.1M,
banco K≥16 radios log en [0.1, 0.5], **primario o1 [0.1,0.5] + ancla o0**;
el o2 óptimo-en-sesgo es inusable en 3D por varianza). Siguiente:
corridas de producción — su prerrequisito, el dato l>0 en 3D
(gaussiana × Y_lm), quedó implementado el 2026-07-11.

---

## 1. Hipótesis y alcance

- **H2 (titular):** *la dominación cinética cerca de la singularidad de
  Schwarzschild borra la estructura de vacío tipo Higgs; se recuperan
  asintóticas logarítmicas lineales* (Fournodavlos–Sbierski; **enunciado
  exacto VERIFICADO 2026-07-09** — ψ = A(t,ω)·log r + B(t,ω) + O(r·log r),
  con genericidad; ver [`phase2/literature.md`](phase2/literature.md)).
- **H1 (respaldo):** decaimiento clásico de vacío catalizado por el agujero
  negro (línea Burda–Gregory–Moss).
- **Alcance declarado honestamente:** aproximación de Cowling (campo de prueba
  sobre fondo fijo), monitoreada cuantitativamente; es un límite explícito del
  estudio, no una omisión.
- **Observable de H2:** el perfil de pendiente logarítmica a(t) del campo
  interior. Para la pregunta Higgs el campo debe **empezar en el vacío**
  (u∞ = v).
- Meta: calidad de paper.

## 2. Infraestructura y entorno

- Campos escalares 3D por FEM sobre fondos de agujero negro: DOLFINx 0.10,
  métricas Kerr-Schild, excisión interior. Paquete `src/rsd/` (`import rsd`),
  CLI `rsd` / `rsd-run` / `rsd-postprocess`.
- Entorno: conda `rsd-dolfinx`; en macOS exportar `CC=/usr/bin/clang`;
  python: `~/miniforge3/envs/rsd-dolfinx/bin/python`. Para instalar paquetes
  usar **mamba** (el solver clásico de conda se cuelga >45 min).
- Oráculo 1D: `src/rsd/reference/spherical1d.py` — reducción esférica exacta
  por modos l sobre Schwarzschild-KS, validada contra Leaver (QNM l=1: 0.1 %
  Re ω / 1.9 % Im ω; l=2: 1 % / 2.2 %).
- Suite rápida de tests: 170 en verde (a 2026-07-11; corre con
  `python -m pytest -m "not slow"` dentro del entorno).
- Nota histórica: el paquete fue renombrado el 2026-07-07 (commit `f5f3f74`);
  los commits anteriores usan el nombre viejo del proyecto — es historia, no
  contenido vivo.

## 3. Fases

### 3.0 Fase 0 — Viabilidad (CERRADA 2026-07-07 — **GO**)

Informe completo: [`phase0/report.md`](phase0/report.md). Números clave:

- Dominación cinética interior: r*_int = 0.281M (cuadrático/Higgs) y 0.474M
  (sombrero mexicano); cociente cinético/potencial R_pot ~ r^2.78.
- Invariancia de amplitud: < 2·10⁻⁴ relativo en A ∈ [10⁻³, 10⁻¹].
- Observable a(t) plano a ~1–10 % (fase activa t ≈ 5–8M; |s/A| hasta 2.1).
- Cowling: ζ ≤ 1.2·10⁻³ dentro del horizonte a A = 0.1 (mejora hacia la
  singularidad). Supresión de junk ×1.67 con dato `ingoing_curved`.
- Sondas 3D A/B/C estables hasta r_inner = 0.1M; costo ~140–225 s cada una.
- Caveat: `r_star` en el JSON del piloto es el cruce EXTERIOR (artefacto de
  definición) — usar los valores interiores recalculados del informe.

### 3.1 Fase 1 — Infraestructura de producción (EN CURSO)

**Hecho y commiteado:**

- **Fast path** del solver (operador lineal preensamblado + resto cúbico
  exacto): ×4.2 (lineal) / ×2.1 (Higgs) a 262k celdas; tests A/B 5/5
  (`tests/test_operator_fastpath.py`). Tras esto dominan los solves de masa.
- **Ventana de excisión Kerr:** `docs/math/excision_window.md` +
  `kerr_excision_window()`; intervalo válido (√(r₋²+a²), r₊), se cierra en
  a > 0.9718M (excisión esferoidal pendiente). El default de r_inner ahora es
  el punto medio automático (a=0.9 → 1.249; el viejo 1.0 era inseguro) — los
  tests lentos de Kerr necesitan recalibración.
- **Energía de Killing** (`docs/math/killing_energy.md`, `series/killing.csv`):
  cierra exacto en slicings estacionarios; validada (1D cruce de horizonte
  ∫F_in = 1.000·E0; malla sonda-B: residual 11 % vs 290 % euleriano).
  **killing.csv es EL balance de referencia bajo excisión.** El flujo interior
  euleriano (`inner_flux()`) es solo cualitativo: ni la versión conormal ni la
  conormal+advección cierran el balance.
- **Capítulo de convergencia CERRADO** (interpretación:
  [`phase1/convergence/note.md`](phase1/convergence/note.md)): la métrica
  válida es la auto-convergencia de waveforms; en la ventana física (t < 30M)
  el esquema da ~2.º orden (1.87 triplete grueso / 1.3–1.4 fino, igual con
  celdas facetadas que curvas — hipótesis de geometría **refutada**, mismo
  coste de pared). La cola t > 40M está dominada por un **modo de cavidad
  Mω = 0.209** (+armónico 0.419), independiente de resolución, atrapado entre
  la barrera de potencial (~3M) y la esponja (R=20, r>15): **no converge** y
  debe excluirse de toda métrica. Los errores ω-vs-Leaver (4–19 %) son
  fit-limited (< 1 ciclo útil de ring en l=1), no de malla.
- **Disipación honesta (parte 1):** rename KO→filter + documentación del
  operador FEM real (`docs/math/dissipation.md`, commit `2a3ace1`).
- **Capítulo de disipación CERRADO** (2026-07-09, commits `6d5ec58` +
  `47c106c`; interpretación:
  [`phase1/dissipation/note.md`](phase1/dissipation/note.md)):
  - La divergencia ~10¹⁴⁸ del barrido original a ε=0.05 era la cota de
    estabilidad ε·dt·λmax < 2 (dependiente de malla: λmax ~ 1/h_min²)
    cruzada sin aviso; en la malla del estudio ε_max = 0.02005 y el viejo
    ε=0.02 operaba al 99.7 % de la cota. **El solver ahora estima λmax
    para ambos órdenes y rechaza configuraciones inestables con
    `RuntimeError` que reporta el ε_max de la malla** (y loguea
    ε·dt·λmax si pasa).
  - Sesgo medido vs ε=0 (interior, t=20M): orden 2 desplaza 4–25 % en el
    rango práctico de ε (∝ ε·λ_k, inusable para observables al 1 %);
    **orden 4 a ε=0.02: φ ≤ 2.2 %, E ≤ 3 %, balance de Killing
    0.4–0.7 %**.
  - Regla de producción: corridas de referencia con ε=0; si hace falta
    control de junk no lineal, **sólo orden 4** con ε ≤ 0.5·ε_max.
  - v1 divergente preservada como evidencia en
    `phase1/dissipation/v1_diverged/`.
- **Mass lumping HECHO** (2026-07-09, commit `1c5ac2b`):
  `optimization.mass_lumping` (row-sum, sólo P1; degree>1 lanza error)
  convierte el solve de masa de cada etapa RK en escala puntual:
  **251 → 7.2 ms/paso (×35) a 262k celdas**. Desviación vs masa
  consistente = brecha O(hᵖ) esperada en campos resueltos (2.8 % L2 a
  lc=0.8, cae con h); frentes sub-resueltos difieren O(1) en L∞ por
  diseño espectral — no es defecto del lumping. Default OFF; hacer un A/B
  en una config de producción antes de adoptarlo para corridas de
  extracción. Tests `tests/test_mass_lumping.py`, benchmark
  `benchmarks/benchmark_mass_lumping.py`.

- **Capítulo de cavidad CERRADO** (2026-07-09; interpretación:
  [`phase1/cavity/note.md`](phase1/cavity/note.md)): la espectroscopía de
  F2 queda DISEÑADA — **l=2, r_ext=6, R=20, ventanas ancladas** (3.5 %/2 %
  a lc=1.0, scatter ×9 menor que l=1; resolución-limitado, mejorable por
  convergencia). Post-proceso l=1 (ajuste conjunto, sustracción) probado y
  descartado; pulso angosto nulo; r_ext=4 sube pico/suelo a 6.4 pero
  sesga (near-zone). Sub-1 % si el paper lo exige: una validación
  R≥40 + esponja ancha (~×8/corrida). El doblete de cavidad l=2
  (0.351±0.003 / 0.560±0.007) tiene canario:
  `tests/test_cavity_mode_slow.py`. Nuevo estimador de líneas
  `rsd.analysis.ringdown.fit_tail_lines` (regresión conjunta + perfil de
  verosimilitud; la cifra histórica "0.209+armónico 0.419" era artefacto
  de bins de FFT y está retractada en la nota de convergencia).

**Resto de F1 (pendiente):** opción interpolate-coefficients; excisión
esferoidal (a > 0.9718M); checkpoint/restart; recalibrar tests lentos de
Kerr. Pregunta menor abierta: degradación residual de orden en el triplete
fino (¿interpolación de extracción / esponja / ruido de remallado no
anidado?).

### 3.2 Fase 2 — Producción física (EN CURSO)

- **Pase de literatura, capítulo 1 CERRADO (2026-07-09)** — enunciado exacto
  de Fournodavlos–Sbierski verificado contra el texto completo
  ([`phase2/literature.md`](phase2/literature.md)): expansión
  ψ = A(t,ω)·log r + B(t,ω) + O(r·log r) hacia {r=0} (Thm 1.7), con A =
  L²-lim ψ/log r ≡ **nuestro a(t)**; jerarquía completa Σ ζₙ rⁿ log r +
  Σ ηₙ rⁿ (Thm 1.18); genericidad: abierto en topología de energía sobre
  toda {r=0} (Thm 1.17) y datos suaves genéricos cerca de los extremos con
  |A₀| ~ |t|^{−3 ó −4} vía AAG (Cor 1.16). H2 sobrevive sin enmiendas; el
  contraste **lineal-vs-Higgs con dato idéntico** queda fijado como
  discriminador primario. Refs H1 (Burda–Gregory–Moss) ancladas + contrapunto
  2023 anotado. El related-work amplio queda para F3.
- **Diagnóstico interior CERRADO (2026-07-09/10;
  [`phase2/interior/note.md`](phase2/interior/note.md))** — estimador
  `rsd.analysis.interior.fit_log_profile` (base F–S truncada, órdenes 0–2)
  + **calibración 1D del sesgo de ventana**
  (`scripts/interior_window_calibration.py`) + **estimador 3D multipolar**
  (`MultiRadiusExtractor`: banco de K radios, cobertura validada al
  construir; config `analysis.interior_profile`; el CLI escribe
  `interior_profiles.npz` + `interior_alm.csv`) + **humo A/B 3D
  lineal-vs-mexhat con dato idéntico** (`scripts/interior_ab_smoke.py`,
  malla sonda-C, K=32, validado contra el oráculo 1D denso). Decisiones:
  - **[0.25, 0.5] RECHAZADA** para a(t) cuantitativo; corridas interiores
    con **r_inner = 0.1M**;
  - por sesgo (1D) el óptimo era o2 [0.1,0.5], pero el humo mostró que
    **en 3D la varianza (error de malla correlacionado × cond) invierte
    la jerarquía**: o2 y o1-angosto inusables; **primario 3D = o1
    [0.1, 0.5] + o0 de ancla de fase** (presupuesto medido: sesgo ≤2.3 %
    med + malla ≈10–15 % med a lc_inner=0.04; junk l>0 ~1–1.6 %);
  - primer número 3D del discriminador de H2 (dato idéntico): cociente L2
    a_hat/a_lin = 0.94 (3D) vs 1.03 (oráculo 1D) — O(1) en ambos mundos,
    consistente con H2; es humo, no medición;
  - caveats: balance de Killing NO citable a r_inner=0.1 sin estudio
    propio (residual 40 % por cuadratura del flujo en la esfera facetada);
    ζ de Cowling del monitor es global (dominado por el exterior débil),
    no comparable con el ζ interior de F0; verdad l=2 con piso 7 %
    (recalibrar a n≥2600 si el interior l=2 exige menos).
- **SIGUIENTE — corridas de producción:** campo iniciando en el vacío
  (u∞ = v), dato `ingoing_curved`, A ≤ 0.1, modos l = 1, 2, interiores con
  r_inner = 0.1M y banco interior activado (K≥16; el humo usó 32), con
  escalera de convergencia para bajar el término de malla del 10–15 %
  (+ posible corrida ℓ=0 larga para tasas tardías, opcional);
  espectroscopía exterior según diseño del capítulo de cavidad (l=2,
  r_ext=6, R=20, ventanas ancladas — sin cambios). **Dato l>0 en 3D
  implementado (2026-07-11):** gaussiana × Y_lm real ortonormal en
  `GaussianBump` (config `initial_conditions.l`/`.m`); usa la misma
  `real_ylm` del extractor ⇒ c_lm(r, 0) = A·g(r) en el canal (l, m),
  igual al modo del oráculo 1D con dato idéntico; el momento factoriza
  con el mismo ansatz radial del oráculo para todo l; l=0 conserva la
  normalización esférica histórica (sin factor Y_00). Tests:
  `tests/test_initial_data_ylm.py` (convención dato↔extracción pineada
  exacta; fuga de canal ≤ 2.5 % en φ sobre la malla de test,
  convergente con h). Caveat: con potencial no lineal la reducción 1D
  solo es exacta para l=0 — el mexhat con l>0 acopla multipolos y es
  física exclusiva del 3D (el oráculo lo advierte en runtime).

### 3.3 Fase 3 — Pipeline de paper (PENDIENTE)

## 4. Hechos técnicos establecidos (no re-derivar)

- Kerr-Schild: la velocidad característica entrante es exactamente 1; el ring
  llega con retardo tipo tortuga — anclar ventanas de ajuste al pico
  detectado. El QNM l=0 está mal condicionado (Q≈0.5): validar con l=1,2.
- El perfil interior ES logarítmico sobre 2 décadas durante la fase activa
  (pendiente log independiente de r); la amplitud a r fijo decae conforme la
  cola de Price alimenta el horizonte.
- Ventana radial del observable interior a(t): el perfil en [0.25, 0.5]M
  está fuertemente contaminado por la jerarquía ζ/η (fits con errores O(1)
  y de signo en la fase activa — el "zona log hasta r~0.5" de F0 era
  visual); [0.1, 0.5] con fit de orden 2 da ≤ 2.6 % (l=0,1). El término
  ζ₁·r·ln r tiene pendiente logarítmica nula en r = 1/e ≈ 0.37: los sesgos
  NO son monótonos entre ventanas — calibrar por ventana, no extrapolar.
- Sesgo vs varianza del fit interior EN 3D: el error de malla correlacionado
  entre radios se amplifica ∝ cond de la base — a resolución práctica el
  o2 (cond ~5·10⁴) da series sin sentido aunque su sesgo 1D sea el menor.
  Primario 3D = o1 [0.1,0.5] (cond ~600) + o0 de ancla de fase (cond ~6).
  El cociente puntual a_hat/a_lin es mal condicionado en el cruce por cero
  de a_lin: citar cocientes de picos/L2 sobre la fase fuerte.
- La BC característica es exactamente compatible con un vacío constante
  (φ=cte, Π=0 ⇒ término de borde nulo): correcta para corridas mexhat con
  u∞=v. La `sommerfeld_spherical` (término φ/r) advectaría el vacío — no
  usarla con u∞ ≠ 0.
- Balance de Killing a r_inner=0.1: la cuadratura del flujo interior sobre
  la esfera de excisión facetada (lc_inner=0.04 ⇒ facetas ~23°) pierde
  ~40 % de la energía absorbida durante el cruce del pulso — el residual
  NO es métrica de calidad del campo en esta geometría sin su propio
  estudio de resolución. El ζ_max del monitor de Cowling es global y lo
  domina el exterior de curvatura débil; el número interior de F0
  (ζ ≤ 1.2·10⁻³) sale de la versión restringida al interior.
- Modos de cavidad del dominio R=20 (barrera↔esponja): el suelo de cola
  (~2–4·10⁻⁴) es cuasi-estacionario, independiente de resolución Y de la
  excitación — NO es cola de Price ni error de malla. En l=2 es un doblete
  limpio w₁ = 0.351±0.003 / w₂ = 0.560±0.007 (no armónico; canario
  `test_cavity_mode_slow.py`); en l=1 NO es un doblete limpio. La cifra
  histórica "Mω=0.209 + armónico 0.419" era artefacto de bins de FFT
  (Δω = 2π/30M) — retractada. Pico/suelo ≈ 4 en ambos l e invariante del
  pulso; extraer en r_ext=4 lo sube a 6.4 a costa de sesgo near-zone.

## 5. Mapa de artefactos

| Qué | Dónde |
|---|---|
| Informe Fase 0 (GO) | `docs/research/phase0/report.md` |
| Convergencia F1 (interpretación) | `docs/research/phase1/convergence/note.md` |
| Disipación F1 (interpretación + datos) | `docs/research/phase1/dissipation/note.md` |
| Cavidad F1 (diseño espectroscopía F2) | `docs/research/phase1/cavity/note.md` |
| Literatura F2 (enunciado F–S verificado + refs ancla) | `docs/research/phase2/literature.md` |
| Diagnóstico interior F2 (estimador a(t) 1D+3D, calibración, humo A/B) | `docs/research/phase2/interior/note.md` |
| Humo A/B interior (números + protocolo) | `docs/research/phase2/interior/ab_smoke.json` + `scripts/interior_ab_smoke.py` |
| Matemática: 3+1, excisión, energía, Killing, disipación | `docs/math/*.md` |
| Validación general | `docs/validation/summary.md` |
| Oráculo 1D | `src/rsd/reference/spherical1d.py` |
| Solver 3D | `src/rsd/solvers/first_order.py` |
| Sondas F0 / estudios | `scripts/phase0_probes/`, `scripts/dissipation_study.py` |

## 6. Reglas de trabajo para quien continúe (humano o modelo)

1. Leer este archivo antes de tocar nada; al cerrar un capítulo, actualizar la
   sección correspondiente aquí (no solo dejar datos sueltos).
2. Encadenar fases autónomamente; detenerse solo ante un go/no-go ambiguo o
   acciones externas/irreversibles (acuerdo vigente con Marco).
3. Cada capítulo produce una nota interpretativa en `docs/research/`
   (números + decisión), no solo datos crudos.
4. Honestidad metodológica: alcances declarados (Cowling), límites
   cuantificados, nombres que describen lo que el código hace de verdad.
5. Los commits los coordina Marco (a veces vía GitHub Desktop); no asumir
   permiso para commitear/pushear.
