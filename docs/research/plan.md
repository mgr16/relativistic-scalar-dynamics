# Plan de investigación — Relativistic Scalar Dynamics (RSD)

**Estado a 2026-07-12.** Este archivo es la **copia canónica y versionada** del
plan del programa (antes vivía solo en la memoria local de Claude Code).
Cualquier persona, herramienta o modelo que continúe el trabajo debe **leer
este archivo primero** y **actualizar la sección correspondiente al cerrar cada
capítulo**. El plan original se estableció el 2026-06-12/13.

**Estado en una línea:** Fase 0 cerrada (GO). Fase 1: todo lo que bloquea la
física está CERRADO; quedan extras de §3.1 que no bloquean H2. **Fase 2
COMPLETA en sus dos capítulos de medición**: literatura (F–S) CERRADA;
diagnóstico interior CERRADO (2026-07-10); **PRODUCCIÓN INTERIOR CERRADA
(2026-07-11)** — a_hat/a_lin = O(1) en todos los modos (L2 0.87–0.94,
escalera estable al 2–3 %): **el borrado de H2 confirmado a ~10–15 % y por
modo**; acoplamiento no lineal ACOTADO < 5 %; mass lumping RECHAZADO para
el interior. **ESPECTROSCOPÍA EXTERIOR DE PRODUCCIÓN CERRADA (2026-07-12)**
— QNM l=2 validado vs Leaver: Re −1.9 % (escalera R=20, p≈1.8) / +0.8 % y
−Im +5.0 % ± 5.6 % (ventanas tardías R=40); los dos sistemáticos de −Im
identificados (suelo de cavidad R=20 — punto-estimado del capítulo de
cavidad RETRACTADO — y overtone n=1 no separable); modo de cavidad
confirmado por experimento de dominio R=40 apareado. **F3 ABIERTA
(2026-07-12)**: pipeline definido en §3.3 (C1–C5); **C1 related-work
CERRADO** (scoop check limpio; declaración de novedad borrador en
[`phase3/related_work.md`](phase3/related_work.md)); **C2 congelado de
números CERRADO** (mismo día; implementó GPT-5.6 Sol bajo
[`phase3/HANDOFF.md`](phase3/HANDOFF.md), revisó/cerró el orquestador):
calibración o1 = diagnóstico parcial (~31 % del déficit L2 explicado;
peak_ratio degradado a no-citable — el discriminador citable es
l2_ratio + ratio_median/IQR; l>0 sin corrección por piso de verdad) +
tabla canónica `phase3/numbers.{json,md}` (222 entradas, procedencia
RFC 6901, 0 pendientes); suite 184. **Desde el cierre de C2 rige el
contrato AUTÓNOMO `phase3/HANDOFF.md` §8** (orquestador offline hasta
~07-15): **C3 pipeline de figuras CERRADO por el implementador, pendiente
de auditoría Fable** (5 figuras / 10 PDF+PNG reproducibles, solo fuentes
versionadas, suite rápida 192); **C4 MANUSCRITO CERRADO por el
implementador, pendiente de auditoría Fable** (revtex4-2/PRD, 7 páginas,
16 referencias verificadas, 5 figuras, 97 macros trazables, suite rápida
207); **C5 EMPAQUETADO CERRADO por el implementador, pendiente de auditoría
Fable** (README, manifest SHA-256 de 32 artefactos, PDF byte-reproducible,
suite compuesta 214/214). La implementación F3 C1–C5 queda completa sin tag,
commit ni depósito externo. **AUDITORÍA Fable 2026-07-16 (HANDOFF §10):
C3 PASS · C5 PASS · PR#18 PASS · C4 PASS-CON-HALLAZGOS → reabierto como
RONDA R (C4b: completar manuscrito → robustez → depósito), contrato
autónomo HANDOFF §11, auditoría de ronda §12.** Los hallazgos que
bloquean arXiv son de completitud, no de validez: setup numérico ausente
del texto, ζ de Cowling sin citar, mecanismo cinético aseverado sin
mostrarse, sensibilidad de punto único. Opcionales de F2 no bloquean.

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
- Suite rápida de tests: 172 en verde (a 2026-07-12; corre con
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
- **PRODUCCIÓN INTERIOR CERRADA (2026-07-11;
  [`phase2/production/note.md`](phase2/production/note.md))** — matriz de
  12 corridas (0 fallos, 88 min pool-3, `scripts/interior_production.py`
  idempotente): escalera l=0 lc_inner {0.056, 0.04, 0.028} × {lineal,
  mexhat u∞=v}, l=1 @ 0.04, l=2 @ {0.04, 0.028} (banco lmax=4), pulso
  idéntico al humo. Resultados:
  - **Mass lumping RECHAZADO en la config de producción** (A/B etapa 0:
    Δa00 mediana 27.7 %, máx 38.6 % — el caveat de F1 materializado en el
    perfil interior sub-resuelto); producción = masa consistente.
  - Presupuesto: rung 0.056 fuera de régimen (no citable); diferencia
    0.04→0.028 rms 9–12 % (l=0) / 19–21 % (l=2); dev vs oráculo en fino
    13 %/8.6 % mediana (lin/hat l=0) con piso de sistemáticas de
    comparación 3D↔1D, σ_a OLS sí contrae (7.6 % en fino). l>0: error
    ABSOLUTO igual a l=0; relativo ×2 porque la señal cae con l (barrera
    centrífuga).
  - **El número de H2: a_hat/a_lin = O(1) en todos los modos** (L2
    0.92/0.94/0.94 escalera l=0; 0.87 l=1; 0.88–0.90 l=2; picos
    0.86–1.06; oráculo 1.01) — el borrado por dominación cinética se
    confirma por modo al nivel ~10–15 %; el déficit sistemático ~10 % del
    L2 3D (estable bajo refinamiento, se cancela parcialmente por
    correlación del par) queda abierto — paso barato: calibrar sesgo o1
    por perfil con el oráculo mexhat denso.
  - **Acoplamiento de modos no lineal: COTA < 4–5 % de la señal del modo**
    (A=0.1, fase fuerte); el "×3.3 en l=2 fino" pre-tope era junk tardío
    (retractado). Única señal débil: l=1 →0 ×2.2.
  - **Fase fuerte con tope t ≤ 10M** (nuevo): el junk tardío de dominio
    (R=15 sin esponja) crece a ~escala de señal en canales l>0 tras
    t≈10M; sin tope contamina dev/discriminador.
- **Dato l>0 en 3D implementado (2026-07-11):** gaussiana × Y_lm real
  ortonormal en `GaussianBump` (config `initial_conditions.l`/`.m`); usa
  la misma `real_ylm` del extractor ⇒ c_lm(r, 0) = A·g(r) en el canal
  (l, m), igual al modo del oráculo 1D con dato idéntico; el momento
  factoriza con el mismo ansatz radial del oráculo para todo l; l=0
  conserva la normalización esférica histórica (sin factor Y_00). Tests:
  `tests/test_initial_data_ylm.py` (convención dato↔extracción pineada
  exacta; fuga de canal ≤ 2.5 % en φ sobre la malla de test, convergente
  con h). Caveat: con potencial no lineal la reducción 1D solo es exacta
  para l=0 — el mexhat con l>0 acopla multipolos y es física exclusiva
  del 3D (el oráculo lo advierte en runtime).
- **ESPECTROSCOPÍA EXTERIOR DE PRODUCCIÓN CERRADA (2026-07-12;
  [`phase2/exterior/note.md`](phase2/exterior/note.md))** — diseño del
  capítulo de cavidad sin cambios (l=2, r_ext=6, ventanas ancladas) +
  validación de dominio R=40 con **graduación apareada** (misma lc(r) en
  r<20 ⇒ aísla dominio de resolución a ~×2 de costo) y esponja ancha:
  - **Re Mω:** escalera R=20 converge como el diseño predijo (−3.5 % →
    **−1.9 %** a lc=0.7, p≈1.8); dominio acotado ≲0.5 % (par apareado);
    pooled tardío R=40: +0.8 % ± 2.6 %.
  - **−Im Mω = 0.1016 ± 0.0055 (+5.0 % ± 5.6 %)** por barrido declarado de
    ventanas tardías en R=40 (14 ventanas × 2 rungs) — imposible en R=20.
    Dos sistemáticos medidos: (a) el suelo de cavidad R=20 hace in-medible
    −Im ahí (deriva de abanico; **el "−Im err 2.1 %" del capítulo de
    cavidad queda RETRACTADO como punto-estimado**); (b) el overtone n=1
    (Δω 4 %, decaimiento ×3) no es separable por Prony y sesga el abanico
    temprano +14–16 % independiente de la resolución (p≈−0.3: protocolo,
    no malla).
  - Modo de cavidad CONFIRMADO por manipulación del pozo: en R=40 el
    doblete no llena a t=70 (round-trip ~74M), suelo ×25–50 menor,
    pico/suelo 4 → 109–196; el doblete R=20 a lc=0.7 sigue en las bandas
    del canario.
  - Costos: lc=0.7 1 779 s; R=40 apareado 1 025 s (lc=1-eq) / 3 177 s
    (0.7-eq). Sub-1 % en Re (lc=0.5, ~2.4 h) y −Im < 5 % (R≥60,
    t_end≳100M) quedan como vías costeadas opcionales.
- **Opcionales de F2 (no bloquean):** corrida larga ℓ=0 para la era
  tardía (necesita esponja/R mayor y probablemente checkpoint/restart);
  calibración del sesgo o1 por perfil con el oráculo mexhat denso (ataca
  el déficit sistemático ~10 % del L2 del discriminador).
- **SIGUIENTE:** F3 — pipeline de paper.

### 3.3 Fase 3 — Pipeline de paper (ABIERTA 2026-07-12)

Objetivo: manuscrito arXiv-ready (inglés; **default revtex4-2 estilo PRD** —
el venue final lo decide Marco y no bloquea) con reproducibilidad total:
cada número y figura del paper debe regenerarse por script desde artefactos
versionados, nada re-tipeado a mano.

Capítulos, en orden:

1. **C1 — Related work (CERRADO 2026-07-12;
   [`phase3/related_work.md`](phase3/related_work.md)):** scoop check
   LIMPIO — no existe verificación numérica de F–S ni medición del perfil
   a(t) hacia r=0 (test field, lineal o Higgs). Lo más cercano:
   Thuestad–Khanna–Price 2017 (interior test-field, oscilaciones-l, sin
   asintóticas — cross-check pendiente y citable), Traykova–Braden–Peiris
   2018 + Marsden et al. 2024 (Higgs–BH EXTERIOR: tortuga 1+1 ⇒ nunca
   entran al horizonte; cascarones/burbujas) — somos su complemento
   interior. Contexto de quiescencia anclado (Rodnianski–Speck, FRS,
   +2026); contrastes en gravedad modificada identificados (Bars).
   Métodos: claim FEM = "poco común/to our knowledge", no "primero"; el
   sesgo de overtone conecta con el debate start-time (Giesler+ y
   2025–26). Declaración de novedad borrador en §5 de la nota; regla:
   toda ref [S] se promueve a [A/T] en el pase de bib de C4.
2. **C2 — Congelado de números (CERRADO 2026-07-12; implementó GPT-5.6
   Sol bajo contrato [`phase3/HANDOFF.md`](phase3/HANDOFF.md), revisó y
   cerró el orquestador — log §7 del handoff):**
   - **Calibración o1 por perfil** (`scripts/o1_profile_calibration.py`
     + `phase3/o1_calibration.json`, status reviewed-diagnostic):
     veredicto **parcial** — el sesgo de ventana o1 explica ~19–31 % del
     dev l=0 en los rungs en régimen (fino: 10.31→7.11 % lin,
     4.83→3.89 % hat) y ~31 % del déficit L2 del discriminador en
     soporte común (6.04→4.16 puntos vs oráculo 1.0055); el residual
     4.16 pts queda declarado como sistemática 3D real. c(t) estable en
     resolución (≤0.5 %) y muestreo K=32 subdominante (≤0.44 %); piso
     TRUTH_SCAN 1.8–2.4 % (l=0). Los números corregidos son DIAGNÓSTICO
     con soporte declarado — nunca sustituyen a `production.json`.
   - **La alarma contractual del discriminador tuvo mecanismo verificado**
     (réplica exacta del orquestador): el pico del par vive en tiempos
     distintos por miembro (lin t≈4.45 entrada del pulso, c≈1.11; hat
     t=5.88 fase fuerte, c≈0.99) ⇒ el peak_ratio congelado arrastra
     ~11 % de sesgo diferencial de ventana. **peak_ratio queda
     no-citable como número de H2; el discriminador citable es l2_ratio
     (primario) + ratio_median/IQR (secundario).** Refuerza H2: el l2
     corregido tiende al oráculo en los tres rungs (0.964/0.978/1.010).
   - **Corrección l>0 DENEGADA en forma definitiva** (piso TRUTH_SCAN de
     la propia verdad 10.1 %/6.6 % ≥ efecto; c_l>0 oscila ±40–80 %);
     el discriminador l>0 citable es el congelado.
   - **Tabla canónica** (`scripts/paper_numbers.py` →
     `phase3/numbers.{json,md}`): 222 entradas con procedencia
     `archivo::/JSON-Pointer` (RFC 6901) revalidada al emitir, cero
     números re-tipeados; transformaciones e incertidumbres con
     procedencia propia. Counts: 34 citable / 156 citable-con-caveat /
     22 no-citable / 10 degradado-a-prosa / 0 pendiente. Spot-checks
     del orquestador con resolutor independiente: 7/7.
   - Suite tras C2: **184 rápidos en verde** (+12 tests C2).
3. **C3 — Pipeline de figuras CERRADO 2026-07-12 (cierre por
   implementador; AUDITADO Fable 2026-07-16: PASS):**
   `scripts/paper_figures.py` (idempotente, `--check`) regenera 5 figuras
   de dos paneles desde JSON/NPZ versionados → 10 artefactos en
   `paper/figures/` (PDF vector + PNG). La fuente 3D faltante del perfil
   H2 se promovió como copia exacta a `phase3/data/` con hashes validados;
   el script no lee `results/`. Los números anotados se resuelven por id de
   `numbers.json` y el catálogo rechaza status no citables; se omitió el
   overlay L2 corregido y todo `peak_ratio`. Ocho tests C3 cubren fuentes,
   status, determinismo byte a byte, atomicidad y `--check`; suite rápida
   total: 192 verdes (MPI re-verificado fuera del sandbox).
4. **C4 — Manuscrito CERRADO 2026-07-12 (cierre por implementador;
   AUDITADO Fable 2026-07-16: PASS-CON-HALLAZGOS → reabierto como
   RONDA R, ver punto 6):**
   `paper/main.tex` + `paper/refs.bib` + `paper/numbers.tex` +
   `paper/main.pdf`, revtex4-2/PRD dos columnas. Manuscrito de 7 páginas
   con 5 figuras, 16/16 referencias verificadas/citadas, Cowling explícito
   en abstract/setup, novedad consistente con §5 y caveats conservadores.
   `paper_tex_numbers.py` genera 97 macros desde la tabla ampliada a 257
   entradas (39 altas JSON-backed de protocolo/malla/RMS); cero decimal de
   resultado re-tipeado y cero valor retractado/no-citable. Compila con
   Tectonic 0.16.9 + RevTeX 4.2e sin errores, overfull, referencias o citas
   indefinidas; QA visual de las 7 páginas aprobado. Trece tests C4 nuevos;
   suite rápida total tras la actualización de toolchain: 207 verdes + 7
   slow deseleccionados.
5. **C5 — Empaquetado reproducible CERRADO 2026-07-12 (cierre por
   implementador; AUDITADO Fable 2026-07-16: PASS):**
   `paper/README.md` documenta números → macros → figuras → PDF → manifest →
   tests. `scripts/paper_manifest.py --check` verifica 32 artefactos con
   SHA-256, rutas relativas, inventario explícito, rechazo de escapes/symlinks
   y escritura atómica/idempotente; 7 tests nuevos. El build probado fija
   Tectonic 0.16.9, bundle v33 y `SOURCE_DATE_EPOCH=1783814400`: dos builds
   consecutivos dieron el mismo PDF SHA-256
   `70ecbe1ea20e64f387f51b074444843c0ac66397c397ea9dd6299e4f7c064046`.
   Los checks secuenciales y la suite compuesta 214/214 están verdes; entrada
   `[Unreleased]` agregada a CHANGELOG. Tag propuesto `v3.3.0-paper`, no
   creado; depósito externo, tag y decisión de subir `pyproject.toml` desde
   3.2.0 siguen reservados a Marco.
6. **RONDA R — revisión C4b (ABIERTA 2026-07-16; contrato autónomo
   [`phase3/HANDOFF.md`](phase3/HANDOFF.md) §11; hallazgos de auditoría
   §10; auditoría de ronda §12):** Sol encadena R0 (saneo) → R1
   (manuscrito autocontenido: tabla de protocolo numérico JSON-backed,
   mecanismo cinético medido en 1D con artefacto nuevo versionado, ζ de
   Cowling citado, párrafo de detectabilidad, pase de jerga, apéndice de
   dato inicial, convergencia en prosa, data availability) → R2
   (robustez: barrido de sensibilidad 1D λ×A, +8 refs verificadas,
   HPC CI diagnosticado/pineado, lockfile del entorno; opcionales:
   TKP17, b_l, nivel lc=0.020 solo con OK de Marco) → R3 (preparación de
   depósito: re-scoop-check, drill de clon limpio, checklist de release
   — tag/Zenodo/arXiv = SOLO Marco). Gate de arXiv = cierre de R1; gate
   de PRD = cierre de R2. Fable audita la ronda completa a su vuelta.

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
- Mass lumping y el observable interior NO se llevan: en la config de
  producción (P1, lc_inner=0.04) desplaza a00 un 28 % mediana / 39 % máx
  (A/B 2026-07-11) aunque acelere ×3 — el perfil logarítmico junto a la
  excisión facetada es una estructura sub-resuelta en el sentido del
  caveat de F1. Interior = masa consistente, siempre.
- Los cocientes A/B con dato idéntico son mucho más estables que los dev
  individuales (spread de escalera 2–3 % vs dev 10–15 %): malla y
  estimador idénticos correlacionan los errores del par y se cancelan al
  primer orden. Diseñar TODA medición comparativa como par con dato
  idéntico, no como valores absolutos por corrida.
- Junk tardío de dominio en los canales interiores l>0 (R=15 sin
  esponja): crece hasta ~la escala de la señal después de t ≈ 10M.
  Ventana fuerte del interior SIEMPRE con tope superior (producción usa
  t ∈ [4, 10] M) — sin él, dev/discriminador/acoplamiento se contaminan
  (un falso "acoplamiento ×3.3" salió de ahí y está retractado en la
  nota de producción).
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
  CONFIRMACIÓN experimental (2026-07-12): alargar el pozo (R=40, esponja
  r>30) evita que llene a t=70 — suelo ×25–50 menor y sin doblete.
- El −Im de un QNM NO es medible con suelo de cavidad a pico/suelo ≈ 4:
  las ventanas tardías del abanico caen al suelo y el −Im por ventana
  deriva monótonamente (la media del abanico es ruido de esa deriva — así
  murió el "−Im err 2.1 %" de l=2/R=20, retractado). Medir −Im exige
  dominio grande (R≥40, pico/suelo ≳ 100) Y ventanas tardías: el overtone
  n=1 (Δω 4 %, decaimiento ×3) no es separable por Prony (devuelve el
  conjugado como "segundo modo") y sesga las ventanas tempranas +14–16 %
  con independencia de la resolución.
- Validación de dominio barata para mallas graduadas lc(r) lineal:
  aparear la graduación (lc_out' = lc_inner + (lc − lc_inner)·R'/R_base)
  reproduce la malla base en r < R_base y aísla el efecto de dominio del
  de resolución a ~×2 de costo (R'=2R_base), no ×8.
- El peak_ratio de un par A/B compara argmax que viven en TIEMPOS
  distintos por miembro (lin: entrada del pulso t≈4.5; hat: fase fuerte
  t≈5.9), y el sesgo de ventana o1 difiere ~11 % entre esos instantes
  (c(t) va de ~1.18 en la entrada a ~0.97 en el corazón): todo cociente
  de picos hereda ese diferencial. Citar L2 (integral, mismo soporte
  para ambos miembros) como primario del discriminador; el sesgo de
  ventana o1 explica ~19–31 % del dev l=0 en régimen y el resto es
  malla/extracción (C2, `phase3/o1_calibration.json`). La fase fuerte
  del ORÁCULO tiene un hueco interno (cruce por cero de a_truth,
  t≈5.1–5.6) y termina en t≈7.1 < tope 3D t=10: toda corrección por
  perfil se define solo en la intersección, sin relleno.

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
| Producción interior F2 (escalera l=0,1,2 + número de H2 por modo) | `docs/research/phase2/production/note.md` + `production.json` + `scripts/interior_production.py` |
| Espectroscopía exterior F2 (QNM l=2 vs Leaver + dominio R=40) | `docs/research/phase2/exterior/note.md` + `spectroscopy.json` + `scripts/exterior_spectroscopy.py` |
| Related work F3 (linaje de refs + declaración de novedad) | `docs/research/phase3/related_work.md` |
| Contrato C2 + log de orquestación F3 | `docs/research/phase3/HANDOFF.md` |
| Calibración o1 F3 (diagnóstico de sistemática del discriminador) | `docs/research/phase3/o1_calibration.json` + `scripts/o1_profile_calibration.py` |
| Tabla canónica de números del paper (procedencia RFC 6901) | `docs/research/phase3/numbers.{json,md}` + `scripts/paper_numbers.py` |
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
