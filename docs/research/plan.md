# Plan de investigación — Relativistic Scalar Dynamics (RSD)

**Estado a 2026-07-09.** Este archivo es la **copia canónica y versionada** del
plan del programa (antes vivía solo en la memoria local de Claude Code).
Cualquier persona, herramienta o modelo que continúe el trabajo debe **leer
este archivo primero** y **actualizar la sección correspondiente al cerrar cada
capítulo**. El plan original se estableció el 2026-06-12/13.

**Estado en una línea:** Fase 0 cerrada (GO). Fase 1 en curso — capítulos de
convergencia y disipación CERRADOS, mass lumping hecho; queda el resto de
§3.1 (nada en vuelo: árbol de trabajo limpio a 2026-07-09).

---

## 1. Hipótesis y alcance

- **H2 (titular):** *la dominación cinética cerca de la singularidad de
  Schwarzschild borra la estructura de vacío tipo Higgs; se recuperan
  asintóticas logarítmicas lineales* (à la Fournodavlos–Sbierski; el enunciado
  exacto se verificará en el pase de literatura de Fase 2).
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
- Suite rápida de tests: 144 en verde (a 2026-07-09; corre con
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

**Resto de F1 (pendiente):** opción interpolate-coefficients; excisión
esferoidal (a > 0.9718M); checkpoint/restart; recalibrar tests lentos de Kerr;
**mitigación del modo de cavidad antes de la espectroscopía de F2** (R ≥ 40 +
esponja más ancha, o l=2 — periodo 13M —, o ajuste conjunto ring+cavidad).
Pregunta menor abierta: degradación residual de orden en el triplete fino
(¿interpolación de extracción / esponja / ruido de remallado no anidado?).

### 3.2 Fase 2 — Producción física (PENDIENTE)

Pase de literatura (verificar el enunciado exacto de Fournodavlos–Sbierski) +
diagnósticos interiores 3D + corridas de producción: campo iniciando en el
vacío (u∞ = v), r_inner = 0.25M (la zona logarítmica llega hasta r ~ 0.5M),
dato `ingoing_curved`, A ≤ 0.1, modos l = 1, 2.

### 3.3 Fase 3 — Pipeline de paper (PENDIENTE)

## 4. Hechos técnicos establecidos (no re-derivar)

- Kerr-Schild: la velocidad característica entrante es exactamente 1; el ring
  llega con retardo tipo tortuga — anclar ventanas de ajuste al pico
  detectado. El QNM l=0 está mal condicionado (Q≈0.5): validar con l=1,2.
- El perfil interior ES logarítmico sobre 2 décadas durante la fase activa
  (pendiente log independiente de r); la amplitud a r fijo decae conforme la
  cola de Price alimenta el horizonte.
- Modo de cavidad Mω = 0.209: suelo espectral plano ~2·10⁻⁴, independiente de
  resolución — NO es cola de Price ni error de malla.

## 5. Mapa de artefactos

| Qué | Dónde |
|---|---|
| Informe Fase 0 (GO) | `docs/research/phase0/report.md` |
| Convergencia F1 (interpretación) | `docs/research/phase1/convergence/note.md` |
| Disipación F1 (interpretación + datos) | `docs/research/phase1/dissipation/note.md` |
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
