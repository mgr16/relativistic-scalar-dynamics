# F3 · Capítulo 1 — Related work (pase amplio)

**Fecha:** 2026-07-12 · **Estado:** CERRADO (primer pase; el pase de
bibliografía fina se repite mecánicamente en C4 al construir `refs.bib`).
Este es el pase amplio que F2 difirió a F3
([`plan.md`](../plan.md) §3.3-C1). Objetivo: (a) verificar que la medición
central no está publicada por nadie (scoop check), (b) fijar el linaje de
referencias que el manuscrito debe citar, (c) borrador de la declaración
de novedad.

## 0. Método y niveles de verificación

Búsquedas web dirigidas (2026-07-12) en cuatro frentes: competencia
directa interior, línea Higgs–BH, contexto matemático de quiescencia,
métodos. Cada referencia lleva su nivel de verificación:

- **[T]** texto completo leído (heredado del capítulo de literatura de F2).
- **[A]** abstract/resumen verificado hoy contra la fuente (arXiv/PMC).
- **[S]** solo snippet de búsqueda — título+afiliación plausibles, contenido
  no verificado. **Regla:** ninguna afirmación del paper puede apoyarse en
  una ref [S] sin promoverla antes a [A] o [T] (pase de bib de C4).
- IDs marcados *(ID de memoria)* provienen del conocimiento del modelo, no
  de la búsqueda: verificar el número exacto en C4 antes de citar.

## 1. Competencia directa: campos en el interior, asintóticas hacia r=0

- **[T] Fournodavlos & Sbierski**, *Generic blow-up results for the wave
  equation in the interior of a Schwarzschild black hole*, ARMA 2019,
  arXiv:1804.01941 — el teorema que H2 pone a prueba (ψ = A·log r + B +
  O(r·log r), genericidad). Verificado a texto completo en F2
  ([`../phase2/literature.md`](../phase2/literature.md)).
- **[A] Thuestad, Khanna & Price**, *Scalar fields in black hole
  spacetimes*, PRD 96, 024020 (2017), arXiv:1705.04949 — **la numérica de
  campo de prueba interior más cercana que existe**: evolución tardía en
  Schwarzschild y Kerr incluyendo el interior; reportan oscilaciones
  interiores caracterizadas por el índice l. El abstract NO menciona
  perfiles logarítmicos, ni r→0, ni comparación con F–S (el paper es
  anterior a F–S). **TODO para C4:** comprobar en nuestras series si sus
  oscilaciones-l aparecen (validación cruzada barata y citable).
- **[S] Línea interior autogravitante clásica** (contexto, no
  competencia — otro sistema: colapso con carga/self-gravity):
  Burko & Ori, *Black hole singularities: a numerical approach*
  (gr-qc/9506067); Burko, *Homogeneous spacelike singularities inside
  spherical black holes* (gr-qc/9711012); *Late-time evolution of
  nonlinear gravitational collapse* (gr-qc/9703067); *Numerical evolution
  of the interior geometry of charged black holes* (arXiv:2001.02788;
  interior "scalarized Kasner" cerca de r=0); *Towards a spacelike
  characterization of the null singularity inside a black hole*
  (arXiv:2503.20969, 2025).
- **[S] Interiores de BH con pelo / holografía**: *The interior of the
  scalar hairy black hole with inverted Higgs potential*
  (arXiv:2603.08067, 2026) — línea "interior Kasner de BHs con pelo"
  (Hartnoll et al. como raíz *(atribución de memoria)*): autogravitante,
  pelo en el horizonte; no es campo de prueba ni discriminador de vacío.

**Veredicto scoop-check interior:** ninguna búsqueda devolvió una
verificación numérica de las asintóticas log de F–S ni una medición del
perfil a(t) hacia r=0 para campo de prueba, lineal o Higgs. El hueco que
ocupamos sigue abierto.

## 2. Línea Higgs–BH (física de H2 y H1)

- **[A→casi T] Traykova, Braden & Peiris**, *Accretion of a
  symmetry-breaking scalar field by a Schwarzschild black hole*, Phil.
  Trans. R. Soc. A 376:20170122 (2018), arXiv:1711.00854 (leído vía PMC) —
  campo de prueba con doble pozo dependiente de temperatura
  V = σ(T/T₀)²φ² − λφ⁴ sobre Schwarzschild, **1+1 esférico,
  pseudo-espectral (Chebyshev), coordenada tortuga (el horizonte se va a
  −∞ ⇒ por construcción NUNCA entran al interior)**. Resultado: la
  dilatación temporal congela el campo cerca del horizonte — "no se rompe
  la simetría cerca del horizonte" mientras lejos sí. **Es el complemento
  exacto de H2 desde afuera**: ellos ven la estructura de vacío congelada
  en el exterior; nosotros medimos su borrado en el interior. Cita
  obligada en la intro.
- **[A] Marsden, Aurrekoetxea, Clough & Ferreira**, *Symmetry restoration
  and vacuum decay from accretion around black holes*, arXiv:2403.17595
  (2024) — acreción clásica ⇒ cascarones de restauración de simetría
  alrededor del BH ⇒ cataliza el decaimiento del vacío (burbuja que
  expande a ~c). Todo exterior (el abstract no menciona interior ni
  singularidad). Refuerza que la región interior es el hueco.
- **[T/S] H1 (respaldo, anclas ya fijadas en F2):** Burda–Gregory–Moss
  (arXiv:1501.04937 PRL, 1503.07331 JHEP) y el contrapunto 2023 *BHs
  don't source fast Higgs vacuum decay* (JHEP 03(2023)039) — pinneados en
  [`../phase2/literature.md`](../phase2/literature.md). Nuevos [S] del
  pase de hoy: *On thermal false vacuum decay around black holes*
  (arXiv:2210.08028); Canaletti & Moss, *Seeding the decay of the false
  vacuum*, PRD 110, 105015 (2024; arXiv:2408.12229); *The Signals of
  Doomsday I* (arXiv:2501.15848, 2025). Suficiente para el párrafo de
  contexto H1 del paper; H1 sigue sin activarse.
- **Contraste (afirman lo contrario en teorías MODIFICADAS — citables en
  la discusión, no compiten con la medición):**
  - **[A] Bars**, *The Higgs field governs the interior spacetime of
    black holes*, arXiv:2509.06800 (2025; hep-th, 40 pp.) — en su
    programa i(SM+GR) localmente conforme (con antigravedad), el Higgs
    extiende el espaciotiempo más allá de la singularidad. GR modificada:
    útil como contraste explícito de la discusión ("en GR mínima el
    interior BORRA el Higgs; para que el Higgs gobierne el interior hace
    falta salir de GR"). Versión carta: Phys. Lett. B 2026 [S].
  - **[S]** *Black hole singularity avoidance by the Higgs scalar field*
    (EPJC, arXiv:1901.05295): sector Higgs-fermión viola condición de
    energía dominante ⇒ evitación; también fuera de nuestro régimen.
  - **[A, NO citar salvo necesidad]** Musielak, Fry & Kanan
    (arXiv:2604.01246, 2026, **physics.gen-ph**): Klein-Gordon
    "modificado por teoría de gauge y grupos", analítico, campo regular
    en la singularidad. Categoría gen-ph y ecuación no estándar.

## 3. Contexto matemático: dominación cinética y quiescencia

La historia física detrás de H2 en el caso AUTOGRAVITANTE está probada
rigurosamente: el campo escalar (materia stiff) vuelve quiescente la
singularidad — sin oscilaciones BKL — y el comportamiento hacia la
singularidad es monótono, dominado por el término cinético. Nuestro
resultado es la sombra test-field de esa historia dentro del horizonte
de Schwarzschild, con F–S como puente lineal riguroso.

- **[S] Rodnianski & Speck**, *Stable Big Bang formation in near-FLRW
  solutions to the Einstein-scalar field and Einstein-stiff fluid
  systems*, Selecta Math. 2018 (arXiv:1407.6298).
- **[S] Fournodavlos, Rodnianski & Speck**, estabilidad de Kasner-scalar
  en todo el régimen subcrítico (arXiv:2012.05888 *(ID de memoria)*).
- **[S]** *Formation of quiescent big bang singularities*
  (arXiv:2309.11370) y *Complete asymptotics in the formation of
  quiescent big bang singularities* (arXiv:2602.02373, **feb 2026** —
  citar como estado del arte).
- **[S]** *Scattering towards the singularity for the wave equation and
  the linearized Einstein-scalar field system in Kasner spacetimes*
  (arXiv:2401.08437) y *The wave equation near flat FLRW and Kasner Big
  Bang singularities* (JHDE 2019, doi 10.1142/S0219891619500140; Alho–
  Fournodavlos–Franzen *(atribución de memoria)*) — asintóticas de ondas
  hacia singularidades tipo Kasner: el pariente cosmológico de F–S.
- **[S]** *Stable space-like singularity formation for axi-symmetric and
  polarized near-Schwarzschild black hole interiors* (arXiv:2004.00692)
  y An & Gajic *(atribución de memoria)*, *Curvature blow-up rates in
  spherically symmetric gravitational collapse to a Schwarzschild black
  hole* (arXiv:2004.11831) — la singularidad de Schwarzschild como
  atractor local: apuntala la relevancia del fondo fijo.

## 4. Métodos

- **FEM en relatividad numérica es raro** (nuestro claim metodológico es
  defendible como "poco común", NO como "primero"): [S] Sopuerta &
  Laguna, FEM toy model para EMRIs (gr-qc/0507112); serie *Binary black
  hole simulation with an adaptive finite element method* I–III
  (arXiv:1805.10640, 1805.10642); review *Finite elements in numerical
  relativity* — un snippet de búsqueda afirma explícitamente que "no hay
  código FEM usado para merger de BBH". Nadie apareció usando
  FEniCS/DOLFINx para evolución con excisión sobre fondo BH; formular
  como "to our knowledge". [S] arXiv:2507.18934 (drag de dark matter
  escalar sobre binaria, 2025) salió en la búsqueda FEM — verificar
  método en C4 antes de usarla de ejemplo.
- **Espectroscopía QNM — el debate de overtones y tiempo de inicio** (a
  donde se enchufa nuestro sesgo de ventana temprana +14–16 % por n=1 no
  separable, y el retracto del −Im de cavidad): [S] Giesler, Isi, Scheel
  & Teukolsky, *Black hole ringdown: the importance of overtones*
  (arXiv:1903.08284 *(ID de memoria)*); *Agnostic black hole
  spectroscopy* (arXiv:2302.03050); *Quasinormal mode content of binary
  black hole ringdowns* (arXiv:2510.13954); *High-overtone ringdown
  fits: start time, no-hair tests, and correlations* (arXiv:2512.08098);
  *Deterministic trust regions for finite-window black-hole spectroscopy
  in GW250114* (arXiv:2604.17558, 2026). El paper debe presentar nuestro
  presupuesto de −Im como un caso limpio y controlado de ese debate
  (campo de prueba: sin no-linealidad de fondo, overtone conocido).
- **Artefactos de dominio finito / esponjas**: práctica estándar
  documentada ([S] Buchman & Sarbach, *Improved outer boundary
  conditions* (gr-qc/0703129); esponjas en códigos espectrales, p. ej.
  arXiv:1206.3015). No encontré una caracterización publicada del **modo
  de cavidad barrera↔esponja con confirmación experimental por
  manipulación del pozo + graduación apareada** — nuestro capítulo de
  cavidad/dominio es citable como contribución metodológica menor.
  Tangencial: *absorption-induced mode excitation* (arXiv:2112.11168).

## 5. Declaración de novedad (borrador para el paper, EN)

> We present, to our knowledge, the first numerical measurement of the
> logarithmic blow-up profile a(t,ω) of test scalar fields approaching
> the Schwarzschild singularity, verifying the Fournodavlos–Sbierski
> asymptotics mode by mode in 3+1D, and the first identical-data
> linear-vs-Higgs comparison of that profile, which quantifies the
> kinetic-domination erasure of vacuum structure in the black hole
> interior (a_hat/a_lin = O(1) at the 10–15% level across l=0,1,2).
> Prior numerical work on symmetry-breaking scalars near black holes
> [TBP18, MACF24] concerns exterior accretion and freezes at the
> horizon by construction; prior interior test-field evolutions [TKP17]
> did not extract near-singularity asymptotics. On the methods side we
> contribute a rare FEM/excision evolution stack (DOLFINx) with a
> Killing-energy balance diagnostic, a calibrated multi-radius interior
> estimator, and an honest QNM error budget that isolates two
> systematics (finite-domain cavity floor; unresolved n=1 overtone)
> relevant to the ringdown start-time debate.

Cada "first/to our knowledge" está respaldado por el veredicto de §1–§4;
si el pase de bib de C4 encuentra un contraejemplo, se degrada la frase,
no se defiende.

## 6. Impacto en el manuscrito

1. **Intro:** encuadrar como el eslabón interior que falta entre (a) la
   línea exterior Higgs–BH (TBP18 congelado en horizonte; MACF24
   cascarones/burbujas) y (b) el teorema F–S + la historia de quiescencia
   autogravitante (RS/FRS). H1 queda en un párrafo con sus anclas.
2. **Validación:** añadir el cross-check de las oscilaciones-l de TKP17
   sobre nuestras series interiores (tarea nueva, barata, para C2/C4).
3. **Discusión:** contraste explícito con teorías modificadas (Bars) —
   nuestra medición fija el comportamiento en GR mínima.
4. **Sección QNM:** citar el debate de overtones/start-time y presentar
   nuestros dos sistemáticos como su versión controlada de campo de
   prueba.
5. **C4 bib pass (regla):** promover toda ref [S] usada a [A/T],
   verificar los *(ID de memoria)* y las atribuciones de autor marcadas.

## 7. Cobertura y límites del pase

Cubierto: interior/asintóticas, Higgs–BH clásico, quiescencia matemática,
FEM-NR, práctica QNM, artefactos de dominio. No cubierto (no bloquea, se
abre solo si el paper lo toca): superradiancia/nubes bosónicas, QFT en
espaciotiempo curvo cerca de singularidades (efectos cuánticos quedan
explícitamente fuera del alcance declarado), literatura de Cowling en
estrellas de neutrones (el término viene de ahí; buscar la cita canónica
en C4 si el referee la pide).
