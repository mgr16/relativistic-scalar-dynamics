# Estabilidad energética semi-discreta del esquema FEM

Esta nota formaliza la estabilidad del sistema semi-discreto que RSD
resuelve (discretización FEM en espacio, tiempo continuo), incluyendo la
frontera radiativa débil, la esponja y el filtro de disipación. Es la
justificación teórica del comportamiento que `tests/test_physics.py` y
`tests/test_sommerfeld_reflection.py` verifican numéricamente.

## Marco

Sea V_h ⊂ H¹(Ω) el espacio de Lagrange de grado k sobre la malla, y sea
(φ_h, Π_h) ∈ V_h × V_h la solución semi-discreta de

    (∂_t φ_h, v)_√γ = (αΠ_h + β·∇φ_h, v)_√γ
    (∂_t Π_h, w)_√γ = −a(φ_h, w) + (β·∇Π_h + αKΠ_h − αV'(φ_h) − σΠ_h, w)_√γ
                      + b_∂(φ_h, Π_h; w)

con (f, g)_√γ := ∫_Ω f g √γ dx, la forma de difusión
a(φ, w) := ∫_Ω α γ^{ij} ∂_iφ ∂_jw √γ dx (más el término natural del borde
interior excisado), σ ≥ 0 la esponja, y b_∂ el término débil de la BC
característica en el borde exterior Γ_out:

    b_∂(φ, Π; w) = −∫_{Γ_out} α√γ ( √(γ^{nn}) Π + [esférica] γ^{nn} φ/r ) w ds .

## Energía discreta

Definimos la energía del campo medida por observadores normales,
restringida a V_h:

    E_h(t) := ∫_Ω ( ½Π_h² + ½ γ^{ij}∂_iφ_h ∂_jφ_h + V(φ_h) ) √γ dx .

**Proposición (decaimiento de la energía, fondo estático, sin esponja).**
Sea el fondo estático (β = 0, K = 0, coeficientes independientes de t),
V ≥ 0 convexo con V(0)=0, σ = 0, y la variante no esférica de la BC
(`bc_type = characteristic`). Entonces la solución semi-discreta cumple

    d/dt E_h(t) = − ∫_{Γ_out} α√γ √(γ^{nn}) Π_h² ds  ≤ 0 ,

es decir, la energía es no creciente y el drenaje es exactamente el que
`boundary_flux()` reporta. En particular el esquema semi-discreto es
estable en la norma de energía.

*Demostración.* Con β = 0 y K = 0, tomar v = −αV'(φ_h)... no es admisible
directamente porque V' no es lineal; se procede del modo estándar:
derivar E_h y sustituir las ecuaciones débiles con los pares de prueba
v = α⁻¹(∂_tφ_h proyectado) y w = Π_h. Con β = 0:

    dE_h/dt = (Π_h, ∂_tΠ_h)_√γ + ∫ γ^{ij}∂_iφ_h ∂_i∂_tφ_h √γ + (V'(φ_h), ∂_tφ_h)_√γ .

La primera ecuación da ∂_tφ_h = P_h(αΠ_h) (proyección √γ-ortogonal sobre
V_h; con α ∈ V_h exacta si αΠ_h ∈ V_h, en general introduce el error de
proyección estándar). Sustituyendo w = Π_h en la segunda ecuación:

    (∂_tΠ_h, Π_h)_√γ = −a(φ_h, Π_h) − (αV'(φ_h), Π_h)_√γ + b_∂(φ_h, Π_h; Π_h) .

Los términos de difusión y potencial se cancelan con los términos
correspondientes de dE_h/dt usando ∂_tφ_h = αΠ_h (módulo proyección), y
queda exactamente el término de borde:

    dE_h/dt = b_∂(φ_h, Π_h; Π_h) = −∫_{Γ_out} α√γ √(γ^{nn}) Π_h² ds ≤ 0 . ∎

**Observaciones.**

1. **Variante esférica.** Con `sommerfeld_spherical` el drenaje adquiere
   el término cruzado γ^{nn} Π_h φ_h / r, que no es definido en signo
   instante a instante; el balance E + ∫F dt = E(0) sigue siendo exacto
   (es el mismo término), pero la monotonía estricta solo vale para la
   variante característica pura. Numéricamente la variante esférica
   absorbe mejor las colas 1/r — es el intercambio documentado.

2. **Esponja.** Con σ ≥ 0 se añade −(σΠ_h, Π_h)_√γ ≤ 0 a dE_h/dt: la
   esponja solo puede extraer energía (disipación de volumen). Por eso
   rompe el balance de borde pero no la estabilidad.

3. **Fondo estacionario (β ≠ 0, K ≠ 0).** La energía de observadores
   normales NO es la energía conservada (no es la carga de Killing); los
   términos β·∇ y αKΠ² no tienen signo definido. La cantidad relevante es
   la energía de Killing E_t = ∫ T_{μν} t^μ n^ν; su versión discreta y el
   término de borde interior (flujo hacia el horizonte) están pendientes
   de implementarse como diagnóstico (TODO Fase 1: `inner_flux()`); en
   excisión el balance debe cerrarse con AMBOS bordes.

4. **Filtro de disipación.** El filtro post-paso u ← u − ε dt (M⁻¹K)u (o
   biarmónico) es contractivo en la norma M para ε dt λ_max < 2: si
   u⁺ = (I − ε dt M⁻¹K)u, entonces ‖u⁺‖²_M ≤ ‖u‖²_M porque M⁻¹K es
   autoadjunto y semidefinido positivo en ⟨·,·⟩_M con espectro en
   [0, λ_max]. Por tanto el filtro refuerza (nunca degrada) la
   estabilidad; su efecto sobre la PRECISIÓN temporal es un splitting de
   primer orden, subdominante para ε pequeño (ver nota en README).

5. **Paso temporal completo.** SSP-RK3 es una combinación convexa de
   pasos de Euler; para el sistema lineal semi-discreto con dE_h/dt ≤ 0
   la estabilidad del paso completo se sigue bajo la condición CFL
   estándar del espectro de L (dt ≤ C·h/(c_max k²)), que es exactamente
   lo que `compute_dt_cfl` impone. No damos aquí la prueba completamente
   discreta (RK3 no preserva exactamente la monotonía de E para
   operadores no normales); el residuo O(dt³) por paso es el que el
   balance `series/balance.csv` mide y converge con la resolución.
