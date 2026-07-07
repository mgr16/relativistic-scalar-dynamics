# Energía de Killing: el balance que sí cierra en foliaciones estacionarias

La energía euleriana E = ∫√γ ρ d³x (con ρ = T_ab n^a n^b) **no** satisface
una ley de conservación con flujo puro de superficie sobre fondos
Kerr-Schild: el observador normal n^a no es de Killing, y los términos de
volumen β/K aparecen como producción aparente (ver
`energy_stability.md`). Esa es la razón por la que el residual de
`balance.csv` no converge a cero en runs con excisión, aun con el flujo
interior advectivo (validado en Fase 0: el término −(β·n)ρ corrige el signo
pero sobrecuenta; `docs/research/phase0/report.md`, §5).

La cantidad correcta es la **energía de Killing** asociada a ξ = ∂_t, que en
Kerr-Schild es exactamente Killing (fondo estacionario). La corriente
J^a = −T^a_b ξ^b satisface ∇_a J^a = 0 on-shell sin términos de volumen.

## Forma 3+1

Con n^a el normal unitario futuro y ξ = α n + β:

    ε_K ≡ T_ab n^a ξ^b = α ρ + Π β^i ∂_i φ,
    E_K = ∫_Σ ε_K √γ d³x,

donde usamos T_ab n^a γ^b_i = Π ∂_i φ. Fuera del horizonte (α≈1, β≈0)
E_K ≈ E; dentro, ξ es espacial y ε_K puede cambiar de signo (energía "vista
desde infinito" — exactamente lo que absorbe el agujero).

## El flujo es exacto y simple

Para cualquier métrica, T^i_t = ∂^i φ ∂_t φ (el término de presión cae
porque g^{ia} g_{at} = δ^i_t = 0). En variables del solver:

    ∂_t φ = α Π + β·∇φ,
    ∂^i φ = β^i Π / α + γ^{ij} ∂_j φ,

y la ley de conservación en coordenadas (∂_μ(√−g T^μ_t) = 0, √−g = α√γ) da

    d E_K/dt = ∮_∂Ω α √γ T^i_t n̂_i dS,

con n̂ el normal coordenado saliente del dominio (el `FacetNormal` del FEM y
la medida `ds` estándar: la identidad de Gauss se aplica en coordenadas).
La **energía que sale** por cada borde (positiva al salir) es

    F = −∮ √γ (α Π + β·∇φ) (Π β·n̂ + α n̂·γ⁻¹∇φ) dS ,

sin divisiones por α (seguro dentro del horizonte). En el borde excisado n̂
apunta hacia el agujero y F > 0 es absorción; en el exterior F > 0 es
radiación. El balance exacto del continuo es

    E_K(t) − E_K(0) + ∫ (F_outer + F_inner) dt = 0 ,

y el residual discreto debe converger a 0 con la resolución (esquema de
2.º orden en espacio; RK3 en tiempo). El filtro de disipación (`ko_eps>0`)
extrae energía fuera de este balance: para tests de cierre usar ko_eps=0 o
incluir su efecto en la tolerancia. La esponja también lo rompe por diseño.

## Reducción esférica (oráculo 1D)

Para φ = u(t,r) Y_lm sobre Schwarzschild-KS (f = 1+2M/r, α = f^{−1/2},
β^r = (2M/r)/f, w = r²√f):

    ε_K = α ρ + β^r Π ∂_r u,          E_K = ∫ w ε_K dr,
    T^r_t = (α Π + β^r u′) · α (2M/r · Π + α u′),
    d/dt ∫ w ε_K dr = [α w T^r_t]_{r_min}^{r_max}.

Chequeos analíticos incorporados en la derivación:
- g^{rt} g_tt + g^{rr} g_rt = 0 (el término −L del flujo cae exactamente);
- onda entrante nula (u̇ = u′): Π = α u′, ε_K = α u′² > 0 y
  T^r_t = u′² > 0 ⇒ el flujo por r_min es positivo saliente (absorción);
- M → 0: ε_K = ρ, T^r_t = Π u′ (resultado plano estándar).

Implementación: `SphericalOracle1D.energy_killing()` /
`killing_boundary_flux()` (1D), `FirstOrderKGSolver.energy_killing()` /
`killing_flux()` / `killing_inner_flux()` (3D), serie `series/killing.csv`
del CLI. Validación: `tests/test_killing_energy.py`.
