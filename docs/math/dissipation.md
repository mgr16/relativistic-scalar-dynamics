# Disipación numérica: qué es (y qué no es) el filtro del solver 3D

Honestidad de nomenclatura primero: en el solver FEM 3D
(`rsd.solvers.first_order`) la disipación **no es Kreiss–Oliger** en el
sentido estricto de diferencias finitas. Es un **filtro espectral FEM**
construido con la matriz de rigidez `K` y la de masa `M` del propio espacio
de elementos. Lo llamamos "estilo KO" porque su *efecto* es el de KO
—amortiguar los modos de malla de alta frecuencia dejando casi intactos los
modos físicos suaves— pero el operador es distinto. El KO auténtico (D⁴
adimensional en índices de grilla, post-paso) sí existe y está
correctamente nombrado en el oráculo 1D `rsd.reference.spherical1d`, que es
una discretización por diferencias finitas.

Los parámetros canónicos son `solver.filter_strength` (ε) y
`solver.filter_order` (2 ó 4). `solver.ko_eps` / `solver.ko_order` se
conservan como **alias retrocompatibles**.

## El operador

Tras cada paso SSP-RK3 se aplica una corrección explícita al estado
`u = (φ, Π)`. Sea `A = M⁻¹K` el laplaciano FEM discreto ponderado por la
métrica (`K` ensambla `∫ √γ (γ⁻¹∇·,∇·)`, `M` ensambla `∫ √γ (·,·)`):

    orden 2 (laplaciano):   u ← u − ε·dt · A u
    orden 4 (biarmónico):   u ← u − (ε·dt / λmax) · A² u

con `λmax = λmax(A)` estimado por iteración de potencias (15 iteraciones,
margen +10 %). `A` es simétrico y semidefinido positivo en el producto de
masa (⟨u, Av⟩_M = ∫ √γ ∇u·∇v ≥ 0), así que sus autovalores λ_k ≥ 0 y sus
autovectores forman una base M-ortogonal.

## Interpretación espectral y por qué el orden 4 es honesto

Descompón `u = Σ_k c_k v_k` en autovectores de `A` (A v_k = λ_k v_k). Un
paso del filtro multiplica cada coeficiente por un factor de amortiguación:

    orden 2:   c_k ← (1 − ε·dt·λ_k) c_k
    orden 4:   c_k ← (1 − ε·dt·λ_k · (λ_k/λmax)) c_k

En ambos casos el modo de malla más rápido (λ_k ≈ λmax) se amortigua con la
**misma** tasa `ε·dt·λmax`, de modo que la condición de estabilidad es
idéntica: `ε·dt·λmax < 2` (fuera de eso el filtro amplifica). La
normalización por `λmax` es justo lo que iguala esa cota entre órdenes; sin
ella `h²` subestima λmax en mallas con celdas de mala calidad y el orden 4
(sensibilidad cuadrática) se desestabiliza.

**La cota se verifica en tiempo de ejecución.** Como `λmax ~ 1/h_min²`, el
ε máximo estable **depende de la malla** (`ε_max = 2/(dt·λmax)`, y con
`dt ∝ h_min` por CFL, `ε_max ∝ h_min`): un ε válido en una malla gruesa
puede divergir en una fina. Cruzar la cota no "disipa más" — convierte el
filtro en un amplificador exponencial (así se descubrió: un barrido con
ε = 0.05 sobre la malla interior fina divergió a ~10¹⁴⁸ en t = 20M;
ver `docs/research/phase1/dissipation/note.md`). Por eso el solver estima
λmax por iteración de potencias para **ambos** órdenes y, al primer paso,
lanza `RuntimeError` con el `ε_max` concreto de la malla si
`ε·dt·λmax ≥ 2`, además de registrar el número de amortiguación en el log.

La diferencia está en los modos **suaves** (λ_k ≪ λmax), que son los que
portan la física:

    factor de daño orden 2  ∝  λ_k
    factor de daño orden 4  ∝  λ_k² / λmax  =  λ_k · (λ_k/λmax)  ≪  λ_k

es decir, el orden 4 es más **transparente** a los modos suaves por el
factor `λ_k/λmax ≪ 1`. Ésa es la propiedad KO que sí se conserva: filtrar
la basura de malla sin tocar (apreciablemente) la señal. `tests/test_physics.py`
verifica cuantitativamente que, a igual ε, el orden 4 disipa un pulso suave
mucho menos que el orden 2.

## Consecuencia para el balance de energía

El filtro extrae energía **fuera** de las leyes de balance físicas: ni el
balance euleriano ni el de Killing (`docs/math/killing_energy.md`) lo
contabilizan como flujo de superficie. Por eso todo test de cierre exacto
del balance corre con `filter_strength = 0`. En producción esto no es un
problema porque —ver abajo— las corridas interiores de referencia usan
ε = 0.

## Efecto sobre los observables (honestidad de resultados)

La pregunta que un referee hará: *¿los observables del programa H2 (perfil
de log-slope a(t) en el interior, cierre del balance de Killing, forma de
onda extraída) son artefactos de esta disipación artificial?*

Dos respuestas, en orden de fuerza:

1. **Las corridas interiores de referencia usan `filter_strength = 0`.** Los
   probes 3D de Fase-0 (A/B/C, `scripts/phase0_probes/`) y la escalera de
   convergencia corrieron **sin filtro** y fueron estables
   (`docs/research/phase0/report.md`). Donde no hay filtro no hay
   contaminación posible: es la afirmación más limpia.

2. **Cuando el filtro se enciende, el sesgo es acotado y pequeño.** Para las
   corridas donde el filtro sí ayuda (p. ej. control de ruido de malla en el
   caso no lineal de Higgs), cuantificamos el desplazamiento de los
   observables entre ε = 0 y el ε de trabajo. Resultados y regresión en
   [`docs/research/phase1/dissipation/note.md`](../research/phase1/dissipation/note.md).

## Configuración

| clave (canónica) | alias | significado |
|---|---|---|
| `solver.filter_strength` | `solver.ko_eps` | amplitud ε ≥ 0 (0 = sin filtro) |
| `solver.filter_order` | `solver.ko_order` | 2 (laplaciano) ó 4 (biarmónico) |

Si se dan ambos nombres, gana el canónico. El oráculo 1D
(`rsd.reference.spherical1d`) mantiene su propio `ko_eps` porque allí sí es
Kreiss–Oliger de diferencias finitas.
