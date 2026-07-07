# Ventana admisible del radio de excisión en Kerr-Schild

La frontera interior excisada usa la condición natural ("do-nothing"),
válida solo si **ninguna característica re-entra al dominio** por ella.
Esta nota deriva la condición exacta para esferas de excisión cartesianas
y el rango admisible de `mesh.r_inner` en función del spin.

## Geometría

En Kerr-Schild cartesiano, el radio de Boyer-Lindquist r satisface
r² = ½(ρ² − a² + √((ρ²−a²)² + 4a²z²)), con ρ² = x²+y²+z². Sobre la esfera
cartesiana ρ = R_exc:

- en los polos (x=y=0):      r = R_exc        (máximo)
- en el ecuador (z=0):       r = √(R_exc²−a²) (mínimo, si R_exc > a)

Es decir, la esfera cartesiana NO es una superficie de r constante: barre
r ∈ [√(R_exc²−a²), R_exc].

## Condición de outflow

En la región atrapada r₋ < r < r₊ (con r± = M ± √(M²−a²)) toda curva
causal dirigida al futuro tiene dr/dτ < 0 estrictamente (g^{μν}∂_μr∂_νr =
Δ/Σ < 0). Si la esfera de excisión está CONTENIDA en la región atrapada,
ninguna curva causal puede salir del agujero y re-entrar al dominio:
si entrara por un punto con r = r_in ≥ min esfera = √(R_exc²−a²) y saliera
por otro con r = r_out ≥ √(R_exc²−a²) = min, la monotonía estricta de r
exigiría r_out < r_in con ambos sobre la esfera, imposible en el punto de
mínimo. Por tanto la condición suficiente y verificable en forma cerrada es

    √(r₋² + a²)  <  R_exc  <  r₊                              (★)

- La cota superior garantiza que el polo (r = R_exc) no asome fuera del
  horizonte de eventos.
- La cota inferior garantiza que el ecuador (r = √(R_exc²−a²)) no caiga
  dentro del horizonte de Cauchy r₋, donde la región deja de ser atrapada
  y las características pueden volver a moverse hacia r creciente
  (do-nothing dejaría de ser consistente).
- La singularidad anular (r=0: ρ=a, z=0) queda automáticamente excisada
  porque √(r₋²+a²) > a.

Para Schwarzschild (a=0): ventana (0, 2M) — cualquier esfera interior.

## La ventana se cierra a spin alto

(★) es no vacía sólo si r₋² + a² < r₊², es decir a² < 4M√(M²−a²).
Con x = (a/M)²: x² + 16x − 16 = 0 ⟹ x = √320/2 − 8 ≈ 0.9443:

    a/M < 0.9718...   (excisión esférica cartesiana admisible)

Por encima de ese spin NINGUNA esfera cartesiana cabe en la región
atrapada. La solución correcta es excisar una superficie r = r₀ constante,
que en cartesianas es el esferoide oblato

    (x² + y²)/(r₀² + a²) + z²/r₀² = 1 ,

con ventana completa r₀ ∈ (r₋, r₊) para todo a < M (TODO Fase 1:
`mesh.excision_shape = "spheroid"` vía elipsoide de Gmsh).

## Valores de referencia

| a/M  | r₋/M   | r₊/M   | ventana R_exc/M (★) | nota |
|------|--------|--------|----------------------|------|
| 0.0  | 0      | 2.000  | (0, 2)               | |
| 0.5  | 0.134  | 1.866  | (0.518, 1.866)       | |
| 0.7  | 0.286  | 1.714  | (0.756, 1.714)       | |
| 0.9  | 0.564  | 1.436  | (1.062, 1.436)       | r_inner=1.0 usado históricamente queda FUERA (por debajo) |
| 0.95 | 0.688  | 1.312  | (1.174, 1.312)       | ventana estrecha |
| 0.97 | 0.757  | 1.243  | (1.236, 1.243)       | casi cerrada |

**Implicación práctica**: los runs históricos de Kerr a=0.9 con
r_inner = 1.0 tienen el casquete ecuatorial de la esfera de excisión
dentro del horizonte de Cauchy (r_ecuador = √(1−0.81) = 0.436 < r₋ =
0.564). El sesgo sistemático de +30–40% observado en las frecuencias
puede tener aquí una contribución; el estudio de convergencia debe
repetirse con r_inner dentro de (★). `validate_config` impone (★) desde
la Fase 1.
