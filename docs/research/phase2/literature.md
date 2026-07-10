# Fase 2 — Pase de literatura (capítulo 1): el enunciado exacto de Fournodavlos–Sbierski

**Fecha:** 2026-07-09. **Estado:** capítulo cerrado.
**Método:** verificación contra el texto completo del paper (arXiv/ar5iv) y las
páginas de las revistas; los enunciados de abajo usan la notación del propio
paper. Ningún resultado de este capítulo depende de corridas nuestras.

## 1. Qué había que verificar

H2 se enunció desde el principio como *"la dominación cinética cerca de la
singularidad de Schwarzschild borra la estructura de vacío tipo Higgs; se
recuperan asintóticas logarítmicas lineales (à la Fournodavlos–Sbierski)"*,
con la salvedad explícita de que el enunciado exacto quedaba por verificar.
Este capítulo cierra esa salvedad: qué dice exactamente el teorema, bajo qué
hipótesis, dónde vive el coeficiente logarítmico, y qué implica todo eso para
el observable a(t) del programa.

## 2. El resultado exacto

**Paper ancla:** G. Fournodavlos, J. Sbierski, *"Generic Blow-Up Results for
the Wave Equation in the Interior of a Schwarzschild Black Hole"*,
Arch. Ration. Mech. Anal. (2019), DOI
[10.1007/s00205-019-01434-0](https://link.springer.com/article/10.1007/s00205-019-01434-0),
[arXiv:1804.01941](https://arxiv.org/abs/1804.01941). Soluciones suaves de
□_g ψ = 0 sobre Schwarzschild **fijo** (interior), coordenadas de
Schwarzschild (t, r, ω) — en el interior r es temporal y la singularidad
{r=0} es una hipersuperficie espacial ≅ ℝ_t × S².

- **Teorema 1.7 (expansión de primer orden).** Hipótesis: decaimiento a lo
  largo de ambos horizontes de eventos ℋ⁺∩{v≥1}: |ψ| ≲ v^{−q} y
  |∂_t^{(i)} Ω^{(j)} ψ| ≲ v^{−(q+δ)} para 1 ≤ i+j ≤ 6, con q > 0, δ ≥ 0.
  Conclusión: existen A, B ∈ C^∞(ℝ×S²) y un resto P tales que, en
  {0 < r < r₀} (r₀ cerca de 2m):

  ```
  ψ(t, r, ω) = A(t, ω) · log r + B(t, ω) + P(t, r, ω)
  |A| ≲ (|t|+1)^{−(q+δ)},  |B| ≲ (|t|+1)^{−q},  |P| ≲ r·|log r|·(|t|+1)^{−q}
  ```

  con definiciones intrínsecas **A := L²-lim_{r→0} ψ/log r** y
  **B := L²-lim_{r→0} (ψ − A log r)**. El coeficiente A es una función sobre
  la hipersuperficie singular: depende de t **y del ángulo ω**.

- **Teorema 1.18 (expansión completa).** ψ = Σₙ ζₙ(t,ω) rⁿ log r +
  Σₙ ηₙ(t,ω) rⁿ + R_N, con ζ₀ = A, η₀ = B y los coeficientes superiores
  determinados **recursivamente** por (A, B) y sus derivadas (∂_t, Δ_{S²}).
  Es decir: el par (A, B) codifica toda la asintótica.

- **Teorema 1.21 (scattering hacia la singularidad).** Isomorfismo entre
  soluciones en {0 < r ≤ r₀} (con decaimiento prescrito) y pares (A, B)
  suaves con el decaimiento correspondiente.

- **Teorema 1.10 (blow-up en los extremos).** Si además los modos ℓ≥1 decaen
  más rápido (|∂_t^{(i)} Ω^{(j)} ψ_{ℓ≥1}| ≲ v^{−(q+1+ε)}) y el modo esférico
  tiene cota inferior |∂_t ψ₀| ≳ v^{−(q+1)} en el horizonte, entonces ψ
  explota puntual y logarítmicamente en un entorno de los extremos |t| = ∞
  de {r=0}, con |A₀| ∼ (|t|+1)^{−(q+1)}.

- **Teorema 1.14 (insumo externo) + Corolario 1.16 (genericidad tardía).**
  El insumo es Angelopoulos–Aretakis–Gajic
  ([arXiv:1612.01566](https://arxiv.org/abs/1612.01566), Adv. Math. 323
  (2018) 529–621): datos suaves **genéricos** en la superficie de Cauchy Σ
  (con cierta cota de decaimiento hacia i⁰) cumplen las hipótesis del
  Teorema 1.10 con ε = 2 y **q = 2 ó 3** según la cota. Conclusión
  (Cor. 1.16): genéricamente hay blow-up logarítmico cerca de los extremos,
  con tasa tardía |A₀| ∼ |t|^{−3} ó |t|^{−4}.

- **Teorema 1.17 (genericidad global).** Existe un conjunto **abierto** de
  datos de Cauchy en Σ (topología inducida por una energía, §4 del paper)
  cuyas soluciones explotan logarítmicamente en **toda** la hipersuperficie
  {r=0}, no solo en los extremos.

Los autores señalan la analogía de A con la "velocidad asintótica" de las
dinámicas tipo Kasner/BKL — misma familia estructural que la reducción
cinética de H2.

## 3. Mapeo al programa RSD

- **Identificación del observable.** Nuestro a(t) *es* el A(t,ω) de F–S
  (proyectado al modo angular que excitamos). La definición
  A = L²-lim ψ/log r valida el estimador del piloto: pendiente de φ contra
  ln r, ventana interior. El piloto 1D de F0 ya vio exactamente esta
  estructura empíricamente (perfil logarítmico sobre 2 décadas, pendiente
  independiente de r, más un término acotado b(t)).
- **Rebanadas KS vs coordenada t de Schwarzschild.** Medimos sobre
  rebanadas de tiempo Kerr-Schild; la relación exacta es
  t_KS = t_Schw + 2M·ln|r/2M − 1|, de modo que **t_KS → t_Schw en r → 0** y
  el drift de t_Schw a lo largo de una ventana radial [0.1M, 0.5M] a t_KS
  fijo es ≈ 0.47M — pequeño frente a las escalas de variación de A y
  **cuantificable**: entra como sistemático conocido del diagnóstico
  interior (verificarlo numéricamente en el capítulo de diagnósticos).
- **H2 operacionalizada** (predicciones falsables, de más barata a más cara):
  1. **Perfil:** el campo interior (u − v para Higgs) sigue
     a·ln r + b + O(r ln r) en la zona cinética (r ≲ 0.5M). [Visto en 1D;
     confirmar en 3D.]
  2. **Borrado (el corazón de H2):** a(t,ω) del run no lineal con u∞ = v
     coincide con el a(t,ω) del run **lineal** con el mismo dato inicial,
     con desviación que cae hacia r → 0 acorde a la supresión del potencial
     (F0 midió R_pot ~ r^2.78). El A/B lineal-vs-Higgs con dato idéntico es
     el discriminador primario (la estructura del piloto F0 ya era esa).
  3. **Estructura de la jerarquía:** dónde entra la primera corrección del
     potencial en la expansión ζₙ/ηₙ del Teorema 1.18 — derivación corta
     pendiente; comparar el exponente resultante con el R_pot ~ r^2.78
     medido en F0. (Sub-pregunta abierta, teórica, barata.)
  4. **Tasas tardías (estiramiento, solo modo ℓ=0):** |a₀| ~ t^{−(q+1)} con
     q ∈ {2,3}. Requiere una corrida con componente esférica y tiempos
     largos; el diseño espectroscópico actual (l=2) no la cubre. Opcional.
- **El valor agregado 3D:** F–S demuestra que A depende del ángulo. El
  oráculo 1D solo ve a(t) por modo; el 3D puede medir **a(t, ω)** (su
  descomposición en armónicos) — ese es el diagnóstico interior a diseñar.
- **Cowling es el marco del teorema, no una debilidad del mapeo:** F–S es
  exactamente campo de prueba sobre Schwarzschild fijo. Lo que RSD agrega
  sobre F–S es la **no-linealidad del potencial** (Higgs/sombrero), no el
  backreaction. El límite Cowling sigue declarándose como hasta ahora.

## 4. Lo que F–S NO cubre (límites honestos del ancla)

- **Lineal y sin masa.** Nada de potenciales: la pregunta de H2 (si la
  estructura de vacío sobrevive) está genuinamente abierta y es nuestra.
- **Extensión maximal (dos horizontes).** Las hipótesis se imponen en ambos
  horizontes; nuestro dominio solo se alimenta del exterior derecho con un
  pulso entrante (dato efectivamente trivial en el otro extremo). Caveat de
  mapeo menor; anotarlo en el paper.
- **Genericidad por partes:** la versión "toda la singularidad" (Thm 1.17)
  es para un abierto de datos en topología de energía; la versión "datos
  suaves genéricos" (Cor 1.16) solo cubre los extremos |t| → ∞. Al comparar
  con corridas, citar el teorema que corresponda a la ventana temporal
  medida.

## 5. Consecuencias de diseño para el resto de F2

1. El siguiente capítulo natural es el **diagnóstico interior 3D**:
   estimador de a(t,ω) (fit de φ vs ln r por rayos/armónicos sobre la
   ventana [r_inner, ~0.5M]) + comparador lineal/Higgs con dato idéntico.
2. **Ventana radial:** con r_inner = 0.25M la palanca del fit es < 1 década.
   Las sondas F0 fueron estables hasta r_inner = 0.1M; cuantificar con el
   oráculo 1D el sesgo de truncamiento del fit de a sobre [0.25, 0.5] vs
   [0.1, 0.5] y decidir r_inner de producción con ese número.
3. Los términos siguientes de la expansión (ζ₁, η₁) son medibles en 1D como
   test de consistencia fuerte del estimador antes de gastarlo en 3D.
4. Para la predicción de tasas tardías haría falta una corrida ℓ=0 larga
   (recordar que el QNM l=0 está mal condicionado para espectroscopía, pero
   aquí el objetivo sería la cola interior, no el ring).

## 6. Referencias ancla verificadas

| Ref | Para qué | Dónde |
|---|---|---|
| Fournodavlos–Sbierski, ARMA 2019 | Enunciado exacto de H2 (Thms 1.7/1.10/1.14/1.17/1.18/1.21) | [arXiv:1804.01941](https://arxiv.org/abs/1804.01941) · [DOI](https://link.springer.com/article/10.1007/s00205-019-01434-0) |
| Angelopoulos–Aretakis–Gajic, Adv. Math. 323 (2018) 529–621 | Insumo Price-law del Thm 1.14 (q = 2 ó 3, ε = 2) | [arXiv:1612.01566](https://arxiv.org/abs/1612.01566) |
| Burda–Gregory–Moss, PRL 115, 071303 (2015) | H1 (respaldo): decaimiento de vacío catalizado por BH | [arXiv:1501.04937](https://arxiv.org/abs/1501.04937) |
| Burda–Gregory–Moss, JHEP 08 (2015) 114 | H1: metaestabilidad con agujeros negros | [arXiv:1503.07331](https://arxiv.org/abs/1503.07331) · [DOI](https://link.springer.com/article/10.1007/JHEP08(2015)114) |
| *Black holes don't source fast Higgs vacuum decay*, JHEP 03 (2023) 039 | Contrapunto moderno a la línea H1 — leer a fondo solo si H1 se activa | [DOI](https://link.springer.com/article/10.1007/JHEP03(2023)039) |

(Extracción del texto completo de F–S vía ar5iv el 2026-07-09; el pase de
literatura amplio para related-work del paper queda para F3 — este capítulo
solo debía anclar H2.)

## 7. Decisión

- **H2 queda anclada** al Teorema 1.7 (expansión con A(t,ω)·log r),
  Teorema 1.17 / Corolario 1.16 (genericidad) y Teorema 1.18 (jerarquía
  completa). La frase "el enunciado exacto se verificará en F2" se retira
  del plan: verificado.
- La hipótesis tal como está formulada **sobrevive sin enmiendas** — el
  observable a(t) coincide con el objeto matemático correcto (A), y el
  contraste lineal-vs-Higgs con dato idéntico es exactamente el experimento
  que el marco de F–S sugiere.
- **GO al siguiente capítulo de F2:** diagnóstico interior 3D (a(t,ω) +
  comparador lineal/Higgs), con la calibración de ventana radial vía
  oráculo 1D como primer paso.
