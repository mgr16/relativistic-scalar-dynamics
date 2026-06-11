# PSYOP - Simulación de Campos Escalares en Relatividad General

## Descripción

PSYOP es un simulador de campos escalares evolucionando en fondos de agujeros negros usando elementos finitos. El proyecto ha sido completamente renovado con una arquitectura modular avanzada y métodos numéricos de alto orden.

##  Mejoras Implementadas (Versión 2.1)

### **Mejora 1: Formulación de Primer Orden con SSP-RK3**
- **Sistema de primer orden**: (φ, Π) con Π como momento 3+1, \(\Pi=(\partial_t\phi-\beta^i\partial_i\phi)/\alpha\)
- **Integración temporal SSP-RK3**: Strong Stability Preserving Runge-Kutta de orden 3
- **CFL adaptativo**: Paso de tiempo automático basado en el tamaño de malla
- **Solver de matriz de masa**: Inversión eficiente usando PETSc/HYPRE

### **Mejora 2: Condiciones de Frontera Sommerfeld (característica)**
- **Condición física característica**: sustituye el flujo conormal natural usando `Π = −√(γ^{nn}) ∂_n φ` (en espacio plano se reduce al término absorbente clásico `−∫ Π v ds`)
- **Implementación débil consistente**: término de borde con pesos métricos `α √γ` sobre facetas `tag=2` vía `ds(tag)`
- **Absorción de ondas**: reduce reflexiones sin Robin ad-hoc

### **Soporte de agujeros negros con excisión**
- **Excisión del interior**: `mesh.r_inner > 0` genera una cáscara esférica con borde interior etiquetado (`tag=3`)
- **Borde interior "do-nothing"**: válido cuando las características salen del dominio (foliaciones horizon-penetrating)
- **Curvatura extrínseca de Kerr-Schild**: `K = (1/(α√γ)) ∂_i(√γ β^i)` evaluada simbólicamente (ya no se asume K=0)
- Las métricas `schwarzschild`/`kerr` **requieren** `mesh.r_inner > 0` (sugerido: `~M/2` para Schwarzschild isotrópico, `~M` para Kerr-Schild)
- **Malla graduada**: `mesh.lc_inner < mesh.lc` refina radialmente cerca del horizonte

### **Diagnósticos y absorción avanzados**
- **Momento inicial consistente**: `initial_conditions.direction = "ingoing"|"outgoing"` (pulso esférico puro, Π = ±(∂_rφ + φ/r)); `"static"` (Π=0) divide el pulso en mitades
- **Capa esponja**: `solver.sponge {enabled, width, strength}` amortigua las colas dispersivas de campos masivos que la BC característica no absorbe. *Tuning*: la anchura debe ser comparable a la longitud de onda a absorber — una esponja angosta y fuerte refleja los modos lentos en vez de absorberlos
- **Disipación de 4.º orden**: `solver.ko_order = 4` (filtro biarmónico normalizado por λmax: misma estabilidad que el de 2.º orden pero casi no toca los modos suaves)
- **Extracción multipolar**: `analysis.extraction {enabled, radius, lmax}` proyecta φ sobre armónicos esféricos reales en una esfera de extracción → `series/multipoles.csv`
- **Balance de energía**: `series/balance.csv` registra `E(t) + ∫F dt − E(0)` (residuo converge ~h²)

> **Supuesto físico**: el campo escalar es un campo de prueba sobre fondo fijo
> (aproximación de Cowling); no hay backreaction sobre la métrica.

### **Mejora 3: Arquitectura Modular Avanzada**
- **Implementación DOLFINx-only**: migración completa desde soporte dual
- **Generación de mallas**: Gmsh con etiquetas de frontera automáticas
- **Potenciales generalizados**: Higgs, cuadrático, sombrero mexicano
- **Condiciones iniciales flexibles**: Gaussian bump, ondas planas, etc.

## Estructura del Proyecto

```
PSYOP/
├── main.py                    # Script principal
├── src/psyop/                 # Paquete principal
│   ├── analysis/              # Análisis (QNM, espectros)
│   ├── backends/              # Abstracciones numéricas DOLFINx
│   ├── mesh/                  # Generación de mallas (Gmsh, cajas)
│   ├── physics/               # Métricas, potenciales, condiciones iniciales
│   ├── solvers/               # Solvers numéricos
│   └── utils/                 # Utilidades (CFL, análisis de malla)
├── docs/reviews/              # Documentación consolidada (PR #7)
├── scripts/                   # Scripts auxiliares
├── tests/                     # Pruebas
└── README.md                  # Esta documentación
```

## Documentación consolidada (PR #7)

Los documentos de revisión y mejoras se movieron a `docs/reviews/` para mantener la raíz del proyecto más limpia:

- [IMPLEMENTATION_SUMMARY](docs/reviews/IMPLEMENTATION_SUMMARY.md)
- [IMPROVEMENT_ROADMAP](docs/reviews/IMPROVEMENT_ROADMAP.md)
- [PROJECT_REVIEW](docs/reviews/PROJECT_REVIEW.md)
- [QUICK_START_IMPROVEMENTS](docs/reviews/QUICK_START_IMPROVEMENTS.md)
- [REVIEW_INDEX](docs/reviews/REVIEW_INDEX.md)
- [REVIEW_SUMMARY](docs/reviews/REVIEW_SUMMARY.md)
- [UPDATE_NOTES](docs/reviews/UPDATE_NOTES.md)

## Instalación

### Instalación automatizada (Opción A)
```bash
# Desde la raíz del proyecto
./scripts/setup_conda_env.sh --yes

# Alternativas útiles
./scripts/setup_conda_env.sh --env-name psyop-dolfinx --python 3.10 --yes
./scripts/setup_conda_env.sh --install-dev --yes
```

El script crea (o reutiliza) un entorno conda, instala `fenics-dolfinx` (módulo `dolfinx`) y dependencias desde `conda-forge`, instala el paquete local en modo editable y valida imports críticos.

### Entorno recomendado (DOLFINx-only)
```bash
# Crear entorno conda
conda create -n psyop-dolfinx python=3.10
conda activate psyop-dolfinx

# Instalar DOLFINx
conda install -c conda-forge fenics-dolfinx

# Dependencias adicionales
conda install -c conda-forge gmsh numpy matplotlib scipy petsc4py
```

### Verificación de la instalación
```bash
# Probar imports, configuración y postproceso liviano
pytest -q

# Probar la CLI instalada
psyop --test
```

## Uso Rápido

### Simulación básica
```bash
psyop run --config config_example.json --output results
```

También quedan instalados los aliases compatibles:
```bash
psyop-run --config config_example.json --output results
psyop-postprocess --run results/run_YYYYmmdd_HHMMSS --qnm --method fft
```

### Postproceso QNM
```bash
psyop postprocess --run results/run_YYYYmmdd_HHMMSS --qnm --method fft --plots
```

### Visualización en vivo (`--live`)
Abre una ventana interactiva PyVista con un corte z=0 del campo φ que se
actualiza durante la evolución (barra de color fija calibrada con el estado
inicial y el tiempo t en pantalla). Requiere pyvista:

```bash
conda install -n psyop-dolfinx -c conda-forge pyvista
```

```bash
psyop run --config config_example.json --live                 # refresca cada output_every pasos
psyop run --config config_example.json --live --live-every 5  # refresco cada 5 pasos
```

Caveats:
- Pensado para demos y debugging, **no para producción**: el render frena el
  lazo de evolución.
- **Solo en serie**: con MPI > 1 rank se desactiva con un warning y la
  simulación continúa normalmente.
- Necesita sesión gráfica: en entornos headless, o si pyvista no está
  instalado, se loggea un warning y la corrida sigue sin ventana (sin `--live`
  el costo es cero).

### Configuración personalizada
Edita `config_example.json` o crea un JSON propio con las mismas claves:

```json
{
    "mesh": {
        "type": "gmsh",
        "R": 15.0,
        "lc": 1.0,
        "r_inner": 0.0
    },
    "metric": {
        "type": "flat",
        "M": 1.0
    },
    "solver": {
        "degree": 1,
        "cfl": 0.3,
        "potential_type": "higgs",
        "potential_params": {
            "m_squared": 1.0,
            "lambda_coupling": 0.1
        },
        "bc_type": "characteristic",
        "enable_sommerfeld": true
    },
    "evolution": {
        "t_end": 20.0,
        "output_every": 10
    }
}
```

Para fondos de agujero negro, usa `metric.type = "schwarzschild"` o `"kerr"` y
fija `mesh.r_inner > 0` (excisión); la validación lo exige y sugiere valores.

## Física y Métodos Numéricos

### Ecuaciones Fundamentales

**Sistema de primer orden:**
```
Π = (∂tφ - β·∇φ)/α
∂tφ = αΠ + β·∇φ
∂tΠ = αDᵢDⁱφ + DⁱαDᵢφ + β·∇Π + αKΠ - αV'(φ)
```

**Condición de salida Sommerfeld (característica):**
```
c_out = α − β·n
```
Se implementa como flujo saliente en el término de borde del RHS.

Ver derivación y convenciones completas en: `docs/math/3p1_scalar_field.md`.
Resumen de validación y reproducibilidad: `docs/validation/summary.md`.

**Potencial de Higgs:**
```
V(φ) = ½m²φ² + ¼λφ⁴
V'(φ) = m²φ + λφ³
```

### Esquema de Integración SSP-RK3

```
u⁽¹⁾ = uⁿ + dt · L(uⁿ)
u⁽²⁾ = ¾uⁿ + ¼u⁽¹⁾ + ¼dt · L(u⁽¹⁾)
uⁿ⁺¹ = ⅓uⁿ + ⅔u⁽²⁾ + ⅔dt · L(u⁽²⁾)
```

### Condición CFL Adaptativa

```
dt = CFL_factor × h_min / (c_max × degree²)
```
donde `h_min` es el tamaño mínimo de celda, `c_max` la velocidad característica
máxima del fondo y el factor `degree²` es el escalado estándar para elementos
de orden alto (sin efecto para `degree = 1`).

##  Características Avanzadas

### Generación de Mallas
- **Gmsh**: Mallas esféricas con etiquetas de frontera automáticas
- **Fallback**: Mallas cúbicas básicas si Gmsh no está disponible
- **Etiquetas**: Frontera externa marcada con `tag=2` para condiciones Sommerfeld

### Análisis de Modos Quasi-Normales
- **Muestreo temporal**: Registro del campo en puntos específicos
- **FFT**: Análisis espectral para identificar frecuencias características
- **Visualización**: Gráficos automáticos del espectro de frecuencias

### Compatibilidad
- **Framework numérico**: DOLFINx
- **Paralelización**: MPI + PETSc
- **Salida**: XDMF para postprocesado

## 🔧 Desarrollo y Extensiones

### Añadir un nuevo potencial
```python
# En potential.py
class CustomPotential:
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def evaluate(self, phi):
        return self.param1 * phi**6  # Ejemplo
    
    def derivative(self, phi):
        return 6 * self.param1 * phi**5
```

### Añadir condiciones iniciales
```python
# En initial_conditions.py
class CustomInitialCondition:
    def __init__(self, mesh, V, **params):
        # Implementar lógica personalizada
        pass
```

### Modificar el solver
El solver principal está en `src/psyop/solvers/first_order.py`. Métodos clave:
- `ssp_rk3_step()`: Integración temporal
- `_sommerfeld_boundary_term()`: Condiciones de frontera
- `_assemble_rhs_and_solve_du()`: Evaluación/solve del lado derecho

## Resultados y Validación

### Salidas del programa
- **Campos finales**: φ y Π guardados en formato VTK/XDMF
- **Series temporales**: Evolución del campo en puntos específicos
- **Espectro QNM**: Análisis de frecuencias características
- **Métricas de convergencia**: Normas y estadísticas

### Archivos generados
```
results/run_YYYYmmdd_HHMMSS/
├── config.json
├── manifest.json
├── fields/
│   └── phi_evolution.xdmf
├── series/
│   ├── time_series.csv
│   ├── energy.csv
│   ├── flux.csv
│   ├── qnm_spectrum.csv / qnm_peak.json
│   └── qnm_modes.json (prony)
└── plots/
    └── qnm_spectrum.png   # opcional (postprocess --plots)
```

##  Rendimiento

### Optimizaciones implementadas
- **Matriz de masa precalculada**: Factorización reutilizada
- **Solver PETSc**: Algoritmos paralelos eficientes
- **CFL adaptativo**: Pasos de tiempo óptimos automáticamente
- **Evaluación vectorizada**: Potenciales evaluados en arrays NumPy

### Benchmarks típicos
- **Mesh 10³ elementos**: ~1-5 segundos por unidad de tiempo físico
- **Mesh 20³ elementos**: ~10-30 segundos por unidad de tiempo físico
- **Escalabilidad**: Excelente con número de cores (PETSc paralelo)

##  Solución de Problemas

### Error común: "Import could not be resolved"
**Causa**: DOLFINx no instalado
**Solución**: 
```bash
conda install -c conda-forge fenics-dolfinx
```

### Error: "Gmsh not available"
**Causa**: Gmsh no instalado
**Solución**: El programa usa mallas de fallback automáticamente. Para instalar Gmsh:
```bash
conda install -c conda-forge gmsh
```

### Error de JIT en macOS: `ld: -lto_library library filename must be 'libLTO.dylib'`
**Causa**: conflicto entre el clang de conda-forge y el linker de Xcode al compilar las formas (FFCx JIT).
**Solución**: usar el compilador del sistema para el JIT:
```bash
export CC=/usr/bin/clang
```
(agregalo a tu activación del entorno o al perfil del shell).

### Error de convergencia en el solver
**Causa**: Paso de tiempo demasiado grande o malla muy gruesa
**Solución**: Reducir `solver.cfl` o reducir `mesh.lc`

### Memoria insuficiente
**Causa**: Malla demasiado fina
**Solución**: Aumentar `mesh.lc`

## Referencias Técnicas

### Métodos numéricos
- **SSP-RK3**: Gottlieb et al. (2001) "Strong Stability-Preserving High-Order Time Discretization Methods"
- **Elementos Finitos**: Brenner & Scott "The Mathematical Theory of Finite Element Methods"
- **Condiciones Sommerfeld**: Engquist & Majda (1977) "Absorbing boundary conditions for numerical simulation of waves"

### Física
- **Klein-Gordon**: Relativistic quantum mechanics and field theory textbooks
- **Modos Quasi-Normales**: Berti et al. (2009) "Eigenvalues and eigenfunctions of spin-weighted spheroidal harmonics"
- **Agujeros Negros**: Wald "General Relativity", Misner-Thorne-Wheeler "Gravitation"


---

**Versión**: 2.1 (incluye cambios del PR #7)  
**Compatibilidad**: DOLFINx 0.7+  
**Python**: 3.9+  
**Licencia**: Proyecto de investigación académica
