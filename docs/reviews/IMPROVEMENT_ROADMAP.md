# PSYOP - Roadmap de Mejoras / Improvement Roadmap

**Based on objective review - Grade: B+ (83/100)**

---

## 🎯 Sprint 1 (Semana 1-2): Tests y Validación Crítica

### Objetivos
- Alcanzar 60%+ cobertura de tests
- Establecer CI/CD básico
- Validar corrección numérica

### Tareas Específicas

#### ✅ Tarea 1.1: Tests de Conservación de Energía
**Prioridad:** CRÍTICA  
**Esfuerzo:** 2 días  
**Archivo:** `tests/test_energy_conservation.py`

```python
import pytest
import numpy as np
from psyop.solvers.first_order import FirstOrderKGSolver
from psyop.mesh.gmsh import build_ball_mesh

@pytest.mark.parametrize("cfl", [0.1, 0.3, 0.5])
def test_energy_conservation_varying_cfl(cfl):
    """Verify energy conservation within numerical tolerance."""
    mesh, _, facet_tags = build_ball_mesh(R=5.0, lc=2.0)
    solver = FirstOrderKGSolver(mesh, cfl_factor=cfl)
    
    # Initialize with Gaussian bump
    # ... setup ...
    
    initial_energy = solver.energy()
    
    # Evolve for t_final
    for _ in range(100):
        solver.ssp_rk3_step(dt=0.01)
    
    final_energy = solver.energy()
    
    # Allow 1% energy drift (adjust based on expected numerical error)
    rel_error = abs(final_energy - initial_energy) / initial_energy
    assert rel_error < 0.01, f"Energy drift: {rel_error:.4%} for CFL={cfl}"
```

**Criterio de éxito:** Tests pasan para CFL=[0.1, 0.3, 0.5]

---

#### ✅ Tarea 1.2: Tests de Potenciales
**Prioridad:** ALTA  
**Esfuerzo:** 1 día  
**Archivo:** `tests/test_potentials.py`

```python
import pytest
import numpy as np
from psyop.physics.potential import HiggsPotential, QuadraticPotential

def test_higgs_potential_derivative():
    """Verify analytical vs numerical derivative."""
    pot = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
    
    phi_test = np.linspace(-5, 5, 100)
    
    # Analytical derivative
    V_prime_analytical = pot.derivative(phi_test)
    
    # Numerical derivative (finite differences)
    epsilon = 1e-6
    V_prime_numerical = (pot.evaluate(phi_test + epsilon) - 
                         pot.evaluate(phi_test - epsilon)) / (2 * epsilon)
    
    np.testing.assert_allclose(V_prime_analytical, V_prime_numerical, 
                               rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("pot_type,params", [
    ("higgs", {"m_squared": 1.0, "lambda_coupling": 0.1}),
    ("quadratic", {"m_squared": 2.0}),
    ("mexican_hat", {"m_squared": -1.0, "lambda_coupling": 0.5}),
])
def test_potential_factory(pot_type, params):
    """Verify all potential types are constructible."""
    from psyop.physics.potential import get_potential
    pot = get_potential(pot_type, **params)
    
    # Test evaluation doesn't crash
    phi = np.array([0.1, 0.5, 1.0])
    V = pot.evaluate(phi)
    V_prime = pot.derivative(phi)
    
    assert V.shape == phi.shape
    assert V_prime.shape == phi.shape
```

**Criterio de éxito:** Derivadas analíticas coinciden con numéricas (error < 1e-5)

---

#### ✅ Tarea 1.3: Tests de Sommerfeld BC
**Prioridad:** ALTA  
**Esfuerzo:** 2 días  
**Archivo:** `tests/test_sommerfeld_bc.py`

```python
import pytest
import numpy as np

def test_sommerfeld_reflection_coefficient():
    """Measure reflection at boundary with outgoing wave."""
    # Setup: Gaussian pulse traveling outward
    mesh, _, facet_tags = build_ball_mesh(R=15.0, lc=1.0)
    solver = FirstOrderKGSolver(mesh, cfl_factor=0.3)
    solver.enable_sommerfeld(facet_tags, outer_tag=2)
    
    # Initialize wave packet moving outward
    # v0 > 0 means outward velocity
    ic = GaussianBump(mesh, A=0.1, r0=8.0, w=2.0, v0=1.0)
    solver.set_initial_conditions(ic.get_function())
    
    # Measure amplitude before boundary
    phi_before = solver.measure_at_radius(r=12.0)
    
    # Evolve until wave reaches boundary
    solver.evolve(t_final=5.0, dt=0.01)
    
    # Measure reflected amplitude
    phi_after = solver.measure_at_radius(r=8.0)
    
    # Reflection coefficient: should be < 5%
    reflection = abs(phi_after) / abs(phi_before)
    assert reflection < 0.05, f"Reflection too high: {reflection:.2%}"
```

**Criterio de éxito:** Coeficiente de reflexión < 5%

---

#### ✅ Tarea 1.4: CI/CD con GitHub Actions
**Prioridad:** ALTA  
**Esfuerzo:** 1 día  
**Archivo:** `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Micromamba
      uses: mamba-org/setup-micromamba@v1
      with:
        environment-file: environment.yml
        cache-environment: true
    
    - name: Install dependencies
      run: |
        micromamba install -c conda-forge pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=psyop --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

**Criterio de éxito:** Tests ejecutándose automáticamente en cada push

---

## 🔧 Sprint 2 (Semana 3-4): Robustez y Error Handling

### Objetivos
- Eliminar bare except clauses
- Añadir validación de entradas
- Implementar logging module

### Tareas Específicas

#### ✅ Tarea 2.1: Reemplazar Bare Except
**Prioridad:** CRÍTICA  
**Esfuerzo:** 1 día  
**Archivos:** `psyop/solvers/first_order.py`, otros

**Buscar y reemplazar:**
```bash
# Encontrar todos los bare except
grep -rn "except:" psyop/

# Patrón a reemplazar:
except:                              # ❌ MALO
    print("Fallback")

# Por:
except (RuntimeError, AttributeError) as e:  # ✅ BUENO
    logger.warning(f"Fallback due to: {e}")
```

**Líneas específicas a corregir:**
- `first_order.py:390` - Mass matrix assembly fallback
- `first_order.py:498` - Sommerfeld BC setup
- `initial_conditions.py:*` - Import fallbacks
- `main.py:*` - Framework detection

**Criterio de éxito:** 0 bare except clauses en psyop/

---

#### ✅ Tarea 2.2: Input Validation
**Prioridad:** ALTA  
**Esfuerzo:** 1-2 días  
**Archivo:** `psyop/solvers/first_order.py`

```python
class FirstOrderKGSolver:
    def __init__(self, mesh, degree=1, potential_type="higgs", 
                 potential_params=None, cfl_factor=0.5, domain_radius=10.0, **kwargs):
        # ADD VALIDATION
        if not 0 < cfl_factor <= 1:
            raise ValueError(
                f"CFL factor must be in (0, 1], got {cfl_factor}. "
                f"For SSP-RK3, typical range is [0.1, 0.5]."
            )
        
        if domain_radius <= 0:
            raise ValueError(f"Domain radius must be positive, got {domain_radius}")
        
        if degree < 1 or degree > 5:
            raise ValueError(
                f"FEM degree must be in [1, 5], got {degree}. "
                f"Higher degrees may be unstable without h-refinement."
            )
        
        if potential_type not in ["higgs", "quadratic", "mexican_hat"]:
            raise ValueError(
                f"Unknown potential type: {potential_type}. "
                f"Valid options: higgs, quadratic, mexican_hat"
            )
        
        # Validate mesh
        if mesh.topology().dim() != 3:
            raise ValueError(f"Solver requires 3D mesh, got {mesh.topology().dim()}D")
        
        # Continue with initialization...
```

**Test de validación:**
```python
def test_input_validation():
    mesh, _, _ = build_ball_mesh(R=5.0, lc=2.0)
    
    # Should raise for invalid CFL
    with pytest.raises(ValueError, match="CFL factor"):
        FirstOrderKGSolver(mesh, cfl_factor=1.5)
    
    # Should raise for negative radius
    with pytest.raises(ValueError, match="Domain radius"):
        FirstOrderKGSolver(mesh, domain_radius=-1.0)
```

**Criterio de éxito:** Tests de validación pasan, errores claros para inputs inválidos

---

#### ✅ Tarea 2.3: Logging Module
**Prioridad:** ALTA  
**Esfuerzo:** 1 día  
**Archivo nuevo:** `psyop/utils/logger.py`

```python
"""
Centralized logging configuration for PSYOP.
"""
import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "psyop",
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Configure logging for PSYOP.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Convenience function
def get_logger(name: str = "psyop") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)
```

**Uso en código:**
```python
# En first_order.py
from psyop.utils.logger import get_logger
logger = get_logger(__name__)

# Reemplazar print() por logger
logger.info("FirstOrderKGSolver initialized")
logger.warning("Gmsh not available, using fallback mesh")
logger.error(f"Failed to assemble mass matrix: {e}")
logger.debug(f"CFL timestep: dt={dt:.6e}")
```

**Criterio de éxito:** 0 print() statements en psyop/ (excepto debugging temporal)

---

## 📊 Sprint 3 (Semana 5-6): Type Hints y Documentación

### Objetivos
- Añadir type hints a componentes críticos
- Configurar mypy para type checking
- Actualizar docstrings con tipos

#### ✅ Tarea 3.1: Type Hints en Solver
**Prioridad:** MEDIA  
**Esfuerzo:** 2 días  
**Archivo:** `psyop/solvers/first_order.py`

```python
from typing import Tuple, Optional, Dict, Any
import numpy.typing as npt

# Conditional imports with types
if HAS_DOLFINX:
    from dolfinx.fem import Function as DolfinxFunction
    FEMFunction = DolfinxFunction
else:
    from fenics import Function as FenicsFunction
    FEMFunction = FenicsFunction

class FirstOrderKGSolver:
    def __init__(
        self,
        mesh: Any,  # Union of fenics.Mesh or dolfinx.mesh.Mesh
        degree: int = 1,
        potential_type: str = "higgs",
        potential_params: Optional[Dict[str, float]] = None,
        cfl_factor: float = 0.5,
        domain_radius: float = 10.0,
        **kwargs: Any
    ) -> None:
        """Initialize solver with validated parameters."""
        ...
    
    def ssp_rk3_step(self, dt: float) -> None:
        """Advance solution by time dt using SSP-RK3."""
        ...
    
    def get_fields(self) -> Tuple[FEMFunction, FEMFunction]:
        """Return (phi, Pi) field functions."""
        ...
    
    def energy(self) -> float:
        """Compute total energy."""
        ...
    
    def boundary_flux(self) -> float:
        """Compute flux through outer boundary."""
        ...
```

**Configurar mypy:**
```ini
# mypy.ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False  # Gradual typing
ignore_missing_imports = True  # For FEniCS/DOLFINx

[mypy-psyop.solvers.*]
disallow_untyped_defs = True  # Enforce in solvers

[mypy-psyop.physics.*]
disallow_untyped_defs = True  # Enforce in physics
```

**Criterio de éxito:** mypy pasa sin errores en psyop/solvers/ y psyop/physics/

---

## 🚀 Sprint 4 (Semana 7-8): Performance y Optimización

### Objetivos
- Añadir benchmarks
- Perfilar hotspots
- Documentar rendimiento esperado

#### ✅ Tarea 4.1: Benchmarks
**Prioridad:** MEDIA  
**Esfuerzo:** 2 días  
**Archivo:** `benchmarks/benchmark_solver.py`

```python
import time
import numpy as np
from psyop.solvers.first_order import FirstOrderKGSolver
from psyop.mesh.gmsh import build_ball_mesh

def benchmark_mesh_size(mesh_sizes=[5.0, 10.0, 15.0], lc_values=[2.0, 1.5, 1.0]):
    """Benchmark solver performance vs mesh size."""
    results = []
    
    for R in mesh_sizes:
        for lc in lc_values:
            mesh, _, facet_tags = build_ball_mesh(R=R, lc=lc)
            n_cells = mesh.num_cells()
            
            solver = FirstOrderKGSolver(mesh, cfl_factor=0.3)
            
            # Warm up
            solver.ssp_rk3_step(dt=0.01)
            
            # Benchmark 100 steps
            start = time.perf_counter()
            for _ in range(100):
                solver.ssp_rk3_step(dt=0.01)
            elapsed = time.perf_counter() - start
            
            time_per_step = elapsed / 100
            
            results.append({
                "R": R,
                "lc": lc,
                "n_cells": n_cells,
                "time_per_step": time_per_step,
                "cells_per_second": n_cells / time_per_step
            })
    
    return results

if __name__ == "__main__":
    results = benchmark_mesh_size()
    
    print("Benchmark Results:")
    print(f"{'R':>6} {'lc':>6} {'Cells':>10} {'Time/step':>12} {'Cells/sec':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['R']:>6.1f} {r['lc']:>6.1f} {r['n_cells']:>10} "
              f"{r['time_per_step']:>12.6f} {r['cells_per_second']:>12.0f}")
```

**Documentar resultados en README:**
```markdown
## Performance Benchmarks

| Mesh Size | Resolution | Cells | Time/step | Throughput |
|-----------|------------|-------|-----------|------------|
| R=5.0     | lc=2.0     | ~500  | 0.05s     | 10k cells/s |
| R=10.0    | lc=1.5     | ~2k   | 0.15s     | 13k cells/s |
| R=15.0    | lc=1.0     | ~10k  | 1.2s      | 8k cells/s  |

*Benchmarks on: Intel i7-10700K, 16GB RAM, single core*
```

---

## 📈 Métricas de Éxito / Success Metrics

| Métrica | Actual | Meta Sprint 1-2 | Meta Sprint 3-4 |
|---------|--------|-----------------|-----------------|
| **Test coverage** | ~5% | 60% | 80% |
| **Bare except clauses** | 4+ | 0 | 0 |
| **Type hints coverage** | ~10% | 30% | 60% |
| **CI/CD** | ❌ | ✅ | ✅ |
| **Logging** | print() | logging module | logging module |
| **Input validation** | ❌ | ✅ | ✅ |
| **Documentation** | A (95%) | A+ (98%) | A+ (100%) |

---

## 🎯 Definición de "Listo" / Definition of Done

Un sprint está completo cuando:

1. ✅ Todos los tests pasan (`pytest tests/ -v`)
2. ✅ Coverage >= objetivo del sprint
3. ✅ Code review aprobado (si aplicable)
4. ✅ Documentación actualizada (README, docstrings)
5. ✅ CI/CD pasa (GitHub Actions green)
6. ✅ No hay bare except clauses introducidos
7. ✅ mypy pasa sin errores (Sprint 3+)

---

## 🔄 Proceso de Implementación

### Workflow recomendado:

```bash
# 1. Crear branch por tarea
git checkout -b feature/test-energy-conservation

# 2. Implementar cambios
# ... editar archivos ...

# 3. Ejecutar tests localmente
pytest tests/test_energy_conservation.py -v

# 4. Verificar coverage
pytest tests/ --cov=psyop --cov-report=term-missing

# 5. Commit y push
git add tests/test_energy_conservation.py
git commit -m "Add energy conservation tests for multiple CFL values"
git push origin feature/test-energy-conservation

# 6. Crear PR en GitHub
# 7. Esperar CI/CD
# 8. Merge cuando todo esté verde
```

---

## 📞 Contacto y Soporte

**Preguntas sobre roadmap:**
- Abrir issue en GitHub con tag `[roadmap]`
- Incluir sprint y tarea específica

**Contribuciones:**
- Seguir workflow de branches
- Escribir tests para nuevas features
- Actualizar ROADMAP.md si es necesario

---

**Este roadmap es un documento vivo. Actualizar después de cada sprint con lecciones aprendidas.**

**Última actualización:** Febrero 2026  
**Próxima revisión:** Después de Sprint 1
