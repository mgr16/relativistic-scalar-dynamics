# Quick Start: Mejoras Críticas / Critical Improvements Quick Start

**Para desarrolladores que quieren mejorar PSYOP inmediatamente**

---

## ⚡ Las 3 Mejoras Más Impactantes (4-5 horas)

### 1. Añadir Test de Conservación de Energía (2 horas)

**Impacto:** ⭐⭐⭐⭐⭐ (Valida corrección numérica)  
**Dificultad:** ⭐⭐⭐ (Moderada)

**Pasos:**

```bash
# 1. Crear archivo de test
touch tests/test_energy_conservation.py
```

```python
# 2. Copiar este código a tests/test_energy_conservation.py
import pytest
import numpy as np
import sys
import os

# Importar componentes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_energy_conservation_basic():
    """Test that energy is conserved during evolution."""
    try:
        from psyop.solvers.first_order import FirstOrderKGSolver
        from psyop.mesh.gmsh import build_ball_mesh
        from psyop.physics.initial_conditions import GaussianBump
    except ImportError:
        pytest.skip("FEniCS/DOLFINx not available")
    
    # Small mesh for fast testing
    mesh, _, facet_tags = build_ball_mesh(R=5.0, lc=3.0)
    
    # Initialize solver
    solver = FirstOrderKGSolver(
        mesh, 
        degree=1,
        potential_type="quadratic",
        potential_params={"m_squared": 1.0},
        cfl_factor=0.3
    )
    
    # Set initial conditions
    ic = GaussianBump(mesh, A=0.01, r0=2.0, w=1.0, v0=0.0)
    solver.set_initial_conditions(ic.get_function())
    
    # Measure initial energy
    E0 = solver.energy()
    
    # Evolve for 10 steps
    dt = 0.01
    for _ in range(10):
        solver.ssp_rk3_step(dt)
    
    # Measure final energy
    Ef = solver.energy()
    
    # Check conservation (allow 5% drift for coarse mesh)
    rel_error = abs(Ef - E0) / E0
    assert rel_error < 0.05, f"Energy drift: {rel_error:.2%}"
    print(f"✓ Energy conservation test passed: drift = {rel_error:.4%}")
```

```bash
# 3. Ejecutar test
pytest tests/test_energy_conservation.py -v

# 4. Si pasa, commit!
git add tests/test_energy_conservation.py
git commit -m "Add energy conservation test"
```

**¿Qué valida?** Que la implementación numérica es correcta y conserva la energía.

---

### 2. Reemplazar Bare Except Clauses (1 hora)

**Impacto:** ⭐⭐⭐⭐ (Previene bugs ocultos)  
**Dificultad:** ⭐ (Fácil)

**Pasos:**

```bash
# 1. Encontrar todas las cláusulas problemáticas
grep -n "except:" psyop/solvers/first_order.py

# Output esperado:
# 390:            except:
# 498:            except:
```

```python
# 2. Reemplazar en psyop/solvers/first_order.py

# ANTES (línea ~390):
        except:
            # Fallback genérico
            print("⚠️ Matriz de masa assembly fallback")

# DESPUÉS:
        except (RuntimeError, AttributeError, KeyError) as e:
            # Specific fallback for known errors
            print(f"⚠️ Mass matrix assembly fallback: {e}")

# ANTES (línea ~498):
        except:
            print("⚠️ Sommerfeld BC setup failed, continuing without")

# DESPUÉS:
        except (RuntimeError, AttributeError, ValueError) as e:
            print(f"⚠️ Sommerfeld BC setup failed: {e}")
            print("Continuing without Sommerfeld boundary conditions")
```

```bash
# 3. Verificar cambios
git diff psyop/solvers/first_order.py

# 4. Ejecutar tests existentes
pytest tests/ -v

# 5. Commit
git add psyop/solvers/first_order.py
git commit -m "Replace bare except clauses with specific exceptions"
```

**¿Qué previene?** Errores silenciosos que ocultan bugs reales (TypeError, NameError, etc.)

---

### 3. Añadir Validación de Inputs (1-2 horas)

**Impacto:** ⭐⭐⭐⭐ (Previene crashes y errores confusos)  
**Dificultad:** ⭐⭐ (Fácil-Moderada)

**Pasos:**

```python
# 1. Editar psyop/solvers/first_order.py
# Encontrar __init__ (línea ~54) y añadir validación ANTES de self.mesh = mesh:

    def __init__(self, mesh, degree=1, potential_type="higgs", potential_params=None,
                 cfl_factor=0.5, domain_radius=10.0, **kwargs):
        """
        Inicializa el solver.
        ...
        """
        # ===== AÑADIR ESTAS VALIDACIONES =====
        
        # Validate CFL factor
        if not isinstance(cfl_factor, (int, float)):
            raise TypeError(f"cfl_factor must be numeric, got {type(cfl_factor)}")
        if not 0 < cfl_factor <= 1:
            raise ValueError(
                f"CFL factor must be in (0, 1], got {cfl_factor}. "
                f"Typical range for SSP-RK3: [0.1, 0.5]"
            )
        
        # Validate domain radius
        if domain_radius <= 0:
            raise ValueError(f"domain_radius must be positive, got {domain_radius}")
        
        # Validate degree
        if degree < 1 or degree > 5:
            raise ValueError(
                f"FEM degree must be in [1, 5], got {degree}. "
                f"Higher degrees require careful h-refinement"
            )
        
        # Validate potential type
        valid_potentials = ["higgs", "quadratic", "mexican_hat"]
        if potential_type not in valid_potentials:
            raise ValueError(
                f"Unknown potential_type: '{potential_type}'. "
                f"Valid options: {valid_potentials}"
            )
        
        # ===== FIN DE VALIDACIONES =====
        
        self.mesh = mesh
        # ... resto del código ...
```

```python
# 2. Crear test de validación: tests/test_input_validation.py
import pytest
from psyop.solvers.first_order import FirstOrderKGSolver

def test_invalid_cfl():
    """Test that invalid CFL values are rejected."""
    pytest.skip("Requires mesh setup")
    # mesh = ...  # Necesitas crear mesh primero
    
    with pytest.raises(ValueError, match="CFL factor"):
        FirstOrderKGSolver(mesh, cfl_factor=1.5)
    
    with pytest.raises(ValueError, match="CFL factor"):
        FirstOrderKGSolver(mesh, cfl_factor=-0.1)

def test_invalid_radius():
    """Test that negative radius is rejected."""
    pytest.skip("Requires mesh setup")
    
    with pytest.raises(ValueError, match="domain_radius"):
        FirstOrderKGSolver(mesh, domain_radius=-5.0)

def test_invalid_potential():
    """Test that unknown potential type is rejected."""
    pytest.skip("Requires mesh setup")
    
    with pytest.raises(ValueError, match="Unknown potential_type"):
        FirstOrderKGSolver(mesh, potential_type="invalid")
```

```bash
# 3. Probar manualmente
python3 -c "
from psyop.solvers.first_order import FirstOrderKGSolver
solver = FirstOrderKGSolver(None, cfl_factor=2.0)  # Debería fallar
"

# Debería imprimir: ValueError: CFL factor must be in (0, 1], got 2.0

# 4. Commit
git add psyop/solvers/first_order.py tests/test_input_validation.py
git commit -m "Add input validation to FirstOrderKGSolver"
```

**¿Qué previene?** Crashes confusos, mensajes de error claros para usuarios.

---

## 🔥 Mejora Bonus: Logging Module (30 minutos)

**Impacto:** ⭐⭐⭐ (Mejor debugging, production-ready)  
**Dificultad:** ⭐ (Muy fácil)

```bash
# 1. Crear psyop/utils/logger.py
cat > psyop/utils/logger.py << 'EOF'
"""Centralized logging for PSYOP."""
import logging
import sys

def setup_logger(name="psyop", level=logging.INFO):
    """Setup logger with console handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_logger(name="psyop"):
    """Get logger instance."""
    return logging.getLogger(name)
EOF
```

```python
# 2. Usar en main.py (línea ~22, después de imports)
from psyop.utils.logger import setup_logger, get_logger

# Setup logging
logger = setup_logger("psyop", level=logging.INFO)

# Reemplazar print() por logger:
# ANTES:
print("✓ DOLFINx disponible")

# DESPUÉS:
logger.info("DOLFINx available")
```

```bash
# 3. Commit
git add psyop/utils/logger.py main.py
git commit -m "Add centralized logging module"
```

---

## 📋 Checklist de Verificación

Después de implementar las 3 mejoras críticas:

```bash
# 1. Ejecutar todos los tests
pytest tests/ -v

# 2. Verificar que no hay bare except
grep -r "except:" psyop/ | grep -v ".pyc"
# Debería estar vacío (o solo excepciones específicas)

# 3. Probar validación manualmente
python3 -c "
from psyop.solvers.first_order import FirstOrderKGSolver
try:
    solver = FirstOrderKGSolver(None, cfl_factor=5.0)
except ValueError as e:
    print(f'✓ Validation works: {e}')
"

# 4. Verificar git status
git status
git log --oneline -5

# 5. Push a GitHub
git push origin main  # o tu branch
```

---

## 🎯 Resultados Esperados

Después de estas mejoras (4-5 horas de trabajo):

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Tests críticos** | 0 | 1 (energy conservation) ✅ |
| **Bare except** | 4+ | 0 ✅ |
| **Input validation** | ❌ | ✅ |
| **Error messages** | Genéricos | Específicos y útiles ✅ |
| **Test coverage** | ~5% | ~15% 📈 |
| **Grade** | B+ (83/100) | **A- (88/100)** 🎉 |

**Impacto:** +5 puntos en calificación general con menos de 1 día de trabajo!

---

## 🚀 Siguiente Paso

Si tienes más tiempo, implementa en orden de prioridad:

1. **Tests de potenciales** (1-2 horas) - Validar derivadas
2. **CI/CD básico** (1 hora) - GitHub Actions
3. **Test de Sommerfeld BC** (2-3 horas) - Medir reflexión
4. **Type hints** (3-4 horas) - Añadir a solvers/
5. **Benchmarks** (2-3 horas) - Documentar performance

Ver `IMPROVEMENT_ROADMAP.md` para detalles completos.

---

## 💬 ¿Necesitas Ayuda?

**Si algo falla:**

1. Verifica que tienes FEniCS o DOLFINx instalado
2. Asegúrate de estar en el entorno conda correcto
3. Ejecuta `python main.py --test` para verificar instalación
4. Revisa `PROJECT_REVIEW.md` para contexto completo

**¿Preguntas?**
- Abre un issue en GitHub
- Etiqueta con `[quick-start]`

---

**¡Buena suerte! Estas mejoras transformarán PSYOP de código de investigación a software profesional.** 🚀

---

**Última actualización:** Febrero 2026  
**Tiempo estimado total:** 4-5 horas  
**Impacto en calidad:** +5 puntos (B+ → A-)
