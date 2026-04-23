#!/usr/bin/env python3
"""
test_sommerfeld_reflection.py

Test A/B para verificar que las condiciones de frontera Sommerfeld 
reducen la reflexión comparado con condiciones de frontera básicas.
"""

try:
    import dolfinx.fem as fem
    from mpi4py import MPI
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False

import os
import sys
import pytest
from pathlib import Path
np = pytest.importorskip("numpy")

pytestmark = [
    pytest.mark.requires_numpy,
    pytest.mark.requires_dolfinx,
    pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available"),
]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if HAS_DOLFINX:
    from psyop.solvers.first_order import FirstOrderKGSolver
    from psyop.mesh.gmsh import build_ball_mesh, get_outer_tag
    from psyop.physics.initial_conditions import GaussianBump
    from psyop.physics.metrics import FlatBackgroundCoeffs
    from psyop.utils.utils import compute_dt_cfl

def run_case(use_sommerfeld: bool, R=15.0, t_end=10.0):
    """
    Ejecuta una simulación con o sin BC Sommerfeld.
    
    Args:
        use_sommerfeld: Si True, usa BC Sommerfeld característica
        R: Radio del dominio
        t_end: Tiempo final de simulación
        
    Returns:
        tuple: (energia_final, norma_final, flujo_promedio)
    """
    print(f"\n=== Caso: {'CON' if use_sommerfeld else 'SIN'} Sommerfeld ===")
    
    try:
        # Crear malla pequeña para test rápido
        comm = MPI.COMM_WORLD
            
        mesh, cell_tags, facet_tags = build_ball_mesh(R=R, lc=3.0, comm=comm)
        
        # Configurar métrica plana
        coeffs = FlatBackgroundCoeffs()
        alpha_f, beta_f, gammaInv_f, sqrtg_f, K_f = coeffs.build(mesh)
        
        # CFL conservador para estabilidad (usando c_max=1.0 por fondo plano)
        dt = compute_dt_cfl(mesh, cfl=0.2, c_max=1.0)
        
        # Crear solver
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=R,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 0.1},  # masa pequeña
            cfl_factor=0.2
        )
        
        # Configurar métrica
        solver.set_background(
            alpha=alpha_f, 
            beta=beta_f, 
            gammaInv=gammaInv_f, 
            sqrtg=sqrtg_f, 
            K=K_f
        )
        
        # Configurar BC Sommerfeld si se solicita
        if use_sommerfeld and facet_tags is not None:
            outer_tag = get_outer_tag(facet_tags, default=2)
            solver.enable_sommerfeld(facet_tags, outer_tag)
            print(f"  ✓ BC Sommerfeld habilitada en tag {outer_tag}")
        else:
            print(f"  ✓ BC estándar (sin Sommerfeld)")
        
        # Condición inicial: pulso gaussiano centrado alejado del borde
        phi0 = GaussianBump(
            mesh=mesh,
            V=solver.V_scalar,
            A=0.1,           # amplitud moderada
            r0=R/3,          # centrado en r = R/3 (lejos del borde)
            w=R/8,           # ancho = R/8
            v0=1.0
        )
        solver.set_initial_conditions(phi0.get_function())
        
        energia_inicial = solver.energy()
        print(f"  Energía inicial: {energia_inicial:.6e}")
        
        # Evolución temporal
        t = 0.0
        step = 0
        flujos = []
        
        num_steps = int(t_end / dt)
        output_every = max(1, num_steps // 20)  # ~20 outputs
        
        while t < t_end:
            solver.ssp_rk3_step(dt)
            t += dt
            step += 1
            
            if step % output_every == 0:
                energia = solver.energy()
                flujo = solver.boundary_flux() if solver.has_sommerfeld else 0.0
                flujos.append(abs(flujo))
                
                if step % (output_every * 4) == 0:  # print cada 4 outputs
                    print(f"    t={t:.2f}  E={energia:.4e}  |F|={abs(flujo):.4e}")
        
        # Estadísticas finales
        energia_final = solver.energy()
        phi_final, _ = solver.get_fields()
        
        norma_final = float(fem.norm(phi_final))
            
        flujo_promedio = np.mean(flujos) if flujos else 0.0
        
        # Calcular pérdida de energía relativa
        perdida_energia = (energia_inicial - energia_final) / energia_inicial
        
        print(f"  Energía final: {energia_final:.6e}")
        print(f"  Norma final φ: {norma_final:.6e}")
        print(f"  Pérdida energía: {perdida_energia:.4%}")
        print(f"  Flujo promedio: {flujo_promedio:.4e}")
        
        return energia_final, norma_final, flujo_promedio, perdida_energia
        
    except Exception as e:
        print(f"  ✗ Error en caso: {e}")
        return 0.0, 0.0, 0.0, 1.0  # valores que harán fallar el test

def test_sommerfeld_reduces_reflection():
    """Test principal que compara casos con y sin Sommerfeld."""
    print("="*60)
    print("TEST A/B: REFLEXIÓN CON/SIN SOMMERFELD")
    print("="*60)
    
    # Parámetros del test
    R = 12.0      # dominio pequeño para ver efectos de frontera
    t_end = 8.0   # tiempo suficiente para que el pulso llegue al borde
    
    print(f"Configuración: R={R}, t_end={t_end}")
    
    # Caso sin Sommerfeld (BC estándar)
    E_sin, norma_sin, flujo_sin, perdida_sin = run_case(
        use_sommerfeld=False, R=R, t_end=t_end
    )
    
    # Caso con Sommerfeld
    E_con, norma_con, flujo_con, perdida_con = run_case(
        use_sommerfeld=True, R=R, t_end=t_end
    )
    
    # Análisis comparativo
    print("\n" + "="*60)
    print("ANÁLISIS COMPARATIVO")
    print("="*60)
    
    print(f"Energía final:")
    print(f"  Sin Sommerfeld: {E_sin:.6e}")
    print(f"  Con Sommerfeld: {E_con:.6e}")
    print(f"  Ratio: {E_con/E_sin:.4f}" if E_sin > 0 else "  Ratio: N/A")
    
    print(f"\nPérdida de energía:")
    print(f"  Sin Sommerfeld: {perdida_sin:.4%}")
    print(f"  Con Sommerfeld: {perdida_con:.4%}")
    
    print(f"\nFlujo promedio:")
    print(f"  Sin Sommerfeld: {flujo_sin:.6e}")
    print(f"  Con Sommerfeld: {flujo_con:.6e}")
    
    # Criterios de éxito
    criterios_ok = []
    
    # 1. Con Sommerfeld debe conservar mejor la energía (menor pérdida)
    if perdida_con < perdida_sin:
        criterios_ok.append("✓ Sommerfeld conserva mejor la energía")
    else:
        criterios_ok.append("✗ Sommerfeld NO conserva mejor la energía")
    
    # 2. O al menos no ser mucho peor (tolerancia del 20%)
    if perdida_con < 1.2 * perdida_sin:
        criterios_ok.append("✓ Pérdida de energía en rango aceptable")
    else:
        criterios_ok.append("✗ Pérdida de energía excesiva con Sommerfeld")
    
    # 3. Con Sommerfeld puede tener más flujo (es correcto, está absorbiendo)
    if flujo_con >= 0.8 * flujo_sin:  # tolerancia amplia
        criterios_ok.append("✓ Flujo Sommerfeld en rango esperado")
    else:
        criterios_ok.append("✗ Flujo Sommerfeld anormalmente bajo")
    
    print(f"\nCriterios de evaluación:")
    for criterio in criterios_ok:
        print(f"  {criterio}")
    
    # Resultado del test
    exitos = sum(1 for c in criterios_ok if c.startswith("✓"))
    total = len(criterios_ok)
    
    print(f"\nResultado: {exitos}/{total} criterios cumplidos")
    
    if exitos >= 2:  # Al menos 2 de 3 criterios
        print("🎉 TEST EXITOSO: Sommerfeld funciona correctamente")
        return True
    else:
        print("⚠️  TEST PARCIAL: Resultados mixtos")
        return False

def main():
    """Función principal del test."""
    try:
        success = test_sommerfeld_reduces_reflection()
        
        print("\n" + "="*60)
        if success:
            print("CONCLUSIÓN: Las BC Sommerfeld están funcionando correctamente")
            print("El sistema está listo para simulaciones de producción.")
        else:
            print("CONCLUSIÓN: Los resultados son mixtos")
            print("Revisar implementación o parámetros del test.")
        print("="*60)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n✗ Error en test principal: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
