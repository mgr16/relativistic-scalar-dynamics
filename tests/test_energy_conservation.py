"""
Test energy conservation for the Klein-Gordon solver.

These tests verify that the numerical scheme conserves energy
to within acceptable tolerances for different CFL factors.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import dolfinx
    from mpi4py import MPI
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
class TestEnergyConservation:
    """Test suite for energy conservation."""
    
    def test_energy_conservation_quadratic_potential(self):
        """Test energy conservation with quadratic potential."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        from psyop.mesh.gmsh import build_ball_mesh
        from psyop.physics.initial_conditions import GaussianBump
        
        # Small mesh for fast testing
        mesh, _, facet_tags = build_ball_mesh(R=5.0, lc=3.0, comm=MPI.COMM_WORLD)
        
        # Initialize solver
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=5.0,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=0.3,
        )
        
        # Set initial conditions (small amplitude for linear regime)
        ic = GaussianBump(mesh, A=0.01, r0=2.0, w=1.0, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        
        # Measure initial energy
        E0 = solver.energy()
        assert E0 > 0, "Initial energy should be positive"
        
        # Evolve for multiple steps
        dt = 0.01
        n_steps = 20
        for _ in range(n_steps):
            solver.ssp_rk3_step(dt)
        
        # Measure final energy
        Ef = solver.energy()
        
        # Check conservation (allow 10% drift for coarse mesh)
        rel_error = abs(Ef - E0) / E0
        assert rel_error < 0.10, f"Energy drift too large: {rel_error:.2%}"
    
    @pytest.mark.parametrize("cfl", [0.1, 0.2, 0.3])
    def test_energy_conservation_varying_cfl(self, cfl):
        """Test energy conservation with different CFL factors."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        from psyop.mesh.gmsh import build_ball_mesh
        from psyop.physics.initial_conditions import GaussianBump
        
        # Small mesh
        mesh, _, facet_tags = build_ball_mesh(R=5.0, lc=2.5, comm=MPI.COMM_WORLD)
        
        # Initialize solver with varying CFL
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=5.0,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=cfl,
        )
        
        # Set initial conditions
        ic = GaussianBump(mesh, A=0.01, r0=2.0, w=1.0, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        
        # Initial energy
        E0 = solver.energy()
        
        # Evolve
        dt = 0.01
        for _ in range(15):
            solver.ssp_rk3_step(dt)
        
        # Final energy
        Ef = solver.energy()
        
        # Check conservation (smaller CFL should have better conservation)
        rel_error = abs(Ef - E0) / E0
        max_allowed = 0.15  # 15% for coarse mesh
        assert rel_error < max_allowed, \
            f"Energy drift {rel_error:.2%} > {max_allowed:.2%} for CFL={cfl}"
    
    def test_zero_potential_conservation(self):
        """Test that zero potential gives perfect conservation (within numerical error)."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        from psyop.mesh.gmsh import build_ball_mesh
        from psyop.physics.initial_conditions import GaussianBump
        
        mesh, _, _ = build_ball_mesh(R=5.0, lc=2.0, comm=MPI.COMM_WORLD)
        
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=5.0,
            degree=1,
            potential_type="zero",
            cfl_factor=0.2,
        )
        
        ic = GaussianBump(mesh, A=0.01, r0=2.0, w=1.0, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        
        E0 = solver.energy()
        
        # Evolve longer
        dt = 0.01
        for _ in range(30):
            solver.ssp_rk3_step(dt)
        
        Ef = solver.energy()
        
        # Zero potential should conserve very well
        rel_error = abs(Ef - E0) / E0
        assert rel_error < 0.05, f"Energy drift for zero potential: {rel_error:.2%}"


@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
class TestEnergyNonNegativity:
    """Test that energy remains non-negative."""
    
    def test_energy_positive(self):
        """Test that energy is always positive."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        from psyop.mesh.gmsh import build_ball_mesh
        from psyop.physics.initial_conditions import GaussianBump
        
        mesh, _, _ = build_ball_mesh(R=5.0, lc=2.5, comm=MPI.COMM_WORLD)
        
        solver = FirstOrderKGSolver(
            mesh=mesh,
            domain_radius=5.0,
            degree=1,
            potential_type="quadratic",
            potential_params={"m_squared": 1.0},
            cfl_factor=0.3,
        )
        
        ic = GaussianBump(mesh, A=0.05, r0=2.0, w=1.0, v0=0.0)
        solver.set_initial_conditions(ic.get_function())
        
        # Check energy at multiple time steps
        dt = 0.01
        for step in range(20):
            E = solver.energy()
            assert E >= 0, f"Energy became negative at step {step}: E={E}"
            solver.ssp_rk3_step(dt)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
