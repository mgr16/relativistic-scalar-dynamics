"""
Test input validation for FirstOrderKGSolver.

Verifies that invalid inputs are properly rejected with clear error messages.
"""

import pytest
try:
    import dolfinx
    from mpi4py import MPI
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
class TestSolverInputValidation:
    """Test input validation in FirstOrderKGSolver."""
    
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for testing."""
        from psyop.mesh.gmsh import build_ball_mesh
        mesh, _, _ = build_ball_mesh(R=5.0, lc=3.0, comm=MPI.COMM_WORLD)
        return mesh
    
    def test_invalid_cfl_negative(self, simple_mesh):
        """Test that negative CFL is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="CFL factor must be in"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                cfl_factor=-0.1
            )
    
    def test_invalid_cfl_zero(self, simple_mesh):
        """Test that zero CFL is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="CFL factor must be in"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                cfl_factor=0.0
            )
    
    def test_invalid_cfl_too_large(self, simple_mesh):
        """Test that CFL > 1 is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="CFL factor must be in"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                cfl_factor=1.5
            )
    
    def test_valid_cfl_boundary(self, simple_mesh):
        """Test that CFL = 1.0 (boundary) is accepted."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        # Should not raise
        solver = FirstOrderKGSolver(
            mesh=simple_mesh,
            domain_radius=5.0,
            cfl_factor=1.0
        )
        assert solver.cfl_factor == 1.0
    
    def test_invalid_domain_radius_negative(self, simple_mesh):
        """Test that negative domain radius is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="domain_radius must be positive"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=-5.0
            )
    
    def test_invalid_domain_radius_zero(self, simple_mesh):
        """Test that zero domain radius is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="domain_radius must be positive"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=0.0
            )
    
    def test_invalid_degree_too_small(self, simple_mesh):
        """Test that degree < 1 is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="degree must be in"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                degree=0
            )
    
    def test_invalid_degree_too_large(self, simple_mesh):
        """Test that degree > 5 is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="degree must be in"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                degree=10
            )
    
    def test_valid_degrees(self, simple_mesh):
        """Test that valid degrees [1-5] are accepted."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        for degree in [1, 2, 3, 4, 5]:
            solver = FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                degree=degree
            )
            assert solver.degree == degree
    
    def test_invalid_potential_type(self, simple_mesh):
        """Test that invalid potential type is rejected."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        with pytest.raises(ValueError, match="Unknown potential type"):
            FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                potential_type="invalid_potential"
            )
    
    def test_valid_potential_types(self, simple_mesh):
        """Test that all valid potential types are accepted."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        valid_types = ["quadratic", "higgs", "mexican_hat", "zero"]
        
        for pot_type in valid_types:
            params = {}
            if pot_type == "higgs":
                params = {"m_squared": 1.0, "lambda_coupling": 0.1}
            elif pot_type == "mexican_hat":
                params = {"m_squared": -1.0, "lambda_coupling": 0.5}
            elif pot_type == "quadratic":
                params = {"m_squared": 1.0}
            
            solver = FirstOrderKGSolver(
                mesh=simple_mesh,
                domain_radius=5.0,
                potential_type=pot_type,
                potential_params=params
            )
            assert solver.potential is not None


@pytest.mark.skipif(not HAS_DOLFINX, reason="DOLFINx not available")
class TestSolverInitialization:
    """Test solver initialization and setup."""
    
    @pytest.fixture
    def simple_mesh(self):
        """Create a simple mesh for testing."""
        from psyop.mesh.gmsh import build_ball_mesh
        mesh, _, _ = build_ball_mesh(R=5.0, lc=3.0, comm=MPI.COMM_WORLD)
        return mesh
    
    def test_solver_creates_function_spaces(self, simple_mesh):
        """Test that solver creates function spaces."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        solver = FirstOrderKGSolver(
            mesh=simple_mesh,
            domain_radius=5.0,
            degree=1
        )
        
        assert solver.V_scalar is not None
        assert solver.V_vector is not None
    
    def test_solver_initializes_fields(self, simple_mesh):
        """Test that solver initializes phi and Pi fields."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        solver = FirstOrderKGSolver(
            mesh=simple_mesh,
            domain_radius=5.0,
            degree=1
        )
        
        phi, Pi = solver.get_fields()
        assert phi is not None
        assert Pi is not None
    
    def test_solver_potential_initialization(self, simple_mesh):
        """Test that solver initializes potential correctly."""
        from psyop.solvers.first_order import FirstOrderKGSolver
        
        solver = FirstOrderKGSolver(
            mesh=simple_mesh,
            domain_radius=5.0,
            potential_type="quadratic",
            potential_params={"m_squared": 2.0}
        )
        
        assert solver.potential is not None
        # Test potential can be evaluated
        import numpy as np
        V = solver.potential.evaluate(np.array([0.0, 1.0]))
        assert V.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
