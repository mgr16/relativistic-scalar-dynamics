"""
Test potential implementations and their derivatives.

Verifies that analytical derivatives match numerical derivatives
for all potential types.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestPotentialDerivatives:
    """Test analytical vs numerical derivatives for all potentials."""
    
    def test_quadratic_potential_derivative(self):
        """Test QuadraticPotential derivative."""
        from psyop.physics.potential import QuadraticPotential
        
        pot = QuadraticPotential(m_squared=1.0)
        
        # Test at multiple points
        phi_test = np.linspace(-5, 5, 100)
        
        # Analytical derivative
        V_prime_analytical = pot.derivative(phi_test)
        
        # Numerical derivative (finite differences)
        epsilon = 1e-6
        V_prime_numerical = (
            pot.evaluate(phi_test + epsilon) - 
            pot.evaluate(phi_test - epsilon)
        ) / (2 * epsilon)
        
        # Compare
        np.testing.assert_allclose(
            V_prime_analytical, 
            V_prime_numerical,
            rtol=1e-5, 
            atol=1e-8,
            err_msg="Quadratic potential derivative mismatch"
        )
    
    def test_higgs_potential_derivative(self):
        """Test HiggsPotential derivative."""
        from psyop.physics.potential import HiggsPotential
        
        pot = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
        
        phi_test = np.linspace(-5, 5, 100)
        
        # Analytical
        V_prime_analytical = pot.derivative(phi_test)
        
        # Numerical
        epsilon = 1e-6
        V_prime_numerical = (
            pot.evaluate(phi_test + epsilon) - 
            pot.evaluate(phi_test - epsilon)
        ) / (2 * epsilon)
        
        # Compare
        np.testing.assert_allclose(
            V_prime_analytical,
            V_prime_numerical,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Higgs potential derivative mismatch"
        )
    
    def test_mexican_hat_potential_derivative(self):
        """Test MexicanHatPotential derivative."""
        from psyop.physics.potential import MexicanHatPotential
        
        pot = MexicanHatPotential(m_squared=-1.0, lambda_coupling=0.5)
        
        phi_test = np.linspace(-3, 3, 100)
        
        # Analytical
        V_prime_analytical = pot.derivative(phi_test)
        
        # Numerical
        epsilon = 1e-6
        V_prime_numerical = (
            pot.evaluate(phi_test + epsilon) - 
            pot.evaluate(phi_test - epsilon)
        ) / (2 * epsilon)
        
        # Compare
        np.testing.assert_allclose(
            V_prime_analytical,
            V_prime_numerical,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Mexican hat potential derivative mismatch"
        )
    
    def test_zero_potential_derivative(self):
        """Test ZeroPotential derivative."""
        from psyop.physics.potential import ZeroPotential
        
        pot = ZeroPotential()
        
        phi_test = np.linspace(-10, 10, 50)
        
        # Analytical
        V_prime_analytical = pot.derivative(phi_test)
        
        # Should be exactly zero
        assert np.all(V_prime_analytical == 0), "Zero potential derivative should be zero"
    
    @pytest.mark.parametrize("pot_type,params", [
        ("quadratic", {"m_squared": 2.0}),
        ("higgs", {"m_squared": 1.0, "lambda_coupling": 0.1}),
        ("mexican_hat", {"m_squared": -1.0, "lambda_coupling": 0.5}),
        ("zero", {}),
    ])
    def test_potential_factory(self, pot_type, params):
        """Test that all potential types can be constructed via factory."""
        from psyop.physics.potential import get_potential
        
        pot = get_potential(pot_type, **params)
        
        # Test evaluation doesn't crash
        phi = np.array([0.1, 0.5, 1.0])
        V = pot.evaluate(phi)
        V_prime = pot.derivative(phi)
        
        assert V.shape == phi.shape, "Potential evaluation shape mismatch"
        assert V_prime.shape == phi.shape, "Derivative shape mismatch"


class TestPotentialProperties:
    """Test physical properties of potentials."""
    
    def test_quadratic_potential_minimum(self):
        """Test that quadratic potential has minimum at phi=0."""
        from psyop.physics.potential import QuadraticPotential
        
        pot = QuadraticPotential(m_squared=1.0)
        
        # Derivative at phi=0 should be zero
        V_prime_zero = pot.derivative(np.array([0.0]))
        assert abs(V_prime_zero[0]) < 1e-10, "Quadratic potential should have minimum at phi=0"
        
        # Value at phi=0 should be zero
        V_zero = pot.evaluate(np.array([0.0]))
        assert abs(V_zero[0]) < 1e-10, "Quadratic potential should be zero at phi=0"
    
    def test_higgs_potential_symmetry(self):
        """Test that Higgs potential is symmetric."""
        from psyop.physics.potential import HiggsPotential
        
        pot = HiggsPotential(m_squared=1.0, lambda_coupling=0.1)
        
        phi_values = np.array([-2.0, -1.0, 1.0, 2.0])
        V_pos = pot.evaluate(phi_values[2:])
        V_neg = pot.evaluate(phi_values[:2])
        
        # V(phi) = V(-phi)
        np.testing.assert_allclose(V_pos, V_neg[::-1], rtol=1e-10)
    
    def test_mexican_hat_double_well(self):
        """Test that Mexican hat potential has double-well structure."""
        from psyop.physics.potential import MexicanHatPotential
        
        pot = MexicanHatPotential(m_squared=-1.0, lambda_coupling=1.0)
        
        # Check that V(0) > 0 (barrier)
        V_zero = pot.evaluate(np.array([0.0]))
        assert V_zero[0] > 0, "Mexican hat should have positive value at phi=0"
        
        # Check that minima exist at phi = ±sqrt(|m²|/λ)
        phi_min = np.sqrt(1.0 / 1.0)  # sqrt(|m²|/λ)
        V_min = pot.evaluate(np.array([phi_min, -phi_min]))
        
        # Minima should have negative or zero energy
        assert V_min[0] <= 0, "Mexican hat minima should have non-positive energy"
        assert V_min[1] <= 0, "Mexican hat minima should have non-positive energy"


class TestPotentialValidation:
    """Test input validation for potentials."""
    
    def test_invalid_potential_type(self):
        """Test that invalid potential type raises error."""
        from psyop.physics.potential import get_potential
        
        with pytest.raises(ValueError, match="Unknown potential type"):
            get_potential("invalid_potential")
    
    def test_higgs_invalid_lambda(self):
        """Test that Higgs potential validates lambda > 0."""
        from psyop.physics.potential import HiggsPotential
        
        with pytest.raises(ValueError):
            HiggsPotential(m_squared=1.0, lambda_coupling=-0.1)
    
    def test_mexican_hat_invalid_lambda(self):
        """Test that Mexican hat validates lambda > 0."""
        from psyop.physics.potential import MexicanHatPotential
        
        with pytest.raises(ValueError):
            MexicanHatPotential(m_squared=-1.0, lambda_coupling=-0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
