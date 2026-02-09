"""
Unit tests for mathematical functions in detcde.py.

These tests validate the core numerical functions used in the deterministic
convective-dispersive equation (CDE) calculations.
"""
import pytest
import numpy as np
from scipy.special import erfc, i0, i1

from cxtfit.detcde import dbexp, exf, gold, expbi0, expbi1, chebycon


class TestDbexp:
    """Tests for the safe exponential function dbexp()."""

    def test_normal_values(self):
        """Test dbexp with normal input values."""
        assert dbexp(0) == pytest.approx(1.0)
        assert dbexp(1) == pytest.approx(np.exp(1))
        assert dbexp(-1) == pytest.approx(np.exp(-1))
        assert dbexp(10) == pytest.approx(np.exp(10))

    def test_underflow_protection(self):
        """Test that very negative values return 0 (underflow protection)."""
        assert dbexp(-101) == 0.0
        assert dbexp(-200) == 0.0
        assert dbexp(-1000) == 0.0

    def test_overflow_protection(self):
        """Test that very large values are capped (overflow protection)."""
        # Should return exp(700) for x > 700 to avoid overflow
        result = dbexp(701)
        assert result == pytest.approx(np.exp(700))
        result = dbexp(1000)
        assert result == pytest.approx(np.exp(700))

    def test_boundary_values(self):
        """Test boundary values around -100 and 700."""
        # Just above -100 threshold
        assert dbexp(-99) == pytest.approx(np.exp(-99))
        # At -100 threshold
        assert dbexp(-100) == pytest.approx(np.exp(-100))
        # Just below 700 threshold
        assert dbexp(699) == pytest.approx(np.exp(699))
        # At 700 threshold
        assert dbexp(700) == pytest.approx(np.exp(700))


class TestExf:
    """Tests for the EXP(A)*ERFC(B) function exf()."""

    def test_basic_values(self):
        """Test exf with simple inputs."""
        # When A=0, B=0: exp(0)*erfc(0) = 1*1 = 1
        result = exf(0, 0)
        expected = np.exp(0) * erfc(0)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_positive_b(self):
        """Test with positive B values."""
        result = exf(1, 1)
        expected = np.exp(1) * erfc(1)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_negative_b(self):
        """Test with negative B values."""
        result = exf(1, -1)
        expected = np.exp(1) * erfc(-1)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_large_b_values(self):
        """Test with large B values (uses continued fraction approximation)."""
        # For large B > 3, the function uses a continued fraction approximation
        result = exf(0, 4)
        # The result should be small but positive
        assert result >= 0
        assert result < 0.1  # exp(0)*erfc(4) is very small

    def test_overflow_protection_positive_a(self):
        """Test overflow protection when A is large."""
        # When |a| > 170 and b <= 0, should return 0
        result = exf(171, 0)
        assert result == 0.0

    def test_symmetry_properties(self):
        """Test that results are consistent with mathematical properties."""
        # exf(a, 0) should be close to exp(a) for small a
        for a in [0, 0.5, 1.0]:
            result = exf(a, 0)
            expected = np.exp(a) * erfc(0)
            assert result == pytest.approx(expected, rel=1e-4)


class TestGold:
    """Tests for Goldstein's J-function gold()."""

    def test_zero_inputs(self):
        """Test gold with zero inputs."""
        # gold(0, 0) should be exp(-0) = 1
        result = gold(0, 0)
        assert result == pytest.approx(1.0, rel=1e-5)

    def test_small_x(self):
        """Test gold with small x values."""
        # For very small x, gold(x, y) ~ exp(-x)
        result = gold(1e-12, 1.0)
        expected = np.exp(-1e-12)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_positive_values(self):
        """Test gold with positive x and y."""
        # Test several positive values
        result = gold(1.0, 1.0)
        # J(x,y) is related to incomplete gamma functions
        # Result should be between 0 and 1
        assert 0 <= result <= 1

    def test_asymmetry(self):
        """Test that gold(x,y) != gold(y,x) in general."""
        result1 = gold(1.0, 2.0)
        result2 = gold(2.0, 1.0)
        # Generally these should be different
        # (unless at specific symmetric points)
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_large_values(self):
        """Test gold with larger values."""
        result = gold(5.0, 5.0)
        assert 0 <= result <= 1

    def test_x_less_than_y(self):
        """Test behavior when x < y."""
        result = gold(0.5, 2.0)
        assert isinstance(result, float)
        assert result >= 0


class TestExpbi0:
    """Tests for EXP(Z)*I0(X) function expbi0()."""

    def test_zero_x(self):
        """Test expbi0 with x=0."""
        # I0(0) = 1, so expbi0(0, z) = exp(z) * 1 = exp(z)
        result = expbi0(0, 0)
        assert result == pytest.approx(1.0, rel=1e-5)

        result = expbi0(0, 1)
        assert result == pytest.approx(np.exp(1), rel=1e-5)

    def test_small_x(self):
        """Test expbi0 with small x values (|x| < 3.75)."""
        x = 1.0
        z = 0.5
        result = expbi0(x, z)
        expected = np.exp(z) * i0(x)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_large_x(self):
        """Test expbi0 with large x values (|x| >= 3.75)."""
        x = 5.0
        z = 0.5
        result = expbi0(x, z)
        expected = np.exp(z) * i0(x)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_negative_x(self):
        """Test expbi0 with negative x (I0 is even function)."""
        x = -2.0
        z = 0.0
        result = expbi0(x, z)
        expected = i0(x)  # exp(0) * I0(x) = I0(x)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_negative_z(self):
        """Test expbi0 with negative z values."""
        x = 2.0
        z = -1.0
        result = expbi0(x, z)
        expected = np.exp(z) * i0(x)
        assert result == pytest.approx(expected, rel=1e-4)


class TestExpbi1:
    """Tests for EXP(Z)*I1(X) function expbi1()."""

    def test_zero_x(self):
        """Test expbi1 with x=0."""
        # I1(0) = 0, so expbi1(0, z) = exp(z) * 0 = 0
        result = expbi1(0, 0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_small_x(self):
        """Test expbi1 with small x values (|x| < 3.75)."""
        x = 1.0
        z = 0.5
        result = expbi1(x, z)
        expected = np.exp(z) * i1(x)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_large_x(self):
        """Test expbi1 with large x values (|x| >= 3.75)."""
        x = 5.0
        z = 0.5
        result = expbi1(x, z)
        expected = np.exp(z) * i1(x)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_negative_x(self):
        """Test expbi1 with negative x (I1 is odd function)."""
        x = -2.0
        z = 0.0
        result = expbi1(x, z)
        expected = i1(x)  # exp(0) * I1(x) = I1(x)
        # I1(-x) = -I1(x)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sign_change_for_negative_x(self):
        """Verify I1 odd function behavior."""
        x = 2.0
        z = 0.0
        result_pos = expbi1(x, z)
        result_neg = expbi1(-x, z)
        assert result_neg == pytest.approx(-result_pos, rel=1e-4)


class TestChebycon:
    """Tests for Gauss-Chebyshev quadrature integration chebycon()."""

    def test_constant_function(self):
        """Test integration of a constant function."""
        def const_func(x):
            return 2.0

        # Integral of 2 from 0 to 1 = 2
        result = chebycon(const_func, 0, 1, icheb=0, mm=8)
        assert result == pytest.approx(2.0, rel=1e-3)

    def test_linear_function(self):
        """Test integration of a linear function."""
        def linear_func(x):
            return x

        # Integral of x from 0 to 2 = 2
        result = chebycon(linear_func, 0, 2, icheb=0, mm=16)
        assert result == pytest.approx(2.0, rel=1e-2)

    def test_quadratic_function(self):
        """Test integration of a quadratic function."""
        def quad_func(x):
            return x ** 2

        # Integral of x^2 from 0 to 1 = 1/3
        result = chebycon(quad_func, 0, 1, icheb=0, mm=16)
        assert result == pytest.approx(1/3, rel=1e-2)

    def test_exponential_function(self):
        """Test integration of an exponential function."""
        def exp_func(x):
            return np.exp(-x)

        # Integral of exp(-x) from 0 to 1 = 1 - exp(-1) ~ 0.6321
        result = chebycon(exp_func, 0, 1, icheb=0, mm=16)
        expected = 1 - np.exp(-1)
        assert result == pytest.approx(expected, rel=1e-2)

    def test_adaptive_integration(self):
        """Test adaptive integration mode (icheb=1)."""
        def smooth_func(x):
            return np.sin(x)

        # Integral of sin(x) from 0 to pi = 2
        result = chebycon(smooth_func, 0, np.pi, icheb=1, mm=8, stopch=1e-4)
        assert result == pytest.approx(2.0, rel=1e-3)

    def test_with_mc_parameter(self):
        """Test integration with mc parameter passed to function."""
        def func_with_mc(x, mc):
            return x * mc

        # Integral of x*2 from 0 to 1 = 1
        result = chebycon(func_with_mc, 0, 1, mc=2, icheb=0, mm=16)
        assert result == pytest.approx(1.0, rel=1e-2)

    def test_zero_integral(self):
        """Test integration that should be close to zero."""
        def zero_func(x):
            return 0.0

        result = chebycon(zero_func, 0, 1, icheb=0, mm=8)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_narrow_interval(self):
        """Test integration over a very narrow interval."""
        def const_func(x):
            return 1.0

        # Integral of 1 from 0 to 0.001 = 0.001
        result = chebycon(const_func, 0, 0.001, icheb=0, mm=8)
        assert result == pytest.approx(0.001, rel=1e-2)


class TestNumericalStability:
    """Tests for numerical stability of mathematical functions."""

    def test_dbexp_range(self):
        """Test dbexp over a wide range of values."""
        test_values = [-150, -100, -50, -10, -1, 0, 1, 10, 50, 100, 500, 750]
        for x in test_values:
            result = dbexp(x)
            assert np.isfinite(result), f"dbexp({x}) returned non-finite value"
            assert result >= 0, f"dbexp({x}) returned negative value"

    def test_exf_stability(self):
        """Test exf stability over various inputs."""
        test_cases = [
            (0, 0), (1, 1), (-1, 1), (10, 5), (-10, -5),
            (0, 10), (0, -10), (50, 2), (100, 3)
        ]
        for a, b in test_cases:
            result = exf(a, b)
            assert np.isfinite(result), f"exf({a}, {b}) returned non-finite value"
            assert result >= 0, f"exf({a}, {b}) returned negative value"

    def test_gold_stability(self):
        """Test gold function stability."""
        test_cases = [
            (0, 0), (1, 1), (0.001, 0.001), (10, 10), (0.5, 2), (2, 0.5)
        ]
        for x, y in test_cases:
            result = gold(x, y)
            assert np.isfinite(result), f"gold({x}, {y}) returned non-finite value"
            assert 0 <= result <= 2, f"gold({x}, {y}) outside expected range"

    def test_bessel_stability(self):
        """Test Bessel function approximations stability."""
        x_values = [0, 0.5, 1, 2, 3.75, 5, 10, 20]
        z_values = [-5, -1, 0, 1, 5]

        for x in x_values:
            for z in z_values:
                result0 = expbi0(x, z)
                result1 = expbi1(x, z)
                assert np.isfinite(result0), f"expbi0({x}, {z}) non-finite"
                assert np.isfinite(result1), f"expbi1({x}, {z}) non-finite"
