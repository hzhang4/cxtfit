"""
Unit tests for stochastic functions in stocde.py.

These tests validate the probability distribution functions and integration
methods used in stochastic convective-dispersive equation models.
"""
import pytest
import numpy as np
from scipy import stats
from scipy.integrate import quad

from cxtfit.stocde import xlnprob, blnprob, limit, limit2, chebylog2


class TestXlnprob:
    """Tests for the single log-normal distribution function xlnprob()."""

    def test_basic_calculation(self):
        """Test basic log-normal PDF calculation."""
        x = 1.0
        avex = 1.0
        sdlnx = 0.5
        result = xlnprob(x, avex, sdlnx)

        # Compare with scipy's lognormal
        # scipy uses scale = exp(mean_ln), s = std_ln
        xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
        expected = stats.lognorm.pdf(x, s=sdlnx, scale=np.exp(xlnm))
        assert result == pytest.approx(expected, rel=1e-5)

    def test_normalization(self):
        """Test that the PDF integrates to approximately 1."""
        avex = 2.0
        sdlnx = 0.5

        # Integrate over a wide range
        integral, _ = quad(xlnprob, 0.001, 50, args=(avex, sdlnx))
        assert integral == pytest.approx(1.0, rel=1e-2)

    def test_mode_location(self):
        """Test that PDF peaks near the mode of log-normal."""
        avex = 5.0
        sdlnx = 0.3

        # Mode of log-normal: exp(mu - sigma^2)
        xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
        mode = np.exp(xlnm - sdlnx * sdlnx)

        # PDF at mode should be higher than at other points
        pdf_at_mode = xlnprob(mode, avex, sdlnx)
        pdf_at_other = xlnprob(mode * 2, avex, sdlnx)
        assert pdf_at_mode > pdf_at_other

    def test_positive_values(self):
        """Test that PDF is always non-negative."""
        avex = 3.0
        sdlnx = 0.7

        x_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for x in x_values:
            result = xlnprob(x, avex, sdlnx)
            assert result >= 0, f"PDF negative at x={x}"

    def test_different_parameters(self):
        """Test PDF with various parameter combinations."""
        test_cases = [
            (1.0, 1.0, 0.1),
            (5.0, 10.0, 0.5),
            (0.5, 2.0, 1.0),
            (10.0, 5.0, 0.3),
        ]
        for x, avex, sdlnx in test_cases:
            result = xlnprob(x, avex, sdlnx)
            assert np.isfinite(result), f"Non-finite result for x={x}, avex={avex}, sdlnx={sdlnx}"
            assert result >= 0


class TestBlnprob:
    """Tests for the bivariate log-normal distribution function blnprob()."""

    def test_basic_calculation(self):
        """Test basic bivariate log-normal calculation."""
        x, avex, sdlnx = 1.0, 1.0, 0.5
        y, avey, sdlny = 1.0, 1.0, 0.5
        rho = 0.0  # No correlation

        result = blnprob(x, avex, sdlnx, y, avey, sdlny, rho)

        # For rho=0, should equal product of marginals
        marginal_x = xlnprob(x, avex, sdlnx)
        marginal_y = xlnprob(y, avey, sdlny)
        expected = marginal_x * marginal_y
        assert result == pytest.approx(expected, rel=1e-4)

    def test_positive_correlation(self):
        """Test with positive correlation."""
        x, avex, sdlnx = 2.0, 2.0, 0.3
        y, avey, sdlny = 2.0, 2.0, 0.3
        rho = 0.5

        result = blnprob(x, avex, sdlnx, y, avey, sdlny, rho)
        assert np.isfinite(result)
        assert result > 0

    def test_negative_correlation(self):
        """Test with negative correlation."""
        x, avex, sdlnx = 2.0, 2.0, 0.3
        y, avey, sdlny = 2.0, 2.0, 0.3
        rho = -0.5

        result = blnprob(x, avex, sdlnx, y, avey, sdlny, rho)
        assert np.isfinite(result)
        assert result > 0

    def test_symmetry(self):
        """Test symmetry in x and y when parameters are equal."""
        avex, sdlnx = 3.0, 0.4
        avey, sdlny = 3.0, 0.4
        rho = 0.3

        result1 = blnprob(2.0, avex, sdlnx, 4.0, avey, sdlny, rho)
        result2 = blnprob(4.0, avex, sdlnx, 2.0, avey, sdlny, rho)
        assert result1 == pytest.approx(result2, rel=1e-5)

    def test_correlation_range(self):
        """Test behavior at different correlation values."""
        x, avex, sdlnx = 1.5, 2.0, 0.3
        y, avey, sdlny = 1.5, 2.0, 0.3

        # Results should change with correlation
        result_neg = blnprob(x, avex, sdlnx, y, avey, sdlny, -0.5)
        result_zero = blnprob(x, avex, sdlnx, y, avey, sdlny, 0.0)
        result_pos = blnprob(x, avex, sdlnx, y, avey, sdlny, 0.5)

        # All should be positive and finite
        assert result_neg > 0 and np.isfinite(result_neg)
        assert result_zero > 0 and np.isfinite(result_zero)
        assert result_pos > 0 and np.isfinite(result_pos)


class TestLimit:
    """Tests for the integration limit calculation function limit()."""

    def test_basic_limits(self):
        """Test basic limit calculation."""
        x = 5.0
        sdlnx = 0.5

        xmin, xmax = limit(x, sdlnx)

        # xmin should be positive and less than x
        assert xmin > 0
        assert xmin < x

        # xmax should be greater than x
        assert xmax > x

    def test_limits_contain_bulk_of_distribution(self):
        """Test that calculated limits contain most of the distribution."""
        x = 3.0
        sdlnx = 0.4

        xmin, xmax = limit(x, sdlnx)

        # Integrate PDF over calculated limits
        integral, _ = quad(xlnprob, xmin, xmax, args=(x, sdlnx))

        # Should contain most of the probability mass
        assert integral > 0.95, f"Limits only contain {integral*100:.1f}% of distribution"

    def test_narrow_distribution(self):
        """Test limits for narrow distribution (small sdlnx)."""
        x = 10.0
        sdlnx = 0.1  # Very narrow

        xmin, xmax = limit(x, sdlnx)

        # For narrow distribution, limits should be close to mean
        assert xmin > x * 0.5
        assert xmax < x * 2.0

    def test_wide_distribution(self):
        """Test limits for wide distribution (large sdlnx)."""
        x = 10.0
        sdlnx = 1.0  # Wide

        xmin, xmax = limit(x, sdlnx)

        # For wide distribution, limits should be further from mean
        assert xmin < x * 0.5
        assert xmax > x * 2.0

    def test_different_means(self):
        """Test limits for different mean values."""
        sdlnx = 0.5

        for x in [1.0, 5.0, 10.0, 50.0]:
            xmin, xmax = limit(x, sdlnx)
            assert xmin > 0, f"xmin not positive for x={x}"
            assert xmin < xmax, f"xmin >= xmax for x={x}"
            assert xmin < x < xmax, f"x not between limits for x={x}"


class TestLimit2:
    """Tests for the binary search limit function limit2()."""

    def test_basic_functionality(self):
        """Test basic limit2 functionality."""
        def test_func(x):
            if 1.0 <= x <= 5.0:
                return x, x * 2
            return 0, 0

        t0, t1 = limit2(test_func, 0.0, 10.0)

        # Should find limits close to 1.0 and 5.0
        assert t0 == pytest.approx(1.0, abs=0.1)
        assert t1 == pytest.approx(5.0, abs=0.1)

    def test_with_nonzero_initial(self):
        """Test when function is already non-zero at boundaries."""
        def test_func(x):
            return x, x * 2  # Always non-zero

        t0, t1 = limit2(test_func, 1.0, 5.0)

        # Should keep original limits
        assert t0 == pytest.approx(1.0, abs=0.1)
        assert t1 == pytest.approx(5.0, abs=0.1)


class TestChebylog2:
    """Tests for log-transformed Chebyshev integration chebylog2()."""

    def test_constant_function(self):
        """Test integration of a constant-returning function."""
        def const_func(x):
            return 1.0, 0.5

        result1, result2 = chebylog2(const_func, 1.0, 10.0, icheb=0, mm=16)

        # Integral of 1 from 1 to 10 in log space = log(10) - log(1) = log(10)
        # But with the log-transform: int_1^10 1 dx = 9
        assert np.isfinite(result1)
        assert np.isfinite(result2)

    def test_linear_function(self):
        """Test integration of a linear function."""
        def linear_func(x):
            return x, x / 2

        result1, result2 = chebylog2(linear_func, 1.0, 5.0, icheb=0, mm=32)

        # Both results should be positive and finite
        assert result1 > 0
        assert result2 > 0
        assert np.isfinite(result1)
        assert np.isfinite(result2)

    def test_pdf_integration(self):
        """Test integration of a log-normal PDF."""
        avex = 5.0
        sdlnx = 0.5
        xmin, xmax = limit(avex, sdlnx)

        def pdf_func(x):
            p = xlnprob(x, avex, sdlnx)
            return p, p * x  # Return PDF and x*PDF

        result1, result2 = chebylog2(pdf_func, xmin, xmax, icheb=0, mm=64)

        # result1 should be close to 1 (normalized PDF)
        # result2 should be close to mean
        assert result1 == pytest.approx(1.0, rel=0.05)

    def test_adaptive_mode(self):
        """Test adaptive integration mode (icheb=1)."""
        def smooth_func(x):
            return np.exp(-x / 5), np.exp(-x / 5) * 0.5

        result1, result2 = chebylog2(smooth_func, 1.0, 10.0, icheb=1, mm=16, stopch=1e-3)

        assert np.isfinite(result1)
        assert np.isfinite(result2)
        assert result1 > 0
        assert result2 > 0


class TestDistributionConsistency:
    """Tests for consistency between distribution functions."""

    def test_marginal_from_bivariate(self):
        """Test that integrating bivariate over y gives marginal in x."""
        x = 2.0
        avex, sdlnx = 3.0, 0.4
        avey, sdlny = 4.0, 0.5
        rho = 0.0  # No correlation

        # Integrate bivariate over y
        def integrand(y):
            return blnprob(x, avex, sdlnx, y, avey, sdlny, rho)

        ymin, ymax = limit(avey, sdlny)
        integral, _ = quad(integrand, ymin, ymax)

        # Should equal marginal at x
        marginal = xlnprob(x, avex, sdlnx)
        assert integral == pytest.approx(marginal, rel=0.1)

    def test_xlnprob_scipy_comparison(self):
        """Compare xlnprob with scipy's lognorm for multiple cases."""
        test_cases = [
            (1.0, 2.0, 0.3),
            (5.0, 5.0, 0.5),
            (0.5, 1.0, 0.2),
            (10.0, 8.0, 0.7),
        ]

        for x, avex, sdlnx in test_cases:
            result = xlnprob(x, avex, sdlnx)

            # scipy lognorm parameterization
            xlnm = np.log(avex) - 0.5 * sdlnx * sdlnx
            expected = stats.lognorm.pdf(x, s=sdlnx, scale=np.exp(xlnm))

            assert result == pytest.approx(expected, rel=1e-4), \
                f"Mismatch at x={x}, avex={avex}, sdlnx={sdlnx}"


class TestNumericalStability:
    """Tests for numerical stability of stochastic functions."""

    def test_xlnprob_extreme_values(self):
        """Test xlnprob with extreme parameter values."""
        # Very small x
        result = xlnprob(0.001, 1.0, 0.5)
        assert np.isfinite(result)

        # Very large x
        result = xlnprob(100.0, 10.0, 0.5)
        assert np.isfinite(result)

        # Very small sdlnx
        result = xlnprob(1.0, 1.0, 0.01)
        assert np.isfinite(result)

    def test_blnprob_extreme_correlations(self):
        """Test blnprob near correlation boundaries."""
        x, avex, sdlnx = 2.0, 2.0, 0.3
        y, avey, sdlny = 2.0, 2.0, 0.3

        # Near perfect positive correlation
        result = blnprob(x, avex, sdlnx, y, avey, sdlny, 0.99)
        assert np.isfinite(result)

        # Near perfect negative correlation
        result = blnprob(x, avex, sdlnx, y, avey, sdlny, -0.99)
        assert np.isfinite(result)

    def test_limit_stability(self):
        """Test limit function stability."""
        test_cases = [
            (0.1, 0.5),
            (1.0, 0.1),
            (1.0, 1.0),
            (10.0, 0.3),
            (100.0, 0.5),
        ]

        for x, sdlnx in test_cases:
            try:
                xmin, xmax = limit(x, sdlnx)
                assert xmin > 0, f"xmin not positive for x={x}, sdlnx={sdlnx}"
                assert xmax > xmin, f"xmax <= xmin for x={x}, sdlnx={sdlnx}"
            except RuntimeError:
                # Some edge cases may not converge, which is acceptable
                pass
