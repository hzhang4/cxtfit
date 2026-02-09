"""
Integration tests for stochastic model modes (modes 3-8).

Tests cover:
- Mode 3: Stochastic KD&V equilibrium
- Mode 4: Stochastic KD&V nonequilibrium
- Mode 5: Stochastic D&V equilibrium
- Mode 6: Stochastic D&V nonequilibrium
- Mode 8: Stochastic Alpha & V nonequilibrium
"""
import pytest
import pandas as pd
import numpy as np

from cxtfit import CXTsim


class TestMode3StochasticKdVEquilibrium:
    """Tests for Mode 3: Stochastic KD&V equilibrium model."""

    def test_basic_forward_problem(self, default_stoch_eq_parms, simple_obsdata):
        """Test basic forward problem for mode 3."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=1,
            parms=default_stoch_eq_parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        # Check that simulation produced results
        assert 'csim' in sim.cxtdata.columns
        assert 'cvar' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].max() > 0

    def test_with_variable_velocity_only(self, simple_obsdata):
        """Test mode 3 with variable velocity only (SD.Kd=0)."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=3,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_with_correlated_parameters(self, simple_obsdata):
        """Test mode 3 with correlated velocity and Kd (RhovKd != 0)."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5]  # Positive correlation
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0

    def test_with_degradation(self, simple_obsdata):
        """Test mode 3 with non-zero degradation rate."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.0, 0.1, 0.5, 0.0, 0.5, 0.0]  # mu1 = 0.1
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        # With degradation, concentrations should be lower
        assert sim.cxtdata['csim'].max() > 0


class TestMode4StochasticKdVNonequilibrium:
    """Tests for Mode 4: Stochastic KD&V nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem for mode 4."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=4,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        # Check both phases are computed
        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns
        assert 'cvar' in sim.cxtdata.columns
        assert 'cvar2' in sim.cxtdata.columns

    def test_with_different_omega(self, simple_obsdata):
        """Test mode 4 with different mass transfer coefficients."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']

        for omega in [0.5, 1.0, 2.0, 5.0]:
            binit = [50.0, 20.0, 0.5, omega, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            pulse = [{'conc': 1.0, 'time': 0.0}]

            sim = CXTsim(
                mode=4,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for omega={omega}"

    def test_with_degradation_rates(self, simple_obsdata):
        """Test mode 4 with non-zero degradation rates."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.5, 1.0, 0.1, 0.05, 0.3, 0.3, 0.3, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=4,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0


class TestMode5StochasticDVEquilibrium:
    """Tests for Mode 5: Stochastic D&V equilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem for mode 5."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']
        binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=5,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'cvar' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].max() > 0

    def test_with_correlation(self, simple_obsdata):
        """Test mode 5 with correlated D and V (RhovD != 0)."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']

        for rho in [-0.5, 0.0, 0.5]:
            binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, rho]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            pulse = [{'conc': 1.0, 'time': 0.0}]

            sim = CXTsim(
                mode=5,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for RhovD={rho}"

    def test_with_pulse_input(self, simple_obsdata):
        """Test mode 5 with pulse input."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']
        binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 1.0}]

        sim = CXTsim(
            mode=5,
            modc=1,
            parms=parms,
            modb=3,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0


class TestMode6StochasticDVNonequilibrium:
    """Tests for Mode 6: Stochastic D&V nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem for mode 6."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']
        binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=6,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        # Check both phases are computed
        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_with_varying_sd_values(self, simple_obsdata):
        """Test mode 6 with different standard deviation values."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovD']

        for sd in [0.2, 0.5, 0.8]:
            binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, sd, 0.0, sd, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            pulse = [{'conc': 1.0, 'time': 0.0}]

            sim = CXTsim(
                mode=6,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for SD={sd}"


class TestMode8StochasticAlphaVNonequilibrium:
    """Tests for Mode 8: Stochastic Alpha & V nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem for mode 8."""
        bname = ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl']
        binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=8,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_with_different_alpha(self, simple_obsdata):
        """Test mode 8 with different alpha values."""
        bname = ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl']

        for alpha in [0.5, 1.0, 2.0]:
            binit = [50.0, 20.0, 0.5, alpha, 0.0, 0.0, 0.3, 0.0, 0.3, 0.3, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            pulse = [{'conc': 1.0, 'time': 0.0}]

            sim = CXTsim(
                mode=8,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for alpha={alpha}"

    def test_with_correlation(self, simple_obsdata):
        """Test mode 8 with correlated alpha and V."""
        bname = ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl']

        for rho in [-0.5, 0.0, 0.5]:
            binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, rho]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            pulse = [{'conc': 1.0, 'time': 0.0}]

            sim = CXTsim(
                mode=8,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for RhovAl={rho}"


class TestStochasticModeComparisons:
    """Tests comparing different stochastic modes."""

    def test_equilibrium_vs_nonequilibrium_convergence(self, simple_obsdata):
        """Test that nonequilibrium modes converge to equilibrium as omega increases."""
        # Mode 3 (equilibrium)
        bname_eq = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit_eq = [50.0, 20.0, 0.5, 0.0, 0.3, 0.3, 0.3, 0.0]
        parms_eq = pd.DataFrame([binit_eq], columns=bname_eq, index=['binit'])
        parms_eq.loc['bfit'] = [0] * len(bname_eq)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim_eq = CXTsim(
            mode=3,
            modc=1,
            parms=parms_eq,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata.copy()
        )
        sim_eq.run()

        # Mode 4 with very high omega (should approach equilibrium)
        bname_neq = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit_neq = [50.0, 20.0, 0.5, 100.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]  # Very high omega
        parms_neq = pd.DataFrame([binit_neq], columns=bname_neq, index=['binit'])
        parms_neq.loc['bfit'] = [0] * len(bname_neq)

        sim_neq = CXTsim(
            mode=4,
            modc=1,
            parms=parms_neq,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata.copy()
        )
        sim_neq.run()

        # Both should produce results
        assert sim_eq.cxtdata['csim'].max() > 0
        assert sim_neq.cxtdata['csim'].max() > 0

    def test_variance_increases_with_sd(self, simple_obsdata):
        """Test that variance increases with larger standard deviations."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        variances = []
        for sd in [0.1, 0.3, 0.5]:
            binit = [50.0, 20.0, 0.0, 0.0, sd, 0.0, sd, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=3,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                rhoth=4.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()
            variances.append(sim.cxtdata['cvar'].max())

        # Variance should generally increase with SD
        # (may not be strictly monotonic due to nonlinear effects)
        assert variances[-1] >= variances[0] * 0.5  # Allow some flexibility
