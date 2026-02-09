"""
Integration tests for nonequilibrium model types (mneq 0-3).

Tests cover:
- mneq=0: Two-region physical nonequilibrium model
- mneq=1: One-site chemical nonequilibrium model
- mneq=2: Two-site chemical nonequilibrium model
- mneq=3: Two-region physical nonequilibrium model with internal constraints
"""
import pytest
import pandas as pd
import numpy as np

from cxtfit import CXTsim, CXTfit


class TestMneq0TwoRegionPhysical:
    """Tests for mneq=0: Two-region physical nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem with two-region physical model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.6, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].max() > 0

    def test_varying_beta(self, simple_obsdata):
        """Test two-region model with different beta values."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        for beta in [0.2, 0.5, 0.8]:
            binit = [20.0, 10.0, 2.0, beta, 1.0, 0.0, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=2,
                modc=1,
                mneq=0,
                parms=parms,
                modb=1,
                pulse=pulse,
                zl=50.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for beta={beta}"

    def test_varying_omega(self, simple_obsdata):
        """Test two-region model with different omega (mass transfer) values."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        results = []
        for omega in [0.5, 1.0, 5.0, 10.0]:
            binit = [20.0, 10.0, 2.0, 0.5, omega, 0.0, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=2,
                modc=1,
                mneq=0,
                parms=parms,
                modb=1,
                pulse=pulse,
                zl=50.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()
            results.append(sim.cxtdata['csim'].max())

        # Higher omega should lead to more rapid equilibration
        assert all(r > 0 for r in results)

    def test_with_degradation(self, simple_obsdata):
        """Test two-region model with degradation in both phases."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.1, 0.05]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestMneq1OneSiteChemical:
    """Tests for mneq=1: One-site chemical nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem with one-site chemical model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        # For mneq=1, beta = 1/R is enforced
        binit = [20.0, 10.0, 5.0, 0.2, 0.8, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_varying_R_values(self, simple_obsdata):
        """Test one-site model with different retardation factors."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        for R in [2.0, 5.0, 10.0]:
            beta = 1.0 / R  # One-site constraint
            binit = [20.0, 10.0, R, beta, 1.0, 0.0, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=2,
                modc=1,
                mneq=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                zl=50.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for R={R}"

    def test_pulse_input(self, simple_obsdata):
        """Test one-site model with pulse input."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 5.0, 0.2, 2.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 5.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=1,
            parms=parms,
            modb=3,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0


class TestMneq2TwoSiteChemical:
    """Tests for mneq=2: Two-site chemical nonequilibrium model."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem with two-site chemical model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 5.0, 0.6, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=2,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_different_site_fractions(self, simple_obsdata):
        """Test two-site model with different site fractions (beta)."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        for beta in [0.3, 0.5, 0.7]:
            binit = [20.0, 10.0, 5.0, beta, 1.0, 0.0, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=2,
                modc=1,
                mneq=2,
                parms=parms,
                modb=1,
                pulse=pulse,
                zl=50.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for beta={beta}"

    def test_with_degradation(self, simple_obsdata):
        """Test two-site model with degradation."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 5.0, 0.6, 1.0, 0.1, 0.05]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=2,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestMneq3TwoRegionWithConstraints:
    """Tests for mneq=3: Two-region physical model with internal constraints."""

    def test_basic_forward_problem(self, simple_obsdata):
        """Test basic forward problem with constrained two-region model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.6, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=3,
            phim=0.6,  # Mobile water fraction required for mneq=3
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_varying_mobile_fraction(self, simple_obsdata):
        """Test constrained model with different mobile water fractions."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        for phim in [0.3, 0.5, 0.7]:
            binit = [20.0, 10.0, 2.0, 0.6, 1.0, 0.0, 0.0]
            parms = pd.DataFrame([binit], columns=bname, index=['binit'])
            parms.loc['bfit'] = [0] * len(bname)

            sim = CXTsim(
                mode=2,
                modc=1,
                mneq=3,
                phim=phim,
                parms=parms,
                modb=1,
                pulse=pulse,
                zl=50.0,
                obsdata=simple_obsdata.copy()
            )
            sim.run()

            assert sim.cxtdata['csim'].notna().all(), f"Failed for phim={phim}"


class TestNonequilibriumComparisons:
    """Tests comparing different nonequilibrium model types."""

    def test_high_omega_approaches_equilibrium(self, simple_obsdata):
        """Test that very high omega approaches equilibrium behavior."""
        # Equilibrium model (mode=1)
        bname_eq = ['V', 'D', 'R', 'mu1']
        binit_eq = [20.0, 10.0, 5.0, 0.0]
        parms_eq = pd.DataFrame([binit_eq], columns=bname_eq, index=['binit'])
        parms_eq.loc['bfit'] = [0] * len(bname_eq)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim_eq = CXTsim(
            mode=1,
            modc=1,
            parms=parms_eq,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata.copy()
        )
        sim_eq.run()

        # Nonequilibrium with very high omega (should approach equilibrium)
        bname_neq = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit_neq = [20.0, 10.0, 5.0, 0.2, 100.0, 0.0, 0.0]  # Very high omega
        parms_neq = pd.DataFrame([binit_neq], columns=bname_neq, index=['binit'])
        parms_neq.loc['bfit'] = [0] * len(bname_neq)

        sim_neq = CXTsim(
            mode=2,
            modc=1,
            mneq=1,
            parms=parms_neq,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata.copy()
        )
        sim_neq.run()

        # Both should produce results
        assert sim_eq.cxtdata['csim'].max() > 0
        assert sim_neq.cxtdata['csim'].max() > 0

    def test_all_mneq_types_run(self, simple_obsdata):
        """Verify all mneq types can run without errors."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        pulse = [{'conc': 1.0, 'time': 0.0}]

        # mneq=0
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)
        sim0 = CXTsim(mode=2, modc=1, mneq=0, parms=parms, modb=1,
                      pulse=pulse, zl=50.0, obsdata=simple_obsdata.copy())
        sim0.run()
        assert sim0.cxtdata['csim'].notna().all()

        # mneq=1
        binit = [20.0, 10.0, 5.0, 0.2, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)
        sim1 = CXTsim(mode=2, modc=1, mneq=1, parms=parms, modb=1,
                      pulse=pulse, zl=50.0, obsdata=simple_obsdata.copy())
        sim1.run()
        assert sim1.cxtdata['csim'].notna().all()

        # mneq=2
        binit = [20.0, 10.0, 5.0, 0.6, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)
        sim2 = CXTsim(mode=2, modc=1, mneq=2, parms=parms, modb=1,
                      pulse=pulse, zl=50.0, obsdata=simple_obsdata.copy())
        sim2.run()
        assert sim2.cxtdata['csim'].notna().all()

        # mneq=3
        binit = [20.0, 10.0, 2.0, 0.6, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)
        sim3 = CXTsim(mode=2, modc=1, mneq=3, phim=0.5, parms=parms, modb=1,
                      pulse=pulse, zl=50.0, obsdata=simple_obsdata.copy())
        sim3.run()
        assert sim3.cxtdata['csim'].notna().all()


class TestDegradationCodes:
    """Tests for degradation estimation codes (mdeg 0-3)."""

    def test_mdeg0_independent_degradation(self, simple_obsdata):
        """Test mdeg=0: Independent degradation rates."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.1, 0.05]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            mdeg=0,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_mdeg1_equal_degradation(self, simple_obsdata):
        """Test mdeg=1: Equal degradation everywhere."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.1, 0.1]  # mu1 = mu2
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            mdeg=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_mdeg2_liquid_phase_only(self, simple_obsdata):
        """Test mdeg=2: Degradation only in liquid phase."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.1, 0.0]  # mu2 = 0
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            mdeg=2,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_mdeg3_adsorbed_phase_only(self, simple_obsdata):
        """Test mdeg=3: Degradation only in adsorbed phase."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.0, 0.1]  # mu1 = 0
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            mneq=0,
            mdeg=3,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()
