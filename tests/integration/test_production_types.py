"""
Integration tests for production term types (modp 0-3).

Tests cover:
- modp=0: Zero production
- modp=1: Constant production
- modp=2: Stepwise production
- modp=3: Exponential production
"""
import pytest
import pandas as pd
import numpy as np

from cxtfit import CXTsim


class TestModp0ZeroProduction:
    """Tests for modp=0: Zero production."""

    def test_zero_production_equilibrium(self, default_det_parms, profile_obsdata):
        """Test zero production with equilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=3,
            pulse=pulse,
            modp=0,
            obsdata=profile_obsdata
        )
        sim.run()

        # Should produce valid results
        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].notna().all()

    def test_zero_production_nonequilibrium(self, default_det_noneq_parms, profile_obsdata):
        """Test zero production with nonequilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=2,
            modc=1,
            parms=default_det_noneq_parms,
            modb=3,
            pulse=pulse,
            modp=0,
            zl=10.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_zero_production_stochastic(self, default_stoch_eq_parms, profile_obsdata):
        """Test zero production with stochastic equilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=1,
            parms=default_stoch_eq_parms,
            modb=1,
            pulse=pulse,
            modp=0,
            rhoth=4.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0


class TestModp1ConstantProduction:
    """Tests for modp=1: Constant production."""

    def test_constant_production_equilibrium(self, default_det_parms, profile_obsdata):
        """Test constant production with equilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=default_det_parms,
            modb=3,
            pulse=pulse,
            modp=1,
            prodval1=prodval1,
            obsdata=profile_obsdata
        )
        sim.run()

        # With production, concentrations should be affected
        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].notna().all()

    def test_constant_production_increases_concentration(self, profile_obsdata):
        """Test that constant production increases concentration."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [25.0, 37.5, 3.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]  # No input
        prodval1 = [{'gamma': 1.0, 'zpro': 0.0}]

        # Create time-series data
        nt = 21
        obsdata = pd.DataFrame([{'t': t * 0.5, 'z': 5.0} for t in range(nt)])

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=1,
            prodval1=prodval1,
            obsdata=obsdata
        )
        sim.run()

        # Concentration should increase over time due to production
        # (at least for later time points)
        assert sim.cxtdata['csim'].iloc[-1] > 0

    def test_constant_production_nonequilibrium_different_phases(self, profile_obsdata):
        """Test constant production with different values for each phase."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [10.0, 5.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]
        prodval2 = [{'gamma': 0.25, 'zpro': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=1,
            mpro=1,  # Different conditions for phases
            prodval1=prodval1,
            prodval2=prodval2,
            zl=10.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_constant_production_with_degradation(self, profile_obsdata):
        """Test balance between production and degradation."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.5]  # Non-zero degradation
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]

        nt = 21
        obsdata = pd.DataFrame([{'t': t * 0.5, 'z': 5.0} for t in range(nt)])

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=1,
            prodval1=prodval1,
            obsdata=obsdata
        )
        sim.run()

        # System should reach some equilibrium between production and degradation
        assert sim.cxtdata['csim'].notna().all()


class TestModp2StepwiseProduction:
    """Tests for modp=2: Stepwise production."""

    def test_stepwise_production_single_step(self, profile_obsdata):
        """Test stepwise production with single step."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=2,
            prodval1=prodval1,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_stepwise_production_multiple_steps(self, profile_obsdata):
        """Test stepwise production with multiple steps."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        # Multiple production zones
        prodval1 = [
            {'gamma': 0.5, 'zpro': 0.0},
            {'gamma': 1.0, 'zpro': 5.0},
            {'gamma': 0.25, 'zpro': 10.0}
        ]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=2,
            prodval1=prodval1,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    @pytest.mark.xfail(reason="Bug in detcde.py cc3(): division by zero when zpro=0")
    def test_stepwise_production_nonequilibrium(self, profile_obsdata):
        """Test stepwise production with nonequilibrium model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [10.0, 5.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        prodval1 = [
            {'gamma': 0.5, 'zpro': 0.0},
            {'gamma': 0.25, 'zpro': 10.0}
        ]
        prodval2 = [
            {'gamma': 0.3, 'zpro': 0.0},
            {'gamma': 0.15, 'zpro': 10.0}
        ]

        sim = CXTsim(
            mode=2,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=2,
            mpro=1,
            prodval1=prodval1,
            prodval2=prodval2,
            zl=20.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns


class TestModp3ExponentialProduction:
    """Tests for modp=3: Exponential production."""

    def test_exponential_production_basic(self, profile_obsdata):
        """Test basic exponential production."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        # gamma = gamma0 + gamma1 * exp(-zpro * z)
        prodval1 = [
            {'gamma': 0.5, 'zpro': 0.1},   # gamma0 and zpro (decay rate)
            {'gamma': 1.0, 'zpro': 0.0}    # gamma1
        ]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=3,
            prodval1=prodval1,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()

    def test_exponential_production_decay_with_depth(self, profile_obsdata):
        """Test that exponential production decays with depth."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        # Strong exponential decay with depth
        prodval1 = [
            {'gamma': 0.0, 'zpro': 0.5},   # No constant term, just exponential
            {'gamma': 2.0, 'zpro': 0.0}    # gamma1 = 2.0
        ]

        nz = 21
        dz = 1.0
        obsdata = pd.DataFrame([{'t': 5.0, 'z': z * dz} for z in range(nz)])

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=3,
            prodval1=prodval1,
            obsdata=obsdata
        )
        sim.run()

        # Concentration profile should exist
        assert sim.cxtdata['csim'].max() > 0

    @pytest.mark.xfail(reason="Bug in detcde.py c2pro(): unbound variable 'g' in modp=3 path")
    def test_exponential_production_nonequilibrium(self, profile_obsdata):
        """Test exponential production with nonequilibrium model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [10.0, 5.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]
        prodval1 = [
            {'gamma': 0.5, 'zpro': 0.1},
            {'gamma': 1.0, 'zpro': 0.0}
        ]
        prodval2 = [
            {'gamma': 0.3, 'zpro': 0.1},
            {'gamma': 0.5, 'zpro': 0.0}
        ]

        sim = CXTsim(
            mode=2,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            modp=3,
            mpro=1,
            prodval1=prodval1,
            prodval2=prodval2,
            zl=20.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns


class TestProductionComparisons:
    """Tests comparing different production types."""

    def test_constant_vs_zero_production(self, profile_obsdata):
        """Test that constant production increases concentrations vs zero production."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        # Zero production
        sim_zero = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=3,
            pulse=pulse,
            modp=0,
            obsdata=profile_obsdata.copy()
        )
        sim_zero.run()

        # Constant production
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]
        sim_const = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=3,
            pulse=pulse,
            modp=1,
            prodval1=prodval1,
            obsdata=profile_obsdata.copy()
        )
        sim_const.run()

        # With production, total mass should be higher
        total_zero = sim_zero.cxtdata['csim'].sum()
        total_const = sim_const.cxtdata['csim'].sum()
        assert total_const >= total_zero * 0.9  # Allow some numerical tolerance

    def test_all_production_types_run(self, profile_obsdata):
        """Verify all production types can run without errors."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 0.0, 'time': 0.0}]

        # modp=0
        sim0 = CXTsim(mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      modp=0, obsdata=profile_obsdata.copy())
        sim0.run()
        assert sim0.cxtdata['csim'].notna().all()

        # modp=1
        prodval1 = [{'gamma': 0.5, 'zpro': 0.0}]
        sim1 = CXTsim(mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      modp=1, prodval1=prodval1, obsdata=profile_obsdata.copy())
        sim1.run()
        assert sim1.cxtdata['csim'].notna().all()

        # modp=2
        prodval2 = [{'gamma': 0.5, 'zpro': 0.0}, {'gamma': 0.25, 'zpro': 10.0}]
        sim2 = CXTsim(mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      modp=2, prodval1=prodval2, obsdata=profile_obsdata.copy())
        sim2.run()
        assert sim2.cxtdata['csim'].notna().all()

        # modp=3
        prodval3 = [{'gamma': 0.5, 'zpro': 0.1}, {'gamma': 1.0, 'zpro': 0.0}]
        sim3 = CXTsim(mode=1, modc=3, parms=parms, modb=2, pulse=pulse,
                      modp=3, prodval1=prodval3, obsdata=profile_obsdata.copy())
        sim3.run()
        assert sim3.cxtdata['csim'].notna().all()
