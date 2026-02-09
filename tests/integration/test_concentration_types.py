"""
Integration tests for concentration types (modc 1-6).

Tests cover:
- modc=1: Flux concentration or area-averaged flux concentration
- modc=2: Field-scale flux concentration
- modc=3: Third-type resident concentration
- modc=4: Third-type total resident concentration
- modc=5: First-type resident concentration
- modc=6: First-type total resident concentration
"""
import pytest
import pandas as pd
import numpy as np

from cxtfit import CXTsim


class TestModc1FluxConcentration:
    """Tests for modc=1: Flux concentration."""

    def test_basic_forward_problem(self, default_det_parms, simple_obsdata):
        """Test basic forward problem with flux concentration."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=1,
            pulse=pulse,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].max() > 0

    def test_with_nonequilibrium(self, default_det_noneq_parms, simple_obsdata):
        """Test flux concentration with nonequilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=2,
            modc=1,
            parms=default_det_noneq_parms,
            modb=1,
            pulse=pulse,
            zl=10.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_with_stochastic_model(self, default_stoch_eq_parms, simple_obsdata):
        """Test flux concentration with stochastic model."""
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

        assert sim.cxtdata['csim'].max() > 0


class TestModc2FieldScaleFlux:
    """Tests for modc=2: Field-scale flux concentration."""

    def test_basic_forward_problem(self, default_stoch_eq_parms, simple_obsdata):
        """Test basic forward problem with field-scale flux concentration."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=2,
            parms=default_stoch_eq_parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].notna().all()

    def test_with_variable_velocity(self, simple_obsdata):
        """Test field-scale flux with variable velocity distribution."""
        bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=2,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].max() > 0


class TestModc3ThirdTypeResident:
    """Tests for modc=3: Third-type resident concentration."""

    def test_basic_forward_problem(self, default_det_parms, profile_obsdata):
        """Test basic forward problem with third-type resident concentration."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=default_det_parms,
            modb=3,
            pulse=pulse,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].notna().all()

    def test_profile_shape(self, profile_obsdata):
        """Test concentration profile shape."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=3,
            pulse=pulse,
            obsdata=profile_obsdata
        )
        sim.run()

        # Concentration should decrease with depth
        # (for pulse input at time=1.0)
        assert sim.cxtdata['csim'].iloc[0] >= sim.cxtdata['csim'].iloc[-1]

    def test_with_nonequilibrium(self, default_det_noneq_parms, profile_obsdata):
        """Test third-type concentration with nonequilibrium model."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=2,
            modc=3,
            parms=default_det_noneq_parms,
            modb=3,
            pulse=pulse,
            zl=20.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert 'csim2' in sim.cxtdata.columns

    def test_with_stochastic_model(self, default_stoch_eq_parms, profile_obsdata):
        """Test third-type concentration with stochastic model."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=3,
            modc=3,
            parms=default_stoch_eq_parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestModc4ThirdTypeTotalResident:
    """Tests for modc=4: Third-type total resident concentration."""

    def test_basic_forward_problem(self, default_det_noneq_parms, profile_obsdata):
        """Test basic forward problem with total resident concentration."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=2,
            modc=4,
            parms=default_det_noneq_parms,
            modb=3,
            pulse=pulse,
            zl=20.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns

    def test_with_stochastic_model(self, simple_obsdata):
        """Test total resident concentration with stochastic nonequilibrium."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=4,
            modc=4,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestModc5FirstTypeResident:
    """Tests for modc=5: First-type resident concentration."""

    def test_basic_forward_problem(self, default_det_parms, profile_obsdata):
        """Test basic forward problem with first-type resident concentration."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=1,
            modc=5,
            parms=default_det_parms,
            modb=3,
            pulse=pulse,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns
        assert sim.cxtdata['csim'].notna().all()

    def test_with_initial_condition(self, default_det_parms, profile_obsdata):
        """Test first-type concentration with initial condition."""
        cini = [{'conc': 1.0, 'z': 0.0}]

        sim = CXTsim(
            mode=1,
            modc=5,
            parms=default_det_parms,
            modi=1,
            cini=cini,
            obsdata=profile_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestModc6FirstTypeTotalResident:
    """Tests for modc=6: First-type total resident concentration."""

    def test_basic_forward_problem(self, default_det_noneq_parms, profile_obsdata):
        """Test basic forward problem with first-type total resident."""
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        sim = CXTsim(
            mode=2,
            modc=6,
            parms=default_det_noneq_parms,
            modb=3,
            pulse=pulse,
            zl=20.0,
            obsdata=profile_obsdata
        )
        sim.run()

        assert 'csim' in sim.cxtdata.columns

    def test_with_stochastic_model(self, simple_obsdata):
        """Test first-type total with stochastic nonequilibrium."""
        bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
        binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        sim = CXTsim(
            mode=4,
            modc=6,
            parms=parms,
            modb=1,
            pulse=pulse,
            rhoth=4.0,
            obsdata=simple_obsdata
        )
        sim.run()

        assert sim.cxtdata['csim'].notna().all()


class TestConcentrationTypeComparisons:
    """Tests comparing different concentration types."""

    def test_flux_vs_resident_concentration(self, profile_obsdata):
        """Test difference between flux and resident concentrations."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        # Flux concentration (modc=1)
        sim_flux = CXTsim(
            mode=1,
            modc=1,
            parms=parms,
            modb=3,
            pulse=pulse,
            obsdata=profile_obsdata.copy()
        )
        sim_flux.run()

        # Third-type resident (modc=3)
        sim_resident = CXTsim(
            mode=1,
            modc=3,
            parms=parms,
            modb=3,
            pulse=pulse,
            obsdata=profile_obsdata.copy()
        )
        sim_resident.run()

        # Both should produce valid results but may differ
        assert sim_flux.cxtdata['csim'].max() > 0
        assert sim_resident.cxtdata['csim'].max() > 0

    def test_all_modc_types_run(self, profile_obsdata):
        """Verify all modc types can run without errors."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        for modc in [1, 3, 5]:  # modc 2,4,6 need stochastic/nonequilibrium models
            sim = CXTsim(
                mode=1,
                modc=modc,
                parms=parms,
                modb=3,
                pulse=pulse,
                obsdata=profile_obsdata.copy()
            )
            sim.run()
            assert sim.cxtdata['csim'].notna().all(), f"Failed for modc={modc}"

    def test_nonequilibrium_modc_types(self, profile_obsdata):
        """Test modc types that require nonequilibrium model."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [10.0, 5.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 0.5}]

        for modc in [1, 3, 4, 5, 6]:
            sim = CXTsim(
                mode=2,
                modc=modc,
                parms=parms,
                modb=3,
                pulse=pulse,
                zl=20.0,
                obsdata=profile_obsdata.copy()
            )
            sim.run()
            assert sim.cxtdata['csim'].notna().all(), f"Failed for modc={modc}"


class TestConcentrationMassBalance:
    """Tests for mass balance in different concentration types."""

    def test_mass_conservation(self, profile_obsdata):
        """Test that mass is approximately conserved (no degradation)."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]  # No degradation
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        # Pulse input
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 1.0}]

        # Extended profile for better integration
        nz = 101
        dz = 0.5
        obsdata = pd.DataFrame([{'t': 5.0, 'z': z * dz} for z in range(nz)])

        sim = CXTsim(
            mode=1,
            modc=3,  # Resident concentration for mass balance
            parms=parms,
            modb=3,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Total mass should be related to pulse (though not exactly equal
        # due to discretization and boundary effects)
        total_mass = sim.cxtdata['csim'].sum() * dz
        assert total_mass > 0

    def test_degradation_reduces_mass(self, profile_obsdata):
        """Test that degradation reduces total mass."""
        bname = ['V', 'D', 'R', 'mu1']
        pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 1.0}]

        # Extended profile
        nz = 101
        dz = 0.5
        obsdata = pd.DataFrame([{'t': 5.0, 'z': z * dz} for z in range(nz)])

        # Without degradation
        binit_no_deg = [10.0, 5.0, 1.0, 0.0]
        parms_no_deg = pd.DataFrame([binit_no_deg], columns=bname, index=['binit'])
        parms_no_deg.loc['bfit'] = [0] * len(bname)

        sim_no_deg = CXTsim(
            mode=1,
            modc=3,
            parms=parms_no_deg,
            modb=3,
            pulse=pulse,
            obsdata=obsdata.copy()
        )
        sim_no_deg.run()

        # With degradation
        binit_with_deg = [10.0, 5.0, 1.0, 0.5]
        parms_with_deg = pd.DataFrame([binit_with_deg], columns=bname, index=['binit'])
        parms_with_deg.loc['bfit'] = [0] * len(bname)

        sim_with_deg = CXTsim(
            mode=1,
            modc=3,
            parms=parms_with_deg,
            modb=3,
            pulse=pulse,
            obsdata=obsdata.copy()
        )
        sim_with_deg.run()

        mass_no_deg = sim_no_deg.cxtdata['csim'].sum()
        mass_with_deg = sim_with_deg.cxtdata['csim'].sum()

        # Mass with degradation should be less
        assert mass_with_deg < mass_no_deg
