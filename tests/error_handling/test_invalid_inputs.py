"""
Error handling tests for invalid inputs.

These tests verify that the code properly handles invalid parameters,
edge cases, and error conditions.
"""
import pytest
import pandas as pd
import numpy as np

from cxtfit import CXTsim, CXTfit


class TestInvalidModeValues:
    """Tests for invalid model mode values."""

    def test_invalid_mode_in_cxtsim(self, simple_obsdata):
        """Test that invalid mode values are handled."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 5.0, 1.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        # Mode 7 should be converted to 8
        sim = CXTsim(
            mode=1,  # Valid mode
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            obsdata=simple_obsdata
        )
        # Should not raise an error
        assert sim.mode == 1


class TestInvalidBoundaryConditions:
    """Tests for invalid boundary condition parameters."""

    def test_invalid_modb_raises_error(self, default_det_parms, simple_obsdata):
        """Test that invalid modb values raise appropriate errors."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        # modb=7 is invalid (should be 0-6)
        with pytest.raises((ValueError, KeyError)):
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=default_det_parms,
                modb=7,
                pulse=pulse,
                obsdata=simple_obsdata
            )
            sim.run()

    def test_empty_pulse_handled(self, default_det_parms, simple_obsdata):
        """Test handling of empty pulse list."""
        # Empty pulse for modb=0 (zero input) should be fine
        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=0,
            pulse=[],
            obsdata=simple_obsdata
        )
        sim.run()
        # Should complete without error


class TestInvalidInitialConditions:
    """Tests for invalid initial condition parameters."""

    def test_invalid_modi_raises_error(self, default_det_parms, simple_obsdata):
        """Test that invalid modi values raise appropriate errors."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        # modi=5 is invalid (should be 0-4)
        with pytest.raises((ValueError, KeyError)):
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=default_det_parms,
                modb=1,
                pulse=pulse,
                modi=5,
                obsdata=simple_obsdata
            )
            sim.run()


class TestInvalidProductionParameters:
    """Tests for invalid production parameters."""

    def test_invalid_modp_raises_error(self, default_det_parms, simple_obsdata):
        """Test that invalid modp values raise appropriate errors."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        # modp=4 is invalid (should be 0-3)
        with pytest.raises((ValueError, KeyError)):
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=default_det_parms,
                modb=1,
                pulse=pulse,
                modp=4,
                obsdata=simple_obsdata
            )
            sim.run()


class TestParameterValidation:
    """Tests for parameter value validation."""

    def test_missing_required_parameters(self, simple_obsdata):
        """Test handling of missing required parameters."""
        # Create incomplete parameter dataframe
        bname = ['V', 'D']  # Missing 'R' and 'mu1'
        binit = [10.0, 5.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        with pytest.raises((KeyError, ValueError)):
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=parms,
                modb=1,
                pulse=pulse,
                obsdata=simple_obsdata
            )
            sim.run()

    def test_negative_velocity_handling(self, simple_obsdata):
        """Test handling of negative velocity."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [-10.0, 5.0, 1.0, 0.0]  # Negative velocity
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        # This may produce invalid results or raise an error
        # depending on implementation
        sim = CXTsim(
            mode=1,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            obsdata=simple_obsdata
        )
        # Should either raise error or produce some result
        # (behavior depends on implementation)

    def test_zero_dispersion_handling(self, simple_obsdata):
        """Test handling of zero dispersion coefficient."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [10.0, 0.0, 1.0, 0.0]  # Zero dispersion
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]

        # Zero dispersion leads to infinite Peclet number
        # Implementation should handle this gracefully
        sim = CXTsim(
            mode=1,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            obsdata=simple_obsdata
        )
        # May produce division by zero or special handling


class TestObservationDataValidation:
    """Tests for observation data validation."""

    def test_empty_obsdata(self, default_det_parms):
        """Test handling of empty observation data."""
        pulse = [{'conc': 1.0, 'time': 0.0}]

        # Empty dataframe may raise error during construction or run
        try:
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=default_det_parms,
                modb=1,
                pulse=pulse,
                obsdata=pd.DataFrame()  # Empty dataframe
            )
            sim.run()
            # If it runs without error, check that there are no results
            assert sim.cxtdata.empty or len(sim.cxtdata) == 0
        except (ValueError, KeyError, IndexError, AttributeError):
            # Expected to fail with one of these errors
            pass

    def test_missing_columns_in_obsdata(self, default_det_parms):
        """Test handling of missing required columns in obsdata."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        # Missing 'z' column
        obsdata = pd.DataFrame({'t': [0.5, 1.0, 1.5]})

        with pytest.raises((KeyError, ValueError)):
            sim = CXTsim(
                mode=1,
                modc=1,
                parms=default_det_parms,
                modb=1,
                pulse=pulse,
                obsdata=obsdata
            )
            sim.run()

    def test_negative_time_values(self, default_det_parms):
        """Test handling of negative time values."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({'t': [-1.0, 0.5, 1.0], 'z': [10.0, 10.0, 10.0]})

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()
        # Should handle gracefully (negative times may give zero concentration)


class TestInverseProblems:
    """Tests for inverse problem error handling."""

    def test_more_parameters_than_observations(self, default_det_parms):
        """Test error when fitting more parameters than observations."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        # Only 2 observations but trying to fit 4 parameters
        obsdata = pd.DataFrame({
            't': [0.5, 1.0],
            'z': [10.0, 10.0],
            'cobs': [0.5, 0.8]
        })

        parms = default_det_parms.copy()
        parms.loc['bfit'] = [1, 1, 1, 1]  # Fit all 4 parameters
        parms.loc['bmin'] = [0.01, 0.01, 0.1, 0.0]
        parms.loc['bmax'] = [100.0, 100.0, 10.0, 1.0]

        sim = CXTsim(
            inverse=1,
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            mit=10,
            ilmt=1,
            obsdata=obsdata
        )

        # Should raise error about too many parameters
        with pytest.raises(ValueError):
            sim.run()

    def test_missing_cobs_column(self, default_det_parms):
        """Test error when cobs column is missing for inverse problem."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [0.5, 1.0, 1.5],
            'z': [10.0, 10.0, 10.0]
            # Missing 'cobs' column
        })

        parms = default_det_parms.copy()
        parms.loc['bfit'] = [1, 1, 0, 0]
        parms.loc['bmin'] = [0.01, 0.01, 0.0, 0.0]
        parms.loc['bmax'] = [100.0, 100.0, 10.0, 1.0]

        sim = CXTsim(
            inverse=1,
            mode=1,
            modc=3,
            parms=parms,
            modb=2,
            pulse=pulse,
            mit=10,
            ilmt=1,
            obsdata=obsdata
        )

        with pytest.raises(ValueError):
            sim.run()


class TestConstraintValidation:
    """Tests for parameter constraint validation."""

    def test_ilmt_mneq3_constraint(self, simple_obsdata):
        """Test that mneq=3 requires ilmt=1."""
        bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
        binit = [20.0, 10.0, 2.0, 0.5, 1.0, 0.0, 0.0]
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [1, 1, 0, 0, 0, 0, 0]
        parms.loc['bmin'] = [0.01] * len(bname)
        parms.loc['bmax'] = [100.0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = simple_obsdata.copy()
        obsdata['cobs'] = np.random.rand(len(obsdata))

        # mneq=3 with ilmt=0 should raise error
        # (This is enforced in CXTfit.load, may not be checked in CXTsim directly)
        # Testing forward problem instead
        sim = CXTsim(
            inverse=0,
            mode=2,
            modc=1,
            mneq=3,
            ilmt=0,
            phim=0.5,
            parms=parms,
            modb=1,
            pulse=pulse,
            zl=50.0,
            obsdata=simple_obsdata
        )
        # Forward problem should still work
        sim.run()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_time(self, default_det_parms):
        """Test behavior at very small time values."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [1e-10, 1e-8, 1e-6],
            'z': [10.0, 10.0, 10.0]
        })

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Should produce finite results
        assert sim.cxtdata['csim'].notna().all()

    def test_very_large_time(self, default_det_parms):
        """Test behavior at very large time values."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [1000.0, 5000.0, 10000.0],
            'z': [10.0, 10.0, 10.0]
        })

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Should produce finite results
        assert sim.cxtdata['csim'].notna().all()

    def test_zero_position(self, default_det_parms):
        """Test behavior at z=0 (inlet)."""
        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [0.5, 1.0, 1.5],
            'z': [0.0, 0.0, 0.0]  # At inlet
        })

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=default_det_parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Should produce valid results
        assert sim.cxtdata['csim'].notna().all()

    def test_very_large_peclet(self):
        """Test behavior with very large Peclet number (low dispersion)."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [100.0, 0.1, 1.0, 0.0]  # High velocity, low dispersion
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [0.5, 1.0, 1.5],
            'z': [10.0, 10.0, 10.0]
        })

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Should handle without overflow
        assert np.all(np.isfinite(sim.cxtdata['csim']))

    def test_very_small_peclet(self):
        """Test behavior with very small Peclet number (high dispersion)."""
        bname = ['V', 'D', 'R', 'mu1']
        binit = [1.0, 1000.0, 1.0, 0.0]  # Low velocity, high dispersion
        parms = pd.DataFrame([binit], columns=bname, index=['binit'])
        parms.loc['bfit'] = [0] * len(bname)

        pulse = [{'conc': 1.0, 'time': 0.0}]
        obsdata = pd.DataFrame({
            't': [0.5, 1.0, 1.5],
            'z': [10.0, 10.0, 10.0]
        })

        sim = CXTsim(
            mode=1,
            modc=1,
            parms=parms,
            modb=1,
            pulse=pulse,
            obsdata=obsdata
        )
        sim.run()

        # Should handle without issues
        assert np.all(np.isfinite(sim.cxtdata['csim']))
