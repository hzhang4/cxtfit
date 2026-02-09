"""
Shared pytest fixtures for CXTFIT tests.
"""
import pytest
import os
import pandas as pd
import numpy as np


@pytest.fixture
def input_path():
    """Get the absolute path to the example input files folder."""
    return os.path.join(os.path.dirname(__file__), '..', 'examples', 'input')


@pytest.fixture
def default_det_parms():
    """Default parameters for deterministic equilibrium model (mode=1)."""
    bname = ['V', 'D', 'R', 'mu1']
    binit = [10.0, 5.0, 1.0, 0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)
    parms.loc['bmin'] = [0.0] * len(bname)
    parms.loc['bmax'] = [100.0] * len(bname)
    return parms


@pytest.fixture
def default_det_noneq_parms():
    """Default parameters for deterministic nonequilibrium model (mode=2)."""
    bname = ['V', 'D', 'R', 'beta', 'omega', 'mu1', 'mu2']
    binit = [10.0, 5.0, 2.0, 0.5, 1.0, 0.0, 0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)
    parms.loc['bmin'] = [0.0] * len(bname)
    parms.loc['bmax'] = [100.0] * len(bname)
    return parms


@pytest.fixture
def default_stoch_eq_parms():
    """Default parameters for stochastic equilibrium models (mode=3,5)."""
    bname = ['<V>', '<D>', '<Kd>', 'mu1', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
    binit = [50.0, 20.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)
    parms.loc['bmin'] = [0.0] * len(bname)
    parms.loc['bmax'] = [100.0] * len(bname)
    return parms


@pytest.fixture
def default_stoch_noneq_parms():
    """Default parameters for stochastic nonequilibrium models (mode=4,6)."""
    bname = ['<V>', '<D>', '<Kd>', 'omega', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'RhovKd']
    binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)
    parms.loc['bmin'] = [0.0] * len(bname)
    parms.loc['bmax'] = [100.0] * len(bname)
    return parms


@pytest.fixture
def default_stoch_alpha_parms():
    """Default parameters for stochastic alpha-v nonequilibrium model (mode=8)."""
    bname = ['<V>', '<D>', '<Kd>', 'alpha', 'mu1', 'mu2', 'SD.v', 'SD.Kd', 'SD.D', 'SD.alp', 'RhovAl']
    binit = [50.0, 20.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
    parms = pd.DataFrame([binit], columns=bname, index=['binit'])
    parms.loc['bfit'] = [0] * len(bname)
    parms.loc['bmin'] = [0.0] * len(bname)
    parms.loc['bmax'] = [100.0] * len(bname)
    return parms


@pytest.fixture
def simple_obsdata():
    """Simple observation data for testing."""
    nt = 21
    dt = 0.5
    return pd.DataFrame([{'t': t * dt, 'z': 10.0} for t in range(nt)])


@pytest.fixture
def profile_obsdata():
    """Profile observation data for testing."""
    nz = 21
    dz = 1.0
    return pd.DataFrame([{'t': 1.0, 'z': z * dz} for z in range(nz)])
