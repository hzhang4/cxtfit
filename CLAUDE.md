# CLAUDE.md - AI Assistant Guide for CXTFIT

## Project Overview

CXTFIT is a Python implementation of the original CXTFIT v2.1 Fortran program for **non-linear least-squares analysis of solute transport** using one-dimensional convective-dispersive equations (CDE). It supports both deterministic and stochastic models for equilibrium and nonequilibrium transport.

**Key Reference**: Toride, N., F. J. Leij, and M. Th. van Genuchten. 1999. "The CXTFIT Code for Estimating Transport Parameters from Laboratory or Field Tracer Experiments, Version 2.1", Research Report No. 137, U.S. Salinity Laboratory, USDA, ARS, Riverside, CA.

**Version**: 1.10
**Author**: Hua Zhang (hzhang4@gmail.com)
**License**: MIT

## Directory Structure

```
cxtfit/
├── cxtfit/                    # Main package
│   ├── __init__.py           # Exports: CXTfit, CXTsim, DetCDE, StoCDE
│   ├── cxtfit.py             # CXTfit class - file I/O and multi-case orchestration
│   ├── cxtsim.py             # CXTsim class - simulation and curve fitting
│   ├── detcde.py             # DetCDE class - deterministic CDE solver
│   └── stocde.py             # StoCDE class - stochastic CDE solver
├── examples/
│   ├── input/                # 20 example input files (FIG4-5.IN through FIG7-15.IN)
│   ├── test_cxtfit.py        # Pytest test suite
│   ├── conftest.py           # Pytest fixtures
│   └── cxtfit_examples.ipynb # Jupyter notebook with usage examples
├── doc/
│   └── cxtfit.pdf            # Original Fortran CXTFIT documentation
├── pyproject.toml            # Project configuration
├── README.md                 # Basic documentation
└── LICENSE.md                # MIT License
```

## Class Hierarchy

```
DetCDE (base mathematical solver)
   ↑
StoCDE (stochastic extensions via log-normal integration)
   ↑
CXTsim (simulation, curve fitting, plotting)

CXTfit (manages multiple CXTsim cases, file I/O)
```

## Key Classes and Their Roles

### CXTfit (cxtfit/cxtfit.py)
Main orchestrator for loading/writing input files and managing multiple simulation cases.

```python
from cxtfit import CXTfit

# Load from file
sims = CXTfit.load('input_file.IN', verbose=False)
sims.run(verbose=False)
sims.write('output_file.IN', verbose=False)

# Access individual cases
for case in sims.simcases:
    case.plot_btc()  # breakthrough curve
```

### CXTsim (cxtfit/cxtsim.py)
Core simulation class for forward and inverse problems.

```python
from cxtfit import CXTsim
import pandas as pd

# Define parameters as DataFrame
bname = ['V', 'D', 'R', 'mu1']
binit = [25.0, 37.5, 3.0, 0.0]
parms = pd.DataFrame([binit], columns=bname, index=['binit'])
parms.loc['bfit'] = [0, 0, 0, 0]  # 0=fixed, 1=fit

# Define observation points
obsdata = pd.DataFrame([{'t': 7.5, 'z': z} for z in range(101)])

# Define input pulse
pulse = [{'conc': 1.0, 'time': 0.0}, {'conc': 0.0, 'time': 5.0}]

# Create and run simulation
sim = CXTsim(
    inverse=0,      # 0=forward, 1=inverse problem
    mode=1,         # Model type (1-8)
    modc=3,         # Concentration type
    parms=parms,
    modb=3,         # Boundary condition
    pulse=pulse,
    obsdata=obsdata
)
sim.run(verbose=False)
sim.plot_profile()  # spatial profile
sim.plot_btc()      # breakthrough curve
```

### DetCDE (cxtfit/detcde.py)
Deterministic CDE solver implementing boundary value problems (BVP), initial value problems (IVP), and production value problems (PVP).

### StoCDE (cxtfit/stocde.py)
Stochastic CDE solver extending DetCDE with ensemble averaging over log-normal distributions using Chebyshev quadrature.

## Mode Codes Reference

| Mode | Description |
|------|-------------|
| 1 | Deterministic equilibrium CDE |
| 2 | Deterministic nonequilibrium (two-region/two-site) |
| 3-8 | Stochastic models with various parameter distributions |

| MODB | Boundary Condition |
|------|-------------------|
| 0 | No boundary |
| 1 | Step input (first-type) |
| 2 | Step input (third-type) |
| 3 | Pulse input (first-type) |
| 4 | Pulse input (third-type) |
| 5 | Multiple pulses (first-type) |
| 6 | Multiple pulses (third-type) |

| MODC | Concentration Type |
|------|-------------------|
| 1 | Resident concentration |
| 2 | Flux concentration |
| 3 | Resident (semi-infinite) |
| 4-6 | Various temporal flux modes |

## Development Commands

### Installation
```bash
pip install .
```

### Running Tests
```bash
# Run all tests
pytest examples/test_cxtfit.py

# Run specific test
pytest examples/test_cxtfit.py::test_fig7_3

# Run with verbose output
pytest examples/test_cxtfit.py -v
```

### Dependencies
- numpy: Numerical computations
- scipy: Optimization (least_squares), integration, statistics
- pandas: DataFrame handling for parameters
- matplotlib: Plotting breakthrough curves and profiles

## Code Conventions

### Parameter DataFrames
Parameters are stored in pandas DataFrames with specific row indices:
- `binit`: Initial parameter values
- `bfit`: Fit flags (0=fixed, 1=fit)
- `bmin`: Lower bounds for fitting
- `bmax`: Upper bounds for fitting

### Input Data Structures
- **pulse**: `list[dict]` with keys `conc`, `time`
- **cini**: `list[dict]` with keys `conc`, `z` (initial conditions)
- **prodval1/prodval2**: `list[dict]` with keys `gamma`, `zpro` (production)
- **obsdata**: pandas DataFrame with columns `t`, `z`, and optionally `cobs`

### Private Methods
Internal configuration methods use double-underscore prefix:
- `__set_parm`, `__set_bvp`, `__set_ivp`, `__set_pvp`, `__set_obs`, `__set_const`

### Variable Naming
Mathematical variables follow the original Fortran naming:
- `v`: velocity
- `d`: dispersion coefficient
- `r`: retardation factor
- `z`: distance
- `t`: time
- `c`: concentration

## Common Tasks

### Adding a New Test Case
1. Create input file in `examples/input/` following existing format
2. Add test function in `examples/test_cxtfit.py`:
```python
def test_new_case(input_path):
    sims = CXTfit.load(f'{input_path}/NEW_CASE.IN', verbose=False)
    sims.run(verbose=False)
    for simcase in sims.simcases:
        simcase.plot_profile()
```

### Creating a Programmatic Simulation
```python
import pandas as pd
from cxtfit import CXTsim

# 1. Define parameters
parms = pd.DataFrame([[50., 20., 1., 0.]],
                     columns=['V', 'D', 'R', 'mu1'],
                     index=['binit'])
parms.loc['bfit'] = [0, 0, 0, 0]

# 2. Define observation grid
obsdata = pd.DataFrame([{'t': 1.0, 'z': z*2.0} for z in range(101)])

# 3. Define boundary conditions
pulse = [{'conc': 1.0, 'time': 0.0}]

# 4. Create and run
sim = CXTsim(mode=1, modc=3, parms=parms, modb=1, pulse=pulse, obsdata=obsdata)
sim.run()
sim.plot_profile()
```

### Inverse Problem (Parameter Estimation)
```python
# Set bfit=1 for parameters to estimate
parms.loc['bfit'] = [1, 1, 0, 0]  # Fit V and D
parms.loc['bmin'] = [0.01, 0.01, 999, 999]
parms.loc['bmax'] = [100.0, 100.0, 999, 999]

# Add observed concentrations to obsdata
obsdata['cobs'] = [...]  # measured values

sim = CXTsim(inverse=1, mode=1, modc=3, parms=parms,
             modb=2, pulse=pulse, mit=150, ilmt=1, obsdata=obsdata)
sim.run()
sim.plot_btc()  # Shows fit vs observed
```

## Important Files

| File | Purpose |
|------|---------|
| `cxtfit/cxtfit.py:1-670` | Main CXTfit class, file parsing, multi-case management |
| `cxtfit/cxtsim.py:1-659` | CXTsim simulation class, curve fitting, plotting |
| `cxtfit/detcde.py:1-892` | Deterministic CDE mathematical solver |
| `cxtfit/stocde.py:1-401` | Stochastic CDE with log-normal integration |
| `examples/test_cxtfit.py` | Comprehensive pytest test suite |
| `examples/cxtfit_examples.ipynb` | Interactive usage examples |
| `doc/cxtfit.pdf` | Original Fortran documentation (reference) |

## Notes for AI Assistants

1. **Scientific Domain**: This is a specialized scientific computing package for solute transport modeling. Understanding the CDE equations and their parameters is helpful for meaningful modifications.

2. **Fortran Heritage**: The code preserves many conventions from the original Fortran implementation. Variable names are terse (single letters) following mathematical notation.

3. **Testing**: The test suite in `examples/test_cxtfit.py` validates against known results from the original Fortran code. Run tests after any modifications.

4. **Documentation**: The PDF in `doc/cxtfit.pdf` contains the complete mathematical background and is essential for understanding the physics.

5. **DataFrames**: Always use pandas DataFrames for parameter definitions with proper row indices (`binit`, `bfit`, `bmin`, `bmax`).

6. **Plotting**: Use `plot_btc()` for breakthrough curves (concentration vs time at fixed location) and `plot_profile()` for spatial profiles (concentration vs distance at fixed time).
