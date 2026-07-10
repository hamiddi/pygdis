<p align="center">
  <img src="docs/images/pygdis_logo.png" width="120">
</p>

# pyGDIS

### Generalized Dynamical Instability Score (GDIS)

*A Python package for universal instability quantification, critical transition detection, and chaos analysis.*

---
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-README-blue)](README.md)
[![GitHub Stars](https://img.shields.io/github/stars/hamiddi/pygdis?style=social)](https://github.com/hamiddi/pygdis)
---

## Authors

- **Hamid D. Ismail**
- **Ahmed Harb**
- **Marwan Bikdash**

## Overview

**pyGDIS** is an open-source Python package implementing the **Generalized Dynamical Instability Score (GDIS)**, a bounded, physics-informed score for detecting and quantifying instability, critical transitions, and chaotic behavior in nonlinear dynamical systems.


## What GDIS is Used For

The **Generalized Dynamical Instability Score (GDIS)** is a universal framework for quantifying instability and identifying critical dynamical transitions in nonlinear systems. Designed to be independent of any specific mathematical model, GDIS provides a normalized instability score that enables consistent comparison across different dynamical systems and datasets.

GDIS can be used to:

- **Quantify dynamical instability** using a bounded score between 0 (fully stable) and 1 (highly unstable).
- **Identify stable, near-critical, and chaotic regimes** from parameter sweeps or time-series trajectories.
- **Detect bifurcations and critical transitions** before large-scale qualitative changes in system behavior occur.
- **Provide early-warning indicators** for the onset of instability in complex nonlinear systems.
- **Separate sustained instability from transient transition effects**, allowing continuous instability and localized transition dynamics to be analyzed independently.
- **Compare instability across different systems** using a common, dimensionless metric.
- **Validate instability predictions** against established measures such as Lyapunov exponents, finite-time Lyapunov exponents (FTLE), bifurcation diagrams, recurrence analysis, entropy measures, and other nonlinear dynamical indicators.
- **Analyze both model-based simulations and experimental or observational data**, requiring only ordered trajectories generated along a control parameter or time evolution.

### Applications

GDIS is applicable to a wide range of nonlinear dynamical systems, including:

- Chaotic benchmark systems (Lorenz, Rössler, Chen, Logistic map, and others)
- Phase-Locked Loops (PLLs) and synchronization systems
- Electric power systems and voltage stability analysis
- Nonlinear electronic circuits
- Mechanical and structural vibration systems
- Robotics and autonomous control systems
- Fluid dynamics and turbulence
- Climate and Earth system dynamics
- Ecological and population dynamics
- Financial and economic time-series analysis
- Biological systems and gene regulatory networks
- Single-cell transcriptomics and cell-state transition analysis
- Biomedical systems and disease progression modeling
- Any parameterized nonlinear dynamical system exhibiting critical transitions or chaotic behavior

## Mathematical definition

The GDIS potential is

$$
\Phi(p)=
-\log\left(1-I_{\mathrm{sustained}}(p)\right)
+
I_{\mathrm{transition}}(p)
$$

and the final bounded score is

$$
\mathrm{GDIS}(p)=1-\exp\left[-\Phi(p)\right],
\qquad 0 \leq \mathrm{GDIS}(p) \leq 1.
$$

The sustained term combines local Jacobian instability, trajectory stretching, attractor expansion, entropy complexity, and temporal persistence. The transition term measures parameter-dependent changes in the physical channels near a critical region.

## Repository structure

```text
pygdis/
├── src/gdis/            # Package source code
├── src/gdis/systems/    # Built-in benchmark systems
├── examples/            # Example programs
├── tests/               # Unit tests
├── docs/                # Installation, API, mathematics, and usage
├── environment.yml      # Conda environment
├── pyproject.toml       # Python packaging configuration
├── CITATION.cff         # Citation metadata
└── LICENSE              # MIT license
```

## Installation with pip

### From a cloned repository

```bash
git clone https://github.com/hamiddi/pygdis.git
cd pygdis
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

### From PyPI after release

```bash
pip install pygdis
```

## Installation with Anaconda or Miniconda

```bash
git clone https://github.com/hamiddi/pygdis.git
cd pygdis
conda env create -f environment.yml
conda activate pygdis
python -m pip install -e .
pytest -q
```

A manual environment can also be created:

```bash
conda create -n pygdis python=3.11 numpy pandas scipy matplotlib pytest pip -y
conda activate pygdis
python -m pip install -e .
```

## Short example: built-in Lorenz system

```python
from gdis import GDIS
from gdis.systems import LorenzSystem

system = LorenzSystem()
trajectories, rho = system.generate_sweep()

result = GDIS().fit_transform(
    trajectories=trajectories,
    parameters=rho,
    jacobian_function=system.jacobian,
    critical_value=system.critical_value,
)

print(result.to_dataframe().head())
result.plot()
```

## Data-only example

```python
from gdis import GDIS

result = GDIS().fit_transform(
    trajectories=trajectories,
    parameters=control_parameter_values,
)

print(result.gdis)
```

When a Jacobian is unavailable, pyGDIS uses a data-driven local divergence proxy.

## Command-line interface

Analyze a long-format CSV file:

```bash
gdis analyze data.csv \
  --parameter rho \
  --state-columns x y z \
  --time time \
  --output results
```

Run built-in benchmarks:

```bash
gdis benchmark --output benchmark_results
```

## Documentation

- [Installation guide](docs/INSTALLATION.md)
- [Usage guide](docs/USAGE.md)
- [Applications](docs/APPLICATIONS.md)
- [Mathematical formulation](docs/MATHEMATICS.md)
- [API reference](docs/API.md)
- [Contributing](CONTRIBUTING.md)

## Citation

Please cite the associated GDIS paper and this software. Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

This project is released under the MIT License.
