# pyGDIS

**pyGDIS** is an open-source Python package implementing the **Generalized Dynamical Instability Score (GDIS)**, a bounded, physics-informed score for detecting and quantifying instability, critical transitions, and chaotic behavior in nonlinear dynamical systems.

## Authors

- **Hamid D. Ismail**
- **Ahmed Harb**
- **Marwan Bikdash**

## What GDIS is used for

GDIS is intended for ordered collections of trajectories generated while a control parameter changes. It can be used to:

- identify stable, near-critical, and unstable operating regimes;
- detect bifurcation and transition regions;
- compare instability across nonlinear systems using a bounded score;
- evaluate sustained instability separately from localized transition energy;
- validate instability against Lyapunov exponents, bifurcation diagrams, and other references;
- analyze equation-based simulations and data-only experimental trajectories.

Potential applications include chaotic benchmark systems, phase-locked loops, power-system stability, nonlinear circuits, mechanical oscillators, climate and ecological dynamics, and biological or transcriptomic state transitions.

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
