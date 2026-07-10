# Installation

## Requirements

- Python 3.10 or newer
- NumPy, pandas, SciPy, and Matplotlib

## Pip installation from GitHub

```bash
git clone https://github.com/hamiddi/pygdis.git
cd pygdis
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Conda installation

```bash
git clone https://github.com/hamiddi/pygdis.git
cd pygdis
conda env create -f environment.yml
conda activate pygdis
python -m pip install -e .
```

## Verify installation

```bash
python -c "import gdis; print(gdis.__version__)"
gdis --help
pytest -q
```

## Build distribution files

```bash
python -m pip install build
python -m build
```

The wheel and source distribution will be written to `dist/`.
