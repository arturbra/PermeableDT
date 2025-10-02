# PermeableDT

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PermeableDT** is a comprehensive Python library for permeable pavement digital twin modeling, featuring water flow simulation, genetic algorithm calibration, particle filtering, and weather data acquisition capabilities.

## Features

- 🌊 **Water Flow Modeling**: Physics-based simulation of permeable pavement systems
- 🧬 **Genetic Algorithm Calibration**: Automated parameter optimization using DEAP
- 📊 **Particle Filtering**: Real-time state estimation and uncertainty quantification
- 🌦️ **Weather Data Integration**: HRRR forecast data downloading and processing
- 📊 **Visualization**: Built-in plotting functions for results analysis

## Installation

### Requirements
- **Python** ≥ 3.8  
- Base dependencies are installed automatically (`numpy`, `pandas`).  
- Optional feature groups (install with extras):
  - `calib` → genetic algorithm calibration (`deap`, `scikit-learn`)
  - `pf` → particle filtering (`pypfilt`, `scipy`, `tomlkit`)
  - `plots` → plotting (`matplotlib`)
  - `weather` → HRRR downloads (`herbie-data`, `xarray`, `pytz`)
  - `all` → everything above

---

### Install from PyPI (recommended)

```bash
pip install permeabledt
# or with optional features:
pip install "permeabledt[all]"
# examples:
pip install "permeabledt[calib]"
pip install "permeabledt[pf]"
pip install "permeabledt[plots]"
pip install "permeabledt[weather]"

### Manual Installation

```bash
git clone https://github.com/arturbra/permeabledt.git
cd permeabledt
pip install -e .
```

## Support

- 📧 Email: jose.brasil@utsa.edu
- 🐛 Issues: [GitHub Issues](https://github.com/arturbra/permeabledt/issues)
- 📖 Documentation: [Documentation](https://permeabledt/readthedocs.io)

## Acknowledgments

- DEAP library for genetic algorithms
- pypfilt library for particle filtering
- herbie-data for weather data access