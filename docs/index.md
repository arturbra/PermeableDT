# permeabledt Documentation

**Digital Twin Tools for Permeable Pavement Water Flow Modeling**

Welcome to the comprehensive documentation for PermeableDT, a Python library for permeable pavement digital twin modeling, featuring water flow simulation, genetic algorithm calibration, particle filtering, and weather data acquisition capabilities.

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
data_preparation
```

```{toctree}
:maxdepth: 2
:caption: Technical Documentation

technical/model_theory
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/water_flow
api/calibration
api/particle_filtering
api/weather_data
api/plotting
```

```{toctree}
:maxdepth: 2
:caption: Examples and Tutorials

examples/basic_examples
```

## Features Overview

### 🌊 **Water Flow Modeling**
Physics-based simulation of permeable pavement systems with three-zone modeling:
- Ponding zone with overflow
- Unsaturated zone with evapotranspiration
- Saturated zone with pipe drainage

### 🧬 **Genetic Algorithm Calibration**
Automated parameter optimization using DEAP library:
- Multi-objective optimization
- Parallel processing support
- Comprehensive parameter bounds

### 📊 **Particle Filtering**
Real-time state estimation and uncertainty quantification:
- Probabilistic forecasting
- Uncertainty propagation
- Real-time data assimilation

### 🌦️ **Weather Data Integration**
HRRR forecast data downloading and processing:
- Automated data retrieval
- Format conversion
- Historical and real-time data

### 📊 **Visualization**
Built-in plotting functions for results analysis:
- Rainfall-hydrograph plots
- Calibration convergence
- Particle filter results

## Quick Navigation

| Topic | Description | Key Files |
|-------|-------------|-----------|
| **Installation** | Get permeabledt running | [installation.md](installation.md) |
| **Basic Usage** | Core functionality | [quickstart.md](quickstart.md) |
| **Calibration** | Parameter optimization | [user_guide/calibration.md](user_guide/calibration.md) |
| **Particle Filter** | Real-time forecasting | [user_guide/particle_filtering.md](user_guide/particle_filtering.md) |
| **API Reference** | Function documentation | [api/index.md](api/index.md) |
| **Examples** | Working code samples | [examples/](examples/) |

## Getting Help

- 📖 **Documentation**: You're reading it!
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/permeabledt/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/permeabledt/discussions)
- 📧 **Email**: Contact the development team


## License

PermeableDT is released under the MIT License. See [LICENSE](../LICENSE) for details.