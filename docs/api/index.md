# API Reference

Complete reference documentation for all permeabledt functions, classes, and modules.

## Overview

permeabledt provides a comprehensive API for permeable pavement modeling with the following main modules:

- **[Core Water Flow](water_flow.md)** - Basic simulation functions
- **[Calibration](calibration.md)** - Parameter optimization
- **[Particle Filtering](particle_filtering.md)** - Real-time forecasting
- **[Weather Data](weather_data.md)** - HRRR data integration
- **[Plotting](plotting.md)** - Visualization functions

## Quick Reference

### Core Functions

| Function | Description | Module |
|----------|-------------|---------|
| `run_simulation()` | Run water flow simulation with arrays | `permeabledt` |
| `run_model()` | Run simulation from files | `permeabledt` |
| `initialize_parameters()` | Load parameters from setup file | `permeabledt` |
| `read_setup_file()` | Read INI configuration file | `permeabledt` |
| `calculate_water_balance()` | Compute water balance metrics | `permeabledt` |

### Calibration Functions

| Function | Description | Module |
|----------|-------------|---------|
| `run_calibration()` | Modern calibration interface | `permeabledt` |
| `calibrate()` | Legacy calibration method | `permeabledt` |

### Particle Filter Classes

| Class | Description | Module |
|-------|-------------|---------|
| `PavementModel` | pypfilt.Model implementation | `permeabledt.particle_filter` |
| `PipeObs` | pypfilt.obs.Univariate implementation | `permeabledt.particle_filter` |

### Analysis Classes

## Module Structure

```
permeabledt/
├── __init__.py                          # Main package interface
├── water_flow_module.py                 # Core simulation functions
├── calibration.py                       # Genetic algorithm optimization
├── particle_filter.py                   # Particle filtering classes
├── download_HRRR_historical_forecast.py # Weather data integration
└── plots.py                            # Visualization functions
```

## Import Patterns

### Basic Usage
```python
import permeabledt as pdt

# Core functions are available directly
data, wb = pdt.run_simulation(params, qin, qrain, emax)
setup = pdt.read_setup_file("config.ini")
params = pdt.initialize_parameters(setup)
```

### Specialized Classes
```python
import permeabledt as pdt

# Particle filtering
model = pdt.PavementModel(setup_file, rainfall_file)
obs = pdt.PipeObs(observed_file)

```

### Plotting Module
```python
import permeabledt as pdt

# Plotting functions (if matplotlib available)
if pdt.plots is not None:
    pdt.plots.plot_rainfall_hydrograph(rainfall_file, outflow_data)
```

## Function Categories

### Data I/O Functions
- `read_setup_file()` - Read INI configuration files
- `read_rainfall_dat_file()` - Read rainfall data files
- `results_dataframe()` - Convert results to pandas DataFrame
- `load_input_files()` - Load simulation input files

### Simulation Functions
- `run_simulation()` - Core simulation with arrays
- `run_model()` - File-based simulation
- `run_single_timestep()` - Single timestep execution
- `calculate_water_balance()` - Water balance computation

### Parameter Functions
- `initialize_parameters()` - Parameter initialization
- `modify_parameters()` - Parameter modification
- `rainfall_data_treatment()` - Rainfall data processing

### Calibration Functions
- `run_calibration()` - Modern calibration interface
- `calibrate()` - Legacy calibration method

## Error Handling

All permeabledt functions include comprehensive error handling:

### Common Exceptions
- `FileNotFoundError` - Missing input files
- `ValueError` - Invalid parameter values
- `RuntimeError` - Missing optional dependencies
- `KeyError` - Missing configuration parameters

### Example Error Handling
```python
try:
    data, wb = pdt.run_model(params, "rainfall.dat")
except FileNotFoundError as e:
    print(f"Input file not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except RuntimeError as e:
    print(f"Missing dependency: {e}")
```

## Optional Dependencies

Some functions require optional dependencies:

### Calibration Functions
Require: `deap`, `scikit-learn`
```python
try:
    result = pdt.run_calibration(...)
except RuntimeError as e:
    print("Install with: pip install permeabledt[calib]")
```

### Particle Filter Classes
Require: `pypfilt`, `scipy`, `tomlkit`
```python
if pdt.PavementModel is not None:
    model = pdt.PavementModel(...)
else:
    print("Install with: pip install permeabledt[pf]")
```

### Weather Data
Requires: `herbie-data`, `xarray`, `pytz`
```python
if pdt.HRRRAccumulatedPrecipitationDownloader is not None:
    downloader = pdt.HRRRAccumulatedPrecipitationDownloader(...)
else:
    print("Install with: pip install permeabledt[weather]")
```

### Plotting
Requires: `matplotlib`
```python
if pdt.plots is not None:
    pdt.plots.plot_rainfall_hydrograph(...)
else:
    print("Install with: pip install permeabledt[plots]")
```

## Version Information

```python
import permeabledt as pdt
print(f"permeabledt version: {pdt.__version__}")
```

## Detailed Module Documentation

- **[Core Water Flow API](water_flow.md)** - Complete water flow function reference
- **[Calibration API](calibration.md)** - Parameter optimization functions
- **[Particle Filter API](particle_filtering.md)** - Real-time forecasting classes
- **[Weather Data API](weather_data.md)** - HRRR data integration
- **[Plotting API](plotting.md)** - Visualization functions