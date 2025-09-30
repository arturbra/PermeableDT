# API Reference

Complete reference documentation for all permeabledt functions, classes, and modules.

## Overview

permeabledt provides a comprehensive API for permeable pavement modeling with the following main modules:

- **[Core Water Flow](water_flow.md)** - Basic simulation functions
- **[Calibration](calibration.md)** - Parameter optimization
- **[Particle Filtering](particle_filtering.md)** - Real-time forecasting
- **[Sensitivity Analysis](sensitivity_analysis.md)** - Parameter importance
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

| Class | Description | Module |
|-------|-------------|---------|
| `SobolSensitivityAnalysis` | Global sensitivity analysis | `permeabledt.sensitivity_analysis` |
| `HRRRAccumulatedPrecipitationDownloader` | Weather data downloader | `permeabledt.download_HRRR_historical_forecast` |

## Module Structure

```
permeabledt/
├── __init__.py                          # Main package interface
├── water_flow_module.py                 # Core simulation functions
├── calibration.py                       # Genetic algorithm optimization
├── particle_filter.py                   # Particle filtering classes
├── sensitivity_analysis.py              # Sensitivity analysis tools
├── download_HRRR_historical_forecast.py # Weather data integration
└── plots.py                            # Visualization functions
```

## Import Patterns

### Basic Usage
```python
import permeabledt as gdt

# Core functions are available directly
data, wb = gdt.run_simulation(params, qin, qrain, emax)
setup = gdt.read_setup_file("config.ini")
params = gdt.initialize_parameters(setup)
```

### Specialized Classes
```python
import permeabledt as gdt

# Particle filtering
model = gdt.PavementModel(setup_file, rainfall_file)
obs = gdt.PipeObs(observed_file)

# Sensitivity analysis
sa = gdt.SobolSensitivityAnalysis(setup_file, rainfall_file)

# Weather data
downloader = gdt.HRRRAccumulatedPrecipitationDownloader(lat, lon)
```

### Plotting Module
```python
import permeabledt as gdt

# Plotting functions (if matplotlib available)
if gdt.plots is not None:
    gdt.plots.plot_rainfall_hydrograph(rainfall_file, outflow_data)
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
    data, wb = gdt.run_model(params, "rainfall.dat")
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
    result = gdt.run_calibration(...)
except RuntimeError as e:
    print("Install with: pip install permeabledt[calib]")
```

### Particle Filter Classes
Require: `pypfilt`, `scipy`, `tomlkit`
```python
if gdt.PavementModel is not None:
    model = gdt.PavementModel(...)
else:
    print("Install with: pip install permeabledt[pf]")
```

### Sensitivity Analysis
Requires: `SALib`, `tqdm`
```python
if gdt.SobolSensitivityAnalysis is not None:
    sa = gdt.SobolSensitivityAnalysis(...)
else:
    print("Install with: pip install permeabledt[sensitivity]")
```

### Weather Data
Requires: `herbie-data`, `xarray`, `pytz`
```python
if gdt.HRRRAccumulatedPrecipitationDownloader is not None:
    downloader = gdt.HRRRAccumulatedPrecipitationDownloader(...)
else:
    print("Install with: pip install permeabledt[weather]")
```

### Plotting
Requires: `matplotlib`
```python
if gdt.plots is not None:
    gdt.plots.plot_rainfall_hydrograph(...)
else:
    print("Install with: pip install permeabledt[plots]")
```

## Version Information

```python
import permeabledt as gdt
print(f"permeabledt version: {gdt.__version__}")
```

## Detailed Module Documentation

- **[Core Water Flow API](water_flow.md)** - Complete water flow function reference
- **[Calibration API](calibration.md)** - Parameter optimization functions
- **[Particle Filter API](particle_filtering.md)** - Real-time forecasting classes
- **[Sensitivity Analysis API](sensitivity_analysis.md)** - Parameter analysis tools
- **[Weather Data API](weather_data.md)** - HRRR data integration
- **[Plotting API](plotting.md)** - Visualization functions