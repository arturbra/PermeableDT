# Core Water Flow API

Complete reference for the core water flow simulation functions in permeabledt.

## Overview

The water flow module (`permeabledt.water_flow_module`) provides the fundamental simulation capabilities for permeable pavement systems. This module implements the three-zone conceptual model with ponding, unsaturated, and saturated zones.

## Core Simulation Functions

### run_simulation()

Run a complete water flow simulation using array inputs.

```python
def run_simulation(params, qin, qrain, emax, dt=60.0, output_time_interval=None)
```

**Parameters:**
- `params` (dict): Complete parameter dictionary from `initialize_parameters()`
- `qin` (array-like): External inflow time series [m³/s]
- `qrain` (array-like): Rainfall time series [m³/s]
- `emax` (array-like): Maximum evapotranspiration time series [m/s]
- `dt` (float, optional): Time step in seconds (default: 60.0)
- `output_time_interval` (int, optional): Output interval in timesteps

**Returns:**
- `data` (dict): Simulation results with all state variables
- `water_balance` (dict): Water balance components and metrics

**Example:**
```python
import permeabledt as pdt
import numpy as np

# Load parameters
setup = pdt.read_setup_file("config.ini")
params = pdt.initialize_parameters(setup)

# Create input time series (24 hours)
time_steps = 1440  # minutes
qin = np.zeros(time_steps)
qrain = np.random.exponential(0.001, time_steps)  # Random rainfall
emax = np.full(time_steps, 1e-6)  # Constant ET

# Run simulation
data, wb = pdt.run_simulation(params, qin, qrain, emax)

print(f"Peak ponding depth: {max(data['hp']):.3f} m")
print(f"Water balance error: {wb['error_percent']:.2f}%")
```

### run_model()

Run simulation from file inputs with automatic data loading.

```python
def run_model(params, rainfall_file, qin_file=None, emax_file=None,
              dt=60.0, output_time_interval=None)
```

**Parameters:**
- `params` (dict): Parameter dictionary
- `rainfall_file` (str): Path to rainfall data file (.dat format)
- `qin_file` (str, optional): Path to external inflow file
- `emax_file` (str, optional): Path to evapotranspiration file
- `dt` (float, optional): Time step in seconds
- `output_time_interval` (int, optional): Output interval

**Returns:**
- `data` (dict): Simulation results
- `water_balance` (dict): Water balance metrics

**Example:**
```python
import permeabledt as pdt

# Load configuration
setup = pdt.read_setup_file("pavement.ini")
params = pdt.initialize_parameters(setup)

# Run from files
data, wb = pdt.run_model(params, "rainfall_event.dat")

# Check results
print(f"Simulation completed with {len(data['time'])} timesteps")
print(f"Peak outflow: {max(data['qpipe']):.6f} m³/s")
```

### run_single_timestep()

Execute a single simulation timestep for custom integration.

```python
def run_single_timestep(params, hp, s, hsz, qin, qrain, emax, dt)
```

**Parameters:**
- `params` (dict): Parameter dictionary
- `hp` (float): Current ponding depth [m]
- `s` (float): Current soil moisture content [-]
- `hsz` (float): Current saturated zone depth [m]
- `qin` (float): External inflow rate [m³/s]
- `qrain` (float): Rainfall rate [m³/s]
- `emax` (float): Maximum ET rate [m/s]
- `dt` (float): Time step [s]

**Returns:**
- `hp_new` (float): Updated ponding depth [m]
- `s_new` (float): Updated soil moisture [-]
- `hsz_new` (float): Updated saturated zone depth [m]
- `fluxes` (dict): All computed fluxes for this timestep

**Example:**
```python
# Custom time stepping
hp, s, hsz = 0.0, 0.3, 0.5  # Initial conditions
dt = 60.0  # 1-minute timestep

for i in range(100):  # 100 timesteps
    hp, s, hsz, fluxes = pdt.run_single_timestep(
        params, hp, s, hsz, qin[i], qrain[i], emax[i], dt
    )

    # Custom logic here
    if hp > 0.1:  # High ponding
        print(f"High ponding at timestep {i}: {hp:.3f} m")
```

## Parameter Functions

### initialize_parameters()

Initialize complete parameter set from setup dictionary.

```python
def initialize_parameters(setup, custom_params=None)
```

**Parameters:**
- `setup` (dict): Setup dictionary from `read_setup_file()`
- `custom_params` (dict, optional): Override specific parameters

**Returns:**
- `params` (dict): Complete parameter dictionary with all computed values

**Key Parameter Groups:**
- **Physical**: Depths, areas, porosities
- **Hydraulic**: Conductivities, moisture parameters
- **Drainage**: Pipe geometry and coefficients
- **Overflow**: Weir parameters
- **Initial**: Starting conditions

**Example:**
```python
setup = pdt.read_setup_file("config.ini")
params = pdt.initialize_parameters(setup)

# Override specific parameters
custom = {"Ks": 1e-4, "gama": 2.0}
params_modified = pdt.initialize_parameters(setup, custom)

print(f"Default Ks: {params['Ks']}")
print(f"Modified Ks: {params_modified['Ks']}")
```

### modify_parameters()

Modify existing parameter dictionary with new values.

```python
def modify_parameters(params, modifications)
```

**Parameters:**
- `params` (dict): Existing parameter dictionary
- `modifications` (dict): Parameter modifications

**Returns:**
- `params` (dict): Updated parameter dictionary

**Example:**
```python
# Modify calibrated parameters
modifications = {
    "Ks": 5e-5,        # Reduce hydraulic conductivity
    "sw": 0.15,        # Adjust wilting point
    "Cd": 0.8          # Change discharge coefficient
}

params = pdt.modify_parameters(params, modifications)
```

## Data I/O Functions

### read_setup_file()

Read INI configuration file with parameter definitions.

```python
def read_setup_file(filename)
```

**Parameters:**
- `filename` (str): Path to INI setup file

**Returns:**
- `setup` (dict): Nested dictionary with all configuration sections

**Example:**
```python
setup = pdt.read_setup_file("pavement_config.ini")

# Access parameter sections
print("Physical parameters:", setup['PHYSICAL'])
print("Hydraulic parameters:", setup['HYDRAULIC'])
print("Drainage parameters:", setup['DRAINAGE'])
```

### read_rainfall_dat_file()

Read rainfall data file in DAT format.

```python
def read_rainfall_dat_file(filename)
```

**Parameters:**
- `filename` (str): Path to rainfall DAT file

**Returns:**
- `time` (list): Time values in minutes
- `rainfall` (list): Rainfall intensities [mm/h]

**File Format:**
```
# Rainfall data file
# Time(min) Rainfall(mm/h)
0.0 0.0
60.0 5.2
120.0 12.8
180.0 8.4
```

**Example:**
```python
time, rainfall = pdt.read_rainfall_dat_file("event_01.dat")
print(f"Duration: {max(time)} minutes")
print(f"Peak intensity: {max(rainfall)} mm/h")
```

### results_dataframe()

Convert simulation results to pandas DataFrame for analysis.

```python
def results_dataframe(data, time_unit='minutes')
```

**Parameters:**
- `data` (dict): Simulation results from `run_simulation()`
- `time_unit` (str): Time unit for output ('minutes', 'hours', 'days')

**Returns:**
- `df` (pandas.DataFrame): Results with time index

**Example:**
```python
import pandas as pd

data, wb = pdt.run_simulation(params, qin, qrain, emax)
df = pdt.results_dataframe(data, time_unit='hours')

# Analyze results
print(df.describe())
df.plot(y=['hp', 'qpipe'], secondary_y='qpipe')
```

### load_input_files()

Load multiple input files for simulation.

```python
def load_input_files(rainfall_file, qin_file=None, emax_file=None)
```

**Parameters:**
- `rainfall_file` (str): Rainfall data file
- `qin_file` (str, optional): External inflow file
- `emax_file` (str, optional): Evapotranspiration file

**Returns:**
- `inputs` (dict): Dictionary with all loaded time series

**Example:**
```python
inputs = pdt.load_input_files(
    rainfall_file="rainfall.dat",
    qin_file="inflow.dat",
    emax_file="et.dat"
)

print(f"Data length: {len(inputs['qrain'])} timesteps")
```

## Water Balance Functions

### calculate_water_balance()

Compute detailed water balance for simulation results.

```python
def calculate_water_balance(data, params, inputs)
```

**Parameters:**
- `data` (dict): Simulation results
- `params` (dict): Parameter dictionary
- `inputs` (dict): Input time series

**Returns:**
- `water_balance` (dict): Complete water balance with:
  - `input_volume`: Total rainfall + inflow [m³]
  - `output_volume`: Total outflow + overflow + ET [m³]
  - `storage_change`: Change in system storage [m³]
  - `error_volume`: Mass balance error [m³]
  - `error_percent`: Relative error [%]

**Example:**
```python
wb = pdt.calculate_water_balance(data, params, inputs)

print(f"Input volume: {wb['input_volume']:.3f} m³")
print(f"Output volume: {wb['output_volume']:.3f} m³")
print(f"Storage change: {wb['storage_change']:.3f} m³")
print(f"Balance error: {wb['error_percent']:.2f}%")
```

## Data Processing Functions

### rainfall_data_treatment()

Process and validate rainfall data for simulation.

```python
def rainfall_data_treatment(time, rainfall, dt=60.0, interpolate=True)
```

**Parameters:**
- `time` (array-like): Time values [minutes]
- `rainfall` (array-like): Rainfall intensities [mm/h]
- `dt` (float): Target time step [seconds]
- `interpolate` (bool): Whether to interpolate missing values

**Returns:**
- `time_regular` (array): Regular time grid
- `rainfall_regular` (array): Interpolated rainfall data
- `qrain` (array): Rainfall flow rates [m³/s]

**Example:**
```python
# Load irregular rainfall data
time, rainfall = pdt.read_rainfall_dat_file("irregular_data.dat")

# Process to regular 5-minute intervals
time_reg, rain_reg, qrain = pdt.rainfall_data_treatment(
    time, rainfall, dt=300.0, interpolate=True
)

print(f"Original points: {len(time)}")
print(f"Regularized points: {len(time_reg)}")
```

## Utility Functions

### get_parameter_bounds()

Get reasonable bounds for parameters (used in calibration).

```python
def get_parameter_bounds(param_names)
```

**Parameters:**
- `param_names` (list): List of parameter names to get bounds for

**Returns:**
- `bounds` (dict): Dictionary with 'lower' and 'upper' bound arrays

**Example:**
```python
# Get bounds for calibration parameters
params_to_calibrate = ["Ks", "gama", "sw", "sfc", "Cd"]
bounds = pdt.get_parameter_bounds(params_to_calibrate)

print("Lower bounds:", bounds['lower'])
print("Upper bounds:", bounds['upper'])
```

### validate_inputs()

Validate simulation inputs for consistency and physical realism.

```python
def validate_inputs(params, qin, qrain, emax, dt)
```

**Parameters:**
- `params` (dict): Parameter dictionary
- `qin` (array-like): External inflow
- `qrain` (array-like): Rainfall
- `emax` (array-like): Evapotranspiration
- `dt` (float): Time step

**Returns:**
- `is_valid` (bool): Whether inputs are valid
- `warnings` (list): List of warning messages

**Example:**
```python
is_valid, warnings = pdt.validate_inputs(params, qin, qrain, emax, dt)

if not is_valid:
    print("Validation failed!")
    for warning in warnings:
        print(f"Warning: {warning}")
```

## State Variables

The simulation tracks three primary state variables:

### hp - Ponding Depth [m]
Surface water depth above the pavement surface.
- Range: 0 to overflow height
- Physical meaning: Temporary surface storage

### s - Soil Moisture Content [-]
Volumetric moisture content in the unsaturated zone.
- Range: hygroscopic point to saturation
- Physical meaning: Water storage in pore spaces

### hsz - Saturated Zone Depth [m]
Depth of water in the saturated zone.
- Range: 0 to total system depth
- Physical meaning: Groundwater storage

## Derived Variables

Computed from state variables and parameters:

### husz - Unsaturated Zone Depth [m]
```python
husz = params['L'] - hsz
```

### nsz - Saturated Zone Porosity [-]
Computed using porosity function based on layer structure.

### nusz - Unsaturated Zone Porosity [-]
Computed using porosity function based on layer structure.

## Flux Variables

All fluxes are computed at each timestep:

### Ponding Zone Fluxes
- `qinfp`: Infiltration from ponding [m³/s]
- `qover`: Overflow [m³/s]
- `qetp`: Evaporation from ponding [m³/s]

### Unsaturated Zone Fluxes
- `qfs`: Flow to saturated zone [m³/s]
- `qhc`: Plant water uptake [m³/s]
- `qpf`: Lateral flow [m³/s]

### Saturated Zone Fluxes
- `qpipe`: Pipe drainage [m³/s]
- `qinfsz`: Lateral infiltration [m³/s]

## Error Handling

All functions include comprehensive error checking:

### Parameter Validation
```python
try:
    params = pdt.initialize_parameters(setup)
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

### File I/O Errors
```python
try:
    data, wb = pdt.run_model(params, "rainfall.dat")
except FileNotFoundError:
    print("Rainfall file not found")
```

### Numerical Issues
```python
if wb['error_percent'] > 5.0:
    print("Warning: High water balance error")
    print("Consider reducing time step")
```

## Performance Considerations

### Time Step Selection
- Smaller time steps increase accuracy but slow computation
- Typical range: 10-300 seconds
- Adaptive time stepping available in advanced functions

### Memory Usage
- Output arrays grow with simulation length
- Use `output_time_interval` to reduce memory usage
- Consider data streaming for very long simulations

### Numerical Stability
- Model includes stability checks
- Automatic time step reduction for numerical issues
- Mass conservation enforced at each timestep

## Integration with Other Modules

### Calibration
```python
# Use in optimization
result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="outflow.csv"
)
```

### Particle Filtering
```python
# Real-time forecasting
model = pdt.PavementModel("config.ini", "forecast.dat")
```

### Sensitivity Analysis
```python
# Parameter importance
sa = pdt.SobolSensitivityAnalysis("config.ini", "data.dat")
results = sa.run_analysis()
```

This water flow API provides the foundation for all permeabledt modeling capabilities, with robust error handling, comprehensive validation, and integration with advanced features.