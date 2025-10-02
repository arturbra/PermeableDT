# Calibration API

Complete reference for parameter calibration and optimization functions in PermeableDT.

## Overview

The calibration module (`permeabledt.calibration`) provides genetic algorithm-based parameter optimization for permeable pavement models. It uses the DEAP (Distributed Evolutionary Algorithms in Python) library to efficiently search the parameter space and find optimal parameter sets.

## Installation Requirements

Calibration functions require additional dependencies:

```bash
# Install with calibration support
pip install permeabledt[calib]

# Or install dependencies manually
pip install deap scikit-learn
```

## Core Calibration Functions

### run_calibration()

Modern interface for parameter calibration with comprehensive options.

```python
def run_calibration(setup_file, rainfall_file, observed_file,
                   parameters_to_calibrate=None, bounds=None,
                   population_size=50, generations=20, crossover_prob=0.7,
                   mutation_prob=0.2, tournament_size=3, random_seed=None,
                   objective_function='nse', weights=None,
                   output_dir=None, verbose=True)
```

**Parameters:**

**Input Files:**
- `setup_file` (str): Path to INI configuration file
- `rainfall_file` (str): Path to rainfall data file (.dat format)
- `observed_file` (str): Path to observed outflow data (.csv format)

**Calibration Parameters:**
- `parameters_to_calibrate` (list, optional): Parameter names to optimize
  - Default: `['Ks', 'gama', 'sw', 'sfc', 'Cd']`
- `bounds` (dict, optional): Custom parameter bounds
  - Format: `{'lower': [min_vals], 'upper': [max_vals]}`

**Genetic Algorithm Settings:**
- `population_size` (int): Number of individuals in population (default: 50)
- `generations` (int): Number of generations to evolve (default: 20)
- `crossover_prob` (float): Crossover probability (default: 0.7)
- `mutation_prob` (float): Mutation probability (default: 0.2)
- `tournament_size` (int): Tournament selection size (default: 3)
- `random_seed` (int, optional): Random seed for reproducibility

**Objective Function:**
- `objective_function` (str): Optimization metric
  - Options: 'nse', 'rmse', 'mae', 'r2', 'kge', 'multi'
- `weights` (list, optional): Weights for multi-objective optimization

**Output:**
- `output_dir` (str, optional): Directory to save results
- `verbose` (bool): Print progress information (default: True)

**Returns:**
- `result` (dict): Calibration results containing:
  - `best_params`: Optimal parameter values
  - `best_fitness`: Best objective function value
  - `convergence_history`: Fitness evolution over generations
  - `final_population`: All individuals in final generation
  - `statistics`: Calibration statistics

**Example - Basic Calibration:**
```python
import permeabledt as pdt

# Basic calibration with default settings
result = pdt.run_calibration(
    setup_file="pavement.ini",
    rainfall_file="event_data.dat",
    observed_file="observed_outflow.csv"
)

print(f"Best NSE: {result['best_fitness']:.4f}")
print("Optimal parameters:")
for param, value in result['best_params'].items():
    print(f"  {param}: {value:.6f}")
```

**Example - Custom Calibration:**
```python
# Custom parameter selection and bounds
params_to_cal = ['Ks', 'gama', 'Cd', 'eta']
custom_bounds = {
    'lower': [1e-6, 1.0, 0.5, 0.0],
    'upper': [1e-3, 5.0, 1.0, 1.0]
}

result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="rainfall.dat",
    observed_file="outflow.csv",
    parameters_to_calibrate=params_to_cal,
    bounds=custom_bounds,
    population_size=100,
    generations=50,
    objective_function='kge',
    output_dir="calibration_results",
    random_seed=42
)
```

**Example - Multi-objective Calibration:**
```python
# Multi-objective optimization
result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="obs.csv",
    objective_function='multi',
    weights=[0.7, 0.3],  # 70% NSE, 30% volume error
    population_size=80,
    generations=30
)
```

### calibrate()

Legacy calibration interface (maintained for backward compatibility).

```python
def calibrate(setup_file, rainfall_file, observed_file,
             output_folder, param_names=None, param_bounds=None,
             pop_size=50, n_gen=20, verbose=True)
```

**Parameters:**
- `setup_file` (str): INI configuration file
- `rainfall_file` (str): Rainfall data file
- `observed_file` (str): Observed outflow file
- `output_folder` (str): Output directory
- `param_names` (list, optional): Parameters to calibrate
- `param_bounds` (list, optional): Parameter bounds as [(min, max), ...]
- `pop_size` (int): Population size (default: 50)
- `n_gen` (int): Number of generations (default: 20)
- `verbose` (bool): Print progress (default: True)

**Returns:**
- `best_individual` (list): Best parameter values
- `best_fitness` (float): Best fitness value
- `logbook` (deap.tools.Logbook): Evolution statistics

**Example:**
```python
# Legacy interface
best_params, best_fitness, logbook = pdt.calibrate(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="outflow.csv",
    output_folder="results/"
)

print(f"Best fitness: {best_fitness}")
print(f"Best parameters: {best_params}")
```

## Objective Functions

### Available Metrics

#### Nash-Sutcliffe Efficiency (NSE)
```python
objective_function='nse'
```
- Range: (-∞, 1], perfect = 1
- Emphasizes high flows
- Most commonly used in hydrology

#### Root Mean Square Error (RMSE)
```python
objective_function='rmse'
```
- Range: [0, ∞), perfect = 0
- Penalizes large errors heavily
- Good for peak flow matching

#### Mean Absolute Error (MAE)
```python
objective_function='mae'
```
- Range: [0, ∞), perfect = 0
- Less sensitive to outliers than RMSE
- Good for overall fit

#### Coefficient of Determination (R²)
```python
objective_function='r2'
```
- Range: [0, 1], perfect = 1
- Measures linear correlation
- Good for trend matching

#### Kling-Gupta Efficiency (KGE)
```python
objective_function='kge'
```
- Range: (-∞, 1], perfect = 1
- Balances correlation, bias, and variability
- More balanced than NSE

#### Multi-objective
```python
objective_function='multi'
weights=[0.6, 0.4]  # NSE weight, Volume error weight
```
- Combines multiple objectives
- Weights control relative importance

### Custom Objective Functions

You can define custom objective functions by modifying the calibration code:

```python
def custom_objective(observed, simulated):
    """Custom objective function example."""
    # Peak error component
    peak_obs = max(observed)
    peak_sim = max(simulated)
    peak_error = abs(peak_sim - peak_obs) / peak_obs

    # Volume error component
    vol_obs = sum(observed)
    vol_sim = sum(simulated)
    vol_error = abs(vol_sim - vol_obs) / vol_obs

    # Combined metric (to minimize)
    return peak_error + vol_error
```

## Parameter Configuration

### Default Parameters

The following parameters are calibrated by default:

| Parameter | Symbol | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `Ks` | Saturated hydraulic conductivity [m/s] | 1e-6 to 1e-3 |
| `gama` | Pore-size distribution parameter [-] | 1.0 to 5.0 |
| `sw` | Wilting point moisture content [-] | 0.05 to 0.3 |
| `sfc` | Field capacity moisture content [-] | 0.2 to 0.6 |
| `Cd` | Pipe discharge coefficient [-] | 0.5 to 1.0 |

### Custom Parameter Selection

```python
# Hydraulic parameters only
hydraulic_params = ['Ks', 'gama', 'nf', 'ng']

# Drainage parameters only
drainage_params = ['Cd', 'eta', 'hpipe']

# Moisture parameters only
moisture_params = ['sw', 'sfc', 'ss', 'sh']

# All physical parameters
all_params = ['Ks', 'gama', 'sw', 'sfc', 'ss', 'sh', 'Cd', 'eta', 'nf', 'ng', 'Dtl', 'Dg']

result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="obs.csv",
    parameters_to_calibrate=hydraulic_params
)
```

### Parameter Bounds

#### Default Bounds
```python
default_bounds = {
    'Ks': (1e-6, 1e-3),      # Hydraulic conductivity [m/s]
    'gama': (1.0, 5.0),      # Pore-size parameter [-]
    'sw': (0.05, 0.3),       # Wilting point [-]
    'sfc': (0.2, 0.6),       # Field capacity [-]
    'ss': (0.4, 0.9),        # Saturation [-]
    'sh': (0.01, 0.1),       # Hygroscopic point [-]
    'Cd': (0.5, 1.0),        # Discharge coefficient [-]
    'eta': (0.0, 1.0),       # Drainage coefficient [-]
    'nf': (0.3, 0.6),        # Filter porosity [-]
    'ng': (0.3, 0.5),        # Gravel porosity [-]
    'Dtl': (0.05, 0.3),      # Transition depth [m]
    'Dg': (0.2, 1.0)         # Gravel depth [m]
}
```

#### Custom Bounds
```python
# Tight bounds for well-known parameters
tight_bounds = {
    'lower': [5e-5, 2.0, 0.1, 0.3, 0.6],  # Ks, gama, sw, sfc, Cd
    'upper': [2e-4, 3.0, 0.2, 0.5, 0.9]
}

# Wide bounds for uncertain parameters
wide_bounds = {
    'lower': [1e-6, 1.0, 0.05, 0.2, 0.5],
    'upper': [1e-3, 5.0, 0.3, 0.6, 1.0]
}
```

## Genetic Algorithm Configuration

### Population Size
- **Small (20-50)**: Fast but may miss optimal solutions
- **Medium (50-100)**: Good balance of speed and exploration
- **Large (100-500)**: Thorough search but slower

### Number of Generations
- **Few (10-20)**: Quick calibration, may not converge
- **Moderate (20-50)**: Usually sufficient for convergence
- **Many (50-200)**: Ensures convergence but time-consuming

### Crossover and Mutation
```python
# Conservative settings (slow but steady)
result = pdt.run_calibration(
    crossover_prob=0.5,
    mutation_prob=0.1
)

# Aggressive settings (fast exploration)
result = pdt.run_calibration(
    crossover_prob=0.8,
    mutation_prob=0.3
)

# Balanced settings (recommended)
result = pdt.run_calibration(
    crossover_prob=0.7,
    mutation_prob=0.2
)
```

## Data Format Requirements

### Observed Outflow File

CSV format with time and outflow columns:

```csv
time,outflow
0,0.0
60,0.000015
120,0.000045
180,0.000123
240,0.000089
300,0.000034
360,0.0
```

**Requirements:**
- Time in minutes from start
- Outflow in m³/s
- No missing values in calibration period
- Synchronized with rainfall data

### Rainfall Data File

DAT format with time and rainfall:

```
# Rainfall event data
# Time(min) Rainfall(mm/h)
0.0 0.0
60.0 5.2
120.0 12.8
180.0 8.4
240.0 3.1
300.0 0.0
```

### Setup File

INI format with all model parameters:

```ini
[PHYSICAL]
A = 10.0          # Bottom area [m²]
Ap = 12.0         # Ponding area [m²]
L = 1.2           # Total depth [m]
Df = 0.6          # Filter depth [m]

[HYDRAULIC]
Ks = 1e-4         # Saturated hydraulic conductivity [m/s]
gama = 2.5        # Pore-size parameter [-]

[DRAINAGE]
dpipe = 100       # Pipe diameter [mm]
hpipe = 0.1       # Pipe height [m]
Cd = 0.8          # Discharge coefficient [-]
```

## Output Analysis

### Calibration Results Structure

```python
result = {
    'best_params': {
        'Ks': 8.5e-5,
        'gama': 2.3,
        'sw': 0.12,
        'sfc': 0.35,
        'Cd': 0.78
    },
    'best_fitness': 0.847,  # NSE value
    'convergence_history': [0.12, 0.34, 0.56, 0.78, 0.82, 0.84, 0.847],
    'final_population': [[param_sets], ...],
    'statistics': {
        'mean_fitness': 0.72,
        'std_fitness': 0.15,
        'convergence_generation': 45
    }
}
```

### Result Visualization

```python
import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(result['convergence_history'])
plt.xlabel('Generation')
plt.ylabel('Best Fitness (NSE)')
plt.title('Calibration Convergence')
plt.grid(True)
plt.show()

# Test calibrated parameters
setup = pdt.read_setup_file("config.ini")
params = pdt.initialize_parameters(setup, result['best_params'])
data, wb = pdt.run_model(params, "rainfall.dat")

# Plot results vs observations
plt.figure(figsize=(12, 6))
plt.plot(data['time'], data['qpipe'], label='Simulated', linewidth=2)
# plt.plot(obs_time, obs_flow, label='Observed', linewidth=2)
plt.xlabel('Time [minutes]')
plt.ylabel('Outflow [m³/s]')
plt.legend()
plt.title(f'Calibrated Model Results (NSE = {result["best_fitness"]:.3f})')
plt.grid(True)
plt.show()
```

## Advanced Features

### Multi-Event Calibration

```python
# Calibrate using multiple rainfall events
rainfall_files = ["event_01.dat", "event_02.dat", "event_03.dat"]
observed_files = ["obs_01.csv", "obs_02.csv", "obs_03.csv"]

results = []
for rain_file, obs_file in zip(rainfall_files, observed_files):
    result = pdt.run_calibration(
        setup_file="config.ini",
        rainfall_file=rain_file,
        observed_file=obs_file,
        generations=30
    )
    results.append(result)

# Combine results or select best
best_overall = max(results, key=lambda x: x['best_fitness'])
```

### Parameter Sensitivity During Calibration

```python
# Track parameter evolution
def analyze_parameter_evolution(result):
    """Analyze how parameters evolved during calibration."""
    pop = result['final_population']
    param_names = list(result['best_params'].keys())

    # Calculate parameter statistics
    for i, param in enumerate(param_names):
        values = [individual[i] for individual in pop]
        print(f"{param}:")
        print(f"  Best: {result['best_params'][param]:.6f}")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std:  {np.std(values):.6f}")
        print()

analyze_parameter_evolution(result)
```

### Calibration Validation

```python
# Split data for validation
def split_calibration_validation(rainfall_file, observed_file, split_ratio=0.7):
    """Split data into calibration and validation periods."""
    # Load full datasets
    time_rain, rainfall = pdt.read_rainfall_dat_file(rainfall_file)
    obs_data = pd.read_csv(observed_file)

    # Find split point
    split_time = max(time_rain) * split_ratio

    # Create calibration files
    cal_indices = [i for i, t in enumerate(time_rain) if t <= split_time]
    val_indices = [i for i, t in enumerate(time_rain) if t > split_time]

    # Save split files and return filenames
    # Implementation details...

    return cal_rain_file, cal_obs_file, val_rain_file, val_obs_file

# Use split data
cal_rain, cal_obs, val_rain, val_obs = split_calibration_validation(
    "full_rainfall.dat", "full_observed.csv"
)

# Calibrate on first part
result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file=cal_rain,
    observed_file=cal_obs
)

# Validate on second part
setup = pdt.read_setup_file("config.ini")
params = pdt.initialize_parameters(setup, result['best_params'])
val_data, val_wb = pdt.run_model(params, val_rain)

print(f"Calibration NSE: {result['best_fitness']:.3f}")
# Compare with validation observations...
```

## Error Handling

### Common Issues and Solutions

#### Missing Dependencies
```python
try:
    result = pdt.run_calibration(...)
except RuntimeError as e:
    if "deap" in str(e).lower():
        print("Install with: pip install permeabledt[calib]")
    else:
        print(f"Runtime error: {e}")
```

#### Invalid Parameter Bounds
```python
# Check bounds before calibration
def validate_bounds(bounds, param_names):
    if len(bounds['lower']) != len(param_names):
        raise ValueError("Bounds length mismatch")

    for i, (low, high) in enumerate(zip(bounds['lower'], bounds['upper'])):
        if low >= high:
            raise ValueError(f"Invalid bounds for {param_names[i]}: {low} >= {high}")

validate_bounds(custom_bounds, params_to_calibrate)
```

#### Data Synchronization Issues
```python
# Verify data alignment
def check_data_sync(rainfall_file, observed_file):
    time_rain, _ = pdt.read_rainfall_dat_file(rainfall_file)
    obs_data = pd.read_csv(observed_file)

    if len(time_rain) != len(obs_data):
        print(f"Warning: Data length mismatch")
        print(f"Rainfall: {len(time_rain)} points")
        print(f"Observed: {len(obs_data)} points")

    if max(time_rain) != max(obs_data['time']):
        print(f"Warning: Time range mismatch")

check_data_sync("rainfall.dat", "observed.csv")
```

## Performance Optimization

### Parallel Processing

The genetic algorithm can utilize multiple CPU cores:

```python
# Use all available cores
result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="obs.csv",
    population_size=100,  # Larger population for parallel processing
    generations=50
)
```

### Memory Management

For long-term simulations:

```python
# Reduce memory usage
result = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="data.dat",
    observed_file="obs.csv",
    population_size=50,   # Smaller population
    generations=100       # More generations instead
)
```

This calibration API provides comprehensive tools for optimizing permeable pavement model parameters, with flexible configuration options and robust error handling for reliable results.