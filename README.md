# permeabledt

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**permeabledt** is a comprehensive Python library for permeable pavement digital twin modeling, featuring water flow simulation, genetic algorithm calibration, particle filtering, sensitivity analysis, and weather data acquisition capabilities.

## Features

- 🌊 **Water Flow Modeling**: Physics-based simulation of permeable pavement systems
- 🧬 **Genetic Algorithm Calibration**: Automated parameter optimization using DEAP
- 📊 **Particle Filtering**: Real-time state estimation and uncertainty quantification
- 🌦️ **Weather Data Integration**: HRRR forecast data downloading and processing
- 📊 **Visualization**: Built-in plotting functions for results analysis


### Manual Installation

```bash
git clone https://github.com/arturbra/permeabledt.git
cd permeabledt
pip install -e .
```

## Quick Start

```python
import permeabledt as pdt
import numpy as np

# Basic water flow simulation
params = pdt.initialize_parameters(setup)
data, water_balance = pdt.run_simulation(params, qin, qrain, emax)

# File-based simulation
data, wb = pdt.run_model(params, "rainfall.dat")

# Visualize results
pdt.plots.plot_rainfall_hydrograph("rainfall.dat", data['Qpipe'])
```

## Data Formats and File Structure

### Setup File (.ini)

The setup file contains all model parameters in INI format:

```ini
[GENERAL]
Kc = 0.0                    # Evapotranspiration constant
Df = 0.0508                 # Filter media depth (m)
Dtl = 0.1016               # Transition layer depth (m)
Dg = 0.2032                # Gravel layer depth (m)
nf = 0.32                  # Filter media porosity
nt = 0.4                   # Transition layer porosity
ng = 0.35                  # Gravel layer porosity

[PONDING_ZONE]
Ap = 195.0964              # Surface area (m²)
Hover = 0.5                # Overflow height (m)
Kweir = 1.3                # Weir coefficient
wWeir = 5.0                # Weir width (m)
expWeir = 2.5              # Weir exponent

[UNSATURATED_ZONE]
A = 195.0964               # Bottom area (m²)
Ks = 0.00048               # Saturated hydraulic conductivity (m/s)
sh = 0.01                  # Hygroscopic point moisture
sw = 0.022                 # Wilting point moisture
sfc = 0.084                # Field capacity
ss = 0.096                 # Plant stress moisture
gama = 5.37                # Saturated curve parameter

[SATURATED_ZONE]
hpipe = 0.025              # Pipe height (m)
dpipe = 152.4              # Pipe diameter (mm)
Cd = 0.27                  # Discharge coefficient
eta = 0.232                # Drainage coefficient

[TIMESTEP]
dt = 60                    # Timestep (seconds)

[CALIBRATION]
# Parameter bounds for calibration
Ks_min = 0.000001
Ks_max = 0.01
# ... (other parameter bounds)
pop = 400                  # Population size
gen = 50                   # Number of generations
```

### Rainfall Data (.dat)

Rainfall files should be space-separated with datetime and rainfall intensity:

```
10/05/2023 03:51 0.0
10/05/2023 03:52 0.0
10/05/2023 03:53 0.5
10/05/2023 03:54 1.2
10/05/2023 03:55 0.8
```

**Format**: `MM/DD/YYYY HH:MM rainfall_intensity`

### Observed Outflow Data (.csv)

```csv
date,observed_outflow
10/26/2023 07:13,0.0
10/26/2023 07:14,0.0
10/26/2023 07:15,0.1
10/26/2023 07:16,0.5
```

**Format**: CSV with datetime and outflow in ft³/s (automatically converted to m³/s)

## Usage Examples

### 1. Basic Water Flow Simulation

```python
import permeabledt as pdt

# Load parameters from setup file
setup = pdt.read_setup_file("input_parameters.ini")
params = pdt.initialize_parameters(setup)

# Run simulation with rainfall file
data, water_balance = pdt.run_model(params, "rainfall.dat", rainfall_unit='mm')

# Convert to DataFrame for analysis
df = pdt.results_dataframe(data, save=True, filename="results.csv")

print(f"Peak outflow: {max(data['Qpipe']):.4f} m³/s")
print(f"Total runoff: {water_balance['total_runoff']:.2f} mm")
```

### 2. Parameter Calibration

```python
import permeabledt as pdt
from pathlib import Path

# Define file lists
rainfall_files = [
    "events/event_01_rainfall.dat",
    "events/event_02_rainfall.dat",
    "events/event_03_rainfall.dat"
]

observed_files = [
    "events/event_01_observed_outflow.csv",
    "events/event_02_observed_outflow.csv",
    "events/event_03_observed_outflow.csv"
]

# Run calibration
best_params, calibrated_setup, logbook = pdt.run_calibration(
    calibration_rainfall=rainfall_files,
    calibration_observed_data=observed_files,
    setup_file="input_parameters.ini",
    output_setup_file="calibrated_parameters.ini",
    logbook_output_path="calibration_log.csv",
    seed=42
)

print(f"Best fitness: {best_params.fitness.values[0]:.4f}")
print(f"Calibrated parameters: {best_params}")
```

### 3. Particle Filtering for Real-time Forecasting

```python
import permeabledt as pdt
import pypfilt

# Set up particle filter configuration (pavement.toml)
# See examples/particle_filter/input/pavement.toml for full configuration

# Initialize model and observations
model = pdt.PavementModel(
    setup_file="calibrated_parameters.ini",
    rainfall_file="forecast_rainfall.dat"
)

observations = pdt.PipeObs(
    file="observed_outflow.ssv",
    weir_k=0.006,
    weir_n=2.5532,
    head_error_inches=0.08
)

# Run particle filter
scenario_file = "pavement.toml"
instances = list(pypfilt.load_instances(scenario_file))
instance = instances[0]

ctx = instance.build_context()
results = pypfilt.forecast(ctx, [forecast_time])

# Extract forecast statistics
forecast_mean = results['forecast']['pred_mean']
forecast_ci_50 = results['forecast']['pred_ci_50']
forecast_ci_95 = results['forecast']['pred_ci_95']
```

### 4. Weather Data Download

```python
import permeabledt as pdt
from datetime import datetime

# Initialize HRRR downloader
downloader = pdt.HRRRAccumulatedPrecipitationDownloader(
    lat=40.7589,  # Your site coordinates
    lon=-111.8883,
    timezone='US/Mountain'
)

# Download forecast data
forecast_data = downloader.download_forecast(
    start_local=datetime(2023, 10, 5, 0, 0),
    end_local=datetime(2023, 10, 6, 0, 0),
    output_file="hrrr_forecast.dat"
)

# The data is automatically formatted for permeabledt
print(f"Downloaded {len(forecast_data)} forecast timesteps")
```

### 5. Advanced Plotting

```python
import permeabledt as pdt
import matplotlib.pyplot as plt

# Plot rainfall-hydrograph
fig, axes = pdt.plots.plot_rainfall_hydrograph(
    rainfall_file="rainfall.dat",
    outflow_data=data['Qpipe'],
    rainfall_unit='mm',
    output_path="hydrograph.png"
)

# Add customizations
axes[0].set_title("Rainfall Event Analysis")
axes[1].set_ylabel("Outflow (L/s)")
plt.tight_layout()
plt.show()

# Plot calibration convergence
pdt.plots.plot_calibration_convergence(
    logbook_file="calibration_log.csv",
    output_path="convergence.png"
)
```

## File Organization

Organize your project files as follows:

```
your_project/
├── setup_files/
│   ├── input_parameters.ini
│   └── calibrated_parameters.ini
├── data/
│   ├── rainfall/
│   │   ├── event_01_rainfall.dat
│   │   ├── event_02_rainfall.dat
│   │   └── forecast_rainfall.dat
│   └── observed/
│       ├── event_01_observed_outflow.csv
│       ├── event_02_observed_outflow.csv
│       └── observed_outflow.ssv
├── particle_filter/
│   └── pavement.toml
├── results/
│   ├── simulation_results.csv
│   ├── calibration_log.csv
│   └── plots/
└── scripts/
    ├── run_simulation.py
    ├── calibrate_model.py
    └── particle_filter.py
```

## Creating Your Own Data Files

### Generate Synthetic Rainfall Data

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create synthetic rainfall event
start_time = datetime(2023, 10, 5, 4, 0)
duration_hours = 4
timestep_minutes = 1

times = []
rainfall = []

for i in range(duration_hours * 60):
    time = start_time + timedelta(minutes=i)
    times.append(time.strftime("%m/%d/%Y %H:%M"))

    # Create a triangular hyetograph
    if i < 60:  # Rising limb
        rain = i * 0.5 / 60
    elif i < 180:  # Peak
        rain = 0.5
    else:  # Falling limb
        rain = max(0, 0.5 - (i - 180) * 0.5 / 60)

    rainfall.append(rain)

# Save to file
with open("synthetic_rainfall.dat", "w") as f:
    for time, rain in zip(times, rainfall):
        f.write(f"{time} {rain:.3f}\n")
```

### Convert Observed Data

```python
import pandas as pd

# Convert your observed data to permeabledt format
df = pd.read_csv("your_data.csv")  # Adjust column names as needed
df['date'] = pd.to_datetime(df['timestamp'])
df['observed_outflow'] = df['flow_cfs']  # Convert units if necessary

# Save in required format
df[['date', 'observed_outflow']].to_csv(
    "observed_outflow.csv",
    index=False
)
```

## Advanced Configuration

### Particle Filter Setup (pavement.toml)

The particle filter requires a TOML configuration file. Key sections:

```toml
[model]
setup_file = "path/to/calibrated_parameters.ini"
rainfall_file = "path/to/forecast_rainfall.dat"

# Rainfall uncertainty model
use_magnitude_uncertainty_model = true
zero_rainfall_threshold = 0.01
rainfall_low_threshold = 0.12
low_rainfall_bias = 0.004
high_rainfall_bias_slope = -0.024

[prior]
hp = { name = "uniform", args.loc = 0.0, args.scale = 0.05}
hsz = { name = "uniform", args.loc = 1e-6, args.scale = 0.05}
s = { name = "uniform", args.loc = 0.0, args.scale = 0.2}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "path/to/observed_outflow.ssv"
weir_k = 0.006               # Weir coefficient
weir_n = 2.5532              # Weir exponent
head_error_inches = 0.08     # Measurement uncertainty

[filter]
particles = 1000
prng_seed = 2025
resample.threshold = 0.5
```

## Model Physics

The permeabledt model simulates three main zones in permeable pavement systems:

1. **Ponding Zone**: Surface water storage and overflow
2. **Unsaturated Zone**: Vadose zone with evapotranspiration and infiltration
3. **Saturated Zone**: Groundwater storage with pipe drainage

Key processes modeled:
- Surface runoff and weir overflow
- Unsaturated flow with moisture-dependent hydraulic conductivity
- Evapotranspiration based on soil moisture
- Pipe drainage with orifice/weir equations
- Lateral infiltration (if enabled)

## API Reference

### Core Functions

- `run_simulation(params, qin, qrain, emax)`: Run simulation with data arrays
- `run_model(params, rainfall_file, **kwargs)`: Run simulation from files
- `initialize_parameters(setup)`: Load parameters from setup file
- `calculate_water_balance(data, dt)`: Calculate water balance metrics

### Calibration Functions

- `run_calibration(rainfall_files, observed_files, setup_file, **kwargs)`: Modern calibration interface
- `calibrate(setup_cfg, pavement, events_list)`: Legacy calibration method

### Particle Filter Classes

- `PavementModel`: pypfilt.Model implementation for the pavement system
- `PipeObs`: pypfilt.obs.Univariate implementation for outflow observations

### Analysis Tools

- `SobolSensitivityAnalysis`: Global sensitivity analysis using Sobol indices
- `HRRRAccumulatedPrecipitationDownloader`: Download HRRR weather forecasts

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use permeabledt in your research, please cite:

```bibtex
@software{permeabledt,
  title={permeabledt: Digital Twin Tools for Permeable Pavement Modeling},
  author={Jose Brasil},
  year={2025},
  url={https://github.com/arturbra/permeabledt}
}
```

## Support

- 📧 Email: jose.brasil@utsa.edu
- 🐛 Issues: [GitHub Issues](https://github.com/arturbra/permeabledt/issues)
- 📖 Documentation: Coming soon

## Acknowledgments

- DEAP library for genetic algorithms
- pypfilt library for particle filtering
- SALib library for sensitivity analysis
- herbie-data for weather data access