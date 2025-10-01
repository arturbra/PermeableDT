# Data Preparation Guide

This guide explains how to prepare and format your data for use with permeabledt. Proper data preparation is crucial for successful modeling, calibration, and forecasting.

## Table of Contents

1. [Data Requirements Overview](#data-requirements-overview)
2. [Setup Files (.ini)](#setup-files-ini)
3. [Rainfall Data (.dat)](#rainfall-data-dat)
4. [Observed Outflow Data (.csv)](#observed-outflow-data-csv)
5. [Particle Filter Data (.ssv)](#particle-filter-data-ssv)
6. [Configuration Files (.toml)](#configuration-files-toml)
7. [Data Quality Guidelines](#data-quality-guidelines)
8. [Common Data Issues](#common-data-issues)
9. [Data Conversion Tools](#data-conversion-tools)
10. [Example Datasets](#example-datasets)

## Data Requirements Overview

permeabledt requires different types of data depending on your modeling objectives:

### Basic Simulation
- **Setup file (.ini)**: Model parameters and configuration
- **Rainfall data (.dat)**: Precipitation time series

### Calibration
- **Setup file (.ini)**: Including calibration parameter bounds
- **Rainfall data (.dat)**: Multiple events for calibration
- **Observed outflow (.csv)**: Measured flow data for parameter estimation

### Particle Filtering
- **Setup file (.ini)**: Calibrated model parameters
- **Rainfall data (.dat)**: Including forecast data
- **Observed outflow (.ssv)**: Real-time observations
- **Configuration file (.toml)**: Particle filter settings

## Setup Files (.ini)
Setup files contain all model parameters in INI format with clearly defined sections.

### Required Sections

#### [GENERAL] - Physical Properties
```ini
[GENERAL]
# Evapotranspiration constant
Kc = 0.0
# Layer depths (meters)
Df = 0.0508          # Filter media depth
Dtl = 0.1016         # Transition layer depth
Dg = 0.2032          # Gravel layer depth
# Porosity values (0-1)
nf = 0.32            # Filter media porosity
nt = 0.4             # Transition layer porosity
ng = 0.35            # Gravel layer porosity
```

#### [PONDING_ZONE] - Surface Water Storage
```ini
[PONDING_ZONE]
# Surface area (m²)
Ap = 195.1
# Overflow parameters
Hover = 0.5          # Overflow height (m)
Kweir = 1.3          # Weir coefficient
wWeir = 5.0          # Weir width (m)
expWeir = 2.5        # Weir exponent
# Infiltration parameters
Cs = 0               # Side flow coefficient
Pp = 0               # Unlined perimeter (m)
flagp = 1            # Lining flag (1=lined, 0=unlined)
```

#### [UNSATURATED_ZONE] - Vadose Zone
```ini
[UNSATURATED_ZONE]
# Physical properties
A = 195.1         # Bottom area (m²)
husz = 0.0508        # Initial unsaturated depth (m)
nusz = 0.32          # Initial unsaturated porosity
# Hydraulic properties
Ks = 0.00048         # Saturated hydraulic conductivity (m/s)
gama = 5.37          # Pore-size distribution parameter
# Moisture parameters
sh = 0.01            # Hygroscopic point (0-1)
sw = 0.022           # Wilting point (0-1)
sfc = 0.084          # Field capacity (0-1)
ss = 0.096           # Plant stress point (0-1)
# External infiltration
Kf = 0               # Surrounding soil conductivity (m/s)
```

#### [SATURATED_ZONE] - Groundwater Zone
```ini
[SATURATED_ZONE]
# Infiltration parameters
Psz = 0              # Unlined perimeter (m)
flagsz = 1           # Lining flag (1=lined, 0=unlined)
# Pipe drainage
hpipe = 0            # Pipe height (m)
dpipe = 152.4        # Pipe diameter (mm)
Cd = 0.37            # Discharge coefficient (0-1)
eta = 0.37           # Drainage coefficient (0-1)
```

#### [TIMESTEP] - Simulation Control
```ini
[TIMESTEP]
dt = 900              # Timestep in seconds
```

#### [CALIBRATION] - Parameter Bounds (Optional)
```ini
[CALIBRATION]
# Parameter bounds for optimization
Ks_min = 0.000001
Ks_max = 0.01
sw_min = 0.01
sw_max = 0.1
sfc_min = 0.01
sfc_max = 0.25
ss_min = 0.01
ss_max = 0.3
Cd_min = 0.01
Cd_max = 0.8
ng_min = 0.2
ng_max = 0.5
gama_min = 1
gama_max = 25
hpipe_min = 0
hpipe_max = 0.2
eta_min = 0.001
eta_max = 1
nusz_min = 0.2
nusz_max = 0.32
# Genetic algorithm settings
pop = 100            # Population size
gen = 50             # Number of generations
```

### Parameter Guidelines

#### Physical Dimensions
- **Depths**: Must be positive values in meters
- **Areas**: Must be positive values in square meters
- **Total depth**: L = Df + Dtl + Dg

#### Hydraulic Properties
- **Porosity**: Values between 0 and 1
- **Hydraulic conductivity**: Positive values in m/s
- **Typical Ks ranges**: 1e-6 to 1e-2 m/s for engineered media

#### Moisture Parameters
- **Order requirement**: sh < sw < sfc < ss < 1
- **Typical values**:
  - sh: 0.01-0.05 (very dry)
  - sw: 0.02-0.10 (wilting point)
  - sfc: 0.08-0.30 (field capacity)
  - ss: 0.10-0.40 (saturation)

#### Drainage Parameters
- **Pipe height**: Should be reasonable relative to system depth
- **Discharge coefficient**: 0.1-1.0 (typical range)
- **Drainage coefficient**: 0.1-1.0 (1.0 = perfect drainage)

## Rainfall Data (.dat)

Rainfall files contain time series precipitation data in space-separated format.

### Format Specification

```
MM/DD/YYYY HH:MM precipitation_value
MM/DD/YYYY HH:MM precipitation_value
...
```

### Example File
```
10/05/2023 03:51 0.0
10/05/2023 03:52 0.0
10/05/2023 03:53 0.5
10/05/2023 03:54 1.2
10/05/2023 03:55 0.8
10/05/2023 03:56 0.3
10/05/2023 03:57 0.0
```

### Requirements
- **Time format**: MM/DD/YYYY HH:MM (24-hour format)
- **Units**: Default is inches; specify units with `rainfall_unit` parameter
- **Resolution**: Any time resolution supported (seconds to hours)
- **Missing data**: Use 0.0 for no precipitation
- **Negative values**: Not allowed (will cause errors)

### Creating Rainfall Files

#### From Pandas DataFrame
```python
import pandas as pd
from datetime import datetime, timedelta

# Create sample data
start_time = datetime(2023, 10, 5, 8, 0)
times = [start_time + timedelta(minutes=i) for i in range(120)]
rainfall = [0.1 if 30 <= i <= 90 else 0.0 for i in range(120)]

df = pd.DataFrame({
    'datetime': times,
    'rainfall_mm': rainfall
})

# Convert to permeabledt format
with open('rainfall.dat', 'w') as f:
    for _, row in df.iterrows():
        # Convert mm to inches for permeabledt
        rain_inches = row['rainfall_mm'] / 25.4
        timestamp = row['datetime'].strftime('%m/%d/%Y %H:%M')
        f.write(f"{timestamp} {rain_inches:.4f}\n")
```

#### From CSV File
```python
import pandas as pd

# Read existing CSV file
df = pd.read_csv('my_rainfall.csv')
df['datetime'] = pd.to_datetime(df['timestamp'])

# Convert and save
with open('rainfall.dat', 'w') as f:
    for _, row in df.iterrows():
        timestamp = row['datetime'].strftime('%m/%d/%Y %H:%M')
        rainfall = row['precipitation_mm'] / 25.4  # Convert to inches
        f.write(f"{timestamp} {rainfall:.4f}\n")
```

## Observed Outflow Data (.csv)

Observed outflow data is used for calibration and validation.

### Format Specification

CSV format with datetime and flow columns:

```csv
date,observed_outflow
10/26/2023 07:13,0.0
10/26/2023 07:14,0.0
10/26/2023 07:15,0.1
10/26/2023 07:16,0.5
```

### Requirements
- **Date column**: Named 'date', parseable by pandas
- **Flow column**: Named 'observed_outflow'
- **Units**: ft³/s (automatically converted to m³/s internally)
- **Missing data**: Use NaN or leave blank (will be dropped)
- **Negative values**: Use 0.0 for no flow

### Creating Observed Files

#### From Sensor Data
```python
import pandas as pd
import numpy as np

# Read raw sensor data
sensor_data = pd.read_csv('sensor_raw.csv')
sensor_data['datetime'] = pd.to_datetime(sensor_data['timestamp'])

# Data cleaning
sensor_data = sensor_data.dropna()  # Remove missing values
sensor_data = sensor_data[sensor_data['flow_rate'] >= 0]  # Remove negative flows

# Convert units if necessary (example: L/s to ft³/s)
sensor_data['observed_outflow'] = sensor_data['flow_rate'] / 28.317  # L/s to ft³/s

# Resample to desired frequency
hourly_data = sensor_data.set_index('datetime').resample('H').mean().reset_index()

# Save in permeabledt format
hourly_data[['datetime', 'observed_outflow']].rename(
    columns={'datetime': 'date'}
).to_csv('observed_outflow.csv', index=False)
```

#### From Manual Measurements
```python
import pandas as pd
from datetime import datetime

# Manual measurements
measurements = [
    ('2023-10-05 08:00', 0.0),
    ('2023-10-05 09:00', 0.15),
    ('2023-10-05 10:00', 0.82),
    ('2023-10-05 11:00', 0.45),
    ('2023-10-05 12:00', 0.12),
    ('2023-10-05 13:00', 0.0),
]

# Create DataFrame
df = pd.DataFrame(measurements, columns=['date', 'observed_outflow'])
df['date'] = pd.to_datetime(df['date'])

# Save
df.to_csv('observed_outflow.csv', index=False)
```

## Particle Filter Data (.ssv)

Particle filter observations use space-separated values format.

### Format Specification

```
time value
0 0.000
1 0.001
2 0.015
3 0.032
```

### Requirements
- **Time column**: Integer timestep indices (0, 1, 2, ...)
- **Value column**: Flow in m³/s
- **Separator**: Single space
- **Header**: Column names on first line

### Creating SSV Files

```python
import pandas as pd
def create_ssv_file(csv_file, output_file, forecast_time=None):
    """Convert CSV outflow data to SSV format for particle filter"""

    # Read CSV data
    df = pd.read_csv(csv_file, parse_dates=['date']).dropna()

    # Limit to forecast time if specified
    if forecast_time is not None:
        df.set_index('date', inplace=True)
        df = df.resample("15min").mean().reset_index()  # Resample to 15-min
        forecast_date = df['date'][0] + pd.Timedelta(minutes=forecast_time * 15)
        df = df[df['date'] <= forecast_date]

    # Convert to timestep indices and m³/s
    df['time'] = range(len(df))
    df['value'] = df['observed_outflow'] * 0.028316847  # ft³/s to m³/s

    # Save as space-separated file
    df[['time', 'value']].to_csv(output_file, sep=' ', index=False)

    return df

# Usage
ssv_data = create_ssv_file('observed_outflow.csv', 'observed_outflow.ssv')
```

## Configuration Files (.toml)

TOML files configure particle filter settings.

### Basic Structure

```toml
[components]
model = "permeabledt.particle_filter.PavementModel"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0
until = 149
steps_per_unit = 1

[model]
setup_file = "calibrated_parameters.ini"
rainfall_file = "rainfall.dat"

[prior]
hp = { name = "uniform", args.loc = 0.0, args.scale = 0.05}
hsz = { name = "uniform", args.loc = 1e-6, args.scale = 0.05}
s = { name = "uniform", args.loc = 0.0, args.scale = 0.2}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "observed_outflow.ssv"

[filter]
particles = 1000
prng_seed = 2025
```

### Creating TOML Files

```python
def create_toml_config(setup_file, rainfall_file, obs_file, output_file,
                      sim_length, particles=1000):
    """Create TOML configuration for particle filter"""

    config = f"""[components]
model = "permeabledt.particle_filter.PavementModel"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0
until = {sim_length - 1}
steps_per_unit = 1

[model]
setup_file = "{setup_file}"
rainfall_file = "{rainfall_file}"

[prior]
hp = {{ name = "uniform", args.loc = 0.0, args.scale = 0.05}}
hsz = {{ name = "uniform", args.loc = 1e-6, args.scale = 0.05}}
s = {{ name = "uniform", args.loc = 0.0, args.scale = 0.2}}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "{obs_file}"
weir_k = 0.006
weir_n = 2.5532
head_error_inches = 0.08

[filter]
particles = {particles}
prng_seed = 2025
resample.threshold = 0.5
"""

    with open(output_file, 'w') as f:
        f.write(config)

# Usage
create_toml_config(
    setup_file="calibrated_parameters.ini",
    rainfall_file="rainfall.dat",
    obs_file="observed_outflow.ssv",
    output_file="pavement.toml",
    sim_length=150
)
```