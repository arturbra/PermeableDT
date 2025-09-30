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

### Sensitivity Analysis
- **Setup file (.ini)**: Including parameter ranges for analysis
- **Rainfall data (.dat)**: Representative rainfall events

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
Ap = 195.0964
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
A = 195.0964         # Bottom area (m²)
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
hpipe = 0.02552      # Pipe height (m)
dpipe = 152.4        # Pipe diameter (mm)
Cd = 0.1276          # Discharge coefficient (0-1)
eta = 0.2316         # Drainage coefficient (0-1)
```

#### [TIMESTEP] - Simulation Control
```ini
[TIMESTEP]
dt = 60              # Timestep in seconds
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
pop = 400            # Population size
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
- **Discharge coefficient**: 0.1-0.8 (typical range)
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

### Quality Control

#### Time Series Checks
```python
import pandas as pd

def validate_rainfall_file(filename):
    """Validate rainfall data file"""

    # Read file
    data = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Line {line_num}: Invalid format")
                continue

            date_str = f"{parts[0]} {parts[1]}"
            try:
                timestamp = pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')
                rainfall = float(parts[2])

                if rainfall < 0:
                    print(f"Line {line_num}: Negative rainfall value")

                data.append((timestamp, rainfall))

            except ValueError as e:
                print(f"Line {line_num}: Parse error - {e}")

    # Create DataFrame for analysis
    df = pd.DataFrame(data, columns=['datetime', 'rainfall'])
    df = df.sort_values('datetime')

    # Check for gaps
    time_diffs = df['datetime'].diff()
    expected_dt = time_diffs.mode()[0]  # Most common time step
    gaps = time_diffs[time_diffs > expected_dt * 1.5]

    if len(gaps) > 0:
        print(f"Found {len(gaps)} time gaps larger than expected")

    # Summary statistics
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total records: {len(df)}")
    print(f"Time step: {expected_dt}")
    print(f"Total rainfall: {df['rainfall'].sum():.2f} inches")
    print(f"Max intensity: {df['rainfall'].max():.3f} inches/timestep")

    return df

# Usage
df = validate_rainfall_file('rainfall.dat')
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

### Quality Control

```python
def validate_outflow_file(filename):
    """Validate observed outflow data"""

    # Read file
    try:
        df = pd.read_csv(filename, parse_dates=['date'])
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Check required columns
    required_cols = ['date', 'observed_outflow']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return None

    # Check for negative flows
    negative_flows = df[df['observed_outflow'] < 0]
    if len(negative_flows) > 0:
        print(f"Found {len(negative_flows)} negative flow values")

    # Check for missing data
    missing_data = df['observed_outflow'].isna().sum()
    if missing_data > 0:
        print(f"Found {missing_data} missing flow values")

    # Summary statistics
    print(f"Time range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df)}")
    print(f"Flow range: {df['observed_outflow'].min():.3f} to {df['observed_outflow'].max():.3f} ft³/s")
    print(f"Mean flow: {df['observed_outflow'].mean():.3f} ft³/s")

    return df

# Usage
df = validate_outflow_file('observed_outflow.csv')
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
particles = 3000
prng_seed = 2025
```

### Creating TOML Files

```python
def create_toml_config(setup_file, rainfall_file, obs_file, output_file,
                      sim_length, particles=3000):
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

## Data Quality Guidelines

### General Principles

1. **Consistency**: All files must use consistent time stamps
2. **Completeness**: Minimize missing data gaps
3. **Accuracy**: Validate data against known physical limits
4. **Resolution**: Match temporal resolution to modeling needs
5. **Units**: Clearly document and convert units properly

### Time Synchronization

```python
def synchronize_data(rainfall_file, outflow_file, output_prefix):
    """Synchronize rainfall and outflow data to common time grid"""

    # Read both datasets
    rain_df = pd.read_csv(rainfall_file, sep=' ', header=None,
                         names=['date', 'time', 'rainfall'])
    rain_df['datetime'] = pd.to_datetime(rain_df['date'] + ' ' + rain_df['time'])

    flow_df = pd.read_csv(outflow_file, parse_dates=['date'])
    flow_df = flow_df.rename(columns={'date': 'datetime'})

    # Find common time range
    start_time = max(rain_df['datetime'].min(), flow_df['datetime'].min())
    end_time = min(rain_df['datetime'].max(), flow_df['datetime'].max())

    print(f"Common time range: {start_time} to {end_time}")

    # Filter to common range
    rain_sync = rain_df[(rain_df['datetime'] >= start_time) &
                       (rain_df['datetime'] <= end_time)]
    flow_sync = flow_df[(flow_df['datetime'] >= start_time) &
                       (flow_df['datetime'] <= end_time)]

    # Save synchronized files
    # Rainfall file
    with open(f'{output_prefix}_rainfall.dat', 'w') as f:
        for _, row in rain_sync.iterrows():
            f.write(f"{row['date']} {row['time']} {row['rainfall']:.4f}\n")

    # Outflow file
    flow_sync[['datetime', 'observed_outflow']].rename(
        columns={'datetime': 'date'}
    ).to_csv(f'{output_prefix}_outflow.csv', index=False)

    return rain_sync, flow_sync

# Usage
rain_sync, flow_sync = synchronize_data(
    'rainfall.dat', 'observed_outflow.csv', 'synchronized'
)
```

### Physical Validation

```python
def validate_physical_limits(setup_file, rainfall_file, outflow_file):
    """Check data against physical limits"""

    import permeabledt as pdt

    # Load parameters
    setup = pdt.read_setup_file(setup_file)
    params = pdt.initialize_parameters(setup)

    # Check rainfall data
    rain_df = pdt.read_rainfall_dat_file(rainfall_file)
    max_intensity = rain_df['rain'].max()

    # Physical limits for rainfall (rough guidelines)
    if max_intensity > 10:  # inches per timestep
        print(f"WARNING: Very high rainfall intensity: {max_intensity:.2f} inches/timestep")

    # Check outflow data
    flow_df = pd.read_csv(outflow_file, parse_dates=['date'])
    max_flow = flow_df['observed_outflow'].max()

    # Estimate maximum physical flow capacity
    pipe_area = np.pi * (params['dpipe'] / 1000 / 2)**2  # m²
    max_velocity = 5.0  # m/s (reasonable upper limit)
    max_capacity = pipe_area * max_velocity * 35.315  # Convert to ft³/s

    if max_flow > max_capacity:
        print(f"WARNING: Flow exceeds pipe capacity: {max_flow:.3f} > {max_capacity:.3f} ft³/s")

    # Check for flow during no-rain periods
    # (This requires synchronized data)

    print("Physical validation complete")

# Usage
validate_physical_limits('setup.ini', 'rainfall.dat', 'observed_outflow.csv')
```

## Common Data Issues

### Issue 1: Time Format Problems

**Problem**: Inconsistent datetime formats
```
# Bad
2023-10-05 8:00 AM
Oct 5, 2023 8:00
10/5/23 8:00
```

**Solution**: Standardize to MM/DD/YYYY HH:MM
```python
def standardize_time_format(input_file, output_file):
    """Standardize datetime format"""

    df = pd.read_csv(input_file)

    # Parse flexible datetime formats
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)

    # Convert to standard format
    df['formatted_time'] = df['datetime'].dt.strftime('%m/%d/%Y %H:%M')

    # Save with standard format
    df[['formatted_time', 'value']].to_csv(output_file,
                                          header=['datetime', 'value'],
                                          index=False)
```

### Issue 2: Unit Inconsistencies

**Problem**: Mixed units in dataset
**Solution**: Implement unit conversion functions

```python
def convert_rainfall_units(value, from_unit, to_unit='inches'):
    """Convert rainfall units"""

    # Convert to mm first
    if from_unit == 'inches':
        mm_value = value * 25.4
    elif from_unit == 'cm':
        mm_value = value * 10
    elif from_unit == 'mm':
        mm_value = value
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    # Convert from mm to target
    if to_unit == 'inches':
        return mm_value / 25.4
    elif to_unit == 'mm':
        return mm_value
    elif to_unit == 'cm':
        return mm_value / 10
    else:
        raise ValueError(f"Unknown unit: {to_unit}")

def convert_flow_units(value, from_unit, to_unit='ft3/s'):
    """Convert flow units"""

    # Convert to m³/s first
    if from_unit == 'ft3/s':
        m3s_value = value * 0.028316847
    elif from_unit == 'L/s':
        m3s_value = value / 1000
    elif from_unit == 'gpm':
        m3s_value = value * 6.30902e-5
    elif from_unit == 'm3/s':
        m3s_value = value
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    # Convert from m³/s to target
    if to_unit == 'ft3/s':
        return m3s_value / 0.028316847
    elif to_unit == 'L/s':
        return m3s_value * 1000
    elif to_unit == 'gpm':
        return m3s_value / 6.30902e-5
    elif to_unit == 'm3/s':
        return m3s_value
    else:
        raise ValueError(f"Unknown unit: {to_unit}")
```

### Issue 3: Missing Data

**Problem**: Gaps in time series
**Solution**: Interpolation or gap filling

```python
def fill_missing_data(df, time_col='datetime', value_col='value', method='linear'):
    """Fill missing data in time series"""

    # Ensure datetime index
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Create complete time index
    freq = pd.infer_freq(df.index)
    if freq is None:
        # Infer frequency from most common interval
        intervals = df.index.to_series().diff().value_counts()
        freq = intervals.index[0]

    complete_index = pd.date_range(start=df.index.min(),
                                  end=df.index.max(),
                                  freq=freq)

    # Reindex and fill missing values
    df_complete = df.reindex(complete_index)

    if method == 'linear':
        df_complete[value_col] = df_complete[value_col].interpolate(method='linear')
    elif method == 'zero':
        df_complete[value_col] = df_complete[value_col].fillna(0)
    elif method == 'forward':
        df_complete[value_col] = df_complete[value_col].fillna(method='ffill')

    return df_complete.reset_index()
```

## Data Conversion Tools

### Complete Data Preparation Pipeline

```python
class permeabledtDataPrep:
    """Complete data preparation pipeline for permeabledt"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.data_summary = {}

    def prepare_rainfall(self, source_file, datetime_col, rainfall_col,
                        rainfall_unit='mm', output_file=None):
        """Prepare rainfall data from various formats"""

        if output_file is None:
            output_file = f"{self.project_name}_rainfall.dat"

        # Read source data
        df = pd.read_csv(source_file)
        df['datetime'] = pd.to_datetime(df[datetime_col])
        df = df.sort_values('datetime')

        # Convert units to inches
        rain_inches = convert_rainfall_units(df[rainfall_col], rainfall_unit, 'inches')

        # Write DAT file
        with open(output_file, 'w') as f:
            for i, row in df.iterrows():
                timestamp = row['datetime'].strftime('%m/%d/%Y %H:%M')
                f.write(f"{timestamp} {rain_inches.iloc[i]:.4f}\n")

        # Store summary
        self.data_summary['rainfall'] = {
            'file': output_file,
            'records': len(df),
            'start': df['datetime'].min(),
            'end': df['datetime'].max(),
            'total_mm': df[rainfall_col].sum(),
            'total_inches': rain_inches.sum()
        }

        print(f"Rainfall data prepared: {output_file}")
        return output_file

    def prepare_outflow(self, source_file, datetime_col, flow_col,
                       flow_unit='L/s', output_file=None):
        """Prepare outflow data from various formats"""

        if output_file is None:
            output_file = f"{self.project_name}_outflow.csv"

        # Read source data
        df = pd.read_csv(source_file)
        df['datetime'] = pd.to_datetime(df[datetime_col])
        df = df.sort_values('datetime').dropna()

        # Convert units to ft³/s
        flow_ft3s = convert_flow_units(df[flow_col], flow_unit, 'ft3/s')

        # Create output DataFrame
        output_df = pd.DataFrame({
            'date': df['datetime'],
            'observed_outflow': flow_ft3s
        })

        # Save CSV file
        output_df.to_csv(output_file, index=False)

        # Store summary
        self.data_summary['outflow'] = {
            'file': output_file,
            'records': len(output_df),
            'start': df['datetime'].min(),
            'end': df['datetime'].max(),
            'max_flow_ft3s': flow_ft3s.max(),
            'mean_flow_ft3s': flow_ft3s.mean()
        }

        print(f"Outflow data prepared: {output_file}")
        return output_file

    def create_setup_template(self, area_m2, output_file=None):
        """Create setup file template with reasonable defaults"""

        if output_file is None:
            output_file = f"{self.project_name}_setup.ini"

        setup_content = f"""[GENERAL]
# Physical properties
Kc = 0.0
Df = 0.05              # Filter depth (m)
Dtl = 0.10             # Transition depth (m)
Dg = 0.20              # Gravel depth (m)
nf = 0.32              # Filter porosity
nt = 0.40              # Transition porosity
ng = 0.35              # Gravel porosity

[PONDING_ZONE]
# Surface area (m²)
Ap = {area_m2}
# Overflow parameters
Hover = 0.5            # Overflow height (m)
Kweir = 1.3            # Weir coefficient
wWeir = 5.0            # Weir width (m)
expWeir = 2.5          # Weir exponent
# Infiltration parameters
Cs = 0                 # Side flow coefficient
Pp = 0                 # Unlined perimeter (m)
flagp = 1              # Lining flag (1=lined)

[UNSATURATED_ZONE]
# Physical properties
A = {area_m2}          # Bottom area (m²)
husz = 0.05            # Initial unsaturated depth (m)
nusz = 0.32            # Initial porosity
# Hydraulic properties
Ks = 0.0005            # Saturated hydraulic conductivity (m/s)
gama = 5.0             # Pore-size parameter
# Moisture parameters
sh = 0.01              # Hygroscopic point
sw = 0.02              # Wilting point
sfc = 0.08             # Field capacity
ss = 0.09              # Saturation point
# External infiltration
Kf = 0                 # Surrounding soil K (m/s)

[SATURATED_ZONE]
# Infiltration parameters
Psz = 0                # Unlined perimeter (m)
flagsz = 1             # Lining flag (1=lined)
# Pipe drainage
hpipe = 0.025          # Pipe height (m)
dpipe = 152.4          # Pipe diameter (mm)
Cd = 0.13              # Discharge coefficient
eta = 0.23             # Drainage coefficient

[TIMESTEP]
dt = 60                # Timestep (seconds)

[CALIBRATION]
# Parameter bounds for calibration
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
gama_min = 1
gama_max = 25
hpipe_min = 0
hpipe_max = 0.2
eta_min = 0.001
eta_max = 1
# Optimization settings
pop = 400
gen = 50
"""

        with open(output_file, 'w') as f:
            f.write(setup_content)

        print(f"Setup template created: {output_file}")
        return output_file

    def print_summary(self):
        """Print data preparation summary"""

        print(f"\n=== Data Preparation Summary: {self.project_name} ===")

        for data_type, info in self.data_summary.items():
            print(f"\n{data_type.upper()} DATA:")
            for key, value in info.items():
                print(f"  {key}: {value}")

# Usage example
prep = permeabledtDataPrep("my_project")

# Prepare data files
rainfall_file = prep.prepare_rainfall(
    source_file="raw_weather.csv",
    datetime_col="timestamp",
    rainfall_col="precipitation_mm",
    rainfall_unit="mm"
)

outflow_file = prep.prepare_outflow(
    source_file="flow_measurements.csv",
    datetime_col="measurement_time",
    flow_col="discharge_Ls",
    flow_unit="L/s"
)

setup_file = prep.create_setup_template(area_m2=195.0)

# Print summary
prep.print_summary()
```

## Example Datasets

### Synthetic Test Dataset

```python
def create_synthetic_dataset(output_dir="test_data"):
    """Create a complete synthetic dataset for testing"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Create time series (4-hour event)
    start_time = pd.Timestamp('2023-10-05 08:00')
    times = pd.date_range(start_time, periods=240, freq='1min')

    # Synthetic rainfall (triangular hyetograph)
    rainfall_mm = np.zeros(240)
    peak_time = 120  # 2 hours
    peak_intensity = 2.0  # mm/min

    for i in range(240):
        if i <= peak_time:
            rainfall_mm[i] = peak_intensity * (i / peak_time) if i > 0 else 0
        else:
            rainfall_mm[i] = peak_intensity * (1 - (i - peak_time) / peak_time)
            if rainfall_mm[i] < 0:
                rainfall_mm[i] = 0

    # Synthetic outflow (delayed response with attenuation)
    outflow_ft3s = np.zeros(240)
    delay = 30  # 30-minute delay
    attenuation = 0.3

    for i in range(delay, 240):
        # Simple linear response to rainfall
        rain_input = rainfall_mm[i-delay:i].mean()
        outflow_ft3s[i] = rain_input * attenuation * 0.1  # Convert to ft³/s scale

    # Add some noise
    noise = np.random.normal(0, 0.01, 240)
    outflow_ft3s = np.maximum(0, outflow_ft3s + noise)

    # Save rainfall file
    rainfall_file = os.path.join(output_dir, "synthetic_rainfall.dat")
    with open(rainfall_file, 'w') as f:
        for i, time in enumerate(times):
            rain_inches = rainfall_mm[i] / 25.4
            f.write(f"{time.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")

    # Save outflow file
    outflow_file = os.path.join(output_dir, "synthetic_outflow.csv")
    outflow_df = pd.DataFrame({
        'date': times,
        'observed_outflow': outflow_ft3s
    })
    outflow_df.to_csv(outflow_file, index=False)

    # Create setup file
    setup_file = os.path.join(output_dir, "synthetic_setup.ini")
    prep = permeabledtDataPrep("synthetic")
    prep.create_setup_template(195.0, setup_file)

    print(f"Synthetic dataset created in: {output_dir}")
    print(f"Files created:")
    print(f"  - {rainfall_file}")
    print(f"  - {outflow_file}")
    print(f"  - {setup_file}")

    return rainfall_file, outflow_file, setup_file

# Create test dataset
files = create_synthetic_dataset()
```

This comprehensive data preparation guide provides everything needed to format data correctly for permeabledt. Follow these guidelines to ensure successful modeling, calibration, and forecasting with your permeable pavement systems.