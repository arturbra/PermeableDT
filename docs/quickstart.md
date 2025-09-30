# Quick Start Guide

Get up and running with permeabledt in minutes! This guide will walk you through your first simulation, calibration, and forecasting example.

## Prerequisites

- permeabledt installed ([Installation Guide](installation.md))
- Basic Python knowledge
- Your data files ready ([Data Preparation](data_preparation.md))

## 1. First Water Flow Simulation

### Step 1: Import permeabledt

```python
import permeabledt as pdt
import numpy as np
import pandas as pd
```

### Step 2: Create or Load Parameters

#### Option A: Use Example Parameters

```python
# Create a simple parameter dictionary for testing
params = {
    # Physical dimensions
    'area': 195.0,          # Bottom area (m²)
    'ap': 195.0,            # Ponding area (m²)
    'df': 0.05,             # Filter depth (m)
    'dtl': 0.10,            # Transition layer depth (m)
    'dg': 0.20,             # Gravel depth (m)
    'l': 0.35,              # Total depth (m)

    # Hydraulic properties
    'nf': 0.32,             # Filter porosity
    'nt': 0.40,             # Transition porosity
    'ng': 0.35,             # Gravel porosity
    'ks': 0.0005,           # Saturated hydraulic conductivity (m/s)

    # Moisture parameters
    'sh': 0.01,             # Hygroscopic point
    'sw': 0.02,             # Wilting point
    'sfc': 0.08,            # Field capacity
    'ss': 0.09,             # Plant stress point
    'gama': 5.0,            # Curve parameter

    # Ponding zone
    'hover': 0.5,           # Overflow height (m)
    'kweir': 1.3,           # Weir coefficient
    'wweir': 5.0,           # Weir width (m)
    'expweir': 2.5,         # Weir exponent

    # Pipe drainage
    'hpipe': 0.025,         # Pipe height (m)
    'dpipe': 152.4,         # Pipe diameter (mm)
    'cd': 0.13,             # Discharge coefficient
    'eta': 0.23,            # Drainage coefficient

    # Simulation
    'dt': 60,               # Timestep (seconds)
    'kc': 0.0,              # Evapotranspiration constant
}
```

#### Option B: Load from Setup File

```python
# Load parameters from INI file
setup = pdt.read_setup_file("input_parameters.ini")
params = pdt.initialize_parameters(setup)
```

### Step 3: Create Input Data

```python
# Create synthetic rainfall event (10mm over 4 hours)
timesteps = 240  # 4 hours at 1-minute intervals
time_minutes = np.arange(timesteps)

# Triangular rainfall pattern
rainfall_intensity = np.zeros(timesteps)
peak_time = 120  # Peak at 2 hours
peak_intensity = 0.5  # 0.5 mm/min peak

for i, t in enumerate(time_minutes):
    if t <= peak_time:
        # Rising limb
        rainfall_intensity[i] = peak_intensity * (t / peak_time)
    else:
        # Falling limb
        rainfall_intensity[i] = peak_intensity * (1 - (t - peak_time) / peak_time)

# Convert to flow rates (m³/s per m² of surface)
qin = np.zeros(timesteps)  # No external inflow
qrain = rainfall_intensity * params['area'] / (1000 * 60)  # mm/min to m³/s
emax = np.zeros(timesteps)  # No evapotranspiration

print(f"Total rainfall: {np.sum(rainfall_intensity):.1f} mm")
print(f"Peak intensity: {np.max(rainfall_intensity):.2f} mm/min")
```

### Step 4: Run Simulation

```python
# Run the simulation
data, water_balance = pdt.run_simulation(params, qin, qrain, emax)

# Display results
print("\n=== Simulation Results ===")
print(f"Peak outflow: {np.max(data['Qpipe']):.6f} m³/s")
print(f"Total outflow: {water_balance['total_outflow']:.2f} mm")
print(f"Final ponding depth: {data['hp'][-1]:.4f} m")
print(f"Final soil moisture: {data['s'][-1]:.3f}")

# Convert to DataFrame for easy analysis
df = pdt.results_dataframe(data, save=True, filename="first_simulation.csv")
print(f"Results saved to: first_simulation.csv")
```

### Step 5: Basic Visualization

```python
# Simple plotting (if matplotlib available)
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Rainfall plot
    ax1.bar(time_minutes, rainfall_intensity, width=1, alpha=0.7, color='blue')
    ax1.set_ylabel('Rainfall (mm/min)')
    ax1.set_title('Rainfall and Outflow Response')
    ax1.invert_yaxis()

    # Outflow plot
    outflow_lps = np.array(data['Qpipe']) * 1000  # Convert to L/s
    ax2.plot(time_minutes, outflow_lps, 'r-', linewidth=2)
    ax2.set_ylabel('Outflow (L/s)')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('first_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as: first_simulation.png")

except ImportError:
    print("Matplotlib not available. Install with: pip install permeabledt[plots]")
```

## 2. Parameter Calibration Example

### Step 1: Prepare Calibration Data

```python
# Create synthetic observed data for calibration
np.random.seed(42)  # For reproducible results

# Simulate "observed" outflow with some noise
observed_outflow = np.array(data['Qpipe']) * 1000  # Convert to L/s
# Add realistic measurement noise
noise = np.random.normal(0, np.maximum(observed_outflow * 0.1, 0.001))
observed_outflow_noisy = np.maximum(0, observed_outflow + noise)

# Save as CSV file for calibration
import pandas as pd
from datetime import datetime, timedelta

start_time = datetime(2023, 10, 5, 8, 0)  # 8 AM start
timestamps = [start_time + timedelta(minutes=i) for i in range(len(observed_outflow_noisy))]

obs_df = pd.DataFrame({
    'date': timestamps,
    'observed_outflow': observed_outflow_noisy / 35.315  # Convert L/s to ft³/s for permeabledt format
})
obs_df.to_csv('observed_outflow.csv', index=False)

# Create rainfall file
rain_df = pd.DataFrame({
    'date': timestamps,
    'rain': rainfall_intensity / 25.4  # Convert mm to inches for permeabledt format
})

# Save rainfall in .dat format
with open('rainfall.dat', 'w') as f:
    for i, row in rain_df.iterrows():
        f.write(f"{row['date'].strftime('%m/%d/%Y %H:%M')} {row['rain']:.4f}\n")

print("Calibration files created:")
print("- observed_outflow.csv")
print("- rainfall.dat")
```

### Step 2: Create Setup File for Calibration

```python
# Create a setup file with calibration bounds
setup_content = f"""[GENERAL]
Kc = {params['kc']}
Df = {params['df']}
Dtl = {params['dtl']}
Dg = {params['dg']}
nf = {params['nf']}
nt = {params['nt']}
ng = {params['ng']}

[PONDING_ZONE]
Ap = {params['ap']}
Hover = {params['hover']}
Kweir = {params['kweir']}
wWeir = {params['wweir']}
expWeir = {params['expweir']}
Cs = 0
Pp = 0
flagp = 1

[UNSATURATED_ZONE]
A = {params['area']}
husz = 0.05
nusz = {params['nf']}
Ks = {params['ks']}
sh = {params['sh']}
sw = {params['sw']}
sfc = {params['sfc']}
ss = {params['ss']}
gama = {params['gama']}
Kf = 0

[SATURATED_ZONE]
Psz = 0
hpipe = {params['hpipe']}
flagsz = 1
dpipe = {params['dpipe']}
Cd = {params['cd']}
eta = {params['eta']}

[TIMESTEP]
dt = {params['dt']}

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
gama_min = 1
gama_max = 25
hpipe_min = 0
hpipe_max = 0.2
eta_min = 0.001
eta_max = 1
# Optimization settings
pop = 50
gen = 10
"""

with open('setup_calibration.ini', 'w') as f:
    f.write(setup_content)

print("Setup file created: setup_calibration.ini")
```

### Step 3: Run Calibration

```python
# Run parameter calibration
try:
    print("\n=== Starting Calibration ===")
    print("This may take a few minutes...")

    best_params, calibrated_setup, logbook = pdt.run_calibration(
        calibration_rainfall=['rainfall.dat'],
        calibration_observed_data=['observed_outflow.csv'],
        setup_file='setup_calibration.ini',
        output_setup_file='calibrated_parameters.ini',
        logbook_output_path='calibration_log.csv',
        seed=42
    )

    print(f"\n=== Calibration Complete ===")
    print(f"Best fitness (lower is better): {best_params.fitness.values[0]:.4f}")
    print("Calibrated parameters saved to: calibrated_parameters.ini")
    print("Optimization log saved to: calibration_log.csv")

    # Show improvement
    print(f"\nParameter changes:")
    original_setup = pdt.read_setup_file('setup_calibration.ini')
    for section in calibrated_setup.sections():
        if section == 'CALIBRATION':
            continue
        for key in calibrated_setup[section]:
            if key in original_setup[section]:
                old_val = float(original_setup[section][key])
                new_val = float(calibrated_setup[section][key])
                if abs(old_val - new_val) > 1e-6:
                    print(f"  {key}: {old_val:.6f} → {new_val:.6f}")

except ImportError:
    print("Calibration requires DEAP. Install with: pip install permeabledt[calib]")
except Exception as e:
    print(f"Calibration error: {e}")
    print("Using original parameters for next steps...")
```

## 3. Particle Filter Forecasting Example

### Step 1: Prepare Particle Filter Data

```python
# Create forecasting scenario
print("\n=== Particle Filter Setup ===")

# Use first half of data as "observed", second half as "forecast"
forecast_start_time = len(timestamps) // 2
obs_for_pf = obs_df.iloc[:forecast_start_time].copy()
obs_for_pf.to_csv('observed_for_pf.csv', index=False)

# Create TOML configuration for particle filter
toml_config = f"""[components]
model   = "permeabledt.particle_filter.PavementModel"
time    = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0
until = {forecast_start_time - 1}
steps_per_unit = 1

[model]
setup_file = "setup_calibration.ini"
rainfall_file = "rainfall.dat"

[prior]
hp = {{ name = "uniform", args.loc = 0.0, args.scale = 0.05}}
hsz = {{ name = "uniform", args.loc = 1e-6, args.scale = 0.05}}
s = {{ name = "uniform", args.loc = 0.0, args.scale = 0.2}}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "observed_pf.ssv"
weir_k = 0.006
weir_n = 2.5532
head_error_inches = 0.08

[filter]
particles = 1000
prng_seed = 2025
resample.threshold = 0.5
"""

with open('pavement.toml', 'w') as f:
    f.write(toml_config)

# Convert observed data to space-separated format for pypfilt
obs_pf = obs_for_pf.copy()
obs_pf['time'] = range(len(obs_pf))
obs_pf['value'] = obs_pf['observed_outflow'] * 0.028316847  # ft³/s to m³/s
obs_pf[['time', 'value']].to_csv('observed_pf.ssv', sep=' ', index=False)

print("Particle filter files created:")
print("- pavement.toml")
print("- observed_pf.ssv")
```

### Step 2: Run Particle Filter

```python
try:
    import pypfilt

    print("Running particle filter...")

    # Load and run particle filter
    instances = list(pypfilt.load_instances('pavement.toml'))
    instance = instances[0]

    ctx = instance.build_context()
    results = pypfilt.forecast(ctx, [forecast_start_time])

    # Extract results
    est_df = pd.DataFrame(results.estimation.tables['forecast'])
    fcst_df = pd.DataFrame(results.forecasts[forecast_start_time].tables['forecast'])

    # Get median forecast
    forecast_median = (
        fcst_df[fcst_df['prob'] == 50]
        .assign(median=lambda d: (d['ymin'] + d['ymax']) / 2)
    )

    print(f"\n=== Particle Filter Results ===")
    print(f"Estimation completed for {len(est_df)} timesteps")
    print(f"Forecast generated for {len(forecast_median)} timesteps")
    print(f"Forecast range: {forecast_median['median'].min():.6f} - {forecast_median['median'].max():.6f} m³/s")

    # Simple accuracy check
    true_future = observed_outflow_noisy[forecast_start_time:forecast_start_time + len(forecast_median)]
    forecast_values = forecast_median['median'].values * 1000  # Convert to L/s

    if len(true_future) > 0 and len(forecast_values) > 0:
        min_len = min(len(true_future), len(forecast_values))
        rmse = np.sqrt(np.mean((true_future[:min_len] - forecast_values[:min_len])**2))
        print(f"Forecast RMSE: {rmse:.3f} L/s")

except ImportError:
    print("Particle filtering requires pypfilt. Install with: pip install permeabledt[pf]")
except Exception as e:
    print(f"Particle filter error: {e}")
```

## 4. Sensitivity Analysis Example

```python
try:
    print("\n=== Sensitivity Analysis ===")

    # Initialize sensitivity analysis
    sa = pdt.SobolSensitivityAnalysis(
        setup_file='setup_calibration.ini',
        rainfall_file='rainfall.dat'
    )

    # Run analysis with smaller sample for quick demo
    print("Running sensitivity analysis (this may take a moment)...")
    results = sa.run_analysis(
        n_samples=100,  # Small sample for demo (use 1000+ for real analysis)
        metrics=['peak_outflow', 'total_outflow'],
        n_cores=1
    )

    print(f"\n=== Sensitivity Results ===")

    # Show first-order Sobol indices for peak outflow
    peak_s1 = results['peak_outflow']['S1']
    print("First-order sensitivity indices for peak outflow:")
    for param, sensitivity in zip(sa.param_names, peak_s1):
        print(f"  {param}: {sensitivity:.3f}")

    # Find most important parameter
    most_important = sa.param_names[np.argmax(peak_s1)]
    print(f"\nMost important parameter: {most_important} (S1 = {max(peak_s1):.3f})")

except ImportError:
    print("Sensitivity analysis requires SALib. Install with: pip install permeabledt[sensitivity]")
except Exception as e:
    print(f"Sensitivity analysis error: {e}")
```

## 5. Weather Data Integration Example

```python
try:
    print("\n=== Weather Data Integration ===")

    # Initialize HRRR downloader (example coordinates for Salt Lake City)
    downloader = pdt.HRRRAccumulatedPrecipitationDownloader(
        lat=40.7589,
        lon=-111.8883,
        timezone='US/Mountain'
    )

    print("HRRR downloader initialized successfully")
    print("Note: Actual data download requires internet connection and valid dates")
    print("Example usage:")
    print("""
    from datetime import datetime

    # Download forecast data
    forecast_data = downloader.download_forecast(
        start_local=datetime(2023, 10, 5, 0, 0),
        end_local=datetime(2023, 10, 6, 0, 0),
        output_file="hrrr_forecast.dat"
    )
    """)

except ImportError:
    print("Weather data integration requires herbie-data. Install with: pip install permeabledt[weather]")
except Exception as e:
    print(f"Weather data error: {e}")
```

## Summary

Congratulations! You've completed the permeabledt quick start tutorial. You've learned how to:

✅ **Run basic water flow simulations**
✅ **Calibrate model parameters**
✅ **Perform particle filter forecasting**
✅ **Conduct sensitivity analysis**
✅ **Integrate weather data**

### Files Created

During this tutorial, you created several files:
- `first_simulation.csv` - Simulation results
- `first_simulation.png` - Visualization (if matplotlib available)
- `observed_outflow.csv` - Synthetic observed data
- `rainfall.dat` - Rainfall input file
- `setup_calibration.ini` - Model parameters
- `calibrated_parameters.ini` - Optimized parameters
- `calibration_log.csv` - Optimization history
- `pavement.toml` - Particle filter configuration
- `observed_pf.ssv` - Particle filter observations

## Next Steps

Now that you've seen the basics, explore these topics:

### Deep Dive into Features
- **[Water Flow Modeling](user_guide/water_flow.md)** - Advanced simulation options
- **[Parameter Calibration](user_guide/calibration.md)** - Multi-objective optimization
- **[Particle Filtering](user_guide/particle_filtering.md)** - Real-time forecasting
- **[Sensitivity Analysis](user_guide/sensitivity_analysis.md)** - Parameter importance
- **[Visualization](user_guide/plotting.md)** - Advanced plotting options

### Working with Real Data
- **[Data Preparation](data_preparation.md)** - Format your datasets
- **[File Formats](technical/file_formats.md)** - Detailed format specifications
- **[Configuration Files](technical/configuration.md)** - Advanced setup options

### Advanced Applications
- **[Case Studies](examples/case_studies.md)** - Real-world examples
- **[Advanced Tutorials](examples/advanced_tutorials.md)** - Complex workflows
- **[API Reference](api/index.md)** - Complete function documentation

### Get Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/permeabledt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/permeabledt/discussions)
- **Documentation**: Continue reading these docs!

Happy modeling! 🌊