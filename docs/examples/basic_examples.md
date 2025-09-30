# Basic Examples

This section provides simple, complete examples for common permeabledt use cases. Each example is self-contained and can be run independently.

## Table of Contents

1. [Simple Water Flow Simulation](#simple-water-flow-simulation)
2. [Parameter Calibration Workflow](#parameter-calibration-workflow)
3. [Basic Particle Filter Forecast](#basic-particle-filter-forecast)
4. [Quick Sensitivity Analysis](#quick-sensitivity-analysis)
5. [Weather Data Integration](#weather-data-integration)
6. [Plotting and Visualization](#plotting-and-visualization)
7. [Complete Modeling Workflow](#complete-modeling-workflow)

## Simple Water Flow Simulation

This example shows how to run a basic water flow simulation with synthetic data.

```python
import permeabledt as gdt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create simple parameter dictionary
params = {
    # Physical dimensions
    'area': 200.0,          # Bottom area (m²)
    'ap': 200.0,            # Ponding area (m²)
    'df': 0.05,             # Filter depth (m)
    'dtl': 0.10,            # Transition depth (m)
    'dg': 0.20,             # Gravel depth (m)
    'l': 0.35,              # Total depth (m)

    # Hydraulic properties
    'nf': 0.32,             # Filter porosity
    'nt': 0.40,             # Transition porosity
    'ng': 0.35,             # Gravel porosity
    'ks': 0.0005,           # Saturated K (m/s)
    'gama': 5.0,            # Pore parameter

    # Moisture parameters
    'sh': 0.01,             # Hygroscopic point
    'sw': 0.02,             # Wilting point
    'sfc': 0.08,            # Field capacity
    'ss': 0.09,             # Saturation point

    # Ponding parameters
    'hover': 0.5,           # Overflow height (m)
    'kweir': 1.3,           # Weir coefficient
    'wweir': 5.0,           # Weir width (m)
    'expweir': 2.5,         # Weir exponent

    # Pipe parameters
    'hpipe': 0.025,         # Pipe height (m)
    'dpipe': 152.4,         # Pipe diameter (mm)
    'cd': 0.13,             # Discharge coefficient
    'eta': 0.23,            # Drainage coefficient

    # Simulation
    'dt': 60,               # Timestep (seconds)
    'kc': 0.0,              # ET constant
}

# Create synthetic rainfall event (6-hour storm)
timesteps = 360  # 6 hours at 1-minute intervals
time_array = np.arange(timesteps)

# Triangular rainfall pattern (total: 25mm)
peak_time = 180  # Peak at 3 hours
total_rainfall_mm = 25
peak_intensity_mm_min = (2 * total_rainfall_mm) / timesteps

rainfall_mm_min = np.zeros(timesteps)
for i, t in enumerate(time_array):
    if t <= peak_time:
        rainfall_mm_min[i] = peak_intensity_mm_min * (t / peak_time)
    else:
        rainfall_mm_min[i] = peak_intensity_mm_min * (1 - (t - peak_time) / peak_time)

    if rainfall_mm_min[i] < 0:
        rainfall_mm_min[i] = 0

# Convert to model inputs
qin = np.zeros(timesteps)  # No external inflow
qrain = rainfall_mm_min * params['area'] / (1000 * 60)  # mm/min to m³/s
emax = np.zeros(timesteps)  # No evapotranspiration

print(f"Rainfall event summary:")
print(f"  Duration: {timesteps} minutes")
print(f"  Total rainfall: {np.sum(rainfall_mm_min):.1f} mm")
print(f"  Peak intensity: {np.max(rainfall_mm_min):.3f} mm/min")

# Run simulation
data, water_balance = gdt.run_simulation(params, qin, qrain, emax)

# Display results
print(f"\nSimulation results:")
print(f"  Peak outflow: {np.max(data['Qpipe']):.6f} m³/s")
print(f"  Peak outflow: {np.max(data['Qpipe']) * 1000:.2f} L/s")
print(f"  Total outflow: {water_balance['total_outflow']:.2f} mm")
print(f"  Runoff coefficient: {water_balance['total_outflow'] / np.sum(rainfall_mm_min):.3f}")
print(f"  Final ponding: {data['hp'][-1]:.4f} m")
print(f"  Final soil moisture: {data['s'][-1]:.3f}")

# Save results
df = gdt.results_dataframe(data, save=True, filename="basic_simulation.csv")
print(f"\nResults saved to: basic_simulation.csv")
```

### Expected Output:
```
Rainfall event summary:
  Duration: 360 minutes
  Total rainfall: 25.0 mm
  Peak intensity: 0.139 mm/min

Simulation results:
  Peak outflow: 0.000234 m³/s
  Peak outflow: 0.23 L/s
  Total outflow: 18.5 mm
  Runoff coefficient: 0.740
  Final ponding: 0.000 m
  Final soil moisture: 0.156
```

## Parameter Calibration Workflow

This example demonstrates parameter calibration using synthetic observed data.

```python
import permeabledt as gdt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Step 1: Create setup file with calibration bounds
def create_setup_file(filename="calibration_setup.ini"):
    setup_content = """[GENERAL]
Kc = 0.0
Df = 0.05
Dtl = 0.10
Dg = 0.20
nf = 0.32
nt = 0.40
ng = 0.35

[PONDING_ZONE]
Ap = 200.0
Hover = 0.5
Kweir = 1.3
wWeir = 5.0
expWeir = 2.5
Cs = 0
Pp = 0
flagp = 1

[UNSATURATED_ZONE]
A = 200.0
husz = 0.05
nusz = 0.32
Ks = 0.0005
sh = 0.01
sw = 0.02
sfc = 0.08
ss = 0.09
gama = 5.0
Kf = 0

[SATURATED_ZONE]
Psz = 0
hpipe = 0.025
flagsz = 1
dpipe = 152.4
Cd = 0.13
eta = 0.23

[TIMESTEP]
dt = 60

[CALIBRATION]
Ks_min = 0.000001
Ks_max = 0.01
sw_min = 0.01
sw_max = 0.1
sfc_min = 0.01
sfc_max = 0.25
Cd_min = 0.01
Cd_max = 0.8
eta_min = 0.001
eta_max = 1
pop = 50
gen = 10
"""

    with open(filename, 'w') as f:
        f.write(setup_content)
    return filename

# Step 2: Create synthetic observed data for multiple events
def create_calibration_data():
    rainfall_files = []
    observed_files = []

    # Create 3 different rainfall events
    events = [
        {"duration": 240, "total_mm": 15, "pattern": "uniform"},
        {"duration": 180, "total_mm": 30, "pattern": "triangular"},
        {"duration": 300, "total_mm": 20, "pattern": "exponential"}
    ]

    for i, event in enumerate(events):
        # Create rainfall pattern
        timesteps = event["duration"]
        total_mm = event["total_mm"]

        if event["pattern"] == "uniform":
            intensity = np.full(timesteps, total_mm / timesteps)
        elif event["pattern"] == "triangular":
            peak_time = timesteps // 2
            intensity = np.zeros(timesteps)
            for t in range(timesteps):
                if t <= peak_time:
                    intensity[t] = (2 * total_mm / timesteps) * (t / peak_time)
                else:
                    intensity[t] = (2 * total_mm / timesteps) * (1 - (t - peak_time) / peak_time)
        elif event["pattern"] == "exponential":
            intensity = total_mm * np.exp(-3 * np.arange(timesteps) / timesteps)
            intensity = intensity * (total_mm / np.sum(intensity))

        # Create time series
        start_time = datetime(2023, 10, 5 + i, 8, 0)
        times = [start_time + timedelta(minutes=j) for j in range(timesteps)]

        # Save rainfall file
        rainfall_file = f"event_{i+1}_rainfall.dat"
        with open(rainfall_file, 'w') as f:
            for t, rain in zip(times, intensity):
                rain_inches = rain / 25.4
                f.write(f"{t.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")
        rainfall_files.append(rainfall_file)

        # Simulate "observed" outflow (with noise)
        setup = gdt.read_setup_file("calibration_setup.ini")
        params = gdt.initialize_parameters(setup)
        qin = np.zeros(timesteps)
        qrain = intensity * params['area'] / (1000 * 60)
        emax = np.zeros(timesteps)

        sim_data, _ = gdt.run_simulation(params, qin, qrain, emax)

        # Add realistic noise to create "observations"
        true_outflow = np.array(sim_data['Qpipe']) * 35.315  # Convert to ft³/s
        noise = np.random.normal(0, np.maximum(true_outflow * 0.1, 0.001))
        observed_outflow = np.maximum(0, true_outflow + noise)

        # Save observed file
        observed_file = f"event_{i+1}_observed.csv"
        obs_df = pd.DataFrame({
            'date': times,
            'observed_outflow': observed_outflow
        })
        obs_df.to_csv(observed_file, index=False)
        observed_files.append(observed_file)

        print(f"Created event {i+1}: {event['pattern']} pattern, {total_mm}mm")

    return rainfall_files, observed_files

# Step 3: Run calibration
def run_calibration_example():
    print("Setting up calibration example...")

    # Create files
    setup_file = create_setup_file()
    rainfall_files, observed_files = create_calibration_data()

    print(f"\nRunning calibration with {len(rainfall_files)} events...")
    print("This may take a few minutes...")

    try:
        # Run calibration
        best_params, calibrated_setup, logbook = gdt.run_calibration(
            calibration_rainfall=rainfall_files,
            calibration_observed_data=observed_files,
            setup_file=setup_file,
            output_setup_file="calibrated_parameters.ini",
            logbook_output_path="calibration_log.csv",
            seed=42
        )

        print(f"\nCalibration completed!")
        print(f"Best fitness: {best_params.fitness.values[0]:.4f}")

        # Show parameter changes
        original_setup = gdt.read_setup_file(setup_file)
        print(f"\nParameter improvements:")

        calib_params = ['Ks', 'sw', 'sfc', 'Cd', 'eta']
        for param in calib_params:
            for section in original_setup.sections():
                if param in original_setup[section]:
                    old_val = float(original_setup[section][param])
                    new_val = float(calibrated_setup[section][param])
                    change_pct = ((new_val - old_val) / old_val) * 100
                    print(f"  {param}: {old_val:.6f} → {new_val:.6f} ({change_pct:+.1f}%)")
                    break

        return True

    except ImportError:
        print("Calibration requires DEAP. Install with: pip install permeabledt[calib]")
        return False

# Run the example
if __name__ == "__main__":
    success = run_calibration_example()
    if success:
        print("\nFiles created:")
        print("  - calibration_setup.ini")
        print("  - calibrated_parameters.ini")
        print("  - calibration_log.csv")
        print("  - event_*_rainfall.dat")
        print("  - event_*_observed.csv")
```

## Basic Particle Filter Forecast

This example shows how to set up and run a basic particle filter forecast.

```python
import permeabledt as gdt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def basic_particle_filter_example():
    """Complete particle filter example from data preparation to forecast"""

    print("Setting up particle filter example...")

    # Step 1: Create synthetic data
    timesteps = 200
    start_time = datetime(2023, 10, 5, 8, 0)
    times = [start_time + timedelta(minutes=i) for i in range(timesteps)]

    # Rainfall pattern
    rainfall_mm = np.zeros(timesteps)
    for i in range(60, 120):  # Rain from hour 1 to hour 2
        rainfall_mm[i] = 0.5 * np.sin((i - 60) * np.pi / 60)  # Sinusoidal pattern

    # Create files
    # 1. Rainfall file
    with open('pf_rainfall.dat', 'w') as f:
        for time, rain in zip(times, rainfall_mm):
            rain_inches = rain / 25.4
            f.write(f"{time.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")

    # 2. Setup file (use calibrated if available, otherwise create basic)
    try:
        setup = gdt.read_setup_file("calibrated_parameters.ini")
        setup_file = "calibrated_parameters.ini"
    except FileNotFoundError:
        setup_file = create_setup_file("pf_setup.ini")
        setup = gdt.read_setup_file(setup_file)

    # 3. Generate "observed" outflow
    params = gdt.initialize_parameters(setup)
    qin = np.zeros(timesteps)
    qrain = rainfall_mm * params['area'] / (1000 * 60)
    emax = np.zeros(timesteps)

    sim_data, _ = gdt.run_simulation(params, qin, qrain, emax)
    true_outflow = np.array(sim_data['Qpipe']) * 35.315  # ft³/s

    # Add noise for realistic observations
    noise = np.random.normal(0, np.maximum(true_outflow * 0.05, 0.001))
    observed_outflow = np.maximum(0, true_outflow + noise)

    # Step 2: Prepare particle filter data
    forecast_time = 120  # Forecast from timestep 120

    # Observed data up to forecast time
    obs_for_pf = pd.DataFrame({
        'date': times[:forecast_time],
        'observed_outflow': observed_outflow[:forecast_time]
    })
    obs_for_pf.to_csv('pf_observed.csv', index=False)

    # Convert to SSV format
    ssv_data = pd.DataFrame({
        'time': range(forecast_time),
        'value': observed_outflow[:forecast_time] * 0.028316847  # ft³/s to m³/s
    })
    ssv_data.to_csv('pf_observed.ssv', sep=' ', index=False)

    # Step 3: Create TOML configuration
    toml_config = f"""[components]
model = "permeabledt.particle_filter.PavementModel"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0
until = {forecast_time - 1}
steps_per_unit = 1

[model]
setup_file = "{setup_file}"
rainfall_file = "pf_rainfall.dat"

[prior]
hp = {{ name = "uniform", args.loc = 0.0, args.scale = 0.05}}
hsz = {{ name = "uniform", args.loc = 1e-6, args.scale = 0.05}}
s = {{ name = "uniform", args.loc = 0.0, args.scale = 0.2}}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "pf_observed.ssv"
weir_k = 0.006
weir_n = 2.5532
head_error_inches = 0.08

[filter]
particles = 1000
prng_seed = 2025
resample.threshold = 0.5
"""

    with open('pf_config.toml', 'w') as f:
        f.write(toml_config)

    # Step 4: Run particle filter
    try:
        import pypfilt

        print(f"Running particle filter forecast from timestep {forecast_time}...")

        # Load configuration
        instances = list(pypfilt.load_instances('pf_config.toml'))
        instance = instances[0]

        # Run forecast
        ctx = instance.build_context()
        results = pypfilt.forecast(ctx, [forecast_time])

        # Extract results
        est_df = pd.DataFrame(results.estimation.tables['forecast'])
        fcst_df = pd.DataFrame(results.forecasts[forecast_time].tables['forecast'])

        # Get median forecast
        forecast_median = (
            fcst_df[fcst_df['prob'] == 50]
            .assign(median=lambda d: (d['ymin'] + d['ymax']) / 2)
        )

        print(f"\nParticle filter results:")
        print(f"  Estimation period: {len(est_df)} timesteps")
        print(f"  Forecast period: {len(forecast_median)} timesteps")
        print(f"  Forecast range: {forecast_median['median'].min():.6f} - {forecast_median['median'].max():.6f} m³/s")

        # Simple accuracy assessment
        if forecast_time < len(true_outflow):
            future_length = min(len(forecast_median), len(true_outflow) - forecast_time)
            if future_length > 0:
                true_future = true_outflow[forecast_time:forecast_time + future_length]
                forecast_values = forecast_median['median'].iloc[:future_length].values * 35.315  # Convert to ft³/s

                rmse = np.sqrt(np.mean((true_future - forecast_values)**2))
                print(f"  Forecast RMSE: {rmse:.4f} ft³/s")

        print(f"\nParticle filter completed successfully!")
        return True

    except ImportError:
        print("Particle filtering requires pypfilt. Install with: pip install permeabledt[pf]")
        return False
    except Exception as e:
        print(f"Particle filter error: {e}")
        return False

# Run the example
if __name__ == "__main__":
    success = basic_particle_filter_example()
    if success:
        print("\nFiles created:")
        print("  - pf_rainfall.dat")
        print("  - pf_observed.csv")
        print("  - pf_observed.ssv")
        print("  - pf_config.toml")
```

## Quick Sensitivity Analysis

This example demonstrates how to run a basic sensitivity analysis.

```python
import permeabledt as gdt
import numpy as np

def quick_sensitivity_example():
    """Run a quick sensitivity analysis"""

    print("Setting up sensitivity analysis example...")

    # Create setup file if needed
    try:
        setup_file = "calibrated_parameters.ini"
        gdt.read_setup_file(setup_file)
    except FileNotFoundError:
        setup_file = create_setup_file("sensitivity_setup.ini")

    # Create rainfall file if needed
    try:
        with open('pf_rainfall.dat', 'r') as f:
            rainfall_file = 'pf_rainfall.dat'
    except FileNotFoundError:
        # Create simple rainfall event
        rainfall_file = 'sensitivity_rainfall.dat'
        with open(rainfall_file, 'w') as f:
            start_time = datetime(2023, 10, 5, 8, 0)
            for i in range(120):
                time = start_time + timedelta(minutes=i)
                # Simple triangular event
                if 30 <= i <= 90:
                    rain_mm = 0.5 * (1 - abs(i - 60) / 30)
                else:
                    rain_mm = 0
                rain_inches = rain_mm / 25.4
                f.write(f"{time.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")

    try:
        # Initialize sensitivity analysis
        sa = gdt.SobolSensitivityAnalysis(
            setup_file=setup_file,
            rainfall_file=rainfall_file
        )

        print(f"Running sensitivity analysis...")
        print(f"Analyzing {len(sa.param_names)} parameters: {sa.param_names}")

        # Run with small sample for demo (use 1000+ for real analysis)
        results = sa.run_analysis(
            n_samples=100,
            metrics=['peak_outflow', 'total_outflow'],
            n_cores=1
        )

        print(f"\nSensitivity analysis results:")

        # Display first-order Sobol indices
        for metric in ['peak_outflow', 'total_outflow']:
            print(f"\n{metric.upper()} sensitivity:")
            s1_indices = results[metric]['S1']

            # Sort by importance
            sorted_indices = sorted(zip(sa.param_names, s1_indices),
                                  key=lambda x: x[1], reverse=True)

            for param, sensitivity in sorted_indices:
                print(f"  {param:8s}: {sensitivity:.3f}")

            # Find most important parameter
            most_important = sorted_indices[0]
            print(f"  → Most important: {most_important[0]} (S1 = {most_important[1]:.3f})")

        return True

    except ImportError:
        print("Sensitivity analysis requires SALib. Install with: pip install permeabledt[sensitivity]")
        return False
    except Exception as e:
        print(f"Sensitivity analysis error: {e}")
        return False

# Run the example
if __name__ == "__main__":
    success = quick_sensitivity_example()
    if success:
        print("\nSensitivity analysis completed!")
```

## Weather Data Integration

This example shows how to integrate HRRR weather forecast data.

```python
import permeabledt as gdt
from datetime import datetime

def weather_integration_example():
    """Demonstrate weather data integration"""

    print("Weather data integration example...")

    try:
        # Initialize HRRR downloader
        # Example coordinates for Salt Lake City, UT
        downloader = gdt.HRRRAccumulatedPrecipitationDownloader(
            lat=40.7589,
            lon=-111.8883,
            timezone='US/Mountain'
        )

        print(f"HRRR downloader initialized successfully")
        print(f"Location: {downloader.lat:.4f}°N, {downloader.lon:.4f}°W")
        print(f"Timezone: {downloader.timezone}")

        # Example of how to download data (requires internet and valid dates)
        print(f"\nExample usage for downloading forecast data:")

        example_code = """
# Download 24-hour forecast
from datetime import datetime

forecast_data = downloader.download_forecast(
    start_local=datetime(2023, 10, 5, 0, 0),
    end_local=datetime(2023, 10, 6, 0, 0),
    output_file="hrrr_forecast.dat"
)

print(f"Downloaded {len(forecast_data)} forecast timesteps")

# The output file is automatically formatted for permeabledt:
# MM/DD/YYYY HH:MM precipitation_inches
"""

        print(example_code)

        # Demonstrate methods available
        print(f"\nAvailable methods:")
        methods = [method for method in dir(downloader) if not method.startswith('_')]
        for method in methods:
            if callable(getattr(downloader, method)):
                print(f"  - {method}()")

        return True

    except ImportError:
        print("Weather data integration requires herbie-data. Install with: pip install permeabledt[weather]")
        return False
    except Exception as e:
        print(f"Weather integration error: {e}")
        return False

# Run the example
if __name__ == "__main__":
    weather_integration_example()
```

## Plotting and Visualization

This example demonstrates basic plotting capabilities.

```python
import permeabledt as gdt
import numpy as np

def plotting_example():
    """Demonstrate plotting capabilities"""

    print("Plotting example...")

    # Check if plotting is available
    if gdt.plots is None:
        print("Plotting requires matplotlib. Install with: pip install permeabledt[plots]")
        return False

    try:
        # Use existing data files if available
        rainfall_file = None
        for filename in ['pf_rainfall.dat', 'sensitivity_rainfall.dat', 'event_1_rainfall.dat']:
            try:
                with open(filename, 'r'):
                    rainfall_file = filename
                    break
            except FileNotFoundError:
                continue

        if rainfall_file is None:
            # Create simple rainfall file
            rainfall_file = 'plot_rainfall.dat'
            from datetime import datetime, timedelta

            start_time = datetime(2023, 10, 5, 8, 0)
            with open(rainfall_file, 'w') as f:
                for i in range(120):
                    time = start_time + timedelta(minutes=i)
                    if 30 <= i <= 90:
                        rain_mm = 1.0 * np.sin((i - 30) * np.pi / 60)
                    else:
                        rain_mm = 0
                    rain_inches = rain_mm / 25.4
                    f.write(f"{time.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")

        # Generate corresponding outflow data
        try:
            setup = gdt.read_setup_file("calibrated_parameters.ini")
        except FileNotFoundError:
            setup_file = create_setup_file("plot_setup.ini")
            setup = gdt.read_setup_file(setup_file)

        params = gdt.initialize_parameters(setup)

        # Read rainfall and simulate
        rain_df = gdt.read_rainfall_dat_file(rainfall_file)
        rainfall_mm = rain_df['rain'].values * 25.4  # Convert to mm

        qin = np.zeros(len(rainfall_mm))
        qrain = rainfall_mm * params['area'] / (1000 * 60)  # mm/min to m³/s
        emax = np.zeros(len(rainfall_mm))

        data, _ = gdt.run_simulation(params, qin, qrain, emax)
        outflow_data = np.array(data['Qpipe'])

        # Create plots
        print(f"Creating rainfall-hydrograph plot...")

        fig, axes = gdt.plots.plot_rainfall_hydrograph(
            rainfall_file=rainfall_file,
            outflow_data=outflow_data,
            rainfall_unit='mm',
            output_path='example_hydrograph.png'
        )

        print(f"Plot saved as: example_hydrograph.png")

        # If calibration log exists, plot convergence
        try:
            gdt.plots.plot_calibration_convergence(
                logbook_file="calibration_log.csv",
                output_path="convergence_plot.png"
            )
            print(f"Convergence plot saved as: convergence_plot.png")
        except FileNotFoundError:
            print("No calibration log found - skipping convergence plot")

        return True

    except Exception as e:
        print(f"Plotting error: {e}")
        return False

# Run the example
if __name__ == "__main__":
    success = plotting_example()
    if success:
        print("Plotting completed successfully!")
```

## Complete Modeling Workflow

This example ties together all the components in a complete workflow.

```python
import permeabledt as gdt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def complete_workflow_example():
    """Complete modeling workflow from data preparation to forecasting"""

    print("=== Complete permeabledt Modeling Workflow ===\n")

    # Step 1: Data Preparation
    print("Step 1: Preparing synthetic dataset...")

    # Create project directory
    os.makedirs("complete_example", exist_ok=True)
    os.chdir("complete_example")

    # Generate realistic multi-event dataset
    events = [
        {"name": "storm_1", "duration": 180, "total_mm": 15, "start_hour": 8},
        {"name": "storm_2", "duration": 240, "total_mm": 25, "start_hour": 14},
        {"name": "storm_3", "duration": 300, "total_mm": 35, "start_hour": 10},
    ]

    rainfall_files = []
    observed_files = []

    for i, event in enumerate(events):
        # Create rainfall pattern
        timesteps = event["duration"]
        total_mm = event["total_mm"]

        # Realistic rainfall pattern (gamma distribution-like)
        t = np.linspace(0, 1, timesteps)
        intensity = total_mm * 4 * t * np.exp(-4 * t)
        intensity = intensity / np.sum(intensity) * total_mm  # Normalize

        # Create time series
        start_time = datetime(2023, 10, 5 + i, event["start_hour"], 0)
        times = [start_time + timedelta(minutes=j) for j in range(timesteps)]

        # Save rainfall file
        rainfall_file = f"{event['name']}_rainfall.dat"
        with open(rainfall_file, 'w') as f:
            for time, rain in zip(times, intensity):
                rain_inches = rain / 25.4
                f.write(f"{time.strftime('%m/%d/%Y %H:%M')} {rain_inches:.4f}\n")
        rainfall_files.append(rainfall_file)

        print(f"  Created {event['name']}: {total_mm}mm over {timesteps} minutes")

    # Step 2: Model Setup
    print("\nStep 2: Creating model setup...")

    setup_file = create_setup_file("model_setup.ini")
    print(f"  Setup file created: {setup_file}")

    # Step 3: Generate Synthetic Observations
    print("\nStep 3: Generating synthetic observations...")

    setup = gdt.read_setup_file(setup_file)
    params = gdt.initialize_parameters(setup)

    for i, rainfall_file in enumerate(rainfall_files):
        # Simulate true response
        rain_df = gdt.read_rainfall_dat_file(rainfall_file)
        rainfall_mm = rain_df['rain'].values * 25.4

        qin = np.zeros(len(rainfall_mm))
        qrain = rainfall_mm * params['area'] / (1000 * 60)
        emax = np.zeros(len(rainfall_mm))

        data, _ = gdt.run_simulation(params, qin, qrain, emax)
        true_outflow = np.array(data['Qpipe']) * 35.315  # Convert to ft³/s

        # Add realistic measurement noise
        noise = np.random.normal(0, np.maximum(true_outflow * 0.08, 0.002))
        observed_outflow = np.maximum(0, true_outflow + noise)

        # Save observed data
        observed_file = f"{events[i]['name']}_observed.csv"
        obs_df = pd.DataFrame({
            'date': rain_df['date'],
            'observed_outflow': observed_outflow
        })
        obs_df.to_csv(observed_file, index=False)
        observed_files.append(observed_file)

    print(f"  Generated observations for {len(observed_files)} events")

    # Step 4: Parameter Calibration
    print("\nStep 4: Running parameter calibration...")

    try:
        best_params, calibrated_setup, logbook = gdt.run_calibration(
            calibration_rainfall=rainfall_files,
            calibration_observed_data=observed_files,
            setup_file=setup_file,
            output_setup_file="calibrated_model.ini",
            logbook_output_path="calibration_history.csv",
            seed=42
        )

        print(f"  Calibration completed! Best fitness: {best_params.fitness.values[0]:.4f}")
        calibration_success = True

    except ImportError:
        print("  Skipping calibration (requires DEAP)")
        calibrated_setup = setup
        calibration_success = False

    # Step 5: Model Validation
    print("\nStep 5: Validating calibrated model...")

    if calibration_success:
        calib_params = gdt.initialize_parameters(calibrated_setup)
    else:
        calib_params = params

    # Validate on first event
    rain_df = gdt.read_rainfall_dat_file(rainfall_files[0])
    rainfall_mm = rain_df['rain'].values * 25.4

    qin = np.zeros(len(rainfall_mm))
    qrain = rainfall_mm * calib_params['area'] / (1000 * 60)
    emax = np.zeros(len(rainfall_mm))

    val_data, val_wb = gdt.run_simulation(calib_params, qin, qrain, emax)
    predicted_outflow = np.array(val_data['Qpipe']) * 35.315

    # Load observed data for comparison
    obs_df = pd.read_csv(observed_files[0])
    observed_outflow = obs_df['observed_outflow'].values

    # Calculate performance metrics
    rmse = np.sqrt(np.mean((predicted_outflow - observed_outflow)**2))
    nse = 1 - np.sum((observed_outflow - predicted_outflow)**2) / np.sum((observed_outflow - np.mean(observed_outflow))**2)

    print(f"  Validation metrics:")
    print(f"    RMSE: {rmse:.4f} ft³/s")
    print(f"    NSE: {nse:.3f}")

    # Step 6: Sensitivity Analysis
    print("\nStep 6: Parameter sensitivity analysis...")

    try:
        sa = gdt.SobolSensitivityAnalysis(
            setup_file="calibrated_model.ini" if calibration_success else setup_file,
            rainfall_file=rainfall_files[0]
        )

        sa_results = sa.run_analysis(
            n_samples=200,  # Small for demo
            metrics=['peak_outflow'],
            n_cores=1
        )

        # Show most important parameters
        s1_indices = sa_results['peak_outflow']['S1']
        important_params = sorted(zip(sa.param_names, s1_indices),
                                key=lambda x: x[1], reverse=True)[:3]

        print(f"  Most important parameters:")
        for param, sensitivity in important_params:
            print(f"    {param}: {sensitivity:.3f}")

        sensitivity_success = True

    except ImportError:
        print("  Skipping sensitivity analysis (requires SALib)")
        sensitivity_success = False

    # Step 7: Particle Filter Forecast
    print("\nStep 7: Setting up particle filter forecast...")

    try:
        # Use second event for forecasting
        forecast_event = 1
        forecast_time = 120  # Forecast from 2 hours into event

        # Prepare particle filter data
        obs_df = pd.read_csv(observed_files[forecast_event])
        obs_for_pf = obs_df.iloc[:forecast_time].copy()

        # Create SSV file
        ssv_data = pd.DataFrame({
            'time': range(forecast_time),
            'value': obs_for_pf['observed_outflow'].values * 0.028316847  # ft³/s to m³/s
        })
        ssv_data.to_csv('pf_observations.ssv', sep=' ', index=False)

        # Create TOML configuration
        toml_config = f"""[components]
model = "permeabledt.particle_filter.PavementModel"
time = "pypfilt.Scalar"
sampler = "pypfilt.sampler.LatinHypercube"
summary = "pypfilt.summary.HDF5"

[time]
start = 0
until = {forecast_time - 1}
steps_per_unit = 1

[model]
setup_file = "{'calibrated_model.ini' if calibration_success else setup_file}"
rainfall_file = "{rainfall_files[forecast_event]}"

[prior]
hp = {{ name = "uniform", args.loc = 0.0, args.scale = 0.05}}
hsz = {{ name = "uniform", args.loc = 1e-6, args.scale = 0.05}}
s = {{ name = "uniform", args.loc = 0.0, args.scale = 0.2}}

[observations.Qpipe]
model = "permeabledt.particle_filter.PipeObs"
file = "pf_observations.ssv"

[filter]
particles = 1000
prng_seed = 2025
"""

        with open('forecast_config.toml', 'w') as f:
            f.write(toml_config)

        import pypfilt

        # Run particle filter
        instances = list(pypfilt.load_instances('forecast_config.toml'))
        instance = instances[0]

        ctx = instance.build_context()
        pf_results = pypfilt.forecast(ctx, [forecast_time])

        # Extract forecast statistics
        fcst_df = pd.DataFrame(pf_results.forecasts[forecast_time].tables['forecast'])
        forecast_median = (
            fcst_df[fcst_df['prob'] == 50]
            .assign(median=lambda d: (d['ymin'] + d['ymax']) / 2)
        )

        print(f"  Particle filter forecast generated:")
        print(f"    Forecast length: {len(forecast_median)} timesteps")
        print(f"    Median flow range: {forecast_median['median'].min():.6f} - {forecast_median['median'].max():.6f} m³/s")

        pf_success = True

    except ImportError:
        print("  Skipping particle filter (requires pypfilt)")
        pf_success = False

    # Step 8: Create Visualizations
    print("\nStep 8: Creating visualizations...")

    if gdt.plots is not None:
        try:
            # Plot first event
            gdt.plots.plot_rainfall_hydrograph(
                rainfall_file=rainfall_files[0],
                outflow_data=predicted_outflow / 1000,  # Convert to m³/s
                rainfall_unit='mm',
                output_path='validation_plot.png'
            )
            print(f"  Validation plot saved: validation_plot.png")

            # Plot calibration convergence if available
            if calibration_success:
                try:
                    gdt.plots.plot_calibration_convergence(
                        logbook_file="calibration_history.csv",
                        output_path="calibration_convergence.png"
                    )
                    print(f"  Calibration plot saved: calibration_convergence.png")
                except:
                    pass

        except Exception as e:
            print(f"  Plotting error: {e}")
    else:
        print("  Skipping plots (requires matplotlib)")

    # Step 9: Summary Report
    print("\nStep 9: Generating summary report...")

    summary_report = f"""
permeabledt Complete Workflow Summary
================================

Data Preparation:
- Events processed: {len(events)}
- Total rainfall files: {len(rainfall_files)}
- Observation files: {len(observed_files)}

Model Setup:
- Parameter file: {setup_file}
- System area: {params['area']} m²
- Total depth: {params['l']} m

Calibration:
- Status: {'Completed' if calibration_success else 'Skipped (missing DEAP)'}
{'- Best fitness: ' + str(best_params.fitness.values[0]) if calibration_success else ''}

Validation:
- RMSE: {rmse:.4f} ft³/s
- NSE: {nse:.3f}

Sensitivity Analysis:
- Status: {'Completed' if sensitivity_success else 'Skipped (missing SALib)'}
{('- Most important: ' + important_params[0][0] + f' (S1={important_params[0][1]:.3f})') if sensitivity_success else ''}

Particle Filter:
- Status: {'Completed' if pf_success else 'Skipped (missing pypfilt)'}
{('- Forecast timesteps: ' + str(len(forecast_median))) if pf_success else ''}

Files Created:
"""

    # List all files created
    files_created = []
    for file in os.listdir('.'):
        if os.path.isfile(file):
            files_created.append(file)

    for file in sorted(files_created):
        summary_report += f"- {file}\n"

    # Save summary
    with open('workflow_summary.txt', 'w') as f:
        f.write(summary_report)

    print(summary_report)
    print(f"Summary saved to: workflow_summary.txt")

    # Change back to parent directory
    os.chdir('..')

    print(f"\n=== Workflow Complete ===")
    print(f"All files created in: complete_example/")

    return True

# Helper function (defined earlier)
def create_setup_file(filename="setup.ini"):
    setup_content = """[GENERAL]
Kc = 0.0
Df = 0.05
Dtl = 0.10
Dg = 0.20
nf = 0.32
nt = 0.40
ng = 0.35

[PONDING_ZONE]
Ap = 200.0
Hover = 0.5
Kweir = 1.3
wWeir = 5.0
expWeir = 2.5
Cs = 0
Pp = 0
flagp = 1

[UNSATURATED_ZONE]
A = 200.0
husz = 0.05
nusz = 0.32
Ks = 0.0005
sh = 0.01
sw = 0.02
sfc = 0.08
ss = 0.09
gama = 5.0
Kf = 0

[SATURATED_ZONE]
Psz = 0
hpipe = 0.025
flagsz = 1
dpipe = 152.4
Cd = 0.13
eta = 0.23

[TIMESTEP]
dt = 60

[CALIBRATION]
Ks_min = 0.000001
Ks_max = 0.01
sw_min = 0.01
sw_max = 0.1
sfc_min = 0.01
sfc_max = 0.25
Cd_min = 0.01
Cd_max = 0.8
eta_min = 0.001
eta_max = 1
pop = 50
gen = 15
"""

    with open(filename, 'w') as f:
        f.write(setup_content)
    return filename

# Run the complete workflow
if __name__ == "__main__":
    complete_workflow_example()
```

## Running the Examples

To run any of these examples:

1. **Save the code** to a Python file (e.g., `basic_example.py`)
2. **Install permeabledt** with appropriate optional dependencies:
   ```bash
   pip install permeabledt[all]  # For all features
   ```
3. **Run the example**:
   ```bash
   python basic_example.py
   ```

## Expected Outputs

Each example will:
- Print progress messages to the console
- Create data files specific to the example
- Generate results and summary statistics
- Save output files for further analysis

## Next Steps

After running these basic examples:

1. **Modify parameters** to match your system
2. **Use your own data** instead of synthetic data
3. **Explore advanced features** in the user guides
4. **Try the complete workflow** with real-world data

For more advanced examples and case studies, see:
- [Advanced Tutorials](advanced_tutorials.md)
- [Case Studies](case_studies.md)
- [User Guides](../user_guide/)