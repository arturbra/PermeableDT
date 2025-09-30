# Plotting API

Complete reference for visualization and plotting functions in permeabledt.

## Overview

The plotting module (`permeabledt.plots`) provides comprehensive visualization tools for permeable pavement modeling results. It includes functions for time series plots, hydrograph analysis, parameter visualization, and publication-quality figures.

## Installation Requirements

Plotting functions require matplotlib:

```bash
# Install with plotting support
pip install permeabledt[plots]

# Or install matplotlib manually
pip install matplotlib
```

The plotting module is conditionally imported and available as `permeabledt.plots` if matplotlib is installed.

## Availability Check

```python
import permeabledt as pdt

if pdt.plots is not None:
    print("Plotting functions available")
    # Use plotting functions
    pdt.plots.plot_rainfall_hydrograph(...)
else:
    print("Install matplotlib for plotting: pip install permeabledt[plots]")
```

## Core Plotting Functions

### plot_rainfall_hydrograph()

Create rainfall and hydrograph plots for simulation results.

```python
def plot_rainfall_hydrograph(rainfall_file, simulation_data,
                           observed_file=None, title=None,
                           save_plot=False, output_file=None,
                           figsize=(12, 8), show_components=False)
```

**Parameters:**
- `rainfall_file` (str): Path to rainfall data file (.dat format)
- `simulation_data` (dict): Simulation results from `run_simulation()`
- `observed_file` (str, optional): Path to observed outflow data (.csv)
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size in inches
- `show_components` (bool): Show individual flow components

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects

**Example - Basic Plot:**
```python
import permeabledt as pdt

# Run simulation
setup = pdt.read_setup_file("pavement.ini")
params = pdt.initialize_parameters(setup)
data, wb = pdt.run_model(params, "rainfall_event.dat")

# Create plot
if pdt.plots is not None:
    fig, axes = pdt.plots.plot_rainfall_hydrograph(
        rainfall_file="rainfall_event.dat",
        simulation_data=data,
        title="Storm Event Analysis",
        save_plot=True,
        output_file="storm_analysis.png"
    )
```

**Example - Comparison with Observations:**
```python
# Plot with observed data
fig, axes = pdt.plots.plot_rainfall_hydrograph(
    rainfall_file="event_data.dat",
    simulation_data=data,
    observed_file="observed_outflow.csv",
    title="Model Validation: Event 2024-03-15",
    show_components=True,  # Show individual flow components
    figsize=(14, 10)
)
```

### plot_state_variables()

Visualize state variable evolution over time.

```python
def plot_state_variables(simulation_data, title=None,
                        save_plot=False, output_file=None,
                        figsize=(12, 10), show_storage=True)
```

**Parameters:**
- `simulation_data` (dict): Simulation results
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `show_storage` (bool): Include storage calculations

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects

**Example:**
```python
# Plot state variable evolution
fig, axes = pdt.plots.plot_state_variables(
    simulation_data=data,
    title="System State Evolution",
    show_storage=True,
    save_plot=True,
    output_file="state_variables.png"
)

# Customize individual plots
axes[0].set_ylabel("Ponding Depth [m]", fontsize=12)
axes[1].set_ylabel("Soil Moisture [-]", fontsize=12)
axes[2].set_ylabel("Saturated Zone Depth [m]", fontsize=12)
```

### plot_water_balance()

Visualize water balance components and cumulative flows.

```python
def plot_water_balance(simulation_data, water_balance, title=None,
                      save_plot=False, output_file=None,
                      figsize=(14, 8), show_error=True)
```

**Parameters:**
- `simulation_data` (dict): Simulation results
- `water_balance` (dict): Water balance from `calculate_water_balance()`
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `show_error` (bool): Display water balance error

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects

**Example:**
```python
# Create water balance plot
fig, axes = pdt.plots.plot_water_balance(
    simulation_data=data,
    water_balance=wb,
    title=f"Water Balance (Error: {wb['error_percent']:.2f}%)",
    show_error=True,
    save_plot=True,
    output_file="water_balance.png"
)
```

### plot_parameter_sensitivity()

Visualize sensitivity analysis results.

```python
def plot_parameter_sensitivity(sensitivity_results, plot_type='bar',
                             title=None, save_plot=False, output_file=None,
                             figsize=(12, 8), show_interactions=False)
```

**Parameters:**
- `sensitivity_results` (dict): Results from sensitivity analysis
- `plot_type` (str): Plot type ('bar', 'tornado', 'scatter')
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `show_interactions` (bool): Include interaction effects

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (matplotlib.Axes or list): Axes object(s)

**Example - Bar Chart:**
```python
# Run sensitivity analysis
sa = pdt.SobolSensitivityAnalysis("config.ini", "data.dat")
sa_results = sa.run_analysis(n_samples=2000)

# Create sensitivity plot
fig, ax = pdt.plots.plot_parameter_sensitivity(
    sensitivity_results=sa_results,
    plot_type='bar',
    title="Parameter Sensitivity Analysis",
    show_interactions=True,
    save_plot=True,
    output_file="sensitivity_analysis.png"
)
```

**Example - Tornado Plot:**
```python
# Tornado plot for parameter ranges
fig, ax = pdt.plots.plot_parameter_sensitivity(
    sensitivity_results=sa_results,
    plot_type='tornado',
    title="Parameter Importance Ranking",
    figsize=(10, 8)
)
```

### plot_particle_filter_results()

Visualize particle filter state estimation and forecasting results.

```python
def plot_particle_filter_results(pf_results, observations=None,
                                title=None, save_plot=False, output_file=None,
                                figsize=(14, 10), confidence_levels=[0.68, 0.95])
```

**Parameters:**
- `pf_results` (dict): Particle filter results
- `observations` (dict, optional): Observed data for comparison
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `confidence_levels` (list): Confidence intervals to show

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects

**Example:**
```python
# Run particle filter
model = pdt.PavementModel("config.ini", "forecast.dat")
observations = [pdt.PipeObs("observed.csv")]
config = {'model': model, 'observations': observations, 'n_particles': 1000}
pf_results = pypfilt.run(config)

# Plot results
fig, axes = pdt.plots.plot_particle_filter_results(
    pf_results=pf_results,
    observations={'time': obs_time, 'outflow': obs_flow},
    title="Real-time State Estimation",
    confidence_levels=[0.68, 0.95],
    save_plot=True,
    output_file="particle_filter_results.png"
)
```

### plot_calibration_results()

Visualize parameter calibration convergence and results.

```python
def plot_calibration_results(calibration_results, validation_data=None,
                           title=None, save_plot=False, output_file=None,
                           figsize=(15, 10))
```

**Parameters:**
- `calibration_results` (dict): Results from calibration
- `validation_data` (dict, optional): Validation dataset
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects

**Example:**
```python
# Run calibration
cal_results = pdt.run_calibration(
    setup_file="config.ini",
    rainfall_file="cal_data.dat",
    observed_file="cal_obs.csv",
    generations=30
)

# Plot calibration results
fig, axes = pdt.plots.plot_calibration_results(
    calibration_results=cal_results,
    title="Parameter Calibration Results",
    save_plot=True,
    output_file="calibration_results.png"
)
```

## Specialized Plotting Functions

### plot_performance_metrics()

Create performance metric visualizations for model evaluation.

```python
def plot_performance_metrics(observed, simulated, metrics=None,
                           title=None, save_plot=False, output_file=None,
                           figsize=(12, 8))
```

**Parameters:**
- `observed` (array): Observed values
- `simulated` (array): Simulated values
- `metrics` (dict, optional): Pre-calculated metrics
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects
- `metrics` (dict): Calculated performance metrics

**Example:**
```python
# Load observed and simulated data
observed = pd.read_csv("observed.csv")['outflow'].values
simulated = data['qpipe']

# Create performance plot
fig, axes, metrics = pdt.plots.plot_performance_metrics(
    observed=observed,
    simulated=simulated,
    title="Model Performance Evaluation",
    save_plot=True,
    output_file="performance_metrics.png"
)

print(f"NSE: {metrics['nse']:.3f}")
print(f"RMSE: {metrics['rmse']:.6f} m³/s")
print(f"R²: {metrics['r2']:.3f}")
```

### plot_parameter_correlation()

Visualize parameter correlation matrix from calibration or sensitivity analysis.

```python
def plot_parameter_correlation(parameter_data, parameter_names,
                             title=None, save_plot=False, output_file=None,
                             figsize=(10, 8), method='pearson')
```

**Parameters:**
- `parameter_data` (array): Parameter sample matrix [n_samples × n_parameters]
- `parameter_names` (list): Parameter names
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `method` (str): Correlation method ('pearson', 'spearman')

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `ax` (matplotlib.Axes): Axes object
- `correlation_matrix` (array): Correlation coefficients

**Example:**
```python
# Extract parameter data from calibration
final_population = cal_results['final_population']
param_names = list(cal_results['best_params'].keys())

# Plot correlation matrix
fig, ax, corr_matrix = pdt.plots.plot_parameter_correlation(
    parameter_data=final_population,
    parameter_names=param_names,
    title="Parameter Correlation Matrix",
    save_plot=True,
    output_file="parameter_correlation.png"
)
```

### plot_system_schematic()

Create a schematic diagram of the permeable pavement system.

```python
def plot_system_schematic(params, title=None, save_plot=False,
                        output_file=None, figsize=(10, 8),
                        show_dimensions=True, show_flows=True)
```

**Parameters:**
- `params` (dict): Parameter dictionary with system dimensions
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `show_dimensions` (bool): Show dimension labels
- `show_flows` (bool): Show flow arrows

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `ax` (matplotlib.Axes): Axes object

**Example:**
```python
# Create system schematic
setup = pdt.read_setup_file("pavement.ini")
params = pdt.initialize_parameters(setup)

fig, ax = pdt.plots.plot_system_schematic(
    params=params,
    title="Permeable Pavement System Configuration",
    show_dimensions=True,
    show_flows=True,
    save_plot=True,
    output_file="system_schematic.png"
)
```

## Advanced Visualization

### plot_uncertainty_bands()

Create plots with uncertainty bands from ensemble or particle filter results.

```python
def plot_uncertainty_bands(time_data, ensemble_data, observed_data=None,
                         confidence_levels=[0.5, 0.68, 0.95],
                         title=None, save_plot=False, output_file=None,
                         figsize=(12, 6))
```

**Parameters:**
- `time_data` (array): Time values
- `ensemble_data` (array): Ensemble data [n_timesteps × n_members]
- `observed_data` (array, optional): Observed values for comparison
- `confidence_levels` (list): Confidence levels for bands
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `ax` (matplotlib.Axes): Axes object

**Example:**
```python
# Create uncertainty bands from particle filter ensemble
time = pf_results['time']
particles = pf_results['particles'][:, :, 0]  # hp state variable

fig, ax = pdt.plots.plot_uncertainty_bands(
    time_data=time,
    ensemble_data=particles.T,  # Transpose for correct shape
    observed_data=observed_hp,
    confidence_levels=[0.5, 0.68, 0.95],
    title="Ponding Depth Uncertainty",
    save_plot=True,
    output_file="uncertainty_bands.png"
)
```

### plot_forecast_verification()

Create forecast verification plots comparing predictions with observations.

```python
def plot_forecast_verification(forecast_data, observed_data, lead_times,
                             title=None, save_plot=False, output_file=None,
                             figsize=(14, 8), metrics=['mae', 'rmse', 'bias'])
```

**Parameters:**
- `forecast_data` (dict): Forecast data with different lead times
- `observed_data` (array): Observed values
- `lead_times` (list): Forecast lead times
- `title` (str, optional): Plot title
- `save_plot` (bool): Save plot to file
- `output_file` (str, optional): Output file path
- `figsize` (tuple): Figure size
- `metrics` (list): Verification metrics to calculate

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `axes` (list): List of axes objects
- `verification_metrics` (dict): Calculated verification metrics

**Example:**
```python
# Verify forecasts at different lead times
forecast_data = {
    '1h': forecast_1h,
    '6h': forecast_6h,
    '12h': forecast_12h,
    '24h': forecast_24h
}

fig, axes, metrics = pdt.plots.plot_forecast_verification(
    forecast_data=forecast_data,
    observed_data=observed_outflow,
    lead_times=['1h', '6h', '12h', '24h'],
    title="Forecast Skill vs Lead Time",
    metrics=['mae', 'rmse', 'bias', 'correlation'],
    save_plot=True,
    output_file="forecast_verification.png"
)
```

## Publication-Quality Plots

### create_publication_figure()

Create publication-quality figures with proper formatting.

```python
def create_publication_figure(plot_data, plot_type='timeseries',
                            style='publication', dpi=300,
                            title=None, save_formats=['png', 'pdf'],
                            output_prefix='figure', figsize=(8, 6))
```

**Parameters:**
- `plot_data` (dict): Data for plotting
- `plot_type` (str): Type of plot ('timeseries', 'scatter', 'bar', 'performance')
- `style` (str): Plot style ('publication', 'presentation', 'web')
- `dpi` (int): Resolution for saved figures
- `title` (str, optional): Figure title
- `save_formats` (list): Output formats
- `output_prefix` (str): Output file prefix
- `figsize` (tuple): Figure size in inches

**Returns:**
- `fig` (matplotlib.Figure): Figure object
- `saved_files` (list): List of saved file paths

**Example:**
```python
# Create publication-quality figure
plot_data = {
    'time': data['time'],
    'simulated': data['qpipe'],
    'observed': observed_outflow,
    'rainfall': rainfall_data
}

fig, saved_files = pdt.plots.create_publication_figure(
    plot_data=plot_data,
    plot_type='timeseries',
    style='publication',
    dpi=300,
    title="Model Performance During Storm Event",
    save_formats=['png', 'pdf', 'svg'],
    output_prefix='figure_1',
    figsize=(10, 6)
)

print(f"Saved files: {saved_files}")
```

### setup_plot_style()

Configure matplotlib style for consistent plotting.

```python
def setup_plot_style(style='permeabledt', font_size=12, line_width=2,
                    color_palette='viridis')
```

**Parameters:**
- `style` (str): Plot style ('permeabledt', 'publication', 'presentation')
- `font_size` (int): Base font size
- `line_width` (float): Default line width
- `color_palette` (str): Color palette name

**Example:**
```python
# Setup consistent plotting style
pdt.plots.setup_plot_style(
    style='publication',
    font_size=12,
    line_width=1.5,
    color_palette='tab10'
)

# All subsequent plots will use this style
fig, ax = pdt.plots.plot_rainfall_hydrograph(
    rainfall_file="data.dat",
    simulation_data=data
)
```

## Interactive Plots

### create_interactive_dashboard()

Create interactive dashboard using matplotlib widgets (optional feature).

```python
def create_interactive_dashboard(simulation_function, parameter_ranges,
                               rainfall_file, figsize=(14, 10))
```

**Parameters:**
- `simulation_function` (callable): Function to run simulations
- `parameter_ranges` (dict): Parameter ranges for sliders
- `rainfall_file` (str): Rainfall data file
- `figsize` (tuple): Figure size

**Returns:**
- `fig` (matplotlib.Figure): Interactive figure

**Example:**
```python
# Create interactive parameter exploration dashboard
def run_simulation_wrapper(params_dict):
    setup = pdt.read_setup_file("config.ini")
    params = pdt.initialize_parameters(setup, params_dict)
    data, wb = pdt.run_model(params, "rainfall.dat")
    return data

param_ranges = {
    'Ks': [1e-5, 1e-3],
    'gama': [1.0, 5.0],
    'Cd': [0.5, 1.0]
}

if pdt.plots is not None:
    fig = pdt.plots.create_interactive_dashboard(
        simulation_function=run_simulation_wrapper,
        parameter_ranges=param_ranges,
        rainfall_file="rainfall.dat"
    )
```

## Utility Functions

### save_all_plots()

Save multiple plots with consistent formatting.

```python
def save_all_plots(figures, output_dir='plots', formats=['png'],
                  dpi=300, bbox_inches='tight')
```

**Parameters:**
- `figures` (dict): Dictionary of figure names and objects
- `output_dir` (str): Output directory
- `formats` (list): Output formats
- `dpi` (int): Resolution
- `bbox_inches` (str): Bounding box setting

**Example:**
```python
# Create multiple plots
figures = {}

figures['hydrograph'] = pdt.plots.plot_rainfall_hydrograph(
    "rainfall.dat", data
)[0]

figures['states'] = pdt.plots.plot_state_variables(data)[0]

figures['balance'] = pdt.plots.plot_water_balance(data, wb)[0]

# Save all plots
pdt.plots.save_all_plots(
    figures=figures,
    output_dir='simulation_results',
    formats=['png', 'pdf'],
    dpi=300
)
```

### get_default_colors()

Get default color scheme for consistent plotting.

```python
def get_default_colors(n_colors=10, palette='permeabledt')
```

**Parameters:**
- `n_colors` (int): Number of colors needed
- `palette` (str): Color palette name

**Returns:**
- `colors` (list): List of color values

**Example:**
```python
# Get consistent colors for multiple time series
colors = pdt.plots.get_default_colors(n_colors=5, palette='permeabledt')

fig, ax = plt.subplots()
for i, series in enumerate(data_series):
    ax.plot(time, series, color=colors[i], label=f'Series {i+1}')
```

## Error Handling

### Plot Validation

```python
def validate_plot_data(data, required_keys):
    """Validate data for plotting functions."""

    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required data keys: {missing_keys}")

    # Check for empty arrays
    for key in required_keys:
        if len(data[key]) == 0:
            raise ValueError(f"Empty data array for key: {key}")

    return True

# Example usage in plotting functions
try:
    validate_plot_data(simulation_data, ['time', 'qpipe', 'hp'])
    fig, ax = pdt.plots.plot_rainfall_hydrograph(rainfall_file, simulation_data)
except ValueError as e:
    print(f"Data validation error: {e}")
```

This plotting API provides comprehensive visualization capabilities for all aspects of permeable pavement modeling, from basic time series to advanced uncertainty visualization and publication-quality figures.