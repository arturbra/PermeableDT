# Sensitivity Analysis API

Complete reference for global sensitivity analysis and parameter importance assessment in permeabledt.

## Overview

The sensitivity analysis module (`permeabledt.sensitivity_analysis`) provides global sensitivity analysis tools using the SALib (Sensitivity Analysis Library) framework. It implements Sobol' sensitivity indices to quantify parameter importance and interactions in permeable pavement models.

## Installation Requirements

Sensitivity analysis requires additional dependencies:

```bash
# Install with sensitivity analysis support
pip install permeabledt[sensitivity]

# Or install dependencies manually
pip install SALib tqdm
```

## Core Class

### SobolSensitivityAnalysis

Main class for conducting global sensitivity analysis using Sobol' indices.

```python
class SobolSensitivityAnalysis
```

**Initialization:**
```python
def __init__(self, setup_file, rainfall_file, output_variable='qpipe_peak',
             parameters_to_analyze=None, parameter_bounds=None)
```

**Parameters:**
- `setup_file` (str): Path to INI configuration file
- `rainfall_file` (str): Path to rainfall data file (.dat format)
- `output_variable` (str): Target output variable for analysis
  - Options: 'qpipe_peak', 'qpipe_volume', 'hp_peak', 'storage_max'
- `parameters_to_analyze` (list, optional): Parameter names to analyze
- `parameter_bounds` (dict, optional): Custom parameter bounds

**Default Parameters:**
```python
default_parameters = [
    'Ks',      # Saturated hydraulic conductivity
    'gama',    # Pore-size distribution parameter
    'sw',      # Wilting point
    'sfc',     # Field capacity
    'ss',      # Saturation point
    'Cd',      # Discharge coefficient
    'eta',     # Drainage coefficient
    'nf',      # Filter porosity
    'ng'       # Gravel porosity
]
```

**Example - Basic Setup:**
```python
import permeabledt as pdt

# Basic sensitivity analysis
sa = pdt.SobolSensitivityAnalysis(
    setup_file="pavement.ini",
    rainfall_file="design_storm.dat",
    output_variable='qpipe_peak'
)

print(f"Analyzing {len(sa.parameters)} parameters")
print(f"Target variable: {sa.output_variable}")
```

**Example - Custom Configuration:**
```python
# Custom parameter selection
hydraulic_params = ['Ks', 'gama', 'nf', 'ng']

# Custom bounds (tighter than defaults)
custom_bounds = {
    'Ks': [1e-5, 1e-4],      # Narrow range for known soil
    'gama': [2.0, 3.0],      # Literature-based range
    'nf': [0.35, 0.45],      # Measured porosity range
    'ng': [0.30, 0.40]       # Measured porosity range
}

sa = pdt.SobolSensitivityAnalysis(
    setup_file="calibrated_pavement.ini",
    rainfall_file="100yr_storm.dat",
    output_variable='hp_peak',
    parameters_to_analyze=hydraulic_params,
    parameter_bounds=custom_bounds
)
```

## Core Methods

### run_analysis()

Execute the complete sensitivity analysis workflow.

```python
def run_analysis(self, n_samples=1000, calc_second_order=True,
                parallel=True, random_seed=None, progress_bar=True)
```

**Parameters:**
- `n_samples` (int): Number of samples for Monte Carlo estimation
  - Minimum: 100 (for testing)
  - Recommended: 1000-10000 (for reliable results)
  - High accuracy: >10000 (for publication-quality results)
- `calc_second_order` (bool): Calculate second-order (interaction) indices
- `parallel` (bool): Use parallel processing for simulations
- `random_seed` (int, optional): Random seed for reproducibility
- `progress_bar` (bool): Show progress bar during analysis

**Returns:**
- `results` (dict): Complete sensitivity analysis results

**Example - Standard Analysis:**
```python
# Standard analysis with moderate sample size
results = sa.run_analysis(
    n_samples=2000,
    calc_second_order=True,
    parallel=True,
    random_seed=42
)

print("First-order indices:")
for param, s1 in zip(sa.parameters, results['S1']):
    print(f"  {param}: {s1:.4f}")

print(f"\nTotal-order indices:")
for param, st in zip(sa.parameters, results['ST']):
    print(f"  {param}: {st:.4f}")
```

**Example - High-Resolution Analysis:**
```python
# High-resolution analysis for publication
results = sa.run_analysis(
    n_samples=10000,
    calc_second_order=True,
    parallel=True,
    random_seed=123,
    progress_bar=True
)

# Check convergence
if results['convergence_achieved']:
    print("Analysis converged successfully")
else:
    print("Warning: Analysis may not have converged")
    print(f"Consider increasing n_samples above {results['recommended_samples']}")
```

### generate_samples()

Generate parameter samples for sensitivity analysis.

```python
def generate_samples(self, n_samples, random_seed=None)
```

**Parameters:**
- `n_samples` (int): Number of samples to generate
- `random_seed` (int, optional): Random seed

**Returns:**
- `samples` (numpy.ndarray): Parameter samples [n_samples × n_parameters]

**Example:**
```python
# Generate samples for custom analysis
samples = sa.generate_samples(n_samples=5000, random_seed=42)
print(f"Generated {samples.shape[0]} samples for {samples.shape[1]} parameters")

# Inspect sample distribution
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(samples, columns=sa.parameters)
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
```

### evaluate_model()

Evaluate the model for given parameter samples.

```python
def evaluate_model(self, samples, parallel=True, progress_bar=True)
```

**Parameters:**
- `samples` (numpy.ndarray): Parameter samples
- `parallel` (bool): Use parallel processing
- `progress_bar` (bool): Show progress bar

**Returns:**
- `outputs` (numpy.ndarray): Model outputs for each sample

**Example:**
```python
# Custom evaluation workflow
samples = sa.generate_samples(1000)
outputs = sa.evaluate_model(samples, parallel=True)

# Analyze output distribution
print(f"Output range: {outputs.min():.6f} to {outputs.max():.6f}")
print(f"Output mean: {outputs.mean():.6f}")
print(f"Output std: {outputs.std():.6f}")
```

### calculate_indices()

Calculate Sobol' sensitivity indices from samples and outputs.

```python
def calculate_indices(self, samples, outputs, calc_second_order=True)
```

**Parameters:**
- `samples` (numpy.ndarray): Parameter samples
- `outputs` (numpy.ndarray): Model outputs
- `calc_second_order` (bool): Calculate interaction indices

**Returns:**
- `indices` (dict): Sensitivity indices

**Example:**
```python
# Manual calculation workflow
samples = sa.generate_samples(2000)
outputs = sa.evaluate_model(samples)
indices = sa.calculate_indices(samples, outputs, calc_second_order=True)

print("Manual calculation results:")
print(f"S1: {indices['S1']}")
print(f"ST: {indices['ST']}")
if 'S2' in indices:
    print(f"S2: {indices['S2']}")
```

## Output Variables

### Available Output Variables

#### qpipe_peak
Peak pipe outflow rate [m³/s]
- Most sensitive to drainage parameters
- Critical for flood control design

#### qpipe_volume
Total pipe outflow volume [m³]
- Sensitive to storage and hydraulic parameters
- Important for water balance analysis

#### hp_peak
Peak ponding depth [m]
- Sensitive to overflow and infiltration parameters
- Critical for surface flooding assessment

#### storage_max
Maximum total system storage [m³]
- Sensitive to porosity and depth parameters
- Important for storage capacity design

#### outflow_delay
Time to peak outflow [minutes]
- Sensitive to hydraulic conductivity
- Important for hydrograph timing

### Custom Output Variables

```python
def custom_output_function(data, params):
    """Define custom output variable."""
    # Example: 95th percentile of ponding depth
    return np.percentile(data['hp'], 95)

# Modify the class to use custom output
class CustomSensitivityAnalysis(pdt.SobolSensitivityAnalysis):
    def extract_output_variable(self, data, params):
        if self.output_variable == 'hp_95th':
            return custom_output_function(data, params)
        else:
            return super().extract_output_variable(data, params)

# Use custom analysis
sa_custom = CustomSensitivityAnalysis(
    setup_file="config.ini",
    rainfall_file="data.dat",
    output_variable='hp_95th'
)
```

## Sensitivity Indices

### First-order Indices (S₁)

Measures the main effect of each parameter on the output variance.

```python
# Interpretation of S1 values:
# S1 > 0.1:  Significant influence (>10% of output variance)
# S1 > 0.05: Moderate influence (5-10% of output variance)
# S1 < 0.05: Minor influence (<5% of output variance)

def interpret_first_order_indices(results):
    """Interpret first-order sensitivity indices."""
    s1_values = results['S1']
    parameters = results['parameters']

    interpretation = {}
    for param, s1 in zip(parameters, s1_values):
        if s1 > 0.1:
            level = "High"
        elif s1 > 0.05:
            level = "Moderate"
        else:
            level = "Low"

        interpretation[param] = {
            'value': s1,
            'influence': level,
            'variance_explained': f"{s1*100:.1f}%"
        }

    return interpretation
```

### Total-order Indices (Sₜ)

Measures the total effect including interactions with other parameters.

```python
def analyze_parameter_interactions(results):
    """Analyze parameter interactions using total-order indices."""
    s1_values = results['S1']
    st_values = results['ST']
    parameters = results['parameters']

    interactions = {}
    for param, s1, st in zip(parameters, s1_values, st_values):
        interaction_effect = st - s1

        interactions[param] = {
            'main_effect': s1,
            'total_effect': st,
            'interaction_effect': interaction_effect,
            'interaction_ratio': interaction_effect / st if st > 0 else 0
        }

        # Interpretation
        if interaction_effect > 0.05:
            interactions[param]['interpretation'] = "Strong interactions"
        elif interaction_effect > 0.02:
            interactions[param]['interpretation'] = "Moderate interactions"
        else:
            interactions[param]['interpretation'] = "Weak interactions"

    return interactions
```

### Second-order Indices (S₂)

Measures pairwise interactions between parameters.

```python
def analyze_pairwise_interactions(results):
    """Analyze second-order interactions."""
    if 'S2' not in results:
        return "Second-order indices not calculated"

    s2_matrix = results['S2']
    parameters = results['parameters']
    n_params = len(parameters)

    # Find significant interactions
    significant_interactions = []

    for i in range(n_params):
        for j in range(i+1, n_params):
            s2_value = s2_matrix[i, j]
            if s2_value > 0.01:  # Threshold for significance
                significant_interactions.append({
                    'param1': parameters[i],
                    'param2': parameters[j],
                    'interaction': s2_value,
                    'description': f"{parameters[i]} × {parameters[j]}"
                })

    # Sort by interaction strength
    significant_interactions.sort(key=lambda x: x['interaction'], reverse=True)

    return significant_interactions
```

## Advanced Analysis Methods

### Multi-output Sensitivity Analysis

```python
def multi_output_sensitivity_analysis(setup_file, rainfall_file, output_vars):
    """Perform sensitivity analysis for multiple output variables."""

    results = {}

    for output_var in output_vars:
        print(f"Analyzing {output_var}...")

        sa = pdt.SobolSensitivityAnalysis(
            setup_file=setup_file,
            rainfall_file=rainfall_file,
            output_variable=output_var
        )

        result = sa.run_analysis(n_samples=2000, random_seed=42)
        results[output_var] = result

    return results

# Example usage
output_variables = ['qpipe_peak', 'qpipe_volume', 'hp_peak', 'storage_max']
multi_results = multi_output_sensitivity_analysis(
    "pavement.ini", "design_storm.dat", output_variables
)

# Compare parameter importance across outputs
for param in multi_results['qpipe_peak']['parameters']:
    print(f"\nParameter: {param}")
    for output_var in output_variables:
        idx = multi_results[output_var]['parameters'].index(param)
        s1 = multi_results[output_var]['S1'][idx]
        print(f"  {output_var}: S1 = {s1:.4f}")
```

### Temporal Sensitivity Analysis

```python
def temporal_sensitivity_analysis(setup_file, rainfall_files, time_labels):
    """Analyze parameter sensitivity across different time periods."""

    temporal_results = {}

    for rainfall_file, label in zip(rainfall_files, time_labels):
        sa = pdt.SobolSensitivityAnalysis(
            setup_file=setup_file,
            rainfall_file=rainfall_file,
            output_variable='qpipe_peak'
        )

        result = sa.run_analysis(n_samples=1500, random_seed=42)
        temporal_results[label] = result

    return temporal_results

# Example: Seasonal sensitivity analysis
rainfall_files = [
    "spring_storms.dat",
    "summer_storms.dat",
    "fall_storms.dat",
    "winter_storms.dat"
]
seasons = ["Spring", "Summer", "Fall", "Winter"]

seasonal_results = temporal_sensitivity_analysis(
    "pavement.ini", rainfall_files, seasons
)

# Analyze seasonal variations
for param in seasonal_results['Spring']['parameters']:
    print(f"\nParameter: {param}")
    for season in seasons:
        idx = seasonal_results[season]['parameters'].index(param)
        s1 = seasonal_results[season]['S1'][idx]
        print(f"  {season}: S1 = {s1:.4f}")
```

### Convergence Analysis

```python
def check_convergence(sa, sample_sizes, output_variable='qpipe_peak'):
    """Check convergence of sensitivity indices with sample size."""

    convergence_results = {}

    for n_samples in sample_sizes:
        print(f"Testing with {n_samples} samples...")

        result = sa.run_analysis(
            n_samples=n_samples,
            calc_second_order=False,  # Faster for convergence testing
            random_seed=42
        )

        convergence_results[n_samples] = {
            'S1': result['S1'].copy(),
            'ST': result['ST'].copy()
        }

    return convergence_results

# Example convergence study
sample_sizes = [500, 1000, 2000, 5000, 10000]
conv_results = check_convergence(sa, sample_sizes)

# Plot convergence
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, param in enumerate(sa.parameters[:5]):  # Top 5 parameters
    s1_values = [conv_results[n]['S1'][i] for n in sample_sizes]
    st_values = [conv_results[n]['ST'][i] for n in sample_sizes]

    ax1.plot(sample_sizes, s1_values, 'o-', label=param)
    ax2.plot(sample_sizes, st_values, 'o-', label=param)

ax1.set_xlabel('Sample Size')
ax1.set_ylabel('First-order Index (S1)')
ax1.set_title('S1 Convergence')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Total-order Index (ST)')
ax2.set_title('ST Convergence')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Visualization and Reporting

### Basic Plots

```python
def plot_sensitivity_indices(results, save_plot=False, filename=None):
    """Create standard sensitivity analysis plots."""
    import matplotlib.pyplot as plt

    parameters = results['parameters']
    s1_values = results['S1']
    st_values = results['ST']

    # Sort parameters by total-order indices
    sorted_indices = np.argsort(st_values)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First-order indices
    bars1 = ax1.barh(range(len(parameters)),
                     [s1_values[i] for i in sorted_indices],
                     color='skyblue', alpha=0.7, label='S1')
    ax1.set_yticks(range(len(parameters)))
    ax1.set_yticklabels([parameters[i] for i in sorted_indices])
    ax1.set_xlabel('First-order Index (S1)')
    ax1.set_title('Main Effects')
    ax1.grid(True, axis='x', alpha=0.3)

    # Total-order indices
    bars2 = ax2.barh(range(len(parameters)),
                     [st_values[i] for i in sorted_indices],
                     color='orange', alpha=0.7, label='ST')
    ax2.set_yticks(range(len(parameters)))
    ax2.set_yticklabels([parameters[i] for i in sorted_indices])
    ax2.set_xlabel('Total-order Index (ST)')
    ax2.set_title('Total Effects (including interactions)')
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_plot and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

# Use the plotting function
plot_sensitivity_indices(results, save_plot=True, filename='sensitivity_analysis.png')
```

### Advanced Visualization

```python
def create_comprehensive_sensitivity_report(results, output_file='sensitivity_report.png'):
    """Create comprehensive sensitivity analysis visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(20, 12))

    # 1. Main effects bar chart
    ax1 = plt.subplot(2, 3, 1)
    parameters = results['parameters']
    s1_values = results['S1']
    sorted_idx = np.argsort(s1_values)[::-1]

    bars = ax1.bar(range(len(parameters)), [s1_values[i] for i in sorted_idx])
    ax1.set_xticks(range(len(parameters)))
    ax1.set_xticklabels([parameters[i] for i in sorted_idx], rotation=45)
    ax1.set_ylabel('First-order Index (S1)')
    ax1.set_title('Parameter Main Effects')
    ax1.grid(True, alpha=0.3)

    # Color bars by importance
    for i, bar in enumerate(bars):
        if s1_values[sorted_idx[i]] > 0.1:
            bar.set_color('red')
        elif s1_values[sorted_idx[i]] > 0.05:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')

    # 2. Total effects vs main effects
    ax2 = plt.subplot(2, 3, 2)
    st_values = results['ST']
    ax2.scatter(s1_values, st_values, alpha=0.7, s=100)

    # Add parameter labels
    for i, param in enumerate(parameters):
        ax2.annotate(param, (s1_values[i], st_values[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Add diagonal line (ST = S1)
    max_val = max(max(s1_values), max(st_values))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax2.set_xlabel('First-order Index (S1)')
    ax2.set_ylabel('Total-order Index (ST)')
    ax2.set_title('Main Effects vs Total Effects')
    ax2.grid(True, alpha=0.3)

    # 3. Interaction effects
    ax3 = plt.subplot(2, 3, 3)
    interaction_effects = st_values - s1_values
    bars3 = ax3.bar(range(len(parameters)), [interaction_effects[i] for i in sorted_idx])
    ax3.set_xticks(range(len(parameters)))
    ax3.set_xticklabels([parameters[i] for i in sorted_idx], rotation=45)
    ax3.set_ylabel('Interaction Effect (ST - S1)')
    ax3.set_title('Parameter Interaction Effects')
    ax3.grid(True, alpha=0.3)

    # 4. Second-order interactions heatmap (if available)
    if 'S2' in results:
        ax4 = plt.subplot(2, 3, 4)
        s2_matrix = results['S2']
        im = ax4.imshow(s2_matrix, cmap='Blues', vmin=0, vmax=np.max(s2_matrix))
        ax4.set_xticks(range(len(parameters)))
        ax4.set_yticks(range(len(parameters)))
        ax4.set_xticklabels(parameters, rotation=45)
        ax4.set_yticklabels(parameters)
        ax4.set_title('Second-order Interactions (S2)')
        plt.colorbar(im, ax=ax4)

    # 5. Parameter ranking
    ax5 = plt.subplot(2, 3, 5)
    ranking_data = [(param, st) for param, st in zip(parameters, st_values)]
    ranking_data.sort(key=lambda x: x[1], reverse=True)

    y_pos = range(len(ranking_data))
    ranking_values = [x[1] for x in ranking_data]
    ranking_params = [x[0] for x in ranking_data]

    ax5.barh(y_pos, ranking_values)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(ranking_params)
    ax5.set_xlabel('Total-order Index (ST)')
    ax5.set_title('Parameter Importance Ranking')
    ax5.grid(True, axis='x', alpha=0.3)

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate summary statistics
    total_s1 = np.sum(s1_values)
    total_st = np.sum(st_values)
    max_interaction = np.max(interaction_effects)
    most_important = parameters[np.argmax(st_values)]

    summary_text = f"""
    SENSITIVITY ANALYSIS SUMMARY

    Total S1 (main effects): {total_s1:.3f}
    Total ST (total effects): {total_st:.3f}

    Most important parameter: {most_important}
    (ST = {np.max(st_values):.3f})

    Strongest interaction effect: {max_interaction:.3f}

    Number of significant parameters:
    S1 > 0.1: {np.sum(s1_values > 0.1)}
    S1 > 0.05: {np.sum(s1_values > 0.05)}

    Model explained variance: {total_s1:.1%}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Global Sensitivity Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# Create comprehensive report
create_comprehensive_sensitivity_report(results, 'comprehensive_sensitivity_report.png')
```

## Performance Optimization

### Parallel Processing

```python
def optimize_parallel_performance():
    """Optimize parallel processing for sensitivity analysis."""
    import multiprocessing as mp

    # Determine optimal number of processes
    n_cores = mp.cpu_count()
    optimal_processes = min(n_cores - 1, 8)  # Leave one core free, max 8

    print(f"Available cores: {n_cores}")
    print(f"Using {optimal_processes} processes")

    # Configure for parallel execution
    sa = pdt.SobolSensitivityAnalysis(
        setup_file="config.ini",
        rainfall_file="data.dat"
    )

    # Run with optimized settings
    results = sa.run_analysis(
        n_samples=5000,
        parallel=True,
        progress_bar=True
    )

    return results
```

### Memory Management

```python
def memory_efficient_analysis(sa, n_samples):
    """Perform memory-efficient sensitivity analysis for large problems."""

    # Process samples in chunks to manage memory
    chunk_size = 1000
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    all_outputs = []

    for chunk in range(n_chunks):
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, n_samples)
        chunk_samples = end_idx - start_idx

        print(f"Processing chunk {chunk + 1}/{n_chunks} ({chunk_samples} samples)")

        # Generate samples for this chunk
        samples = sa.generate_samples(chunk_samples, random_seed=42 + chunk)

        # Evaluate model
        outputs = sa.evaluate_model(samples, parallel=True, progress_bar=False)

        all_outputs.extend(outputs)

        # Force garbage collection
        import gc
        gc.collect()

    return np.array(all_outputs)
```

## Integration with Other Modules

### Calibration-Informed Sensitivity Analysis

```python
def calibration_informed_sensitivity_analysis(calibration_results):
    """Perform sensitivity analysis using calibrated parameter distributions."""

    # Extract calibrated parameter values
    best_params = calibration_results['best_params']
    final_population = calibration_results['final_population']

    # Calculate parameter statistics from calibration
    param_stats = {}
    for i, param in enumerate(best_params.keys()):
        values = [individual[i] for individual in final_population]
        param_stats[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    # Use calibrated bounds for sensitivity analysis
    calibrated_bounds = {}
    for param, stats in param_stats.items():
        # Use ±2σ range around calibrated mean
        calibrated_bounds[param] = [
            max(stats['min'], stats['mean'] - 2 * stats['std']),
            min(stats['max'], stats['mean'] + 2 * stats['std'])
        ]

    # Perform sensitivity analysis with calibrated bounds
    sa = pdt.SobolSensitivityAnalysis(
        setup_file="config.ini",
        rainfall_file="data.dat",
        parameter_bounds=calibrated_bounds
    )

    return sa.run_analysis(n_samples=2000)
```

### Uncertainty Propagation

```python
def uncertainty_propagation_analysis(sa_results, forecast_data):
    """Propagate parameter uncertainty through forecasts."""

    # Extract parameter importance
    important_params = []
    s1_values = sa_results['S1']
    parameters = sa_results['parameters']

    for param, s1 in zip(parameters, s1_values):
        if s1 > 0.05:  # Include parameters with >5% contribution
            important_params.append(param)

    print(f"Propagating uncertainty for {len(important_params)} important parameters")

    # Monte Carlo uncertainty propagation
    n_realizations = 1000
    forecast_outputs = []

    for realization in range(n_realizations):
        # Sample parameter values based on sensitivity analysis
        perturbed_params = {}
        for param in important_params:
            param_idx = parameters.index(param)
            # Use uniform sampling within sensitivity bounds
            param_range = sa.parameter_bounds[param]
            perturbed_params[param] = np.random.uniform(param_range[0], param_range[1])

        # Run forecast with perturbed parameters
        setup = pdt.read_setup_file("config.ini")
        params = pdt.initialize_parameters(setup, perturbed_params)
        data, wb = pdt.run_model(params, forecast_data)

        forecast_outputs.append(data['qpipe'])

    # Calculate uncertainty bounds
    forecast_array = np.array(forecast_outputs)
    uncertainty_bounds = {
        'mean': np.mean(forecast_array, axis=0),
        'std': np.std(forecast_array, axis=0),
        'q05': np.percentile(forecast_array, 5, axis=0),
        'q25': np.percentile(forecast_array, 25, axis=0),
        'q75': np.percentile(forecast_array, 75, axis=0),
        'q95': np.percentile(forecast_array, 95, axis=0)
    }

    return uncertainty_bounds
```

This sensitivity analysis API provides comprehensive tools for understanding parameter importance and model behavior in permeable pavement systems, with advanced features for multi-output analysis, convergence checking, and integration with other modeling components.