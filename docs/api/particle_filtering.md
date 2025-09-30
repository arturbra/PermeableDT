# Particle Filter API

Complete reference for real-time forecasting and data assimilation using particle filtering in permeabledt.

## Overview

The particle filter module (`permeabledt.particle_filter`) provides real-time state estimation and forecasting capabilities for permeable pavement systems. It implements the pypfilt framework for sequential Monte Carlo methods, enabling data assimilation and uncertainty quantification.

## Installation Requirements

Particle filtering requires additional dependencies:

```bash
# Install with particle filter support
pip install permeabledt[pf]

# Or install dependencies manually
pip install pypfilt scipy tomlkit
```

## Core Classes

### PavementModel

Implementation of `pypfilt.Model` for permeable pavement systems.

```python
class PavementModel(pypfilt.Model)
```

**Initialization:**
```python
def __init__(self, setup_file, rainfall_file, observations_file=None)
```

**Parameters:**
- `setup_file` (str): Path to INI configuration file
- `rainfall_file` (str): Path to rainfall forecast data (.dat format)
- `observations_file` (str, optional): Path to observed outflow data

**Methods:**

#### state_size()
Returns the number of state variables (3: hp, s, hsz).

```python
def state_size(self) -> int
```

#### can_smooth()
Returns False (smoothing not currently implemented).

```python
def can_smooth(self) -> bool
```

#### init()
Initialize particle states with random perturbations around initial conditions.

```python
def init(self, ctx, vec)
```

**Parameters:**
- `ctx`: pypfilt context
- `vec`: State vector array [n_particles × 3]

#### update()
Advance particles one timestep using the water flow model.

```python
def update(self, ctx, time_step, vec)
```

**Parameters:**
- `ctx`: pypfilt context
- `time_step`: Current time step index
- `vec`: State vector array [n_particles × 3]

#### describe()
Return parameter descriptions for logging.

```python
def describe(self) -> dict
```

**Example - Basic Model Setup:**
```python
import permeabledt as gdt

# Create model instance
model = gdt.PavementModel(
    setup_file="pavement.ini",
    rainfall_file="forecast_rainfall.dat",
    observations_file="observed_outflow.csv"
)

print(f"State size: {model.state_size()}")
print(f"Can smooth: {model.can_smooth()}")
print("Parameters:", model.describe())
```

### PipeObs

Implementation of `pypfilt.obs.Univariate` for pipe outflow observations.

```python
class PipeObs(pypfilt.obs.Univariate)
```

**Initialization:**
```python
def __init__(self, observed_file, obs_error_std=1e-6)
```

**Parameters:**
- `observed_file` (str): Path to observed outflow CSV file
- `obs_error_std` (float): Observation error standard deviation [m³/s]

**Methods:**

#### from_state()
Extract pipe outflow from state vector.

```python
def from_state(self, ctx, state_vec)
```

**Parameters:**
- `ctx`: pypfilt context
- `state_vec`: State vector [n_particles × 3]

**Returns:**
- Pipe outflow observations [n_particles]

#### log_llhd()
Calculate log-likelihood of observations.

```python
def log_llhd(self, ctx, obs_vec, model_obs)
```

**Parameters:**
- `ctx`: pypfilt context
- `obs_vec`: Observed values
- `model_obs`: Model predictions

**Returns:**
- Log-likelihood values [n_particles]

**Example - Observation Setup:**
```python
# Create observation instance
obs = gdt.PipeObs(
    observed_file="outflow_data.csv",
    obs_error_std=5e-7  # 0.5 μm³/s error
)

# Use in particle filter configuration
observations = [obs]
```

## Particle Filter Workflow

### Basic Particle Filter Setup

```python
import pypfilt
import permeabledt as gdt

def run_basic_particle_filter():
    """Run basic particle filter for real-time forecasting."""

    # 1. Create model and observations
    model = gdt.PavementModel(
        setup_file="config.ini",
        rainfall_file="forecast.dat",
        observations_file="observed.csv"
    )

    observations = [gdt.PipeObs("observed.csv")]

    # 2. Configure particle filter
    config = {
        'model': model,
        'observations': observations,
        'n_particles': 1000,
        'resample': pypfilt.resample.residual,
        'prng_seed': 42
    }

    # 3. Run particle filter
    results = pypfilt.run(config)

    return results
```

### Advanced Configuration

```python
def run_advanced_particle_filter():
    """Advanced particle filter with custom settings."""

    # Model setup
    model = gdt.PavementModel("pavement.ini", "rainfall_forecast.dat")

    # Multiple observation types
    pipe_obs = gdt.PipeObs("pipe_outflow.csv", obs_error_std=1e-6)
    observations = [pipe_obs]

    # Advanced resampling
    resample_config = {
        'regularise': True,
        'smooth': 0.1,
        'roughen': True
    }

    # Custom prior distribution
    def custom_prior(ctx, state_vec):
        """Custom prior with physical constraints."""
        n_particles = state_vec.shape[0]

        # Sample from reasonable ranges
        state_vec[:, 0] = np.random.uniform(0, 0.1, n_particles)      # hp
        state_vec[:, 1] = np.random.uniform(0.2, 0.8, n_particles)    # s
        state_vec[:, 2] = np.random.uniform(0.1, 1.0, n_particles)    # hsz

        return state_vec

    # Configure particle filter
    config = {
        'model': model,
        'observations': observations,
        'n_particles': 2000,
        'resample': pypfilt.resample.systematic,
        'resample_params': resample_config,
        'prior': custom_prior,
        'prng_seed': 123
    }

    # Run with progress tracking
    results = pypfilt.run(config, verbose=True)

    return results
```

## Configuration Management

### TOML Configuration Files

Particle filter settings can be managed using TOML files:

```toml
# particle_filter_config.toml

[model]
setup_file = "pavement.ini"
rainfall_file = "forecast_data.dat"
observations_file = "observed_outflow.csv"

[filter]
n_particles = 1000
prng_seed = 42
resample_method = "systematic"

[observations]
pipe_error_std = 1e-6

[output]
save_states = true
save_forecasts = true
output_dir = "pf_results"
```

### Loading TOML Configuration

```python
def load_pf_config(config_file):
    """Load particle filter configuration from TOML file."""
    try:
        import tomlkit
    except ImportError:
        raise RuntimeError("tomlkit required for TOML configuration")

    with open(config_file, 'r') as f:
        config = tomlkit.load(f)

    return config

# Use configuration
config = load_pf_config("pf_config.toml")
model = gdt.PavementModel(
    setup_file=config['model']['setup_file'],
    rainfall_file=config['model']['rainfall_file']
)
```

## State Estimation and Forecasting

### Real-time State Estimation

```python
def real_time_estimation(model, observations, forecast_steps=60):
    """Perform real-time state estimation with forecasting."""

    # Configure particle filter
    config = {
        'model': model,
        'observations': observations,
        'n_particles': 1000,
        'resample': pypfilt.resample.residual
    }

    # Run particle filter
    results = pypfilt.run(config)

    # Extract state estimates
    state_estimates = {
        'time': results['time'],
        'hp_mean': results['state_mean'][:, 0],
        'hp_std': results['state_std'][:, 0],
        'hp_quantiles': results['state_quantiles'][:, 0, :],
        's_mean': results['state_mean'][:, 1],
        's_std': results['state_std'][:, 1],
        's_quantiles': results['state_quantiles'][:, 1, :],
        'hsz_mean': results['state_mean'][:, 2],
        'hsz_std': results['state_std'][:, 2],
        'hsz_quantiles': results['state_quantiles'][:, 2, :]
    }

    # Generate forecasts from final state
    final_particles = results['final_particles']
    forecasts = generate_forecasts(model, final_particles, forecast_steps)

    return state_estimates, forecasts
```

### Forecast Generation

```python
def generate_forecasts(model, particles, n_steps):
    """Generate forecasts from particle ensemble."""

    n_particles = particles.shape[0]
    forecasts = np.zeros((n_steps, n_particles, 3))

    # Initialize with final particle states
    current_states = particles.copy()

    # Propagate forward without observations
    for step in range(n_steps):
        # Update particles using model
        ctx = None  # Simplified context
        time_step = step

        # Advance each particle
        for p in range(n_particles):
            current_states[p] = model.update(ctx, time_step, current_states[p])

        forecasts[step] = current_states

    # Calculate forecast statistics
    forecast_stats = {
        'mean': np.mean(forecasts, axis=1),
        'std': np.std(forecasts, axis=1),
        'quantiles': np.percentile(forecasts, [5, 25, 50, 75, 95], axis=1)
    }

    return forecast_stats
```

## Uncertainty Quantification

### Confidence Intervals

```python
def calculate_confidence_intervals(results, confidence_levels=[0.68, 0.95]):
    """Calculate confidence intervals for state estimates."""

    intervals = {}

    for level in confidence_levels:
        alpha = 1 - level
        lower_q = (alpha / 2) * 100
        upper_q = (1 - alpha / 2) * 100

        intervals[f'{level:.0%}'] = {
            'hp_lower': np.percentile(results['particles'][:, :, 0], lower_q, axis=1),
            'hp_upper': np.percentile(results['particles'][:, :, 0], upper_q, axis=1),
            's_lower': np.percentile(results['particles'][:, :, 1], lower_q, axis=1),
            's_upper': np.percentile(results['particles'][:, :, 1], upper_q, axis=1),
            'hsz_lower': np.percentile(results['particles'][:, :, 2], lower_q, axis=1),
            'hsz_upper': np.percentile(results['particles'][:, :, 2], upper_q, axis=1)
        }

    return intervals
```

### Effective Sample Size

```python
def monitor_filter_performance(results):
    """Monitor particle filter performance metrics."""

    # Calculate effective sample size
    weights = results['weights']
    ess = 1.0 / np.sum(weights**2, axis=1)

    # Calculate resampling frequency
    resampling_times = np.where(np.diff(results['resampled']) > 0)[0]
    resampling_freq = len(resampling_times) / len(results['time'])

    # Weight degeneracy
    weight_entropy = -np.sum(weights * np.log(weights + 1e-16), axis=1)
    max_entropy = np.log(weights.shape[1])
    normalized_entropy = weight_entropy / max_entropy

    performance = {
        'effective_sample_size': ess,
        'mean_ess': np.mean(ess),
        'min_ess': np.min(ess),
        'resampling_frequency': resampling_freq,
        'weight_entropy': normalized_entropy,
        'mean_entropy': np.mean(normalized_entropy)
    }

    return performance
```

## Data Assimilation

### Multiple Observation Types

```python
def setup_multi_observation_filter():
    """Setup particle filter with multiple observation types."""

    # Different observation sources
    pipe_obs = gdt.PipeObs("pipe_outflow.csv", obs_error_std=1e-6)

    # Custom observation for ponding depth (if available)
    class PondingObs(pypfilt.obs.Univariate):
        def __init__(self, observed_file, obs_error_std=0.01):
            self.obs_data = pd.read_csv(observed_file)
            self.obs_error_std = obs_error_std

        def from_state(self, ctx, state_vec):
            return state_vec[:, 0]  # hp (ponding depth)

        def log_llhd(self, ctx, obs_vec, model_obs):
            diff = obs_vec - model_obs
            return -0.5 * (diff / self.obs_error_std)**2

    # ponding_obs = PondingObs("ponding_data.csv")

    observations = [pipe_obs]  # Add ponding_obs if data available

    return observations
```

### Adaptive Observation Error

```python
def adaptive_observation_error(results, base_error=1e-6):
    """Adapt observation error based on model performance."""

    # Calculate innovation (observation - prediction)
    innovations = results['innovations']

    # Estimate time-varying observation error
    window_size = 10
    adaptive_errors = []

    for i in range(len(innovations)):
        start = max(0, i - window_size)
        end = i + 1

        # Use rolling standard deviation of innovations
        window_innovations = innovations[start:end]
        if len(window_innovations) > 1:
            adaptive_error = np.std(window_innovations)
        else:
            adaptive_error = base_error

        # Bound the error
        adaptive_error = np.clip(adaptive_error, base_error * 0.1, base_error * 10)
        adaptive_errors.append(adaptive_error)

    return np.array(adaptive_errors)
```

## Model Validation

### Cross-validation

```python
def cross_validate_particle_filter(model_config, n_folds=5):
    """Perform k-fold cross-validation for particle filter."""

    # Load full dataset
    obs_data = pd.read_csv(model_config['observations_file'])

    # Split into folds
    fold_size = len(obs_data) // n_folds
    results = []

    for fold in range(n_folds):
        # Create train/test split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size

        train_data = pd.concat([
            obs_data[:test_start],
            obs_data[test_end:]
        ])
        test_data = obs_data[test_start:test_end]

        # Train on training data
        train_file = f"train_fold_{fold}.csv"
        train_data.to_csv(train_file, index=False)

        # Setup model with training data
        model = gdt.PavementModel(
            setup_file=model_config['setup_file'],
            rainfall_file=model_config['rainfall_file'],
            observations_file=train_file
        )

        observations = [gdt.PipeObs(train_file)]

        # Run particle filter
        config = {
            'model': model,
            'observations': observations,
            'n_particles': 1000
        }

        pf_results = pypfilt.run(config)

        # Evaluate on test data
        test_metrics = evaluate_forecasts(pf_results, test_data)
        results.append(test_metrics)

    return results
```

### Performance Metrics

```python
def evaluate_particle_filter(results, observations):
    """Evaluate particle filter performance."""

    # Extract predictions and observations
    predictions = results['state_mean'][:, 0]  # hp predictions
    obs_values = observations['outflow'].values

    # Calculate metrics
    metrics = {}

    # Root Mean Square Error
    metrics['rmse'] = np.sqrt(np.mean((predictions - obs_values)**2))

    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(predictions - obs_values))

    # Nash-Sutcliffe Efficiency
    ss_res = np.sum((obs_values - predictions)**2)
    ss_tot = np.sum((obs_values - np.mean(obs_values))**2)
    metrics['nse'] = 1 - (ss_res / ss_tot)

    # Coverage probability (for confidence intervals)
    ci_95 = results['state_quantiles'][:, 0, :]  # hp quantiles
    coverage = np.mean(
        (obs_values >= ci_95[:, 0]) & (obs_values <= ci_95[:, 4])
    )
    metrics['coverage_95'] = coverage

    # Reliability (Rank Histogram)
    n_particles = results['particles'].shape[1]
    ranks = np.zeros(len(obs_values))

    for i, obs in enumerate(obs_values):
        particles = results['particles'][i, :, 0]  # hp particles
        rank = np.sum(particles < obs)
        ranks[i] = rank

    # Chi-square test for uniformity
    hist, _ = np.histogram(ranks, bins=n_particles//10)
    expected = len(ranks) / len(hist)
    chi2 = np.sum((hist - expected)**2 / expected)
    metrics['reliability_chi2'] = chi2

    return metrics
```

## Real-time Implementation

### Streaming Data Processing

```python
def real_time_processor():
    """Real-time particle filter for streaming data."""

    # Initialize particle filter
    model = gdt.PavementModel("config.ini", "current_forecast.dat")
    observations = [gdt.PipeObs("current_obs.csv")]

    # Initialize particle ensemble
    n_particles = 1000
    particles = np.zeros((n_particles, 3))

    # Initialize states
    ctx = None
    model.init(ctx, particles)

    # Simulation loop
    while True:
        # Get new observation (from data stream)
        new_obs = get_latest_observation()

        if new_obs is not None:
            # Update particles with new observation
            time_step = get_current_timestep()

            # Predict step
            for i in range(n_particles):
                particles[i] = model.update(ctx, time_step, particles[i])

            # Update step (data assimilation)
            log_weights = observations[0].log_llhd(ctx, new_obs, particles)
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)

            # Resample if needed
            ess = 1.0 / np.sum(weights**2)
            if ess < n_particles * 0.5:
                indices = pypfilt.resample.systematic(weights)
                particles = particles[indices]
                weights = np.ones(n_particles) / n_particles

            # Generate current state estimate
            state_estimate = {
                'hp_mean': np.mean(particles[:, 0]),
                'hp_std': np.std(particles[:, 0]),
                's_mean': np.mean(particles[:, 1]),
                's_std': np.std(particles[:, 1]),
                'hsz_mean': np.mean(particles[:, 2]),
                'hsz_std': np.std(particles[:, 2])
            }

            # Send to output stream
            send_state_estimate(state_estimate)

        # Sleep until next timestep
        time.sleep(60)  # 1-minute intervals

def get_latest_observation():
    """Get latest observation from data stream."""
    # Implementation depends on data source
    pass

def send_state_estimate(estimate):
    """Send state estimate to output stream."""
    # Implementation depends on output format
    pass
```

## Error Handling and Diagnostics

### Common Issues

```python
def diagnose_particle_filter(results):
    """Diagnose common particle filter issues."""

    diagnostics = {}

    # Check for particle degeneracy
    final_weights = results['weights'][-1]
    max_weight = np.max(final_weights)

    if max_weight > 0.9:
        diagnostics['particle_degeneracy'] = True
        diagnostics['max_weight'] = max_weight

    # Check effective sample size
    ess = 1.0 / np.sum(final_weights**2)
    n_particles = len(final_weights)

    if ess < n_particles * 0.1:
        diagnostics['low_ess'] = True
        diagnostics['ess_ratio'] = ess / n_particles

    # Check for filter divergence
    state_variance = np.var(results['particles'][-1], axis=0)

    if np.any(state_variance > 1.0):  # Threshold depends on problem
        diagnostics['filter_divergence'] = True
        diagnostics['state_variance'] = state_variance

    # Check resampling frequency
    resampling_count = np.sum(np.diff(results['resampled']) > 0)
    resampling_freq = resampling_count / len(results['time'])

    if resampling_freq > 0.8:
        diagnostics['excessive_resampling'] = True
        diagnostics['resampling_frequency'] = resampling_freq

    return diagnostics
```

### Troubleshooting

```python
def fix_particle_filter_issues(config, diagnostics):
    """Suggest fixes for particle filter issues."""

    suggestions = []

    if diagnostics.get('particle_degeneracy', False):
        suggestions.append("Increase number of particles")
        suggestions.append("Increase process noise")
        suggestions.append("Use regularized resampling")
        config['n_particles'] *= 2

    if diagnostics.get('low_ess', False):
        suggestions.append("Reduce observation error")
        suggestions.append("Check observation-model mismatch")
        config['obs_error_std'] *= 0.5

    if diagnostics.get('filter_divergence', False):
        suggestions.append("Increase observation frequency")
        suggestions.append("Add process noise")
        suggestions.append("Check model physics")

    if diagnostics.get('excessive_resampling', False):
        suggestions.append("Increase observation error")
        suggestions.append("Use adaptive resampling threshold")

    return suggestions, config
```

This particle filter API provides comprehensive tools for real-time state estimation and forecasting in permeable pavement systems, with robust error handling and performance monitoring capabilities.