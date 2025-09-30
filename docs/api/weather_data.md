# Weather Data API

Complete reference for HRRR weather data integration and precipitation forecasting in permeabledt.

## Overview

The weather data module (`permeabledt.download_HRRR_historical_forecast`) provides tools for downloading and processing High-Resolution Rapid Refresh (HRRR) weather model data from NOAA. This enables real-time precipitation forecasting for permeable pavement systems.

## Installation Requirements

Weather data functions require additional dependencies:

```bash
# Install with weather data support
pip install permeabledt[weather]

# Or install dependencies manually
pip install herbie-data xarray pytz
```

## Core Class

### HRRRAccumulatedPrecipitationDownloader

Main class for downloading and processing HRRR precipitation forecasts.

```python
class HRRRAccumulatedPrecipitationDownloader
```

**Initialization:**
```python
def __init__(self, latitude, longitude, timezone='UTC')
```

**Parameters:**
- `latitude` (float): Latitude in decimal degrees (WGS84)
- `longitude` (float): Longitude in decimal degrees (WGS84)
- `timezone` (str): Timezone string (e.g., 'UTC', 'US/Eastern', 'US/Pacific')

**Example - Basic Setup:**
```python
import permeabledt as pdt

# Create downloader for specific location
downloader = pdt.HRRRAccumulatedPrecipitationDownloader(
    latitude=40.7128,      # New York City
    longitude=-74.0060,
    timezone='US/Eastern'
)

print(f"Location: {downloader.latitude:.4f}°N, {downloader.longitude:.4f}°W")
print(f"Timezone: {downloader.timezone}")
```

**Example - Multiple Locations:**
```python
# Setup for different cities
locations = {
    'NYC': {'lat': 40.7128, 'lon': -74.0060, 'tz': 'US/Eastern'},
    'LA': {'lat': 34.0522, 'lon': -118.2437, 'tz': 'US/Pacific'},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'tz': 'US/Central'},
    'Seattle': {'lat': 47.6062, 'lon': -122.3321, 'tz': 'US/Pacific'}
}

downloaders = {}
for city, coords in locations.items():
    downloaders[city] = pdt.HRRRAccumulatedPrecipitationDownloader(
        latitude=coords['lat'],
        longitude=coords['lon'],
        timezone=coords['tz']
    )
```

## Core Methods

### download_forecast()

Download the latest HRRR precipitation forecast.

```python
def download_forecast(self, forecast_hours=48, model_run='latest',
                     output_file=None, output_format='dat')
```

**Parameters:**
- `forecast_hours` (int): Forecast length in hours (1-48 for HRRR)
- `model_run` (str): Model run time
  - 'latest': Most recent available run
  - 'YYYYMMDDHH': Specific run (e.g., '2024010112' for Jan 1, 2024 12Z)
- `output_file` (str, optional): Output file path
- `output_format` (str): Output format ('dat', 'csv', 'json')

**Returns:**
- `forecast_data` (dict): Forecast data with time and precipitation
- `metadata` (dict): Forecast metadata and quality information

**Example - Latest Forecast:**
```python
# Download latest 24-hour forecast
forecast, metadata = downloader.download_forecast(
    forecast_hours=24,
    model_run='latest',
    output_file='latest_forecast.dat',
    output_format='dat'
)

print(f"Forecast initialized: {metadata['model_run_time']}")
print(f"Forecast length: {len(forecast['time'])} timesteps")
print(f"Total precipitation: {sum(forecast['precipitation']):.2f} mm")
```

**Example - Specific Model Run:**
```python
# Download specific model run
forecast, metadata = downloader.download_forecast(
    forecast_hours=48,
    model_run='2024030112',  # March 1, 2024 12Z
    output_file='specific_forecast.dat'
)

# Check forecast quality
print(f"Model run: {metadata['model_run_time']}")
print(f"Download time: {metadata['download_time']}")
print(f"Data quality: {metadata['quality_flags']}")
```

### download_historical_data()

Download historical HRRR data for model validation and analysis.

```python
def download_historical_data(self, start_date, end_date,
                           forecast_hours=24, output_dir='historical_data')
```

**Parameters:**
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
- `forecast_hours` (int): Forecast length for each run
- `output_dir` (str): Directory to save historical data

**Returns:**
- `download_summary` (dict): Summary of downloaded data

**Example - Historical Download:**
```python
# Download one week of historical forecasts
summary = downloader.download_historical_data(
    start_date='2024-01-01',
    end_date='2024-01-07',
    forecast_hours=24,
    output_dir='january_2024_forecasts'
)

print(f"Downloaded {summary['successful_downloads']} forecasts")
print(f"Failed downloads: {summary['failed_downloads']}")
print(f"Total data size: {summary['total_size_mb']:.1f} MB")
```

### get_precipitation_time_series()

Extract precipitation time series from HRRR data.

```python
def get_precipitation_time_series(self, model_run_time, forecast_hours=24,
                                 accumulation_period=1, units='mm/h')
```

**Parameters:**
- `model_run_time` (str): Model run time ('YYYYMMDDHH')
- `forecast_hours` (int): Forecast length
- `accumulation_period` (int): Accumulation period in hours
- `units` (str): Output units ('mm/h', 'mm', 'in/h', 'in')

**Returns:**
- `time_series` (dict): Time series data with time and precipitation

**Example - Hourly Precipitation:**
```python
# Get hourly precipitation forecast
time_series = downloader.get_precipitation_time_series(
    model_run_time='2024030112',
    forecast_hours=24,
    accumulation_period=1,  # 1-hour accumulation
    units='mm/h'
)

print(f"Time range: {time_series['time'][0]} to {time_series['time'][-1]}")
print(f"Peak intensity: {max(time_series['precipitation']):.1f} mm/h")
```

**Example - Sub-hourly Data:**
```python
# Get 15-minute precipitation data
time_series = downloader.get_precipitation_time_series(
    model_run_time='2024030112',
    forecast_hours=6,
    accumulation_period=0.25,  # 15-minute accumulation
    units='mm/h'
)

print(f"Temporal resolution: {len(time_series['time'])} timesteps in 6 hours")
```

### convert_to_permeabledt_format()

Convert HRRR data to permeabledt-compatible format.

```python
def convert_to_permeabledt_format(self, precipitation_data, output_file,
                             time_step=60, start_time_offset=0)
```

**Parameters:**
- `precipitation_data` (dict): Raw precipitation data
- `output_file` (str): Output file path
- `time_step` (int): Time step in seconds (default: 60)
- `start_time_offset` (int): Start time offset in seconds

**Returns:**
- `conversion_info` (dict): Conversion statistics and metadata

**Example - Standard Conversion:**
```python
# Download and convert forecast
forecast, metadata = downloader.download_forecast(forecast_hours=24)

# Convert to 1-minute timesteps for permeabledt
conversion_info = downloader.convert_to_permeabledt_format(
    precipitation_data=forecast,
    output_file='permeabledt_forecast.dat',
    time_step=60,  # 1-minute timesteps
    start_time_offset=0
)

print(f"Converted {conversion_info['original_timesteps']} to {conversion_info['output_timesteps']} timesteps")
print(f"Interpolation method: {conversion_info['interpolation_method']}")
```

## Data Processing Methods

### validate_forecast_data()

Validate and quality-check downloaded forecast data.

```python
def validate_forecast_data(self, forecast_data)
```

**Parameters:**
- `forecast_data` (dict): Forecast data to validate

**Returns:**
- `validation_report` (dict): Validation results and quality flags

**Example:**
```python
# Download and validate forecast
forecast, metadata = downloader.download_forecast(forecast_hours=24)
validation = downloader.validate_forecast_data(forecast)

print("Validation Results:")
print(f"  Data completeness: {validation['completeness']:.1%}")
print(f"  Temporal consistency: {validation['temporal_consistency']}")
print(f"  Physical realism: {validation['physical_realism']}")

if validation['warnings']:
    print("Warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
```

### interpolate_precipitation()

Interpolate precipitation data to different temporal resolutions.

```python
def interpolate_precipitation(self, time_data, precip_data, target_timestep,
                            method='linear', preserve_total=True)
```

**Parameters:**
- `time_data` (array): Original time values
- `precip_data` (array): Original precipitation values
- `target_timestep` (float): Target time step in same units as time_data
- `method` (str): Interpolation method ('linear', 'cubic', 'nearest')
- `preserve_total` (bool): Preserve total precipitation volume

**Returns:**
- `interpolated_data` (dict): Interpolated time series

**Example:**
```python
# Download hourly data and interpolate to 5-minute resolution
forecast, metadata = downloader.download_forecast(forecast_hours=12)

# Interpolate from hourly to 5-minute data
interpolated = downloader.interpolate_precipitation(
    time_data=forecast['time'],
    precip_data=forecast['precipitation'],
    target_timestep=5.0,  # 5 minutes
    method='linear',
    preserve_total=True
)

print(f"Original resolution: {len(forecast['time'])} points")
print(f"Interpolated resolution: {len(interpolated['time'])} points")
print(f"Total preserved: {abs(sum(forecast['precipitation']) - sum(interpolated['precipitation'])) < 0.01}")
```

### apply_bias_correction()

Apply bias correction to HRRR precipitation forecasts.

```python
def apply_bias_correction(self, forecast_data, correction_method='linear',
                         correction_parameters=None, observed_data=None)
```

**Parameters:**
- `forecast_data` (dict): Raw forecast data
- `correction_method` (str): Correction method ('linear', 'power', 'quantile')
- `correction_parameters` (dict): Pre-computed correction parameters
- `observed_data` (dict, optional): Observed data for real-time correction

**Returns:**
- `corrected_data` (dict): Bias-corrected forecast data

**Example - Linear Bias Correction:**
```python
# Apply linear bias correction
correction_params = {
    'slope': 0.85,      # HRRR tends to overestimate by 15%
    'intercept': 0.1    # Small positive bias
}

corrected_forecast = downloader.apply_bias_correction(
    forecast_data=forecast,
    correction_method='linear',
    correction_parameters=correction_params
)

print(f"Original peak: {max(forecast['precipitation']):.1f} mm/h")
print(f"Corrected peak: {max(corrected_forecast['precipitation']):.1f} mm/h")
```

## Real-time Operations

### setup_real_time_monitoring()

Setup real-time monitoring for new HRRR forecasts.

```python
def setup_real_time_monitoring(self, update_interval=3600,
                             forecast_hours=24, callback_function=None)
```

**Parameters:**
- `update_interval` (int): Check interval in seconds (default: 1 hour)
- `forecast_hours` (int): Forecast length to download
- `callback_function` (callable): Function to call with new forecasts

**Returns:**
- `monitor` (object): Real-time monitoring object

**Example:**
```python
def process_new_forecast(forecast_data, metadata):
    """Process newly downloaded forecast."""
    print(f"New forecast available: {metadata['model_run_time']}")

    # Convert to permeabledt format
    downloader.convert_to_permeabledt_format(
        forecast_data,
        f"forecasts/forecast_{metadata['model_run_time']}.dat"
    )

    # Run particle filter with new forecast
    # (Implementation depends on specific use case)

# Setup monitoring
monitor = downloader.setup_real_time_monitoring(
    update_interval=1800,  # Check every 30 minutes
    forecast_hours=48,
    callback_function=process_new_forecast
)
```

### check_forecast_updates()

Check for new forecast availability.

```python
def check_forecast_updates(self, last_model_run=None)
```

**Parameters:**
- `last_model_run` (str, optional): Last known model run time

**Returns:**
- `update_info` (dict): Information about available updates

**Example:**
```python
# Check for updates
last_run = '2024030112'
update_info = downloader.check_forecast_updates(last_model_run=last_run)

if update_info['new_forecast_available']:
    print(f"New forecast: {update_info['latest_run']}")
    print(f"Hours newer: {update_info['hours_newer']}")

    # Download new forecast
    forecast, metadata = downloader.download_forecast(
        model_run=update_info['latest_run']
    )
else:
    print("No new forecasts available")
```

## Data Quality and Validation

### assess_forecast_skill()

Assess forecast skill using historical data.

```python
def assess_forecast_skill(self, forecast_period_days=30,
                         validation_metrics=['bias', 'rmse', 'correlation'])
```

**Parameters:**
- `forecast_period_days` (int): Period for skill assessment
- `validation_metrics` (list): Metrics to calculate

**Returns:**
- `skill_assessment` (dict): Forecast skill metrics

**Example:**
```python
# Assess 30-day forecast skill
skill = downloader.assess_forecast_skill(
    forecast_period_days=30,
    validation_metrics=['bias', 'rmse', 'correlation', 'pod', 'far']
)

print("Forecast Skill Assessment:")
print(f"  Bias: {skill['bias']:.3f}")
print(f"  RMSE: {skill['rmse']:.3f} mm/h")
print(f"  Correlation: {skill['correlation']:.3f}")
print(f"  Probability of Detection: {skill['pod']:.3f}")
print(f"  False Alarm Ratio: {skill['far']:.3f}")
```

### detect_data_gaps()

Detect gaps and missing data in time series.

```python
def detect_data_gaps(self, time_series, expected_interval=3600)
```

**Parameters:**
- `time_series` (dict): Time series data to check
- `expected_interval` (int): Expected time interval in seconds

**Returns:**
- `gap_report` (dict): Information about detected gaps

**Example:**
```python
# Check for data gaps
gap_report = downloader.detect_data_gaps(
    time_series=forecast,
    expected_interval=3600  # Hourly data
)

if gap_report['gaps_found']:
    print(f"Found {gap_report['number_of_gaps']} gaps:")
    for gap in gap_report['gap_details']:
        print(f"  Gap from {gap['start']} to {gap['end']} ({gap['duration']} hours)")
else:
    print("No data gaps detected")
```

## Integration with permeabledt

### create_forecast_ensemble()

Create ensemble forecasts for uncertainty quantification.

```python
def create_forecast_ensemble(self, base_forecast, ensemble_size=10,
                           perturbation_method='gaussian',
                           perturbation_magnitude=0.1)
```

**Parameters:**
- `base_forecast` (dict): Base forecast data
- `ensemble_size` (int): Number of ensemble members
- `perturbation_method` (str): Method for creating ensemble spread
- `perturbation_magnitude` (float): Magnitude of perturbations

**Returns:**
- `ensemble_forecasts` (list): List of perturbed forecasts

**Example:**
```python
# Create ensemble for uncertainty analysis
base_forecast, metadata = downloader.download_forecast(forecast_hours=24)

ensemble = downloader.create_forecast_ensemble(
    base_forecast=base_forecast,
    ensemble_size=20,
    perturbation_method='gaussian',
    perturbation_magnitude=0.15  # 15% uncertainty
)

print(f"Created ensemble with {len(ensemble)} members")

# Run particle filter with ensemble
for i, member in enumerate(ensemble):
    # Convert to permeabledt format
    downloader.convert_to_permeabledt_format(
        member,
        f"ensemble_forecasts/member_{i:02d}.dat"
    )
```

### integrate_with_particle_filter()

Integrate weather forecasts with particle filtering workflow.

```python
def integrate_with_particle_filter(self, model_config, update_frequency='hourly'):
    """Integrate HRRR forecasts with particle filter operations."""

    # Download latest forecast
    forecast, metadata = self.download_forecast(
        forecast_hours=48,
        model_run='latest'
    )

    # Convert to permeabledt format
    forecast_file = f"pf_forecast_{metadata['model_run_time']}.dat"
    self.convert_to_permeabledt_format(forecast, forecast_file)

    # Setup particle filter with new forecast
    model = pdt.PavementModel(
        setup_file=model_config['setup_file'],
        rainfall_file=forecast_file,
        observations_file=model_config['observations_file']
    )

    # Configure particle filter
    observations = [pdt.PipeObs(model_config['observations_file'])]

    config = {
        'model': model,
        'observations': observations,
        'n_particles': 1000,
        'resample': 'systematic'
    }

    return config

# Use integration
pf_config = downloader.integrate_with_particle_filter({
    'setup_file': 'pavement.ini',
    'observations_file': 'realtime_obs.csv'
})
```

## Error Handling and Troubleshooting

### Common Issues and Solutions

```python
def diagnose_download_issues():
    """Diagnose common HRRR download issues."""

    try:
        # Test basic connectivity
        test_downloader = pdt.HRRRAccumulatedPrecipitationDownloader(
            latitude=40.0, longitude=-100.0
        )

        # Try simple download
        forecast, metadata = test_downloader.download_forecast(
            forecast_hours=6,
            model_run='latest'
        )

        print("Download successful!")
        return True

    except ConnectionError:
        print("Error: No internet connection or HRRR server unavailable")
        return False

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install permeabledt[weather]")
        return False

    except ValueError as e:
        print(f"Error: Invalid parameters - {e}")
        return False

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Run diagnostics
success = diagnose_download_issues()
```

### Data Quality Checks

```python
def comprehensive_data_quality_check(forecast_data):
    """Perform comprehensive quality check on forecast data."""

    quality_report = {
        'overall_quality': 'Good',
        'issues_found': [],
        'recommendations': []
    }

    # Check for negative precipitation
    if any(p < 0 for p in forecast_data['precipitation']):
        quality_report['issues_found'].append('Negative precipitation values')
        quality_report['overall_quality'] = 'Poor'

    # Check for unrealistic values
    max_precip = max(forecast_data['precipitation'])
    if max_precip > 100:  # mm/h
        quality_report['issues_found'].append(f'Unrealistic precipitation: {max_precip:.1f} mm/h')
        quality_report['overall_quality'] = 'Questionable'

    # Check temporal consistency
    time_diffs = np.diff(forecast_data['time'])
    if not np.allclose(time_diffs, time_diffs[0], rtol=0.01):
        quality_report['issues_found'].append('Irregular time intervals')
        quality_report['recommendations'].append('Consider temporal interpolation')

    # Check for missing data
    if len(forecast_data['time']) < 0.9 * 24:  # Less than 90% of expected hourly data
        quality_report['issues_found'].append('Significant missing data')
        quality_report['overall_quality'] = 'Poor'

    return quality_report

# Use quality check
forecast, metadata = downloader.download_forecast(forecast_hours=24)
quality = comprehensive_data_quality_check(forecast)

print(f"Data quality: {quality['overall_quality']}")
if quality['issues_found']:
    print("Issues:")
    for issue in quality['issues_found']:
        print(f"  - {issue}")
```

## Advanced Features

### Custom Model Domains

```python
def setup_custom_domain(bounds, resolution=3):
    """Setup custom domain for multiple locations."""

    # Define domain bounds
    lat_min, lat_max = bounds['lat_range']
    lon_min, lon_max = bounds['lon_range']

    # Create grid of downloaders
    lat_points = np.arange(lat_min, lat_max + resolution/111, resolution/111)  # ~3km spacing
    lon_points = np.arange(lon_min, lon_max + resolution/111, resolution/111)

    domain_downloaders = {}

    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            key = f"point_{i:02d}_{j:02d}"
            domain_downloaders[key] = pdt.HRRRAccumulatedPrecipitationDownloader(
                latitude=lat,
                longitude=lon
            )

    return domain_downloaders

# Example usage
bounds = {
    'lat_range': (40.0, 41.0),  # 1 degree latitude
    'lon_range': (-75.0, -74.0)  # 1 degree longitude
}

domain = setup_custom_domain(bounds, resolution=3)
print(f"Created domain with {len(domain)} grid points")
```

This weather data API provides comprehensive tools for integrating HRRR precipitation forecasts with permeabledt permeable pavement models, enabling real-time forecasting and data-driven decision making.