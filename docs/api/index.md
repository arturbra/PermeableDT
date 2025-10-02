# API Reference

High-level map of the `permeabledt` public API.  Detailed pages for each module
are linked throughout.

- **[Core Water Flow](water_flow.md)** – Simulation entry points and utilities.
- **[Calibration](calibration.md)** – Genetic-algorithm calibration helpers.
- **[Particle Filtering](particle_filtering.md)** – pypfilt model and observation
  adapters.
- **[Weather Data](weather_data.md)** – HRRR accumulated-precipitation tools.
- **[Plotting](plotting.md)** – Optional matplotlib helpers.

## Quick reference

### Core water-flow helpers (`permeabledt` top-level)

| Function | Description |
| --- | --- |
| `run_simulation(params, rainfall_file, *, inflow=None, evapotranspiration=None, rainfall_unit='mm', verbose=True, plot_outflow=False, output_path=None)` | Read a rainfall `.dat` file, simulate the event, and return pandas outputs. |
| `run_model(params, rainfall_file, inflow=None, evapotranspiration=None, rainfall_unit='mm')` | Lower-level driver that returns raw lists for each series. |
| `run_from_files(pavement, event, input_folder='input_files', calibrated_parameters=None, verbose=True)` | Legacy wrapper compatible with the historical folder layout. |
| `read_setup_file(path)` | Load an INI file with `configparser`. |
| `initialize_parameters(setup)` | Build the parameter dictionary expected by the solver. |
| `modify_parameters(parameters, calibrated_params)` | Apply calibration overrides to a parameter dictionary. |
| `results_dataframe(results)` | Convert the dictionary from `run_model` into a `DataFrame`. |
| `calculate_water_balance(data, dt)` | Summarise volumes and peaks from a simulation. |

### Calibration (`permeabledt` top-level)

| Function | Description |
| --- | --- |
| `run_calibration(calibration_rainfall, calibration_observed_data, setup_file, ...)` | Run the DEAP-based GA using parallel rainfall/observation file lists. |
| `calibrate(*args, **kwargs)` | Thin wrapper around the legacy `calibration.main`. |

### Particle filtering (`permeabledt.particle_filter`)

| Class | Description |
| --- | --- |
| `PavementModel` | `pypfilt.Model` subclass that steps the pavement states. |
| `PipeObs` | Observation model that provides a normal likelihood for pipe flow. |

### Weather data (`permeabledt.download_HRRR_historical_forecast`)

| Class | Description |
| --- | --- |
| `HRRRAccumulatedPrecipitationDownloader` | Download, save, and compare HRRR accumulated precipitation. |

### Plotting (`permeabledt.plots` – optional)

| Function | Description |
| --- | --- |
| `plot_rainfall_hydrograph(...)` | Plot rainfall bars and simulated outflow. |
| `plot_event_comparison(...)` | Compare modeled vs observed outflow for multiple events. |
| `plot_calibration_summary(...)` | Summarise calibration metrics. |

## Typical import pattern

```python
import permeabledt as pdt

setup = pdt.read_setup_file("configs/tc_pf_example.ini")
params = pdt.initialize_parameters(setup)

data, wb = pdt.run_simulation(
    params,
    "data/rainfall_event.dat",
    rainfall_unit="mm",
    verbose=False,
)
```

Optional features (calibration, particle filtering, weather downloads, plotting)
are imported lazily.  Attempting to use them without the corresponding extras
raises `RuntimeError` with installation guidance, e.g.:

```python
try:
    best, calibrated_setup, logbook = pdt.run_calibration(rain_events, observed, setup_file)
except RuntimeError:
    print("Install with: pip install 'permeabledt[calib]'")
```

Refer to the module-specific pages linked above for full parameter tables and
usage notes.
