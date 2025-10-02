# Core Water Flow API

Reference documentation for the functions exposed from `permeabledt.water_flow_module`.

The module simulates the three-zone permeable pavement conceptual model.  Most
users interact with `run_simulation` (for new code) or the legacy
`run_from_files` wrapper.  Lower-level utilities are available for reading input
files, assembling parameter dictionaries, and post-processing results.

## Simulation entry points

### `run_simulation(params, rainfall_file, inflow=None, evapotranspiration=None, rainfall_unit='mm', verbose=True, plot_outflow=False, output_path=None)`

Execute the full water-flow model for a rainfall event stored in a `.dat` file.
The helper reads, converts, and simulates the event, returning tidy pandas
outputs.

**Parameters**
- `params` (`dict`): Parameter dictionary produced by
  `initialize_parameters` (and optionally updated via `modify_parameters`).
- `rainfall_file` (`str` or `pathlib.Path`): Path to a rainfall `.dat` file with
  `mm/dd/YYYY HH:MM rain` columns.
- `inflow` (`Sequence[float]` or `None`): Optional external inflow series in
  m³/s.  Defaults to zeros that match the rainfall length.
- `evapotranspiration` (`Sequence[float]` or `None`): Optional maximum ET series
  in m/s.  Defaults to zeros.
- `rainfall_unit` (`str`): Unit stored in `rainfall_file` (`'mm'` or `'in'`).
- `verbose` (`bool`): When `True`, prints the water balance summary and elapsed
  time.
- `plot_outflow` (`bool`): When `True`, plots rainfall vs. outflow using
  `plots.plot_rainfall_hydrograph`.
- `output_path` (`str` or `pathlib.Path` or `None`): Optional figure path passed
  to the plotting helper when `plot_outflow` is enabled.

**Returns**
- `data` (`pandas.DataFrame`): Time-step results containing rainfall, inflow,
  state variables, and fluxes (see `results_dataframe`).
- `water_balance` (`pandas.DataFrame`): Single-row summary returned by
  `calculate_water_balance`.

**Example**
```python
import permeabledt as pdt

setup = pdt.read_setup_file("inputs/tc_pf_example.ini")
params = pdt.initialize_parameters(setup)

data, wb = pdt.run_simulation(
    params,
    "inputs/rainfall_event.dat",
    rainfall_unit="in",
    verbose=False,
)

print(wb[["Vtotal_in", "Vtotal_pipe (m3)"]])
```

### `run_model(parameters, rainfall_file, inflow=None, evapotranspiration=None, rainfall_unit='mm')`

Lower-level driver that performs the time-stepping loop and returns raw lists of
series.  This function is primarily used internally and by particle filtering
code that needs access to the full dictionary of arrays.

**Returns**

A `dict` with keys such as `date`, `t`, `tQrain`, `tQin`, `tQpipe`, `thp`, `ts`,
etc.  Use `results_dataframe` to convert this dictionary to a tidy
`pandas.DataFrame`.

### `run_from_files(pavement, event, input_folder='input_files', calibrated_parameters=None, verbose=True)`

Backwards-compatible wrapper that reproduces the original package behaviour.  It
loads the legacy configuration and rainfall folders, optionally applies
calibrated parameters, and then calls `run_simulation`.

**Parameters**
- `pavement` (`str`): Pavement identifier used in the legacy folder layout.
- `event` (`int`): Event number within the legacy folders.
- `input_folder` (`str`): Base folder that contains legacy configuration files.
- `calibrated_parameters` (`dict` or `None`): Optional overrides applied via
  `modify_parameters` before running the model.
- `verbose` (`bool`): Passed to `run_simulation`.

**Returns**

Same as `run_simulation`.

### `run_single_timestep(parameters, qin, qrain, emax, hp_prev, s_prev, hsz_prev, husz_prev, nusz_prev, nsz_prev)`

Advance the model by one time step given the previous state and instantaneous
forcings.  Designed for the particle filter implementation where individual
particles evolve independently.

**Parameters**
- `parameters` (`dict`): Model parameters dictionary.
- `qin`, `qrain`, `emax` (`float`): External inflow, rainfall, and ET forcing for
  the current step.
- `hp_prev`, `s_prev`, `hsz_prev`, `husz_prev`, `nusz_prev`, `nsz_prev` (`float`):
  Previous state values.

**Returns**

A dictionary containing the updated state (`hp`, `s`, `hsz`, `husz`, `nusz`,
`nsz`) and the diagnostic flow `Qpipe` for the current step.

## Parameter utilities

### `read_setup_file(setup_file)`

Read an INI configuration file using `configparser.ConfigParser` and return the
loaded parser object.

### `initialize_parameters(setup)`

Build the parameter dictionary expected by the simulation from a
`ConfigParser`.  The dictionary includes geometry, hydraulic properties, and
time-step information.

### `modify_parameters(parameters, calibrated_params=None)`

Update a parameters dictionary with a calibration result.  When thickness values
are modified, dependent totals (such as `l`) are recomputed before returning the
new dictionary.

## Data loading helpers

### `read_rainfall_dat_file(rainfall_file)`

Parse a rainfall `.dat` file into a DataFrame with `date` (timezone-naive
`datetime`) and `rain` columns.

### `rainfall_data_treatment(rainfall_file, surface_area, dt, rainfall_unit='mm')`

Read a rainfall file and convert the totals into m³/s flow rates based on the
specified surface area and time step.  Returns a tuple `(flow_series, dates)`.

### `load_input_files(parameters, rainfall_file, evt_file=None, inflow_file=None, rainfall_unit='in')`

Legacy helper that converts rainfall to flow and optionally loads inflow and ET
series from separate files.  Returns `(inflow, rainfall_flow, evapotranspiration)`
lists sized to the rainfall record.

### `results_dataframe(results, save=False, filename='water_flow_results.csv')`

Convert the raw dictionary returned by `run_model` into a `pandas.DataFrame`.
Optionally saves the CSV when `save=True`.

### `calculate_water_balance(data, dt)`

Summarise a simulation by integrating inflow/outflow volumes and peak metrics.
Returns a single-row `pandas.DataFrame` with columns such as `Vtotal_in`,
`Vtotal_pipe (m3)`, `Qpeak_over (L/s)`, etc.

## Additional notes

- All time series are assumed to share the same time step defined by `params['dt']`.
- Rainfall `.dat` files are expected to contain intensities per time step; the
  helpers take care of converting them to flows using the pavement area.
- Plotting functionality is optional; import errors will occur if `matplotlib`
  is not installed when `plot_outflow=True`.
