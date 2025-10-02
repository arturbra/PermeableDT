# Plotting API

`permeabledt.plots` contains a small set of optional matplotlib helpers.  Import
errors are suppressed at package import time; check `permeabledt.plots is not
None` before calling these functions.

Install the plotting extras with:

```bash
pip install "permeabledt[plots]"
```

## `plot_rainfall_hydrograph(rainfall_file, outflow_data, rainfall_unit='mm', output_path=None)`

Plot rainfall bars and simulated outflow on a shared timeline.

**Parameters**
- `rainfall_file` (`str | Path`): Rainfall `.dat` file to read via
  `read_rainfall_dat_file`.
- `outflow_data` (`array-like`): Outflow time series (m³/s).  The helper converts
  values to L/s before plotting.
- `rainfall_unit` (`str`): Unit label for rainfall totals (`'mm'` or `'in'`).
- `output_path` (`str | Path | None`): Optional file path to save the resulting
  figure.

**Returns**

`(fig, (ax_rain, ax_flow))` – the created matplotlib figure and axes.

## `plot_event_comparison(rainfall_files, observed_files, parameters, rainfall_unit='in', output_folder=None, figure_size=(12, 10), ncols=2)`

Run the model for multiple events, compare simulated vs observed outflow, and
annotate rainfall totals for each event.

**Parameters**
- `rainfall_files` (`Sequence[str | Path]`): Rainfall `.dat` files used for each
  event.
- `observed_files` (`Sequence[str | Path]`): Observed outflow CSV files.
- `parameters` (`dict`): Parameter dictionary passed to `run_simulation` for each
  event.
- `rainfall_unit` (`str`): Rainfall unit for labelling and bar inversion.
- `output_folder` (`str | Path | None`): When provided, saves per-event figures
  plus a combined grid figure into this folder.
- `figure_size` (`tuple[float, float]`): Dimensions of each subplot pair.
- `ncols` (`int`): Number of columns in the combined grid.

**Returns**
- `fig` (`matplotlib.figure.Figure`): Combined comparison grid.
- `axes` (`list[tuple[Axes, Axes]]`): List of `(ax_rain, ax_flow)` tuples for each
  event.
- `metrics` (`dict`): Dictionary keyed by event name containing RMSE, NSE, and R²
  statistics computed from the observed vs modeled series.

## `plot_calibration_summary(metrics, output_path=None)`

Summarise the per-event metrics returned by `plot_event_comparison` in a 2×2 grid
of bar charts (RMSE, NSE, R², and MAE).

**Parameters**
- `metrics` (`dict`): Mapping of event labels to dictionaries containing metric
  values.  Raises `ValueError` if empty.
- `output_path` (`str | Path | None`): Optional path to save the summary figure.

**Returns**

`(fig, axes)` – the matplotlib figure and axes array.
