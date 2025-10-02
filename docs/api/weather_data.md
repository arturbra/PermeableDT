# Weather Data API

The `permeabledt.download_HRRR_historical_forecast` module exposes a single
helper class for working with HRRR accumulated precipitation data via the
[herbie-data](https://github.com/blaylockbk/Herbie) client.  Install the optional
extras with:

```bash
pip install "permeabledt[weather]"
```

## `HRRRAccumulatedPrecipitationDownloader`

### Constructor

`HRRRAccumulatedPrecipitationDownloader(lat, lon, timezone='US/Central')`

Initialise the downloader with the location of interest.

- `lat`, `lon` (`float`): Coordinates in decimal degrees.
- `timezone` (`str`): Local timezone identifier used when converting timestamps
  (defaults to US/Central).

### Exploration helper

`explore_precipitation_variables(sample_date=None, product='subh')`

Inspect the GRIB inventory for accumulated-precipitation variables for a given
run.  Prints matches for the requested date (defaults to 28 April 2024 06:00
UTC) and product (`'subh'` or `'sfc'`).

### Download accumulated forecasts

`download_date_range(start_date, end_date, forecast_hours=6)`

Loop over model runs between the supplied start and end dates (inclusive) and
collect accumulated precipitation data.  For each run the helper attempts to
download the subhourly product and falls back to the surface product if
necessary.

- Accepts either strings or `datetime` objects for `start_date`/`end_date`.
- Returns a list of pandas DataFrames (one per successful model run) with columns
  including `forecast_time`, `precipitation_mm`, `model_run`, and
  `forecast_time_local` (converted to the configured timezone).

### Persist downloads

`save_to_csv(forecast_dataframes, output_dir='hrrr_accumulated')`

Write the list of DataFrames returned by `download_date_range` to individual CSV
files.  Filenames follow the pattern
`hrrr_accumulated_YYYYMMDD_HHMM_UTC.csv` based on the model run time.

### Visual comparison

`plot_comparison(comp, metrics, output_dir='hrrr_comparison', cumulative=False)`

Create a matplotlib plot comparing observed rainfall (`comp['obs']`) with the
forecasted series (`comp['fcst']`).  The `metrics` dictionary should contain the
MAE, RMSE, and bias values used for annotation.  When `cumulative=True`, the
helper plots cumulative sums; otherwise it shows 15-minute increments.  Figures
are saved under `output_dir`.

### Forecast vs observation analysis

`compare_with_observed(forecast_dir, observed, cumulative=False, plot=False, output_dir='plots_output')`

Load previously saved forecast CSVs from `forecast_dir`, align them with an
observed rainfall DataFrame (`date` / `rain` columns), and compute performance
statistics.  Optionally calls `plot_comparison` when `plot=True`.

Returns a tuple `(metrics_list, all_comp_df)` where `metrics_list` contains the
per-run MAE/RMSE/bias dictionaries and `all_comp_df` concatenates the
per-forecast comparison tables.
