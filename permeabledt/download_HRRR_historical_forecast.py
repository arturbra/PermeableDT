"""
HRRR Accumulated Precipitation Downloader

This script downloads ONLY accumulated precipitation data from HRRR,
excluding instantaneous rate variables like PRATE.

Requirements:
    pip install herbie-data pandas xarray numpy pytz

Author: Modified to use only accumulated precipitation variables
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from herbie import Herbie
import warnings
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')


class HRRRAccumulatedPrecipitationDownloader:
    """
    A class to download HRRR accumulated precipitation data.
    """

    def __init__(self, lat, lon, timezone='US/Central'):
        """
        Initialize the downloader with coordinates and timezone.

        Parameters:
        -----------
        lat : float
            Latitude of the point of interest
        lon : float
            Longitude of the point of interest
        timezone : str
            Timezone string (default: 'US/Central')
        """
        self.lat = lat
        self.lon = lon
        self.timezone = pytz.timezone(timezone)
        self.utc = pytz.UTC

    def _convert_to_utc(self, dt_local):
        """Convert local datetime to UTC."""
        if dt_local.tzinfo is None:
            dt_local = self.timezone.localize(dt_local)
        return dt_local.astimezone(self.utc).replace(tzinfo=None)

    def _convert_to_local(self, dt_utc):
        """Convert UTC datetime to local time."""
        if dt_utc.tzinfo is None:
            dt_utc = self.utc.localize(dt_utc)
        return dt_utc.astimezone(self.timezone)

    def explore_precipitation_variables(self, sample_date=None, product='subh'):
        """
        Explore available precipitation variables in HRRR files.

        Parameters:
        -----------
        sample_date : datetime or str
            Date to explore (default: uses April 28, 2024)
        product : str
            HRRR product type ('subh' or 'sfc')
        """
        if sample_date is None:
            sample_date = datetime(2024, 4, 28, 6, 0, 0)
        elif isinstance(sample_date, str):
            sample_date = pd.to_datetime(sample_date)

        print(f"\n{'='*80}")
        print(f"EXPLORING PRECIPITATION VARIABLES")
        print(f"Date: {sample_date} UTC, Product: {product}")
        print(f"{'='*80}\n")

        for fxx in [0, 1, 2]:
            print(f"\nForecast hour {fxx}:")

            try:
                H = Herbie(
                    sample_date,
                    model='hrrr',
                    product=product,
                    fxx=fxx
                )

                # Try to get inventory without downloading
                inventory = H.inventory()

                if inventory is not None and not inventory.empty:
                    # Look for accumulated precipitation variables
                    accum_vars = []
                    for idx, row in inventory.iterrows():
                        if 'APCP' in row.get('variable', '') or 'Total Precipitation' in row.get('parameterName', ''):
                            accum_vars.append({
                                'variable': row.get('variable', 'N/A'),
                                'parameterName': row.get('parameterName', 'N/A'),
                                'level': row.get('level', 'N/A'),
                                'forecast_time': row.get('forecast_time', 'N/A')
                            })

                    if accum_vars:
                        print(f"  Found {len(accum_vars)} accumulated precipitation variable(s)")
                        for var in accum_vars:
                            print(f"    - {var['variable']} | {var['parameterName']} | {var['level']}")
                    else:
                        print("  No accumulated precipitation variables found")

            except Exception as e:
                print(f"  Error: {e}")

    def _extract_nearest_point(self, ds):
        """Extract data at the nearest point to our coordinates."""
        try:
            # Try Herbie's pick_points method first
            if hasattr(ds, 'herbie') and hasattr(ds.herbie, 'pick_points'):
                points_df = pd.DataFrame({
                    'latitude': [self.lat],
                    'longitude': [self.lon]
                })
                return ds.herbie.pick_points(points_df, method='nearest')

            # Fallback to manual extraction
            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lats = ds.latitude.values
                lons = ds.longitude.values

                if len(lats.shape) == 2:  # 2D grid
                    lat_diff = lats - self.lat
                    lon_diff = lons - self.lon
                    distances = np.sqrt(lat_diff**2 + lon_diff**2)
                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    return ds.isel(y=min_idx[0], x=min_idx[1])
                else:  # 1D coordinates
                    lat_idx = np.argmin(np.abs(lats - self.lat))
                    lon_idx = np.argmin(np.abs(lons - self.lon))
                    return ds.isel(latitude=lat_idx, longitude=lon_idx)

        except Exception as e:
            print(f"  Error extracting point: {e}")
            return None

    def _download_forecast_accumulated(self, run_time, forecast_hours=6, product='subh'):
        """
        Download HRRR forecast using ONLY accumulated precipitation.

        Parameters:
        -----------
        run_time : datetime
            Model run time (UTC)
        forecast_hours : int
            Number of forecast hours
        product : str
            HRRR product type

        Returns:
        --------
        pandas.DataFrame or None
        """
        forecast_data = []

        for fhour in range(forecast_hours + 1):
            try:
                H = Herbie(
                    run_time,
                    model='hrrr',
                    product=product,
                    fxx=fhour
                )

                # ONLY try to download accumulated precipitation
                # For subhourly, this is typically 'tp' (Total Precipitation)
                ds = None
                var_found = None

                # Try APCP pattern first (most common)
                try:
                    ds = H.xarray(':APCP:', remove_grib=True)
                    if ds is not None and len(ds.data_vars) > 0:
                        # Find the precipitation variable
                        for var in ds.data_vars:
                            if 'tp' in var.lower() or 'apcp' in var.lower():
                                var_found = var
                                break
                except:
                    pass

                # If APCP didn't work, try TP (Total Precipitation)
                if ds is None or var_found is None:
                    try:
                        ds = H.xarray(':TP:', remove_grib=True)
                        if ds is not None and len(ds.data_vars) > 0:
                            for var in ds.data_vars:
                                var_found = var
                                break
                    except:
                        pass

                if ds is None or var_found is None:
                    print(f"  Fhour {fhour}: No accumulated precipitation data available")
                    continue

                # Extract the variable
                precip_var = ds[var_found]

                # Check that it's accumulated (not instantaneous)
                step_type = precip_var.attrs.get('GRIB_stepType', '')
                if step_type != 'accum':
                    print(f"  Fhour {fhour}: Skipping {var_found} - not accumulated (stepType={step_type})")
                    continue

                # Get metadata
                units = precip_var.attrs.get('units', 'unknown')
                step_range = precip_var.attrs.get('GRIB_stepRange', 'unknown')

                print(f"  Fhour {fhour}: Found {var_found}, units={units}, stepRange={step_range}")

                # Extract point data
                point_data = self._extract_nearest_point(ds)

                if point_data is None:
                    continue

                # Get the precipitation values
                point_precip = point_data[var_found]

                # Handle multiple time steps in subhourly data
                if 'step' in point_precip.dims and len(point_precip.step) > 1:
                    # Multiple 15-minute steps in this file
                    steps = point_precip.step.values
                    times = point_precip.valid_time.values
                    values = point_precip.values

                    for i, (step, time, value) in enumerate(zip(steps, times, values)):
                        # Convert time
                        vt_dt = pd.to_datetime(time)
                        if hasattr(vt_dt, 'tz') and vt_dt.tz is not None:
                            vt_dt = vt_dt.tz_convert(None).tz_localize(None)

                        # The value should already be 15-minute accumulation
                        precip_mm = float(value) if not np.isnan(value) else 0.0

                        forecast_data.append({
                            'forecast_time': vt_dt,
                            'precipitation_mm': precip_mm,
                            'forecast_hour': fhour,
                            'model_run': run_time,
                            'variable_used': var_found,
                            'units': units,
                            'step_range': step_range
                        })
                else:
                    # Single time step
                    time = point_precip.valid_time.values
                    value = float(point_precip.values) if not np.isnan(point_precip.values) else 0.0

                    vt_dt = pd.to_datetime(time)
                    if hasattr(vt_dt, 'tz') and vt_dt.tz is not None:
                        vt_dt = vt_dt.tz_convert(None).tz_localize(None)

                    forecast_data.append({
                        'forecast_time': vt_dt,
                        'precipitation_mm': value,
                        'forecast_hour': fhour,
                        'model_run': run_time,
                        'variable_used': var_found,
                        'units': units,
                        'step_range': step_range
                    })

            except Exception as e:
                print(f"  Error processing forecast hour {fhour}: {e}")
                continue

        if forecast_data:
            df = pd.DataFrame(forecast_data)
            df['forecast_time_local'] = df['forecast_time'].apply(self._convert_to_local)
            return df.sort_values('forecast_time').reset_index(drop=True)
        else:
            return None

    def download_date_range(self, start_date, end_date, forecast_hours=6):
        """
        Download accumulated precipitation for a date range.

        Parameters:
        -----------
        start_date : str or datetime
            Start date
        end_date : str or datetime
            End date
        forecast_hours : int
            Forecast hours per run

        Returns:
        --------
        list : List of DataFrames
        """
        if isinstance(start_date, str):
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = start_date

        if isinstance(end_date, str):
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = end_date

        start_utc = self._convert_to_utc(start_dt)
        end_utc = self._convert_to_utc(end_dt)

        current_time = start_utc.replace(minute=0, second=0, microsecond=0)
        forecast_dataframes = []

        print(f"\nDownloading HRRR ACCUMULATED precipitation data")
        print(f"Period: {start_dt} to {end_dt}")
        print(f"Coordinates: {self.lat}, {self.lon}")
        print("-" * 50)

        while current_time <= end_utc:
            print(f"\nProcessing model run: {current_time} UTC")

            # Try subhourly first (has 15-minute data)
            df = self._download_forecast_accumulated(current_time, forecast_hours, 'subh')

            if df is None or len(df) == 0:
                print(f"  Trying surface product...")
                df = self._download_forecast_accumulated(current_time, forecast_hours, 'sfc')

            if df is not None and len(df) > 0:
                forecast_dataframes.append(df)
                print(f"  ✅ Downloaded {len(df)} forecast points")
                print(f"  Variables: {df['variable_used'].unique()}")
            else:
                print(f"  ❌ No accumulated precipitation data available")

            current_time += timedelta(hours=1)

        print(f"\nCompleted! Downloaded {len(forecast_dataframes)} forecast datasets")
        return forecast_dataframes

    def save_to_csv(self, forecast_dataframes, output_dir='hrrr_accumulated'):
        """Save forecast DataFrames to CSV files."""
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, df in enumerate(forecast_dataframes):
            if df is not None and len(df) > 0:
                model_run = df['model_run'].iloc[0]
                filename = f"hrrr_accumulated_{model_run.strftime('%Y%m%d_%H%M')}_UTC.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")

    def plot_comparison(self, comp, metrics, output_dir='hrrr_comparison', cumulative=False):
        """
        Create a comparison plot between observed and forecasted rainfall.

        Parameters
        ----------
        comp : pd.DataFrame
            DataFrame with columns: date, rain, obs, fcst
        metrics : dict
            Dictionary with MAE, RMSE, and bias values
        output_dir : str
            Directory to save the plot
        cumulative : bool
            If True, plot cumulative totals; else 15-min increments
        """

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting
        if cumulative:
            obs_data = comp['obs'].cumsum()
            fcst_data = comp['fcst'].cumsum()
            ylabel = 'Cumulative Rainfall (mm)'
            title_suffix = 'Cumulative'
        else:
            obs_data = comp['obs']
            fcst_data = comp['fcst']
            ylabel = '15-min Rainfall (mm)'
            title_suffix = '15-minute'

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot observed and forecasted data
        ax.plot(comp['date'], obs_data, 'b-', linewidth=2, label='Observed', marker='o', markersize=4)
        ax.plot(comp['date'], fcst_data, 'r-', linewidth=2, label='HRRR Forecast', marker='s', markersize=4)

        # Customize the plot
        ax.set_xlabel('Date/Time', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'HRRR {title_suffix} Rainfall Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add metrics text box
        metrics_text = f"MAE: {metrics['MAE']:.2f} mm\nRMSE: {metrics['RMSE']:.2f} mm\nBias: {metrics['bias']:.2f} mm"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Format x-axis dates
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add some statistics to the plot
        obs_total = obs_data.iloc[-1] if cumulative else obs_data.sum()
        fcst_total = fcst_data.iloc[-1] if cumulative else fcst_data.sum()

        stats_text = f"Total Observed: {obs_total:.2f} mm\nTotal Forecast: {fcst_total:.2f} mm"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Adjust layout
        plt.tight_layout()

        # Save the plot with unique filename using start and end times
        plot_type = 'cumulative' if cumulative else '15min'

        # Extract start and end times for unique filename
        start_time = comp['date'].min().strftime('%Y%m%d_%H%M')
        end_time = comp['date'].max().strftime('%Y%m%d_%H%M')

        filename = f'hrrr_comparison_{start_time}_to_{end_time}_{plot_type}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {filepath}")

        # Show the plot (optional, comment out if running in batch)
        plt.show()

        # Close the figure to free memory
        plt.close()

    def compare_with_observed(self,
                              forecast_dir: str,
                              observed: pd.DataFrame,
                              cumulative=False,
                              plot=False,
                              output_dir='plots_output'):
        """
        Compare HRRR forecasts (saved as CSVs) against observed rainfall.

        Parameters
        ----------
        forecast_dir : str
            Path to folder containing CSVs named like
            'hrrr_accumulated_YYYYMMDD_HHMM_UTC.csv'.
        observed : pd.DataFrame
            Observed rainfall with columns ['date', 'rain'].
        cumulative : bool
            If True, plot cumulative totals; else 15‑min increments.
        plot : bool
            If True, plot a comparison plot between observed and forecasted rainfall.
        output_dir : str
            Directory where comparison plot(s) will be saved.
        Returns
        -------
        metrics_list : list of dicts
            Metrics for each forecast.
        all_comp_df : pd.DataFrame
            Concatenated comparison DataFrame for all forecasts.
        """
        # — load all forecast CSVs —
        dfs = []
        for fn in sorted(os.listdir(forecast_dir)):
            if not fn.endswith('.csv'):
                continue
            df = pd.read_csv(
                os.path.join(forecast_dir, fn),
                parse_dates=['forecast_time_local'],
                index_col=False
            )
            dfs.append(df)

        obs = observed.sort_values('date').copy()
        obs['date'] = pd.to_datetime(obs['date']).dt.tz_localize('US/Central')
        obs.set_index('date', inplace=True)
        obs = obs.resample("15min").sum().reset_index()
        obs['obs'] = obs['rain'] * 25.4

        # — prepare for plotting & metrics —
        all_metrics = []
        all_comparisons = []

        for df in dfs:
            df = (df.rename(columns={'forecast_time_local': 'date',
                                     'precipitation_mm': 'fcst'})
                  .set_index(pd.DatetimeIndex(df['forecast_time_local']))
                  )
            # localize to observed tz
            ser = df['fcst'].sort_index().reset_index()
            ser.columns = ['date', 'fcst']
            comp = obs.merge(ser, how='inner', on='date')
            if comp.empty:
                print(f"[Warning] No overlapping timestamps for the model run.")
                continue
            else:
                comp['forecast_time'] = comp['date'] - comp['date'].iloc[0]

            # compute stats
            mae = round(mean_absolute_error(comp['obs'], comp['fcst']), 2)
            rmse = round(root_mean_squared_error(comp['obs'], comp['fcst']), 2)
            bias = round(float((comp['fcst'] - comp['obs']).mean()), 2)

            metrics = {'MAE': mae, 'RMSE': rmse, 'bias': bias}
            all_metrics.append(metrics)
            all_comparisons.append(comp)

            # Create the plot for this forecast
            if plot:
                self.plot_comparison(comp, metrics, output_dir, cumulative)

        all_comp_df = pd.concat(all_comparisons, ignore_index=True)
        return all_metrics, all_comp_df
