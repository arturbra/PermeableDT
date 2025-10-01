from permeabledt import download_HRRR_historical_forecast
import pandas as pd
import os
import permeabledt as pdt

base_dir = os.getcwd()
rainfall_file = os.path.join(base_dir, "input", "event_00_rainfall.dat")
rainfall_obs = pdt.water_flow_module.read_rainfall_dat_file(rainfall_file)
rainfall_obs['date'] = pd.to_datetime(rainfall_obs['date'])
start_date = rainfall_obs['date'][0].floor("h")
end_date = rainfall_obs['date'][len(rainfall_obs) - 1].ceil("h")
output_dir = os.path.join(base_dir, 'output', str(start_date.date()))
# Permeable Pavement Site - San Antonio, TX coordinates
lat = 29.629438
lon = -98.476345

# Initialize downloader
downloader = download_HRRR_historical_forecast.HRRRAccumulatedPrecipitationDownloader(lat, lon, timezone='US/Central')

# Explore available variables
downloader.explore_precipitation_variables(
    sample_date=start_date,
    product='subh'
)

# Download accumulated precipitation data
forecast_dataframes = downloader.download_date_range(
    start_date=start_date,
    end_date=end_date,
    forecast_hours=6
)

# Display results
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

all_data = []
for i, df in enumerate(forecast_dataframes):
    if df is not None and len(df) > 0:
        model_run = df['model_run'].iloc[0]
        print(f"\nForecast {i + 1}: Model run at {model_run} UTC")
        print(f"Total points: {len(df)}")
        print(f"Variables used: {df['variable_used'].unique()}")
        print(f"\nFirst 10 rows:")
        print(df[['forecast_time_local', 'precipitation_mm', 'step_range']].head(10))

        # Collect all data
        all_data.append(df)

    # Save results
    if forecast_dataframes:
        downloader.save_to_csv(forecast_dataframes, output_dir)

        # Create a combined file with all forecasts
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['forecast_time', 'model_run']).reset_index(drop=True)
            combined_df.to_csv(os.path.join(f'output\hrrr_all_accumulated_forecasts_{start_date.date()}.csv'), index=False)
            print(f"\nSaved combined file: output\\hrrr_all_accumulated_forecasts_{start_date.date()}.csv")

            # Show summary statistics
            print(f"\nSummary Statistics:")
            print(f"Total forecast points: {len(combined_df)}")
            print(f"Time range: {combined_df['forecast_time_local'].min()} to {combined_df['forecast_time_local'].max()}")
            print(f"Total precipitation: {combined_df['precipitation_mm'].sum():.2f} mm")
            print(f"Max 15-min precipitation: {combined_df['precipitation_mm'].max():.2f} mm")


# Plot and extract the metrics to compare with the observed rainfall
metrics, _ = downloader.compare_with_observed(
    forecast_dir=output_dir,
    observed=rainfall_obs,
    cumulative=False,
    plot=True,
    output_dir = rf'output\plots\{str(start_date.date())}'
)