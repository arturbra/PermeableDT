import permeabledt as gdt
from permeabledt import plots
import os
from pathlib import Path
import pandas as pd

print(os.getcwd())

base_dir = os.getcwd()

setup_file = os.path.join(base_dir, "input", "input_parameters.ini")
print(setup_file)

calibration_path = Path(os.path.join(base_dir, "input", "calibration"))
calibration_rainfall = sorted(calibration_path.glob('*.dat'))
calibration_observed_data = sorted(calibration_path.glob('*.csv'))

# Output files
output_setup_file = os.path.join(base_dir, "output", f"calibrated_parameters.ini")
logbook_path = os.path.join(base_dir, "output", "calibration_logbook.csv")

# # Run calibration
best_params, calibrated_setup, logbook = gdt.run_calibration(
    calibration_rainfall=calibration_rainfall,
    calibration_observed_data=calibration_observed_data,
    setup_file=setup_file,
    output_setup_file=output_setup_file,
    logbook_output_path=logbook_path,
    seed=2025  # For reproducibility
)

# Plot the calibration events
setup_calibration = gdt.read_setup_file(output_setup_file)
parameters = gdt.initialize_parameters(setup_calibration)

# Plot comparisons
_, _, metrics_calib = plots.plot_event_comparison(
    rainfall_files=calibration_rainfall,
    observed_files=calibration_observed_data,
    parameters=parameters,
    rainfall_unit='in',
    output_folder= os.path.join(base_dir, "output", 'calibration_plots'),
    ncols=1  # Number of columns in the grid
)

validation_path = Path(os.path.join(base_dir, "input", "validation"))
validation_rainfall = sorted(validation_path.glob('*.dat'))
validation_observed_data = sorted(validation_path.glob('*.csv'))

_, _, metrics_valid = plots.plot_event_comparison(
    rainfall_files=validation_rainfall,
    observed_files=validation_observed_data,
    parameters=parameters,
    rainfall_unit='in',
    output_folder= os.path.join(base_dir, "output", 'validation_plots'),
    ncols=1  # Number of columns in the grid
)

df_calib = pd.DataFrame(metrics_calib).T
df_valid = pd.DataFrame(metrics_valid).T

print(df_calib.to_string())
print(df_valid.to_string())