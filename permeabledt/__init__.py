"""
permeabledt
=======

Digital-twin tools for permeable-pavement water-flow modelling,
genetic-algorithm calibration, particle filtering,
and weather data acquisition.

Quick start
-----------
>>> import permeabledt as pdt
>>>
>>> # --- Core water flow simulation ---
>>> data, wb = pdt.run_simulation(params, qin, qrain, emax)
>>> # or run from files
>>> data, wb = pdt.run_model(params, rainfall_file)
>>>
>>> # --- Calibration with genetic algorithm (requires DEAP) ---
>>> best, setup, log = pdt.run_calibration(rainfall_files, observed_files, setup_file)
>>>
>>> # --- Particle filtering (requires pypfilt) ---
>>> model = pdt.PavementModel(setup_file, rainfall_file)
>>> obs = pdt.PipeObs(observed_data)
>>>
>>> # --- Weather data download (requires herbie-data) ---
>>> downloader = pdt.HRRRAccumulatedPrecipitationDownloader(lat, lon)
>>> data = downloader.download_forecast(start_date, end_date)
>>>
>>> # --- Plotting (requires matplotlib) ---
>>> pdt.plots.plot_rainfall_hydrograph(rainfall_file, outflow_data)
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# Package version (PEP 621 / wheel metadata)
# -------------------------------------------------------------------------
try:
    from importlib import metadata as _md  # Python 3.8+
    __version__: str = _md.version(__name__)
except Exception:  # pragma: no cover
    __version__ = "0.1"

# -------------------------------------------------------------------------
# 1. Core water-flow functionality  (lightweight dependencies)
# -------------------------------------------------------------------------
from .water_flow_module import (           # noqa: E402  (import after future)
    run_simulation,
    run_from_files,
    read_setup_file,
    initialize_parameters,
    calculate_water_balance,
    run_model,
    results_dataframe,
    modify_parameters,
    read_rainfall_dat_file,
    rainfall_data_treatment,
    load_input_files,
    run_single_timestep,
)

# -------------------------------------------------------------------------
# 2. Plotting functionality  (optional: matplotlib, pandas)
# -------------------------------------------------------------------------
try:
    from . import plots  # noqa: E402
except ModuleNotFoundError:            # pragma: no cover
    plots = None     # keep attribute but signal unavailability

# -------------------------------------------------------------------------
# 3. GA calibration  (optional: DEAP, numpy, pandas)
#    We wrap it to avoid import-time failure if those libs are missing.
# -------------------------------------------------------------------------
def calibrate(*args, **kwargs):
    """
    Thin wrapper around :pyfunc:`permeabledt.calibration.main`.

    This is the legacy calibration method that works with folder structures.
    For new code, consider using :pyfunc:`run_calibration` instead.

    Raises
    ------
    RuntimeError
        If the optional `deap` (and friends) dependency is not installed.
    """
    try:
        from . import calibration as _calib  # local import to defer heavy deps
        return _calib.main(*args, **kwargs)
    except ModuleNotFoundError as exc:       # pragma: no cover
        raise RuntimeError(
            "The calibration feature requires extra dependencies "
            "(deap, numpy, pandas…). Install them with:\n"
            "    pip install 'permeabledt[calib]'"
        ) from exc


def run_calibration(*args, **kwargs):
    """
    Thin wrapper around :pyfunc:`permeabledt.calibration.run_calibration`.

    Run calibration using lists of rainfall and observed data files.

    Parameters
    ----------
    calibration_rainfall : list[Path or str]
        List of paths to rainfall files
    calibration_observed_data : list[Path or str]
        List of paths to observed outflow files
    setup_file : Path or str
        Path to setup file with calibration parameters
    output_setup_file : Path or str, optional
        Path to save calibrated setup file
    logbook_output_path : Path or str, optional
        Path to save logbook CSV
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    best_individual : list
        List of best parameter values
    calibrated_setup : ConfigParser
        ConfigParser object with calibrated parameters
    logbook : Logbook
        DEAP logbook with optimization history

    Raises
    ------
    RuntimeError
        If the optional `deap` (and friends) dependency is not installed.
    """
    try:
        from . import calibration as _calib  # local import to defer heavy deps
        return _calib.run_calibration(*args, **kwargs)
    except ModuleNotFoundError as exc:       # pragma: no cover
        raise RuntimeError(
            "The calibration feature requires extra dependencies "
            "(deap, numpy, pandas…). Install them with:\n"
            "    pip install 'permeabledt[calib]'"
        ) from exc


# -------------------------------------------------------------------------
# 4. Particle-filter model & observation classes  (optional: pypfilt, scipy)
# -------------------------------------------------------------------------
try:
    from .particle_filter import PavementModel, PipeObs  # noqa: E402
except ModuleNotFoundError:            # pragma: no cover
    PavementModel = PipeObs = None     # keep attributes but signal unavailability

# -------------------------------------------------------------------------
# 5. HRRR weather data downloader  (optional: herbie-data, xarray)
# -------------------------------------------------------------------------
try:
    from .download_HRRR_historical_forecast import HRRRAccumulatedPrecipitationDownloader  # noqa: E402
except ModuleNotFoundError:            # pragma: no cover
    HRRRAccumulatedPrecipitationDownloader = None     # keep attribute but signal unavailability


# -------------------------------------------------------------------------
# Public API surface
# -------------------------------------------------------------------------
__all__: list[str] = [
    "__version__",
    # water-flow core
    "run_simulation",
    "run_from_files",
    "read_setup_file",
    "initialize_parameters",
    "calculate_water_balance",
    "run_model",
    "results_dataframe",
    "modify_parameters",
    "read_rainfall_dat_file",
    "rainfall_data_treatment",
    "load_input_files",
    "run_single_timestep",
    # plotting
    "plots",
    # calibration
    "calibrate",
    "run_calibration",
    # particle filter
    "PavementModel",
    "PipeObs",
    # weather data
    "HRRRAccumulatedPrecipitationDownloader",
]