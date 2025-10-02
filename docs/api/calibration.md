# Calibration API

Reference for the genetic-algorithm calibration utilities in
`permeabledt.calibration`.

The public interface consists of the modern `run_calibration` function plus the
legacy `calibrate` wrapper kept for backward compatibility.  Both require the
optional calibration dependencies (`pip install "permeabledt[calib]"`).

## `run_calibration(calibration_rainfall, calibration_observed_data, setup_file, output_setup_file=None, logbook_output_path=None, seed=None)`

Run the DEAP-based single-objective genetic algorithm across one or more
calibration events.

**Arguments**
- `calibration_rainfall` (`list[str | Path]`): Paths to rainfall `.dat` files.
  Each entry must align with the observed outflow file at the same index.
- `calibration_observed_data` (`list[str | Path]`): Paths to observed outflow CSV
  files.  Each file must contain the modeled discharge column as the second
  column (converted internally from ft³/s to L/s).
- `setup_file` (`str | Path`): INI file containing calibration parameter bounds
  under the `[CALIBRATION_PARAMETERS]` section and genetic algorithm settings in
  `[CALIBRATION]`.
- `output_setup_file` (`str | Path`, optional): When provided, the calibrated
  setup is written back to disk using these parameter values.
- `logbook_output_path` (`str | Path`, optional): When set, a CSV export of the
  DEAP logbook (generation statistics and hall-of-fame individual) is written to
  this path.
- `seed` (`int`, optional): Random seed applied to both Python's `random` module
  and NumPy for reproducible runs.

**Behaviour**
- All rainfall/observed files must be supplied as lists of equal length.  An
  empty list raises a `ValueError`.
- Calibration parameter names and bounds are parsed from `setup_file`.  GA
  options such as population size, number of generations, crossover probability,
  and mutation probability are also read from the INI.
- Fitness evaluation delegates to the `evaluate` helper, which runs the water
  flow model for each event and compares modeled vs observed outflow.

**Returns**
- `best_individual` (`list[float]`): Parameter values from the hall-of-fame
  individual.
- `calibrated_setup` (`configparser.ConfigParser`): The modified configuration
  object with calibrated values inserted.
- `logbook` (`deap.tools.Logbook`): Evolutionary history, including generation
  statistics and the hall-of-fame individual per iteration.

**Example**
```python
import permeabledt as pdt

rain_events = ["data/event1_rain.dat", "data/event2_rain.dat"]
observed    = ["data/event1_obs.csv", "data/event2_obs.csv"]
setup_file  = "configs/tc_pf_site.ini"

best, calibrated_setup, logbook = pdt.run_calibration(
    rain_events,
    observed,
    setup_file,
    output_setup_file="outputs/calibrated.ini",
    logbook_output_path="outputs/logbook.csv",
    seed=123,
)

print("Best candidate:", best)
print("Generations recorded:", len(logbook))
```

## `calibrate(*args, **kwargs)`

Convenience wrapper that forwards all arguments to the legacy
`permeabledt.calibration.main` routine.  It is preserved for existing scripts
that rely on the historical folder layout and command-line behaviour.

**Notes**
- Raises `RuntimeError` if the optional calibration dependencies are missing.
- Returns whatever `calibration.main` returns (typically `None`).  For new code
  prefer `run_calibration`, which surfaces the best individual, calibrated setup,
  and DEAP logbook directly.
