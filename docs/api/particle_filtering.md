# Particle Filtering API

`permeabledt.particle_filter` provides helper classes that integrate the water
flow solver with [pypfilt](https://pypfilt.readthedocs.io/) for sequential Monte
Carlo assimilation.  The module exposes a `pypfilt.Model` implementation for the
pavement dynamics and an observation model for pipe outflow measurements.

Particle filtering is an optional feature.  Install the extra dependencies via:

```bash
pip install "permeabledt[pf]"
```

## `PavementModel`

Subclass of `pypfilt.Model` that wraps the single-timestep water-flow solver.
Rather than using a bespoke constructor, all configuration is supplied through
the pypfilt context (`ctx.settings`).

### Required context settings

Place these keys in your TOML configuration or programmatic `ctx.settings`:

- `model.setup_file`: Path to the INI file used by
  `water_flow_module.read_setup_file`.
- `model.rainfall_file`: Path to the rainfall `.dat` file that provides the
  forcing history.  The loader converts the rainfall to flows internally.
- Optional knobs controlling forecast behaviour (dry latch, KDE file paths,
  uncertainty tuning) mirror the dictionary keys consumed inside `update`.

### Interface overview

- `field_types(ctx)` – returns the structured dtype describing the state vector.
  The state contains `hp`, `s`, `hsz`, plus derived quantities (`husz`, `nusz`,
  `nsz`) and the diagnostic `Qpipe` flow.
- `init(ctx, state_vec)` – reads configuration, loads rainfall/ET series, and
  populates the initial particle states (including derived variables).
- `update(ctx, time_step, is_fs, prev, curr)` – advances each particle one step
  by calling `run_single_timestep` and applies rainfall uncertainty logic.  It
  also manages forecast “dry latch” behaviour and optional handoff cooling.
- `can_smooth()` – returns the set of state fields that support smoothing.

### Minimal usage example

```python
import pathlib
import pypfilt
import permeabledt as pdt

cfg = pypfilt.toml.parse(
    {
        "time": {"start": 0, "stop": 96, "step": 1},
        "model": {
            "cls": "permeabledt.particle_filter.PavementModel",
            "setup_file": "configs/tc_pf_site.ini",
            "rainfall_file": "data/rain_event.dat",
        },
        "observations": {
            "cls": "permeabledt.particle_filter.PipeObs",
        },
        "resample": {"method": "systematic"},
        "particles": {"size": 500},
    }
)

results = pypfilt.run(cfg)
```

All per-particle calculations use SI units that match the underlying solver
(metres, seconds, cubic metres per second).

## `PipeObs`

Observation model for pipe outflow.  Inherits from `pypfilt.obs.Univariate` and
returns a `scipy.stats.norm` distribution via its `distribution` method.

### Configuration

Observation uncertainty is driven by nested dictionaries inside
`ctx.settings['observations']['Qpipe']`:

- `weir_k`, `weir_n` – rating-curve coefficients (default 0.006 and 2.5532).
- `head_error_inches` – head measurement error bounds (uniform, converted to a
  standard deviation internally).
- `weir_rel_error` – relative error on the rating curve (uniform bounds).
- `min_absolute_uncertainty`, `min_relative_uncertainty` – numerical floors.

### Method summary

- `distribution(ctx, snapshot)` – receives the current particle snapshot,
  extracts the modeled `Qpipe` series, and returns a `norm` distribution whose
  variance combines head measurement error and rating-curve uncertainty using a
  delta-method approximation.  Negative flows are handled gracefully by clipping
  only the uncertainty calculations.

### Example likelihood usage

```python
obs_model = pdt.PipeObs()

def log_likelihood(ctx, snapshot, observation_value):
    dist = obs_model.distribution(ctx, snapshot)
    return dist.logpdf(observation_value)
```

This observation model is typically referenced from pypfilt’s TOML configuration
by specifying the class path (`permeabledt.particle_filter.PipeObs`).
