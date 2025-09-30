import numpy as np
import pypfilt
from scipy.stats import norm
import pickle
import os
from permeabledt.water_flow_module import read_setup_file, initialize_parameters, load_input_files, run_single_timestep, read_rainfall_dat_file, cnsz, cnusz
from scipy.stats import norm

class PavementModel(pypfilt.Model):
    """
    A pypfilt.Model wrapper around your single‐timestep water flow function.

    Adds:
      • Dry-latch logic for sustained zero-forecast rainfall:
          - rain forced to 0 for all particles
          - process noise scaled down (often to 0)
          - optional temporary "cooling" (ensemble spread shrink)
      • One-time "handoff cooling" at the first forecast step
    """

    PRIORS = {
        "hp":   (0.0,   0.05),
        "hsz":  (1e-6,  0.20),
        "s":    (0.0,   1.0),
        # Derived variables - no priors needed as they are computed from the minimal state
        # "husz": computed from L - hsz
        # "nusz": computed from husz, hsz, and parameters
        # "nsz":  computed from hsz and parameters
        # "Qpipe": diagnostic output, not a state variable
    }

    # --------- helpers ----------
    @staticmethod
    def _count_back(arr, idx, predicate, limit):
        """Count consecutive steps backwards from idx that satisfy predicate."""
        c = 0
        for j in range(limit):
            k = idx - j
            if k < 0:
                break
            if predicate(arr[k]):
                c += 1
            else:
                break
        return c

    # --------- PF API ----------
    def field_types(self, ctx):
        return [
            ("hp", np.float64),
            ("s", np.float64),
            ("hsz", np.float64),
            ("husz", np.float64),
            ("nusz", np.float64),
            ("nsz", np.float64),
            ("Qpipe", np.float64),
        ]

    def init(self, ctx, state_vec):
        # Model parameters
        setup_file = ctx.settings['model']["setup_file"]
        setup = read_setup_file(setup_file)
        parameters = initialize_parameters(setup)
        ctx.settings["params"] = parameters

        # Forcings
        rainfall_file = ctx.settings['model']["rainfall_file"]
        inflow, rainfall_converted, evt = load_input_files(parameters, rainfall_file)
        ctx.settings["inflow"] = np.array(inflow)
        ctx.settings["rainfall"] = np.array(rainfall_converted)  # flow-converted
        ctx.settings["evt"] = np.array(evt)

        # Raw rainfall (inches) for uncertainty logic
        rainfall_df = read_rainfall_dat_file(rainfall_file)
        ctx.settings["rainfall_base"] = np.array(rainfall_df['rain'].values)

        # Optional KDE
        kde_model = None
        rainfall_kde_file = ctx.settings['model'].get("rainfall_kde_file", None)
        if rainfall_kde_file and os.path.exists(rainfall_kde_file):
            with open(rainfall_kde_file, 'rb') as f:
                kde_model = pickle.load(f)
        ctx.settings["kde_model"] = kde_model

        # Persistent flags
        self._forecast_start_time = None
        self._dry_latched = False
        self._dry_cool_steps = 0

        # Init minimal states from prior
        prior = ctx.data["prior"]
        p = ctx.settings["params"]
        L = p["l"]

        # Initialize minimal state variables from priors first
        for key in ["hp", "s", "hsz"]:
            if key in prior and key in self.PRIORS:
                state_vec[key] = prior[key]
            elif key == "hp":
                state_vec[key] = prior.get("hp", 0.0)
            elif key == "s":
                state_vec[key] = prior.get("s", 0.0)
            elif key == "hsz":
                state_vec[key] = prior.get("hsz", max(p["hpipe"], 1e-6) if p["hpipe"] > 0 else p["dpipe"])

        # Now compute derived variables for all particles
        n_particles = len(state_vec)
        for i in range(n_particles):
            # Compute husz from hsz
            state_vec["husz"][i] = max(L - state_vec["hsz"][i], 1e-6)

            # Compute nsz from hsz
            state_vec["nsz"][i] = max(cnsz(state_vec["hsz"][i], L, p["dtl"], p["dg"], p["nf"], p["nt"], p["ng"]), 1e-6)

            # Compute nusz from husz and hsz
            state_vec["nusz"][i] = max(cnusz(state_vec["husz"][i], state_vec["hsz"][i], p["nusz_ini"], p["ng"], p["dg"], p["df"]), 1e-6)

            # Initialize Qpipe
            state_vec["Qpipe"][i] = 0.0

    def update(self, ctx, time_step, is_fs, prev, curr):
        step_idx = int(time_step.end)

        t = time_step.end
        p  = ctx.settings["params"]
        qin  = ctx.settings["inflow"][step_idx]
        emax = ctx.settings["evt"][step_idx]

        n_particles = prev["hp"].shape[0]

        # Forecast window detection
        if is_fs and self._forecast_start_time is None:
            self._forecast_start_time = step_idx

            # One-time handoff cooling (shrinks analysis spread before forecast)
            # Only apply to minimal state variables
            gamma0 = ctx.settings['model'].get("handoff_cooling_gamma", 0.92)
            for key in ["hp", "s", "hsz"]:
                arr = prev[key]
                mu = float(np.mean(arr))
                prev[key] = mu + gamma0 * (arr - mu)

            # Recompute derived variables after cooling
            L = p["l"]
            n_particles = len(prev["hp"])
            for i in range(n_particles):
                prev["husz"][i] = max(L - prev["hsz"][i], 1e-6)
                prev["nsz"][i] = max(cnsz(prev["hsz"][i], L, p["dtl"], p["dg"], p["nf"], p["nt"], p["ng"]), 1e-6)
                prev["nusz"][i] = max(cnusz(prev["husz"][i], prev["hsz"][i], p["nusz_ini"], p["ng"], p["dg"], p["df"]), 1e-6)

        in_forecast_period = (self._forecast_start_time is not None and step_idx >= self._forecast_start_time)

        # Rainfall uncertainty
        kde_model     = ctx.settings.get("kde_model", None)
        rainfall_base = ctx.settings.get("rainfall_base", None)

        # Defaults for dry logic (TOML knobs)
        use_mag = ctx.settings['model'].get("use_magnitude_uncertainty_model", True)
        zero_thr_in = ctx.settings['model'].get("zero_rainfall_threshold", 0.02)     # inches (≈0.5 mm)
        enter_zeros = ctx.settings['model'].get("consecutive_zero_periods", 4)      # steps to enter dry (1 h @ 15 min)
        exit_wets   = ctx.settings['model'].get("consecutive_wet_periods", 2)       # steps to exit dry
        dry_noise_scale = ctx.settings['model'].get("dry_noise_scale", 0.0)         # 0.0 => deterministic
        dry_cooling_steps = ctx.settings['model'].get("dry_cooling_steps", 4)       # shrink N steps after latch
        dry_cooling_gamma = ctx.settings['model'].get("dry_cooling_gamma", 0.90)

        # Compute rainfall samples
        if kde_model and (rainfall_base is not None) and in_forecast_period:
            base_rain_in = rainfall_base[step_idx]  # inches (from file you created)

            # --- DRY LATCH decision ---
            if use_mag:
                is_zero_now = (base_rain_in <= zero_thr_in)
                zeros_back = self._count_back(rainfall_base, step_idx, lambda r: r <= zero_thr_in, 64)
                wets_back  = self._count_back(rainfall_base, step_idx, lambda r: r >  zero_thr_in, 64)

                if (not self._dry_latched) and (zeros_back >= enter_zeros):
                    self._dry_latched = True
                    self._dry_cool_steps = dry_cooling_steps

                if self._dry_latched and (wets_back >= exit_wets) and (not is_zero_now):
                    self._dry_latched = False

            # If latched dry: force rain=0 for all particles
            if self._dry_latched:
                rainfall_values_inches = np.zeros(n_particles, dtype=float)
            else:
                # ---- your magnitude-based residual model ----
                low_threshold  = ctx.settings['model'].get("rainfall_low_threshold", 0.12)
                high_threshold = ctx.settings['model'].get("rainfall_high_threshold", 0.12)

                if base_rain_in < low_threshold:
                    bias = ctx.settings['model'].get("low_rainfall_bias", 0.004)
                    std  = ctx.settings['model'].get("low_rainfall_variance", 0.020)
                    rainfall_values_inches = np.random.normal(loc=base_rain_in + bias, scale=std, size=n_particles)
                elif base_rain_in >= high_threshold:
                    b_slope = ctx.settings['model'].get("high_rainfall_bias_slope", -0.024)
                    b_int   = ctx.settings['model'].get("high_rainfall_bias_intercept", 0.059)
                    v_slope = ctx.settings['model'].get("high_rainfall_variance_slope", 0.85)
                    v_int   = ctx.settings['model'].get("high_rainfall_variance_intercept", 0.008)
                    bias = b_slope * base_rain_in + b_int
                    std  = v_slope * base_rain_in + v_int
                    rainfall_values_inches = np.random.normal(loc=base_rain_in + bias, scale=std, size=n_particles)
                else:
                    # blend between low and high
                    blend = (base_rain_in - low_threshold) / max(1e-9, (high_threshold - low_threshold))
                    low_bias = ctx.settings['model'].get("low_rainfall_bias", 0.004)
                    low_std  = ctx.settings['model'].get("low_rainfall_variance", 0.020)
                    b_slope = ctx.settings['model'].get("high_rainfall_bias_slope", -0.024)
                    b_int   = ctx.settings['model'].get("high_rainfall_bias_intercept", 0.059)
                    v_slope = ctx.settings['model'].get("high_rainfall_variance_slope", 0.85)
                    v_int   = ctx.settings['model'].get("high_rainfall_variance_intercept", 0.008)
                    high_bias = b_slope * base_rain_in + b_int
                    high_std  = v_slope * base_rain_in + v_int
                    bias = low_bias * (1 - blend) + high_bias * blend
                    std  = low_std  * (1 - blend) + high_std  * blend
                    rainfall_values_inches = np.random.normal(loc=base_rain_in + bias, scale=std, size=n_particles)

            # clip negatives and convert to flow units
            rainfall_values_inches = np.maximum(0.0, rainfall_values_inches)
            area = ctx.settings["params"]['area']
            dt   = ctx.settings["params"]['dt']
            rainfall_values = (rainfall_values_inches * 0.0254 * area) / dt

        else:
            # historical period or no KDE: deterministic forcing from pre-converted series
            base_rainfall_flow = ctx.settings.get("rainfall", [0.0])[step_idx]
            rainfall_values = np.full(n_particles, base_rainfall_flow, dtype=float)

        # --------- propagate particles ---------
        # In filtering window, process noise factor
        base_proc_fac = 0.002
        if in_forecast_period and self._dry_latched:
            proc_fac = base_proc_fac * dry_noise_scale
        else:
            proc_fac = base_proc_fac

        for i in range(n_particles):
            qrain = rainfall_values[i]

            out = run_single_timestep(
                p, qin, qrain, emax,
                prev["hp"][i], prev["s"][i], prev["hsz"][i], prev["husz"][i],
                prev["nusz"][i], prev["nsz"][i]
            )

            # perturb & clip during filtering only
            if t <= ctx.settings["filter"].get("forecast_time", np.inf):
                # Adaptive process noise based on system activity
                current_flow = out["Qpipe"]
                rainfall_rate = qrain

                # Activity-based scaling factor (higher activity = more noise)
                base_activity = max(current_flow * 1e3, 0.001)  # Convert to L/s, minimum activity
                rain_activity = max(rainfall_rate * 1e6, 0.001)  # Scale rainfall appropriately
                activity_factor = min(np.sqrt(base_activity + rain_activity) / 10.0, 1.0)

                # Reduce process noise significantly during low-activity periods
                adaptive_proc_fac = proc_fac * max(activity_factor, 0.1)  # Minimum 10% of base noise

                for key in ["hp", "s", "hsz"]:
                    lo, scale = self.PRIORS[key]
                    hi = lo + scale
                    val = out[key] + np.random.normal(0.0, adaptive_proc_fac * scale)
                    curr[key][i] = np.clip(val, lo, hi)
            else:
                curr["hp"][i]  = np.clip(out["hp"],  *self.PRIORS["hp"])
                curr["s"][i]   = np.clip(out["s"],   *self.PRIORS["s"])
                curr["hsz"][i] = np.clip(out["hsz"], *self.PRIORS["hsz"])

            # dependent/diagnostic states
            L = p["l"]
            curr["husz"][i] = max(L - curr["hsz"][i], 1e-6)
            curr["nsz"][i]  = max(cnsz(curr["hsz"][i], L, p["dtl"], p["dg"], p["nf"], p["nt"], p["ng"]), 1e-6)
            curr["nusz"][i] = max(cnusz(curr["husz"][i], curr["hsz"][i], p["nusz_ini"], p["ng"], p["dg"], p["df"]), 1e-6)
            curr["Qpipe"][i] = out["Qpipe"]


        # --- Enhanced forecast transition cooling ---
        if in_forecast_period:
            should_cool = False
            gamma = 0.95  # Default cooling factor

            # Dry latched cooling (existing behavior)
            if self._dry_latched and (self._dry_cool_steps > 0):
                should_cool = True
                gamma = float(dry_cooling_gamma)
                self._dry_cool_steps -= 1

            # Early forecast cooling (new: prevent sudden uncertainty expansion)
            elif step_idx - self._forecast_start_time < 8:  # Cool for first 2 hours (8 * 15min)
                should_cool = True
                # Gradual cooling decay: more cooling initially, less over time
                steps_into_forecast = step_idx - self._forecast_start_time
                cooling_strength = max(0.3, 1.0 - (steps_into_forecast / 8.0))
                gamma = 0.85 + (0.15 * (1.0 - cooling_strength))  # Range from 0.85 to 1.0

            # Low activity cooling (new: prevent uncertainty growth during recession)
            elif not self._dry_latched:
                # Check if system is in low-activity state
                median_flow = np.median(curr["Qpipe"]) * 1e3  # Convert to L/s
                median_rain = np.median(rainfall_values) * 1e6  # Scale appropriately

                if median_flow < 0.05 and median_rain < 0.01:  # Very low activity
                    should_cool = True
                    gamma = 0.98  # Gentle cooling to prevent excessive spread

            if should_cool:
                # Only cool minimal state variables
                for key in ["hp", "s", "hsz"]:
                    arr = curr[key]
                    mu = float(np.mean(arr))
                    curr[key] = mu + gamma * (arr - mu)

                # Recompute derived variables after cooling
                L = p["l"]
                for i in range(n_particles):
                    curr["husz"][i] = max(L - curr["hsz"][i], 1e-6)
                    curr["nsz"][i] = max(cnsz(curr["hsz"][i], L, p["dtl"], p["dg"], p["nf"], p["nt"], p["ng"]), 1e-6)
                    curr["nusz"][i] = max(cnusz(curr["husz"][i], curr["hsz"][i], p["nusz_ini"], p["ng"], p["dg"], p["df"]), 1e-6)

        # expose latch flag in context (useful for plot/diagnostics or disabling regularisation hooks)
        ctx.settings["__dry_latched__"] = self._dry_latched
        ctx.settings['final_hp'] = curr['hp']
        ctx.settings['final_s'] = curr['s']
        ctx.settings['final_hsz'] = curr['hsz']

    def can_smooth(self):
        return {"hp", "s", "hsz", "husz", "Qpipe", "nusz", "nsz"}


class PipeObs(pypfilt.obs.Univariate):
    def distribution(self, ctx, snapshot):
        expected = snapshot.state_vec["Qpipe"]      # e.g. shape (N,)
        
        # Analytical uncertainty using delta method for weir rating curve
        # Parameters for 6-inch Thel-Mar weir: Q(h) = k * h^n
        k = ctx.settings.get('observations', {}).get('Qpipe', {}).get('weir_k', 0.006)
        n = ctx.settings.get('observations', {}).get('Qpipe', {}).get('weir_n', 2.5532)
        
        # Head measurement error (converted from uniform bounds to standard deviation)
        sigma_h_inches = ctx.settings.get('observations', {}).get('Qpipe', {}).get('head_error_inches', 0.08)
        sigma_h = (sigma_h_inches / np.sqrt(3)) * 0.0254  # Convert inches to meters, uniform to std dev
        
        # Weir calibration relative error (converted from uniform bounds to standard deviation)
        u_r_uniform = ctx.settings.get('observations', {}).get('Qpipe', {}).get('weir_rel_error', 0.05)
        u_r = u_r_uniform / np.sqrt(3)  # Convert uniform bounds to standard deviation
        
        # Handle zero/near-zero flows to avoid numerical issues
        min_flow = 1e-6  # Minimum flow for numerical stability (1 μm³/s)
        Q_safe = np.maximum(np.abs(expected), min_flow)
        
        # Head error propagated through weir rating curve
        # σ_{Q,head}(Q) = n * k^(1/n) * σ_h * Q^((n-1)/n)
        sigma_Q_head = n * (k**(1/n)) * sigma_h * (Q_safe**((n-1)/n))
        
        # Weir calibration error (multiplicative)
        # σ_{Q,weir}(Q) = u_r * Q
        sigma_Q_weir = u_r * Q_safe
        
        # Total linear-space variance (assuming independence)
        # σ_Q^2(Q) = σ_{Q,head}^2(Q) + σ_{Q,weir}^2(Q)
        sigma_Q_linear = np.sqrt(sigma_Q_head**2 + sigma_Q_weir**2)
        
        # Apply uncertainty floors for numerical stability
        min_absolute_std = ctx.settings.get('observations', {}).get('Qpipe', {}).get('min_absolute_uncertainty', 5e-6)
        min_relative_factor = ctx.settings.get('observations', {}).get('Qpipe', {}).get('min_relative_uncertainty', 0.05)
        
        # Ensure minimum uncertainty levels
        normal_std = np.maximum(sigma_Q_linear, np.maximum(min_absolute_std, Q_safe * min_relative_factor))
        
        # For negative flows, ensure we don't create invalid probabilities by truncating at zero
        # The normal distribution naturally handles this through its support over all real numbers
        return norm(loc=expected, scale=normal_std)  # Use original expected values, not Q_safe
