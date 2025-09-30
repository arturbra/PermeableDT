import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from greendt.water_flow_module import read_rainfall_dat_file
import greendt as gdt
import os


def plot_rainfall_hydrograph(rainfall_file, outflow_data, rainfall_unit='mm', output_path=None):
    """
    Plot rainfall (top) and outflow (bottom) hydrograph with auto bar width.

    Parameters:
    -----------
    rain_df : pandas.DataFrame
        Must contain 'date' (datetime) and 'rain' columns.
    outflow_data : array-like or pandas.Series
        Outflow discharge time series (same length as rain_df).

    Returns:
    --------
    fig, (ax_rain, ax_flow)
    """

    outflow_data = outflow_data * 1000  # Convert to L/s
    rain_df = read_rainfall_dat_file(rainfall_file)

    # Calculate time step in days (auto width for bars)
    if len(rain_df) > 1:
        dt_seconds = (rain_df['date'].iloc[1] - rain_df['date'].iloc[0]).total_seconds()
        bar_width = dt_seconds / (24 * 60 * 60) * 4
    else:
        bar_width = 0.0007  # fallback

    fig, (ax_rain, ax_flow) = plt.subplots(
        2, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [1.5, 2.5]}
    )
    # Rainfall bars
    ax_rain.bar(rain_df['date'], rain_df['rain'],
                width=bar_width,
                alpha=0.7,
                color='black',
                label='Rainfall')
    ax_rain.invert_yaxis()
    ax_rain.set_ylabel(f'Rainfall ({rainfall_unit})')
    ax_rain.grid(True, alpha=0.3)

    total_rainfall = rain_df['rain'].sum()
    ax_rain.annotate(
        f"Total Rainfall: {total_rainfall:.2f} {rainfall_unit}",
        xy=(0.97, 0.9), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
    )

    # Outflow line plot
    ax_flow.plot(rain_df['date'], outflow_data,
                 linewidth=2,
                 color='black',
                 label='Outflow',
                 marker='o',
                 markersize=3)
    ax_flow.set_ylabel('Flow (L/s)')
    ax_flow.set_xlabel('Time')
    ax_flow.grid(True, alpha=0.3)

    plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    return fig, (ax_rain, ax_flow)


def plot_event_comparison(rainfall_files,
                          observed_files,
                          parameters,
                          rainfall_unit='in',
                          output_folder=None,
                          figure_size=(12, 10),
                          ncols=2):
    """
    Plot modeled vs observed outflow with rainfall for multiple calibration events,
    including annotation of total rainfall volume.

    Parameters:
    -----------
    rainfall_files : list of pathlib.Path or str
        List of paths to rainfall files (.dat)
    observed_files : list of pathlib.Path or str
        List of paths to observed outflow files (.csv)
    parameters : Parameters object or dict
        Model parameters to use for simulation
    rainfall_unit : str
        Unit for rainfall ('mm' or 'in')
    output_folder : str, optional
        Folder to save individual plot files (and combined figure)
    figure_size : tuple
        Figure size for each subplot (width, height)
    ncols : int
        Number of columns in subplot grid

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with all subplots
    axes : list of (ax_rain, ax_flow) tuples
    metrics : dict
        Dictionary with performance metrics for each event
    """
    n_events = len(rainfall_files)
    nrows    = int(np.ceil(n_events / ncols))

    fig     = plt.figure(figsize=(figure_size[0] * ncols,
                                   figure_size[1] * nrows))
    axes    = []
    metrics = {}

    for idx, (rain_file, obs_file) in enumerate(zip(rainfall_files, observed_files)):
        # --- 1) run model ---
        results, _ = gdt.run_simulation(
            parameters,
            str(rain_file),
            rainfall_unit=rainfall_unit,
            verbose=False,
            plot_outflow=False
        )

        # --- 2) load observed (CFS → L/s) ---
        df_obs   = pd.read_csv(obs_file)
        observed = df_obs.iloc[:, 1] * 28.316847

        # --- 3) modeled Qpipe (m³/s → L/s) ---
        modeled = results["Qpipe"] * 1000

        # --- 4) rainfall series + bar width ----
        rain_df = read_rainfall_dat_file(rain_file)
        if len(rain_df) > 1:
            dt          = (rain_df.date.iloc[1] - rain_df.date.iloc[0]).total_seconds()
            bar_width   = dt / (24 * 3600) * 4
        else:
            bar_width = 0.0007

        # Compute total rainfall volume
        total_rainfall = rain_df['rain'].sum()

        event_name = os.path.basename(rain_file).replace('.dat', '')

        # ==== Combined grid ====
        ax_rain = plt.subplot(nrows * 2, ncols, 2 * idx + 1)
        ax_flow = plt.subplot(nrows * 2, ncols, 2 * idx + 2,
                              sharex=ax_rain)

        # rainfall bars
        ax_rain.bar(rain_df.date, rain_df.rain,
                    width=bar_width, alpha=0.7, color='skyblue')
        ax_rain.invert_yaxis()
        ax_rain.set_ylabel(f'Rainfall ({rainfall_unit})')
        ax_rain.set_title(f'Event: {event_name}', weight='bold')
        ax_rain.grid(True, alpha=0.3)
        # annotate total rainfall
        ax_rain.annotate(
            f"Total Rainfall: {total_rainfall:.2f} {rainfall_unit}",
            xy=(0.97, 0.9), xycoords='axes fraction',
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      edgecolor='gray', alpha=0.8)
        )

        # flows with smaller markers
        ax_flow.plot(rain_df.date, observed,
                     marker='o', markersize=3, linestyle='-',
                     color='red', label='Observed')
        ax_flow.plot(rain_df.date, modeled[:len(observed)],
                     marker='s', markersize=2, linestyle='--',
                     color='blue', label='Modeled')
        ax_flow.set_ylabel('Flow (L/s)')
        ax_flow.set_xlabel('Time')
        ax_flow.grid(True, alpha=0.3)
        ax_flow.legend(loc='upper right')
        plt.setp(ax_flow.xaxis.get_majorticklabels(), rotation=45)

        # compute & annotate metrics
        obs_arr = np.asarray(observed, float)
        mod_arr = np.asarray(modeled[:len(observed)], float)
        valid   = ~np.isnan(obs_arr) & ~np.isnan(mod_arr)
        if valid.any():
            o    = obs_arr[valid]
            m    = mod_arr[valid]
            rmse = np.sqrt(np.mean((o - m)**2))
            nse  = 1 - ((o - m)**2).sum() / ((o - o.mean())**2).sum()
            r2   = np.corrcoef(o, m)[0,1]**2

            metrics[event_name] = {'RMSE': rmse, 'NSE': nse, 'R2': r2}
            txt = f'RMSE: {rmse:.2f}\nNSE:  {nse:.3f}\nR²:   {r2:.3f}'
            ax_flow.annotate(
                txt,
                xy=(0.96, 0.85), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor='gray', alpha=0.8)
            )

        axes.append((ax_rain, ax_flow))

        # ==== Individual event ====
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            indiv_fig, (ir, ifl) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True,
                gridspec_kw={'height_ratios': [1.5, 2.5]}
            )

            # rainfall
            ir.bar(rain_df.date, rain_df.rain,
                   width=bar_width, alpha=0.7, color='skyblue')
            ir.invert_yaxis()
            ir.set_ylabel(f'Rainfall ({rainfall_unit})')
            ir.set_title(f'Event: {event_name}', weight='bold')
            ir.grid(True, alpha=0.3)
            # annotate total rainfall
            ir.annotate(
                f"Rainfall Volume: {total_rainfall:.2f} {rainfall_unit}",
                xy=(0.97, 0.9), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor='gray', alpha=0.8)
            )

            # flows (small markers again)
            ifl.plot(rain_df.date, observed,
                     marker='o', markersize=3, linestyle='-',
                     color='red', label='Observed')
            ifl.plot(rain_df.date, modeled[:len(observed)],
                     marker='s', markersize=2, linestyle='--',
                     color='blue', label='Modeled')
            ifl.set_ylabel('Flow (L/s)')
            ifl.set_xlabel('Time')
            ifl.grid(True, alpha=0.3)
            ifl.legend(loc='upper right')
            plt.setp(ifl.xaxis.get_majorticklabels(), rotation=45)

            # annotate metrics
            if event_name in metrics:
                rmse = metrics[event_name]['RMSE']
                nse  = metrics[event_name]['NSE']
                r2   = metrics[event_name]['R2']
                txt = f'RMSE: {rmse:.2f}\nNSE:  {nse:.3f}\nR²:   {r2:.3f}'
                ifl.annotate(
                    txt,
                    xy=(0.96, 0.85), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white',
                              edgecolor='gray', alpha=0.8)
                )

            indiv_fig.tight_layout()
            indiv_fig.savefig(
                os.path.join(output_folder,
                             f'{event_name}_comparison.png'),
                dpi=300, bbox_inches='tight'
            )
            plt.close(indiv_fig)

    plt.tight_layout()
    if output_folder:
        fig.savefig(os.path.join(output_folder, 'all_events_comparison.png'),
                    dpi=300, bbox_inches='tight')

    return fig, axes, metrics


def plot_calibration_summary(metrics, output_path=None):
    """
    Create a summary bar plot of calibration metrics across all events.

    Parameters:
    -----------
    metrics : dict
        Dictionary with performance metrics for each event
        (output from plot_calibration_comparison)
    output_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """

    if not metrics:
        raise ValueError("No metrics to plot")

    # Prepare data for plotting
    events = list(metrics.keys())
    metric_names = ['RMSE', 'NSE', 'R²', 'MAE']

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics[event].get(metric_name.replace('²', '2'), 0) for event in events]

        # Create bar plot
        bars = ax.bar(range(len(events)), values, color='steelblue', alpha=0.8)

        # Customize based on metric
        if metric_name == 'NSE':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(-1, 1)
        elif metric_name == 'R²':
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)

        ax.set_xticks(range(len(events)))
        ax.set_xticklabels(events, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Event')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    return fig, axes