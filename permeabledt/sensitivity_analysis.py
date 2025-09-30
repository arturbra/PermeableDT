import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import configparser
import os
from permeabledt.water_flow_module import initialize_parameters, modify_parameters, run_model, results_dataframe, calculate_water_balance


# Import your model functions (assuming they're in a module called 'model')
# from model import initialize_parameters, modify_parameters, run_model, results_dataframe, calculate_water_balance


class SobolSensitivityAnalysis:
    """
    Performs Sobol sensitivity analysis on the permeable pavement model
    for peak outflow, total outflow, and time to peak.
    """

    def __init__(self, setup_file, rainfall_file):
        """
        Initialize the sensitivity analysis with model setup and data files.

        Parameters:
        -----------
        setup_file : str
            Path to the .ini setup file
        rainfall_file : str
            Path to the rainfall data file
        """
        self.setup_file = setup_file
        self.rainfall_file = rainfall_file

        # Read base parameters
        self.setup = self._read_setup_file(setup_file)
        self.base_params = initialize_parameters(self.setup)

        # Extract calibration bounds
        self.param_bounds = self._extract_calibration_bounds()
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)

    def _read_setup_file(self, setup_file):
        """Read the setup configuration file"""
        setup = configparser.ConfigParser()
        setup.read(setup_file)
        return setup

    def _extract_calibration_bounds(self):
        """Extract parameter bounds from the CALIBRATION section"""
        bounds = {}

        if 'CALIBRATION' in self.setup:
            cal_section = self.setup['CALIBRATION']

            # Extract unique parameter names
            param_names = set()
            for key in cal_section:
                if key.endswith('_min') or key.endswith('_max'):
                    param_name = key.rsplit('_', 1)[0]
                    param_names.add(param_name)

            # Build bounds dictionary
            for param in param_names:
                if f"{param}_min" in cal_section and f"{param}_max" in cal_section:
                    bounds[param] = [
                        float(cal_section[f"{param}_min"]),
                        float(cal_section[f"{param}_max"])
                    ]

        return bounds

    def _run_single_simulation(self, param_values):
        """
        Run a single simulation with given parameter values.

        Returns peak outflow, total outflow, and time to peak.
        """
        # Create modified parameters
        param_dict = dict(zip(self.param_names, param_values))
        modified_params = modify_parameters(self.base_params.copy(), param_dict)

        try:
            # Run the model
            results = run_model(modified_params, self.rainfall_file)
            data = results_dataframe(results)
            wb = calculate_water_balance(data, modified_params['dt'])

            # Extract the three key outputs
            outputs = {
                'peak_outflow': data['Qpipe'].max() * 1000,  # L/s
                'total_outflow': wb['Vtotal_pipe (m3)'].iloc[0],  # m3
                'time_to_peak': wb['tpeak (min)'].iloc[0],  # min
            }

            return outputs

        except Exception as e:
            print(f"Error in simulation: {e}")
            # Return NaN values if simulation fails
            return {
                'peak_outflow': np.nan,
                'total_outflow': np.nan,
                'time_to_peak': np.nan
            }

    def run_analysis(self, n_samples=1024, calc_second_order=True, parallel=True, n_workers=None):
        """
        Perform Sobol sensitivity analysis.

        Parameters:
        -----------
        n_samples : int
            Number of samples for the analysis (should be power of 2)
        calc_second_order : bool
            Whether to calculate second-order indices
        parallel : bool
            Whether to run simulations in parallel
        n_workers : int, optional
            Number of parallel workers (defaults to CPU count)

        Returns:
        --------
        results : dict
            Dictionary containing Sobol indices for each output
        """
        print("Generating Sobol samples...")

        # Define the problem
        problem = {
            'num_vars': self.n_params,
            'names': self.param_names,
            'bounds': list(self.param_bounds.values())
        }

        # Generate samples
        param_samples = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
        n_runs = param_samples.shape[0]
        print(f"Total simulations to run: {n_runs}")

        # Initialize output storage
        output_names = ['peak_outflow', 'total_outflow', 'time_to_peak']
        outputs = {name: np.zeros(n_runs) for name in output_names}

        # Run simulations
        print("Running simulations...")
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self._run_single_simulation, params)
                           for params in param_samples]

                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                                total=n_runs, desc="Simulations")):
                    result = future.result()
                    for output_name in output_names:
                        outputs[output_name][i] = result[output_name]
        else:
            for i, params in enumerate(tqdm(param_samples, desc="Simulations")):
                result = self._run_single_simulation(params)
                for output_name in output_names:
                    outputs[output_name][i] = result[output_name]

        # Analyze results
        print("\nAnalyzing Sobol indices...")
        sobol_results = {}

        for output_name in output_names:
            # Filter out NaN values
            valid_mask = ~np.isnan(outputs[output_name])
            if np.sum(valid_mask) < n_runs * 0.9:  # If more than 10% failed
                print(f"Warning: {output_name} has {np.sum(~valid_mask)} failed simulations")

            try:
                Si = sobol.analyze(problem, outputs[output_name], calc_second_order=calc_second_order)
                sobol_results[output_name] = Si
            except Exception as e:
                print(f"Error analyzing {output_name}: {e}")
                sobol_results[output_name] = None

        return sobol_results

    def plot_indices(self, sobol_results, figsize=(15, 5)):
        """
        Plot Sobol sensitivity indices for all three outputs.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        output_names = ['peak_outflow', 'total_outflow', 'time_to_peak']
        titles = ['Peak Outflow (L/s)', 'Total Outflow (m³)', 'Time to Peak (min)']

        for ax, output_name, title in zip(axes, output_names, titles):
            if output_name not in sobol_results or sobol_results[output_name] is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(title)
                continue

            Si = sobol_results[output_name]

            # Prepare data for plotting
            params = self.param_names
            S1 = Si['S1']
            ST = Si['ST']

            # Sort by total-order indices
            sorted_indices = np.argsort(ST)[::-1]
            params_sorted = [params[i] for i in sorted_indices]
            S1_sorted = S1[sorted_indices]
            ST_sorted = ST[sorted_indices]

            # Create grouped bar plot
            x = np.arange(len(params))
            width = 0.35

            bars1 = ax.bar(x - width / 2, S1_sorted, width, label='First-order', alpha=0.8)
            bars2 = ax.bar(x + width / 2, ST_sorted, width, label='Total-order', alpha=0.8)

            ax.set_xlabel('Parameters')
            ax.set_ylabel('Sobol Index')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(params_sorted, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return fig

    def save_results(self, results, output_dir='sensitivity_results'):
        """
        Save sensitivity analysis results to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine all results into one DataFrame
        all_results = []

        for output_name, Si in results.items():
            if Si is None:
                continue

            for i, param in enumerate(self.param_names):
                all_results.append({
                    'Output': output_name,
                    'Parameter': param,
                    'S1': Si['S1'][i],
                    'S1_conf': Si['S1_conf'][i],
                    'ST': Si['ST'][i],
                    'ST_conf': Si['ST_conf'][i],
                    'Interaction': Si['ST'][i] - Si['S1'][i]
                })

        df = pd.DataFrame(all_results)
        filename = f"{output_dir}/sobol_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

        # Also save a summary of most influential parameters
        summary = []
        for output_name in ['peak_outflow', 'total_outflow', 'time_to_peak']:
            if output_name in results and results[output_name] is not None:
                Si = results[output_name]
                for i, param in enumerate(self.param_names):
                    if Si['ST'][i] > 0.05:  # Only include parameters with >5% influence
                        summary.append({
                            'Output': output_name,
                            'Parameter': param,
                            'Total_Effect': Si['ST'][i],
                            'Rank': np.argsort(Si['ST'])[::-1].tolist().index(i) + 1
                        })

        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.sort_values(['Output', 'Rank'])
        summary_filename = f"{output_dir}/influential_parameters_{timestamp}.csv"
        df_summary.to_csv(summary_filename, index=False)
        print(f"Summary saved to {summary_filename}")

    def print_summary(self, results):
        """
        Print a summary of the most influential parameters.
        """
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("=" * 60)

        for output_name in ['peak_outflow', 'total_outflow', 'time_to_peak']:
            if output_name not in results or results[output_name] is None:
                continue

            Si = results[output_name]
            print(f"\n{output_name.upper().replace('_', ' ')}:")
            print("-" * 40)

            # Sort by total-order indices
            param_importance = sorted(zip(self.param_names, Si['ST'], Si['S1']),
                                      key=lambda x: x[1], reverse=True)

            print(f"{'Parameter':<12} {'Total':<8} {'First':<8} {'Interaction':<12}")
            print("-" * 40)

            for param, st, s1 in param_importance:
                if st > 0.01:  # Only show parameters with >1% influence
                    interaction = st - s1
                    print(f"{param:<12} {st:<8.3f} {s1:<8.3f} {interaction:<12.3f}")


# Example usage
if __name__ == "__main__":
    # Initialize the sensitivity analysis
    setup_file = r"C:\Users\Artur\PycharmProjects\PermeableDT\tests\output\calibrated_parameters_PA.ini"
    rainfall_file = r"C:\Users\Artur\PycharmProjects\PermeableDT\tests\input\calibration\rainfall_event_1.dat"

    # Create analysis instance
    gsa = SobolSensitivityAnalysis(setup_file, rainfall_file)

    # Run Sobol analysis
    results = gsa.run_analysis(
        n_samples=1024,  # Use power of 2 for better convergence
        calc_second_order=True,
        parallel=True
    )

    # Generate plots
    fig = gsa.plot_indices(results)
    plt.savefig('sobol_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    gsa.save_results(results)

    # Print summary
    gsa.print_summary(results)