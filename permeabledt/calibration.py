import random
import numpy as np
from deap import base, creator, tools, algorithms
import greendt as gdt
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path


def parse_calibration_parameters(setup):
    calibration_params = {}
    for key, val in setup['CALIBRATION'].items():
        if key.endswith('_min') or key.endswith('_max'):
            base_key = key.replace('_min', '').replace('_max', '')
            # Preserve the original case by using the base_key as it is
            if base_key not in calibration_params:
                calibration_params[base_key] = {}
            if key.endswith('_min'):
                calibration_params[base_key]['min'] = float(val)
            else:
                calibration_params[base_key]['max'] = float(val)
    return calibration_params


def update_setup_with_values(setup, values):
    """
    Given a ConfigParser `setup` and a list of floats `values` (in the same
    order as parse_calibration_parameters), find each parameter in the
    other sections and overwrite it with the corresponding value.
    """
    calib = parse_calibration_parameters(setup)
    names = list(calib.keys())
    if len(names) != len(values):
        raise ValueError(f"Expected {len(names)} values, got {len(values)}")

    for name, val in zip(names, values):
        placed = False
        # look through every section (except CALIBRATION) for an option matching name
        for section in setup.sections():
            if section == 'CALIBRATION':
                continue
            if setup.has_option(section, name):
                setup.set(section, name, str(val))
                placed = True
                break
        if not placed:
            raise KeyError(f"Could not find a section containing option '{name}' to update.")
    return setup


def nash_sutcliffe(observed, modeled):
    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(modeled, dtype=float)
    valid = ~np.isnan(obs) & ~np.isnan(mod)
    if not np.any(valid):
        return np.nan

    observed_mean = obs[valid].mean()
    numerator = ((obs[valid] - mod[valid]) ** 2).sum()
    denominator = ((obs[valid] - observed_mean) ** 2).sum()
    return 1 - (numerator / denominator)


def root_mean_square_error(observed, modeled):
    obs = np.asarray(observed, dtype=float)
    mod = np.asarray(modeled, dtype=float)
    valid = ~np.isnan(obs) & ~np.isnan(mod)
    if not np.any(valid):
        return np.nan

    return np.sqrt(mean_squared_error(obs[valid], mod[valid]))


def individual_to_dict(calibration_params, individual):
    # Ensure that the number of parameters matches the length of the individual list
    if len(calibration_params) != len(individual):
        raise ValueError("The number of parameters does not match the length of the individual list")

    # Create a new dictionary to hold the associations
    associated_dict = {}

    # Iterate through the keys of calibration_params and associate with individual values
    for i, key in enumerate(calibration_params.keys()):
        associated_dict[key] = individual[i]

    return associated_dict


def evaluate(individual,
             calibration_params,
             calibration_rainfall,
             calibration_observed_data,
             setup_file):
    """
    Evaluate an individual by running simulations for all calibration events
    and comparing with observed data.

    Parameters:
    - individual: list of parameter values
    - calibration_params: dict of calibration parameter bounds
    - calibration_rainfall: list of paths to rainfall files
    - calibration_observed_data: list of paths to observed outflow files
    - setup_file: path to setup file
    """
    total_score = 0

    # Read the base setup
    setup = gdt.water_flow_module.read_setup_file(setup_file)

    # Update setup with individual's parameter values
    modified_params = individual_to_dict(calibration_params, individual)
    setup = update_setup_with_values(setup, individual)

    # Initialize parameters from the modified setup
    parameters = gdt.water_flow_module.initialize_parameters(setup)

    # Update parameters with modified values
    for param_name, value in modified_params.items():
        if hasattr(parameters, param_name):
            setattr(parameters, param_name, value)

    # Run simulation for each event
    for rainfall_file, observed_file in zip(calibration_rainfall, calibration_observed_data):
        try:
            # Run the simulation
            results, water_balance = gdt.run_simulation(
                parameters,
                str(rainfall_file),
                rainfall_unit='in',
                verbose=False,
                plot_outflow=False
            )

            # Load observed data
            observed_data = pd.read_csv(observed_file)
            observed = observed_data.iloc[:, 1] * 0.028316847 # Convert from CFS to m3/s

            modeled = results["Qpipe"]
            modeled = modeled # Convert units if needed

            # Calculate RMSE
            rmse = root_mean_square_error(observed, modeled)

            # Calculate NSE
            nse = -nash_sutcliffe(observed, modeled)

            total_score += nse

        except Exception as e:
            print(f"Error evaluating {rainfall_file}: {e}")
            total_score += float('inf')  # Penalize failed evaluations

    return (total_score,)


def custom_init(calib_params):
    individual = creator.Individual(
        [random.uniform(calib_params[key]["min"], calib_params[key]["max"])
         for key in calib_params.keys()]
    )
    for i in range(len(individual)):
        param_name = list(calib_params.keys())[i]
        individual[i] = max(min(individual[i], calib_params[param_name]['max']),
                            calib_params[param_name]['min'])
    return individual


def custom_mate(ind1, ind2, calib_params, alpha=0.5):
    for i in range(len(ind1)):
        beta = random.uniform(0, 1)
        ind1[i] = ind1[i] * (1.0 - alpha) + ind2[i] * alpha * beta
        ind2[i] = ind2[i] * (1.0 - alpha) + ind1[i] * alpha * (1 - beta)

    # Ensure values are within limits
    for i in range(len(ind1)):
        param_name = list(calib_params.keys())[i]
        ind1[i] = max(min(ind1[i], calib_params[param_name]["max"]),
                      calib_params[param_name]["min"])
        ind2[i] = max(min(ind2[i], calib_params[param_name]["max"]),
                      calib_params[param_name]["min"])

    return ind1, ind2


def custom_mutate(individual, calib_params, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            param_name = list(calib_params.keys())[i]
            individual[i] = random.uniform(calib_params[param_name]['min'],
                                           calib_params[param_name]['max'])

    # Ensure values are within limits
    for i in range(len(individual)):
        param_name = list(calib_params.keys())[i]
        individual[i] = max(min(individual[i], calib_params[param_name]['max']),
                            calib_params[param_name]['min'])

    return individual,


def run_calibration(calibration_rainfall,
                    calibration_observed_data,
                    setup_file,
                    output_setup_file=None,
                    logbook_output_path=None,
                    seed=None):
    """
    Run calibration using genetic algorithm.

    Parameters:
    - calibration_rainfall: list of paths to rainfall files
    - calibration_observed_data: list of paths to observed outflow files
    - setup_file: path to setup file with calibration parameters
    - output_setup_file: path to save calibrated setup file (optional)
    - logbook_output_path: path to save logbook CSV (optional)
    - seed: random seed for reproducibility

    Returns:
    - best_individual: list of best parameter values
    - calibrated_setup: ConfigParser object with calibrated parameters
    - logbook: DEAP logbook with optimization history
    """
    # Validate inputs
    if len(calibration_rainfall) != len(calibration_observed_data):
        raise ValueError("Number of rainfall files must match number of observed data files")

    if len(calibration_rainfall) == 0:
        raise ValueError("At least one calibration event is required")

    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Read setup and parse calibration parameters
    setup = gdt.water_flow_module.read_setup_file(setup_file)
    calib_params = parse_calibration_parameters(setup)
    lower_bound = [calib_params[key]['min'] for key in calib_params.keys()]
    upper_bound = [calib_params[key]['max'] for key in calib_params.keys()]

    # Register DEAP classes
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", custom_init, calib_params=calib_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", custom_mate, calib_params=calib_params, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=5,
                     low=lower_bound, up=upper_bound, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate,
                     calibration_params=calib_params,
                     calibration_rainfall=calibration_rainfall,
                     calibration_observed_data=calibration_observed_data,
                     setup_file=setup_file)

    # Get GA parameters from setup
    npop = int(setup['CALIBRATION'].get('pop', 50))
    ngen = int(setup['CALIBRATION'].get('gen', 100))
    cxpb = float(setup['CALIBRATION'].get('cxpb', 0.5))
    mutpb = float(setup['CALIBRATION'].get('mutpb', 0.2))

    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Initialize population and hall of fame
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max", "hof_ind"
    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)

    print(f"Starting calibration with {len(calibration_rainfall)} events")
    print(f"Population size: {npop}, Generations: {ngen}")
    print(f"Calibrating {len(calib_params)} parameters:")
    for param, bounds in calib_params.items():
        print(f"  {param}: [{bounds['min']:.4f}, {bounds['max']:.4f}]")

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    hof.update(pop)
    logbook.record(gen=0, evals=len(invalid_ind), hof_ind=list(hof[0]), **record)
    print(logbook.stream)

    # Run GA
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = toolbox.select(pop + offspring, npop)
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), hof_ind=list(hof[0]), **record)
        print(logbook.stream)

    # Get best individual
    best_individual = hof[0]
    print(f"\nBest individual found: {best_individual}")
    print(f"Best fitness: {best_individual.fitness.values[0]}")

    # Update setup with best parameters
    calibrated_setup = update_setup_with_values(setup, best_individual)

    # Save calibrated setup file if requested
    if output_setup_file:
        with open(output_setup_file, 'w') as f:
            calibrated_setup.write(f)
        print(f"\nCalibrated setup saved to: {output_setup_file}")

    # Save logbook if requested
    if logbook_output_path:
        os.makedirs(os.path.dirname(logbook_output_path), exist_ok=True)
        pd.DataFrame(logbook).to_csv(logbook_output_path, index=False)
        print(f"Logbook saved to: {logbook_output_path}")

    return best_individual, calibrated_setup, logbook