import permeabledt as pdt
import os

base_dir = os.getcwd()

# load input files
setup_file = os.path.join(base_dir, "input_parameters.ini")
rainfall_file = os.path.join(base_dir, "rainfall.dat")
output_path = os.path.join(base_dir, "test_plot.png")

# read the parameters from the setup file
setup = pdt.water_flow_module.read_setup_file(setup_file)
parameters = pdt.water_flow_module.initialize_parameters(setup)

# # run the simulation
data, mb = pdt.run_simulation(parameters,
                              rainfall_file,
                              rainfall_unit='in',
                              verbose=True,
                              plot_outflow=True,
                              output_path=output_path)
