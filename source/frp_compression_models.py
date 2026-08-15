"""
Run tensile tests on the FRP sample (without wires) with different material models. This are the models used
for section 3.4.
Note: tensile models can be easily created in the same way.
"""
from utils import *

from utils.abaqus import *
from utils.abaqus.numerical_tests import CompressionTest
from utils.abaqus.materials import create_FRP_material_hashin

from abaqusConstants import THREADS

num_cpus = 1  # Parallelization is not supported in Learning Edition

mean_E = 38640*ureg.MPa
std_E = 833.2*ureg.MPa
mean_E_end = 31860*ureg.MPa
mean_compressive_strength = 389.9*ureg.MPa
std_compressive_strength = 32.64*ureg.MPa
G_fcc_init = 32*ureg.mJ/ureg.mm**2

##
# Abaqus model
##

# To avoid accidentally overwriting anything, and verify the script contains no errors, set this to True.
# The script will skip the Abaqus model creation and analysis steps, but will still create the input file.
# Once the script is validated, set it to False to effectively run the analysis.
false_run = True

# Initial visualisation
frp_material = create_FRP_material_hashin("FRP Material compr v1", E_1=mean_E, E_2=mean_E, X_C=mean_compressive_strength, Y_C=mean_compressive_strength, G_fcc=G_fcc_init, G_mcc=G_fcc_init)
test = CompressionTest(directory=DIR_MODELS / "Compression", name=f"Com_FRPOnly_InitialParameters", frp_material=frp_material, coarse_mesh_size=0.75*ureg.mm)
test.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
test.save_cae()
test.run_job(false_run=false_run)
test.extract_results(DIR_DATA_OUT / "Compression", false_run=false_run)

clear_model_database()

# Study to determine G_xcc
false_run = True
G_xcc_range = [5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25]*ureg.mJ/ureg.mm**2
tests = []
for G_xcc in G_xcc_range:
    frp_material = create_FRP_material_hashin("FRP Material compr Gxcc adapted", E_1=mean_E, E_2=mean_E, X_C=mean_compressive_strength, Y_C=mean_compressive_strength, G_fcc=G_xcc, G_mcc=G_xcc)
    test = CompressionTest(directory=DIR_MODELS / "Compression" / "ParamSensitivity_Gxcc", name=f"Com_FRPOnly_paramGxcc_{int(G_xcc.m*100)}", frp_material=frp_material, coarse_mesh_size=0.75*ureg.mm)
    test.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
    tests.append(test)

tests[-1].save_cae(filename="ParamSensitivity_Gxcc")

for test in tests:
    test.run_job(false_run=false_run)
    test.extract_results(DIR_DATA_OUT / "Compression" / "ParamSensitivity_Gxcc", false_run=false_run)

clear_model_database()

# G_mcc = 9*ureg.mJ/ureg.mm**2 seems the best value for this mesh size

##
# Checking sensitivity to the parameters: using mean values, and mean +/- std for E and compressive strength
##
false_run = True
param_sets = {
    "stdminus2": {"E_1": mean_E-2*std_E, "E_2": mean_E-2*std_E, "X_C": mean_compressive_strength-2*std_compressive_strength, "Y_C": mean_compressive_strength-2*std_compressive_strength},
    "stdminus1": {"E_1": mean_E-std_E, "E_2": mean_E-std_E, "X_C": mean_compressive_strength-std_compressive_strength, "Y_C": mean_compressive_strength-std_compressive_strength},
    "mean": {"E_1": mean_E, "E_2": mean_E, "X_C": mean_compressive_strength, "Y_C": mean_compressive_strength},
    "stdplus1": {"E_1": mean_E+std_E, "E_2": mean_E+std_E, "X_C": mean_compressive_strength+std_compressive_strength, "Y_C": mean_compressive_strength+std_compressive_strength},
    "stdplus2": {"E_1": mean_E+2*std_E, "E_2": mean_E+2*std_E, "X_C": mean_compressive_strength+2*std_compressive_strength, "Y_C": mean_compressive_strength+2*std_compressive_strength},
    "Eend": {"E_1": mean_E_end, "E_2": mean_E_end, "X_C": mean_compressive_strength, "Y_C": mean_compressive_strength}
}

param_set_name_to_label = {
    "stdminus2": "Mean - 2*std",
    "stdminus1": "Mean - std",
    "mean": "Mean",
    "stdplus1": "Mean + std",
    "stdplus2": "Mean + 2*std",
    "Eend": "Mean end modulus"
}

tests = []
for name, param_set in param_sets.items():
    G_xcc = 9*ureg.mJ/ureg.mm**2
    frp_material = create_FRP_material_hashin(f"FRP Material Compressive {param_set_name_to_label[name]}", **param_set, G_fcc=G_xcc, G_mcc=G_xcc)
    test = CompressionTest(directory=DIR_MODELS / "Compression" / "ParamSensitivity_E_XY", name=f"Comp_FRPOnly_paramEXY_{name}", frp_material=frp_material, coarse_mesh_size=0.75*ureg.mm)
    test.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
    tests.append(test)

tests[-1].save_cae()

for test in tests:
    test.run_job(false_run=false_run)
    test.extract_results(DIR_DATA_OUT / "Compression" / "ParamSensitivity_E_XY", false_run=false_run)

clear_model_database()
