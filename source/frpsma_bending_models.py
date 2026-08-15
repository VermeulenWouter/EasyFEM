"""
Creates and runs three-point-bending models of hybrid composites with different configurations
(wire positioning, mesh, prestrain, wire volume fraction, cyclic loading vs monotonic loading).

Note this file requires the Educational License of Abaqus, as there are too many nodes for the inp files to
be created with the Student Edition.

Sharthand names:
Oxxyy and Cxxyy
O = Off-centred
C = Centred
xx = wire percentage
yy = prestrain percentage
"""

import numpy as np
from abaqusConstants import THREADS

from utils import *
from utils.abaqus import *
from utils.abaqus.materials import create_NiTiCo_material, create_FRP_material_hashin, create_FRP_material_elastic
from utils.abaqus.numerical_tests import ThreePointBendingTest

##
# Mimic experimental tests to check model fidelity
##
# To avoid accidentally overwriting anything, and verify the script contains no errors, set this to True.
# The script will skip the Abaqus model creation and analysis steps, but will still create the input file.
# Once the script is validated, set it to False to effectively run the analysis.
false_run = True

num_cpus = 1  # Parallelization is not supported in Learning Edition


nitico = create_NiTiCo_material("NiTiCo")
frp = create_FRP_material_hashin("FRP", E_1=32776*ureg.MPa, E_2=32776*ureg.MPa, X_C=291.98 * ureg.MPa, Y_C=291.98 * ureg.MPa, G_mcc=10*ureg.mJ/ureg.mm**2, G_fcc=10*ureg.mJ/ureg.mm**2, bidirectional=True)

model_centred_2_percent = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name="3PB_NiTiCo_Centred_volume2_prestrain4", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
model_centred_2_percent.add_wires(wire_material=nitico, wire_percentage=2*ureg.percent, wire_prestrain=4*ureg.percent, wire_configuration="centred")
model_centred_2_percent.create_job()
model_centred_2_percent.save_cae()
if not false_run:
    model_centred_2_percent.run_job(false_run=false_run)
    model_centred_2_percent.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

##

model_offcentred_2_percent = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name="3PB_NiTiCo_OffCentred_volume2_prestrain4", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
model_offcentred_2_percent.add_wires(wire_material=nitico, wire_percentage=2*ureg.percent, wire_prestrain=4*ureg.percent)
model_offcentred_2_percent.create_job()
model_offcentred_2_percent.save_cae()
if not false_run:
    model_offcentred_2_percent.run_job(false_run=false_run)
    model_offcentred_2_percent.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

##

model_offcentred_4_percent = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name="3PB_NiTiCo_OffCentred_volume4_prestrain4", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
model_offcentred_4_percent.add_wires(wire_material=nitico, wire_percentage=4*ureg.percent, wire_prestrain=4*ureg.percent)
model_offcentred_4_percent.create_job()
model_offcentred_4_percent.save_cae()

if not false_run:
    model_offcentred_4_percent.run_job(false_run=false_run)
    model_offcentred_4_percent.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

clear_model_database()

##
# Check results for prestrain 0% - 10% and wire percentage 1% - 10%
##
tests = []
for prestrain in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for wire_percentage in [1, 2, 3, 4, 5, 6, 7, 8]:
        model_offcentred = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name=f"3PB_NiTiCo_OffCentred_volume{wire_percentage:02d}_prestrain{prestrain:02d}", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
        model_offcentred.add_wires(wire_material=nitico, wire_percentage=wire_percentage*ureg.percent, wire_prestrain=prestrain*ureg.percent)
        model_offcentred.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
        tests.append(model_offcentred)
tests[-1].save_cae(filename="HybridOffcentred_PrestrainPercentageParameterSweep")

for test in tests:
    test.run_job(false_run=false_run)
    test.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

clear_model_database()

# As all results should be quasi-same, don't run per percentage
tests = []
for prestrain in [0, 4, 8]:
    for wire_percentage in [2, 4, 6, 8]:
        model_centred = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name=f"3PB_NiTiCo_Centred_volume{wire_percentage:02d}_prestrain{prestrain:02d}", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
        model_centred.add_wires(wire_material=nitico, wire_percentage=wire_percentage*ureg.percent, wire_prestrain=prestrain*ureg.percent, wire_configuration="centred")
        model_centred.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
        tests.append(model_centred)

tests[-1].save_cae(filename="HybridCentred_PrestrainPercentageParameterSweep")

for test in tests:
    test.run_job(false_run=false_run)
    test.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

clear_model_database()

##
# Cyclic (initial test)
##

model_offcentred_4_percent_cyclic = ThreePointBendingTest(DIR_MODELS / "ThreePointBending", frp_material=frp, name=f"3PB_cyclic_NiTiCo_OffCentred_volume04_prestrain04", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True, loading_amplitude=[(i, -((i+1)%2)*(0.1*ureg.mm+(i//2)*0.5*ureg.mm)) for i in range(39*2)])
model_offcentred_4_percent_cyclic.add_wires(wire_material=nitico, wire_percentage=4*ureg.percent, wire_prestrain=4*ureg.percent)
model_offcentred_4_percent_cyclic.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
model_offcentred_4_percent_cyclic.save_cae()
model_offcentred_4_percent_cyclic.run_job(false_run=false_run)
model_offcentred_4_percent_cyclic.extract_results(DIR_DATA_OUT / "ThreePointBending", false_run=false_run)

clear_model_database()

##
# Illustrate the prestrain
##
tests = []
for prestrain in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    model = ThreePointBendingTest(DIR_MODELS / "ThreePointBending" / "PrestrainTest", frp_material=frp, name=f"3PB_prestrainoutput_NiTiCo_OffCentred_volume02_prestrain{prestrain:02d}", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
    model.add_wires(wire_material=nitico, wire_percentage=2*ureg.percent, wire_prestrain=prestrain*ureg.percent, create_output_request=True)
    model.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
    tests.append(model)
tests[-1].save_cae(filename="HybridOffcentred_PrestrainTest")

for test in tests:
    test.run_job(false_run=false_run)
    test.extract_prestrain_results(DIR_DATA_OUT / "ThreePointBending" / "PrestrainTest", false_run=false_run)
clear_model_database()


##
# Illustrate double counting
##
tests = []
frp_elastic = create_FRP_material_elastic("FRP Elastic", E_1=32776*ureg.MPa, E_2=32776*ureg.MPa)
for wire_percentage in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for prestrain in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        model = ThreePointBendingTest(DIR_MODELS / "ThreePointBending" / "DoubleCountingTest", frp_material=frp_elastic, name=f"3PB_doublecounting_OffCentred_volume{wire_percentage:02d}_prestrain{prestrain:02d}", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
        model.add_wires(wire_material=frp_elastic, wire_percentage=wire_percentage*ureg.percent, wire_prestrain=prestrain*ureg.percent)
        model.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
        tests.append(model)
tests[-1].save_cae(filename="HybridOffcentred_DoubleCountingTest")
for test in tests:
    test.run_job(false_run=false_run)
    test.extract_results(DIR_DATA_OUT / "ThreePointBending" / "DoubleCountingTest", false_run=false_run)

clear_model_database()


##
# Mesh convergence
##
try:
    tests = []
    for fine_mesh in (7.5*ureg.mm) / np.array([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 60]):
        test = ThreePointBendingTest(directory=DIR_MODELS / "ThreePointBending" / "MeshSensitivity", name=f"3PB_NiTiCo_OffCentred_volume04_prestrain04_mesh{int(fine_mesh.m*1000)}", frp_material=frp, coarse_mesh_size=1.25*ureg.mm, fine_mesh_size=fine_mesh, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True)
        test.add_wires(wire_material=nitico, wire_percentage=4*ureg.percent, wire_prestrain=4*ureg.percent)
        test.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
        tests.append(test)

    tests[-1].save_cae(filename="MeshSensitivityStudy")  # Will save all of them
except:
    tprint("Error setting up mesh sensitivity study. Skipping.")

for test in tests:
    try:
        test.run_job(false_run=false_run)
        test.extract_results(DIR_DATA_OUT / "ThreePointBending" / "MeshSensitivity", false_run=false_run)
    except:
        tprint(f"Error running job for mesh sensitivity test with fine mesh size {test.frp_sample.info['Fine mesh size']}. Skipping.")

try:
    clear_model_database()
except:
    tprint("Error clearing model database. Skipping.")


##
# Check cyclic loading
##
tests = []
for prestrain in [0, 2, 3, 4, 6, 8, 10]:
    for wire_percentage in [2, 4, 6, 8]:
        model_offcentred = ThreePointBendingTest(DIR_MODELS / "ThreePointBending" / "Cyclic", frp_material=frp, name=f"3PB_cyclic_NiTiCo_OffCentred_volume{wire_percentage:02d}_prestrain{prestrain:02d}", fine_mesh_size=0.75*ureg.mm, loading_tip_diameter=10*ureg.mm, include_wire_prestrain_step=True, loading_amplitude=[(i, -((i+1)%2)*(0.1*ureg.mm+(i//2)*0.5*ureg.mm)) for i in range(39*2)])
        model_offcentred.add_wires(wire_material=nitico, wire_percentage=wire_percentage*ureg.percent, wire_prestrain=prestrain*ureg.percent)
        model_offcentred.create_job(numCpus=num_cpus, numDomains=num_cpus, multiprocessingMode=THREADS)
        tests.append(model_offcentred)
tests[-1].save_cae(filename="HybridOffcentred_Cyclic")

for test in tests:
    try:
        test.run_job(false_run=false_run)
        test.extract_results(DIR_DATA_OUT / "ThreePointBending" / "Cyclic", false_run=false_run)
    except Exception as e:
        tprint(e)
clear_model_database()

