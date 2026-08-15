"""
Creates and runs a tensile test of NiTiCo wire and plots the resulting stress-strain curve.
"""

import numpy as np

from utils import *
from utils.abaqus import *
from utils.abaqus.materials import create_NiTiCo_material
from utils.abaqus.numerical_tests import WireTensileTest

EXP_SAMPLE_FILES = [DIR_DATA_IN / "2025_PRBE" / f"TEST_tensile_cyclic_NiTiCo_{i}" for i in [5, 6, 7]]

figdestination = FigDestinations.LATEX

false_run = True
# Yield stress and strain are commented out to not depend on the experimental input files used to determine them initially
nitico = create_NiTiCo_material("NiTiCo FINAL")  # yield_stress=[final_str*ureg.MPa for final_str in final_stress], yield_strain=[final_str.m*ureg.dimensionless for final_str in final_strain])
test = WireTensileTest(DIR_MODELS / "NiTiCo_tensile", nitico, name="TensileMonotonic")
test.create_job()
test.save_cae()
test.run_job(false_run=false_run)
test.extract_results(DIR_DATA_OUT / "NiTiCo_tensile", false_run=false_run)

fig, ax = subplots(destination=figdestination)

sample = DataFile(DIR_DATA_OUT / "NiTiCo_tensile" / "TensileMonotonic")
data = sample.load()
sample_true_stain = np.log(1 + data["Engineering Strain"])
sample_true_stress = data["Engineering Stress"] * (1 + data["Engineering Strain"])

ax.plot(sample_true_stain*100, sample_true_stress, "o-", color="C0", markersize=2, label=f"Numerical model")
ax.set_ylabel(f"True stress (MPa)")
ax.set_xlabel(r"True strain (\%)")
fig.savefig(DIR_FIGURE_OUT / "Exp_tensile_NiTiCo_new_vs_old")
