"""
Creates the plots for the majority of the figures in Chapter 4

Sharthand names:
Oxxyy and Cxxyy
O = Off-centred
C = Centred
xx = wire percentage
yy = prestrain percentage
"""

import numpy as np

from utils import *
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors


DRAFT = False

figdestination = FigDestinations.LATEX

# Note DIR_DATA_IN = "data/input"
EXP_FILES = {
    "O0000": [DIR_DATA_IN / "2023_Yvan" / f"TEST_bending_{identifier}" for identifier in ["1B", "1C", "1D", "2A", "2B"]],
    "O0204": [DIR_DATA_IN / "2023_Yvan" / f"TEST_bending_{identifier}" for identifier in ["O21A"]],
    "O0404": [DIR_DATA_IN / "2023_Yvan" / f"TEST_bending_{identifier}" for identifier in ["O41A", "O41B", "O42A", "O42B"]],
    "C0204": [DIR_DATA_IN / "2023_Yvan" / f"TEST_bending_{identifier}" for identifier in ["C22A", "C22B", "C23A", "C23B"]]
}

MODEL_FILES = {"O0000": [DIR_DATA_OUT / "ThreePointBending" / "3PB_FRPOnly_FinalMaterialProperties"],
    "O0204": [DIR_DATA_OUT / "ThreePointBending" / "3PB_NiTiCo_Offcentred_volume02_prestrain04"],
    "O0404": [DIR_DATA_OUT / "ThreePointBending" / "3PB_NiTiCo_Offcentred_volume04_prestrain04"],
    "C0204": [DIR_DATA_OUT / "ThreePointBending" / "3PB_NiTiCo_Centred_volume02_prestrain04"],
}

MODEL_FILES_CYCLIC = {
    "O0204": [DIR_DATA_OUT / "ThreePointBending" / "Cyclic" / "3PB_cyclic_NiTiCo_Offcentred_volume02_prestrain04"],
    "O0404": [DIR_DATA_OUT / "ThreePointBending" / "Cyclic" / "3PB_cyclic_NiTiCo_Offcentred_volume04_prestrain04"],
}

cut_envelope_retrace = {
    "1B": 1, "O41A": 1, "O41B": 17, "O42A": 1, "O42B": 1
}

label_to_exp_label = {
    "Without wires": "O0000",
    "Centred 2% Vol Prestrain 4%": "C0204",
    "OffCentred 2% Vol Prestrain 4%": "O0204",
    "OffCentred 4% Vol Prestrain 4%": "O0404"
}


def force_to_stress(force, sample):
    return 3/2 * force * sample.L0 / (sample.width * sample.thickness**2)


def disp_to_strain(disp, sample):
    return 6 * disp * sample.thickness / sample.L0**2


def extract_envelope(data, retrace=0):
    """Extract the max envelope of the cyclic bending test data, appended with the last incomplete cycle.
    Keep the full data in the output dict, but add a suffix "_original" to the original arrays."""
    cycles = data["Cycle"]
    forces = data["Force"]

    # Convert to magnitudes for comparisons
    cycle_vals = cycles.m
    force_vals = forces.m

    max_cycle = int(cycle_vals.max())

    selected_indices = []

    for c in range(1, max_cycle + 1):
        # indices belonging to this cycle
        mask = cycle_vals == c
        idxs = np.where(mask)[0]

        if idxs.size == 0:
            continue  # skip missing cycles

        # find index of max force *within this cycle*
        local_forces = force_vals[idxs]
        local_max_pos = np.argmax(local_forces)
        global_index = idxs[local_max_pos]

        selected_indices.append(global_index)

    # Build output dict with the same structure
    displ_orig = data["Displacement"]
    force_orig = data["Force"]
    if retrace == 0:
        data.update({key: arr[selected_indices] - arr[0] for key, arr in data.items()})
    else:
        data.update({key: arr[selected_indices[:-retrace]] - arr[0] for key, arr in data.items()})

    data["Displacement_original"] = displ_orig - displ_orig[0]
    data["Force_original"] = force_orig - force_orig[0]

    return data


def extract_from_bending_curve(file, model=True, keep_cyclic=False, print_output=False):
    """
    Flexural modulus, ultimate stress, failure deflection (back-calculated from max strain), ductility index

    :param file:
    :param envelope_only:
    :return:
    """
    sample = DataFile(file)
    data = sample.load()

    if not model:
        envelope = extract_envelope(data, retrace=cut_envelope_retrace.get(sample.identifier, 0))
        displ = envelope["Displacement"]
        force = envelope["Force"]
    else:
        displ = data["Displacement"]
        force = -data["Force"]

    time = data["Time"] - data["Time"][0]

    stress = force_to_stress(force, sample)
    strain = disp_to_strain(displ, sample)

    flexural_modulus_idx1 = np.where(displ > 1*ureg.mm)[0][0]
    flexural_modulus_idx2 = np.where(displ > 2*ureg.mm)[0][0]
    flexural_modulus = (force[flexural_modulus_idx2] - force[flexural_modulus_idx1]) / (displ[flexural_modulus_idx2] - displ[flexural_modulus_idx1]) * sample.L0**3 / (48*(sample.width * sample.thickness**3/12))

    ultimate_stress = stress.max()
    failure_idx = np.argmax(stress)
    failure_deflection = displ[failure_idx]
    failure_strain = strain[failure_idx]

    total_energy = np.trapezoid(stress[:failure_idx+1].m, strain[:failure_idx+1].m) * stress.units * strain.units  # Energy per unit volume
    strain_yield = failure_strain - np.sqrt(failure_strain**2 - 2*total_energy/flexural_modulus)
    stress_yield = flexural_modulus * strain_yield

    ductility_index = failure_strain / strain_yield

    tprint(f"{"Model" if model else "Sample"} {sample.identifier}:\n\tFlexural modulus = {flexural_modulus:~P}\n\tUltimate stress = {ultimate_stress:~P}\n\tFailure deflection = {failure_deflection:~}\n\tFailure strain = {failure_strain:~}\n\tStrain at yield = {strain_yield:~}\n\tStress at yield = {stress_yield:~P}\n\tDuctility index = {ductility_index:~}")

    return {
        "sample": sample,
        "data": data,
        "displ": displ,
        "force": force,
        "displ_orig": envelope["Displacement_original"] if keep_cyclic else displ,
        "force_orig": envelope["Force_original"] if keep_cyclic else force,
        "flexural_modulus": flexural_modulus,
        "ultimate_stress": ultimate_stress,
        "failure_deflection": failure_deflection,
        "failure_strain": failure_strain,
        "strain_yield": strain_yield,
        "stress_yield": stress_yield,
        "ductility_index": ductility_index,
        "stress": stress,
        "strain": strain
    }


exp_samples = {label: [extract_from_bending_curve(file, model=False, keep_cyclic=True, print_output=True) for file in files] for label, files in EXP_FILES.items()}
model_samples = {label: [extract_from_bending_curve(file, print_output=True) for file in files] for label, files in MODEL_FILES.items()}
cyclic_model_samples = {label: [extract_from_bending_curve(file, print_output=True) for file in files] for label, files in MODEL_FILES_CYCLIC.items()}


# Means from experimental
for config, samples in exp_samples.items():
    flexural_modulus = np.mean(quantity_array([sample["flexural_modulus"] for sample in samples]))
    ultimate_stress = np.mean(quantity_array([sample["ultimate_stress"] for sample in samples]))
    failure_deflection = np.mean(quantity_array([sample["failure_deflection"] for sample in samples]))
    failure_strain = np.mean(quantity_array([sample["failure_strain"] for sample in samples]))
    strain_yield = np.mean(quantity_array([sample["strain_yield"] for sample in samples]))
    stress_yield = np.mean(quantity_array([sample["stress_yield"] for sample in samples]))
    ductility_index = np.mean(quantity_array([sample["ductility_index"] for sample in samples]))

    flexural_modulus_std = np.std(quantity_array([sample["flexural_modulus"] for sample in samples]))
    ultimate_stress_std = np.std(quantity_array([sample["ultimate_stress"] for sample in samples]))
    failure_strain_std = np.std(quantity_array([sample["failure_strain"] for sample in samples]))
    ductility_index_std = np.std(quantity_array([sample["ductility_index"] for sample in samples]))

    tprint(f"Mean experimental {config}:\n\tFlexural modulus = {flexural_modulus:~P}\n\tUltimate stress = {ultimate_stress:~P}\n\tFailure deflection = {failure_deflection:~}\n\tFailure strain = {failure_strain:~}\n\tStrain at yield = {strain_yield:~}\n\tStress at yield = {stress_yield:~P}\n\tDuctility index = {ductility_index:~}")
    tprint(f"Std experimental {config}:\n\tFlexural modulus = {flexural_modulus_std:~P}\n\tUltimate stress = {ultimate_stress_std:~P}\n\tFailure strain = {failure_strain_std:~}\n\tDuctility index = {ductility_index_std:~}")

# Cyclic vs monotonic comparison
if not DRAFT:
    model = "O0204"
    fig, ax = subplots(destination=figdestination)
    if model in exp_samples:
        ax.plot(exp_samples[model][0]["displ_orig"].m, exp_samples[model][0]["force_orig"].m+3.5, label="Experimental", color="lightgray", linestyle="-", linewidth=1)
    ax.plot(cyclic_model_samples[model][0]["displ"].m, cyclic_model_samples[model][0]["force"].m, label="Cyclic numerical model", color="C1", linestyle="--")
    ax.plot(model_samples[model][0]["displ"].m, model_samples[model][0]["force"].m, "o-", label="Monotonic numerical model", color="C0", markersize=2)
    ax.set_xlabel("Deflection (mm)")
    ax.set_ylabel("Force (N)")
    fig.savefig(DIR_FIGURE_OUT / f"bending_{model}_cyclic_vs_monotonic", legend_loc="upper right")


# Mesh convergence
model_files_mesh_study = {}
el_in_width = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])  # , 28, 30, 60])

for fine_mesh in (7.5*ureg.mm) / el_in_width:
    model_files_mesh_study[fine_mesh] = [DIR_DATA_OUT / "ThreePointBending" / "HybridOffCentred" / "MeshSensitivity" / f"N_mesh{int(fine_mesh.m*1000)}"]

model_samples_mesh_study = {mesh: [extract_from_bending_curve(file) for file in files] for mesh, files in model_files_mesh_study.items()}


flex_moduli = quantity_array([samples[0]["flexural_modulus"] for samples in model_samples_mesh_study.values()])
ult_stress = quantity_array([samples[0]["ultimate_stress"] for samples in model_samples_mesh_study.values()])
ult_strain = quantity_array([samples[0]["failure_strain"] for samples in model_samples_mesh_study.values()])
ductility = quantity_array([samples[0]["ductility_index"] for samples in model_samples_mesh_study.values()])
increment_damage_initiation_time = quantity_array([samples[0]["sample"].metadata["Model type"]["Damage initiation"]["Time"] for samples in model_samples_mesh_study.values()])
# Time and displacement are linearly dependent, so we can use the time to compare the damage initiation time between meshes

flex_moduli = -(flex_moduli[2] - flex_moduli) / flex_moduli[2]
ult_stress = -(ult_stress[2] - ult_stress) / ult_stress[2]
ult_strain = -(ult_strain[2] - ult_strain) / ult_strain[2]
ductility = -(ductility[2] - ductility) / ductility[2]
damage_init = -(increment_damage_initiation_time[2] - increment_damage_initiation_time) / increment_damage_initiation_time[2]

if not DRAFT:
    fig, ax = subplots(destination=figdestination)
    ax.plot(el_in_width, flex_moduli.m*100, label="Flexural modulus $E^{b}$", marker="o")
    ax.plot(el_in_width, damage_init.m*100, label="Eq. flexural strain at damage initiation", marker="o")
    ax.plot(el_in_width, ult_stress.m*100, label=r"Ultimate eq. flexural stress $\sigma^{bu}$", marker="o")
    ax.plot(el_in_width, ult_strain.m*100, label=r"Ultimate eq. flexural strain $\varepsilon^{bu}$", marker="o")
    ax.plot(el_in_width, ductility.m*100, label=r"Ductility index $\mu$", marker="o")

    ax.set_xticks(el_in_width)

    ax.set_xlabel("Number of elements in width (-)")
    ax.set_ylabel("Relative difference with respect to\n" + r"10 elements in width (\%)")
    fig.savefig(DIR_FIGURE_OUT / "hybrid_mesh_convergence", legend_loc="best")

# C0204 and O0204 and O0404 comparison
for model in ["C0204", "O0204", "O0404"]:
    if not DRAFT:
        fig, ax = subplots(destination=figdestination, figsize=FigSizes.HALF_WIDTH)

        for sample in exp_samples[model]:
            ax.plot(sample["strain"].m*100, sample["stress"].m, label=f"Experiments (n={len(exp_samples[model])})" if sample["sample"].identifier == exp_samples[model][0]["sample"].identifier else None, color="lightgray", linestyle="-")  #, linewidth=2)
        ax.plot(model_samples[model][0]["strain"].m*100, model_samples[model][0]["stress"].m, "-o", label="Numerical model", color="C0", markersize=2)

        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+110)
        ax.set_xlabel(r"Equivalent flexural strain $\varepsilon^{b}$ (\%)")
        ax.set_ylabel(r"Equivalent flexural stress $\sigma^{b}$ (MPa)")
        fig.savefig(DIR_FIGURE_OUT / f"bending_{model}_exp_vs_model", legend_loc="upper left")

# HEATMAPS
ALL_MODEL_FILES = {f"O{wire_percentage:02d}{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_NiTiCo_Offcentred_volume{wire_percentage:02d}_prestrain{prestrain:02d}" for wire_percentage in [1, 2, 3, 4, 5, 6, 7, 8] for prestrain in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
ALL_MODEL_FILES.update({f"C{wire_percentage:02d}{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_NiTiCo_Centred_volume{wire_percentage:02d}_prestrain{prestrain:02d}" for wire_percentage in [1, 2, 3, 4, 5, 6, 7, 8] for prestrain in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

ALL_MODEL_FILES = {f"AISI_O{wire_percentage:02d}{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_AISI_Offcentred_volume{wire_percentage:02d}_prestrain{prestrain:02d}" for wire_percentage in [1, 2, 3, 4, 5, 6, 7, 8] for prestrain in [0, 1, 2, 3, 4]}
ALL_MODEL_FILES.update({f"AISI_C{wire_percentage:02d}{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_AISI_Centred_volume{wire_percentage:02d}_prestrain{prestrain:02d}" for wire_percentage in [2, 4, 6, 8] for prestrain in [0, 4]})

# To make it easier to create the heatmaps, also fill these (but they are greyed out in the heatmaps)
ALL_MODEL_FILES.update({f"O00{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_FRPOnly_FinalMaterialProps" for prestrain in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
ALL_MODEL_FILES.update({f"C00{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_FRPOnly_FinalMaterialProps" for prestrain in [0, 4, 8]})
ALL_MODEL_FILES.update({f"AISI_O00{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_FRPOnly_FinalMaterialProps" for prestrain in [0, 1, 2, 3, 4]})
ALL_MODEL_FILES.update({f"AISI_C00{prestrain:02d}": DIR_DATA_OUT / "ThreePointBending" / f"3PB_FRPOnly_FinalMaterialProps" for prestrain in [0, 4]})


ALL_MODEL_DATA = {key: extract_from_bending_curve(file, model=True) for key, file in ALL_MODEL_FILES.items()}


MODELS_WITH_WRONG_SIGMA_MAX = [
    "AISI_O0400", "AISI_O0500", "AISI_O0600", "AISI_O0700", "AISI_O0800",
    *[f"AISI_O{wire_percentage:02d}{prestrain:02d}" for wire_percentage in [3, 4, 5, 6, 7, 8] for prestrain in [1, 2, 3, 4]],
    "O0703", "O0803", "O0704", "O0804", "O0605", "O0705", "O0805", "O0706", "O0806", "O0707", "O0807", "O0808", "O0809"


]


def heatmap(var_name, title, cmap_var_name, filename, conf: str = "O", wire_percentages: list = [0, 1, 2, 3, 4, 5, 6, 7, 8], prestrains: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], percent_output=True, ref_value_legend=None, end_of_curve_output=True):
    """
    Generate a heatmap for a given variable from the model data.
    """
    fig, ax = subplots(destination=figdestination, figsize=FigSizes.HALF_WIDTH)

    # Reference value (no wires)
    ref_value_full = ALL_MODEL_DATA[f"{conf}0000"][var_name]
    reference_value = ALL_MODEL_DATA[f"{conf}0000"][var_name].m

    if percent_output:
        flexural_moduli = np.array([
            [ALL_MODEL_DATA[f"{conf}{wire_percentage:02d}{prestrain:02d}"][var_name].m
             for prestrain in prestrains]
            for wire_percentage in wire_percentages
        ])
        flexural_moduli = (flexural_moduli-reference_value) / reference_value * 100
        tprint(np.max(flexural_moduli), np.min(flexural_moduli))
    else:
        flexural_moduli = np.array([
            [ALL_MODEL_DATA[f"{conf}{wire_percentage:02d}{prestrain:02d}"][var_name].m
             for prestrain in prestrains]
            for wire_percentage in wire_percentages
        ])

    if not percent_output:
        vmax_deviation = np.max(np.abs(flexural_moduli - reference_value))
        norm = mcolors.TwoSlopeNorm(vmin=reference_value - vmax_deviation,
                                     vcenter=reference_value,
                                     vmax=reference_value + vmax_deviation)
    else:
        vmax_deviation = np.max(np.abs(flexural_moduli))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax_deviation,
                                     vcenter=0,
                                     vmax=vmax_deviation)

    im = ax.imshow(
        flexural_moduli,
        cmap="RdYlGn",
        norm=norm,
        aspect="auto",
        origin="lower",
        extent=[
            min(prestrains) - 0.5,
            max(prestrains) + 0.5,
            min(wire_percentages) - 0.5,
            max(wire_percentages) + 0.5,
        ]
    )

    if end_of_curve_output:
        for prestrain in prestrains:
            for wire_percentage in wire_percentages:
                if f"{conf}{wire_percentage:02d}{prestrain:02d}" in MODELS_WITH_WRONG_SIGMA_MAX:
                     # Add a white cross over the cell
                    ax.plot(
                        [prestrain - 0.5, prestrain + 0.5],
                        [wire_percentage - 0.5, wire_percentage + 0.5],
                        "-",
                        color="white",
                        linewidth=2,
                        zorder=3,
                    )
                    ax.plot(
                        [prestrain - 0.5, prestrain + 0.5],
                        [wire_percentage + 0.5, wire_percentage - 0.5],
                        "-",
                        color="white",
                        linewidth=2,
                        zorder=3,
                    )

    for wire_percentage in wire_percentages:
        if wire_percentage != 0:
            continue

        for prestrain in prestrains:
            if prestrain == 0:
                continue

            # Grey out cells where prestrain is not possible for 0% wire
            ax.add_patch(
                plt.Rectangle(
                    (prestrain - 0.5, wire_percentage - 0.5),
                    1,
                    1,
                    facecolor="lightgray",
                    edgecolor="none",
                    zorder=2,
                )
            )

    ax.grid(False)  # kill the default gray grid
    ax.set_xticks(np.array(prestrains), minor=False)
    ax.set_yticks(np.array(wire_percentages), minor=False)
    ax.set_xticks(np.array(prestrains) - 0.5, minor=True)
    ax.set_yticks(np.array(wire_percentages) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel(r"Wire prestrain (\%)")
    ax.set_ylabel(r"Wire volume fraction (\%)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    if percent_output:
        cbar.set_label(f"{cmap_var_name}\n" + r"(\% deviation from reference: " + f"{ref_value_legend})")
    else:
        cbar.set_label(f"{cmap_var_name} ({ALL_MODEL_DATA['O0000'][var_name].u:~})")

    fig.savefig(DIR_FIGURE_OUT / filename, no_grid=True, no_legend=True)


if not DRAFT:
    heatmap("flexural_modulus", None, r"Flexural modulus $E^{b}$", "NiTiCO_O_FlexuralModulus", ref_value_legend="32.7 GPa", end_of_curve_output=False)
    heatmap("ultimate_stress", None, r"Ultimate equivalent flexural stress $\sigma_{eq}^{bu}$", "NiTiCO_O_UltimateStress", ref_value_legend="496 MPa")
    heatmap("ductility_index", None, r"Ductility index $\mu$", "NiTiCO_O_DuctilityIndex", ref_value_legend=r"1.09")
    heatmap("failure_strain", None, r"Equivalent flexural strain at $\sigma_{eq}^{bu}$", "NiTiCO_O_FailureStrain", ref_value_legend=r"1.65 \%")

    heatmap("flexural_modulus", None, "Flexural modulus $E^{b}$", "NiTiCO_C_FlexuralModulus", conf = "C", prestrains=[0, 4, 8], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend="32.7 GPa", end_of_curve_output=False)
    heatmap("ultimate_stress", None, r"Ultimate equivalent flexural stress $\sigma_{eq}^{bu}$", "NiTiCO_C_UltimateStress", conf = "C", prestrains=[0, 4, 8], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend="496 MPa")
    heatmap("ductility_index", None, r"Ductility index $\mu$", "NiTiCO_C_DuctilityIndex_Centred", conf = "C", prestrains=[0, 4, 8], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend=r"1.09")
    heatmap("failure_strain", None, r"Equivalent flexural strain at $\sigma_{eq}^{bu}$", "NiTiCO_C_FailureStrain", conf = "C", prestrains=[0, 4, 8], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend=r"1.65 \%")

    heatmap("flexural_modulus", None, r"Flexural modulus $E^{b}$", "AISI_O_FlexuralModulus", conf = "AISI_O", prestrains=[0, 1, 2, 3, 4], ref_value_legend="32.7 GPa", end_of_curve_output=False)
    heatmap("ultimate_stress", None, r"Ultimate equivalent flexural stress $\sigma_{eq}^{bu}$", "AISI_O_UltimateStress", conf = "AISI_O", prestrains=[0, 1, 2, 3, 4], wire_percentages=[0, 1, 2 ,3], ref_value_legend="496 MPa")
    heatmap("ductility_index", None, r"Ductility index $\mu$", "AISI_O_DuctilityIndex", conf = "AISI_O", prestrains=[0, 1, 2, 3, 4], wire_percentages=[0, 1, 2 ,3], ref_value_legend=r"1.09")
    heatmap("failure_strain", None, r"Equivalent flexural strain at $\sigma_{eq}^{bu}$", "AISI_O_FailureStrain", conf = "AISI_O", prestrains=[0, 1, 2, 3, 4], ref_value_legend=r"1.65 \%")

    heatmap("flexural_modulus", None, r"Flexural modulus $E^{b}$", "AISI_C_FlexuralModulus", conf = "AISI_C", prestrains=[0, 4], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend="32.7 GPa", end_of_curve_output=False)
    heatmap("ultimate_stress", None, r"Ultimate equivalent flexural stress $\sigma_{eq}^{bu}$", "AISI_C_UltimateStress", conf = "AISI_C", prestrains=[0, 4], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend="496 MPa")
    heatmap("ductility_index", None, r"Ductility index $\mu$", "AISI_C_DuctilityIndex", conf = "AISI_C", prestrains=[0, 4], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend=r"1.09")
    heatmap("failure_strain", None, r"Equivalent flexural strain at $\sigma_{eq}^{bu}$", "AISI_C_FailureStrain", conf = "AISI_C", prestrains=[0, 4], wire_percentages=[0, 2, 4, 6, 8], ref_value_legend=r"1.65 \%")
