"""

Three basic extractor functions are defined, and they can be easily combined to get to more complex ones.


@author: Wouter Vermeulen
@date: 2026-04-24
"""

from odbAccess import openOdb
import numpy as np
from numpy.typing import NDArray

from utils import *
from utils.abaqus import *
from utils.data import Data


def filter_regions_for_variable(output_variable: str, region_keys: list[str], coords: NDArray | None = None) -> list[str] | tuple[list[str], NDArray]:
    """
    Determine the geometry type (e.g. node, element) that contains the specified output variable for the given region keys,
    then filter the region keys to only include those that correspond to that geometry type.

    If coords is provided, the coords array gets filtered the same way as the region keys, such that the order of the region keys and the coords array still correspond to each other.

    :param output_variable:
    :param region_keys:
    :return:
    """
    if output_variable in ["U1", "U2", "U3", "RF1", "RF2", "RF3"]:
        geom_type = "Node"
    elif output_variable in ["TEMP", "S11", "S22", "S33", "S12", "S13", "S23", "E11", "E22", "E33", "E12", "E13", "E23", "THE11", "THE22", "THE33", "LE11", "LE22", "LE33"]:
        geom_type = "Element"
        if coords is not None:
            raise ValueError(f"Output variable {output_variable} corresponds to element geometry, but coords array is provided. Cannot filter coords array for element geometry.")
    else:
        # If this raises an error, simply add the output variable to the correct list above (depending on if it
        # corresponds to the node or element geometry).
        raise ValueError(f"Unsupported output variable: {output_variable}. Cannot determine geometry type.")

    mask = np.array([key.startswith(geom_type) for key in region_keys])
    if coords is not None:
        coords = coords[mask]

    filtered_region_keys = [key for key in region_keys if key.startswith(geom_type)]
    if len(filtered_region_keys) == 0:
        raise ValueError(f"No region keys correspond to the geometry type {geom_type} for output variable {output_variable}.")

    if coords is not None:
        return filtered_region_keys, coords

    return filtered_region_keys


def extract_sum(history_regions, output_variables: AbaqusOutputVariable | list[AbaqusOutputVariable], region_keys: list[str], **kwargs):
    """
    Extract a variable from the output history, for each node in region_keys, and sum their values (e.g. for reaction
    force)

    :param history_regions:
    :param region_keys: list of region keys corresponding to the nodes in the node set, in the same order as the coords
    array (e.g. ["Node instance.1", "Node instance.2", ...])
    :param output_variable: the variable to extract and sum across the nodes (e.g. "RF2" for reaction force in the 2
    direction)

    :param kwargs: define the metadata for the output, with keys:
        - output_variable_name: the name to use for the output variable in the metadata
        (default: same as output_variable)
        - output_variable_unit: the unit to use for the output variable in the metadata (default: dimensionless)
        - output_variable_description: the description to use for the output variable in the metadata
        (default: "Summed {output_variable} value across all nodes in the node set.")
    :return:
    """
    if isinstance(output_variables, AbaqusOutputVariable):
        output_variables = [output_variables]

    result = Data()

    var_t_reference = None
    for output_variable in output_variables:
        var_t_sum = None
        var_v_sum = None

        for region in filter_regions_for_variable(output_variable.abaqus_name, region_keys):
            if output_variable.abaqus_name not in history_regions[region].historyOutputs:
                raise RuntimeError(f"[ERROR] {output_variable.abaqus_name} not defined for node {region}.")

            var_raw = history_regions[region].historyOutputs[output_variable.abaqus_name].data

            var_t = []  # Time points
            var_v = []  # Variable values
            for t, v in var_raw:
                if not (isinstance(v, str) and v.lower() == "novalue"):
                    var_t.append(t)
                    var_v.append(v)

            if len(var_t) == 0:
                raise RuntimeError(f"No valid data points found for {output_variable.abaqus_name} in node {region}.")

            if var_t_sum is None:
                var_t_sum = np.array(var_t)
            elif not np.allclose(var_t_sum, np.array(var_t)):
                raise RuntimeError(f"Time points for {output_variable.abaqus_name} are different between nodes. Cannot sum values.")

            if var_v_sum is None:
                var_v_sum = np.array(var_v)
            else:
                var_v_sum += np.array(var_v)

        if "Time" not in result.columns:
            result.add_column(var_t_sum, "Time", unit=Q_.pref_dim("s"), description="Time points at which the variables were recorded.")
            var_t_reference = var_t_sum
        elif not (np.allclose(var_t_reference, var_t_sum)):
            tprint(f"\tTime points for {output_variable.abaqus_name} are different from reference time points. Interpolating to first variable's time points.", color="yellow")
            var_v_sum = np.interp(var_t_reference, var_t_sum, var_v_sum)

        result.add_column(var_v_sum, **output_variable.to_dict())

    return result


def extract_identical(history_regions, output_variables: AbaqusOutputVariable | list[AbaqusOutputVariable], region_keys: list[str], **kwargs):
    if isinstance(output_variables, AbaqusOutputVariable):
        output_variables = [output_variables]

    result = Data()

    var_t_reference = None
    for output_variable in output_variables:
        var_t_id = None
        var_v_id = None  # id meaning identical
        i = 0
        for region in filter_regions_for_variable(output_variable.abaqus_name, region_keys):
            if output_variable.abaqus_name not in history_regions[region].historyOutputs:
                raise RuntimeError(f"[ERROR] {output_variable.abaqus_name} not defined for node {region}.")

            var_raw = history_regions[region].historyOutputs[output_variable.abaqus_name].data

            var_t = []  # Time points
            var_v = []  # Variable values
            for t, v in var_raw:
                if not (isinstance(v, str) and v.lower() == "novalue"):
                    var_t.append(t)
                    var_v.append(v)

            if len(var_t) == 0:
                raise RuntimeError(f"No valid data points found for {output_variable.abaqus_name} in node {region}.")

            if var_t_id is None:
                var_t_id = np.array(var_t)
            elif not np.allclose(var_t_id, np.array(var_t)):
                raise RuntimeError(f"Time points for {output_variable.abaqus_name} are different between nodes. Cannot sum values.")

            atol = kwargs.get("atol", kwargs.get(f"{output_variable.abaqus_name}_atol", 1e-8))  # Allow variable-specific atol or general atol
            rtol = kwargs.get("rtol", 1e-5)
            if var_v_id is None:
                var_v_id = np.array(var_v)
            elif not np.allclose(var_v_id, np.array(var_v), atol=atol, rtol=rtol):
                if kwargs.get("strict", True):
                    raise RuntimeError(f"Variable values for {output_variable.abaqus_name} are different between nodes. Cannot extract identical values.")
                else:
                    var_v_id = (var_v_id * i + np.array(var_v)) / (i+1)  # Update the average value
            i += 1

        if "Time" not in result.columns:
            result.add_column(var_t_id, "Time", unit=Q_.pref_dim("s"), description="Time points at which the variables were recorded.")
            var_t_reference = var_t_id
        elif not(np.allclose(var_t_reference, var_t_id)):
            tprint(f"\tTime points for {output_variable.abaqus_name} are different from reference time points. Interpolating to first variable's time points.", color="yellow")
            var_v_id = np.interp(var_t_reference, var_t_id, var_v_id)

        result.add_column(var_v_id, **output_variable.to_dict())

    return result


def extract_center_node(history_regions, output_variables: AbaqusOutputVariable | list[AbaqusOutputVariable],  region_keys: list[str], coords: NDArray, model_symetry: str | None = "double", **kwargs):
    if model_symetry not in ["double", "single", None]:
        raise ValueError("Invalid value for model_symmetry. Expected 'double', 'single' or None.")

    if isinstance(output_variables, AbaqusOutputVariable):
        output_variables = [output_variables]

    result = Data()

    var_t_reference = None
    for output_variable in output_variables:

        region_keys_filtered, coords_filtered = filter_regions_for_variable(output_variable.abaqus_name, region_keys, coords=coords)

        if model_symetry == "double":
            ideal_point = coords_filtered.max(axis=0)  # Use furthest away corner as ideal point (given this is the centre point when the symetry is added back in)
        elif model_symetry == "single":
            raise NotImplementedError()
        else:
            ideal_point = coords_filtered.mean(axis=0)  # Take centroid as ideal point, which could cause the center node to not be perfectly in the center

        distances = np.linalg.norm(coords_filtered - ideal_point, axis=1)
        center_idx = np.argmin(distances)
        center_region_key = region_keys_filtered[center_idx]

        if output_variable.abaqus_name not in history_regions[center_region_key].historyOutputs:
            raise RuntimeError(f"[ERROR] {output_variable.abaqus_name} not defined for node {center_region_key}.")

        var_raw = history_regions[center_region_key].historyOutputs[output_variable.abaqus_name].data

        var_t = []  # Time points
        var_v = []  # Variable values
        for t, v in var_raw:
            if not (isinstance(v, str) and v.lower() == "novalue"):
                var_t.append(t)
                var_v.append(v)

        if len(var_t) == 0:
            raise RuntimeError(f"No valid data points found for {output_variable.abaqus_name} in node {center_region_key}.")

        var_t = np.array(var_t)
        var_v = np.array(var_v)

        if "Time" not in result.columns:
            result.add_column(var_t, "Time", unit=Q_.pref_dim("s"), description="Time points at which the variables were recorded.")
            var_t_reference = var_t
        elif not (np.allclose(var_t_reference, var_t)):
            tprint(f"\tTime points for {output_variable.abaqus_name} are different from reference time points. Interpolating to first variable's time points.", color="yellow")
            var_v = np.interp(var_t_reference, var_t, var_v)

        result.add_column(var_v, **output_variable.to_dict())

    return result


def extract_midpoint_force_displacement(history_regions, region_keys: list[str], coords: NDArray, **kwargs):
    """
    An example of an extractor function combining the base extractors above

    :param history_regions:
    :param region_keys:
    :param coords:
    :param kwargs:
    :return:
    """
    symmetry_factor = kwargs.get("symmetry_factor", 1)
    result = Data()

    ov_reaction_force = AbaqusOutputVariable("Force", "RF2", Q_.pref_dim("N"), "Reaction force in the loading direction (2 direction), by summing the contributionso f each node on the loading line.")
    ov_displacement = AbaqusOutputVariable("Displacement", "U2", Q_.pref_dim("mm"), "Displacement in the loading direction (2 direction), taken from the centre node (= corner node in double symetry model).")

    forces = extract_sum(history_regions, output_variables=ov_reaction_force, region_keys=region_keys, coords=coords, **kwargs)
    forces.columns["Force"] = symmetry_factor * forces.columns["Force"]  # Symmetry + making force positive
    displacements = extract_center_node(history_regions, output_variables=ov_displacement, region_keys=region_keys, coords=coords, **kwargs)
    displacements.columns["Displacement"] = -displacements.columns["Displacement"]  # Making displacement positive

    result.extend(displacements)

    # Interpolate time between the two if necessary
    if not np.allclose(forces.columns["Time"], displacements.columns["Time"]):
        tprint(f"\tTime points for reaction force and displacement are different. Interpolating to displacement time points.", color="yellow")
        forces_interp = np.interp(displacements.columns["Time"], forces.columns["Time"], forces.columns[ov_reaction_force.name])
        result.add_column(forces_interp, **ov_reaction_force.to_dict())
    else:
        result.add_column(forces.columns[ov_reaction_force.name], **ov_reaction_force.to_dict())
    return result


def extract_tensile_results(history_regions, region_keys: list[str], coords: NDArray, **kwargs):
    """
    Another example of an extractor function combining the base extractors above, for extracting tensile test results.

    :param history_regions:
    :param region_keys:
    :param coords:
    :param kwargs:
    :return:
    """
    symmetry_factor = kwargs.get("symmetry_factor", 1)  # Default to 1 for no symmetry
    result = Data()

    ov_reaction_force = AbaqusOutputVariable("Reaction Force", "RF2", Q_.pref_dim("N"), "Reaction force in the loading direction (2 direction), by summing the contributions of each node on the loading line.")
    ov_displacement = AbaqusOutputVariable("Displacement", "U2", Q_.pref_dim("mm"), "Displacement in the loading direction (2 direction), taken from the centre node (= corner node in double symetry model).")

    forces = extract_sum(history_regions, output_variables=ov_reaction_force, region_keys=region_keys, coords=coords, **kwargs)
    displacements = extract_identical(history_regions, output_variables=ov_displacement, region_keys=region_keys, coords=coords, **kwargs)
    result.extend(displacements)

    # Interpolate time between the two if necessary
    if not np.allclose(forces.columns["Time"], displacements.columns["Time"]):
        tprint(f"\tTime points for reaction force and displacement are different. Interpolating to displacement time points.", color="yellow")
        forces_interp = np.interp(displacements.columns["Time"], forces.columns["Time"], forces.columns[ov_reaction_force.name])
        result.add_column(forces_interp*symmetry_factor, **ov_reaction_force.to_dict())
    else:
        result.add_column(forces.columns[ov_reaction_force.name]*symmetry_factor, **ov_reaction_force.to_dict())

    if not kwargs.get("wire_only_test", False):
        result.add_column(result.columns["Reaction Force"] / (kwargs.get("width") * kwargs.get("thickness")), "Engineering Stress", Q_.pref_dim("MPa"), "Engineering stress, calculated as reaction force divided by original cross-sectional area.")
        result.add_column(result.columns["Displacement"] / kwargs.get("gauge_length"), "Engineering Strain", None, "Engineering strain, calculated as displacement divided by original gauge length.")
    else:
        result.add_column(result.columns["Reaction Force"] / (np.pi * (kwargs.get("diameter")/2)**2), "Engineering Stress", Q_.pref_dim("MPa"), "Engineering stress, calculated as reaction force divided by original cross-sectional area.")
        result.add_column(result.columns["Displacement"] / kwargs.get("gauge_length"), "Engineering Strain", None, "Engineering strain, calculated as displacement divided by original gauge length.")

    return result


def extract_prestrain_results(history_regions, region_keys: list[str], coords: NDArray, **kwargs):
    """
    Another example of an extractor function combining the base extractors above, for extracting prestrain results.

    :param history_regions:
    :param region_keys:
    :param coords:
    :param kwargs:
    :return:
    """
    result = Data()
    ov_temperature = AbaqusOutputVariable("Temperature", "TEMP", Q_.pref_dim("degC"), "Temperature in the wire.")
    ov_log_strain = AbaqusOutputVariable("Logarithmic strain", "LE11", None, "Logarithmic strain in the wire")
    ov_thermal_strain = AbaqusOutputVariable("Thermal strain", "THE11", None, "Thermal strain in the wire")
    ov_stress = AbaqusOutputVariable("Stress", "S11", Q_.pref_dim("MPa"), "Stress in the wire, calculated as stress in the loading direction.")

    temperature = extract_identical(history_regions, output_variables=ov_temperature, region_keys=region_keys, coords=coords, **kwargs)
    thermal_strain = extract_identical(history_regions, output_variables=ov_thermal_strain, region_keys=region_keys, coords=coords, **kwargs)
    log_strain = extract_identical(history_regions, output_variables=ov_log_strain, region_keys=region_keys, coords=coords, **kwargs)
    stress = extract_identical(history_regions, output_variables=ov_stress, region_keys=region_keys, coords=coords, **kwargs)

    result.extend(temperature)
    result.add_column(thermal_strain.columns[ov_thermal_strain.name], **ov_thermal_strain.to_dict())
    result.add_column(log_strain.columns[ov_log_strain.name], **ov_log_strain.to_dict())
    result.add_column(stress.columns[ov_stress.name], **ov_stress.to_dict())

    return result


def extract_history_output(odb_file_path: Path, node_set_name: str, results_dir: Path, sample_data: dict, extractor, steps_to_process: str | list[str] | None = None, allow_multiple_instances: bool = False, results_file_suffix: str = "", **kwargs):
    """
    Extract the history output of the prestrain step for the wires, and save it to a .csv file.

    :param odb_file_path: path to the .odb file to extract the data from
    :param node_set_name: name of the node set to extract the data from (name can be found by inspecting the .odb file, because can be different from the name of the node set in the model (e.g. capitalization))
    :param results_dir: path to the directory where the extracted data should be saved
    :param sample_data: a dict of metadata about the sample, that will be included in the metadata output file
    :param extractor: a function that will extract the desired variable from the history output, given the history regions, region keys, and coordinates of the nodes in the node set. The function should return a tuple of (data, metadata), where data is a 2D numpy array of shape (n_variables, n_time_points), and metadata is a dict of metadata about the extracted data columns (e.g. variable names, units, descriptions)
    :param steps_to_process: a list of step names to process. If None, all steps will be processed.
    :param allow_multiple_instances: whether to allow the node set to be defined on multiple instances or include nodes of multiple instances. E.g. if the goal is to sum reaction forces, this should remain False as the results could otherwise be unexpected...
    :param kwargs: additional keyword arguments to be passed to the extractor function

    :return: path to the saved .csv file
    """
    if isinstance(steps_to_process, str):
        steps_to_process = [steps_to_process]

    tprint(f"Extracting output history from ODB file '{odb_file_path}' for node set '{node_set_name}'...")
    if isinstance(odb_file_path, Path) and not str(odb_file_path).endswith(".odb"):
        odb = openOdb(str(odb_file_path) + ".odb")
    else:
        odb = openOdb(str(odb_file_path))

    history = Data()
    try:  # Avoid locking odb file in case of errors
        odb_steps = odb.steps.keys()
        steps_to_process = steps_to_process or odb_steps  # If no specific steps provided, process all steps

        step_index = 0
        for odb_step_name, odb_step in odb.steps.items():
            step_index += 1

            if odb_step_name not in steps_to_process:
                tprint(f"\tSkipping step {odb_step_name}.")
                continue

            odb_history_regions = odb_step.historyRegions.keys()

            tprint(f"\tProcessing step {odb_step_name} ({step_index}/{len(odb_steps)})")

            # Resolve node set
            region_keys = []
            coords = []
            if node_set_name in odb.rootAssembly.nodeSets:
                node_sets_per_instance = odb.rootAssembly.nodeSets[node_set_name].nodes
                if len(node_sets_per_instance) == 0:
                    raise RuntimeError(f"Node set '{node_set_name}' is empty.")

                # Extract nodes
                if len(node_sets_per_instance) > 1:
                    if not allow_multiple_instances:
                        raise RuntimeError(f"Node set '{node_set_name}' is defined on multiple instances, which was disallowed. Set 'allow_multiple_instances=True' to allow this.")

                    nodes = []
                    for instance_nodes in node_sets_per_instance:
                        nodes.extend([node for node in instance_nodes])
                else:
                    nodes = node_sets_per_instance[0]

                for node in nodes:
                    region_keys.append(f"Node {node.instanceName}.{node.label}")
                    coords.append(node.coordinates)

            # Resolve element set
            if node_set_name in odb.rootAssembly.elementSets:

                element_sets_per_instance = odb.rootAssembly.elementSets[node_set_name].elements
                if len(element_sets_per_instance) > 1:
                    if not allow_multiple_instances:
                        raise RuntimeError(f"Element set '{node_set_name}' is defined on multiple instances, which was disallowed. Set 'allow_multiple_instances=True' to allow this.")

                    elements = []
                    for instance_elements in element_sets_per_instance:
                        elements.extend([element for element in instance_elements])
                else:
                    elements = element_sets_per_instance[0]

                elements_keys = {f"{element.instanceName}.{element.label}": element for element in elements}

                for key in odb_history_regions:
                    if key.startswith(f"Element "):
                        _, odb_element, _, _, integration_point = key.split(" ")
                        if odb_element in elements_keys:
                            region_keys.append(key)
                            coords.append([np.nan, np.nan, np.nan])  # Note: getting coordinates for integration points is complicated, and shouldn't be needed

            if len(region_keys) == 0:
                raise RuntimeError(f"Node set or element set '{node_set_name}' not found in ODB file '{odb_file_path}'.")

            coords = np.array(coords)

            result = extractor(odb_step.historyRegions, region_keys, coords=coords, **kwargs)

            step_indices = np.full(result.column_length, step_index, dtype=int)  # Create an array of the same length as the number of time points, filled with the step index
            result.add_column(step_indices, "Step index", unit=None, description="An integer index of the step in the Abaqus model (starting from 1). Use the array in '['Model type']['Steps']' to get the step names corresponding to these indices.")

            history.extend(result)
    finally:
        odb.close()

    output_file = DataFile(results_dir / f"{odb_file_path.stem}{results_file_suffix}")
    now = datetime.now()
    output_file_metadata = {
        "General": {
            "Year": now.year,
            "Month": now.month,
            "Day": now.day,
            "Time": str(now.time()),
            "Location": "ISAE-Supméca",
            "Operator": "Wouter Vermeulen"
        },
        "Model type": {
            "Type": "",
            "ODB file": str(odb_file_path),
            "Job name": str(odb_file_path.name[:-4]),
            "Steps": list(odb_steps),
        },
        **sample_data,
        "Columns": history.column_metadata
    }

    output_file.save(data=history.data.T, metadata=output_file_metadata)
    tprint(color_text(f"\tAll steps processed successfully to {output_file.path}.", color="green"))

    return output_file.path


def extract_eigenfrequencies(odb_file_path: Path, step: str, results_dir: Path, sample_data: dict):
    """
    Extract the eigenfrequencies from the first step of the ODB file, and save it to a .csv file.

    :param odb_file_path: path to the .odb file to extract the data from
    :param step: name of the step to extract the data from
    :param results_dir: path to the directory where the extracted data should be saved
    :param sample_data: a dict of metadata about the sample, that will be included in the metadata output file

    :return: path to the saved .csv file
    """
    odb = openOdb(path=str(odb_file_path))
    odb_step = odb.steps[step]
    region = odb_step.historyRegions['Assembly ASSEMBLY']
    freqData = region.historyOutputs['EIGFREQ'].data

    frequencies = [f[1] for f in freqData]
    DataFile(results_dir / f"{odb_file_path.stem}_eigenfrequencies").save(
        data=np.array(frequencies), metadata=sample_data)

