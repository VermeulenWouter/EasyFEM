# EasyFEM: A Python framework for finite element analysis with Abaqus

## Description
This repository contains the code, data and documentation for the TFE (*Travail de Fin d'Études*, Master's Thesis) of Wouter Vermeulen (2025-2026). The TFE focuses on numerical modelling of hybrid FRP-SMA and FRP-steel samples. The EasyFEM code is a wrapper around the Abaqus Python API, allowing for easy and fast implementation of finite element models for compressive, tensile and three-point bending tests.

## Author
Main author:
* Wouter Vermeulen: [woutervermeul@gmail.com](mailto:woutervermeul@gmail.com) (UCLouvain - EPL)

Supervisor and co-supervisors:
* Prof. [João Almeida](https://www.researchgate.net/profile/Joao-Pacheco-Almeida) (UCLouvain - EPL)
* Prof. [Olga Klinkova](https://www.researchgate.net/profile/Olga-Klinkova) (ISAE-Supméca)
* Prof. [Stefania Lo Feudo](https://www.researchgate.net/profile/Stefania-Lo-Feudo) (ISAE-Supméca)


## Getting started

### Clone the repository
To get started with this repository, clone it to your local machine using the following command:

```bash
git clone https://github.com/VermeulenWouter/EasyFEM.git
```

or download the full repository as a ZIP file and extract it to your desired location.

### Install/verify dependencies

#### Abaqus
To run the Python scripts that interact with Abaqus, ensure you have Abaqus installed on your machine. These scripts do not require Python to be installed separately, as Abaqus comes with its own Python interpreter. The code has been tested with Abaqus Learning Edition 2024 (on Windows 10) and Abaqus 2024 (on Linux server).

#### Python
If you want to run only Python scripts that do not interact with Abaqus, it suffices to have Python 3.10 or higher installed. You can download it from the [official Python website](https://www.python.org/downloads/).

#### Python packages
Abaqus will not find any packages installed in the system Python. Therefore, the required Python packages are bundled in the `site-packages` folder.

Alternatively, it is also possible to install the packages manually (not copying the `site-packages` folder) by running the following command in the root directory of the repository (the folder containing this `README.md` file) for each dependency listed in the `requirements.txt` file, :

```bash
abaqus python -m abqPip install <dependency>
```

### Setup of path logic
This codebase generates and reads a multitude of files. A helper file (`utils/paths.py`) abstracts most of the path handling away, allowing to use for example `DIR_MODELS` for the main models directory, or `DIR_DATA_OUT` for the main directory to which to save output data. Some folders are generated depending on the source file: e.g. `DIR_FIGURE_OUT` is located in the same folder as the script that is ran and allows to easily find back saved figures without hardcoding full paths. To ensure this last functionality works, ensure the paths are properly set up.

If Python 3.10 or higher is installed, first run
 ```bash
 python3.10 -m source.<folder>.<script_name>
 ```
 (`python3.10` can also be simply `python3` or `python`, depending on the installation), where `<script_name>` must not contain the `.py`.

 If Python 3.10 or higher is not installed, manually adapt the path in the 
 `source/config/paths_config.txt` file:
 ```text
Project root directory: <root>
Caller file: <root>\source\<folder>\<script>  # (use `/` for Linux instead of `\`)
Safe caller name: <a safe name for the caller name (ie. valid for the file system, as a folder with this name will be created), e.g. `compression_py`>
Caller parent directory: <root>\source\<folder>  # (use `/` for Linux instead of `\`)
 ```
where

| Identifier  | Explanation                                                                                                                                                                                                                |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `<root>` | the root directory of the repository (the folder containing this `README.md` file). For example, on the development system it is `C:\Users\Wouter\Documents\TFE` (Windows) or `/home/vermeulenw/TFE` (on the Linux server) |
| `<folder>`    | the folder in which the Python script is located, e.g. `compression` for `source/compression/compression.py`. This can be omitted if the script is directly in the `source/` folder.                                       |
| `<script>`    | the name of the Python script to run, e.g. `compression.py` for `source/compression/compression.py`.                                                                                                                       |


## Running the code
Each script in the `source/` folder contains a docstring at the top explaining its purpose, required inputs, and expected outputs. You can run these scripts using the Python interpreter or within the Abaqus environment, depending on their functionality.

Note scripts are expected to be run from the root directory of the repository (that is, the folder this `README.md` file is in) to ensure correct file path handling. The run command (to be entered on the command line) would thus look like this:

```bash
python -m source.<folder>.<script_name>
```
or, for scripts using Abaqus (or if Python is not separately installed):
```bash
abaqus cae nogui=source/<folder>/<script>
```

**Replace `<script_name>` with the actual names of the scripts you want to run, without the `.py` extension, and `<script>` with the `.py` extension.*

## Repository Structure

### Code (Python)


| Folder         | Content                                                                                                                                                                                                            |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data/raw/`    | Raw data files, e.g. directly obtained from experiments. Can be too big for the git repository (which shall be indicated in the appropriate `README` files in the subfolders of `data/raw`).                       |
| `data/input/`  | Input data files, e.g. (cleaned) data from experiments.                                                                                                                                                            |
| `data/output/` | Output data files generated by the code, e.g. simulation results or data analyses. Can then be used for generation of figures.                                                                                     |
| `source/`      | Contains scripts that analyse the data, create numerical models and extract useful data from the model results. The ones included in this repository are but examples of what can be done with the provided utils! |
| `utils/`       | Utility scripts and functions that can be used in all other code files, helping to keep this codebase maintainable (e.g. handling of file paths, plot generation, ...)                                             |


Within `utils`, the subpackage `abaqus/` implements all interfacing:


| File                        | Content                                                                                                                                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `abaqus/__init__.py`        | A `Model` class to easily create and execute jobs                                                                                                                                                                    |
| `abaqus/materials.py`       | Material classes and functions, wrapping around the Abaqus commands, for easy material definitions.                                                                                                                  |
| `abaqus/numerical_tests.py` | Functions to easily create experimental tests in Abaqus: compression, tension, three-point bending and dynamic. These are each implemented in a class, allowing for a simple create -> execute -> extract data loop. |
| `abaqus/postprocessing.py`  | Low-level functions to extract data from Abaqus output databases.                                                                                                                                                    |


## Example usage

The `numerical_tests` module contains the different test types used in this project: `TensileTest`, `CompressiveTest`, `ThreePointBendingTest`, `DynamicTest`. Each test type is implemented as a class, which takes care of the creation of the Abaqus model, the definition of the material properties, the application of the boundary conditions and loads, and the output requests. The test classes are designed to be flexible and allow for customization of the test parameters, while still providing a default setup that can be used for quick testing (i.e. they are modularised, so that for example creating the mesh is relatively independent of the geometry, it is easy to implement a new section definition, ...). The classes also allow the model to be run and extract the results! Below an example of a three-point-bending test and dynamic test are presented, after a short introduction to the material definition and `Quantity` class.

### The `Quantity` class
To ensure that the units of the input parameters are correct, the `Quantity` class, an adaptation of the `pint` module, is used. This class allows you to specify the value and unit of a parameter, and it will automatically convert it to the desired unit when needed. In the ``units.py`` file, the default unit system is defined (e.g. ``mm - t - s - K - N - MPa - mJ``). In the code, define dimensional inputs as a `Quantity`, for example:

```python
from utils.units import ureg
sample_width = 15 * ureg.mm
yield_stress = 100 * ureg.MPa
youngs_modulus = 34 * ureg.GPa  # Auto-converted to 34000 MPa!
```
When using this as input for a function, the `Quantity` converts to the default unit system, so a function could be written as:

```python
from utils.units import Quantity, ureg
def calculate_stress(force: Quantity, area: Quantity) -> Quantity:
    stress = force / area
    return stress

print(f"{calculate_stress(5000 * ureg.N, 1 * ureg.m**2):~P}")
# >> 0.5 MPa
```
Independently of the units used for `force` and `area`, the output will always be in the default unit system (in this case, will be in MPa), because the inputs will have been converted at the moment of definition.

The general goal is to use these Quantity objects until calling the Abaqus API (which only accepts floats), and to convert the floats given by the Abaqus ODB API back to `Quantity` objects as soon as possible, so that the user can work with the units in a consistent way throughout the code.

### Materials
The `materials` module contains the `Material` class, which is used to define the material properties of the model, and different subclasses for the materials used in this project: `FRPMaterial`, `NiTiCoMaterial` and `StainlessSteelMaterial` (as they use substantially different constitutive descriptions and have only the elastic property in common). The effective creation of a material has been implemented with `create_<material>` functions. They use default values for the material properties, which can be overridden by the user by keyword arguments. For example, to create a NiTiCo material with a custom austenite Young's modulus, you can do:

```python
from utils.abaqus.materials import create_NiTiCo_material
from utils.units import ureg
nitico_material = create_NiTiCo_material(name="NiTiCo adapted austenite Young's modulus", E=50*ureg.GPa)
```
This will create a NiTiCo material with a Young's modulus of 50 000 MPa, while all other properties will be set to their default values. The main currently predefined materials are: `create_FRP_material_hashin` for a fibre-reinforced polymer material with Hashin damage, `create_NiTiCo_material` for a NiTiCo shape memory alloy material, and `create_AISI302_material` for a stainless steel AISI302 material.

*One small note: the `create_<material>` functions do not return a `Material` object directly, but a factory function: the material must be associated with a `Model`, and this solution allows for the material to be defined before the model is created (and in multiple models if needed). The association with a model is done automatically in the `<TestType>Test` classes when creating an Abaqus Model.*


### A three-point-bending test example script
Below is the code to generate a hybrid model using NiTiCo wires (4% wire volume) prestrained to 4%. The output is a `.csv` file in the `./results` folder, containing the force-displacement pairs from all increments where they were registred by Abaqus; and an associated `.json` file with the metadata. A full step-by-step example to create the same model manually in Abaqus/CAE is given in the `./StepByStepAbaqusModel.pdf` file.

```python
from utils.abaqus.materials import create_FRP_material_hashin, create_NiTiCo_material
from utils.abaqus.numerical_tests import ThreePointBendingTest
from utils.units import ureg

from pathlib import Path

frp_material = create_FRP_material_hashin(name="FRP material for three-point-bending test")
nitico_material = create_NiTiCo_material(name="NiTiCo material for three-point-bending test")

testmodel = ThreePointBendingTest(
    # Name of the model (visible in Abaqus/CAE)
    name="Three-point-bending model demonstration",
    # Directory where the model will be created
    directory=Path("./"),  
    # Material (factory) for the FRP part of the model
    frp_material=frp_material,  
    # Mesh size for the fine mesh (around the load application point)
    fine_mesh_size=0.375*ureg.mm,  
    # Set to `True` if prestrain will be applied (to fix step order)
    include_wire_prestrain_step=True 
)

# Wire insertion is separated, as it is not always wanted
testmodel.add_wires(
    # Material (factory) for the wires
    wire_material=nitico_material,  
    # Percentage of the cross-sectional area of the FRP part that is occupied by the wires (e.g. 4% means that the wires will occupy 4% of the cross-sectional area of the FRP part)
    wire_percentage=4*ureg.percent, 
    # Prestrain of the wires (e.g. 2% means that the wires will be prestrained to 2% of their original length, which will induce a compressive stress in the FRP part of the model)
    wire_prestrain=4*ureg.percent,  
    # Allow to change between centred and off-centred wire configuration
    wire_configuration="offcentred" 
)

# Create a job for the model, which can be run in Abaqus/CAE or from the command line
# To use multiple processors (requires a full license, not suppported by Abaqus Learning Edition), add `numCpus=<n>, numDomains=<n>, multiprocessingMode=THREADS` as arguments; where <n> is the number of processors to use (e.g. 4)
testmodel.create_job()  

# Save the model to a .cae file, which can be opened in Abaqus/CAE and inspected there (and run from there, as a job will be pre-created as well)
testmodel.save_cae() 

# Run the model directly from Python (the model will be run, which can take some time - the script waits for completion)
testmodel.run_job()

# Extract the results (e.g. the force-displacement curve). Two files will be created: one with the raw data (in this case: Force and Displacement of the centre of the sample - these columns are described in the metadata file) and one with metadata: type of test, material properties used, mesh size
testmodel.extract_results(results_dir=Path("./results"))
```

### A dynamic test example script
A dynamic test to output the eigenfrequencies of a sample can be created as well. The example below shows how to create a dynamic test with an FRP material and a NiTiCo wire, very similarly to the three-point-bending test above. The main differences are that the wire inclusion is specified in the constructor of the class, and that the job is also created by default.


```python
from utils.abaqus.materials import create_FRP_material_hashin, create_NiTiCo_material
from utils.abaqus.numerical_tests import DynamicTest
from utils.units import ureg

from pathlib import Path

frp_material = create_FRP_material_hashin(name="FRP material for dynamic test")
nitico_material = create_NiTiCo_material(name="NiTiCo material for dynamic test")

testmodel = DynamicTest(
    name="Dynamic model demonstration",
    directory=Path("./"),
    frp_material=frp_material, 
    wire_material=nitico_material, 
    wire_percentage=4*ureg.percent, 
    wire_prestrain=2*ureg.percent,  
    wire_configuration="offcentred"
)

testmodel.save_cae()
testmodel.run_job()
testmodel.extract_results(results_dir=Path("./results"))
```

Note the output for eigenfrequencies is a `.csv` file containing only the eigenfrequencies - mode shapes should be checked in the `.odb` file.

## Other notes
### Easy control of the model parameters
Even though a lot of the control parameters for the model are set to default values, they can be easily overridden by the user. For example, to make the three-point-bending test cyclic, it suffices to pass an `amplitude` list to the `ThreePointBendingTest` class. To use a boundary condition directly to the displacement of the load application point, instead of applying it through a 3D loading tip analytical surface, it suffices to pass `loading_tip="Direct BC"`, ... Some are also in the code base through `**kwargs` in the function headers. For example, `full_sample_length` to change the sample length for the three-point-bending test.

### On the postprocessing of the `.odb` file
The idea behind the module is to allow for easy extension with new extractors, while keeping the set-identification logic and the `.odb` file opening and closing logic in one place.

The postprocessing of the `.odb` file is done with the `postprocessing.py` module. This contains one general function `extract_history_output` that opens the `.odb` and looks for a particular node or element set (without requiring the node or element numbers to be specified, these are found by the code - although this requires iterating over all sets in the model, which can take some time for large models), then extracts the data using one of the `extract_<description>` functions, making it easily adaptable. Predefined functions are for example `extract_identical` which requires the output values to be identical for all nodes or elements in the set (e.g. to check the prestrain is properly applied on the whole length of a wire), `extract_sum` which sums the output values for all nodes or elements in the set (e.g. to get the total force on a surface), `extract_center_node` which extracts the output values for the node closest to the centre of the sample (note: this one is already applied for a symmetrical sample and the three-point-bending test). They can then be combined into `extract_midpoint_force_displacement` or `extract_tensile_results` to get the force-displacement curve for a three-point-bending test or a tensile test, respectively. The output is then saved in a `.csv` file, together with a metadata file containing the model parameters.

## Licence and citation
Copyright (c) Wouter Vermeulen 2026. Please contact the author for any non-academic use.

To reference this code, please cite either the master's thesis for which this code was developed, or the software repository itself:
```bibtex
@mastersthesis{Vermeulen2026,
    title={Finite element modelling of hybrid fibre-reinforced polymer composites},
    subtitle = {Damage modelling, wire integration, and interface behaviour},
    type={Master's Thesis},
    author={Vermeulen, Wouter},
    school={Ecole polytechnique de Louvain, Université catholique de Louvain},
    address = {Louvain-la-Neuve, Belgium},
    year={2026},
    note={Supervisors: Saraiva Esteves Pacheco De Almeida, João; 
    Lo Feudo, Stefania; Klinkova, Olga}
}

@software{EasyFEM,
    title = {EasyFEM: A Python framework for finite element analysis with Abaqus},
    author = {Vermeulen, Wouter},
    year = {2026},
    month = {8},
    publisher = {Zenodo},
    doi = {10.5281/zenodo.21175593},
    version = {v1.0.0},
}
```