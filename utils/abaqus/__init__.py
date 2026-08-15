"""
Abaqus Model class and helper functions to run models from Python, without using the Abaqus GUI.

@author: Wouter Vermeulen
@date: 2026-05-28 (refactor from previous versions)
"""

from abaqus import mdb
from abaqusConstants import STEP, SOLVER_DEFAULT, UNSET, NONE, OMIT, FINITE, ON, MODEL
from step import *  # DO NOT DELETE - using this import forces abaqus to load the module, necessary for e.g. HistoryOutputRequest to work
from interaction import *  # DO NOT DELETE - using this import forces abaqus to load the module, necessary for e.g. EmbeddedRegion to work

from datetime import datetime
import os
from pathlib import Path
import subprocess

from utils import ureg, Quantity
from utils.prettyprint import tprint


def clear_model_database() -> None:
    """
    Clear the model database by deleting all models. This is useful to avoid saving the same model multiple times
    when calling ``save_mdb_to_cae_file`` multiple times, or to simply clear the database after saving.
    """
    mdb.Model("Model-1")  # Create default model again, because abaqus doesn't allow a completely empty mdb
    if clear_model_database:
        for model in Model.all_created_models:
            del mdb.models[model.name]
        Model.last_created_model = None
        Model.all_created_models = []


def save_mdb_to_cae_file(file_path: str | Path, clear_models: bool = False) -> Path:
    """
    Save all models currently in the model database to a .cae file. By default, the model database is cleared
    after saving, to avoid saving the same model twice by calling this function multiple times. Use
    ``clear_model_database=False`` to keep the models in the database after saving.

    :param file_path: the path to the .cae file to save the models to (the file will be created)
    :param clear_models: if ``True``, clear the model database after saving (default: ``True``)

    :return: the path to the saved .cae file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path).absolute()

    # Gets created by default but doesn't contain anything
    if "Model-1" in mdb.models and "Model-1" not in [model.name for model in Model.all_created_models]:
        del mdb.models["Model-1"]

    mdb.saveAs(pathName=str(file_path))

    if clear_models:
        clear_model_database()

    return file_path


def insert_in_input_file(inp_file_path: str, new_text: str, line_above: str | None = None,
        line_below: str | None = None, offset: int = 0):
    """
    Helper function to insert text in the .inp file once it is generated, and before it is run.
    """
    if line_above is None and line_below is None:
        raise ValueError("Either line_above or line_below needs to be defined")
    if line_above is not None and line_below is not None:
        raise ValueError("Only one of line_above or line_below can be defined, not both")

    line_to_search = (line_below if line_below is not None else line_above).strip().lower()
    offset = 1 + offset if line_above is not None else offset

    with open(inp_file_path) as f:
        inp_file_content = f.readlines()

    found = False
    for i, line in enumerate(inp_file_content):
        if line.strip().lower().startswith(line_to_search):
            inp_file_content.insert(i + offset, new_text)
            found = True
            break

    if not found:
        raise ValueError(f"Could not find line starting with '{line_to_search}' in input file to insert new text. "
                         f"Please check the line_above or line_below argument and make sure it matches the content of "
                         f"the input file (ignoring whitespace and case).")

    with open(inp_file_path, "w") as f:
        f.write("".join(inp_file_content))


def check_job_status(directory: Path, job_name: str) -> bool:
    """
    Check the .sta file of the job to see if it completed successfully. If the .sta file is not found, or if it does
    not contain "COMPLETED", a warning is printed (but code execution continues).

    :param directory: the directory where the .sta file is located (usually the same as the job working directory)
    :param job_name: the name of the job (the .sta file is expected to be named "{job_name}.sta" and located in
        the specified directory)

    :return: True if the job completed successfully, False otherwise (a warning is also printed in that case).
    """
    sta_file = directory / f"{job_name}.sta"

    if os.path.exists(sta_file):
        with open(sta_file, 'r') as f:
            lines = f.readlines()
            if any("COMPLETED" in line for line in lines):
                tprint("Analysis completed successfully.", color="green")
                return True
            else:
                tprint("Warning: Analysis did not complete.", color="yellow")
                return False
    else:
        tprint("Warning: .sta file not found. Unable to check job status.", color="yellow")
        return False


class Model:

    # Class variables to keep track of created models (useful for saving to .cae file and clearing model database)
    last_created_model = None
    all_created_models = []

    def __init__(self, name: str, description: str = None, model_type: str = "Standard",
            directory: Path | None = None, **kwargs) -> None:
        """
        Create a new empty model in the model database.

        :param name: the name of the model (must be unique in the model database)
        :param description: a description of the model
        :param model_type: the type of the model, either ``Standard`` or ``Explicit``
            This will allow to automatically create the appropriate step type, and if used in conjunction with any of
            the classes in ``utils.abaqus.tests`` the mesh element type.
        :param directory: the directory where the model is created (if None, the current working directory is used)
            If the directory doesn't exist yet, it is created. Even if no cae or inp files are created, the
            directory is used for temporary files during model creation!
        """
        self.type = model_type

        if directory is not None:
            if not directory.exists():
                directory.mkdir(parents=True)
            os.chdir(directory)

        self.directory = directory

        # Setup
        self._model = mdb.Model(name=name, description=description or "")
        Model.last_created_model = self
        Model.all_created_models.append(self)
        self._previous_step = "Initial"

        # List of actions to perform on the .inp file - these will be executed in order when the model is built
        self.actions_on_inp = []

        # Values to be filled during job creation
        self._inp_file_path = None
        self.job = None

        # Default creations (for easy use later on)
        self.create_amplitude("LinearIncrease", data=[(0.0, 0.0), (1.0, 1.0)])

    def create_amplitude(self, name: str, data: list[tuple[float, float]]):
        """Create a tabular amplitude based on data."""
        return self._model.TabularAmplitude(data=tuple(data), name=name, smooth=SOLVER_DEFAULT, timeSpan=STEP)

    def create_step(self, name: str, minimum_number_of_increments: int | None = 10, step_time: float = 1.0,
            nlgeom: bool = True) -> None:
        """
        Create a step for the model, with appropriate settings based on the model type.

        :param name: the name of the step (must be unique in the model)
        :param minimum_number_of_increments: the minimum number of increments to use for the step (only used for
            Standard step, ignored for Explicit step)
        :param step_time: the total time for the step
        :param nlgeom: switch the ``nlgeom`` option on or off for the step
        """
        maxInc = 1 / minimum_number_of_increments
        minInc = maxInc / 1e5
        initialInc = maxInc / 100
        maxNumInc = 1e5 if minimum_number_of_increments is None else int(1 / minInc)
        if self.type == "Standard":
            self._model.StaticStep(name=name, previous=self._previous_step, initialInc=initialInc, minInc=minInc,
                maxInc=maxInc, maxNumInc=maxNumInc, timePeriod=step_time, nlgeom=nlgeom)
        elif self.type == "Explicit":
            self._model.ExplicitDynamicsStep(name=name, previous=self._previous_step, timePeriod=step_time,
                nlgeom=nlgeom)

        self._previous_step = name

    def create_frequency_step(self, name: str = "Frequency", max_frequency: Quantity = 2500 * ureg.Hz):
        """
        Create a frequency step for the model, with appropriate settings based on the model type.

        :param name: the name of the step (must be unique in the model)
        :param max_frequency: the maximum frequency to consider for the eigenfrequencies
        :return:
        """
        self._model.FrequencyStep(limitSavedEigenvectorRegion=None, maxEigen=max_frequency.m, name=name,
            previous=self._previous_step)

    def create_job(self, name: str | None = None, description: str | None = None, **kwargs) -> str:
        """
        Create a job for the model, and create its input file.

        :param name: the name of the job (if None, the model name is used with spaces replaced by underscores)
        :param description: the description of the job
        :param kwargs: any additional arguments to pass to the ``mdb.Job`` constructor (e.g. ``numCpus``,
            ``memory``, etc.)

        :return: the name of the created job (can be slightly different form the input to avoid spaces which are
        not allowed in job names)
        """
        if name is None:
            name = self._model.name
        name = name.replace(" ", "_")

        if name in mdb.jobs.keys():
            del mdb.jobs[name]

        self.job = mdb.Job(model=self._model.name, name=name, description=description, **kwargs)
        self.job.writeInput()
        if len(self.actions_on_inp) > 0:
            self._inp_file_path = self.job.name + ".inp"

            for action in self.actions_on_inp:
                action(self._inp_file_path)

        return self.job.name

    def save_cae(self, directory: Path | None = None, filename: str | None = None) -> Path:
        """
        Save the model to a .cae file in the specified directory. By default, the model database is cleared
        after saving, to avoid saving the same model twice by calling this function multiple times. Use
        ``clear_model_database=False`` to keep the models in the database after saving.

        :param directory: the directory where the .cae file is saved (the file will be created)
        :param filename: the name of the .cae file (if None, the model name is used with spaces replaced by underscores)

        :return: the path to the saved .cae file
        """
        if directory is None:
            if self.directory is None:
                raise ValueError("No directory specified for saving the .cae file, and no default directory set for "
                                 "the model.")
            directory = self.directory

        if not directory.exists():
            directory.mkdir(parents=True)

        if filename is not None:
            cae_file_path = directory / filename
        else:
            cae_file_path = directory / f"{self.name}.cae"
        return save_mdb_to_cae_file(cae_file_path)

    def run_job(self, directory: Path | None = None, false_run: bool = False) -> bool:
        """
        Run the job, and wait for completion. If actions on the .inp file were defined, the job will be run using
        subprocess instead of the Abaqus API, to ensure the changes to the .inp file are included.

        *Note: each model is supposed to only have one associated model*

        :param directory: the directory where the job is run (usually the same as the job working directory)
        :param false_run: if True, the job is not actually run, but the function simulates a run
            (useful for testing the code without running actual simulations)

        :return: True if the job completed successfully, False otherwise (a warning is also printed in that case).
        """
        if false_run:
            tprint(f"Simulating run of job '{self.job.name}' ...", color="yellow")
            return True

        if directory is None:
            if self.directory is None:
                raise ValueError("No directory specified for running the job, and no default directory set for "
                                 "the model.")
            directory = self.directory

        tprint(f"Submitting job '{self.job.name}' ...")

        if len(self.actions_on_inp) > 0:
            time_start = datetime.now()
            subprocess.run(["abaqus", f"job={self.job.name}", f"input={directory / self.job.name}.inp"], check=True)
            time_end = datetime.now()
            time_simulation = time_end - time_start
            tprint(f"Total simulation time: {time_simulation}")
        else:
            time_start = datetime.now()
            self.job.submit()
            self.job.waitForCompletion()
            time_end = datetime.now()
            time_simulation = time_end - time_start
            tprint(f"Total simulation time: {time_simulation}")

        return check_job_status(directory, self.job.name)

    # Given Abaqus doesn't allow to inherit from its ``Model`` class, create methods in this class to act as though
    # it inherited them.

    def Material(self, name: str, description: str = None, **kwargs):
        """Shortcut to create a material in the model."""
        return self._model.Material(name=name, description=description or "", **kwargs)

    def ConstrainedSketch(self, name: str, sheetSize: float = 200.0, **kwargs):
        """Shortcut to create a constrained sketch in the model."""
        return self._model.ConstrainedSketch(name=name, sheetSize=sheetSize, **kwargs)

    def Part(self, name: str, dimensionality: str, type: str, **kwargs):
        """Shortcut to create a part in the model."""
        return self._model.Part(name=name, dimensionality=dimensionality, type=type, **kwargs)

    def TrussSection(self, name: str, material: str, area: float = 1):
        """Shortcut to create a truss section in the model."""
        return self._model.TrussSection(name=name, material=material, area=area)

    def TabularAmplitude(self, name: str, data: list[tuple[float, float]], smooth=SOLVER_DEFAULT, timeSpan=STEP):
        """Shortcut to create a tabular amplitude in the model."""
        return self._model.TabularAmplitude(name=name, data=tuple(data), smooth=smooth, timeSpan=timeSpan)

    def DisplacementBC(self, name: str, createStepName: str, region, u1=UNSET, u2=UNSET, u3=UNSET, ur1=UNSET,
            ur2=UNSET, ur3=UNSET, localCsys=None, **kwargs):
        """Shortcut to create a displacement boundary condition in the model."""
        return self._model.DisplacementBC(name=name, createStepName=createStepName, region=region, u1=u1, u2=u2, u3=u3,
            ur1=ur1, ur2=ur2, ur3=ur3, localCsys=localCsys, **kwargs)

    def EncastreBC(self, name: str, createStepName: str, region, localCsys=None, **kwargs):
        """Shortcut to create an encastre boundary condition in the model."""
        return self._model.EncastreBC(name=name, createStepName=createStepName, region=region, localCsys=localCsys,
            **kwargs)

    def XsymmBC(self, name: str, createStepName: str, region, localCsys=None, **kwargs):
        """Shortcut to create a Xsymm boundary condition in the model."""
        return self._model.XsymmBC(name=name, createStepName=createStepName, region=region, localCsys=localCsys,
            **kwargs)

    def YsymmBC(self, name: str, createStepName: str, region, localCsys=None, **kwargs):
        """Shortcut to create a Ysymm boundary condition in the model."""
        return self._model.YsymmBC(name=name, createStepName=createStepName, region=region, localCsys=localCsys,
            **kwargs)

    def ZsymmBC(self, name: str, createStepName: str, region, localCsys=None, **kwargs):
        """Shortcut to create a Zsymm boundary condition in the model."""
        return self._model.ZsymmBC(name=name, createStepName=createStepName, region=region, localCsys=localCsys,
            **kwargs)

    def Temperature(self, name: str, createStepName: str, region, magnitudes, **kwargs):
        """Shortcut to create a temperature boundary condition in the model."""
        return self._model.Temperature(name=name, createStepName=createStepName, region=region, magnitudes=magnitudes,
            **kwargs)

    def EmbeddedRegion(self, name: str, embeddedRegion, hostRegion, **kwargs):
        """Shortcut to create an embedded region in the model."""
        return self._model.EmbeddedRegion(name=name, embeddedRegion=embeddedRegion, hostRegion=hostRegion, **kwargs)

    def ContactProperty(self, name: str):
        """Shortcut to create a contact property in the model."""
        return self._model.ContactProperty(name=name)

    def SurfaceToSurfaceContactStd(self, name: str, createStepName: str, main, secondary, sliding=FINITE, thickness=ON,
            interactionProperty=None, adjustMethod=NONE, initialClearance=OMIT, datumAxis=None,
            clearanceRegion=None, **kwargs):
        """Shortcut to create a surface-to-surface contact interaction in the model."""
        return self._model.SurfaceToSurfaceContactStd(name=name, createStepName=createStepName, main=main,
            secondary=secondary, sliding=sliding, thickness=thickness, interactionProperty=interactionProperty,
            adjustMethod=adjustMethod, initialClearance=initialClearance, datumAxis=datumAxis,
            clearanceRegion=clearanceRegion, **kwargs)

    def HistoryOutputRequest(self, name: str, createStepName: str, region, variables: tuple[str, ...], **kwargs):
        """Shortcut to create a history output request in the model."""
        return self._model.HistoryOutputRequest(createStepName=createStepName, name=name, region=region,
            variables=variables, **kwargs)

    def FieldOutputRequest(self, name: str, createStepName: str, variables: tuple[str, ...], region=MODEL, **kwargs):
        """Shortcut to create a field output request in the model."""
        return self._model.FieldOutputRequest(createStepName=createStepName, name=name, region=region,
            variables=variables, **kwargs)

    @property
    def rootAssembly(self):
        """Shortcut to access the root assembly of the model."""
        return self._model.rootAssembly

    @property
    def name(self):
        """The name of the model."""
        return self._model.name

    @property
    def historyOutputRequests(self):
        """Shortcut to access the history output requests of the model."""
        return self._model.historyOutputRequests

    @property
    def fieldOutputRequests(self):
        """Shortcut to access the field output requests of the model."""
        return self._model.fieldOutputRequests


class AbaqusOutputVariable:
    def __init__(self, name: str, abaqus_name: str, unit: str, description: str):
        self.name = name
        self.abaqus_name = abaqus_name
        self.unit = unit
        self.description = description

    def to_dict(self):
        return {"name": self.name, "abaqus_name": self.abaqus_name, "unit": self.unit, "description": self.description}
