from pathlib import Path
from json import load, dump

import numpy as np
from numpy.typing import NDArray

from utils.units import *

from dataclasses import dataclass, field


def load_with_quantities(d: dict | list) -> dict | list:
    """Recursively convert all dicts with "Value" and "Unit" keys to pint Quantities."""
    if isinstance(d, dict):
        if "Value" in d and "Unit" in d:
            return to_ureg(d)
        else:
            return {k: load_with_quantities(v) for k, v in d.items()}
    elif isinstance(d, list):
        if all(isinstance(el, dict) and "Value" in el and "Unit" in el for el in d):
            return quantity_array([to_ureg(el) for el in d])
        else:
            return [load_with_quantities(el) for el in d]
    else:
        return d


@dataclass
class Data:
    """Collected time-series data from an ODB history region, with column metadata."""
    columns: dict[str, NDArray] = field(default_factory=dict)   # name -> values
    metadata: dict[str, dict] = field(default_factory=dict)         # name -> {Unit, Description}
    __column_length: int = 0

    def add_column(self, values: NDArray | None, name: str, unit: str | None, description: str, **kwargs):
        """
        Add a column of data with its metadata. The metadata dict for this column will contain the keys "Name",
        "Unit", and "Description", as well as any additional kwargs provided.

        :param name: the name of the column (e.g. "Displacement", "Force", ...)
        :param values: the values of the column as a numpy array (e.g. [0, 0.1, 0.2, ...])
        :param unit: the unit of the column as a pint unit string, or None if dimensionless
        :param description: the description of the column (e.g. "Displacement of the center node in the loading direction", "Reaction force at the supports", ...)
        :param kwargs: any additional metadata to add for this column (e.g. 'abaqus_var_name' to indicate the abbreviation used in abaqus, ...)
        :return:
        """
        if values is None:
            values = np.array([])

        if self.__column_length == 0:
            self.__column_length = len(values)
        elif len(values) != self.__column_length:
            raise ValueError(f"All columns must have the same length. Expected {self.__column_length}, got {len(values)} for column '{name}'.")

        self.columns[name] = values
        self.metadata[name] = {"Name": name, "Unit": unit, "Description": description, **kwargs}

    def extend(self, other: "Data"):
        """Append another Data's rows (time axis) to this one."""
        for name, values in other.columns.items():
            if name in self.columns:
                self.columns[name] = np.concatenate([self.columns[name], values])
                if self.metadata[name] != other.metadata[name]:
                    tprint(f"Warning: metadata for column '{name}' differs between the two Data objects. Keeping the existing metadata.")
                self.__column_length += other.__column_length
            else:
                self.columns[name] = values
                self.metadata[name] = other.metadata[name]

                if self.__column_length == 0:
                    self.__column_length = other.__column_length

        for name in self.metadata.keys():
            if len(self.columns[name]) != self.__column_length:
                raise ValueError(f"After extending, all columns must have the same length. Expected {self.__column_length}, got {len(self.columns[name])} for column '{name}'.")

    @property
    def data(self) -> np.ndarray:
        """Return (n_columns, n_timepoints) array, column-ordered to match self.columns."""
        return np.array(list(self.columns.values()))

    @property
    def column_metadata(self) -> list[dict]:
        """Return the list of column metadata dicts, in the same order as self.data."""
        return list(self.metadata.values())

    @property
    def column_length(self) -> int:
        """Return the length of the columns (number of time points)."""
        return self.__column_length


class DataFile:
    def __init__(self, path: str | Path):
        self.path = path if isinstance(path, Path) else Path(path)
        self.data_path = self._data_path()
        self.metadata_path = self._metadata_path()
        self.data = None
        self.metadata = None

    def _data_path(self):
        return self.path.parent / f"{self.path.name}_data.csv"

    def _metadata_path(self):
        return self.path.parent / f"{self.path.name}_meta.json"

    def load(self):
        """
        Load the data and metadata and return a dict of str: array where the data has its units added (using pint).
        The metadata is available afterward as a dict in the ``metadata`` attribute of this class. The raw data
        (without units) is available as a numpy array in the ``data`` attribute of this class.
        :return:
        """
        self.data = np.loadtxt(self.data_path, delimiter=";", unpack=True)
        self.metadata = load_with_quantities(load(open(self.metadata_path, "r", encoding="utf-8")))

        data = {}

        for i, col in enumerate(self.metadata["Columns"]):
            mult = 1
            if "Unit" in col and col["Unit"] is not None:
                if col["Unit"] == "%":
                    unit = ureg.dimensionless
                    mult = 0.01
                elif col["Unit"] == "um/m":
                    unit = ureg.dimensionless
                    mult = 1e-6
                elif col["Unit"] != "":
                    unit = ureg.Unit(col["Unit"])
                else:
                    unit = ureg.dimensionless
            else:
                unit = ureg.dimensionless

            data[col["Name"]] = np.array(self.data[i] * mult) * unit

        return data

    def save(self, data, metadata):
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w') as f:
            np.savetxt(f, data, delimiter=";")
        with open(self.metadata_path, 'w', encoding="utf-8") as f:
            dump(metadata, f, indent=4, ensure_ascii=False)

    @property
    def identifier(self):
        if "Sample data" in self.metadata and "Sample identifier" in self.metadata["Sample data"]:
            return self.metadata["Sample data"]["Sample identifier"]
        elif "Model type" in self.metadata and "Job name" in self.metadata["Model type"]:
            return self.metadata["Model type"]["Job name"]
        raise ValueError("No identifier found in metadata. Expected 'Sample data' -> 'Sample identifier' or 'Model type' -> 'Job name'.")

    @property
    def width(self):
        if "Sample type" in self.metadata and "FRP Sample" in self.metadata["Sample type"]:
            mult = 2 if "lateral" in self.metadata["Sample type"]["FRP Sample"]["Symmetry"] else 1
            return self.metadata["Sample type"]["FRP Sample"]["Width"] * mult
        else:
            return self.metadata["Sample data"]["Sample type"]["Cross-section width"]

    @property
    def thickness(self):
        if "Sample type" in self.metadata and "FRP Sample" in self.metadata["Sample type"]:
            return self.metadata["Sample type"]["FRP Sample"]["Thickness"]
        else:
            return self.metadata["Sample data"]["Sample type"]["Cross-section height"]

    @property
    def L0(self):
        if "Sample type" in self.metadata and "FRP Sample" in self.metadata["Sample type"]:
            mult = 2 if "longitudinal" in self.metadata["Sample type"]["FRP Sample"]["Symmetry"] else 1
            return self.metadata["Sample type"]["FRP Sample"]["L0"] * mult
        else:
            return self.metadata["Sample data"]["Sample type"]["L0"]

    @property
    def mesh_size(self):
        return self.metadata["Sample data"]["Mesh"]["Size"]

    @property
    def diameter(self):
        return self.metadata["Sample data"]["Sample type"]["Diameter"]


class AbaqusRptFile:
    def __init__(self, path: str | Path):
        self.path = path if isinstance(path, Path) else Path(path)
        self.data = None

    def load(self, first_data_line: int = 3, column_prepend: list = [], column_names: list = None, transpose: bool = False):
        with open(self.path, "r") as f:
            lines = f.readlines()

        if column_names:
            names = column_names
        else:
            names = column_prepend
            for el in lines[1].split("  "):
                if el != "" and el != "\n":
                    names.append(el.strip())

        data = []
        col_num = len(names)
        col_num_current = 0
        for line in lines[first_data_line:]:
            l_data = []
            el = line.split(" ")
            for e in el:
                e = e.strip()
                if e == "NoValue":
                    l_data.append(np.nan)
                elif e != "" and e != "\n":
                    l_data.append(float(e))
            if l_data != []:
                if col_num_current == 0:
                    col_num_current = len(l_data)
                    data.append(l_data)
                    if col_num_current == col_num:
                        col_num_current = 0
                else:
                    data[-1].extend(l_data)
                    col_num_current += len(l_data)
                    if col_num_current == col_num:
                        col_num_current = 0
                    elif col_num_current > col_num:
                        raise ValueError(f"Too many columns in line. Expected {col_num}, got {col_num_current}.")

        if transpose:
            self.data = np.array(data).T
        else:
            self.data = np.array(data)
        data = {}
        for i, name in enumerate(names):
            data[name] = self.data[i]
        return data

    def convert_to_datafile(self, metadata: dict, datafile_path: str | Path, **kwargs):
        """
        Convert the Abaqus RPT file to a DataFile object and save it to the specified path.

        :param metadata: dict containing the metadata for the DataFile
        :param datafile_path: path to save the DataFile
        :return: None
        """
        data = self.load(**kwargs)
        columns = list(data.keys())
        metadata["Columns"] = [{"Name": name, "Unit": None, "Description": ""} for name in columns]
        data_raw = list(data.values())
        datafile = DataFile(datafile_path)
        datafile.save(data_raw, metadata)


def to_ureg(qty: dict):
    """
    Convert a quantity dict (with "Value" and "Unit" keys) to a pint Quantity object.
    :param qty: dict with "Value" and "Unit" keys
    :return: pint Quantity object
    """
    value = qty["Value"]
    if qty["Unit"] is None or qty["Unit"] == "":
        unit = ureg.dimensionless
    else:
        unit = ureg.Unit(qty["Unit"])
    return value * unit
