"""
Wrapper around pint to automatically convert to preferred units for certain dimensions after any operation.
Ensures consistent units throughout the codebase and simplifies unit handling by automatically converting to preferred
units, for example when using the Abaqus scripting interface where units are not explicitly defined.

@author: Wouter Vermeulen
@date 2026-03-06
"""
import numpy as np
import pint

from utils import tprint

# Define preferred units for specific dimensions - can be expanded and modified to fit the needs of the project.
# The keys are the dimension strings as defined by pint, and the values are the preferred units for those dimensions.
# Note: some consistency is checked (e.g. magnitudes are correct, e.g. MPa = N/mm^2), but user is responsible for
# ensuring that the preferred units are consistent with each other and with the dimensionality.
PREFERRED_UNITS = {
    "[length]": "mm",
    "[force]": "N",
    "[pressure]": "MPa",
    "[energy]": "mJ",
    "[mass]": "Mg",
    "[time]": "s",
    "[temperature]": "°K"
}

# Define preferred derived units for specific combinations of base units, to ensure that they are automatically
# converted to more readable forms. Note: user responsible for ensuring consistency.
_PREFERRED_DERIVED_UNITS = {
    "mm**-1 * Mg**1 * s**-2": "MPa"
}


###
# Unit handling with automatic conversion to preferred units
# Do not touch the code below unless you know what you're doing
###


def create_unit_from_container(container: dict[str, int]) -> str:
    """Helper function to create a unit string from a dictionary of base quantities and their powers."""
    unit_str = ""
    container = dict(sorted(container.items()))
    for base_qty, power in container.items():
        unit_str += f"{PREFERRED_UNITS[base_qty]}**{power} * "
    return unit_str[:-3]  # Remove trailing " * "


ureg = pint.UnitRegistry(autoconvert_to_preferred=True)
ureg.define("micro- = 1e-6 = µ-")  # Visually more pleasant micro prefix


# Check preferred units are consistent with dimensionality
for dim, unit in PREFERRED_UNITS.items():
    if ureg.get_dimensionality(unit) != ureg.get_dimensionality(dim):
        raise ValueError(f"{unit} does not match dimensionality {dim}")

# Check most important preferred units are consistent with each other (e.g., pressure = force/area)
_PREFERRED_UNIT_CHECKS = {
    "[pressure]": {"[force]": 1, "[length]": -2},
    "[energy]": {"[force]": 1, "[length]": 1},
    "[force]": {"[mass]": 1, "[length]": 1, "[time]": -2}
}
for dim, check_dims in _PREFERRED_UNIT_CHECKS.items():
    initial = 1 * ureg.Unit(create_unit_from_container({dim: 1}))
    other = 1 * ureg.Unit(create_unit_from_container(check_dims))

    if initial.dimensionality != other.dimensionality:
        raise ValueError(f"Preferred unit {PREFERRED_UNITS[dim]} for dimension {dim} is not consistent with the "
                         f"preferred units of its base dimensions {check_dims}.")
    if initial.to_base_units().m != other.to_base_units().m:
        raise ValueError(f"Preferred unit {PREFERRED_UNITS[dim]} for dimension {dim} is not consistent with the "
                         f"preferred units of its base dimensions {check_dims} in terms of magnitude.")


# Create a mapping of the preferred units for each dimension, to allow speedup of automatic conversion
# Note: not optimized
PREFERRED_UNITS_LONG = []
for dim, unit in PREFERRED_UNITS.items():
    PREFERRED_UNITS_LONG.append(ureg.Unit(unit))


class PreferredQuantity(ureg.Quantity):
    """
    A subclass of pint.Quantity that automatically converts to preferred units for
    certain dimensions after any operation.
    """
    def _convert_in_place(self):
        if self.dimensionless:
            return  # No conversion needed for dimensionless quantities
        if self.units in PREFERRED_UNITS_LONG:
            return  # Already in a preferred unit

        # Convert to the preferred unit for each base quantity in the dimensionality
        dest_unit = create_unit_from_container(self.dimensionality)

        # Convert to the destination unit for some derived ones
        if dest_unit in _PREFERRED_DERIVED_UNITS:
            dest_unit = _PREFERRED_DERIVED_UNITS[dest_unit]

        self.ito(dest_unit)

    def _wrap(self, result: "pint.Quantity") -> "PreferredQuantity":
        """"Wrap Pint result into PreferredQuantity and convert."""
        q = PreferredQuantity(result.magnitude, result.units)
        q._convert_in_place()
        return q

    def __mul__(self, other):
        return self._wrap(super().__mul__(other))

    def __rmul__(self, other):
        return self._wrap(super().__rmul__(other))

    def __truediv__(self, other):
        return self._wrap(super().__truediv__(other))

    def __rtruediv__(self, other):
        return self._wrap(super().__rtruediv__(other))

    def __add__(self, other):
        return self._wrap(super().__add__(other))

    def __radd__(self, other):
        return self._wrap(super().__radd__(other))

    def __sub__(self, other):
        return self._wrap(super().__sub__(other))

    def __rsub__(self, other):
        return self._wrap(super().__rsub__(other))

    def __pow__(self, other):
        return self._wrap(super().__pow__(other))

    def __iter__(self):
        """Iterate over the Quantity objects.
        Explicit definition to avoid type warnings."""
        return super().__iter__()

    def __len__(self):
        """Return the length of the array magnitude.
        Explicit definition to avoid type warnings."""
        return super().__len__()

    def __getitem__(self, item):
        return self._wrap(super().__getitem__(item))

    @staticmethod
    def zero(units: "ureg.Unit") -> "PreferredQuantity":
        """Create a zero quantity with the specified units."""
        return PreferredQuantity(0, units)

    @staticmethod
    def dim_str(dimension: str | None) -> str | None:
        """Get the preferred unit string for a given dimension."""
        if dimension is None:
            return None

        if dimension in PREFERRED_UNITS:
            return PREFERRED_UNITS[dimension]
        raise ValueError(f"Dimension {dimension} not recognised.")

    @staticmethod
    def dim(dimension: str) -> "ureg.Unit":
        """Get the preferred unit for a given dimension."""
        return ureg.Unit(PreferredQuantity.dim_str(dimension))

    @staticmethod
    def pref_dim(unit: str) -> str:
        """Get the preferred unit string for a given unit (e.g. 'mm' if input was 'm')."""
        return f"{PreferredQuantity(1 * ureg.Unit(unit)).units:~P}"

    @property
    def dict(self):
        """Return a dictionary representation of the quantity, with magnitude and preferred unit string."""
        return {"Value": float(self.m), "Unit": f"{self.units:~P}"}


class PreferredQuantityArray(PreferredQuantity, np.ndarray):

    def __iter__(self):
        """Iterate over the Quantity objects.
        Explicit definition to avoid type warnings."""
        return super().__iter__()

    def __len__(self):
        """Return the length of the array magnitude.
        Explicit definition to avoid type warnings."""
        return super().__len__()

    def __getitem__(self, item):
        return self._wrap(super().__getitem__(item))


def quantity_array(lst: list[PreferredQuantity]) -> PreferredQuantityArray:
    """
    Convert a list of PreferredQuantities to a single PreferredQuantity with an array magnitude.
    All quantities in the list must have the same units, which will be the units of the resulting quantity.
    """
    if lst is None or len(lst) == 0:
        raise ValueError("Input list cannot be empty.")
    if not all(isinstance(qty, PreferredQuantity) or (qty is None) for qty in lst):
        raise ValueError("All elements in the input list must be PreferredQuantities.")

    units = next(qty.units for qty in lst if qty is not None)
    if not all((qty is None) or qty.units == units for qty in lst):
        raise ValueError("All quantities in the input list must have the same units.")
    magnitudes = [qty.m if qty is not None else None for qty in lst]
    return Quantity(magnitudes, units)


# Ensure that all quantities created in the codebase are PreferredQuantities, so that they automatically
# convert to preferred units.
ureg.Quantity = PreferredQuantity
Q_ = PreferredQuantity
Quantity = PreferredQuantity
QuantityArray = PreferredQuantityArray
