"""
This module sets up consistent global configurations for this project, including:

* ``paths``: form directory paths to manage data input/output and figure/table storage
* ``styles``: matplotlib settings such as color cycles, font styles, and figure sizes

:author: Wouter Vermeulen
:date: 2025-11-11
"""

__author__ = "Wouter Vermeulen"
__version__ = "1.1.0"
__date__ = "2026-01-18"

import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(rf"{BASE_DIR}/../site-packages/python3.10")

# Path handling
from utils.paths import *

# Style handling for plots
from utils.styles import *
from utils.styles import _preconfigure_matplotlib
_preconfigure_matplotlib()

from utils.prettyprint import *

# Unit handling
from utils.units import Q_, Quantity, ureg, quantity_array

from utils.data import DataFile





