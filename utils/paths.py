"""
Configuration module for managing project directory paths.

With a lot of data input and output, and many figures being generated, it's convenient to have easy-to-use global
path variables for commonly used directories. These paths are defined here, and can be imported in any script within
the project.

Following directories are defined (and created if missing):

* Raw data directory: ``DIR_DATA_RAW``
* Input data directory: ``DIR_DATA_IN``
* Output data directory: ``DIR_DATA_OUT``
* Figure output directory: ``DIR_FIGURE_OUT``
* Table output directory: ``DIR_TABLE_OUT``

:author: Wouter Vermeulen
:date: 2026-01-18
"""

from pathlib import Path
from sys import argv
from .prettyprint import color_text, tprint

# Determine project root and caller file location dynamically
# Assuming this file is located at __PROJECT_ROOT/utils/paths.py (using __name__ isn't reliable)
__PROJECT_ROOT = Path(__file__).resolve().parent.parent
__CALLER = Path(argv[0]).resolve()
__SAFE_CALLER_NAME = __CALLER.name.replace(".", "_")
__CURRENT_CALLER_PARENT = __CALLER.parent


# If the caller is from Abaqus CAE, the caller path is not reliable, so read from the config file instead
# (the config file should just have been created by a previous run from a non-Abaqus caller)
if not ("DassaultSystems/Abaqus/CAE" in str(__CURRENT_CALLER_PARENT).replace("\\", "/") or
        "SIMULIA/EstProducts" in str(__CURRENT_CALLER_PARENT).replace("\\", "/")):
    with open(__PROJECT_ROOT / "source" / "config" / "paths_config.txt", "w") as f:
        f.write(f"Project root directory: {__PROJECT_ROOT}\n")
        f.write(f"Caller file: {__CALLER}\n")
        f.write(f"Safe caller name: {__SAFE_CALLER_NAME}\n")
        f.write(f"Caller parent directory: {__CURRENT_CALLER_PARENT}\n")
else:
    with open(__PROJECT_ROOT / "source" / "config" / "paths_config.txt", "r") as f:
        __PROJECT_ROOT = Path(f.readline().split(": ", 1)[1].strip())
        __CALLER = Path(f.readline().split(": ", 1)[1].strip())
        __SAFE_CALLER_NAME = Path(f.readline().split(": ", 1)[1].strip())
        __CURRENT_CALLER_PARENT = Path(f.readline().split(": ", 1)[1].strip())

# Data directories (always stable filepaths)
DIR_DATA_RAW = __PROJECT_ROOT / "data" / "raw"
"""Raw data directory (should be used as read-only). Path is fixed."""

DIR_DATA_IN = __PROJECT_ROOT / "data" / "input"
"""Input data directory (for cleaned input data). Path is fixed."""

DIR_DATA_OUT = __PROJECT_ROOT / "data" / "output"
"""Output data directory (for processed or intermediary data). Path is fixed."""

DIR_MODELS = __PROJECT_ROOT / "models"
"""Models directory (for storing created *.cae or *.odb files). Path is fixed."""


# Figures and tables directories (relative to file generating them)
if __CURRENT_CALLER_PARENT.name == "Figures":
    # Caller inside a Figures directory (for a Presentation or Manuscript Chapter)
    DIR_FIGURE_OUT = __CURRENT_CALLER_PARENT
    """Figure output directory (for saving generated figures). Path depends on caller location. Is ``None`` if the 
    caller is inside a ``/Tables`` directory."""
    DIR_TABLE_OUT = None
    """Table output directory (for saving generated tables). Path depends on caller location. Is ``None`` if the 
    caller is inside a ``/Figures`` directory."""
elif __CURRENT_CALLER_PARENT.name == "Tables":
    # Caller inside a Tables directory (for a Presentation or Manuscript Chapter)
    DIR_FIGURE_OUT = None
    """Figure output directory (for saving generated figures). Path depends on caller location. Is ``None`` if the 
    caller is inside a ``/Tables`` directory."""
    DIR_TABLE_OUT = __CURRENT_CALLER_PARENT
    """Table output directory (for saving generated tables). Path depends on caller location. Is ``None`` if the
    caller is inside a ``/Figures`` directory."""
else:
    # Caller outside Figures or Tables directories (e.g., main analysis scripts)
    DIR_FIGURE_OUT = __CURRENT_CALLER_PARENT / __SAFE_CALLER_NAME / "Figures"
    """Figure output directory (for saving generated figures). Path depends on caller location. Is ``None`` if the 
    caller is inside a ``/Tables`` directory."""
    DIR_TABLE_OUT = __CURRENT_CALLER_PARENT / __SAFE_CALLER_NAME / "Tables"
    """Table output directory (for saving generated tables). Path depends on caller location. Is ``None`` if the 
    caller is inside a ``/Figures`` directory."""


# Create directories automatically if missing
for d in [DIR_DATA_RAW, DIR_DATA_IN, DIR_DATA_OUT, DIR_MODELS, DIR_FIGURE_OUT, DIR_TABLE_OUT]:
    d.mkdir(parents=True, exist_ok=True)


# Informative printout
tprint(color_text(f"[INFO] Paths module initialized. Data and output directories are set up.", "green"))
tprint(color_text(f"\tRaw data directory: {DIR_DATA_RAW}", "green"))
tprint(color_text(f"\tInput data directory: {DIR_DATA_IN}", "green"))
tprint(color_text(f"\tOutput data directory: {DIR_DATA_OUT}", "green"))
if DIR_FIGURE_OUT is not None:
    tprint(color_text(f"\tFigure output directory: {DIR_FIGURE_OUT}", "green"))
if DIR_TABLE_OUT is not None:
    tprint(color_text(f"\tTable output directory: {DIR_TABLE_OUT}", "green"))
