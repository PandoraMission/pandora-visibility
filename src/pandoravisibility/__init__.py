import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

# Find Version Number
import importlib.metadata
__version__ = importlib.metadata.version("pandoravisibility")
version = __version__

from .utils import (
    analyze_yearly_visibility,
    calculate_visibility_statistics,
    find_continuous_periods,
    plot_yearly_visibility,
)
from .visibility import Visibility  # noqa: E402, F401

__all__ = [
    "Visibility",
    "analyze_yearly_visibility",
    "find_continuous_periods",
    "calculate_visibility_statistics",
    "plot_yearly_visibility",
]
