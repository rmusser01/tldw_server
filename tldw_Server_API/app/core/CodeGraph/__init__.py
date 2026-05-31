"""Native CodeGraph foundation package."""

from .config import CodeGraphSettings
from .dependencies import DependencyHealth, probe_codegraph_dependencies

__all__ = [
    "CodeGraphSettings",
    "DependencyHealth",
    "probe_codegraph_dependencies",
]
