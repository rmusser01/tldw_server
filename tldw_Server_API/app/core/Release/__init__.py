"""Verified release metadata and paired-distribution completeness boundary."""

from collections import Counter
from collections.abc import Sequence

from .manifest import Artifact, ManifestError

PAIRED_BUNDLE_FILES = (
    "README.md",
    "compose.yaml",
    "start.sh",
    "stop.sh",
    "status.sh",
    "start.ps1",
    "stop.ps1",
    "status.ps1",
)


def require_paired_inventory(artifacts: Sequence[Artifact]) -> None:
    """Require one signed helper record per required path on the selected platform."""
    paths = Counter(artifact.path for artifact in artifacts if artifact.kind == "file")
    if any(paths[path] != 1 for path in PAIRED_BUNDLE_FILES):
        raise ManifestError("paired bundle lacks a unique required file record")
