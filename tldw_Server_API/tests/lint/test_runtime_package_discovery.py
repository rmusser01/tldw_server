"""Keep runtime imports available in regular-package distributions."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib
from setuptools import find_packages

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def packaged_modules() -> set[str]:
    """Use the actual configured regular-package discovery across all roots."""
    root = Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)["tool"]["setuptools"]["packages"]["find"]
    return {
        package
        for directory in config["where"]
        for package in find_packages(
            where=str(root / directory),
            include=config["include"],
            exclude=config["exclude"],
        )
    }


@pytest.mark.parametrize(
    "package_suffix",
    [
        "Chunking.utils",
        "Embeddings.services",
        "Explainer",
        "Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming",
        "Ingestion_Media_Processing.Email",
        "Ingestion_Media_Processing.MediaWiki",
        "Ingestion_Media_Processing.OCR.backends",
        "Ingestion_Media_Processing.PDF",
        "Ingestion_Media_Processing.VLM.backends",
        "Logging",
        "MCP_unified.modules.implementations",
        "MCP_unified.monitoring",
        "MCP_unified.security",
        "MCP_unified.transport",
        "Prototype_Workspaces",
        "Sandbox.runners",
        "Scheduler.handlers",
        "Telegram",
        "Tools",
    ],
)
def test_runtime_package_is_included_by_project_package_discovery(
    package_suffix: str, packaged_modules: set[str]
) -> None:
    """A wheel must include modules imported by its runtime consumers."""
    expected = f"tldw_Server_API.app.core.{package_suffix}"
    # Pytest assertion verifies the actual configured distribution inventory.
    assert expected in packaged_modules  # nosec B101
