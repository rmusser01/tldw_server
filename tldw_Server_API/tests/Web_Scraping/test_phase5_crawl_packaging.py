"""Crawl must ship as a regular package, not merely import from a source tree."""

import importlib
from pathlib import Path

import pytest
import tomllib
from setuptools import find_packages

pytestmark = pytest.mark.unit


def test_configured_package_discovery_includes_importable_crawl_checkpoint_modules():
    root = Path(__file__).resolve().parents[3]
    configuration = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]
    assert configuration["namespaces"] is False
    packages = {
        package
        for directory in configuration["where"]
        for package in find_packages(
            where=str(root / directory),
            include=configuration["include"],
            exclude=configuration["exclude"],
        )
    }
    name = "tldw_Server_API.app.core.Web_Scraping.crawl"
    assert name in packages
    package = importlib.import_module(name)
    assert Path(package.__file__).resolve() == root / "tldw_Server_API/app/core/Web_Scraping/crawl/__init__.py"
    for module in ("checkpoints", "models", "engine", "frontier", "discovery", "profiles"):
        imported = importlib.import_module(f"{name}.{module}")
        assert Path(imported.__file__).resolve() == Path(package.__file__).parent / f"{module}.py"
    assert importlib.import_module(f"{name}.checkpoints").CheckpointStore.__module__ == f"{name}.checkpoints"
