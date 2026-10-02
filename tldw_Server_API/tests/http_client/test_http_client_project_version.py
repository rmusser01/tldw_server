"""The User-Agent version falls back to pyproject.toml in an uninstalled source checkout."""

from __future__ import annotations

import importlib.metadata

import pytest

from tldw_Server_API.app.core import http_client


@pytest.mark.unit
def test_version_falls_back_to_pyproject_when_package_not_installed(monkeypatch):
    def not_installed(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("tldw-server")

    monkeypatch.delenv("TLDW_VERSION", raising=False)
    monkeypatch.setattr(http_client, "_CACHED_VERSION", None)
    monkeypatch.setattr(http_client._importlib_metadata, "version", not_installed)

    version = http_client._get_project_version()

    assert version and version[0].isdigit()
