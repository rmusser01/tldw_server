"""Explicit pytest plugin: restore only the guest's advertised workspace root."""

import pytest


@pytest.fixture(autouse=True)
def permit_valid_workspace(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Keep real admission enabled and require actual supported-metadata execution."""
    monkeypatch.setattr(request.module, "_INJECT_MISMATCH", False)
