"""Explicit pytest plugin: let the real fault guest complete readiness and exec."""

import pytest


@pytest.fixture(autouse=True)
def permit_real_ready(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Change only this test's challenge, not the runtime readiness gate."""
    monkeypatch.setattr(request.module, "_WITHHOLD_READY", False)
