"""Explicit pytest plugin: launch the preserved original guest agent."""

import pytest


@pytest.fixture(autouse=True)
def permit_original_agent(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Change only the guest challenge for this one negative-control case."""
    monkeypatch.setattr(request.module, "_START_AGENT", True)
