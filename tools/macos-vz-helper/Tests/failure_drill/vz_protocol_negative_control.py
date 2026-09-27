"""Explicit pytest plugin: change only the fixture's guest handshake version."""

import pytest


@pytest.fixture(autouse=True)
def permit_valid_protocol(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Leave the Swift validation gate intact and require actual guest execution."""
    monkeypatch.setattr(request.module, "_INJECT_MISMATCH", False)
