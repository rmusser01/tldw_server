"""Explicit pytest plugin: bypass only the runner's create-time capability gate."""

from collections.abc import Mapping
from typing import Optional

import pytest


@pytest.fixture(autouse=True)
def remove_only_runner_create_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live guest must actually execute for this control to be accepted."""
    from tldw_Server_API.app.core.Sandbox.runners import vz_linux_runner

    original = vz_linux_runner.classify_vz_linux_guest_agent

    def no_guard(details: Optional[Mapping[str, object]]) -> dict[str, object]:
        """Preserve real metadata while bypassing only compatibility classification."""
        return {**original(details), "compatibility": "compatible"}

    monkeypatch.setattr(vz_linux_runner, "classify_vz_linux_guest_agent", no_guard)
