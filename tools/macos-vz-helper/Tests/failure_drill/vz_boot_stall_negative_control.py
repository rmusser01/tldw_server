"""Explicit pytest plugin: continue through the same initramfs wrapper."""

import pytest


@pytest.fixture(autouse=True)
def continue_original_boot(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Change only the disposable initramfs challenge mode."""
    monkeypatch.setattr(request.module, "_STALL_BOOT", False)
