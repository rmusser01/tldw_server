from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_personalization_consolidation():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_personalization_consolidation", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_personalization_consolidation")


@pytest.mark.asyncio
async def test_shutdown_personalization_consolidation_stops_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_personalization = _import_shutdown_personalization_consolidation()
    observed = {"stopped": False}

    class _FakeService:
        async def stop(self) -> None:
            observed["stopped"] = True

    monkeypatch.setattr(
        shutdown_personalization,
        "_get_consolidation_service",
        lambda: _FakeService(),
    )

    await shutdown_personalization.shutdown_personalization_consolidation(
        guard_exceptions=(RuntimeError,),
    )

    assert observed["stopped"] is True


@pytest.mark.asyncio
async def test_shutdown_personalization_consolidation_swallows_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_personalization = _import_shutdown_personalization_consolidation()
    warning_messages: list[str] = []

    class _FakeService:
        async def stop(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        shutdown_personalization,
        "_get_consolidation_service",
        lambda: _FakeService(),
    )
    monkeypatch.setattr(
        shutdown_personalization.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )

    await shutdown_personalization.shutdown_personalization_consolidation(
        guard_exceptions=(RuntimeError,),
    )

    assert any("Personalization consolidation shutdown failed" in message for message in warning_messages)
