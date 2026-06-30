from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_owned_job_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_owned_job_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_owned_job_pollers")


def test_prepare_startup_owned_job_pollers_returns_list_and_publishes_inventory() -> None:
    startup_owned = _import_startup_owned_job_pollers()
    app = SimpleNamespace()
    recorded_calls: list[tuple[object, list[object]]] = []

    def _fake_publish_shutdown_job_poller_inventory(seen_app, seen_handles) -> None:
        recorded_calls.append((seen_app, seen_handles))

    handles = startup_owned.prepare_startup_owned_job_pollers(
        app=app,
        publish_shutdown_job_poller_inventory=_fake_publish_shutdown_job_poller_inventory,
    )

    assert handles == []
    assert recorded_calls == [(app, handles)]
