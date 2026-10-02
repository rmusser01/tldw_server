"""Repeated reloads must release apps while registered callbacks remain live."""

from __future__ import annotations

import gc
import weakref

import pytest
from fastapi import FastAPI
from fastapi.dependencies import models

from tldw_Server_API.tests.helpers.app_main_state import app_main_isolated, import_app_main, reload_app_main


@pytest.mark.unit
@pytest.mark.parametrize("ultra_minimal", [True, False])
def test_registered_main_callbacks_release_the_retired_app(
    monkeypatch: pytest.MonkeyPatch,
    ultra_minimal: bool,
) -> None:
    """Hold first-profile consumers, then release a subsequent isolated app."""
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "1" if ultra_minimal else "0")
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("TEST_MODE", "true")
    callbacks = (
        "root",
        "favicon",
        "serve_setup_page",
        "metrics",
        "api_metrics",
        "health_check",
        "internal_readiness_check",
        "readiness_check",
        "readiness_alias",
        "_set_diagnostics_no_store",
    )
    original = import_app_main()
    with app_main_isolated():
        # Hold the first import in this profile as a control for one-time
        # consumers; measure a later reload while those consumers remain live.
        baseline = reload_app_main()
        retired = reload_app_main()
        app_ref = weakref.ref(retired.app)
        endpoints = [getattr(retired, name) for name in callbacks]
        consumer = FastAPI()
        for index, endpoint in enumerate(endpoints[:-1]):
            consumer.add_api_route(f"/retained/{index}", endpoint)
        # Real classification is the native retaining boundary; no cache clearing.
        for endpoint in endpoints:
            assert models._is_coroutine_callable(endpoint) is True
            assert models._is_gen_callable(endpoint) is False
            assert models._is_async_gen_callable(endpoint) is False
    # A later real reload replaces process-global logging hooks. This app is
    # retired from both the import namespace and the active logger configuration.
    with app_main_isolated():
        replacement = reload_app_main()
        assert replacement is not original
        assert replacement is not baseline
    del retired
    gc.collect()
    assert app_ref() is None, "registered control-plane callbacks retained the retired app"


@pytest.mark.asyncio
async def test_api_metrics_preserves_its_existing_call_counter() -> None:
    """Moving the callback must preserve existing monitoring series."""
    from tldw_Server_API.app.api.v1.endpoints.control_plane import api_metrics
    from tldw_Server_API.app.core.Metrics import get_metrics_registry

    registry = get_metrics_registry()
    name = "tldw_Server_API.app.main.api_metrics_calls_total"
    before = registry.get_metric_stats(name).get("sum", 0)
    await api_metrics()
    assert registry.get_metric_stats(name).get("sum", 0) == before + 1
