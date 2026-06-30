from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _context(settings: Any | None = None) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app="app",
        settings={} if settings is None else settings,
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _single_spec(module: Any) -> Any:
    [spec] = module.provide_usage_aggregator_worker_specs()
    return spec


@pytest.mark.parametrize(
    ("module_name", "provider_name", "loop_name", "spec_name"),
    [
        (
            "tldw_Server_API.app.services.usage_aggregator",
            "provide_usage_aggregator_worker_specs",
            "_aggregator_loop",
            "usage_aggregator",
        ),
        (
            "tldw_Server_API.app.services.llm_usage_aggregator",
            "provide_llm_usage_aggregator_worker_specs",
            "_aggregator_loop",
            "llm_usage_aggregator",
        ),
    ],
)
def test_usage_aggregator_worker_specs_match_legacy_worker_contract(
    module_name: str,
    provider_name: str,
    loop_name: str,
    spec_name: str,
) -> None:
    module = __import__(module_name, fromlist=[provider_name, loop_name])

    [spec] = getattr(module, provider_name)()

    assert spec.name == spec_name
    assert spec.task_name == spec_name
    assert spec.category == "usage"
    assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert spec.timeout_sec == 5.0
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None


def test_usage_aggregator_worker_spec_factory_delegates_to_existing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import usage_aggregator

    calls: list[object] = []
    monkeypatch.setattr(
        usage_aggregator,
        "_aggregator_loop",
        lambda stop_event: calls.append(stop_event) or "usage-awaitable",
    )

    spec = _single_spec(usage_aggregator)

    assert spec.factory is not None
    assert spec.factory(_context(), "usage-stop") == "usage-awaitable"
    assert calls == ["usage-stop"]


def test_llm_usage_aggregator_worker_spec_factory_delegates_to_existing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import llm_usage_aggregator

    calls: list[object] = []
    monkeypatch.setattr(
        llm_usage_aggregator,
        "_aggregator_loop",
        lambda stop_event: calls.append(stop_event) or "llm-usage-awaitable",
    )

    [spec] = llm_usage_aggregator.provide_llm_usage_aggregator_worker_specs()

    assert spec.factory is not None
    assert spec.factory(_context(), "llm-usage-stop") == "llm-usage-awaitable"
    assert calls == ["llm-usage-stop"]


def test_usage_aggregator_worker_spec_enabled_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import usage_aggregator

    monkeypatch.setattr(
        usage_aggregator,
        "get_settings",
        lambda: SimpleNamespace(USAGE_LOG_ENABLED=True),
    )
    assert _single_spec(usage_aggregator).enabled(_context()) is True

    monkeypatch.setattr(
        usage_aggregator,
        "get_settings",
        lambda: SimpleNamespace(USAGE_LOG_ENABLED=False),
    )
    assert _single_spec(usage_aggregator).enabled(_context()) is False


def test_llm_usage_aggregator_worker_spec_enabled_uses_settings(
) -> None:
    from tldw_Server_API.app.services import llm_usage_aggregator

    [enabled_spec] = llm_usage_aggregator.provide_llm_usage_aggregator_worker_specs()
    assert enabled_spec.enabled(
        _context(SimpleNamespace(LLM_USAGE_AGGREGATOR_ENABLED=True)),
    ) is True

    [disabled_spec] = llm_usage_aggregator.provide_llm_usage_aggregator_worker_specs()
    assert disabled_spec.enabled(
        _context(SimpleNamespace(LLM_USAGE_AGGREGATOR_ENABLED=False)),
    ) is False


def test_shutdown_usage_aggregators_direct_stop_module_is_removed() -> None:
    assert (
        importlib.util.find_spec("tldw_Server_API.app.services.shutdown_usage_aggregators")
        is None
    )


def test_post_worker_tail_no_longer_has_usage_aggregator_direct_stop_adapter() -> None:
    from tldw_Server_API.app.services import shutdown_post_worker_services as shutdown_services

    assert not hasattr(shutdown_services, "_stop_usage_aggregators")
