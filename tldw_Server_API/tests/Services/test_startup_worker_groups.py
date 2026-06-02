from __future__ import annotations

import importlib
import sys

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

pytestmark = pytest.mark.unit


def _import_startup_worker_groups():
    sys.modules.pop("tldw_Server_API.app.services.startup_worker_groups", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_worker_groups")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_collect_startup_worker_specs_uses_declarative_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_worker_groups()
    provider_calls: list[str] = []

    def _provider(name: str):
        def _collect(_context: WorkerLifecycleContext):
            provider_calls.append(name)
            return ()

        return _collect

    providers = (
        _provider("primary"),
        _provider("study"),
        _provider("content"),
        _provider("sidecar"),
        _provider("notifications"),
        _provider("cleanup"),
        _provider("compactor"),
        _provider("claims"),
        _provider("usage"),
        _provider("llm-usage"),
        _provider("runtime"),
        _provider("optional"),
        _provider("auxiliary"),
        _provider("infra"),
        _provider("maintenance"),
        _provider("recurring"),
    )
    monkeypatch.setattr(
        startup_groups,
        "startup_worker_spec_providers",
        lambda: providers,
    )

    specs = startup_groups.collect_startup_worker_specs(_context())

    assert specs == ()
    assert provider_calls == [
        "primary",
        "study",
        "content",
        "sidecar",
        "notifications",
        "cleanup",
        "compactor",
        "claims",
        "usage",
        "llm-usage",
        "runtime",
        "optional",
        "auxiliary",
        "infra",
        "maintenance",
        "recurring",
    ]


def test_collect_startup_worker_specs_accepts_real_provider_graph() -> None:
    startup_groups = _import_startup_worker_groups()

    specs = startup_groups.collect_startup_worker_specs(_context())
    spec_names = {spec.name for spec in specs}

    assert len(specs) == len(spec_names)
    assert {
        "core_jobs_task",
        "claims_rebuild",
        "jobs_metrics_task",
        "connectors_sync_sched_task",
    }.issubset(spec_names)


def test_startup_worker_groups_no_longer_exposes_legacy_group_start_api() -> None:
    startup_groups = _import_startup_worker_groups()

    assert not hasattr(startup_groups, "StartupWorkerGroupHandles")
    assert not hasattr(startup_groups, "start_worker_groups")
