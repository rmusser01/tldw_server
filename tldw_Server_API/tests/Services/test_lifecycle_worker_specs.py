import pytest
from fastapi import FastAPI


async def _noop_callback() -> None:
    return None


def _worker_spec(**overrides: object):
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        ShutdownPhase,
        WorkerSpec,
    )

    values = {
        "name": "worker_a",
        "task_name": "worker-a-task",
        "category": "jobs",
        "phase": ShutdownPhase.JOB_POLLER_QUIESCE,
        "factory": lambda _context, _stop_event: _stop_event.wait(),
    }
    values.update(overrides)
    return WorkerSpec(**values)


def _context(*, route_enabled):
    from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=route_enabled,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


@pytest.mark.unit
def test_stop_event_worker_spec_builds_standard_stop_event_spec_and_delegates_directly() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        ShutdownPhase,
        WorkerStrategy,
        stop_event_worker_spec,
    )

    calls = []

    def _worker_service(stop_event):
        calls.append(stop_event)
        return "worker-awaitable"

    spec = stop_event_worker_spec(
        name="example_jobs_task",
        category="jobs",
        phase=ShutdownPhase.JOB_POLLER_QUIESCE,
        worker_service=_worker_service,
    )

    assert spec.name == "example_jobs_task"
    assert spec.task_name == "example_jobs_task"
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.timeout_sec == 5.0
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None
    assert spec.factory(_context(route_enabled=lambda *_args, **_kwargs: True), "stop-event") == "worker-awaitable"
    assert calls == ["stop-event"]


@pytest.mark.unit
def test_route_enabled_predicate_forwards_route_and_kwargs_after_env_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        route_enabled_predicate,
    )

    calls = []

    def _route_enabled(*args, **kwargs):
        calls.append((args, kwargs))
        return False

    predicate = route_enabled_predicate(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        default_stable=False,
    )

    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)

    assert predicate(_context(route_enabled=_route_enabled)) is False
    assert calls == []

    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")

    assert predicate(_context(route_enabled=_route_enabled)) is False
    assert calls == [
        (
            ("media-ingest-heavy-jobs",),
            {"default_stable": False},
        )
    ]


@pytest.mark.unit
def test_validate_spec_graph_rejects_duplicate_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    specs = [
        _worker_spec(name="duplicate_worker", task_name="worker-a-task"),
        _worker_spec(name="duplicate_worker", task_name="worker-b-task"),
    ]

    with pytest.raises(WorkerSpecValidationError, match="duplicate.*duplicate_worker"):
        validate_worker_spec_graph(specs)


@pytest.mark.unit
def test_validate_spec_graph_rejects_duplicate_diagnostic_names() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    specs = [
        _worker_spec(name="worker_a", diagnostic_name="shared_diagnostic"),
        _worker_spec(name="worker_b", diagnostic_name="shared_diagnostic"),
    ]

    with pytest.raises(WorkerSpecValidationError, match="diagnostic.*shared_diagnostic"):
        validate_worker_spec_graph(specs)


@pytest.mark.unit
def test_validate_spec_graph_rejects_diagnostic_name_colliding_with_worker_name() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    specs = [
        _worker_spec(name="worker_a", diagnostic_name="worker_b"),
        _worker_spec(name="worker_b"),
    ]

    with pytest.raises(WorkerSpecValidationError, match="diagnostic.*worker_b"):
        validate_worker_spec_graph(specs)


@pytest.mark.unit
def test_validate_spec_graph_rejects_unknown_dependency() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        ShutdownPhase,
        WorkerSpec,
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = WorkerSpec(
        name="child_task",
        task_name="child_task",
        category="jobs",
        phase=ShutdownPhase.JOB_POLLER_QUIESCE,
        depends_on=("missing_task",),
        factory=lambda _context, _stop_event: _stop_event.wait(),
    )

    with pytest.raises(WorkerSpecValidationError, match="child_task.*missing_task"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_dependency_cycles() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    specs = [
        _worker_spec(name="worker_a", depends_on=("worker_b",)),
        _worker_spec(name="worker_b", depends_on=("worker_c",)),
        _worker_spec(name="worker_c", depends_on=("worker_a",)),
    ]

    with pytest.raises(WorkerSpecValidationError, match="cycle.*worker_a.*worker_b.*worker_c"):
        validate_worker_spec_graph(specs)


@pytest.mark.unit
def test_validate_spec_graph_rejects_invalid_phases() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(phase="invalid_phase")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*invalid_phase"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_raw_phase_values() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(phase="job_poller_quiesce")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*ShutdownPhase"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_stop_event_task_without_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(factory=None)

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_non_callable_enabled() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(enabled="not-callable")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*enabled"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_non_callable_stop_event_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(factory="not-callable")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_stop_event_task_with_callback_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(shutdown_callback_factory=lambda _context: _noop_callback)

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*shutdown_callback_factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_callback_only_without_callback_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        WorkerStrategy,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(strategy=WorkerStrategy.CALLBACK_ONLY, factory=None)

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*shutdown_callback_factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_non_callable_callback_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        WorkerStrategy,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(
        strategy=WorkerStrategy.CALLBACK_ONLY,
        factory=None,
        shutdown_callback_factory="not-callable",
    )

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*shutdown_callback_factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_callback_only_with_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        WorkerStrategy,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(
        strategy=WorkerStrategy.CALLBACK_ONLY,
        shutdown_callback_factory=lambda _context: _noop_callback,
    )

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*factory"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_raw_strategy_values() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(strategy="stop_event_task")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*WorkerStrategy"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_raw_failure_policy_values() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(failure_policy="skip")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*WorkerFailurePolicy"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_invalid_failure_policy_values() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(failure_policy="invalid_policy")

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*WorkerFailurePolicy"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_spec_graph_rejects_non_tuple_dependencies() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(depends_on=["parent_worker"])

    with pytest.raises(WorkerSpecValidationError, match="worker_a.*depends_on.*tuple"):
        validate_worker_spec_graph([spec])


@pytest.mark.unit
def test_validate_enabled_worker_dependencies_rejects_disabled_dependency() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerSpecValidationError,
        validate_enabled_worker_dependencies,
        validate_worker_spec_graph,
    )

    graph = validate_worker_spec_graph(
        [
            _worker_spec(name="parent_worker"),
            _worker_spec(name="child_worker", depends_on=("parent_worker",)),
        ]
    )

    with pytest.raises(WorkerSpecValidationError, match="child_worker.*parent_worker"):
        validate_enabled_worker_dependencies(graph, {"child_worker"})


@pytest.mark.unit
def test_validate_spec_graph_accepts_callback_only_with_callback_factory() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        WorkerStrategy,
        validate_worker_spec_graph,
    )

    spec = _worker_spec(
        strategy=WorkerStrategy.CALLBACK_ONLY,
        factory=None,
        shutdown_callback_factory=lambda _context: _noop_callback,
    )

    graph = validate_worker_spec_graph([spec])

    assert graph.specs_by_name == {"worker_a": spec}


@pytest.mark.unit
def test_validate_spec_graph_returns_immutable_name_index() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import (
        validate_worker_spec_graph,
    )

    graph = validate_worker_spec_graph([_worker_spec()])

    with pytest.raises(TypeError):
        graph.specs_by_name["other_worker"] = _worker_spec(name="other_worker")  # type: ignore[index]


@pytest.mark.unit
def test_worker_failure_policy_exposes_abort() -> None:
    from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerFailurePolicy

    assert WorkerFailurePolicy.ABORT.value == "abort"
