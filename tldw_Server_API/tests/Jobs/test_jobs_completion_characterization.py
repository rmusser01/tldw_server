"""Characterize the existing facade before moving any completion ownership."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.Jobs.manager as manager_module
import tldw_Server_API.app.core.Jobs.operations.resource as resource_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs._completion_characterization_support import (
    CompletionConnection,
    CompletionCursor,
    assert_completed,
    completion_arguments,
    durable_state,
    independent_reader,
    observe_python_calls,
    record_completion_boundaries,
    record_completion_observers,
    rows,
    seed_job,
)

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
pytestmark = pytest.mark.integration
USE_SHARED_JOBS_POSTGRES = True


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def completion_manager(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    """Use the existing fixture-owned PostgreSQL route or a temporary SQLite DB."""
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "false")
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "false")
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1048576")
    if request.param == "postgres":
        dsn = request.getfixturevalue("jobs_pg_dsn")
        manager = JobManager(backend="postgres", db_url=dsn)
        _, db_name = request.getfixturevalue("isolated_test_environment")
        assert rows(manager, "SELECT current_database() AS name")[0]["name"] == db_name
        assert "pg_temp_db" not in request.fixturenames
    else:
        monkeypatch.delenv("JOBS_DB_URL", raising=False)
        manager = JobManager(tmp_path / "completion-characterization.db")
        monkeypatch.setenv("JOBS_DB_PATH", str(manager.db_path))
        assert "isolated_test_environment" not in request.fixturenames
    monkeypatch.setattr(manager_module, "_is_test_mode", lambda: False)
    return manager


def test_commit_time_replacement_keeps_original_callback(completion_manager, monkeypatch):
    """Late gauge rebinding must not change an already prepared callback."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(
        manager,
        monkeypatch,
        calls,
        on_commit=lambda: monkeypatch.setattr(
            manager, "_update_gauges", lambda **kwargs: calls.append("replacement_gauges")
        ),
    )
    assert manager.complete_job(job["id"], result={"ok": True}, **completion_arguments(job)) is True
    if manager.backend == "postgres":
        assert calls == [
            "commit",
            "duration",
            "completed",
            "original_gauges",
            "trace",
            "event",
            "cursor_exit",
            "context_exit",
            "close",
        ]
    else:
        assert calls == [
            "commit",
            "duration",
            "completed",
            "original_gauges",
            "trace",
            "event",
            "context_exit",
            "close",
        ]
    assert_completed(reader, job["id"])


def test_suppressed_gauge_binding_failure_preserves_metric_prefix(completion_manager, monkeypatch):
    """A binding failure leaves earlier metric callbacks but never queues trace."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(manager, monkeypatch, calls)
    original_getattribute = JobManager.__getattribute__

    def fail_binding(self, name):
        if self is manager and name == "_update_gauges":
            raise RuntimeError("gauge binding probe")
        return original_getattribute(self, name)

    monkeypatch.setattr(JobManager, "__getattribute__", fail_binding)
    assert manager.complete_job(job["id"], result={"ok": True}, **completion_arguments(job)) is True
    if manager.backend == "postgres":
        assert calls == ["commit", "duration", "completed", "event", "cursor_exit", "context_exit", "close"]
    else:
        assert calls == ["commit", "duration", "completed", "event", "context_exit", "close"]
    assert_completed(reader, job["id"])


def test_unsuppressed_observer_failure_keeps_committed_state(completion_manager, monkeypatch):
    """Outside-tuple observer errors propagate after commit and stop later observers."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(manager, monkeypatch, calls)

    def fail_duration(*args):
        calls.append("duration")
        raise AssertionError("duration observer probe")

    monkeypatch.setattr(manager_module, "observe_duration", fail_duration)
    with pytest.raises(AssertionError, match="duration observer probe"):
        manager.complete_job(job["id"], result={"ok": True}, **completion_arguments(job))
    if manager.backend == "postgres":
        assert calls == ["commit", "duration", "cursor_exit", "context_exit", "close"]
    else:
        assert calls == ["commit", "duration", "context_exit", "close"]
    assert reader.get_job(job["id"])["status"] == "completed"
    assert_completed(reader, job["id"])


@pytest.mark.parametrize("stored_uuid", [None, "", "  NONcanonical-UUID  "])
@pytest.mark.parametrize("matching", [True, False], ids=["matching", "stale"])
@pytest.mark.parametrize(
    "result_kind,serialization_error", [("valid", None), ("object", TypeError), ("circular", ValueError)]
)
def test_completion_identity_short_circuits_result_work(
    completion_manager, monkeypatch, stored_uuid, matching, result_kind, serialization_error
):
    """Terminal identity/replay classification must precede encryption/final encoding."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    rows(
        reader,
        "UPDATE jobs SET uuid=?, status='completed', completion_token=? WHERE id=?",
        (stored_uuid, "binding-token", job["id"]),
    )
    before = durable_state(reader, job["id"])
    transforms: list[Any] = []
    serializations: list[Any] = []
    calls: list[str] = []
    serialization_errors: list[type[Exception]] = []
    result = {"ignored": object() if result_kind == "object" else True}
    if result_kind == "circular":
        result["self"] = result
    original_dumps = manager_module.json.dumps
    if serialization_error is not None:
        # This same payload would fail final encoding if the early exit moved.
        with pytest.raises(serialization_error):
            original_dumps(result)

    def serialize(value, *args, **kwargs):
        if sys._getframe(1).f_code is resource_module.ResourceMutation._freeze_rows.__code__:
            assert value == [{"id": job["id"], "uuid": stored_uuid}]
            return original_dumps(value, *args, **kwargs)
        serializations.append(value)
        try:
            return original_dumps(value, *args, **kwargs)
        except (TypeError, ValueError) as error:
            serialization_errors.append(type(error))
            raise

    json_probe = SimpleNamespace(**{**vars(json), "dumps": serialize})
    monkeypatch.setattr(manager_module, "json", json_probe)
    monkeypatch.setattr(resource_module, "json", json_probe)
    original_getattribute = JobManager.__getattribute__

    def lookup(self, name):
        if self is manager and name in {"_update_gauges", "_stage_completion_sla_breach"}:
            calls.append(f"prepare:{name}")
        return original_getattribute(self, name)

    monkeypatch.setattr(JobManager, "__getattribute__", lookup)
    expected = (stored_uuid or "") if matching else "stale-identity"
    completion_value = completion_arguments(job)["completion_token"]
    observers = {
        manager._maybe_encrypt_json: transforms.append,
        manager_module._parse_dt: lambda args: calls.append("prepare_duration"),
        manager_module._run_post_commit_side_effects: lambda args: calls.append("dispatch"),
    }
    for name in (
        "increment_json_truncated",
        "observe_duration",
        "increment_completed",
        "increment_sla_breach",
        "_record_job_span",
        "observe_job_event",
    ):
        observers[getattr(manager_module, name)] = lambda args, name=name: calls.append(name)
    with observe_python_calls(observers):
        assert (
            manager.complete_job(job["id"], result=result, expected_uuid=expected, completion_token=completion_value)
            is matching
        )
    assert transforms == []
    assert len(serializations) == 1 and serializations[0] is result  # Size probe only, no final encoding.
    assert serialization_errors == ([] if serialization_error is None else [serialization_error])
    assert calls == []
    assert durable_state(reader, job["id"]) == before


@pytest.mark.parametrize("probe", ["identity", "early-exit"])
def test_completion_json_probe_ignores_unrelated_serialization(completion_manager, monkeypatch, probe):
    """Unrelated template JSON must not become a completion serialization."""
    manager = completion_manager
    original_connect = manager._connect
    unrelated: list[str] = []

    def connect():
        unrelated.append(json.dumps({"body": "unrelated template"}))
        return original_connect()

    monkeypatch.setattr(manager, "_connect", connect)
    if probe == "identity":
        test_completion_identity_short_circuits_result_work(
            manager, monkeypatch, "  NONcanonical-UUID  ", True, "object", TypeError
        )
    else:
        test_completion_early_exit_never_prepares_result(
            manager, monkeypatch, "missing", "object", TypeError
        )
    assert unrelated and set(unrelated) == {'{"body": "unrelated template"}'}  # nosec B101 -- Pytest verdict, never a runtime security guard.


@pytest.mark.parametrize("state", ["missing", "stale", "failed", "cancelled", "quarantined", "completed-no-token"])
@pytest.mark.parametrize(
    "result_kind,serialization_error", [("valid", None), ("object", TypeError), ("circular", ValueError)]
)
def test_completion_early_exit_never_prepares_result(
    completion_manager, monkeypatch, state, result_kind, serialization_error
):
    """Misses and other terminal states do not touch result transformation or observers."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    arguments = completion_arguments(job)
    if state in {"failed", "cancelled", "quarantined", "completed-no-token"}:
        rows(
            reader,
            "UPDATE jobs SET status=?, completion_token=? WHERE id=?",
            (state.split("-")[0], "binding-token", job["id"]),
        )
    if state == "completed-no-token":
        arguments["completion_token"] = None
    if state == "stale":
        arguments["expected_uuid"] = "stale"
    job_id = job["id"] + 100 if state == "missing" else job["id"]
    before = durable_state(reader, job["id"])
    transforms: list[Any] = []
    serializations: list[Any] = []
    calls: list[str] = []
    serialization_errors: list[type[Exception]] = []
    result = {"ignored": object() if result_kind == "object" else True}
    if result_kind == "circular":
        result["self"] = result
    original_dumps = manager_module.json.dumps
    if serialization_error is not None:
        # This same payload would fail final encoding if the early exit moved.
        with pytest.raises(serialization_error):
            original_dumps(result)

    def serialize(value, *args, **kwargs):
        if sys._getframe(1).f_code is resource_module.ResourceMutation._freeze_rows.__code__:
            assert value == [{"id": job["id"], "uuid": job["uuid"]}]
            return original_dumps(value, *args, **kwargs)
        serializations.append(value)
        try:
            return original_dumps(value, *args, **kwargs)
        except (TypeError, ValueError) as error:
            serialization_errors.append(type(error))
            raise

    json_probe = SimpleNamespace(**{**vars(json), "dumps": serialize})
    monkeypatch.setattr(manager_module, "json", json_probe)
    monkeypatch.setattr(resource_module, "json", json_probe)
    original_getattribute = JobManager.__getattribute__

    def lookup(self, name):
        if self is manager and name in {"_update_gauges", "_stage_completion_sla_breach"}:
            calls.append(f"prepare:{name}")
        return original_getattribute(self, name)

    monkeypatch.setattr(JobManager, "__getattribute__", lookup)
    observers = {
        manager._maybe_encrypt_json: transforms.append,
        manager_module._parse_dt: lambda args: calls.append("prepare_duration"),
        manager_module._run_post_commit_side_effects: lambda args: calls.append("dispatch"),
    }
    for name in (
        "increment_json_truncated",
        "observe_duration",
        "increment_completed",
        "increment_sla_breach",
        "_record_job_span",
        "observe_job_event",
    ):
        observers[getattr(manager_module, name)] = lambda args, name=name: calls.append(name)
    with observe_python_calls(observers):
        assert manager.complete_job(job_id, result=result, **arguments) is False
    assert transforms == []
    assert len(serializations) == 1 and serializations[0] is result
    assert serialization_errors == ([] if serialization_error is None else [serialization_error])
    assert calls == []
    assert durable_state(reader, job["id"]) == before


@pytest.mark.parametrize("scenario", ["type-connect", "value-connect", "type-deferred", "value-deferred", "rejected"])
def test_completion_serialization_and_connection_error_precedence(completion_manager, monkeypatch, scenario):
    """Preflight is eager; final encoding follows connect, identity, and transformation."""
    manager = completion_manager
    job = seed_job(manager, processing=scenario != "rejected")
    reader = independent_reader(manager)
    before = durable_state(reader, job["id"])
    phases: list[str] = []
    original = manager_module.json.dumps
    original_connect = manager._connect
    result = {"input": "x" * 200}
    transformed = {"stored": True}
    probe_failed = False

    def serialize(value, *args, **kwargs):
        nonlocal probe_failed
        if value is result:
            phases.append("preflight")
            if scenario.startswith(("type-", "value-")) and not probe_failed:
                probe_failed = True
                error = TypeError if scenario.startswith("type-") else ValueError
                raise error("size probe")
        if value is transformed:
            phases.append("final")
        return original(value, *args, **kwargs)

    def connect():
        phases.append("connect")
        if scenario.endswith("-connect"):
            raise ConnectionError("connect wins")
        return original_connect()

    def transform(value, domain):
        assert value is result and domain == "chatbooks"
        phases.append("transform")
        return transformed

    monkeypatch.setattr(
        manager_module, "json", SimpleNamespace(**{**vars(manager_module.json), "dumps": serialize})
    )
    monkeypatch.setattr(manager, "_connect", connect)
    monkeypatch.setattr(manager, "_maybe_encrypt_json", transform)
    if scenario.endswith("-connect"):
        with pytest.raises(ConnectionError, match="connect wins"):
            manager.complete_job(job["id"], result=result, **completion_arguments(job))
        assert phases == ["preflight", "connect"]
    else:
        assert manager.complete_job(job["id"], result=result, **completion_arguments(job)) is (scenario != "rejected")
        # Applied completion also reads SLA policy and updates real queue gauges.
        trailing = ["connect", "connect"] if scenario != "rejected" else []
        assert phases == ["preflight", "connect", "transform", "final", *trailing]
    if scenario in {"type-deferred", "value-deferred"}:
        assert reader.get_job(job["id"])["result"] == transformed
    else:
        assert durable_state(reader, job["id"]) == before


def test_completion_oversize_precedes_connection(completion_manager, monkeypatch):
    """The 128-byte result limit rejects before opening any completion connection."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    before = durable_state(reader, job["id"])
    phases = []
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "128")
    with observe_python_calls(
        {
            manager_module.json.dumps: lambda args: phases.append("preflight"),
            manager._connect: lambda args: phases.append("connect"),
            manager._maybe_encrypt_json: lambda args: phases.append("transform"),
        }
    ):
        with pytest.raises(ValueError, match="Result too large"):
            manager.complete_job(job["id"], result={"input": "x" * 200}, **completion_arguments(job))
    assert phases == ["preflight"]
    assert durable_state(reader, job["id"]) == before


@pytest.mark.parametrize("fail", [False, True], ids=["return", "exception"])
def test_call_observer_preserves_bound_methods_and_previous_profile(completion_manager, fail):
    """Real bound methods stay instance-scoped and the prior profiler always resumes."""
    manager = completion_manager
    other = independent_reader(manager)
    payload = {"ok": True}
    calls = []
    returns = []
    prior_calls = []
    original_profile = sys.getprofile()

    def previous(frame, event, arg):
        if event == "call" and frame.f_code is JobManager._maybe_encrypt_json.__code__:
            prior_calls.append(frame.f_locals["self"])

    sys.setprofile(previous)
    try:
        with pytest.raises(RuntimeError, match="observer cleanup probe") if fail else nullcontext():
            with observe_python_calls(
                {manager._maybe_encrypt_json: calls.append}, returns={manager._maybe_encrypt_json: returns.append}
            ):
                assert manager._maybe_encrypt_json(payload, "chatbooks") is payload
                assert other._maybe_encrypt_json(payload, "chatbooks") is payload
                if fail:
                    raise RuntimeError("observer cleanup probe")
        assert sys.getprofile() is previous
    finally:
        sys.setprofile(original_profile)
    assert len(calls) == 1 and calls[0]["self"] is manager
    assert len(returns) == 1 and returns[0] is payload
    assert prior_calls == [manager, other]


@pytest.mark.parametrize(
    "status,enforce,allowed,mismatch,want",
    [
        ("processing", True, True, None, True),
        ("processing", False, True, "worker", True),
        ("processing", True, True, "worker", False),
        ("processing", True, True, "lease", False),
        ("processing", True, True, "token", False),
        ("processing", False, True, "token", False),
        ("ready", False, True, None, True),
        ("scheduled", False, True, None, True),
        ("ready", False, False, None, False),
        ("scheduled", False, False, None, False),
        ("ready", True, True, None, False),
        ("scheduled", True, True, None, False),
    ],
)
def test_completion_preserves_status_token_and_lease_matrix(
    completion_manager, monkeypatch, status, enforce, allowed, mismatch, want
):
    """Pin missing status/lease branches; existing identity/race suites cover replay."""
    manager = completion_manager
    options = {"available_at": datetime.now(timezone.utc) + timedelta(days=1)} if status == "scheduled" else {}
    job = seed_job(manager, processing=status == "processing", **options)
    reader = independent_reader(manager)
    arguments = completion_arguments(job)
    arguments["enforce"] = enforce
    if mismatch in {"worker", "lease"}:
        arguments[f"{mismatch}_id"] = "other"
    elif mismatch == "token":
        rows(reader, "UPDATE jobs SET completion_token=? WHERE id=?", ("conflicting", job["id"]))
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks" if allowed else "other")
    before = durable_state(reader, job["id"])
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    assert manager.complete_job(job["id"], result={"ok": True}, **arguments) is want
    if not want:
        assert durable_state(reader, job["id"]) == before
        assert calls == []
    else:
        stored, counters, events = durable_state(reader, job["id"])
        assert (stored[0]["status"], stored[0]["completion_token"]) == ("completed", "binding-token")
        assert tuple(counters[0][key] for key in ("ready_count", "scheduled_count", "processing_count")) == (0, 0, 0)
        assert [event["event_type"] for event in events] == (
            ["job.created", "job.acquired", "job.completed"]
            if status == "processing"
            else ["job.created", "job.completed"]
        )


@pytest.mark.parametrize(
    "path,cleanup",
    [
        (path, cleanup)
        for path in ("applied", "replay", "false", "primary-error")
        for cleanup in (None, "close", "close_assertion", "context_exit")
    ]
    + [("commit-error", None)],
)
def test_completion_cleanup_and_commit_failure_order(completion_manager, monkeypatch, path, cleanup):
    """Keep native rollback/exit/close semantics, error masking, and explicit commit order."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    arguments = completion_arguments(job)
    if path == "replay":
        assert manager.complete_job(job["id"], **arguments)
    if path == "false":
        arguments["expected_uuid"] = "stale"
    before = durable_state(reader, job["id"])
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(manager, monkeypatch, calls, fail_at="commit" if path == "commit-error" else cleanup)

    result = {"ok": True}
    if path == "primary-error":
        monkeypatch.setattr(manager, "_maybe_encrypt_json", lambda *args: object())
        # The preflight works; real final JSON serialization rejects the transformed value.
    error = None
    if path == "commit-error":
        error = RuntimeError
    elif cleanup == "close_assertion":
        error = AssertionError
    elif cleanup == "context_exit":
        error = RuntimeError
    elif path == "primary-error":
        error = TypeError
    if error:
        with pytest.raises(error) as raised:
            manager.complete_job(job["id"], result=result, **arguments)
        if cleanup == "close_assertion" and path != "commit-error":
            assert str(raised.value) == "close probe"
        elif cleanup == "context_exit" and path != "commit-error":
            assert str(raised.value) == "context exit probe"
    else:
        assert manager.complete_job(job["id"], result=result, **arguments) is (path != "false")
    prefix = (
        ["commit", "duration", "completed", "original_gauges", "trace", "event"]
        if path == "applied"
        else (["commit"] if path == "commit-error" else [])
    )
    if manager.backend == "postgres":
        assert calls == prefix + ["cursor_exit", "context_exit", "close"]
    else:
        assert calls == prefix + ["context_exit", "close"]
    if path == "applied":
        assert_completed(reader, job["id"])
    else:
        assert durable_state(reader, job["id"]) == before


@contextmanager
def metric_preparation_case(manager):
    """Capture real completion observations; all verdicts belong to the tests."""
    job = seed_job(manager)
    reader = independent_reader(manager)
    raw = rows(reader, "SELECT * FROM jobs WHERE id=?", (job["id"],))[0]
    manager.upsert_sla_policy(
        domain="chatbooks", queue="default", job_type=job["job_type"], max_duration_seconds=999999999
    )
    calls: list[str] = []
    durations: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    clock_samples: list[datetime] = []
    observed = {
        "job": job,
        "reader": reader,
        "raw": raw,
        "calls": calls,
        "durations": durations,
        "traces": traces,
        "clock_samples": clock_samples,
        "result": {"_truncated": True, "len_bytes": 212},
        "projection": {
            key: raw[key]
            for key in (
                "id",
                "uuid",
                "status",
                "completion_token",
                "worker_id",
                "lease_id",
                "domain",
                "queue",
                "job_type",
                "available_at",
                "started_at",
                "acquired_at",
                "trace_id",
                "request_id",
                "owner_user_id",
            )
        },
    }
    yield observed

    def duration(arguments):
        calls.append("duration")
        durations.append(arguments)

    def trace(arguments):
        calls.append("trace")
        traces.append(arguments)

    observed["before"] = datetime.now(timezone.utc)
    with observe_python_calls(
        {
            manager_module.observe_duration: duration,
            manager_module._record_job_span: trace,
            manager_module.increment_json_truncated: lambda args: calls.append("truncated"),
            manager_module.increment_completed: lambda args: calls.append("completed"),
            manager._update_gauges: lambda args: calls.append("original_gauges"),
            manager_module.observe_job_event: lambda args: calls.append("event"),
        },
        returns={manager._clock.now_utc: clock_samples.append},
    ):
        observed["success"] = manager.complete_job(job["id"], result=observed["result"], **completion_arguments(job))
    observed["after"] = datetime.now(timezone.utc)
    stored = rows(reader, "SELECT result FROM jobs WHERE id=?", (job["id"],))[0]["result"]
    if isinstance(stored, str):
        stored = manager_module.json.loads(stored)
    observed["stored"] = stored


@pytest.mark.parametrize("scenario", ["normal", "encrypted", "truncation-error"])
def test_completion_metric_preparation_boundaries(completion_manager, monkeypatch, scenario):
    """Optional truncation failure leaves real metrics, trace, and durable completion intact."""
    manager = completion_manager
    with metric_preparation_case(completion_manager) as observed:

        class BrokenTruncation(dict):
            def get(self, key, default=None):
                if key == "_truncated":
                    raise RuntimeError("truncation preparation probe")
                return super().get(key, default)

        for setting, value in (("JOBS_MAX_JSON_BYTES", "128"), ("JOBS_JSON_TRUNCATE", "true")):
            monkeypatch.setenv(setting, value)
        if scenario == "truncation-error":
            # Exercise the mapping fault directly, without replacing transformation.
            observed["result"] = BrokenTruncation(observed["result"])
        else:
            observed["result"] = {"long": "x" * 200}
        if scenario == "encrypted":
            monkeypatch.setattr(manager, "_should_encrypt", lambda domain: True)
            monkeypatch.setattr(manager_module, "encrypt_json_blob", lambda value: {"cipher": "opaque"})
    assert observed["success"] is True
    assert_completed(observed["reader"], observed["job"]["id"])
    assert observed["calls"] == (
        (["truncated"] if scenario == "normal" else []) + ["duration", "completed", "original_gauges", "trace", "event"]
    )
    assert observed["stored"] == (
        {"_encrypted": {"cipher": "opaque"}} if scenario == "encrypted" else {"_truncated": True, "len_bytes": 212}
    )
    assert observed["traces"] == [{"operation": "job.complete", "job": observed["projection"], "attrs": None}]
    duration = observed["durations"][0]
    assert duration["job"] == {
        "domain": "chatbooks",
        "queue": "default",
        "job_type": observed["job"]["job_type"],
        "trace_id": "trace-1",
        "request_id": "request-1",
    }
    raw = observed["raw"]
    assert duration["started_at"] == datetime.fromisoformat(str(raw["started_at"] or raw["acquired_at"]))
    assert duration["completed_at"].tzinfo is (timezone.utc if manager.backend == "postgres" else None)
    if manager.backend == "postgres":
        assert duration["completed_at"] == observed["clock_samples"][0]
    else:
        assert (
            observed["before"].replace(tzinfo=None)
            <= duration["completed_at"]
            <= observed["after"].replace(tzinfo=None)
        )
    assert len(observed["clock_samples"]) == (2 if manager.backend == "postgres" else 1)
    assert observed["clock_samples"][-1] >= duration["completed_at"].replace(tzinfo=timezone.utc)


def test_completion_parser_metric_preparation_failure(completion_manager, monkeypatch):
    """Parser failure suppresses metrics and the later SLA clock, not completion."""
    manager = completion_manager
    with metric_preparation_case(manager) as observed:

        def parse(value):
            raise ValueError("metric parser probe")

        monkeypatch.setattr(manager_module, "_parse_dt", parse)
        if manager.backend == "postgres":
            record_completion_boundaries(manager, monkeypatch, [])
            original_fetchone = CompletionCursor.__getattr__

            def cursor_getattr(self, name):
                native = original_fetchone(self, name)
                if name != "fetchone":
                    return native

                def fetchone():
                    value = native()
                    if value and "started_at" in value:
                        value = dict(value)
                        for key in ("started_at", "acquired_at"):
                            if value.get(key) is not None:
                                value[key] = str(value[key])
                    return value

                return fetchone

            monkeypatch.setattr(CompletionCursor, "__getattr__", cursor_getattr)
    assert observed["success"] is True
    assert_completed(observed["reader"], observed["job"]["id"])
    assert observed["calls"] == ["truncated", "event"]
    assert observed["clock_samples"] == []
    assert observed["stored"] == {"_truncated": True, "len_bytes": 212}


def test_completion_clock_metric_preparation_failure(completion_manager, monkeypatch):
    """A failed metric clock cannot suppress the later independent SLA evaluation."""
    manager = completion_manager
    with metric_preparation_case(manager) as observed:
        original_clock = manager._clock.now_utc
        samples = []

        def clock():
            samples.append(None)
            if len(samples) == 1:
                raise RuntimeError("metric clock probe")
            return original_clock()

        class MetricDatetime(datetime):
            @classmethod
            def utcnow(cls):
                raise RuntimeError("metric clock probe")

        if manager.backend == "postgres":
            monkeypatch.setattr(manager._clock, "now_utc", clock)
        else:
            monkeypatch.setattr(manager_module, "datetime", MetricDatetime)
    assert observed["success"] is True
    assert_completed(observed["reader"], observed["job"]["id"])
    assert observed["calls"] == ["truncated", "event"]
    assert len(observed["clock_samples"]) == (2 if manager.backend == "postgres" else 1)
    assert observed["stored"] == {"_truncated": True, "len_bytes": 212}


@pytest.mark.parametrize(
    "scenario", ["allow", "deny", "counter-after-metrics", "outbox-snapshot", "observer-settings", "early"]
)
def test_completion_policy_read_timing(completion_manager, monkeypatch, scenario):
    """Only the historical policy read points can affect an in-flight completion."""
    manager = completion_manager
    queued = scenario in {"allow", "deny"}
    job = seed_job(manager, processing=not queued)
    reader = independent_reader(manager)
    calls: list[str] = []
    policy_reads: list[tuple[str, str | None]] = []
    original_getenv = manager_module.os.getenv
    original_getattribute = JobManager.__getattribute__
    # Perturb only environment settings; transformation and observers stay real.
    settings = {
        "initial": (
            {"JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS": "other" if scenario == "allow" else "chatbooks"}
            if queued
            else {}
        ),
        "transform": (
            {"JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS": "chatbooks" if scenario == "allow" else "other"}
            if queued
            else ({"JOBS_EVENTS_OUTBOX": "false"} if scenario == "outbox-snapshot" else {})
        ),
        "binding": {"JOBS_COUNTERS_ENABLED": "false"} if scenario == "counter-after-metrics" else {},
        "commit": (
            {"JOBS_COUNTERS_ENABLED": "false", "JOBS_EVENTS_ENABLED": "true"} if scenario == "observer-settings" else {}
        ),
    }

    def settings_at(phase):
        for name, value in settings[phase].items():
            monkeypatch.setenv(name, value)

    def getenv(arguments):
        name = arguments["key"]
        if name in {
            "JOBS_COUNTERS_ENABLED",
            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
            "JOBS_EVENTS_OUTBOX",
            "JOBS_EVENTS_ENABLED",
        }:
            policy_reads.append((name, original_getenv(name, arguments.get("default"))))

    def transform(arguments):
        calls.append("transform")
        settings_at("transform")

    def getattribute(self, name):
        if self is manager and name == "_update_gauges":
            calls.append("gauge_binding")
            settings_at("binding")
        return original_getattribute(self, name)

    def committed():
        settings_at("commit")

    settings_at("initial")
    record_completion_boundaries(
        manager, monkeypatch, calls, on_commit=committed if scenario == "observer-settings" else None
    )
    monkeypatch.setattr(JobManager, "__getattribute__", getattribute)
    arguments = completion_arguments(job)
    arguments["enforce"] = not queued
    if scenario == "early":
        arguments["expected_uuid"] = "stale"
    before = durable_state(reader, job["id"])
    # Independent reader setup/reads are not facade policy observations.
    policy_reads.clear()
    want = scenario not in {"deny", "early"}
    with observe_python_calls(
        {
            manager_module.os.getenv: getenv,
            manager._maybe_encrypt_json: transform,
            manager_module.set_queue_gauges: lambda args: calls.append("real_gauges"),
        }
    ):
        assert manager.complete_job(job["id"], result={"ok": True}, **arguments) is want
    if scenario == "early":
        assert "transform" not in calls and "gauge_binding" not in calls
        assert policy_reads == [("JOBS_EVENTS_OUTBOX", "true")]
    elif queued:
        assert policy_reads[:2] == [
            ("JOBS_EVENTS_OUTBOX", "true"),
            ("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks" if want else "other"),
        ]
    elif scenario == "counter-after-metrics":
        assert policy_reads == [
            ("JOBS_EVENTS_OUTBOX", "true"),
            ("JOBS_COUNTERS_ENABLED", "false"),
            ("JOBS_COUNTERS_ENABLED", "false"),
            ("JOBS_EVENTS_ENABLED", "false"),
        ]
        assert calls.index("gauge_binding") < calls.index("commit")
    elif scenario == "outbox-snapshot":
        assert policy_reads == [
            ("JOBS_EVENTS_OUTBOX", "true"),
            ("JOBS_COUNTERS_ENABLED", "true"),
            ("JOBS_COUNTERS_ENABLED", "true"),
            ("JOBS_EVENTS_ENABLED", "false"),
        ]
    else:
        assert policy_reads == [
            ("JOBS_EVENTS_OUTBOX", "true"),
            ("JOBS_COUNTERS_ENABLED", "true"),
            ("JOBS_COUNTERS_ENABLED", "false"),
            ("JOBS_EVENTS_ENABLED", "true"),
        ]
        assert calls.index("commit") < calls.index("real_gauges")
    if not want:
        assert durable_state(reader, job["id"]) == before
    else:
        stored, counters, events = durable_state(reader, job["id"])
        assert stored[0]["status"] == "completed"
        assert counters[0]["processing_count"] == (1 if scenario == "counter-after-metrics" else 0)
        assert [event["event_type"] for event in events].count("job.completed") == 1


@pytest.mark.parametrize("failure", [None, "lookup", "close", "outside-tuple"])
def test_completion_sla_policy_lookup_lifetime(completion_manager, monkeypatch, failure):
    """The separate policy connection closes under the owner lock before SLA writes."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    rows(reader, "UPDATE jobs SET started_at=? WHERE id=?", ("2000-01-01 00:00:00", job["id"]))
    manager.upsert_sla_policy(domain="chatbooks", queue="default", job_type=job["job_type"], max_duration_seconds=1)
    before = durable_state(reader, job["id"])
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(manager, monkeypatch, calls)
    original_connect = manager._connect
    owner: list[Any] = []
    policy_connections: list[Any] = []
    policy_statements: list[str] = []
    original_execute = CompletionConnection.execute
    original_cursor_execute = CompletionCursor.execute

    def owning_execute(self, sql, parameters=()):
        if "INSERT INTO job_attachments" in str(sql):
            calls.append("sla_write")
        return original_execute(self, sql, parameters)

    def owning_cursor_execute(self, sql, parameters=None):
        if "INSERT INTO job_attachments" in str(sql):
            calls.append("sla_write")
        return original_cursor_execute(self, sql, parameters)

    def lookup(sql):
        policy_statements.append(str(sql))
        assert str(sql).startswith("SELECT * FROM job_sla_policies")
        if manager.backend == "postgres":
            from psycopg.pq import TransactionStatus

            assert owner[0].info.transaction_status == TransactionStatus.INTRANS
        else:
            assert owner[0].in_transaction
        calls.append("policy_read")
        if failure == "outside-tuple":
            raise AssertionError("policy outside tuple")
        return "SELECT * FROM missing_sla_policy_probe" if failure == "lookup" else sql

    class PolicyCursor:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self.inner.__exit__(*exc)

        def execute(self, sql, parameters=None):
            # RLS SET LOCAL belongs to cursor setup, not the policy query.
            if "FROM job_sla_policies" in str(sql):
                sql = lookup(sql)
                if failure == "lookup":
                    parameters = None
            return self.inner.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    class PolicyConnection:
        def __init__(self, inner):
            self.inner = inner

        def execute(self, sql, parameters=()):
            return self.inner.execute(lookup(sql), () if failure == "lookup" else parameters)

        def cursor(self, *args, **kwargs):
            return PolicyCursor(self.inner.cursor(*args, **kwargs))

        def commit(self):
            pytest.fail("policy lookup cannot own completion commit")

        def close(self):
            calls.append("policy_close")
            self.inner.close()
            if failure == "close":
                raise RuntimeError("policy close probe")

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def connect():
        connection = original_connect()
        if not owner:
            owner.append(connection)
            return connection
        calls.append("policy_open")
        policy_connections.append(connection)
        return PolicyConnection(connection)

    monkeypatch.setattr(manager, "_connect", connect)
    monkeypatch.setattr(CompletionConnection, "execute", owning_execute)
    monkeypatch.setattr(CompletionCursor, "execute", owning_cursor_execute)
    if failure == "outside-tuple":
        with pytest.raises(AssertionError, match="policy outside tuple"):
            manager.complete_job(job["id"], **completion_arguments(job))
        assert durable_state(reader, job["id"]) == before
        assert "commit" not in calls
    else:
        assert manager.complete_job(job["id"], **completion_arguments(job)) is True
        assert reader.get_job(job["id"])["status"] == "completed"
        event_types = [row["event_type"] for row in rows(reader, "SELECT * FROM job_events ORDER BY id")]
        assert event_types == (
            ["job.created", "job.acquired", "job.sla_breached", "job.completed"]
            if failure is None
            else ["job.created", "job.acquired", "job.completed"]
        )
        assert calls.index("policy_close") < calls.index("commit")
        if failure is None:
            assert calls.index("policy_close") < calls.index("sla_write") < calls.index("commit")
            assert len(rows(reader, "SELECT * FROM job_attachments WHERE job_id=?", (job["id"],))) == 1
        else:
            assert "sla_write" not in calls
    assert len(policy_connections) == len(policy_statements) == 1
    connection = policy_connections[0]
    if manager.backend == "postgres":
        assert connection.closed
    else:
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    assert calls[:3] == ["policy_open", "policy_read", "policy_close"]
    assert calls[-2:] == ["context_exit", "close"]


@pytest.mark.parametrize("abort", [False, True], ids=["normal", "diagnostic-abort"])
def test_completion_diagnostic_failure_keeps_pg_commit_guard(completion_manager, monkeypatch, abort):
    """A real diagnostic statement abort must never become a successful PG commit."""
    manager = completion_manager
    job = seed_job(manager)
    reader = independent_reader(manager)
    before = durable_state(reader, job["id"])
    calls: list[str] = []
    entry: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    record_completion_boundaries(manager, monkeypatch, calls)
    original_sqlite_execute = CompletionConnection.execute
    original_pg_execute = CompletionCursor.execute

    def test_mode():
        entry.append("test-mode")
        return True

    def diagnostic(sql):
        normalized = " ".join(str(sql).split())
        if "FOR UPDATE" in normalized or "uuid, status, completion_token" in normalized:
            entry.append("identity")
        if normalized.startswith("SELECT id, status FROM jobs"):
            entry.append("diagnostic")
            if abort:
                return "SELECT * FROM missing_completion_diagnostic_probe"
        return sql

    def sqlite_execute(self, sql, parameters=()):
        rewritten = diagnostic(sql)
        return original_sqlite_execute(self, rewritten, () if rewritten != sql else parameters)

    def pg_execute(self, sql, parameters=None):
        rewritten = diagnostic(sql)
        return original_pg_execute(self, rewritten, None if rewritten != sql else parameters)

    monkeypatch.setattr(manager_module, "_is_test_mode", test_mode)
    connection_type, execute = (
        (CompletionCursor, pg_execute) if manager.backend == "postgres" else (CompletionConnection, sqlite_execute)
    )
    monkeypatch.setattr(connection_type, "execute", execute)
    with observe_python_calls({manager._connect: lambda args: entry.append("connect")}):
        if abort and manager.backend == "postgres":
            import psycopg

            # The next required counter write sees the genuinely aborted transaction.
            with pytest.raises(psycopg.errors.InFailedSqlTransaction):
                manager.complete_job(job["id"], **completion_arguments(job))
            assert durable_state(reader, job["id"]) == before
            assert calls == ["cursor_exit", "context_exit", "close"]
        else:
            assert manager.complete_job(job["id"], **completion_arguments(job)) is True
            assert_completed(reader, job["id"])
    assert entry[:4] == ["connect", "test-mode", "identity", "diagnostic"]
    # The real guard is reachable with counters/outbox disabled: its separate real
    # policy lookup does not issue statements on the aborted owning connection.
    if abort and manager.backend == "postgres":
        for setting in ("JOBS_COUNTERS_ENABLED", "JOBS_EVENTS_OUTBOX"):
            monkeypatch.setenv(setting, "false")
        record_completion_boundaries(manager, monkeypatch, calls)
        calls.clear()
        with pytest.raises(RuntimeError, match="transaction is aborted"):
            manager.complete_job(job["id"], **completion_arguments(job))
        assert "rollback" in calls and "commit" not in calls
        assert durable_state(reader, job["id"]) == before


@pytest.mark.parametrize("mutation", ["late-binding", "wrong-order"])
def test_completion_binding_assertions_reject_adversarial_probes(completion_manager, monkeypatch, mutation):
    """Demonstrate the binding/order oracle rejects scoped test-only adversaries."""
    manager = completion_manager
    job = seed_job(manager)
    calls: list[str] = []
    record_completion_observers(manager, monkeypatch, calls)
    bound_gauge = manager._update_gauges
    original_runner = manager_module._run_post_commit_side_effects
    record_completion_boundaries(
        manager,
        monkeypatch,
        calls,
        on_commit=lambda: monkeypatch.setattr(manager, "_update_gauges", lambda **kwargs: calls.append("wrong_gauges")),
    )

    def adversarial_runner(effects):
        if mutation == "late-binding":
            # Deliberately emulate an extraction which re-reads the manager at dispatch.
            effects = [
                (manager._update_gauges if callback is bound_gauge else callback, args, kwargs)
                for callback, args, kwargs in effects
            ]
        else:
            effects = [effects[-1], *effects[:-1]]  # Same callbacks, wrong order.
        original_runner(effects)

    monkeypatch.setattr(manager_module, "_run_post_commit_side_effects", adversarial_runner)
    assert manager.complete_job(job["id"], result={"ok": True}, **completion_arguments(job)) is True
    expected = ["commit", "duration", "completed", "original_gauges", "trace", "event"]
    expected += ["cursor_exit", "context_exit", "close"] if manager.backend == "postgres" else ["context_exit", "close"]
    with pytest.raises(AssertionError):
        assert calls == expected
    assert ("wrong_gauges" in calls) is (mutation == "late-binding")
    if mutation == "wrong-order":
        assert calls.index("event") < calls.index("duration")
    assert_completed(independent_reader(manager), job["id"])
