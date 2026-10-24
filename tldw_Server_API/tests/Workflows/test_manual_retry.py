"""Manual Retry replays the failed prerequisite once without repeating completed work."""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows as endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows import engine as engine_module
from tldw_Server_API.app.core.Workflows.adapters.control.flow import run_prompt_adapter

pytestmark = pytest.mark.unit


@pytest.fixture
def retry_case(tmp_path, monkeypatch):
    db = WorkflowsDatabase(str(tmp_path / "workflows.db"))
    monkeypatch.setattr(engine_module.WorkflowScheduler, "_inst", engine_module.WorkflowScheduler())
    user = User(id=1, username="retry-user", is_active=True, is_admin=True, roles=["admin"], tenant_id="default")
    try:
        yield db, user
    finally:
        db.close()


async def dispatch(route, db, user):
    before = asyncio.all_tasks()
    if route == "control":
        response = await endpoints.control_run(
            run_id="retry-run", action="retry", request=SimpleNamespace(headers={}), current_user=user, db=db
        )
    else:
        response = await endpoints.retry_run(run_id="retry-run", current_user=user, db=db)
    return response, asyncio.all_tasks() - before


def create_run(db, *, dependent=True):
    steps = [
        {"id": "before", "type": "prompt", "retry": 0, "config": {"kind": "before", "template": "seed"}},
        {"id": "failed", "type": "prompt", "retry": 0, "config": {"kind": "failed", "template": "{{ before.text }}"}},
    ]
    if dependent:
        steps.append(
            {"id": "after", "type": "prompt", "retry": 0, "config": {"kind": "after", "template": "{{ failed.text }}"}}
        )
    db.create_run(
        run_id="retry-run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "manual-retry", "version": 1, "steps": steps},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["control", "legacy"])
async def test_retry_replays_failed_prerequisite_and_keeps_completed_outputs(retry_case, monkeypatch, route):
    db, user = retry_case
    calls = Counter()

    async def adapter(config, context):
        kind = config["kind"]
        calls[kind] += 1
        if kind == "failed":
            if calls[kind] == 1:
                raise RuntimeError("transient_network_error")
            return await run_prompt_adapter({**config, "template": config["template"] + ":recovered"}, context)
        return await run_prompt_adapter(config, context)

    monkeypatch.setattr(engine_module, "get_adapter", lambda step_type: adapter)
    create_run(db)
    await engine_module.WorkflowEngine(db).start_run("retry-run")
    assert db.get_run("retry-run").status == "failed"  # nosec B101 -- causal failed prerequisite.
    assert calls == {"before": 1, "failed": 1}  # nosec B101 -- downstream remains blocked.
    response, tasks = await dispatch(route, db, user)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert response["ok"] is True  # nosec B101 -- normal route acknowledgement.
    assert calls == {"before": 1, "failed": 2, "after": 1}  # nosec B101 -- retry preserves completed effects.
    run = db.get_run("retry-run")
    assert run.status == "succeeded"  # nosec B101 -- repaired prerequisite permits dependent work.
    assert run.outputs_json == '{"text": "seed:recovered"}'  # nosec B101 -- actual recovered dependency output.
    failed_rows = [row for row in db.list_step_runs(run_id="retry-run") if row["step_id"] == "failed"]
    assert [row["status"] for row in failed_rows] == ["failed", "succeeded"]  # nosec B101 -- failed history retained.


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["control", "legacy"])
async def test_retry_of_still_failing_final_step_never_becomes_success(retry_case, monkeypatch, route):
    db, user = retry_case
    calls = Counter()

    async def adapter(config, context):
        calls[config["kind"]] += 1
        if config["kind"] == "failed":
            raise RuntimeError("transient_network_error")
        return await run_prompt_adapter(config, context)

    monkeypatch.setattr(engine_module, "get_adapter", lambda step_type: adapter)
    create_run(db, dependent=False)
    await engine_module.WorkflowEngine(db).start_run("retry-run")
    for attempt in (2, 3):
        _, tasks = await dispatch(route, db, user)
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert db.get_run("retry-run").status == "failed"  # nosec B101 -- no false terminal success.
        assert calls == {"before": 1, "failed": attempt}  # nosec B101 -- each new failure can be legitimately retried.


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["control", "legacy"])
async def test_duplicate_retry_does_not_duplicate_work_or_reopen_success(retry_case, monkeypatch, route):
    db, user = retry_case
    calls = Counter()
    released = asyncio.Event()

    async def adapter(config, context):
        calls[config["kind"]] += 1
        if config["kind"] == "failed":
            if calls["failed"] == 1:
                raise RuntimeError("transient_network_error")
            await released.wait()
        return await run_prompt_adapter(config, context)

    monkeypatch.setattr(engine_module, "get_adapter", lambda step_type: adapter)
    create_run(db)
    await engine_module.WorkflowEngine(db).start_run("retry-run")
    _, first = await dispatch(route, db, user)
    _, second = await dispatch(route, db, user)
    released.set()
    await asyncio.wait_for(asyncio.gather(*(first | second)), timeout=5)
    assert calls == {"before": 1, "failed": 2, "after": 1}  # nosec B101 -- one admitted replay.
    _, repeated = await dispatch(route, db, user)
    await asyncio.wait_for(asyncio.gather(*repeated), timeout=5)
    assert db.get_run("retry-run").status == "succeeded"  # nosec B101 -- repeated control cannot corrupt success.
    assert calls == {"before": 1, "failed": 2, "after": 1}  # nosec B101 -- no completed side effect replay.


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["control", "legacy"])
async def test_retry_refuses_cancelled_run_without_mutating_history(retry_case, monkeypatch, route):
    db, user = retry_case
    calls = []

    async def adapter(config, context):
        calls.append(config["kind"])
        if config["kind"] == "failed":
            raise RuntimeError("transient_network_error")
        return await run_prompt_adapter(config, context)

    monkeypatch.setattr(engine_module, "get_adapter", lambda step_type: adapter)
    create_run(db)
    engine = engine_module.WorkflowEngine(db)
    await engine.start_run("retry-run")
    db.update_run_status("retry-run", status="running")
    engine.cancel("retry-run")
    before = db.get_events("retry-run")
    _, tasks = await dispatch(route, db, user)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    assert not tasks  # nosec B101 -- cancelled workflow must not be dispatched.
    assert db.get_run("retry-run").status == "cancelled"  # nosec B101 -- terminal cancellation retained.
    assert db.get_events("retry-run") == before  # nosec B101 -- no spurious retry/history mutation.
    assert calls == ["before", "failed"]  # nosec B101 -- no cancelled side effects replay.


def test_failed_retry_claim_is_atomic_and_fenced_to_observed_failure(workflows_dual_backend_db):
    _, db = workflows_dual_backend_db
    create_run(db, dependent=False)
    first_failure = "2026-10-21T00:00:00.000001+00:00"
    second_failure = "2026-10-21T00:00:00.000002+00:00"
    db.update_run_status("retry-run", status="failed", ended_at=first_failure)
    other = WorkflowsDatabase(db.db_path, backend=db.backend)
    try:
        assert db.claim_failed_run_retry("retry-run", first_failure)  # nosec B101 -- one winner claims failure.
        assert not other.claim_failed_run_retry("retry-run", first_failure)  # nosec B101 -- distinct DB handle loses.
        db.update_run_status("retry-run", status="failed", ended_at=second_failure)
        assert not other.claim_failed_run_retry("retry-run", first_failure)  # nosec B101 -- stale duplicate cannot claim new failure.
        assert other.claim_failed_run_retry("retry-run", second_failure)  # nosec B101 -- legitimate later retry allowed.
        db.update_run_status("retry-run", status="failed", ended_at=second_failure)
        db.set_cancel_requested("retry-run", True)
        assert not other.claim_failed_run_retry("retry-run", second_failure)  # nosec B101 -- cancellation wins admission.
    finally:
        other.close()


@pytest.mark.asyncio
async def test_continuation_passes_new_structured_outputs_to_dependent_step(retry_case, monkeypatch):
    db, _ = retry_case
    payload = [{"name": "Rowan", "enabled": True}]

    async def adapter(config, context):
        if config["kind"] == "producer":
            return {"items": payload}
        return {"received": config["data"]}

    monkeypatch.setattr(engine_module, "get_adapter", lambda step_type: adapter)
    db.create_run(
        run_id="retry-run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "continued-structured-dependency",
            "version": 1,
            "steps": [
                {"id": "wait", "type": "wait_for_human", "config": {}},
                {"id": "producer", "type": "log", "config": {"kind": "producer"}},
                {"id": "consumer", "type": "log", "config": {"kind": "consumer", "data": "{{ producer.items }}"}},
            ],
        },
    )
    db.update_run_status("retry-run", status="waiting_human")
    await engine_module.WorkflowEngine(db).continue_run("retry-run", after_step_id="wait")
    run = db.get_run("retry-run")
    assert run.status == "succeeded"  # nosec B101 -- normal human/checkpoint continuation remains supported.
    assert json.loads(run.outputs_json) == {"received": payload}  # nosec B101 -- preserve actual list/dict output types.
