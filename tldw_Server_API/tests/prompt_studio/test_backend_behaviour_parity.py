"""Behavioural parity between the two Prompt Studio implementations (TASK-13318, Stage 2).

The signature ratchet (tests/DB_Management/test_prompt_studio_backend_parity.py) only
catches drift in *signatures*. This harness runs the same scenario against the SQLite
implementation and the PostgreSQL one and compares what comes back, so an aggregate can
be moved into a single backend-neutral repository and proven unchanged on both.

Each aggregate gets one scenario. A stage of the consolidation adds its aggregate's
scenario before moving that aggregate.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

from .conftest import _reset_prompt_studio_tables

# Values that legitimately differ per run or per database.
_VOLATILE_KEYS = frozenset(
    {"uuid", "created_at", "updated_at", "last_modified", "started_at", "completed_at", "deleted_at", "leased_until"}
)


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def _outcome(fn: Callable[[], Any]) -> Any:
    """The result, or the exception type: error behaviour must match too."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"raises": type(exc).__name__}


def _seed(db: PromptStudioDatabase) -> dict[str, int]:
    project = db.create_project(name="parity", description="d", user_id="u1")
    prompt = db.create_prompt(project["id"], "p", system_prompt="s", user_prompt="u {x}")
    case = db.create_test_case(project["id"], "c", inputs={"x": 1}, expected_outputs={"y": 2})
    return {"project": project["id"], "prompt": prompt["id"], "case": case["id"]}


def _test_runs(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    run = db.create_test_run(
        project_id=ids["project"],
        prompt_id=ids["prompt"],
        test_case_id=ids["case"],
        model_name="m",
        model_params={"t": 0},
        inputs={"x": 1},
        outputs={"y": 2},
        scores={"acc": 1.0},
        execution_time_ms=5,
        tokens_used=7,
        cost_estimate=0.5,
    )
    return {"create_test_run": run}


def _prompt_versions(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    v2 = db.create_prompt_version(ids["prompt"], change_description="v2", user_prompt="u2 {x}")
    reverted = db.revert_prompt_to_version(ids["prompt"], 1)
    return {
        "create_prompt_version": v2,
        "revert_prompt_to_version": reverted,
        "list_prompt_versions": db.list_prompt_versions(ids["project"], "p"),
    }


def _evaluations(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    ev = db.create_evaluation(
        prompt_id=ids["prompt"], project_id=ids["project"], model_configs={"m": 1}, test_case_ids=[ids["case"]]
    )
    updated = db.update_evaluation(ev["id"], {"status": "completed", "aggregate_metrics": {"acc": 1.0}})
    return {
        "create_evaluation": ev,
        "update_evaluation": updated,
        "get_evaluation": db.get_evaluation(ev["id"]),
        "get_evaluation_missing": db.get_evaluation(ev["id"] + 999),
        "list_evaluations": db.list_evaluations(project_id=ids["project"]),
        "list_evaluations_by_status": db.list_evaluations(status="running"),
    }


def _signatures(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    project = ids["project"]
    a = db.create_signature(project, "alpha", input_schema=[{"name": "x"}], output_schema=[{"name": "y"}])
    b = db.create_signature(project, "beta", input_schema=[], output_schema=[], constraints={"max": 1})
    return {
        "create": a,
        "create_duplicate": _outcome(lambda: db.create_signature(project, "alpha", input_schema=[], output_schema=[])),
        "create_blank": _outcome(lambda: db.create_signature(project, " ", input_schema=[], output_schema=[])),
        "get": db.get_signature(a["id"]),
        "update": db.update_signature(b["id"], {"output_schema": [{"name": "z"}], "ignored": 1}),
        "update_noop": db.update_signature(b["id"], {"ignored": 1}),
        "update_rename_to_duplicate": _outcome(lambda: db.update_signature(b["id"], {"name": "alpha"})),
        "list_search": db.list_signatures(project, search="ALP"),
        "list_paged": db.list_signatures(project, page=1, per_page=1, return_pagination=True)["pagination"],
        "soft_delete": db.delete_signature(a["id"]),
        "soft_delete_again": db.delete_signature(a["id"]),
        "get_deleted": _outcome(lambda: db.get_signature(a["id"])),
        "get_deleted_included": db.get_signature(a["id"], include_deleted=True) is not None,
        "update_deleted": _outcome(lambda: db.update_signature(a["id"], {"name": "gamma"})),
        "list_after_delete": [s["name"] for s in db.list_signatures(project)],
        "list_including_deleted": sorted(s["name"] for s in db.list_signatures(project, include_deleted=True)),
        "hard_delete": db.delete_signature(b["id"], hard_delete=True),
        "hard_delete_missing": db.delete_signature(b["id"], hard_delete=True),
    }


def _projects(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)  # "parity", owned by u1, with one prompt and one test case
    other = db.create_project(name="Other Thing", description="needle", status="active", metadata={"k": 1}, user_id="u2")

    def names(result: dict[str, Any]) -> list[str]:
        return sorted(p["name"] for p in result["projects"])

    return {
        "create": other,
        "create_duplicate": _outcome(lambda: db.create_project(name="parity", user_id="u1")),
        "create_same_name_other_user": db.create_project(name="parity", user_id="u3")["name"],
        "create_blank": _outcome(lambda: db.create_project(name="   ")),
        "create_too_long": _outcome(lambda: db.create_project(name="x" * 1000)),
        "get": db.get_project(ids["project"]),
        "get_missing": _outcome(lambda: db.get_project(99999)),
        "list": db.list_projects(user_id="u1"),
        "list_status": names(db.list_projects(status="active")),
        "list_search_description": names(db.list_projects(search="NEEDLE")),
        "list_paged": db.list_projects(page=2, per_page=1)["pagination"],
        "update": db.update_project(other["id"], {"description": "d2", "metadata": {"k": 2}, "ignored": 1}),
        "update_strips_name": db.update_project(other["id"], name="  Renamed  ")["name"],
        "update_blank_name": _outcome(lambda: db.update_project(other["id"], {"name": ""})),
        "update_noop": db.update_project(other["id"], {"ignored": 1})["name"],
        "update_missing": _outcome(lambda: db.update_project(99999, {"description": "x"})),
        "soft_delete": db.delete_project(other["id"]),
        "soft_delete_again": db.delete_project(other["id"]),
        "get_deleted": _outcome(lambda: db.get_project(other["id"])),
        "get_deleted_included": _outcome(lambda: db.get_project(other["id"], include_deleted=True) is not None),
        "update_deleted": _outcome(lambda: db.update_project(other["id"], {"description": "x"})),
        "list_including_deleted": names(db.list_projects(include_deleted=True)),
        "hard_delete": db.delete_project(other["id"], hard_delete=True),
        "hard_delete_missing": db.delete_project(other["id"], hard_delete=True),
    }


def _prompts(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    project = ids["project"]
    structured = {
        "schema_version": 1,
        "blocks": [{"id": "b1", "name": "sys", "role": "system", "kind": "instructions", "content": "Be brief", "enabled": True, "order": 0}],
        "variables": [],
    }
    second = db.create_prompt(project, "second", system_prompt="s2", few_shot_examples=[{"q": 1}], modules_config={"m": True})
    stub_id = second["id"] + 5
    db.ensure_prompt_stub(prompt_id=stub_id, project_id=project)
    return {
        "create": second,
        "create_duplicate": _outcome(lambda: db.create_prompt(project, "second")),
        # ids are left out below: PostgreSQL burns a sequence value on each failed insert.
        "create_structured": _outcome(
            lambda: {
                k: v
                for k, v in db.create_prompt(
                    project, "structured", prompt_format="structured", prompt_schema_version=1, prompt_definition=structured
                ).items()
                if k != "id"
            }
        ),
        "create_legacy_with_definition": _outcome(lambda: db.create_prompt(project, "bad", prompt_definition=structured)),
        "get_missing": _outcome(lambda: db.get_prompt(99999)),
        "get_with_project": db.get_prompt_with_project(second["id"]),
        "get_with_project_missing": _outcome(lambda: db.get_prompt_with_project(99999)),
        "list_paged": [p["name"] for p in db.list_prompts(project, page=1, per_page=2)["prompts"]],
        "stub": db.get_prompt(stub_id),
        "stub_again_is_noop": db.ensure_prompt_stub(prompt_id=stub_id, project_id=project, name="other"),
        # A stub inserts an explicit id; creates after it must not collide with it.
        "create_after_stub": [_outcome(lambda n=n: db.create_prompt(project, f"after-{n}")["name"]) for n in range(8)],
    }


def _test_cases(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)  # project with test case "c"
    project = ids["project"]
    sig = db.create_signature(project, "sig", input_schema=[], output_schema=[])
    golden = db.create_test_case(
        project, "golden one", inputs={"q": "capital of France"}, expected_outputs={}, description="Geography Quiz",
        tags=["geo", "easy"], is_golden=True, signature_id=sig["id"],
    )
    plain = db.create_test_case(project, "plain", inputs={"q": 2}, tags=["math"], is_generated=True)

    def names(rows: Any) -> list[str]:
        return [r["name"] for r in (rows["test_cases"] if isinstance(rows, dict) else rows)]

    return {
        "create": golden,
        "create_duplicate": _outcome(lambda: db.create_test_case(project, "plain", inputs={})),
        "create_blank": _outcome(lambda: db.create_test_case(project, "  ", inputs={})),
        "create_missing_project": _outcome(lambda: db.create_test_case(99999, "orphan", inputs={})),
        "get_missing": _outcome(lambda: db.get_test_case(99999)),
        "by_ids": names(db.get_test_cases_by_ids([plain["id"], golden["id"], plain["id"]])),
        "list_golden_first": names(db.list_test_cases(project)),
        "list_golden_only": names(db.list_test_cases(project, is_golden=True)),
        "list_by_signature": names(db.list_test_cases(project, signature_id=sig["id"])),
        "list_tags": names(db.list_test_cases(project, tags=["math"])),
        "list_search": names(db.list_test_cases(project, search="QUIZ")),
        "list_paged": db.list_test_cases(project, page=2, per_page=1, return_pagination=True)["pagination"],
        "search_fts": names(db.search_test_cases(project, "France")),
        "by_signature": names(db.get_test_cases_by_signature(sig["id"])),
        "golden": names(db.get_golden_test_cases(project)),
        "stats": _outcome(lambda: db.get_test_case_stats(project)),
        "update": _outcome(
            lambda: db.update_test_case(plain["id"], {"expected_outputs": {"a": 4}, "tags": ["x"], "is_golden": True, "bogus": 1})
        ),
        "update_rename_to_duplicate": _outcome(lambda: db.update_test_case(plain["id"], {"name": "c"})),
        "update_noop": db.update_test_case(plain["id"], {"bogus": 1})["name"],
        "update_missing": _outcome(lambda: db.update_test_case(99999, {"name": "x"})),
        "soft_delete": db.delete_test_case(plain["id"]),
        "soft_delete_again": db.delete_test_case(plain["id"]),
        "reuse_deleted_name": _outcome(lambda: db.create_test_case(project, "plain", inputs={})["name"]),
        "update_deleted": _outcome(lambda: db.update_test_case(plain["id"], {"name": "y"})),
        "hard_delete": db.delete_test_case(golden["id"], hard_delete=True),
        "hard_delete_missing": db.delete_test_case(golden["id"], hard_delete=True),
        "bulk": names(db.create_bulk_test_cases(project, [{"name": "b1", "inputs": {}}, {"name": "b2", "inputs": {"z": 1}}])),
        "bulk_with_duplicate": _outcome(
            lambda: names(db.create_bulk_test_cases(project, [{"name": "b3", "inputs": {}}, {"name": "b1", "inputs": {}}]))
        ),
        "bulk_is_atomic": "b3" in names(db.list_test_cases(project, per_page=100)),
    }


def _optimizations(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    project, prompt = ids["project"], ids["prompt"]
    opt = db.create_optimization(
        project_id=project, name="o1", initial_prompt_id=prompt, optimizer_type="mipro",
        optimization_config={"k": 1}, max_iterations=5, bootstrap_samples=2,
    )
    other = db.create_optimization(project_id=project, name="o2", initial_prompt_id=prompt, optimizer_type="bootstrap")
    oid = opt["id"]
    return {
        "create": opt,
        "get_missing": _outcome(lambda: db.get_optimization(99999)),
        "running": db.set_optimization_status(oid, "running", mark_started=True)["status"],
        "iteration_1": db.record_optimization_iteration(oid, iteration_number=1, prompt_variant={"v": 1}, metrics={"acc": 0.5}, tokens_used=3, cost=0.1, note="n"),
        "iteration_2": db.record_optimization_iteration(oid, iteration_number=2, metrics={"acc": 0.7})["iteration_number"],
        "iterations_page": db.list_optimization_iterations(oid, page=1, per_page=1),
        "update_config": db.update_optimization(oid, {"optimization_config": {"k": 2}, "iterations_completed": 2})["optimization_config"],
        "complete": db.complete_optimization(oid, optimized_prompt_id=prompt, iterations_completed=2, final_metrics={"acc": 0.7}, improvement_percentage=40.0),
        "complete_again_is_guarded": _outcome(
            lambda: db.complete_optimization(oid, iterations_completed=9, _return_transition_applied=True)[1]
        ),
        "fail_after_complete_is_guarded": db.set_optimization_status(oid, "failed", error_message="late")["status"],
        "cancel_pending": db.set_optimization_status(other["id"], "cancelled")["status"],
        "update_missing": _outcome(lambda: db.update_optimization(99999, {"status": "failed"})),
        "update_bad_column": _outcome(lambda: db.update_optimization(oid, {"status = 'x', name": "y"})),
        "update_expected_uuid_mismatch": _outcome(
            lambda: db.update_optimization(other["id"], {"name": "z"}, expected_uuid="nope", _return_transition_applied=True)[1]
        ),
        "list": [o["name"] for o in db.list_optimizations(project_id=project)["optimizations"]],
        "list_status": [o["name"] for o in db.list_optimizations(status="completed")["optimizations"]],
        "list_paged": db.list_optimizations(page=2, per_page=1)["pagination"],
    }


def _jobs(db: PromptStudioDatabase) -> dict[str, Any]:
    ids = _seed(db)
    low = db.create_job("evaluation", 1, {"a": 1}, project_id=ids["project"], priority=1)
    high = db.create_job("optimization", 2, None, project_id=ids["project"], priority=9)
    db.create_job("evaluation", 1, {"a": 2}, project_id=ids["project"], priority=1)

    def summary(job: Any) -> Any:
        return None if job is None else {k: job.get(k) for k in ("id", "job_type", "status", "lease_owner", "retry_count")}

    first = db.acquire_next_job(worker_id="worker-a")
    return {
        "create": low,
        "create_null_payload": high["payload"],
        "acquire_takes_highest_priority": summary(first),
        "renew_by_owner": db.renew_job_lease(high["id"], seconds=30, worker_id="worker-a"),
        "renew_by_other_worker": db.renew_job_lease(high["id"], seconds=30, worker_id="worker-b"),
        "renew_not_processing": db.renew_job_lease(low["id"], seconds=30),
        "acquire_next": summary(db.acquire_next_job(worker_id="  worker-b  ")),
        "complete": summary(db.update_job_status(high["id"], "completed", result={"ok": True})),
        "completed_result": db.get_job(high["id"])["result"],
        "fail": summary(db.update_job_status(low["id"], "failed", error_message="boom")),
        "retry": db.retry_job_record(low["id"]),
        "after_retry": summary(db.get_job(low["id"])),
        "retry_missing": db.retry_job_record(99999),
        "update_missing": db.update_job_status(99999, "failed"),
        "get_by_uuid": summary(db.get_job_by_uuid(high["uuid"])),
        "get_missing": db.get_job(99999),
        "list": [summary(j) for j in db.list_jobs()],
        "list_filtered": [summary(j) for j in db.list_jobs(status="queued", job_type="evaluation")],
        "latest_for_entity": summary(db.get_latest_job_for_entity("evaluation", 1)),
        "for_entity_desc": [j["id"] for j in db.list_jobs_for_entity("evaluation", 1, ascending=False)],
        "cleanup_keeps_recent": db.cleanup_jobs(older_than_days=30),
        "drain": [summary(db.acquire_next_job()) for _ in range(3)],
    }


def _reads(db: PromptStudioDatabase) -> dict[str, Any]:
    # Read paths shared by the later aggregates; pins that PostgreSQL does not leak
    # its tsvector columns through `SELECT *` / `RETURNING *`.
    ids = _seed(db)
    return {
        "list_projects": db.list_projects(),
        "update_project": db.update_project(ids["project"], {"description": "x"}),
        "get_prompt": db.get_prompt(ids["prompt"]),
        "list_prompts": db.list_prompts(ids["project"]),
        "get_test_case": db.get_test_case(ids["case"]),
        "list_test_cases": db.list_test_cases(ids["project"]),
    }


SCENARIOS: dict[str, Callable[[PromptStudioDatabase], dict[str, Any]]] = {
    "test_runs": _test_runs,
    "prompt_versions": _prompt_versions,
    "evaluations": _evaluations,
    "projects": _projects,
    "prompts": _prompts,
    "optimizations": _optimizations,
    "jobs": _jobs,
    "reads": _reads,
    "test_cases": _test_cases,
    "signatures": _signatures,
}


@pytest.fixture
def both_backends(tmp_path, prompt_studio_pg_shared_db):
    sqlite_db = PromptStudioDatabase(str(tmp_path / "parity.sqlite"), prompt_studio_pg_shared_db.client_id)
    _reset_prompt_studio_tables(prompt_studio_pg_shared_db)
    try:
        yield sqlite_db, prompt_studio_pg_shared_db
    finally:
        sqlite_db.close()
        prompt_studio_pg_shared_db.close_connection()


@pytest.mark.parametrize("aggregate", sorted(SCENARIOS))
def test_backends_return_the_same_results(aggregate, both_backends):
    sqlite_db, pg_db = both_backends
    on_sqlite = _normalise(SCENARIOS[aggregate](sqlite_db))
    on_postgres = _normalise(SCENARIOS[aggregate](pg_db))
    diffs = [
        f"{step}: sqlite={on_sqlite[step]!r} postgres={on_postgres[step]!r}"
        for step in on_sqlite
        if on_sqlite[step] != on_postgres[step]
    ]
    assert not diffs, f"{aggregate} differs between backends:\n" + "\n".join(diffs)


def test_moved_writes_retry_transient_contention(tmp_path, monkeypatch):
    """Repositories retry through retry_policy; one transient lock must not fail a write."""
    import sqlite3

    from tldw_Server_API.app.core.DB_Management import retry_policy

    monkeypatch.setattr(retry_policy.time, "sleep", lambda _s: None)
    db = PromptStudioDatabase(str(tmp_path / "retry.sqlite"), "retry")
    try:
        ids = _seed(db)
        real_exec = db._impl._cursor_exec
        failures = {"left": 1}

        def flaky(conn, query, params=None):
            if "INSERT INTO prompt_studio_test_runs" in query and failures["left"]:
                failures["left"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return real_exec(conn, query, params)

        monkeypatch.setattr(db._impl, "_cursor_exec", flaky)
        run = db.create_test_run(
            project_id=ids["project"], prompt_id=ids["prompt"], test_case_id=ids["case"], model_name="m",
            inputs={}, outputs={},
        )
        assert run["model_name"] == "m"
        assert failures["left"] == 0
    finally:
        db.close()


def test_update_evaluation_rejects_columns_outside_the_allowlist(tmp_path):
    """Keys of `updates` become column names in the UPDATE, so they are allowlisted."""
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import InputError

    db = PromptStudioDatabase(str(tmp_path / "allow.sqlite"), "allow")
    try:
        ids = _seed(db)
        ev = db.create_evaluation(prompt_id=ids["prompt"], project_id=ids["project"])
        with pytest.raises(InputError):
            db.update_evaluation(ev["id"], {"status = 'failed', client_id": "x"})
        assert db.get_evaluation(ev["id"])["status"] == "running"
    finally:
        db.close()
