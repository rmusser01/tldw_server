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
    {"uuid", "created_at", "updated_at", "last_modified", "started_at", "completed_at", "deleted_at"}
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
    "reads": _reads,
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
