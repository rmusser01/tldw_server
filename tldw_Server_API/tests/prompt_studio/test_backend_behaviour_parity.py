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
    "reads": _reads,
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
    for step in on_sqlite:
        assert on_sqlite[step] == on_postgres[step], f"{aggregate}.{step} differs between backends"
