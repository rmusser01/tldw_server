"""Regression: list_optimizations must exist on the SQLite backend.

The method was implemented only on _BackendPromptStudioDatabase (PostgreSQL) while the
PromptStudioDatabase facade delegates unconditionally. ADR-020 makes SQLite the default
content backend, so GET /api/v1/prompt-studio/projects/{id}/optimizations raised
AttributeError and returned HTTP 500 on every request, for every project, on a default
deployment. Both existing endpoint tests pass because they substitute a stub.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = Path(tmp.name)
    instance = PromptStudioDatabase(str(path), "test_client")
    yield instance
    try:
        instance.close()
    except Exception:
        pass
    for target in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        target.unlink(missing_ok=True)


def test_facade_defines_list_optimizations_itself() -> None:
    """Since TASK-13318 the method lives once, in the optimizations repository, and the
    facade calls it directly -- it no longer depends on either implementation having it."""
    assert "list_optimizations" in vars(PromptStudioDatabase), (
        "list_optimizations must be defined on the facade, not reached via __getattr__"
    )


def test_list_optimizations_returns_created_records(db) -> None:
    project = db.create_project(name="proj", description="d")
    created = db.create_optimization(
        project_id=project["id"],
        name="opt-1",
        initial_prompt_id=None,
        optimizer_type="bootstrap",
    )

    result = db.list_optimizations(project_id=project["id"], page=1, per_page=20)

    assert isinstance(result, dict)
    names = [o.get("name") for o in result["optimizations"]]
    assert "opt-1" in names, f"created optimization missing from {names}"
    assert result["pagination"]["total"] >= 1
    assert result["pagination"]["page"] == 1
    assert result["pagination"]["per_page"] == 20
    assert created["id"] in [o.get("id") for o in result["optimizations"]]


def test_list_optimizations_filters_and_paginates(db) -> None:
    project = db.create_project(name="proj2")
    for i in range(3):
        db.create_optimization(
            project_id=project["id"],
            name=f"opt-{i}",
            initial_prompt_id=None,
            optimizer_type="bootstrap",
            status="pending" if i < 2 else "running",
        )

    running = db.list_optimizations(project_id=project["id"], status="running")
    assert running["pagination"]["total"] == 1

    page1 = db.list_optimizations(project_id=project["id"], page=1, per_page=2)
    assert len(page1["optimizations"]) == 2
    assert page1["pagination"]["total"] == 3
    assert page1["pagination"]["total_pages"] == 2

    page2 = db.list_optimizations(project_id=project["id"], page=2, per_page=2)
    assert len(page2["optimizations"]) == 1


def test_list_optimizations_rejects_bad_pagination(db) -> None:
    from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import InputError

    with pytest.raises(InputError):
        db.list_optimizations(page=0)
    with pytest.raises(InputError):
        db.list_optimizations(per_page=0)
