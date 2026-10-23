"""On PostgreSQL, a notes step in a run continued after its creator returned succeeds (TASK-13897).

An approval request or a voice command starts ``continue_run`` as a detached task inside its
ChaCha operation, then returns. The notes adapter opens a ChaCha handle per call on the event
loop; on dev that handle joined the closed operation, and the step failed.
"""

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.adapters.knowledge import crud
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def notes_on_postgres(pg_database_config, tmp_path, monkeypatch):
    """Every ChaCha handle the notes adapter opens uses the PostgreSQL content backend, whose checkouts are counted."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    taken, returned = Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(crud, "is_test_mode", lambda: False)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    try:
        yield SimpleNamespace(outstanding=lambda: sum((taken - returned).values()))
    finally:
        pool.close_all()


async def test_a_notes_step_in_a_run_continued_after_its_creator_returned(notes_on_postgres, tmp_path) -> None:
    """On dev the handle's constructor raised 'The ChaCha operation has finished': the run failed, with no note."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    note = {"action": "create", "title": "From a resumed run", "content": "Saved after the request returned."}
    db.create_run(
        run_id="run",
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "run",
            "version": 1,
            "steps": [
                {"id": "review", "type": "log", "config": {"message": "approved"}},
                {"id": "note", "type": "notes", "config": note},
            ],
        },
    )
    db.update_run_status("run", status="waiting_human")
    with chacha_operation(independent=True):  # the approval request, or the voice command
        task = asyncio.create_task(WorkflowEngine(db).continue_run("run", after_step_id="review"))
    await asyncio.wait_for(task, timeout=60)  # the task's first step runs after the operation closed

    finished = db.get_run("run")
    assert (finished.status, finished.error) == ("succeeded", None)
    saved = json.loads(finished.outputs_json)["note"]
    assert (saved["title"], saved["content"]) == (note["title"], note["content"])
    assert notes_on_postgres.outstanding() == 0
