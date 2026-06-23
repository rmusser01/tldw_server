import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_orphan_requeue_resumes_run(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    definition = {
        "name": "orphan-integration",
        "version": 1,
        "steps": [
            {"id": "s1", "type": "prompt", "config": {"template": "Alpha"}},
            {"id": "s2", "type": "prompt", "config": {"template": "Prev: {{ last.text }}"}},
        ],
    }
    run_id = "run-orphan-integration"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )

    step1_run_id = f"{run_id}:s1:1"
    db.create_step_run(
        step_run_id=step1_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s1",
        name="s1",
        step_type="prompt",
    )
    db.complete_step_run(step_run_id=step1_run_id, status="succeeded", outputs={"text": "Alpha"})
    step1_ended_at = db._conn.execute(
        "SELECT ended_at FROM workflow_step_runs WHERE step_run_id = ?",
        (step1_run_id,),
    ).fetchone()[0]

    step2_run_id = f"{run_id}:s2:2"
    db.create_step_run(
        step_run_id=step2_run_id,
        tenant_id="default",
        run_id=run_id,
        step_id="s2",
        name="s2",
        step_type="prompt",
    )
    old = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(seconds=120)).isoformat()
    db._conn.execute(
        "UPDATE workflow_step_runs SET started_at = ?, heartbeat_at = ?, status = 'running' WHERE step_run_id = ?",
        (step1_ended_at, old, step2_run_id),
    )
    db._conn.commit()

    engine = WorkflowEngine(db)
    await engine._reap_orphans()

    deadline = time.time() + 3.0
    status = None
    while time.time() < deadline:
        run = db.get_run(run_id)
        status = run.status if run else None
        if status in {"succeeded", "failed"}:
            break
        await asyncio.sleep(0.05)

    assert status == "succeeded"
    run = db.get_run(run_id)
    outputs = json.loads(run.outputs_json or "{}")
    assert outputs.get("text") == "Prev: Alpha"
