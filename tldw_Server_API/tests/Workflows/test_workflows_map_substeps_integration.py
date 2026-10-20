import asyncio
import json
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Workflows.adapters.control import flow
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine, RunMode


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_map_adapter_supported_substep(tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    definition = {
        "name": "map-substeps",
        "version": 1,
        "steps": [
            {
                "id": "map1",
                "type": "map",
                "config": {
                    "items": [1, 2],
                    "step": {"type": "prompt", "config": {"template": "Item {{ item }}"}},
                    "concurrency": 2,
                },
            }
        ],
    }
    run_id = "run-map-substeps"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )

    engine = WorkflowEngine(db)
    await engine.start_run(run_id, RunMode.SYNC)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "succeeded"
    outputs = json.loads(run.outputs_json or "{}")
    assert outputs.get("count") == 2
    results = outputs.get("results") or []
    assert results and results[0].get("text") == "Item 1"


@pytest.mark.asyncio
async def test_a_failed_map_step_leaves_no_item_running(tmp_path):
    """on_failure lets the run go on; on dev the failed map's other items ran on and wrote to the run (TASK-13863)."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    definition = {
        "name": "map-cancels-items",
        "version": 1,
        "steps": [
            {
                "id": "map1",
                "type": "map",
                "config": {"items": [0, 1, 2, 3, 4, 5], "step": {"type": "summarize", "config": {}}, "concurrency": 2},
                "on_failure": "after",
            },
            # The next step releases the map's items; on dev they are still running when it does.
            {"id": "after", "type": "parallel", "config": {"steps": [{"type": "summarize", "config": {"release": True}}]}},
        ],
    }
    run_id = "run-map-cancels-items"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="user",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot=definition,
    )
    release = asyncio.Event()

    async def sub_step(config, context):
        if config.get("release"):
            release.set()
            for _ in range(20):
                await asyncio.sleep(0)
            return {"released": True}
        if context["item"] == 0:
            await asyncio.sleep(0)
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )
        await release.wait()
        context["append_event"]("map_item_finished", {"item": context["item"]})
        return {"text": str(context["item"])}

    with patch.object(flow, "get_adapter", return_value=sub_step):
        await WorkflowEngine(db).start_run(run_id, RunMode.SYNC)

    run = db.get_run(run_id)
    assert run is not None
    assert run.status == "succeeded"
    assert [(step["step_id"], step["status"]) for step in db.list_step_runs(run_id=run_id)] == [
        ("map1", "failed"),
        ("after", "succeeded"),
    ]
    assert "map_item_finished" not in [event["event_type"] for event in db.get_events(run_id)]
