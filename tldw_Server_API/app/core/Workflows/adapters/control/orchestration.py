"""Workflow orchestration adapters: workflow_call.

These adapters handle sub-workflow execution.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.control._config import WorkflowCallConfig


@registry.register(
    "workflow_call",
    category="control",
    description="Call another workflow as a sub-workflow",
    parallelizable=False,
    tags=["control", "orchestration"],
    config_model=WorkflowCallConfig,
)
async def run_workflow_call_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Call another workflow as a sub-workflow.

    Config:
      - workflow_id: str
      - inputs: dict (inputs for sub-workflow)
      - wait: bool (default True) - wait for completion
      - timeout_seconds: int (default 300)
    Output: { "run_id": str, "status": str, "outputs": dict }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    workflow_id = config.get("workflow_id")
    if not workflow_id:
        return {"error": "missing_workflow_id", "result": None}

    inputs = config.get("inputs") or {}
    wait = config.get("wait", True)
    timeout_seconds = int(config.get("timeout_seconds", 300))

    run_id = None
    try:
        from tldw_Server_API.app.core.Workflows.workflows_db import get_workflows_db

        db = get_workflows_db()
        workflow = db.get_workflow(workflow_id)
        if not workflow:
            return {"error": f"workflow_not_found: {workflow_id}", "result": None}

        # Create a sub-run
        import uuid
        run_id = str(uuid.uuid4())
        tenant_id = context.get("tenant_id", "default")
        user_id = context.get("user_id")

        db.create_run(
            run_id=run_id,
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            user_id=user_id,
            inputs=inputs,
            status="pending",
        )

        if wait:
            from tldw_Server_API.app.core.Workflows.engine import EngineConfig, WorkflowEngine

            engine = WorkflowEngine(db=db, config=EngineConfig(tenant_id=tenant_id))
            await asyncio.wait_for(engine.start_run(run_id, mode="sync"), timeout=timeout_seconds)

            run = db.get_run(run_id)
            if run:
                return {"run_id": run_id, "status": run.status, "outputs": run.outputs or {}, "result": run.outputs}
            return {"run_id": run_id, "status": "unknown", "result": None}
        else:
            # Async execution
            from tldw_Server_API.app.core.Workflows.engine import EngineConfig, WorkflowEngine

            engine = WorkflowEngine(db=db, config=EngineConfig(tenant_id=tenant_id))
            engine.submit(run_id, mode="async")
            return {"run_id": run_id, "status": "submitted", "async": True}

    except asyncio.TimeoutError:
        return {"error": "workflow_timeout", "run_id": run_id}
    except Exception:
        logger.exception("Workflow call error")
        return {"error": "workflow_call_error", "result": None}
