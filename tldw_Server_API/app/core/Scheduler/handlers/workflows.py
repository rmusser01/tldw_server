"""
Scheduler task handlers for Workflows integration.

Registers a `workflow_run` task that can be scheduled or enqueued to
start a workflow run from the internal Workflows engine.

Location: tldw_Server_API/app/core/Scheduler/handlers/workflows.py
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import resolve_user_id_value
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_workflows_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Scheduler.base.registry import task
from tldw_Server_API.app.core.Workflows.daily_ledger import record_workflow_run
from tldw_Server_API.app.core.Workflows.engine import RunMode, WorkflowEngine


def _get_wf_db() -> WorkflowsDatabase:
    backend = get_content_backend_instance()
    return create_workflows_database(backend=backend)


def _resolve_payload_user_id(payload: dict[str, Any]) -> str:
    return resolve_user_id_value(
        payload.get("user_id"),
        missing_detail="workflow_run: user_id is required in multi-user mode",
    )


@task(name="workflow_run", max_retries=0, timeout=3600, queue="workflows")
async def workflow_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Scheduler handler that enqueues/executes a Workflows run.

    Expected payload keys:
      - workflow_id: int (optional if definition_snapshot provided)
      - inputs: dict
      - user_id: str (owner of the run; used for RBAC/attribution)
      - tenant_id: str (optional; defaults to 'default')
      - mode: 'async'|'sync' (default 'async')
      - validation_mode: 'block'|'non-block' (default 'block')
      - definition_snapshot: dict (optional; ad-hoc run definition)
      - metadata: dict (optional; persisted on the run and exposed to adapters)
      - secrets: dict[str,str] (optional; injected ephemerally into engine context)

    Returns:
      { "run_id": str, "status": "queued"|terminal, "succeeded": bool | None }
    """
    db = _get_wf_db()

    # Validate payload minimal shape
    inputs = payload.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError("workflow_run: inputs must be a dict")

    user_id = _resolve_payload_user_id(payload)
    tenant_id = str(payload.get("tenant_id") or "default")
    workflow_id = payload.get("workflow_id")
    definition_snapshot = payload.get("definition_snapshot")
    if workflow_id is None and not definition_snapshot:
        raise ValueError("workflow_run: must provide workflow_id or definition_snapshot")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if isinstance(definition_snapshot, dict):
        definition_snapshot = dict(definition_snapshot)
        existing_definition_metadata = definition_snapshot.get("metadata")
        merged_definition_metadata = {
            **(existing_definition_metadata if isinstance(existing_definition_metadata, dict) else {}),
            **metadata,
        }
        if merged_definition_metadata:
            definition_snapshot["metadata"] = merged_definition_metadata

    run_mode = str(payload.get("mode") or "async").lower()
    mode = RunMode.SYNC if run_mode == "sync" else RunMode.ASYNC
    validation_mode = str(payload.get("validation_mode") or "block")

    # Create run
    run_id = __import__("uuid").uuid4().hex
    try:
        db.create_run(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            inputs=inputs,
            workflow_id=int(workflow_id) if workflow_id is not None else None,
            definition_version=None,
            definition_snapshot=definition_snapshot,
            idempotency_key=None,
            session_id=None,
            validation_mode=validation_mode,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"workflow_run: failed to create run: {e}")
        raise

    # Shadow-write this run into the daily ledger so RG daily caps account for
    # scheduled runs as well. Fail open if ledger unavailable.
    with contextlib.suppress(Exception):
        await record_workflow_run(entity_scope="user", entity_value=str(user_id), run_id=run_id, units=1)

    # Inject secrets ephemerally
    secrets = payload.get("secrets")
    try:
        if isinstance(secrets, dict):
            WorkflowEngine.set_run_secrets(run_id, secrets)  # ephemeral; cleared on terminal state
    except Exception as secrets_error:
        logger.debug("Workflow scheduler failed to cache run secrets", exc_info=secrets_error)

    # Submit to engine (respect internal concurrency scheduler)
    engine = WorkflowEngine(db=db)
    engine.submit(run_id, mode=mode)

    # Optionally wait for terminal state when mode=sync
    if mode == RunMode.SYNC:
        # Poll for completion with backoff (bounded by task timeout)
        deadline = __import__("time").time() + 55 * 60  # 55m safety within 60m timeout
        status: str | None = None
        while __import__("time").time() < deadline:
            r = db.get_run(run_id)
            status = r.status if r else None
            if status in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.5)
        succeeded = status == "succeeded"
        return {"run_id": run_id, "status": status or "unknown", "succeeded": succeeded}

    return {"run_id": run_id, "status": "queued"}
