"""Scheduler handler for ACP agent runs.

Registered as ``acp_run`` -- creates an ACP session, sends a prompt,
and returns the result.  Intended for scheduled/async agent execution.

Location: tldw_Server_API/app/core/Scheduler/handlers/acp.py
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Scheduler.base.registry import task


@task(name="acp_run", max_retries=1, timeout=7200, queue="acp")
async def acp_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute an ACP agent prompt as a scheduled task.

    Expected payload keys
    ---------------------
    user_id : int
        Owner of the run.
    prompt : str | list[dict]
        The prompt to send to the agent.
    cwd : str, optional
        Working directory for the agent (default ``/workspace``).
    agent_type : str, optional
        Agent type identifier.
    persona_id : str, optional
        Persona context for session creation.
    workspace_id : str, optional
        Workspace context for session creation.
    workspace_group_id : str, optional
        Workspace group context for session creation.
    scope_snapshot_id : str, optional
        Scope snapshot for session creation.

    Returns
    -------
    dict
        ``{"session_id": str|None, "result": Any, "usage": dict,
        "duration_ms": int, "error": str|None}``
    """
    start = time.monotonic()
    user_id = payload.get("user_id")
    prompt = payload.get("prompt", "")

    if not user_id:
        return {
            "session_id": None,
            "result": None,
            "usage": {},
            "duration_ms": 0,
            "error": "Missing user_id",
        }

    # Normalize prompt to message-list format
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    session_id: str | None = None
    try:
        # Lazy import to avoid circular imports at module load time.
        # Mirrors the import pattern used by the ACP stage adapter.
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
            get_runner_client,
        )

        runner = await get_runner_client()

        # Create session -- use the same compatibility shim as the
        # workflow ACP stage adapter (_create_session) so both sandbox
        # and non-sandbox runners are supported.
        create_kwargs: dict[str, Any] = {
            "cwd": payload.get("cwd", "/workspace"),
            "user_id": int(user_id),
        }
        if payload.get("agent_type"):
            create_kwargs["agent_type"] = payload["agent_type"]

        # The sandbox runner's create_session may accept extra kwargs
        # (persona_id, workspace_id, etc.).  Pass them when present,
        # falling back to the narrower signature on TypeError -- the
        # same strategy used in the workflow adapter.
        extra_session_keys = (
            "persona_id",
            "workspace_id",
            "workspace_group_id",
            "scope_snapshot_id",
        )
        extras = {k: payload[k] for k in extra_session_keys if payload.get(k) is not None}
        try:
            session_id = await runner.create_session(**create_kwargs, **extras)
        except TypeError:
            session_id = await runner.create_session(**create_kwargs)

        # Send the prompt
        result = await runner.prompt(session_id, prompt)

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        duration_ms = int((time.monotonic() - start) * 1000)

        return {
            "session_id": session_id,
            "result": result,
            "usage": usage,
            "duration_ms": duration_ms,
            "error": None,
        }

    except Exception as exc:
        logger.error("acp_run failed: {}", exc)
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "session_id": session_id,
            "result": None,
            "usage": {},
            "duration_ms": duration_ms,
            "error": str(exc),
        }
    finally:
        # Always attempt to close the session so resources are freed.
        if session_id is not None:
            with contextlib.suppress(Exception):
                await runner.close_session(session_id)  # type: ignore[possibly-undefined]
