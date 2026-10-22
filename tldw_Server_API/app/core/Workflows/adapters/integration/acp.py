"""ACP integration adapter.

This module includes adapters for Agent Client Protocol operations:
- acp_stage: Execute a pipeline stage via ACP session prompt
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPGovernanceDeniedError,
    get_runner_client,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content, resolve_context_user_id
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import ACPStageConfig
from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store

_ACP_STAGE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_ACP_OUTPUT_SCHEMA_VERSION = "1.0"


def _validate_acp_output_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize ACP stage outputs to a stable, versioned contract."""
    payload["acp_output_schema_version"] = _ACP_OUTPUT_SCHEMA_VERSION
    payload.setdefault("status", "error")
    payload.setdefault("stage", "")
    payload.setdefault("session_id", None)
    payload.setdefault("workspace_id", None)
    payload.setdefault("workspace_group_id", None)
    if not isinstance(payload.get("response"), dict):
        payload["response"] = {}
    if not isinstance(payload.get("usage"), dict):
        payload["usage"] = {}
    if not isinstance(payload.get("governance"), dict):
        payload["governance"] = {}
    return payload


def _render_optional_template(value: Any, context: dict[str, Any]) -> str | None:
    """Render an optional templated value and return a stripped string.

    Args:
        value: Candidate value from config/context. Non-string values are stringified.
        context: Workflow execution context used for template rendering.

    Returns:
        Rendered non-empty string, or None when the value is empty/missing.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    rendered = apply_template_to_string(value, context) or value
    text = str(rendered).strip()
    return text or None


def _resolve_str_field(config: dict[str, Any], context: dict[str, Any], key: str) -> str | None:
    """Resolve a string field from config first, then context, with templating."""
    if key in config:
        return _render_optional_template(config.get(key), context)
    return _render_optional_template(context.get(key), context)


def _normalize_error_payload(
    *,
    status: str,
    error_type: str,
    message: str,
    stage: str,
    session_id: str | None,
    workspace_id: str | None,
    workspace_group_id: str | None,
    governance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized ACP error/blocked payload for branchable outcomes."""
    payload: dict[str, Any] = {
        "status": status,
        "error_type": error_type,
        "error": message,
        "stage": stage,
        "session_id": session_id,
        "workspace_id": workspace_id,
        "workspace_group_id": workspace_group_id,
        "response": {},
        "usage": {},
        "governance": governance or {},
    }
    return payload


def _maybe_raise_if_configured(result: dict[str, Any], fail_on_error: bool) -> dict[str, Any]:
    """Raise AdapterError for error/blocked states when fail_on_error is enabled."""
    result = _validate_acp_output_contract(result)
    if not fail_on_error:
        return result
    status = str(result.get("status") or "").strip().lower()
    if status in {"error", "blocked"}:
        err = str(result.get("error_type") or "acp_stage_error")
        raise AdapterError(err)
    return result


def _render_prompt(config: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    """Build an ACP prompt payload from explicit prompt, template, or workflow fallbacks."""
    prompt = config.get("prompt")
    if isinstance(prompt, list):
        rendered: list[dict[str, Any]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            out_msg: dict[str, Any] = {}
            for key, value in msg.items():
                if isinstance(value, str):
                    out_msg[key] = apply_template_to_string(value, context) or value
                else:
                    out_msg[key] = value
            if out_msg:
                rendered.append(out_msg)
        if rendered:
            return rendered
    if isinstance(prompt, str):
        rendered_prompt = apply_template_to_string(prompt, context) or prompt
        return [{"role": "user", "content": rendered_prompt}]

    prompt_template = str(config.get("prompt_template") or "").strip()
    if prompt_template:
        rendered_template = apply_template_to_string(prompt_template, context) or prompt_template
        return [{"role": "user", "content": rendered_template}]

    last = context.get("last") or {}
    fallback = ""
    if isinstance(last, dict):
        fallback = str(last.get("text") or last.get("content") or "").strip()
    if not fallback:
        fallback = str((context.get("inputs") or {}).get("task") or "").strip()
    if fallback:
        return [{"role": "user", "content": fallback}]
    return []


async def _create_session(
    runner: Any,
    *,
    cwd: str,
    agent_type: str | None,
    user_id: int,
    persona_id: str | None,
    workspace_id: str | None,
    workspace_group_id: str | None,
    scope_snapshot_id: str | None,
) -> str:
    """Create an ACP session while preserving compatibility with legacy runners."""
    params: dict[str, Any] = {
        "cwd": cwd,
        "agent_type": agent_type,
        "user_id": user_id,
        "persona_id": persona_id,
        "workspace_id": workspace_id,
        "workspace_group_id": workspace_group_id,
        "scope_snapshot_id": scope_snapshot_id,
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        return await runner.create_session(**params)
    except TypeError:
        # Non-sandbox ACP runner currently only accepts cwd/agent_type/user_id.
        narrowed = {
            "cwd": cwd,
            "agent_type": agent_type,
            "user_id": user_id,
        }
        narrowed = {k: v for k, v in narrowed.items() if v is not None}
        return await runner.create_session(**narrowed)


async def _verify_session_access(
    runner: Any,
    *,
    session_id: str,
    user_id: int,
    stage: str,
    allow_when_verifier_missing: bool = False,
) -> bool:
    """Verify that an ACP session is owned by the workflow user.

    Args:
        runner: ACP runner client instance.
        session_id: Session identifier to authorize.
        user_id: Workflow user id expected to own the session.
        stage: Stage name, used in logs for diagnostics.
        allow_when_verifier_missing: Allow access when a verifier is not implemented.

    Returns:
        True when access is verified (or explicitly allowed), else False.
    """
    verifier = getattr(runner, "verify_session_access", None)
    if not callable(verifier):
        if allow_when_verifier_missing:
            logger.debug(
                "ACP stage adapter: runner missing verify_session_access; allowing session created in adapter. "
                "stage={} session_id={} user_id={}",
                stage,
                session_id,
                user_id,
            )
            return True
        logger.warning(
            "ACP stage adapter: denied session reuse because runner has no verify_session_access. "
            "stage={} session_id={} user_id={}",
            stage,
            session_id,
            user_id,
        )
        return False

    try:
        return bool(await verifier(session_id, user_id))
    except TypeError:
        try:
            return bool(await verifier(session_id=session_id, user_id=user_id))
        except _ACP_STAGE_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "ACP stage adapter: session ownership verification failed. "
                "stage={} session_id={} user_id={} error={}",
                stage,
                session_id,
                user_id,
                type(exc).__name__,
            )
            return False
    except _ACP_STAGE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "ACP stage adapter: session ownership verification failed. "
            "stage={} session_id={} user_id={} error={}",
            stage,
            session_id,
            user_id,
            type(exc).__name__,
        )
        return False


async def _register_workflow_session(
    *,
    session_id: str,
    user_id: int,
    stage: str,
    cwd: str,
    agent_type: str | None,
    persona_id: str | None,
    workspace_id: str | None,
    workspace_group_id: str | None,
    scope_snapshot_id: str | None,
) -> None:
    store = await get_acp_session_store()
    await store.register_session(
        session_id=session_id,
        user_id=int(user_id),
        agent_type=agent_type or "custom",
        name=f"Workflow {stage}",
        cwd=cwd,
        tags=["workflow", "acp_stage", stage],
        persona_id=persona_id,
        workspace_id=workspace_id,
        workspace_group_id=workspace_group_id,
        scope_snapshot_id=scope_snapshot_id,
    )


async def _record_workflow_prompt(
    *,
    session_id: str,
    prompt: list[dict[str, Any]],
    result: dict[str, Any],
    assistant_text: str | None = None,
) -> None:
    store = await get_acp_session_store()
    await store.record_prompt(session_id, prompt, result, assistant_text=assistant_text)


@registry.register(
    "acp_stage",
    category="integration",
    description="Execute an ACP-backed stage via session prompt",
    parallelizable=False,
    tags=["integration", "acp", "pipeline"],
    config_model=ACPStageConfig,
)
async def run_acp_stage_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow stage by prompting an ACP session."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    stage = str(config.get("stage") or "").strip()
    if not stage:
        result = _normalize_error_payload(
            status="error",
            error_type="acp_prompt_error",
            message="missing_stage",
            stage="",
            session_id=None,
            workspace_id=None,
            workspace_group_id=None,
        )
        return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))

    workspace_id = _resolve_str_field(config, context, "workspace_id")
    workspace_group_id = _resolve_str_field(config, context, "workspace_group_id")
    persona_id = _resolve_str_field(config, context, "persona_id")
    scope_snapshot_id = _resolve_str_field(config, context, "scope_snapshot_id")

    review_counter_key = str(config.get("review_counter_key") or "").strip()
    max_review_loops = config.get("max_review_loops")
    if review_counter_key and max_review_loops is not None:
        try:
            current = int(context.get(review_counter_key) or 0)
            maximum = int(max_review_loops)
            if current >= maximum:
                blocked = _normalize_error_payload(
                    status="blocked",
                    error_type="review_loop_exceeded",
                    message="review_loop_exceeded",
                    stage=stage,
                    session_id=None,
                    workspace_id=workspace_id,
                    workspace_group_id=workspace_group_id,
                )
                return _maybe_raise_if_configured(blocked, bool(config.get("fail_on_error")))
            context[review_counter_key] = current + 1
        except _ACP_STAGE_NONCRITICAL_EXCEPTIONS:
            logger.debug("ACP stage adapter: failed to evaluate review loop guard")

    user_id = resolve_context_user_id(context)
    if not user_id:
        result = _normalize_error_payload(
            status="error",
            error_type="acp_session_error",
            message="missing_user_id",
            stage=stage,
            session_id=None,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))
    try:
        workflow_user_id = int(user_id)
    except _ACP_STAGE_NONCRITICAL_EXCEPTIONS:
        result = _normalize_error_payload(
            status="error",
            error_type="acp_session_error",
            message="invalid_user_id",
            stage=stage,
            session_id=None,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))

    session_context_key = str(config.get("session_context_key") or "acp_session_id").strip() or "acp_session_id"
    session_id = _render_optional_template(config.get("session_id"), context) or _render_optional_template(
        context.get(session_context_key), context
    )
    session_created = False
    create_session = bool(config.get("create_session", True))
    runner = await get_runner_client()

    if not session_id:
        if not create_session:
            result = _normalize_error_payload(
                status="error",
                error_type="acp_session_error",
                message="missing_session_id",
                stage=stage,
                session_id=None,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
            )
            return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))

        cwd = _resolve_str_field(config, context, "cwd") or "/workspace"
        agent_type = _resolve_str_field(config, context, "agent_type")
        try:
            session_id = await _create_session(
                runner,
                cwd=cwd,
                agent_type=agent_type,
                user_id=workflow_user_id,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                scope_snapshot_id=scope_snapshot_id,
            )
        except _ACP_STAGE_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "ACP stage adapter: session creation failed. "
                "stage={} user_id={} workspace_id={} workspace_group_id={} error={}",
                stage,
                workflow_user_id,
                workspace_id,
                workspace_group_id,
                type(exc).__name__,
            )
            result = _normalize_error_payload(
                status="error",
                error_type="acp_session_error",
                message="acp_session_create_failed",
                stage=stage,
                session_id=None,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
            )
            return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))
        session_created = True
        context[session_context_key] = session_id
        try:
            await _register_workflow_session(
                session_id=str(session_id),
                user_id=workflow_user_id,
                stage=stage,
                cwd=cwd,
                agent_type=agent_type,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                scope_snapshot_id=scope_snapshot_id,
            )
        except _ACP_STAGE_NONCRITICAL_EXCEPTIONS:
            logger.warning(
                "ACP stage adapter: failed to register workflow-created ACP session in store. "
                "stage={} session_id={} user_id={}",
                stage,
                session_id,
                workflow_user_id,
            )

    access_allowed = await _verify_session_access(
        runner,
        session_id=str(session_id),
        user_id=workflow_user_id,
        stage=stage,
        allow_when_verifier_missing=session_created,
    )
    if not access_allowed:
        denied = _normalize_error_payload(
            status="error",
            error_type="acp_session_error",
            message="session_access_denied",
            stage=stage,
            session_id=session_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(denied, bool(config.get("fail_on_error")))

    prompt_payload = _render_prompt(config, context)
    if not prompt_payload:
        result = _normalize_error_payload(
            status="error",
            error_type="acp_prompt_error",
            message="missing_prompt",
            stage=stage,
            session_id=session_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(result, bool(config.get("fail_on_error")))

    timeout_seconds = int(config.get("timeout_seconds") or 300)
    native_text: list[str] = []
    try:
        raw_response = await asyncio.wait_for(
            runner.prompt(session_id, prompt_payload, on_native_text=native_text.append),
            timeout=float(timeout_seconds),
        )
    except ACPGovernanceDeniedError as exc:
        logger.info(
            "ACP stage adapter: governance blocked prompt. "
            "stage={} session_id={} user_id={} workspace_id={} workspace_group_id={}",
            stage,
            session_id,
            workflow_user_id,
            workspace_id,
            workspace_group_id,
        )
        blocked = _normalize_error_payload(
            status="blocked",
            error_type="acp_governance_blocked",
            message="acp_governance_blocked",
            stage=stage,
            session_id=session_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
            governance=getattr(exc, "governance", None),
        )
        return _maybe_raise_if_configured(blocked, bool(config.get("fail_on_error")))
    except (asyncio.TimeoutError, TimeoutError):
        timed_out = _normalize_error_payload(
            status="error",
            error_type="acp_timeout",
            message="acp_prompt_timeout",
            stage=stage,
            session_id=session_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(timed_out, bool(config.get("fail_on_error")))
    except _ACP_STAGE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "ACP stage adapter: prompt execution failed. "
            "stage={} session_id={} user_id={} workspace_id={} workspace_group_id={} error={}",
            stage,
            session_id,
            workflow_user_id,
            workspace_id,
            workspace_group_id,
            type(exc).__name__,
        )
        failed = _normalize_error_payload(
            status="error",
            error_type="acp_prompt_error",
            message="acp_prompt_failed",
            stage=stage,
            session_id=session_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
        )
        return _maybe_raise_if_configured(failed, bool(config.get("fail_on_error")))

    response = raw_response if isinstance(raw_response, dict) else {"raw_result": raw_response}
    try:
        await _record_workflow_prompt(
            session_id=str(session_id),
            prompt=prompt_payload,
            result=response,
            assistant_text="".join(native_text) if native_text else None,
        )
    except _ACP_STAGE_NONCRITICAL_EXCEPTIONS:
        logger.warning(
            "ACP stage adapter: failed to record prompt in ACP session store. "
            "stage={} session_id={} user_id={}",
            stage,
            session_id,
            workflow_user_id,
        )
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    governance = response.get("governance") if isinstance(response.get("governance"), dict) else {}

    result: dict[str, Any] = {
        "status": "ok",
        "stage": stage,
        "session_id": session_id,
        "workspace_id": workspace_id,
        "workspace_group_id": workspace_group_id,
        "response": response,
        "usage": usage,
        "governance": governance,
    }
    extracted = extract_openai_content(response) or ACPSessionsDB.select_assistant_text(
        response, "".join(native_text) if native_text else None,
    )
    if extracted:
        result["text"] = extracted
    return _validate_acp_output_contract(result)
