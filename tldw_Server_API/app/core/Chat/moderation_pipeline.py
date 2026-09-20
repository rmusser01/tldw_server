"""Output safety processing for non-streaming chat completions."""
from __future__ import annotations

import asyncio
import inspect
import json as _json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.response_processor import (
    NonStreamChoice,
    apply_redaction_to_content,
    extract_text_from_content,
    primary_choice,
    set_choice_content,
)
from tldw_Server_API.app.core.Moderation.moderation_service import get_moderation_service
from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service

_OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ChatAPIError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    _json.JSONDecodeError,
    asyncio.CancelledError,
)

ReviewItemCallback = Callable[..., Awaitable[None] | None]
CompletionMetricCallback = Callable[[str], None]


@dataclass(slots=True)
class OutputModerationRuntime:
    request: Any | None
    client_id: str
    conversation_id: Any | None
    metrics: Any
    audit_service: Any | None = None
    audit_context: Any | None = None
    moderation_getter: Callable[[], Any] | None = None
    self_monitoring_service: Any | None = None
    topic_monitoring_getter: Callable[[], Any | None] | None = get_topic_monitoring_service
    review_item_callback: ReviewItemCallback | None = None
    completion_metric_callback: CompletionMetricCallback | None = None
    audit_event_type: Any | None = AuditEventType.SECURITY_VIOLATION


@dataclass(slots=True)
class OutputSafetyResult:
    content_to_save: Any | None
    content_text_for_usage: str


@dataclass(slots=True)
class _ModerationDecision:
    action: str | None = None
    sample: str | None = None
    redacted_value: Any | None = None
    category: str | None = None
    matched_pattern: str | None = None


async def apply_output_safety_to_choices(
    *,
    choices: list[NonStreamChoice],
    fallback_content: Any | None,
    fallback_content_text: str,
    runtime: OutputModerationRuntime,
) -> OutputSafetyResult:
    """Apply self-monitoring, moderation, and monitoring side effects to output content."""
    current_content = fallback_content
    current_text = fallback_content_text

    def refresh_current() -> None:
        nonlocal current_content, current_text
        first_choice = primary_choice(choices)
        if first_choice is not None:
            current_content = first_choice.content
            current_text = first_choice.content_text
            return
        current_text = extract_text_from_content(current_content)

    def update_fallback(content: Any | None) -> None:
        nonlocal current_content, current_text
        current_content = content
        current_text = extract_text_from_content(content)

    await _apply_output_self_monitoring(
        choices=choices,
        current_text=lambda: current_text,
        update_fallback=update_fallback,
        runtime=runtime,
    )
    refresh_current()

    moderation_result = await _apply_output_moderation(
        choices=choices,
        get_fallback=lambda: current_content,
        get_fallback_text=lambda: current_text,
        update_fallback=update_fallback,
        refresh_current=refresh_current,
        runtime=runtime,
    )
    if moderation_result is not None:
        current_content = moderation_result.content_to_save
        current_text = moderation_result.content_text_for_usage
    else:
        refresh_current()

    return OutputSafetyResult(
        content_to_save=current_content,
        content_text_for_usage=current_text,
    )


async def write_mandatory_moderation_audit(
    *,
    audit_service: Any | None,
    audit_context: Any | None,
    audit_event_type: Any | None,
    action: str,
    result: str,
    metadata: dict[str, Any],
) -> None:
    """Write a moderation audit event and fail closed if persistence is unavailable."""
    if audit_service is None or audit_context is None:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    try:
        await audit_service.log_event(
            event_type=audit_event_type or AuditEventType.SECURITY_VIOLATION,
            context=audit_context,
            action=action,
            result=result,
            metadata=metadata,
        )
        await audit_service.flush(raise_on_failure=True)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        logger.error(
            "Mandatory moderation audit write failed for {} ({}): {}",
            action,
            result,
            exc,
            exc_info=True,
        )
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc


async def _apply_output_self_monitoring(
    *,
    choices: list[NonStreamChoice],
    current_text: Callable[[], str],
    update_fallback: Callable[[Any | None], None],
    runtime: OutputModerationRuntime,
) -> None:
    if not runtime.self_monitoring_service or not (choices or current_text()):
        return

    try:
        user_id = _request_state_attr(runtime.request, "user_id")
        sm_user = str(user_id or runtime.client_id)
        loop = asyncio.get_running_loop()
        sm_targets: list[NonStreamChoice | None] = list(choices) if choices else [None]
        for choice in sm_targets:
            sm_text = choice.content_text if choice is not None else current_text()
            if not sm_text:
                continue
            sm_result = await loop.run_in_executor(
                None,
                lambda text=sm_text: runtime.self_monitoring_service.check_text(
                    text=text,
                    user_id=sm_user,
                    phase="output",
                    conversation_id=_source_id(runtime),
                ),
            )
            if sm_result.action == "block":
                _emit_completion_metric(runtime, "blocked")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=sm_result.block_message or "Output blocked by self-monitoring rule",
                )
            if sm_result.action == "redact" and sm_result.redacted_text is not None:
                if choice is not None:
                    set_choice_content(choice, sm_result.redacted_text)
                else:
                    update_fallback(sm_result.redacted_text)
    except HTTPException:
        raise
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Self-monitoring output check skipped: {exc}")


async def _apply_output_moderation(
    *,
    choices: list[NonStreamChoice],
    get_fallback: Callable[[], Any | None],
    get_fallback_text: Callable[[], str],
    update_fallback: Callable[[Any | None], None],
    refresh_current: Callable[[], None],
    runtime: OutputModerationRuntime,
) -> OutputSafetyResult | None:
    try:
        moderation_targets: list[NonStreamChoice | None] = list(choices) if choices else [None]
        if not any(
            (target.content_text if target is not None else get_fallback_text())
            for target in moderation_targets
        ):
            return None

        moderation_getter = runtime.moderation_getter or get_moderation_service
        moderation = moderation_getter()
        req_user_id = _request_state_attr(runtime.request, "user_id")
        eff_policy = moderation.get_effective_policy(
            str(req_user_id) if req_user_id is not None else runtime.client_id
        )
        if not eff_policy.enabled or not eff_policy.output_enabled:
            return None

        for target in moderation_targets:
            target_content = target.content if target is not None else get_fallback()
            target_text = target.content_text if target is not None else get_fallback_text()
            if not target_text:
                continue

            decision = _evaluate_output_moderation(
                moderation=moderation,
                eff_policy=eff_policy,
                target_text=target_text,
            )
            _schedule_output_topic_monitoring(
                runtime=runtime,
                req_user_id=req_user_id,
                text=target_text,
            )

            if decision.action == "block":
                await _handle_output_block(
                    runtime=runtime,
                    eff_policy=eff_policy,
                    decision=decision,
                    req_user_id=req_user_id,
                )
            if decision.action == "redact":
                redacted_content = _redact_output_content(
                    moderation=moderation,
                    eff_policy=eff_policy,
                    target_content=target_content,
                    target_text=target_text,
                    decision=decision,
                )
                _track_output_redaction(runtime, req_user_id, decision)
                await _audit_output_redaction(runtime, decision)
                if target is not None:
                    set_choice_content(target, redacted_content)
                else:
                    update_fallback(redacted_content)
                refresh_current()
            if decision.action == "warn":
                await _capture_review_item(
                    runtime,
                    phase="output",
                    action="warn",
                    excerpt=decision.sample,
                    category=decision.category,
                    matched_pattern=decision.matched_pattern,
                    effective_policy=eff_policy,
                    source_id=_source_id(runtime),
                    user_id=_topic_user_id(req_user_id, runtime.client_id),
                )
        refresh_current()
        first_choice = primary_choice(choices)
        if first_choice is not None:
            return OutputSafetyResult(first_choice.content, first_choice.content_text)
        fallback_content = get_fallback()
        return OutputSafetyResult(fallback_content, extract_text_from_content(fallback_content))
    except HTTPException:
        raise
    except MandatoryAuditWriteError:
        raise
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Moderation output processing error: {exc}")
        return None


def _evaluate_output_moderation(
    *,
    moderation: Any,
    eff_policy: Any,
    target_text: str,
) -> _ModerationDecision:
    action = None
    sample = None
    redacted_value = None
    category = None
    matched_pattern = None
    match_span = None

    if hasattr(moderation, "evaluate_action_with_match"):
        try:
            eval_res = moderation.evaluate_action_with_match(target_text, eff_policy, "output")
            if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                action, redacted_value, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                category = eval_res[3] if len(eval_res) >= 4 else None
                match_span = eval_res[4] if len(eval_res) >= 5 else None
            else:
                action, redacted_value, matched_pattern = eval_res  # type: ignore[misc]
        except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
            action = None
        if match_span and hasattr(moderation, "build_sanitized_snippet"):
            try:
                sample = moderation.build_sanitized_snippet(target_text, eff_policy, match_span, matched_pattern)
            except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
                sample = None
    elif hasattr(moderation, "evaluate_action"):
        try:
            eval_res = moderation.evaluate_action(target_text, eff_policy, "output")
            if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                action, redacted_value, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                category = eval_res[3] if len(eval_res) >= 4 else None
            else:
                action, redacted_value, matched_pattern = eval_res  # type: ignore[misc]
        except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
            action = None

    if action and action != "pass" and sample is None:
        try:
            _, sample = moderation.check_text(target_text, eff_policy, "output")
        except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
            sample = None
    if not action:
        flagged, sample = moderation.check_text(target_text, eff_policy, "output")
        if flagged:
            action = eff_policy.output_action
            redacted_value = (
                moderation.redact_text(target_text, eff_policy)
                if action == "redact"
                else None
            )

    return _ModerationDecision(
        action=action,
        sample=sample,
        redacted_value=redacted_value,
        category=category,
        matched_pattern=matched_pattern,
    )


async def _handle_output_block(
    *,
    runtime: OutputModerationRuntime,
    eff_policy: Any,
    decision: _ModerationDecision,
    req_user_id: Any | None,
) -> None:
    if runtime.audit_service and runtime.audit_context:
        await write_mandatory_moderation_audit(
            audit_service=runtime.audit_service,
            audit_context=runtime.audit_context,
            audit_event_type=runtime.audit_event_type,
            action="moderation.output",
            result="failure",
            metadata={
                "phase": "output",
                "streaming": False,
                "action": "block",
                "pattern": decision.sample,
            },
        )
        await _capture_review_item(
            runtime,
            phase="output",
            action="block",
            excerpt=decision.sample,
            category=decision.category,
            matched_pattern=decision.matched_pattern,
            effective_policy=eff_policy,
            source_id=_source_id(runtime),
            user_id=_topic_user_id(req_user_id, runtime.client_id),
        )
    _emit_completion_metric(runtime, "blocked")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Output violates moderation policy",
    )


async def _audit_output_redaction(
    runtime: OutputModerationRuntime,
    decision: _ModerationDecision,
) -> None:
    if not runtime.audit_service or not runtime.audit_context:
        return
    await write_mandatory_moderation_audit(
        audit_service=runtime.audit_service,
        audit_context=runtime.audit_context,
        audit_event_type=runtime.audit_event_type,
        action="moderation.output",
        result="success",
        metadata={
            "phase": "output",
            "streaming": False,
            "action": "redact",
            "pattern": decision.sample,
        },
    )


def _redact_output_content(
    *,
    moderation: Any,
    eff_policy: Any,
    target_content: Any | None,
    target_text: str,
    decision: _ModerationDecision,
) -> Any | None:
    if isinstance(target_content, str):
        return (
            decision.redacted_value
            if isinstance(decision.redacted_value, str)
            else moderation.redact_text(target_text, eff_policy)
        )
    return apply_redaction_to_content(
        target_content,
        lambda text: moderation.redact_text(text, eff_policy),
    )


def _track_output_redaction(
    runtime: OutputModerationRuntime,
    req_user_id: Any | None,
    decision: _ModerationDecision,
) -> None:
    if decision.sample is None:
        return
    try:
        runtime.metrics.track_moderation_output(
            str(req_user_id or runtime.client_id),
            "redact",
            streaming=False,
            category=(decision.category or "default"),
        )
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
        pass


async def _capture_review_item(runtime: OutputModerationRuntime, **kwargs: Any) -> None:
    if runtime.review_item_callback is None:
        return
    try:
        result = runtime.review_item_callback(**kwargs)
        if inspect.isawaitable(result):
            await result
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Moderation review capture failed in chat moderation pipeline: {}: {}", type(exc).__name__, str(exc))


def _schedule_output_topic_monitoring(
    *,
    runtime: OutputModerationRuntime,
    req_user_id: Any | None,
    text: str,
) -> None:
    try:
        if runtime.topic_monitoring_getter is None:
            return
        mon = runtime.topic_monitoring_getter()
        team_ids = _request_state_attr(runtime.request, "team_ids")
        org_ids = _request_state_attr(runtime.request, "org_ids")
        if mon is not None and text:
            mon.schedule_evaluate_and_alert(
                user_id=_topic_user_id(req_user_id, runtime.client_id),
                text=text,
                source="chat.output",
                scope_type="user",
                scope_id=_topic_user_id(req_user_id, runtime.client_id),
                team_ids=team_ids,
                org_ids=org_ids,
                source_id=_source_id(runtime),
            )
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Topic monitoring (non-stream final) skipped: {exc}")


def _request_state_attr(request: Any | None, attr: str) -> Any | None:
    try:
        if request is not None and hasattr(request, "state"):
            return getattr(request.state, attr, None)
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS:
        return None
    return None


def _topic_user_id(req_user_id: Any | None, client_id: str) -> str | None:
    return str(req_user_id or client_id) if (req_user_id or client_id) else None


def _source_id(runtime: OutputModerationRuntime) -> str | None:
    return str(runtime.conversation_id) if runtime.conversation_id else None


def _emit_completion_metric(runtime: OutputModerationRuntime, outcome: str) -> None:
    if runtime.completion_metric_callback is None:
        return
    try:
        runtime.completion_metric_callback(outcome)
    except _OUTPUT_SAFETY_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Chat output safety completion metric failed: {}", exc)
