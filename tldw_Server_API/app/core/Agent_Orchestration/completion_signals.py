"""Structured completion signal parsing for ACP orchestration runs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_COMPLETION_MARKER_RE = re.compile(
    r"<acp-task-completion>\s*(?P<payload>.*?)\s*</acp-task-completion>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REVIEW_MARKER_RE = re.compile(
    r"<acp-review-decision>\s*(?P<payload>.*?)\s*</acp-review-decision>",
    flags=re.IGNORECASE | re.DOTALL,
)

_DIRECT_PAYLOAD_KEYS = (
    "taskCompletion",
    "task_completion",
    "completionSignal",
    "completion_signal",
)
_DIRECT_REVIEW_KEYS = (
    "reviewDecision",
    "review_decision",
    "review",
)

_ACCEPTED_STATUSES = {"completed", "complete", "succeeded", "success", "accepted"}
_REJECTED_STATUSES = {"rejected", "failed", "failure", "incomplete"}


class CompletionSignalValidationError(ValueError):
    """Raised when an ACP task completion signal is missing or invalid."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class ReviewDecisionValidationError(ValueError):
    """Raised when an ACP reviewer decision signal is missing or invalid."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TaskCompletionSignal:
    """Validated ACP orchestration completion signal."""

    status: str
    summary: str
    artifacts: list[Any] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskReviewDecision:
    """Validated ACP reviewer decision."""

    approved: bool
    feedback: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


def _coerce_payload(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate)
    if isinstance(candidate, str):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise CompletionSignalValidationError(
                "malformed",
                f"malformed ACP completion signal JSON: {exc.msg}",
            ) from exc
        if isinstance(loaded, dict):
            return loaded
    raise CompletionSignalValidationError(
        "malformed",
        "malformed ACP completion signal: payload must be a JSON object",
    )


def _raise_review_error(reason: str, message: str) -> None:
    raise ReviewDecisionValidationError(reason, message)


def _iter_text_candidates(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_iter_text_candidates(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in ("text", "content", "message", "output"):
            if key in value:
                out.extend(_iter_text_candidates(value[key]))
        return out
    return []


def _extract_payload(acp_result: dict[str, Any]) -> dict[str, Any]:
    for key in _DIRECT_PAYLOAD_KEYS:
        if key in acp_result:
            return _coerce_payload(acp_result[key])

    text_candidates: list[str] = []
    for key in ("content", "message", "output", "text"):
        if key in acp_result:
            text_candidates.extend(_iter_text_candidates(acp_result[key]))

    for text in text_candidates:
        match = _COMPLETION_MARKER_RE.search(text)
        if match is None:
            continue
        try:
            return _coerce_payload(match.group("payload"))
        except CompletionSignalValidationError:
            raise

    raise CompletionSignalValidationError(
        "missing",
        "missing ACP completion signal: expected taskCompletion or <acp-task-completion>{...}</acp-task-completion>",
    )


def _extract_review_payload(acp_result: dict[str, Any]) -> dict[str, Any]:
    for key in _DIRECT_REVIEW_KEYS:
        if key in acp_result:
            try:
                return _coerce_payload(acp_result[key])
            except CompletionSignalValidationError as exc:
                raise ReviewDecisionValidationError(exc.reason, str(exc)) from exc

    text_candidates: list[str] = []
    for key in ("content", "message", "output", "text"):
        if key in acp_result:
            text_candidates.extend(_iter_text_candidates(acp_result[key]))

    for text in text_candidates:
        match = _REVIEW_MARKER_RE.search(text)
        if match is None:
            continue
        try:
            return _coerce_payload(match.group("payload"))
        except CompletionSignalValidationError as exc:
            raise ReviewDecisionValidationError(exc.reason, str(exc)) from exc

    raise ReviewDecisionValidationError(
        "missing",
        "missing ACP review decision: expected reviewDecision or <acp-review-decision>{...}</acp-review-decision>",
    )


def validate_task_completion_signal(acp_result: dict[str, Any]) -> TaskCompletionSignal:
    """Return a validated task completion signal from an ACP prompt result.

    The orchestration contract accepts either a direct structured field such as
    ``taskCompletion`` or a text marker:
    ``<acp-task-completion>{"status":"completed","summary":"..."}</acp-task-completion>``.
    A returned prompt or stop reason alone is deliberately insufficient.
    """
    if not isinstance(acp_result, dict):
        raise CompletionSignalValidationError(
            "malformed",
            "malformed ACP completion signal: prompt result must be an object",
        )

    payload = _extract_payload(acp_result)
    status = str(payload.get("status") or payload.get("outcome") or "").strip().lower()
    summary = str(payload.get("summary") or payload.get("result_summary") or "").strip()
    artifacts = payload.get("artifacts") or []

    if status in _REJECTED_STATUSES:
        detail = f": {summary}" if summary else ""
        raise CompletionSignalValidationError(
            "rejected",
            f"rejected ACP completion signal{detail}",
        )
    if status not in _ACCEPTED_STATUSES:
        raise CompletionSignalValidationError(
            "malformed",
            "malformed ACP completion signal: status must be completed",
        )
    if not isinstance(artifacts, list):
        raise CompletionSignalValidationError(
            "malformed",
            "malformed ACP completion signal: artifacts must be a list",
        )

    return TaskCompletionSignal(
        status="completed",
        summary=summary or "ACP task completed",
        artifacts=list(artifacts),
        raw_payload=payload,
    )


def validate_review_decision_signal(acp_result: dict[str, Any]) -> TaskReviewDecision:
    """Return a validated reviewer decision from an ACP prompt result."""
    if not isinstance(acp_result, dict):
        _raise_review_error(
            "malformed",
            "malformed ACP review decision: prompt result must be an object",
        )

    payload = _extract_review_payload(acp_result)
    raw_approved = payload.get("approved")
    status = str(payload.get("status") or payload.get("decision") or "").strip().lower()

    if isinstance(raw_approved, bool):
        approved = raw_approved
    elif status in {"approved", "approve", "accepted", "pass", "passed"}:
        approved = True
    elif status in {"rejected", "reject", "failed", "fail", "needs_work"}:
        approved = False
    else:
        _raise_review_error(
            "malformed",
            "malformed ACP review decision: approved boolean or approved/rejected status is required",
        )

    feedback = str(payload.get("feedback") or payload.get("summary") or "").strip()
    if not feedback:
        feedback = "Reviewer approved task" if approved else "Reviewer rejected task"

    return TaskReviewDecision(
        approved=approved,
        feedback=feedback,
        raw_payload=payload,
    )
