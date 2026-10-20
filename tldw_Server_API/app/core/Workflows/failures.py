from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

from tldw_Server_API.app.core.Workflows.capabilities import StepCapability, get_step_capability

_NON_RETRIABLE_REASONS: set[str] = {
    "validation_error",
    "authz_error",
    "acp_governance_blocked",
    "session_access_denied",
    "invariant_violation",
    "402",  # an LLM quota refusal (LLMQuotaExceededError): the limit does not reset between attempts
}

_POLICY_REASONS: set[str] = {
    "acp_governance_blocked",
    "authz_error",
    "session_access_denied",
}

_DEFINITION_REASONS: set[str] = {
    "validation_error",
    "invariant_violation",
    "assigned_to_required",
    "branch_loop_exceeded",
}


@dataclass(frozen=True)
class FailureEnvelope:
    reason_code_core: str
    reason_code_detail: str | None
    category: str
    blame_scope: str
    retryable: bool
    retry_recommendation: str
    error_summary: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def normalize_reason_code(error: BaseException | str | None) -> str:
    if error is None:
        return ""
    text = str(error).strip().lower()
    if not text:
        return ""
    for sep in (":", ";", "|", "\n", "\t", " "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text


def is_retriable_reason(reason_code: str | None) -> bool:
    normalized = normalize_reason_code(reason_code)
    if not normalized:
        return True
    return normalized not in _NON_RETRIABLE_REASONS


def build_failure_envelope(
    error: BaseException | str | None,
    *,
    step_type: str,
) -> FailureEnvelope:
    capability = get_step_capability(step_type)
    core = normalize_reason_code(error) or "runtime_error"
    detail = _extract_reason_detail(error, core)
    retryable = is_retriable_reason(core)
    category = _classify_category(core)
    blame_scope = _classify_blame_scope(core, capability)
    retry_recommendation = _classify_retry_recommendation(
        retryable=retryable,
        category=category,
        capability=capability,
    )
    error_summary = _summarize_error(error)
    return FailureEnvelope(
        reason_code_core=core,
        reason_code_detail=detail,
        category=category,
        blame_scope=blame_scope,
        retryable=retryable,
        retry_recommendation=retry_recommendation,
        error_summary=error_summary,
    )


def _extract_reason_detail(error: BaseException | str | None, core: str) -> str | None:
    if error is None:
        return None
    raw = str(error).strip()
    if not raw:
        return None
    lowered = raw.lower()
    for sep in (":", ";", "|", "\n", "\t"):
        if sep in raw:
            tail = raw.split(sep, 1)[1].strip()
            return tail or None
        if sep in lowered:
            tail = lowered.split(sep, 1)[1].strip()
            return tail or None
    exc_name = type(error).__name__.strip().lower() if isinstance(error, BaseException) else ""
    if exc_name and exc_name != core:
        return exc_name
    return None


def _classify_category(reason_code_core: str) -> str:
    if reason_code_core in _POLICY_REASONS:
        return "policy"
    if reason_code_core in _DEFINITION_REASONS:
        return "definition"
    if "timeout" in reason_code_core:
        return "timeout"
    if reason_code_core == "cancelled_by_user":
        return "cancellation"
    return "runtime"


def _classify_blame_scope(reason_code_core: str, capability: StepCapability) -> str:
    if reason_code_core in _POLICY_REASONS:
        return "policy"
    if reason_code_core in _DEFINITION_REASONS:
        return "workflow"
    if capability.idempotency_strategy == "external":
        return "external_dependency"
    return "step"


def _classify_retry_recommendation(
    *,
    retryable: bool,
    category: str,
    capability: StepCapability,
) -> str:
    if not retryable or category in {"definition", "policy"}:
        return "unsafe"
    if capability.requires_human_review_for_rerun:
        return "conditional"
    if capability.replay_safe:
        return "safe"
    return "unsafe"


def _summarize_error(error: BaseException | str | None, *, limit: int = 240) -> str:
    if error is None:
        return ""
    text = str(error).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
