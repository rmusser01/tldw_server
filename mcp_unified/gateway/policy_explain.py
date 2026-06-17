"""Redacted policy explanation service for the standalone MCP gateway."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mcp_unified.interfaces.storage import AuditStore
from mcp_unified.profiles import MCPProfile, explain_profile_tool_decision
from mcp_unified.profiles.subjects import (
    PermissionSubjectLimitError,
    extract_permission_rule_subjects,
)
from mcp_unified.storage.models import AuditEvent

from .policy_simulation import simulate_tool_call_policy

MAX_POLICY_EXPLAIN_ARGUMENT_BYTES = 64 * 1024

PolicyExplainOutcome = Literal["deny", "ask", "allow"]
PolicyExplainRedactionState = Literal["plain", "sanitized", "redacted"]


class PolicyExplainMode(str, Enum):
    """How closely an explanation should mirror runtime policy state."""

    RUNTIME_EFFECTIVE = "runtime_effective"
    STATIC_PROFILE = "static_profile"


class PolicyExplainRequest(BaseModel):
    """Request to explain one profile/tool policy decision."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    capability: str | None = None
    session_id: str | None = None
    mode: PolicyExplainMode = PolicyExplainMode.RUNTIME_EFFECTIVE

    @field_validator("profile_id", "tool_name")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("capability", "session_id")
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("arguments must be an object")
        return dict(value)

    @model_validator(mode="after")
    def _validate_argument_size(self) -> PolicyExplainRequest:
        _validate_serialized_argument_size(self.arguments)
        return self


class ProfileToolPreviewRequest(BaseModel):
    """Request to preview profile decisions across a tool catalog."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    capability: str | None = None
    mode: PolicyExplainMode = PolicyExplainMode.RUNTIME_EFFECTIVE

    @field_validator("profile_id")
    @classmethod
    def _validate_profile_id(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("capability")
    @classmethod
    def _normalize_capability(cls, value: str | None) -> str | None:
        return _optional_text(value)


class PolicyExplainSubject(BaseModel):
    """One redacted subject considered by runtime permission rules."""

    model_config = ConfigDict(extra="forbid")

    type: str
    value: str
    redaction_state: PolicyExplainRedactionState
    outcome: str | None = None
    reason_code: str | None = None
    matched_rules: list[dict[str, Any]] = Field(default_factory=list)


class PolicyExplainResponse(BaseModel):
    """Redacted explanation for one profile/tool policy decision."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    profile_id: str
    tool_name: str
    mode: PolicyExplainMode
    final_outcome: PolicyExplainOutcome
    reason_code: str
    visibility: str
    call_state: str
    requires_approval: bool
    runtime_status: str | None = None
    runtime_reason_code: str | None = None
    subjects: list[PolicyExplainSubject] = Field(default_factory=list)
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    redacted: bool = True


class ProfileToolPreviewEntry(BaseModel):
    """Redacted profile decision for one catalog or policy-listed tool."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    final_outcome: PolicyExplainOutcome
    reason_code: str
    visibility: str
    call_state: str
    requires_approval: bool
    redacted: bool = True


class ProfileToolPreviewSummary(BaseModel):
    """Count summary for profile tool preview entries."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    allow: int = 0
    ask: int = 0
    deny: int = 0


class ProfileToolPreviewResponse(BaseModel):
    """Redacted preview of effective profile tool decisions."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    profile_id: str
    mode: PolicyExplainMode
    entries: list[ProfileToolPreviewEntry] = Field(default_factory=list)
    summary: ProfileToolPreviewSummary = Field(default_factory=ProfileToolPreviewSummary)
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    redacted: bool = True


class PolicyExplainErrorResponse(BaseModel):
    """Stable error envelope for policy explanation failures."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = False
    error: str
    reason_code: str


class GatewayPolicyExplainError(RuntimeError):
    """Expected policy explanation service failure."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

    def to_payload(self) -> PolicyExplainErrorResponse:
        """Return a stable redacted error payload."""

        return PolicyExplainErrorResponse(error=str(self), reason_code=self.reason_code)


class GatewayPolicyExplainService:
    """Assemble redacted policy explanations with strict audit events."""

    def __init__(
        self,
        *,
        profile_resolver: Callable[[str], MCPProfile | Awaitable[MCPProfile | None] | None],
        audit_store: AuditStore | None,
        actor_id: str | None = None,
        catalog_provider: Callable[[], Iterable[Any] | Awaitable[Iterable[Any]]] | None = None,
        policy_grant_store: Any | None = None,
    ) -> None:
        self.profile_resolver = profile_resolver
        self.audit_store = audit_store
        self.actor_id = actor_id
        self.catalog_provider = catalog_provider
        self.policy_grant_store = policy_grant_store

    async def explain_tool_call(
        self,
        request: PolicyExplainRequest,
    ) -> PolicyExplainResponse:
        """Return a redacted explanation for one tool call policy check."""

        profile = await self._resolve_profile(request.profile_id)
        simulator_result = simulate_tool_call_policy(
            profile,
            request.tool_name,
            request.arguments,
            capability=request.capability,
            policy_grant_store=self.policy_grant_store,
            session_id=request.session_id,
        )
        tool_explanation = explain_profile_tool_decision(
            profile,
            request.tool_name,
            capability=request.capability,
        )
        subjects, degraded_reasons = _policy_subjects_from_simulation(
            simulator_result,
            tool_name=request.tool_name,
            arguments=request.arguments,
        )
        overall = _mapping_value(simulator_result, "overall")
        response = PolicyExplainResponse(
            profile_id=profile.id,
            tool_name=request.tool_name,
            mode=request.mode,
            final_outcome=tool_explanation.final_outcome,
            reason_code=tool_explanation.reason_code,
            visibility=tool_explanation.visibility,
            call_state=tool_explanation.call_state,
            requires_approval=tool_explanation.requires_approval,
            runtime_status=(
                _mapping_value(overall, "status")
                if isinstance(overall, Mapping)
                else None
            ),
            runtime_reason_code=(
                _mapping_value(overall, "reason_code")
                if isinstance(overall, Mapping)
                else None
            ),
            subjects=subjects,
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons,
        )
        await _append_audit_event_strict(
            self.audit_store,
            event_type="policy.explain.requested",
            actor_id=self.actor_id,
            profile_id=profile.id,
            target_type="tool",
            target_id=request.tool_name,
            payload=response.model_dump(mode="json", exclude_none=True),
        )
        return response

    async def preview_profile_tools(
        self,
        request: ProfileToolPreviewRequest,
    ) -> ProfileToolPreviewResponse:
        """Return a redacted profile tool preview, degrading without a catalog."""

        profile = await self._resolve_profile(request.profile_id)
        degraded_reasons: list[str] = []
        tool_names = await self._preview_tool_names(profile)
        if self.catalog_provider is None:
            degraded_reasons.append("catalog_unavailable")

        entries = [
            _profile_tool_preview_entry(profile, tool_name, capability=request.capability)
            for tool_name in tool_names
        ]
        response = ProfileToolPreviewResponse(
            profile_id=profile.id,
            mode=request.mode,
            entries=entries,
            summary=_preview_summary(entries),
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons,
        )
        await _append_audit_event_strict(
            self.audit_store,
            event_type="policy.preview.requested",
            actor_id=self.actor_id,
            profile_id=profile.id,
            target_type="profile",
            target_id=profile.id,
            payload=response.model_dump(mode="json", exclude_none=True),
        )
        return response

    async def _resolve_profile(self, profile_id: str) -> MCPProfile:
        profile = await _maybe_await(self.profile_resolver(profile_id))
        if profile is None:
            raise GatewayPolicyExplainError(
                "Profile not found",
                reason_code="profile_not_found",
            )
        return profile

    async def _preview_tool_names(self, profile: MCPProfile) -> list[str]:
        if self.catalog_provider is not None:
            catalog = await _maybe_await(self.catalog_provider())
            return sorted(_tool_names_from_catalog(catalog))

        policy_document = profile.policy_document
        policy_tools = set(policy_document.allowed_tools or [])
        policy_tools.update(policy_document.denied_tools or [])
        return sorted(tool_name for tool_name in policy_tools if isinstance(tool_name, str))


async def _append_audit_event_strict(
    audit_store: AuditStore | None,
    *,
    event_type: str,
    actor_id: str | None,
    profile_id: str | None,
    target_type: str | None,
    target_id: str | None,
    payload: Mapping[str, Any],
) -> None:
    """Append an audit event or fail closed when audit is unavailable."""

    if audit_store is None:
        raise GatewayPolicyExplainError(
            "Audit store unavailable",
            reason_code="audit_store_unavailable",
        )
    event = AuditEvent(
        id=f"policy-explain-{uuid4().hex}",
        event_type=event_type,
        actor_id=actor_id,
        profile_id=profile_id,
        target_type=target_type,
        target_id=target_id,
        payload=dict(payload),
        provenance={"source": "gateway_policy_explain_service"},
        created_at=datetime.now(timezone.utc),
    )
    try:
        await audit_store.append_event(event)
    except Exception as exc:  # noqa: BLE001
        raise GatewayPolicyExplainError(
            "Audit store unavailable",
            reason_code="audit_store_unavailable",
        ) from exc


def _redact_subject_value(
    subject_type: str,
    value: Any,
) -> tuple[str, PolicyExplainRedactionState]:
    """Return a safe subject value and its redaction state."""

    if not isinstance(value, str):
        return "[redacted]", "redacted"

    subject = subject_type.strip().lower()
    text = value.strip()
    if not text:
        return "", "plain"
    if subject == "path":
        return _redact_path(text)
    if subject == "command":
        return "[redacted-command]", "redacted"
    if subject == "domain":
        return _redact_domain(text)
    if subject in {"tool", "mcp", "skill", "agent", "capability", "risk_class"}:
        return text, "plain"
    return "[redacted]", "redacted"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _policy_subjects_from_simulation(
    simulator_result: Mapping[str, Any],
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[list[PolicyExplainSubject], list[str]]:
    degraded_reasons: list[str] = []
    raw_subjects = [
        item
        for item in _sequence_value(simulator_result, "subjects")
        if _mapping_value(item, "subject_type") != "tool"
    ]
    if not raw_subjects:
        try:
            raw_subjects = [
                {
                    "subject_type": subject_type,
                    "value": value,
                    "outcome": None,
                    "reason_code": None,
                    "matched_rules": [],
                }
                for subject_type, value, _argv in extract_permission_rule_subjects(
                    tool_name,
                    arguments,
                )
                if subject_type != "tool"
            ]
        except PermissionSubjectLimitError:
            degraded_reasons.append("subject_limit_exceeded")

    subjects: list[PolicyExplainSubject] = []
    seen: set[tuple[str, str]] = set()
    for raw_subject in raw_subjects:
        subject_type = _mapping_value(raw_subject, "subject_type")
        raw_value = _mapping_value(raw_subject, "value")
        if not isinstance(subject_type, str):
            continue
        value, redaction_state = _redact_subject_value(subject_type, raw_value)
        key = (subject_type, value)
        if key in seen:
            continue
        seen.add(key)
        subjects.append(
            PolicyExplainSubject(
                type=subject_type,
                value=value,
                redaction_state=redaction_state,
                outcome=_optional_string(_mapping_value(raw_subject, "outcome")),
                reason_code=_optional_string(_mapping_value(raw_subject, "reason_code")),
                matched_rules=[
                    _redacted_rule(rule)
                    for rule in _sequence_value(raw_subject, "matched_rules")
                    if isinstance(rule, Mapping)
                ],
            )
        )
    return subjects, degraded_reasons


def _redacted_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    rule_type = _optional_string(rule.get("rule_type"))
    payload: dict[str, Any] = {}
    for key in ("source", "rule_type", "outcome", "reason_code"):
        value = _optional_string(rule.get(key))
        if value is not None:
            payload[key] = value
    pattern = _optional_string(rule.get("pattern"))
    if pattern is not None:
        if rule_type is None:
            payload["pattern"] = "[redacted]"
        else:
            payload["pattern"] = _redact_subject_value(rule_type, pattern)[0]
    return payload


def _profile_tool_preview_entry(
    profile: MCPProfile,
    tool_name: str,
    *,
    capability: str | None,
) -> ProfileToolPreviewEntry:
    explanation = explain_profile_tool_decision(
        profile,
        tool_name,
        capability=capability,
    )
    return ProfileToolPreviewEntry(
        tool_name=_redact_tool_identifier(tool_name),
        final_outcome=explanation.final_outcome,
        reason_code=explanation.reason_code,
        visibility=explanation.visibility,
        call_state=explanation.call_state,
        requires_approval=explanation.requires_approval,
    )


def _preview_summary(
    entries: Iterable[ProfileToolPreviewEntry],
) -> ProfileToolPreviewSummary:
    entry_list = list(entries)
    summary = ProfileToolPreviewSummary(total=len(entry_list))
    for entry in entry_list:
        if entry.final_outcome == "allow":
            summary.allow += 1
        elif entry.final_outcome == "ask":
            summary.ask += 1
        elif entry.final_outcome == "deny":
            summary.deny += 1
    return summary


def _tool_names_from_catalog(catalog: Iterable[Any]) -> set[str]:
    tool_names: set[str] = set()
    for item in catalog:
        tool_name: Any
        if isinstance(item, str):
            tool_name = item
        elif isinstance(item, Mapping):
            tool_name = item.get("name") or item.get("tool_name")
        else:
            tool_name = getattr(item, "name", None) or getattr(item, "tool_name", None)
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.add(tool_name.strip())
    return tool_names


def _redact_path(value: str) -> tuple[str, PolicyExplainRedactionState]:
    normalized = value.replace("\\", "/")
    if "\n" in normalized or "\r" in normalized:
        return "[redacted-path]", "redacted"
    if _is_absolute_path(normalized):
        filename = normalized.rstrip("/").rsplit("/", 1)[-1]
        return (f".../{filename}" if filename else "[path]"), "sanitized"
    return normalized, "plain"


def _redact_domain(value: str) -> tuple[str, PolicyExplainRedactionState]:
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or "[domain]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        return f"{parsed.scheme}://{host}", "sanitized"
    sanitized = value.split("?", 1)[0].split("#", 1)[0]
    if "@" in sanitized:
        sanitized = sanitized.rsplit("@", 1)[-1]
    state: PolicyExplainRedactionState = "sanitized" if sanitized != value else "plain"
    return sanitized, state


def _is_absolute_path(value: str) -> bool:
    return bool(
        value.startswith("/")
        or value.startswith("~")
        or re.match(r"^[A-Za-z]:/", value)
    )


def _redact_tool_identifier(tool_name: str) -> str:
    stripped = tool_name.strip()
    if "\n" in stripped or "\r" in stripped:
        return "[redacted-tool]"
    for command_tool in ("Bash", "Shell", "PowerShell", "Monitor"):
        if stripped.startswith(f"{command_tool}(") and stripped.endswith(")"):
            return f"{command_tool}([redacted-command])"
    return stripped


def _validate_serialized_argument_size(arguments: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        arguments,
        default=lambda _value: "[non-json-value]",
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized.encode("utf-8")) > MAX_POLICY_EXPLAIN_ARGUMENT_BYTES:
        raise ValueError("arguments exceed maximum serialized size")


def _sequence_value(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return list(item)
    return []


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _required_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string")
    text = value.strip()
    if not text:
        raise ValueError("value cannot be blank")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _optional_string(value)


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


__all__ = [
    "GatewayPolicyExplainError",
    "GatewayPolicyExplainService",
    "MAX_POLICY_EXPLAIN_ARGUMENT_BYTES",
    "PolicyExplainErrorResponse",
    "PolicyExplainMode",
    "PolicyExplainRequest",
    "PolicyExplainResponse",
    "PolicyExplainSubject",
    "ProfileToolPreviewEntry",
    "ProfileToolPreviewRequest",
    "ProfileToolPreviewResponse",
    "ProfileToolPreviewSummary",
    "_append_audit_event_strict",
    "_redact_subject_value",
]
