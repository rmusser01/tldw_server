"""Redacted policy explanation service for the standalone MCP gateway."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from mcp_unified.interfaces.storage import AuditStore
from mcp_unified.profiles import MCPProfile, explain_profile_tool_decision
from mcp_unified.profiles.permission_rules import compile_permission_rules
from mcp_unified.profiles.subjects import (
    PermissionSubjectLimitError,
    extract_permission_rule_subjects,
)
from mcp_unified.storage.models import AuditEvent

from .tool_discovery import list_admin_tool_catalog
from .policy_simulation import simulate_tool_call_policy

MAX_POLICY_EXPLAIN_ARGUMENT_BYTES = 64 * 1024
POLICY_EXPLAIN_AUDIT_EVENT_TYPE = "policy.explain.requested"
POLICY_PREVIEW_TOOLS_AUDIT_EVENT_TYPE = "policy.preview_tools.requested"
DEFAULT_AUDIT_ACTOR_ID = "local-cli"

PolicyExplainOutcome = Literal["deny", "ask", "allow"]
PolicyExplainRedactionState = Literal["raw_safe", "sanitized", "redacted", "omitted"]
PolicyExplainVisibility = Literal["visible", "hidden", "deferred"]
STATIC_POLICY_ONLY_SKIPPED_CONTRIBUTORS = [
    "session_grants",
    "approval_grants",
    "runtime_availability",
]

_DEFAULTS_BY_OUTCOME: dict[str, dict[str, Any]] = {
    "deny": {
        "visibility": "hidden",
        "call_state": "blocked",
        "requires_approval": False,
    },
    "ask": {
        "visibility": "visible",
        "call_state": "approval_required",
        "requires_approval": True,
    },
    "allow": {
        "visibility": "visible",
        "call_state": "callable",
        "requires_approval": False,
    },
}


class PolicyExplainMode(str, Enum):
    """How closely an explanation should mirror runtime policy state."""

    RUNTIME_EFFECTIVE = "runtime_effective"
    STATIC_POLICY_ONLY = "static_policy_only"


class PolicyExplainRequest(BaseModel):
    """Request to explain one profile/tool policy decision."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    profile_id: str
    capability: str | None = None
    mode: PolicyExplainMode = PolicyExplainMode.RUNTIME_EFFECTIVE
    limit: int = Field(default=200, ge=1, le=1000)
    cursor: str | None = None
    include_denied: bool = True
    include_recommendations: bool = True
    category: str | None = None

    @field_validator("profile_id")
    @classmethod
    def _validate_profile_id(cls, value: str) -> str:
        return _required_text(value)

    @field_validator("capability")
    @classmethod
    def _normalize_capability(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("category")
    @classmethod
    def _normalize_category(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("cursor")
    @classmethod
    def _normalize_cursor(cls, value: str | None) -> str | None:
        cursor = _optional_text(value)
        if cursor is None:
            return None
        if not cursor.isdigit():
            raise ValueError("cursor must be a non-negative integer offset")
        return cursor


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
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    truncated: bool = False
    final_outcome: PolicyExplainOutcome
    reason_code: str
    visibility: PolicyExplainVisibility
    call_state: str
    requires_approval: bool
    installation_status: str = "unknown"
    runtime_availability: str = "unknown"
    tool_policy_outcome: PolicyExplainOutcome | None = None
    tool_policy_reason_code: str | None = None
    tool_policy_visibility: PolicyExplainVisibility | None = None
    tool_policy_call_state: str | None = None
    tool_policy_requires_approval: bool | None = None
    runtime_status: str | None = None
    runtime_reason_code: str | None = None
    subjects: list[PolicyExplainSubject] = Field(default_factory=list)
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    skipped_contributors: list[str] = Field(default_factory=list)
    redacted: bool = True


class ProfileToolPreviewEntry(BaseModel):
    """Redacted profile decision for one catalog or policy-listed tool."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    outcome: PolicyExplainOutcome
    reason_code: str
    visibility: PolicyExplainVisibility
    call_state: str
    requires_approval: bool
    installation_status: str = "unknown"
    runtime_availability: str = "unknown"
    redacted: bool = True


class ProfileToolPreviewSummary(BaseModel):
    """Count summary for profile tool preview entries."""

    model_config = ConfigDict(extra="forbid")

    total: int = 0
    allow: int = 0
    ask: int = 0
    deny: int = 0
    visible: int = 0
    hidden: int = 0
    deferred: int = 0
    installed: int = 0
    not_installed: int = 0
    unknown_installation: int = 0


class ProfileToolPreviewResponse(BaseModel):
    """Redacted preview of effective profile tool decisions."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    profile_id: str
    mode: PolicyExplainMode
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    truncated: bool = False
    next_cursor: str | None = None
    tools: list[ProfileToolPreviewEntry] = Field(default_factory=list)
    summary: ProfileToolPreviewSummary = Field(default_factory=ProfileToolPreviewSummary)
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    skipped_contributors: list[str] = Field(default_factory=list)
    redacted: bool = True


class PolicyExplainErrorResponse(BaseModel):
    """Stable error envelope for policy explanation failures."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = False
    message: str
    reason_code: str
    details: dict[str, Any] = Field(default_factory=dict)


class GatewayPolicyExplainError(RuntimeError):
    """Expected policy explanation service failure."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

    def to_payload(self) -> PolicyExplainErrorResponse:
        """Return a stable redacted error payload."""

        return PolicyExplainErrorResponse(
            message=str(self),
            reason_code=self.reason_code,
        )


@dataclass(frozen=True, slots=True)
class _PreviewCatalogRow:
    """Internal catalog row with metadata needed for admin preview filtering."""

    tool_name: str
    installation_status: str = "unknown"
    category: str | None = None


def parse_policy_explain_request(payload: Any) -> PolicyExplainRequest:
    """Parse a policy explain request without leaking invalid input in errors."""

    try:
        return PolicyExplainRequest.model_validate(payload)
    except ValidationError:
        raise GatewayPolicyExplainError(
            "Invalid policy explain request",
            reason_code="invalid_policy_explain_request",
        ) from None


def parse_profile_tool_preview_request(payload: Any) -> ProfileToolPreviewRequest:
    """Parse a profile tool preview request without leaking invalid input in errors."""

    try:
        return ProfileToolPreviewRequest.model_validate(payload)
    except ValidationError:
        raise GatewayPolicyExplainError(
            "Invalid policy preview request",
            reason_code="invalid_policy_preview_request",
        ) from None


class GatewayPolicyExplainService:
    """Assemble redacted policy explanations with strict audit events."""

    def __init__(
        self,
        *,
        profile_resolver: Callable[[str], MCPProfile | Awaitable[MCPProfile | None] | None],
        audit_store: AuditStore | None,
        actor_id: str | None = None,
        catalog_provider: Callable[[], Iterable[Any] | Awaitable[Iterable[Any]]] | None = None,
        admin_tool_catalog_provider: Callable[
            [MCPProfile],
            Iterable[Any] | Awaitable[Iterable[Any]],
        ] | None = None,
        installed_tool_catalog: Any | None = None,
        policy_grant_store: Any | None = None,
    ) -> None:
        self.profile_resolver = profile_resolver
        self.audit_store = audit_store
        self.actor_id = actor_id or DEFAULT_AUDIT_ACTOR_ID
        self.catalog_provider = catalog_provider
        self.admin_tool_catalog_provider = admin_tool_catalog_provider
        self.installed_tool_catalog = installed_tool_catalog
        self.policy_grant_store = policy_grant_store

    async def explain_tool_call(
        self,
        request: PolicyExplainRequest,
    ) -> PolicyExplainResponse:
        """Return a redacted explanation for one tool call policy check."""

        try:
            profile = await self._resolve_profile(request.profile_id)
        except GatewayPolicyExplainError as exc:
            await self._audit_explain_failure(request, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            failure = GatewayPolicyExplainError(
                "Profile resolution failed",
                reason_code="profile_resolution_failed",
            )
            await self._audit_explain_failure(request, failure)
            raise failure from exc
        policy_grant_store = (
            self.policy_grant_store
            if request.mode == PolicyExplainMode.RUNTIME_EFFECTIVE
            else None
        )
        session_id = (
            request.session_id
            if request.mode == PolicyExplainMode.RUNTIME_EFFECTIVE
            else None
        )
        safe_tool_name = _redact_tool_identifier(request.tool_name)
        try:
            simulator_result = simulate_tool_call_policy(
                profile,
                request.tool_name,
                request.arguments,
                capability=request.capability,
                policy_grant_store=policy_grant_store,
                session_id=session_id,
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
            effective_decision = _effective_decision_from_simulation(
                simulator_result,
                tool_policy_outcome=tool_explanation.final_outcome,
                tool_policy_reason_code=tool_explanation.reason_code,
            )
            if effective_decision["degraded_reason"] is not None:
                degraded_reasons.append(effective_decision["degraded_reason"])
            response = PolicyExplainResponse(
                profile_id=profile.id,
                tool_name=safe_tool_name,
                mode=request.mode,
                final_outcome=effective_decision["final_outcome"],
                reason_code=effective_decision["reason_code"],
                visibility=effective_decision["visibility"],
                call_state=effective_decision["call_state"],
                requires_approval=effective_decision["requires_approval"],
                tool_policy_outcome=tool_explanation.final_outcome,
                tool_policy_reason_code=tool_explanation.reason_code,
                tool_policy_visibility=_approved_visibility(
                    tool_explanation.visibility,
                    outcome=tool_explanation.final_outcome,
                ),
                tool_policy_call_state=tool_explanation.call_state,
                tool_policy_requires_approval=tool_explanation.requires_approval,
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
                truncated="subject_limit_exceeded" in degraded_reasons,
                skipped_contributors=_skipped_contributors_for_mode(request.mode),
            )
        except GatewayPolicyExplainError:
            raise
        except Exception as exc:  # noqa: BLE001
            failure = GatewayPolicyExplainError(
                "Policy evaluation failed",
                reason_code="policy_evaluation_failed",
            )
            await self._audit_explain_failure(request, failure, profile_id=profile.id)
            raise failure from exc
        await _append_audit_event_strict(
            self.audit_store,
            event_type=POLICY_EXPLAIN_AUDIT_EVENT_TYPE,
            actor_id=self.actor_id,
            profile_id=profile.id,
            target_type="tool",
            target_id=safe_tool_name,
            payload=response.model_dump(mode="json", exclude_none=True),
        )
        return response

    async def preview_profile_tools(
        self,
        request: ProfileToolPreviewRequest,
    ) -> ProfileToolPreviewResponse:
        """Return a redacted profile tool preview, degrading without a catalog."""

        try:
            profile = await self._resolve_profile(request.profile_id)
        except GatewayPolicyExplainError as exc:
            await self._audit_preview_failure(request, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            failure = GatewayPolicyExplainError(
                "Profile resolution failed",
                reason_code="profile_resolution_failed",
            )
            await self._audit_preview_failure(request, failure)
            raise failure from exc
        degraded_reasons: list[str] = []
        try:
            catalog_rows = await self._preview_catalog_rows(
                profile,
                include_recommendations=request.include_recommendations,
            )
        except Exception:  # noqa: BLE001
            degraded_reasons.append("catalog_unavailable")
            catalog_rows = _policy_tool_rows_from_profile(profile)
        else:
            if (
                self.admin_tool_catalog_provider is None
                and self.installed_tool_catalog is None
                and self.catalog_provider is None
            ):
                degraded_reasons.append("catalog_unavailable")

        try:
            argument_sensitive = _has_argument_sensitive_permission_rules(profile)
            filtered_rows = _filter_preview_catalog_rows(
                profile,
                catalog_rows,
                capability=request.capability,
                include_denied=request.include_denied,
                include_recommendations=request.include_recommendations,
                category=request.category,
            )
            page_rows, next_cursor = _preview_page(
                filtered_rows,
                limit=request.limit,
                cursor=request.cursor,
            )
            tools = [
                _profile_tool_preview_entry(
                    profile,
                    row.tool_name,
                    capability=request.capability,
                    argument_sensitive=argument_sensitive,
                    installation_status=row.installation_status,
                )
                for row in page_rows
            ]
            response = ProfileToolPreviewResponse(
                profile_id=profile.id,
                mode=request.mode,
                truncated=next_cursor is not None,
                next_cursor=next_cursor,
                tools=tools,
                summary=_preview_summary(tools),
                degraded=bool(degraded_reasons),
                degraded_reasons=degraded_reasons,
                skipped_contributors=_skipped_contributors_for_mode(request.mode),
            )
        except GatewayPolicyExplainError:
            raise
        except Exception as exc:  # noqa: BLE001
            failure = GatewayPolicyExplainError(
                "Policy evaluation failed",
                reason_code="policy_evaluation_failed",
            )
            await self._audit_preview_failure(request, failure, profile_id=profile.id)
            raise failure from exc
        await _append_audit_event_strict(
            self.audit_store,
            event_type=POLICY_PREVIEW_TOOLS_AUDIT_EVENT_TYPE,
            actor_id=self.actor_id,
            profile_id=profile.id,
            target_type="profile",
            target_id=profile.id,
            payload=_preview_audit_payload(response),
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

    async def _preview_catalog_rows(
        self,
        profile: MCPProfile,
        *,
        include_recommendations: bool,
    ) -> list[_PreviewCatalogRow]:
        if self.admin_tool_catalog_provider is not None:
            catalog = await _maybe_await(self.admin_tool_catalog_provider(profile))
            return _preview_rows_from_catalog(catalog)

        if self.installed_tool_catalog is not None:
            backend_tools = self.installed_tool_catalog
            if callable(backend_tools):
                backend_tools = backend_tools()
            backend_tools = await _maybe_await(backend_tools)
            return _preview_rows_from_catalog(
                list_admin_tool_catalog(
                    profile,
                    backend_tools,
                    include_recommendations=include_recommendations,
                )
            )

        if self.catalog_provider is not None:
            catalog = await _maybe_await(self.catalog_provider())
            return _preview_rows_from_catalog(catalog)

        return _policy_tool_rows_from_profile(profile)

    async def _audit_explain_failure(
        self,
        request: PolicyExplainRequest,
        exc: GatewayPolicyExplainError,
        *,
        profile_id: str | None = None,
    ) -> None:
        event_profile_id = profile_id or request.profile_id
        safe_tool_name = _redact_tool_identifier(request.tool_name)
        payload = {
            "ok": False,
            "profile_id": event_profile_id,
            "tool_name": safe_tool_name,
            "mode": request.mode.value,
            "message": str(exc),
            "reason_code": exc.reason_code,
            "redacted": True,
        }
        await _append_audit_event_strict(
            self.audit_store,
            event_type=POLICY_EXPLAIN_AUDIT_EVENT_TYPE,
            actor_id=self.actor_id,
            profile_id=event_profile_id,
            target_type="tool",
            target_id=safe_tool_name,
            payload=payload,
        )

    async def _audit_preview_failure(
        self,
        request: ProfileToolPreviewRequest,
        exc: GatewayPolicyExplainError,
        *,
        profile_id: str | None = None,
    ) -> None:
        event_profile_id = profile_id or request.profile_id
        payload = {
            "ok": False,
            "profile_id": event_profile_id,
            "mode": request.mode.value,
            "message": str(exc),
            "reason_code": exc.reason_code,
            "redacted": True,
        }
        await _append_audit_event_strict(
            self.audit_store,
            event_type=POLICY_PREVIEW_TOOLS_AUDIT_EVENT_TYPE,
            actor_id=self.actor_id,
            profile_id=event_profile_id,
            target_type="profile",
            target_id=event_profile_id,
            payload=payload,
        )


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
        return "", "omitted"
    if subject == "path":
        return _redact_path(text)
    if subject == "command":
        return "[redacted-command]", "redacted"
    if subject == "domain":
        return _redact_domain(text)
    if subject in {"tool", "mcp", "skill", "agent", "capability", "risk_class"}:
        return text, "raw_safe"
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
    argument_sensitive: bool,
    installation_status: str = "unknown",
) -> ProfileToolPreviewEntry:
    explanation = explain_profile_tool_decision(
        profile,
        tool_name,
        capability=capability,
    )
    if argument_sensitive and explanation.final_outcome == "allow":
        return ProfileToolPreviewEntry(
            tool_name=_redact_tool_identifier(tool_name),
            outcome="ask",
            reason_code="argument_sensitive_policy",
            visibility="deferred",
            call_state="deferred",
            requires_approval=False,
            installation_status=installation_status,
        )
    return ProfileToolPreviewEntry(
        tool_name=_redact_tool_identifier(tool_name),
        outcome=explanation.final_outcome,
        reason_code=explanation.reason_code,
        visibility=_approved_visibility(
            explanation.visibility,
            outcome=explanation.final_outcome,
        ),
        call_state=explanation.call_state,
        requires_approval=explanation.requires_approval,
        installation_status=installation_status,
    )


def _effective_decision_from_simulation(
    simulator_result: Mapping[str, Any],
    *,
    tool_policy_outcome: str,
    tool_policy_reason_code: str,
) -> dict[str, Any]:
    overall = _mapping_value(simulator_result, "overall")
    status = (
        _optional_string(_mapping_value(overall, "status"))
        if isinstance(overall, Mapping)
        else None
    )
    reason_code = (
        _optional_string(_mapping_value(overall, "reason_code"))
        if isinstance(overall, Mapping)
        else None
    )
    denial = _mapping_value(simulator_result, "denial")
    has_explicit_denial = bool(denial)
    degraded_reason: str | None = None
    if (
        status == "denied"
        and tool_policy_outcome == "ask"
        and reason_code == "approval_required"
        and not has_explicit_denial
    ):
        outcome: PolicyExplainOutcome = "ask"
    elif status == "denied":
        outcome: PolicyExplainOutcome = "deny"
    elif status == "approval_required":
        outcome = "ask"
    elif status == "allowed":
        outcome = "allow"
    else:
        outcome = "deny"
        degraded_reason = "unknown_policy_status"

    defaults = _DEFAULTS_BY_OUTCOME[outcome]
    return {
        "final_outcome": outcome,
        "reason_code": (
            "policy_status_unknown"
            if degraded_reason is not None
            else reason_code or tool_policy_reason_code
        ),
        "visibility": defaults["visibility"],
        "call_state": defaults["call_state"],
        "requires_approval": defaults["requires_approval"],
        "degraded_reason": degraded_reason,
    }


def _preview_summary(
    entries: Iterable[ProfileToolPreviewEntry],
) -> ProfileToolPreviewSummary:
    entry_list = list(entries)
    summary = ProfileToolPreviewSummary(total=len(entry_list))
    for entry in entry_list:
        if entry.outcome == "allow":
            summary.allow += 1
        elif entry.outcome == "ask":
            summary.ask += 1
        elif entry.outcome == "deny":
            summary.deny += 1
        if entry.visibility == "visible":
            summary.visible += 1
        elif entry.visibility == "hidden":
            summary.hidden += 1
        elif entry.visibility == "deferred":
            summary.deferred += 1
        if entry.installation_status == "installed":
            summary.installed += 1
        elif entry.installation_status == "not_installed":
            summary.not_installed += 1
        else:
            summary.unknown_installation += 1
    return summary


def _preview_page(
    rows: list[_PreviewCatalogRow],
    *,
    limit: int,
    cursor: str | None,
) -> tuple[list[_PreviewCatalogRow], str | None]:
    start = int(cursor) if cursor is not None else 0
    if start >= len(rows):
        return [], None
    end = min(start + limit, len(rows))
    next_cursor = str(end) if end < len(rows) else None
    return rows[start:end], next_cursor


def _preview_audit_payload(response: ProfileToolPreviewResponse) -> dict[str, Any]:
    return response.model_dump(
        mode="json",
        exclude={"tools"},
        exclude_none=True,
    )


def _tool_names_from_catalog(catalog: Any) -> set[str]:
    tool_names: set[str] = set()
    for item in _catalog_items(catalog):
        tool_name: Any
        if isinstance(item, str):
            tool_name = item
        elif isinstance(item, Mapping):
            tool_name = item.get("tool_id") or item.get("name") or item.get("tool_name")
        else:
            tool_name = (
                getattr(item, "tool_id", None)
                or getattr(item, "name", None)
                or getattr(item, "tool_name", None)
            )
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.add(tool_name.strip())
    return tool_names


def _preview_rows_from_catalog(catalog: Any) -> list[_PreviewCatalogRow]:
    rows_by_name: dict[str, _PreviewCatalogRow] = {}
    for item in _catalog_items(catalog):
        row = _preview_row_from_catalog_item(item)
        if row is None:
            continue
        rows_by_name.setdefault(row.tool_name, row)
    return list(rows_by_name.values())


def _preview_row_from_catalog_item(item: Any) -> _PreviewCatalogRow | None:
    if isinstance(item, str):
        tool_name = item
        installation_status = "unknown"
        category = None
    elif isinstance(item, Mapping):
        tool_name = item.get("tool_id") or item.get("name") or item.get("tool_name")
        installation_status = item.get("installation_status")
        category = item.get("category")
        metadata = item.get("metadata")
        if category is None and isinstance(metadata, Mapping):
            category = metadata.get("category")
    else:
        tool_name = (
            getattr(item, "tool_id", None)
            or getattr(item, "name", None)
            or getattr(item, "tool_name", None)
        )
        installation_status = getattr(item, "installation_status", None)
        category = getattr(item, "category", None)
        metadata = getattr(item, "metadata", None)
        if category is None and isinstance(metadata, Mapping):
            category = metadata.get("category")

    tool_text = _optional_string(tool_name)
    if tool_text is None:
        return None
    return _PreviewCatalogRow(
        tool_name=tool_text,
        installation_status=_preview_installation_status(installation_status),
        category=_optional_string(category),
    )


def _filter_preview_catalog_rows(
    profile: MCPProfile,
    rows: list[_PreviewCatalogRow],
    *,
    capability: str | None,
    include_denied: bool,
    include_recommendations: bool,
    category: str | None,
) -> list[_PreviewCatalogRow]:
    normalized_category = _normalize_preview_category(category)
    filtered: list[_PreviewCatalogRow] = []
    for row in rows:
        if (
            normalized_category is not None
            and _normalize_preview_category(row.category) != normalized_category
        ):
            continue
        if not include_recommendations and row.installation_status == "not_installed":
            continue
        explanation = explain_profile_tool_decision(
            profile,
            row.tool_name,
            capability=capability,
        )
        if not include_denied and explanation.final_outcome == "deny":
            continue
        filtered.append(row)
    return filtered


def _preview_installation_status(value: Any) -> str:
    status = _optional_string(value)
    if status in {"installed", "not_installed"}:
        return status
    if status in {"recommended_unavailable", "unavailable"}:
        return "not_installed"
    return "unknown"


def _normalize_preview_category(value: str | None) -> str | None:
    text = _optional_string(value)
    return text.casefold() if text is not None else None


def _catalog_items(catalog: Any) -> Iterable[Any]:
    if isinstance(catalog, Mapping):
        tools = catalog.get("tools")
        if tools is not None and not isinstance(tools, str):
            if isinstance(tools, Iterable):
                return tools
            return []
        return [catalog]
    if isinstance(catalog, str):
        return [catalog]
    if isinstance(catalog, Iterable):
        return catalog
    return []


def _policy_tool_names_from_profile(profile: MCPProfile) -> list[str]:
    policy_document = profile.policy_document
    policy_tools = set(policy_document.allowed_tools or [])
    policy_tools.update(policy_document.denied_tools or [])
    return sorted(tool_name for tool_name in policy_tools if isinstance(tool_name, str))


def _policy_tool_rows_from_profile(profile: MCPProfile) -> list[_PreviewCatalogRow]:
    return [
        _PreviewCatalogRow(tool_name=tool_name)
        for tool_name in _policy_tool_names_from_profile(profile)
    ]


def _has_argument_sensitive_permission_rules(profile: MCPProfile) -> bool:
    rules = compile_permission_rules(profile.policy_document)
    return any(rule.rule_type in {"command", "domain", "path"} for rule in rules)


def _redact_path(value: str) -> tuple[str, PolicyExplainRedactionState]:
    normalized = value.replace("\\", "/")
    if "\n" in normalized or "\r" in normalized:
        return "[redacted-path]", "redacted"
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() == "file":
        file_path = unquote(parsed.path or "")
        if not file_path:
            return "file://[redacted-path]", "redacted"
        return _redact_path(file_path)
    if parsed.scheme and parsed.netloc:
        return _redacted_url_origin(parsed), "sanitized"
    sanitized = _strip_path_private_syntax(normalized)
    if _is_absolute_path(sanitized):
        filename = sanitized.rstrip("/").rsplit("/", 1)[-1]
        return (f".../{filename}" if filename else "[path]"), "sanitized"
    state: PolicyExplainRedactionState = (
        "sanitized" if sanitized != normalized else "raw_safe"
    )
    return sanitized, state


def _redact_domain(value: str) -> tuple[str, PolicyExplainRedactionState]:
    normalized = value.replace("\\", "/")
    if "\n" in normalized or "\r" in normalized:
        return "[redacted-domain]", "redacted"
    stripped = _strip_path_private_syntax(normalized)
    if _is_absolute_path(stripped):
        return _redact_path(stripped)
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() == "file":
        return _redact_path(normalized)
    if parsed.scheme and parsed.netloc:
        return _redacted_url_origin(parsed), "sanitized"
    sanitized = stripped
    state: PolicyExplainRedactionState = (
        "sanitized" if sanitized != normalized else "raw_safe"
    )
    return sanitized, state


def _strip_path_private_syntax(value: str) -> str:
    sanitized = value.split("?", 1)[0].split("#", 1)[0]
    if "@" in sanitized:
        sanitized = sanitized.rsplit("@", 1)[-1]
    return sanitized


def _redacted_url_origin(parsed: Any) -> str:
    host = parsed.hostname or "[domain]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    return f"{parsed.scheme}://{host}"


def _skipped_contributors_for_mode(mode: PolicyExplainMode) -> list[str]:
    if mode != PolicyExplainMode.STATIC_POLICY_ONLY:
        return []
    return list(STATIC_POLICY_ONLY_SKIPPED_CONTRIBUTORS)


def _approved_visibility(
    value: Any,
    *,
    outcome: str | None = None,
) -> PolicyExplainVisibility:
    if value == "hidden":
        return "hidden"
    if value == "deferred":
        return "deferred"
    if outcome == "deny":
        return "hidden"
    return "visible"


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
    "PolicyExplainVisibility",
    "ProfileToolPreviewEntry",
    "ProfileToolPreviewRequest",
    "ProfileToolPreviewResponse",
    "ProfileToolPreviewSummary",
    "_append_audit_event_strict",
    "parse_policy_explain_request",
    "parse_profile_tool_preview_request",
    "_redact_subject_value",
]
