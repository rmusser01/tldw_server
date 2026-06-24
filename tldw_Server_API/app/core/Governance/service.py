"""Governance service APIs shared by MCP and ACP call paths."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Protocol, cast

from .resolver import resolve_effective_action
from .types import CandidateAction, GovernanceAction, utc_now

CategorySource = Literal["explicit", "metadata", "pattern", "default"]

_FALLBACK_ACTIONS: dict[str, GovernanceAction] = {
    "allow": "allow",
    "allow_only": "allow",
    "open": "allow",
    "warn": "warn",
    "warn_only": "warn",
    "deny": "deny",
    "closed": "deny",
    "require_approval": "require_approval",
    "approval": "require_approval",
}

_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("security", ("auth", "rbac", "oauth", "token", "secret", "permission", "mfa")),
    ("privacy", ("privacy", "pii", "gdpr", "ccpa", "hipaa")),
    ("dependencies", ("dependency", "dependencies", "package", "library", "version")),
    ("compliance", ("compliance", "policy", "governance", "audit", "regulation")),
)

_VALID_ACTIONS = frozenset({"allow", "warn", "require_approval", "deny"})


class InvalidGovernanceCandidateError(ValueError):
    """Raised when a policy source returns a malformed governance candidate."""


class _GapStoreProtocol(Protocol):
    async def upsert_open_gap(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class GovernanceKnowledgeResult:
    """Categorized governance query result."""

    query: str
    category: str
    category_source: CategorySource
    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GovernanceValidationResult:
    """Resolved policy decision for a change request."""

    action: GovernanceAction
    status: GovernanceAction
    category: str
    category_source: CategorySource
    fallback_reason: str | None = None
    matched_rules: tuple[str, ...] = field(default_factory=tuple)


class GovernanceService:
    """Minimal governance service abstraction with shared fallback behavior."""

    def __init__(
        self,
        *,
        store: _GapStoreProtocol,
        policy_loader: Any | None = None,
        default_fallback_mode: str = "warn_only",
    ) -> None:
        self._store = store
        self._policy_loader = policy_loader if policy_loader is not None else store
        self._default_fallback_mode = default_fallback_mode

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").strip().split())

    def _resolve_category(
        self,
        *,
        query: str,
        category: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[str, CategorySource]:
        explicit = self._normalize_text(category).lower()
        if explicit:
            return explicit, "explicit"

        metadata_category = ""
        if metadata and isinstance(metadata.get("category"), str):
            metadata_category = self._normalize_text(metadata.get("category")).lower()
        if metadata_category:
            return metadata_category, "metadata"

        lowered_query = self._normalize_text(query).lower()
        for mapped_category, patterns in _CATEGORY_PATTERNS:
            if any(token in lowered_query for token in patterns):
                return mapped_category, "pattern"
        return "general", "default"

    @staticmethod
    def _coerce_action(value: Any) -> GovernanceAction:
        normalized = GovernanceService._normalize_text(value).lower()
        action = _FALLBACK_ACTIONS.get(normalized, normalized)
        if action not in _VALID_ACTIONS:
            raise InvalidGovernanceCandidateError("invalid_candidate_action")
        return cast(GovernanceAction, action)

    @staticmethod
    def _coerce_int(value: Any, *, field: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise InvalidGovernanceCandidateError(f"invalid_candidate_{field}") from exc

    @staticmethod
    def _coerce_updated_at(value: Any) -> datetime:
        if value is None:
            return utc_now()
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            rendered = GovernanceService._normalize_text(value)
            if not rendered:
                return utc_now()
            if rendered.endswith("Z"):
                rendered = f"{rendered[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(rendered)
            except ValueError as exc:
                raise InvalidGovernanceCandidateError("invalid_candidate_updated_at") from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _coerce_candidate(raw: CandidateAction | Mapping[str, Any]) -> CandidateAction:
        if isinstance(raw, CandidateAction):
            GovernanceService._coerce_action(raw.action)
            return raw
        return CandidateAction(
            action=GovernanceService._coerce_action(raw.get("action", "allow")),
            scope_level=GovernanceService._coerce_int(raw.get("scope_level", 0), field="scope_level"),
            priority=GovernanceService._coerce_int(raw.get("priority", 0), field="priority"),
            updated_at=GovernanceService._coerce_updated_at(raw.get("updated_at")),
            source_id=(None if raw.get("source_id") is None else str(raw.get("source_id"))),
            reason=(None if raw.get("reason") is None else str(raw.get("reason"))),
        )

    async def _load_candidates(self, **kwargs: Any) -> list[CandidateAction]:
        if self._policy_loader is None:
            return []

        loader_method = getattr(self._policy_loader, "get_candidates", None)
        if loader_method is None:
            loader_method = getattr(self._policy_loader, "match_candidates", None)
        if loader_method is None:
            return []

        loaded = loader_method(**kwargs)
        if inspect.isawaitable(loaded):
            loaded = await loaded
        if loaded is None:
            return []
        return [self._coerce_candidate(item) for item in loaded]

    async def query_knowledge(
        self,
        *,
        query: str,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GovernanceKnowledgeResult:
        resolved_category, category_source = self._resolve_category(
            query=query,
            category=category,
            metadata=metadata,
        )
        rules: tuple[dict[str, Any], ...] = ()
        try:
            candidates = await self._load_candidates(
                surface="knowledge_query",
                summary=query,
                category=resolved_category,
                metadata=metadata or {},
            )
            rules = tuple(
                {
                    "action": candidate.action,
                    "scope_level": candidate.scope_level,
                    "priority": candidate.priority,
                    "source_id": candidate.source_id,
                }
                for candidate in candidates
            )
        except (AttributeError, InvalidGovernanceCandidateError, RuntimeError, TypeError, ValueError):
            rules = ()

        return GovernanceKnowledgeResult(
            query=self._normalize_text(query),
            category=resolved_category,
            category_source=category_source,
            rules=rules,
        )

    def resolve_fallback(self, mode: str | None) -> GovernanceAction:
        normalized = self._normalize_text(mode).lower() if mode else ""
        if normalized in _FALLBACK_ACTIONS:
            return _FALLBACK_ACTIONS[normalized]
        return _FALLBACK_ACTIONS.get(self._default_fallback_mode, "warn")

    async def validate_change(
        self,
        *,
        surface: str,
        summary: str,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        fallback_mode: str | None = None,
    ) -> GovernanceValidationResult:
        resolved_category, category_source = self._resolve_category(
            query=summary,
            category=category,
            metadata=metadata,
        )

        effective_fallback_mode = (
            fallback_mode
            or getattr(self._policy_loader, "fallback_mode", None)
            or self._default_fallback_mode
        )

        try:
            candidates = await self._load_candidates(
                surface=surface,
                summary=summary,
                category=resolved_category,
                metadata=metadata or {},
            )
        except InvalidGovernanceCandidateError as exc:
            return GovernanceValidationResult(
                action="deny",
                status="deny",
                category=resolved_category,
                category_source=category_source,
                fallback_reason=str(exc) or "invalid_candidate",
                matched_rules=(),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            fallback_action = self.resolve_fallback(effective_fallback_mode)
            return GovernanceValidationResult(
                action=fallback_action,
                status=fallback_action,
                category=resolved_category,
                category_source=category_source,
                fallback_reason="backend_unavailable",
                matched_rules=(),
            )

        if not candidates:
            return GovernanceValidationResult(
                action="allow",
                status="allow",
                category=resolved_category,
                category_source=category_source,
                fallback_reason=None,
                matched_rules=(),
            )

        effective = resolve_effective_action(candidates)
        matched_rule_ids = tuple(
            candidate.source_id
            for candidate in effective.ordered_candidates
            if candidate.source_id
        )
        return GovernanceValidationResult(
            action=effective.action,
            status=effective.action,
            category=resolved_category,
            category_source=category_source,
            fallback_reason=None,
            matched_rules=matched_rule_ids,
        )

    async def resolve_gap(
        self,
        *,
        question: str,
        category: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        org_id: int | None = None,
        team_id: int | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        resolution_mode: str | None = None,
    ) -> Any:
        resolved_category, _ = self._resolve_category(
            query=question,
            category=category,
            metadata=metadata,
        )
        return await self._store.upsert_open_gap(
            question=question,
            category=resolved_category,
            org_id=org_id,
            team_id=team_id,
            persona_id=persona_id,
            workspace_id=workspace_id,
            resolution_mode=resolution_mode,
        )
