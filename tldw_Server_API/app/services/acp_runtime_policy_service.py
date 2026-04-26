from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple, set)):
        return []
    items: list[str] = []
    for entry in value:
        cleaned = str(entry or "").strip()
        if cleaned:
            items.append(cleaned)
    return items


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class ACPRuntimePolicySnapshot(BaseModel):
    session_id: str
    user_id: int
    policy_snapshot_version: str
    policy_snapshot_fingerprint: str
    policy_snapshot_refreshed_at: str
    policy_summary: dict[str, Any] = Field(default_factory=dict)
    policy_provenance_summary: dict[str, Any] = Field(default_factory=dict)
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    approval_summary: dict[str, Any] = Field(default_factory=dict)
    context_summary: dict[str, Any] = Field(default_factory=dict)
    execution_config: dict[str, Any] = Field(default_factory=dict)
    refresh_error: str | None = None


class ACPRuntimePolicyService:
    """Build and persist ACP runtime policy snapshots from MCP Hub policy."""

    def __init__(
        self,
        *,
        policy_resolver: Any | None = None,
        mcp_hub_service: Any | None = None,
    ) -> None:
        self.policy_resolver = policy_resolver
        self.mcp_hub_service = mcp_hub_service

    async def _get_policy_resolver(self) -> Any:
        if self.policy_resolver is None:
            repo = McpHubRepo(await get_db_pool())
            self.policy_resolver = McpHubPolicyResolver(repo)
        return self.policy_resolver

    @staticmethod
    def _parse_acp_profile(acp_profile: Any) -> tuple[int | None, dict[str, Any], dict[str, Any]]:
        if not isinstance(acp_profile, dict):
            return None, {}, {}

        profile_id: int | None = None
        raw_profile_id = acp_profile.get("id")
        try:
            if raw_profile_id is not None:
                profile_id = int(raw_profile_id)
        except (TypeError, ValueError):
            profile_id = None

        if isinstance(acp_profile.get("profile"), dict):
            payload = dict(acp_profile["profile"])
        elif isinstance(acp_profile.get("profile_json"), str):
            try:
                payload = _as_dict(json.loads(acp_profile["profile_json"]))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {
                key: value
                for key, value in acp_profile.items()
                if key not in {"id", "name", "description", "owner_scope_type", "owner_scope_id"}
            }

        execution_config = _as_dict(payload.get("execution_config")) or {
            key: value for key, value in payload.items() if key != "policy_hints"
        }
        policy_hints = _as_dict(payload.get("policy_hints"))
        return profile_id, execution_config, policy_hints

    async def _load_acp_profile(
        self,
        *,
        acp_profile_id: int | None,
        acp_profile: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if acp_profile is not None:
            return acp_profile
        if acp_profile_id is None or self.mcp_hub_service is None:
            return None
        getter = getattr(self.mcp_hub_service, "get_acp_profile", None)
        if getter is None:
            return None
        return await getter(int(acp_profile_id))

    async def build_resolution_metadata(
        self,
        *,
        session_record: Any,
        acp_profile_id: int | None = None,
        acp_profile: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        loaded_profile = await self._load_acp_profile(
            acp_profile_id=acp_profile_id,
            acp_profile=acp_profile,
        )
        parsed_profile_id, execution_config, policy_hints = self._parse_acp_profile(loaded_profile)
        effective_profile_id = acp_profile_id if acp_profile_id is not None else parsed_profile_id

        metadata: dict[str, Any] = {"mcp_policy_context_enabled": True}
        for key in (
            "persona_id",
            "workspace_id",
            "workspace_group_id",
            "scope_snapshot_id",
        ):
            value = getattr(session_record, key, None)
            if value:
                metadata[key] = value

        mcp_servers = getattr(session_record, "mcp_servers", None)
        if mcp_servers:
            metadata["mcp_servers"] = list(mcp_servers)

        if effective_profile_id is not None:
            metadata["acp_profile_id"] = int(effective_profile_id)

        hint_tags = _unique(
            _as_str_list(policy_hints.get("tags"))
            + _as_str_list(loaded_profile.get("hint_tags") if isinstance(loaded_profile, dict) else None)
        )
        if hint_tags:
            metadata["acp_profile_hint_tags"] = hint_tags

        for key, value in dict(extra_metadata or {}).items():
            if value is not None:
                metadata[key] = value

        return metadata, execution_config

    def _resolve_template_config(
        self,
        *,
        template_name: str,
        session_id: str | None = None,
        persona_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a template config dict, trying DB templates first then flat fallback."""
        # Try DB-backed templates if a session store is available.
        try:
            from tldw_Server_API.app.core.Agent_Client_Protocol.templates import (
                resolve_for_session,
            )
            from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

            db = ACPSessionsDB()
            result = resolve_for_session(db, session_id, persona_id, template_name)
            if result is not None:
                return result
        except Exception:
            logger.exception(
                "ACP runtime policy template lookup failed for template_name={} session_id={} persona_id={}; "
                "falling back to flat templates",
                template_name,
                session_id,
                persona_id,
            )

        # Fallback: use the flat PERMISSION_POLICY_TEMPLATES dict.
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
            PERMISSION_POLICY_TEMPLATES,
        )
        return PERMISSION_POLICY_TEMPLATES.get(template_name)

    async def build_snapshot(
        self,
        *,
        session_record: Any,
        user_id: int,
        acp_profile_id: int | None = None,
        acp_profile: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        template_name: str | None = None,
    ) -> ACPRuntimePolicySnapshot:
        metadata, execution_config = await self.build_resolution_metadata(
            session_record=session_record,
            acp_profile_id=acp_profile_id,
            acp_profile=acp_profile,
            extra_metadata=extra_metadata,
        )
        policy_resolver = await self._get_policy_resolver()
        effective_policy = await policy_resolver.resolve_for_context(
            user_id=user_id,
            metadata=metadata,
        )

        resolved_policy_document = _as_dict(
            effective_policy.get("resolved_policy_document")
            or effective_policy.get("policy_document")
        )

        # Apply permission policy template as a base layer.
        # User / MCP-Hub overrides that are already in the resolved document
        # take precedence over template defaults.
        if template_name:
            template_config = self._resolve_template_config(
                template_name=template_name,
                session_id=str(getattr(session_record, "session_id", "")),
                persona_id=getattr(session_record, "persona_id", None),
            )
            if template_config:
                existing_overrides = resolved_policy_document.get("tool_tier_overrides", {})
                # Template is the base layer -- merge user overrides on top.
                merged = {**template_config.get("tool_tier_overrides", {}), **existing_overrides}
                resolved_policy_document["tool_tier_overrides"] = merged
        allowed_tools = _unique(_as_str_list(resolved_policy_document.get("allowed_tools")))
        denied_tools = _unique(_as_str_list(resolved_policy_document.get("denied_tools")))
        capabilities = _unique(_as_str_list(resolved_policy_document.get("capabilities")))

        sources = list(effective_policy.get("sources") or [])
        provenance = list(effective_policy.get("provenance") or [])
        source_kinds = sorted(
            _unique(
                [
                    str(item.get("source_kind") or "").strip()
                    for item in sources + provenance
                    if str(item.get("source_kind") or "").strip()
                ]
            )
        )
        policy_summary = {
            "allowed_tool_count": len(allowed_tools),
            "denied_tool_count": len(denied_tools),
            "capability_count": len(capabilities),
            "approval_mode": str(resolved_policy_document.get("approval_mode") or "").strip() or None,
            "path_scope_mode": str(resolved_policy_document.get("path_scope_mode") or "").strip() or None,
        }
        approval_summary = {
            "mode": policy_summary["approval_mode"],
            "approval_policy_id": effective_policy.get("approval_policy_id")
            or resolved_policy_document.get("approval_policy_id"),
        }
        policy_provenance_summary = {
            "source_count": len(sources),
            "provenance_entry_count": len(provenance),
            "source_kinds": source_kinds,
        }
        context_summary = dict(metadata)
        fingerprint_payload = {
            "context_summary": context_summary,
            "resolved_policy_document": resolved_policy_document,
            "approval_summary": approval_summary,
            "policy_version": effective_policy.get("policy_version") or "resolved-v1",
        }
        fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()

        return ACPRuntimePolicySnapshot(
            session_id=str(getattr(session_record, "session_id")),
            user_id=int(user_id),
            policy_snapshot_version=str(effective_policy.get("policy_version") or "resolved-v1"),
            policy_snapshot_fingerprint=fingerprint,
            policy_snapshot_refreshed_at=_utcnow_iso(),
            policy_summary=policy_summary,
            policy_provenance_summary=policy_provenance_summary,
            resolved_policy_document=resolved_policy_document,
            approval_summary=approval_summary,
            context_summary=context_summary,
            execution_config=execution_config,
        )

    async def persist_snapshot(
        self,
        *,
        session_store: Any,
        snapshot: ACPRuntimePolicySnapshot,
    ) -> Any:
        return await session_store.update_policy_snapshot_state(
            snapshot.session_id,
            policy_snapshot_version=snapshot.policy_snapshot_version,
            policy_snapshot_fingerprint=snapshot.policy_snapshot_fingerprint,
            policy_snapshot_refreshed_at=snapshot.policy_snapshot_refreshed_at,
            policy_summary=snapshot.policy_summary,
            policy_provenance_summary=snapshot.policy_provenance_summary,
            policy_refresh_error=snapshot.refresh_error,
        )
