from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mcp_unified.interfaces.path_scope import PathScopeCandidate
from mcp_unified.profiles.path_grants import compile_policy_path_grants, has_path_grant_policy

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.services.mcp_hub_multi_root_path_service import (
    McpHubMultiRootPathService,
)
from tldw_Server_API.app.services.mcp_hub_path_scope_service import McpHubPathScopeService
from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import McpHubWorkspaceRootResolver

_FILESYSTEM_CAPABILITIES = frozenset({"filesystem.read", "filesystem.write", "filesystem.delete"})
_PATH_GRANT_ACTIONS = frozenset({"read", "edit", "write"})
_PATH_GRANT_EFFECTS = frozenset({"allow", "deny"})
_SUPPORTED_PATH_ARGUMENT_HINTS = frozenset(
    {"path", "file_path", "target_path", "cwd", "paths", "file_paths", "files[].path"}
)
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:")
_PREVIEW_TOOL_METADATA = {
    "uses_filesystem": True,
    "path_boundable": True,
    "path_argument_hints": ["path"],
}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, Iterable) or isinstance(value, (bytes, bytearray, dict)):
        return []
    out: list[str] = []
    for entry in value:
        cleaned = str(entry or "").strip()
        if cleaned:
            out.append(cleaned)
    return out


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _unique_paths_with_actions(
    raw_paths: list[str],
    candidate_actions: list[str],
    *,
    default_action: str,
) -> tuple[list[str], list[str]]:
    """De-duplicate candidate paths while keeping their action indexes aligned."""

    out_paths: list[str] = []
    out_actions: list[str] = []
    seen: set[str] = set()
    for index, raw_path in enumerate(raw_paths):
        cleaned = str(raw_path or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out_paths.append(cleaned)
        action = candidate_actions[index] if index < len(candidate_actions) else default_action
        out_actions.append(action if action in _PATH_GRANT_ACTIONS else default_action)
    return out_paths, out_actions


def _is_within(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _normalize_candidate_path(raw_path: str, *, base_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_path / candidate
    return candidate.resolve(strict=False)


def _scope_root_from_scope(scope: dict[str, Any]) -> Path | None:
    workspace_root = str(scope.get("workspace_root") or "").strip()
    if not workspace_root:
        return None
    if str(scope.get("path_scope_mode") or "").strip() == "cwd_descendants":
        cwd = str(scope.get("cwd") or "").strip()
        if not cwd:
            return None
        return Path(cwd).expanduser().resolve(strict=False)
    return Path(workspace_root).expanduser().resolve(strict=False)


def _normalize_allowlist_prefix(raw_value: Any) -> str | None:
    value = str(raw_value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = re.sub(r"/+", "/", value)
    if not value or value.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        return None
    parts: list[str] = []
    for part in value.split("/"):
        cleaned = str(part or "").strip()
        if not cleaned or cleaned == ".":
            continue
        if cleaned == "..":
            return None
        parts.append(cleaned)
    if not parts:
        return None
    return "/".join(parts)


def _normalize_workspace_relative_path(raw_value: Any) -> tuple[str | None, str | None]:
    """Normalize a user-supplied path and reject values outside workspace-relative form."""

    value = str(raw_value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = re.sub(r"/+", "/", value)
    if not value:
        return None, "path_required"
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(value):
        return None, "path_must_be_workspace_relative"

    parts: list[str] = []
    for part in value.split("/"):
        cleaned = str(part or "").strip()
        if not cleaned or cleaned == ".":
            continue
        if cleaned == "..":
            return None, "path_traversal_not_allowed"
        parts.append(cleaned)
    if not parts:
        return ".", None
    return "/".join(parts), None


def _policy_allowlist_prefixes(effective_policy: dict[str, Any] | None) -> list[str]:
    policy_document = _as_dict((effective_policy or {}).get("policy_document"))
    out: list[str] = []
    seen: set[str] = set()
    for raw_entry in _as_str_list(policy_document.get("path_allowlist_prefixes")):
        normalized = _normalize_allowlist_prefix(raw_entry)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return sorted(out)


def _policy_path_grants(effective_policy: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    policy_document = _as_dict((effective_policy or {}).get("policy_document"))
    if not has_path_grant_policy(policy_document):
        return None
    return compile_policy_path_grants(policy_document).path_grants


def _allowlist_roots(*, workspace_root: Path, allowlist_prefixes: list[str]) -> list[Path]:
    return [(workspace_root / prefix).resolve(strict=False) for prefix in allowlist_prefixes]


def _tool_metadata(tool_def: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool_def, dict):
        return {}
    return _as_dict(tool_def.get("metadata"))


def _tool_uses_filesystem(metadata: dict[str, Any]) -> bool:
    explicit = _as_bool(metadata.get("uses_filesystem"))
    if explicit is not None:
        return explicit
    capabilities = {cap for cap in _as_str_list(metadata.get("capabilities")) if cap}
    return any(capability in _FILESYSTEM_CAPABILITIES for capability in capabilities)


def _path_boundable(metadata: dict[str, Any]) -> bool:
    return bool(_as_bool(metadata.get("path_boundable")))


def _path_argument_hints(metadata: dict[str, Any]) -> list[str]:
    return [
        hint
        for hint in _unique(_as_str_list(metadata.get("path_argument_hints")))
        if hint in _SUPPORTED_PATH_ARGUMENT_HINTS
    ]


def _extract_candidate_paths(tool_args: Any, hints: list[str]) -> list[str]:
    if not isinstance(tool_args, dict):
        return []
    out: list[str] = []
    for hint in hints:
        if hint in {"path", "file_path", "target_path", "cwd"}:
            value = str(tool_args.get(hint) or "").strip()
            if value:
                out.append(value)
            continue
        if hint in {"paths", "file_paths"}:
            values = tool_args.get(hint)
            if isinstance(values, list):
                out.extend(str(item or "").strip() for item in values if str(item or "").strip())
            continue
        if hint == "files[].path":
            files = tool_args.get("files")
            if isinstance(files, list):
                out.extend(
                    str(item.get("path") or "").strip()
                    for item in files
                    if isinstance(item, dict) and str(item.get("path") or "").strip()
                )
    return _unique(out)


def _path_scope_action(metadata: dict[str, Any]) -> str:
    action = str(metadata.get("path_scope_action") or "").strip().lower()
    if action in _PATH_GRANT_ACTIONS:
        return action
    for flag_name in ("write_capable", "is_write", "mutates_state"):
        if _as_bool(metadata.get(flag_name)) is True:
            return "write"
    return "read"


def _selected_profile_id(effective_policy: dict[str, Any] | None) -> int | None:
    """Return the profile id for the explicitly selected policy assignment when present."""

    policy = dict(effective_policy or {})
    selected_assignment_id = policy.get("selected_assignment_id")
    if selected_assignment_id in (None, ""):
        return None
    sources = policy.get("sources")
    if not isinstance(sources, list):
        return None
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        if source.get("assignment_id") != selected_assignment_id:
            continue
        try:
            profile_id = source.get("profile_id")
            return int(profile_id) if profile_id not in (None, "") else None
        except (TypeError, ValueError):
            return None
    return None


def _preview_outcome(enforcement_result: dict[str, Any]) -> str:
    """Map the path-enforcement result shape into the preview allow/ask/deny outcome."""

    if not bool(enforcement_result.get("enabled")):
        return "allow"
    if bool(enforcement_result.get("within_scope", True)):
        return "allow"
    if bool(enforcement_result.get("force_approval", False)):
        return "ask"
    return "deny"


def _safe_path_decisions(enforcement_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract serializable path-decision dictionaries from direct or nested result payloads."""

    raw_decisions = enforcement_result.get("path_decisions")
    if not isinstance(raw_decisions, list):
        scope_payload = enforcement_result.get("scope_payload")
        raw_decisions = scope_payload.get("path_decisions") if isinstance(scope_payload, dict) else []
    return [dict(decision) for decision in raw_decisions if isinstance(decision, Mapping)]


def _relative_path_for_decision(root: Path, candidate: Path) -> str | None:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return None
    text = relative.as_posix()
    return text if text not in {"", "."} else "."


def _grant_prefix_matches(relative_path: str, prefix: str) -> bool:
    if prefix == ".":
        return True
    return relative_path == prefix or relative_path.startswith(f"{prefix}/")


def _path_grant_decision(
    *,
    relative_path: str,
    action: str,
    path_grants: list[dict[str, Any]],
) -> dict[str, Any]:
    matches = [
        grant
        for grant in path_grants
        if action in set(grant.get("actions") or [])
        and _grant_prefix_matches(relative_path, str(grant.get("prefix") or ""))
    ]
    deny_matches = [grant for grant in matches if grant.get("effect") == "deny"]
    allow_matches = [grant for grant in matches if grant.get("effect") == "allow"]
    selected: dict[str, Any] | None = None
    outcome = "not_granted"
    reason_code: str | None = "path_action_not_granted"
    if deny_matches:
        selected = max(deny_matches, key=lambda grant: len(str(grant.get("prefix") or "")))
        outcome = "denied"
        reason_code = "path_action_denied"
    elif allow_matches:
        selected = max(allow_matches, key=lambda grant: len(str(grant.get("prefix") or "")))
        outcome = "allowed"
        reason_code = None

    return {
        "requested_action": action,
        "normalized_path": relative_path,
        "grant_outcome": outcome,
        "grant_source": "path_grants",
        "matched_grant_prefix": str((selected or {}).get("prefix") or "") or None,
        "matched_grant_effect": str((selected or {}).get("effect") or "") or None,
        "reason_code": reason_code,
        "redacted": True,
    }


class McpHubPathEnforcementService:
    """Evaluate path-scoped MCP Hub policy for a concrete tool call."""

    def __init__(
        self,
        path_scope_service: McpHubPathScopeService | Any | None = None,
        multi_root_path_service: McpHubMultiRootPathService | Any | None = None,
    ) -> None:
        self._path_scope_service = path_scope_service
        self._multi_root_path_service = multi_root_path_service

    async def preview_effective_path_permission(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any | None,
        tool_name: str,
        action: str,
        path: str,
    ) -> dict[str, Any]:
        """Return a redacted effective path-permission explanation for operators."""

        requested_action = str(action or "").strip().lower()
        normalized_path, path_error = _normalize_workspace_relative_path(path)
        safe_requested_path = normalized_path if normalized_path is not None else "<redacted>"
        base_payload: dict[str, Any] = {
            "tool_name": str(tool_name or "").strip(),
            "requested_action": requested_action,
            "requested_path": safe_requested_path,
            "normalized_path": normalized_path,
            "selected_assignment_id": (effective_policy or {}).get("selected_assignment_id"),
            "profile_id": _selected_profile_id(effective_policy),
            "redacted": True,
        }
        if requested_action not in _PATH_GRANT_ACTIONS:
            return {
                **base_payload,
                "outcome": "deny",
                "within_scope": False,
                "reason_code": "invalid_path_action",
                "path_decisions": [],
            }
        if path_error or normalized_path is None:
            return {
                **base_payload,
                "outcome": "deny",
                "within_scope": False,
                "reason_code": path_error,
                "path_decisions": [],
            }

        tool_def = {
            "name": base_payload["tool_name"],
            "metadata": {
                **_PREVIEW_TOOL_METADATA,
                "path_scope_action": requested_action,
            },
        }
        enforcement_result = await self.evaluate_tool_call(
            effective_policy=effective_policy,
            context=context,
            tool_name=base_payload["tool_name"],
            tool_args={"path": normalized_path},
            tool_def=tool_def,
            path_scope_candidates=[
                PathScopeCandidate(
                    path=normalized_path,
                    action=requested_action,  # type: ignore[arg-type]
                    source="effective_permission_preview",
                    display_path=normalized_path,
                )
            ],
        )

        path_decisions = _safe_path_decisions(enforcement_result)
        selected_decision = path_decisions[0] if path_decisions else {}
        scope_payload = enforcement_result.get("scope_payload")
        scope_payload = dict(scope_payload) if isinstance(scope_payload, Mapping) else {}
        preview = {
            **base_payload,
            "outcome": _preview_outcome(enforcement_result),
            "within_scope": bool(enforcement_result.get("within_scope", True)),
            "reason_code": str(enforcement_result.get("reason") or "").strip() or None,
            "grant_source": selected_decision.get("grant_source"),
            "grant_outcome": selected_decision.get("grant_outcome"),
            "matched_grant_prefix": selected_decision.get("matched_grant_prefix"),
            "matched_grant_effect": selected_decision.get("matched_grant_effect"),
            "path_scope_mode": scope_payload.get("path_scope_mode"),
            "workspace_id": scope_payload.get("workspace_id"),
            "path_allowlist_prefixes": list(scope_payload.get("path_allowlist_prefixes") or []),
            "path_decisions": path_decisions,
        }
        return {key: value for key, value in preview.items() if value not in ("", [])}

    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any | None,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        path_scope_candidates: list[PathScopeCandidate] | None = None,
    ) -> dict[str, Any]:
        if self._path_scope_service is None:
            raise RuntimeError("McpHubPathEnforcementService requires an explicit path_scope_service")
        scope = await self._path_scope_service.resolve_for_context(
            effective_policy=effective_policy,
            context=context,
        )
        result = {
            "enabled": bool(scope.get("enabled")),
            "within_scope": True,
            "reason": None,
            "force_approval": False,
            "normalized_paths": [],
            "scope_payload": None,
        }
        if not result["enabled"]:
            return result

        reason = str(scope.get("reason") or "").strip() or None
        if reason:
            if (
                reason == "workspace_root_unavailable"
                and str(scope.get("workspace_id") or "").strip()
                and scope.get("selected_assignment_id") not in (None, "")
                and str(scope.get("selected_workspace_source_mode") or "").strip()
            ):
                return self._blocked_result(
                    scope=scope,
                    reason="workspace_unresolvable_for_trust_source",
                    force_approval=False,
                )
            return self._blocked_result(scope=scope, reason=reason)

        allowed_workspace_ids = _unique(_as_str_list((effective_policy or {}).get("selected_assignment_workspace_ids")))
        active_workspace_id = str(scope.get("workspace_id") or "").strip()
        if allowed_workspace_ids and active_workspace_id not in allowed_workspace_ids:
            return self._blocked_result(
                scope=scope,
                reason="workspace_not_allowed_but_trusted",
                force_approval=True,
                allowed_workspace_ids=allowed_workspace_ids,
            )

        metadata = _tool_metadata(tool_def)
        if not _tool_uses_filesystem(metadata):
            return result

        if not _path_boundable(metadata):
            return self._blocked_result(scope=scope, reason="tool_not_path_boundable")

        inferred_action = _path_scope_action(metadata)
        candidate_actions: list[str] = []
        if path_scope_candidates:
            raw_paths = [candidate.path for candidate in path_scope_candidates]
            candidate_actions = [candidate.action for candidate in path_scope_candidates]
        else:
            hints = _path_argument_hints(metadata)
            raw_paths = _extract_candidate_paths(tool_args, hints)
            candidate_actions = [inferred_action for _raw_path in raw_paths]
        if not raw_paths:
            return self._blocked_result(scope=scope, reason="path_unresolvable")

        workspace_root_text = str(scope.get("workspace_root") or "").strip()
        scope_root = _scope_root_from_scope(scope)
        if not workspace_root_text or scope_root is None:
            return self._blocked_result(scope=scope, reason="workspace_root_unavailable")
        workspace_root = Path(workspace_root_text).expanduser().resolve(strict=False)
        base_path = Path(str(scope.get("cwd") or workspace_root)).expanduser().resolve(strict=False)
        path_allowlist_prefixes = _policy_allowlist_prefixes(effective_policy)
        path_grants = _policy_path_grants(effective_policy)
        is_multi_root_candidate = (
            str(scope.get("path_scope_mode") or "").strip() == "workspace_root" and len(allowed_workspace_ids) > 1
        )

        if is_multi_root_candidate:
            if self._multi_root_path_service is None:
                raise RuntimeError(
                    "McpHubPathEnforcementService requires an explicit multi_root_path_service for multi-root evaluation"
                )
            bundle_raw_paths, bundle_candidate_actions = _unique_paths_with_actions(
                raw_paths,
                candidate_actions,
                default_action=inferred_action,
            )
            multi_root_result = await self._multi_root_path_service.resolve_path_bundle(
                raw_paths=bundle_raw_paths,
                active_workspace_id=active_workspace_id,
                active_workspace_root=str(scope.get("workspace_root") or "").strip() or None,
                active_base_path=str(scope.get("cwd") or workspace_root),
                allowed_workspace_ids=allowed_workspace_ids,
                user_id=str(getattr(context, "user_id", None) or "").strip() or None,
                workspace_trust_source=str(scope.get("selected_workspace_trust_source") or "").strip() or None,
                owner_scope_type=str(scope.get("selected_workspace_scope_type") or "").strip() or None,
                owner_scope_id=scope.get("selected_workspace_scope_id"),
            )
            if not bool(multi_root_result.get("ok")):
                return self._blocked_result(
                    scope=scope,
                    reason=str(multi_root_result.get("reason") or "path_outside_workspace_bundle"),
                    normalized_paths=list(multi_root_result.get("normalized_paths") or []),
                    path_allowlist_prefixes=path_allowlist_prefixes,
                    force_approval=False,
                    allowed_workspace_ids=allowed_workspace_ids,
                    workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                    workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                    path_workspace_map=dict(multi_root_result.get("path_workspace_map") or {}),
                )

            normalized_paths = list(multi_root_result.get("normalized_paths") or [])
            path_workspace_map = dict(multi_root_result.get("path_workspace_map") or {})
            resolved_workspace_roots_by_id = dict(multi_root_result.get("resolved_workspace_roots_by_id") or {})
            for path_index, normalized_text in enumerate(normalized_paths):
                matched_workspace_id = str(path_workspace_map.get(normalized_text) or "").strip()
                matched_root_text = str(resolved_workspace_roots_by_id.get(matched_workspace_id) or "").strip()
                if not matched_root_text:
                    return self._blocked_result(
                        scope=scope,
                        reason="path_outside_workspace_bundle",
                        normalized_paths=normalized_paths,
                        path_allowlist_prefixes=path_allowlist_prefixes,
                        force_approval=False,
                        allowed_workspace_ids=allowed_workspace_ids,
                        workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                        workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                        path_workspace_map=path_workspace_map,
                    )
                normalized = Path(normalized_text).expanduser().resolve(strict=False)
                matched_root = Path(matched_root_text).expanduser().resolve(strict=False)
                allowlist_roots = _allowlist_roots(
                    workspace_root=matched_root,
                    allowlist_prefixes=path_allowlist_prefixes,
                )
                if not _is_within(matched_root, normalized):
                    return self._blocked_result(
                        scope=scope,
                        reason="path_outside_workspace_scope",
                        normalized_paths=normalized_paths,
                        path_allowlist_prefixes=path_allowlist_prefixes,
                        allowed_workspace_ids=allowed_workspace_ids,
                        workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                        workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                        path_workspace_map=path_workspace_map,
                    )
                if (
                    path_grants is None
                    and allowlist_roots
                    and not any(_is_within(root, normalized) for root in allowlist_roots)
                ):
                    return self._blocked_result(
                        scope=scope,
                        reason="path_outside_allowlist_scope",
                        normalized_paths=normalized_paths,
                        path_allowlist_prefixes=path_allowlist_prefixes,
                        allowed_workspace_ids=allowed_workspace_ids,
                        workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                        workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                        path_workspace_map=path_workspace_map,
                    )
                if path_grants is not None:
                    relative_path = _relative_path_for_decision(matched_root, normalized)
                    if relative_path is None:
                        return self._blocked_result(
                            scope=scope,
                            reason="path_outside_workspace_scope",
                            normalized_paths=normalized_paths,
                            path_allowlist_prefixes=path_allowlist_prefixes,
                            allowed_workspace_ids=allowed_workspace_ids,
                            workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                            workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                            path_workspace_map=path_workspace_map,
                        )
                    action = (
                        bundle_candidate_actions[path_index]
                        if path_index < len(bundle_candidate_actions)
                        else inferred_action
                    )
                    decision = _path_grant_decision(
                        relative_path=relative_path,
                        action=action,
                        path_grants=path_grants,
                    )
                    if decision["grant_outcome"] != "allowed":
                        return self._grant_blocked_result(
                            scope=scope,
                            reason=str(decision.get("reason_code") or "path_action_not_granted"),
                            path_decisions=[decision],
                        )
            result["normalized_paths"] = normalized_paths
            path_decisions: list[dict[str, Any]] = []
            if path_grants is not None:
                for path_index, normalized_text in enumerate(normalized_paths):
                    matched_workspace_id = str(path_workspace_map.get(normalized_text) or "").strip()
                    matched_root_text = str(resolved_workspace_roots_by_id.get(matched_workspace_id) or "").strip()
                    matched_root = Path(matched_root_text).expanduser().resolve(strict=False)
                    relative_path = _relative_path_for_decision(
                        matched_root, Path(normalized_text).expanduser().resolve(strict=False)
                    )
                    if relative_path is None:
                        continue
                    action = (
                        bundle_candidate_actions[path_index]
                        if path_index < len(bundle_candidate_actions)
                        else inferred_action
                    )
                    path_decisions.append(
                        _path_grant_decision(
                            relative_path=relative_path,
                            action=action,
                            path_grants=path_grants,
                        )
                    )
                result["path_decisions"] = path_decisions
            result["scope_payload"] = self._scope_payload(
                scope=scope,
                normalized_paths=normalized_paths,
                path_allowlist_prefixes=path_allowlist_prefixes,
                path_decisions=path_decisions,
                allowed_workspace_ids=allowed_workspace_ids,
                workspace_bundle_ids=list(multi_root_result.get("workspace_bundle_ids") or []),
                workspace_bundle_roots=list(multi_root_result.get("workspace_bundle_roots") or []),
                path_workspace_map=path_workspace_map,
            )
            return result

        allowlist_roots = _allowlist_roots(
            workspace_root=workspace_root,
            allowlist_prefixes=path_allowlist_prefixes,
        )

        normalized_paths: list[str] = []
        path_decisions: list[dict[str, Any]] = []
        for raw_path in raw_paths:
            path_index = len(normalized_paths)
            normalized = _normalize_candidate_path(raw_path, base_path=base_path)
            normalized_paths.append(str(normalized))
            if not _is_within(workspace_root, normalized):
                return self._blocked_result(
                    scope=scope,
                    reason="path_outside_workspace_scope",
                    normalized_paths=normalized_paths,
                )
            if not _is_within(scope_root, normalized):
                return self._blocked_result(
                    scope=scope,
                    reason="path_outside_current_folder_scope",
                    normalized_paths=normalized_paths,
                    path_allowlist_prefixes=path_allowlist_prefixes,
                )
            if path_grants is not None:
                relative_path = _relative_path_for_decision(workspace_root, normalized)
                if relative_path is None:
                    return self._blocked_result(
                        scope=scope,
                        reason="path_outside_workspace_scope",
                        normalized_paths=normalized_paths,
                    )
                action = candidate_actions[path_index] if path_index < len(candidate_actions) else inferred_action
                decision = _path_grant_decision(
                    relative_path=relative_path,
                    action=action,
                    path_grants=path_grants,
                )
                path_decisions.append(decision)
                if decision["grant_outcome"] != "allowed":
                    return self._grant_blocked_result(
                        scope=scope,
                        reason=str(decision.get("reason_code") or "path_action_not_granted"),
                        path_decisions=list(path_decisions),
                    )
                continue
            if allowlist_roots and not any(_is_within(root, normalized) for root in allowlist_roots):
                return self._blocked_result(
                    scope=scope,
                    reason="path_outside_allowlist_scope",
                    normalized_paths=normalized_paths,
                    path_allowlist_prefixes=path_allowlist_prefixes,
                )

        result["normalized_paths"] = normalized_paths
        if path_decisions:
            result["path_decisions"] = list(path_decisions)
        result["scope_payload"] = self._scope_payload(
            scope=scope,
            normalized_paths=normalized_paths,
            path_allowlist_prefixes=path_allowlist_prefixes,
            path_decisions=path_decisions,
        )
        return result

    @staticmethod
    def _scope_payload(
        *,
        scope: dict[str, Any],
        normalized_paths: list[str] | None = None,
        reason: str | None = None,
        path_allowlist_prefixes: list[str] | None = None,
        path_decisions: list[dict[str, Any]] | None = None,
        allowed_workspace_ids: list[str] | None = None,
        workspace_bundle_ids: list[str] | None = None,
        workspace_bundle_roots: list[str] | None = None,
        path_workspace_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        scope_root = _scope_root_from_scope(scope)
        payload = {
            "path_scope_mode": str(scope.get("path_scope_mode") or "none").strip() or "none",
            "workspace_root": str(scope.get("workspace_root") or "").strip() or None,
            "workspace_id": str(scope.get("workspace_id") or "").strip() or None,
            "selected_workspace_trust_source": str(scope.get("selected_workspace_trust_source") or "").strip() or None,
            "selected_assignment_id": scope.get("selected_assignment_id"),
            "workspace_source_mode": str(scope.get("selected_workspace_source_mode") or "").strip() or None,
            "scope_root": str(scope_root) if scope_root is not None else None,
        }
        if normalized_paths:
            payload["normalized_paths"] = list(normalized_paths)
        if path_allowlist_prefixes:
            payload["path_allowlist_prefixes"] = list(path_allowlist_prefixes)
        if path_decisions:
            payload["path_decisions"] = [dict(decision) for decision in path_decisions]
        if allowed_workspace_ids:
            payload["allowed_workspace_ids"] = list(allowed_workspace_ids)
        if workspace_bundle_ids:
            payload["workspace_bundle_ids"] = sorted(_unique(list(workspace_bundle_ids)))
        if workspace_bundle_roots:
            payload["workspace_bundle_roots"] = sorted(_unique(list(workspace_bundle_roots)))
        if path_workspace_map:
            payload["path_workspace_map"] = {
                key: value
                for key, value in sorted(path_workspace_map.items(), key=lambda item: item[0])
                if str(key or "").strip() and str(value or "").strip()
            }
        if reason:
            payload["reason"] = reason
        return {key: value for key, value in payload.items() if value not in (None, "", [])}

    @staticmethod
    def _safe_grant_scope_payload(
        *,
        scope: dict[str, Any],
        reason: str,
        path_decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "path_scope_mode": str(scope.get("path_scope_mode") or "none").strip() or "none",
            "workspace_id": str(scope.get("workspace_id") or "").strip() or None,
            "reason": reason,
            "path_decisions": [dict(decision) for decision in path_decisions],
        }
        return {key: value for key, value in payload.items() if value not in (None, "", [])}

    def _blocked_result(
        self,
        *,
        scope: dict[str, Any],
        reason: str,
        normalized_paths: list[str] | None = None,
        path_allowlist_prefixes: list[str] | None = None,
        force_approval: bool = True,
        allowed_workspace_ids: list[str] | None = None,
        workspace_bundle_ids: list[str] | None = None,
        workspace_bundle_roots: list[str] | None = None,
        path_workspace_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(scope.get("enabled")),
            "within_scope": False,
            "reason": reason,
            "force_approval": force_approval,
            "normalized_paths": list(normalized_paths or []),
            "scope_payload": self._scope_payload(
                scope=scope,
                normalized_paths=list(normalized_paths or []),
                reason=reason,
                path_allowlist_prefixes=path_allowlist_prefixes,
                allowed_workspace_ids=allowed_workspace_ids,
                workspace_bundle_ids=workspace_bundle_ids,
                workspace_bundle_roots=workspace_bundle_roots,
                path_workspace_map=path_workspace_map,
            ),
        }

    def _grant_blocked_result(
        self,
        *,
        scope: dict[str, Any],
        reason: str,
        path_decisions: list[dict[str, Any]],
        force_approval: bool = True,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(scope.get("enabled")),
            "within_scope": False,
            "reason": reason,
            "force_approval": force_approval,
            "normalized_paths": [],
            "path_decisions": [dict(decision) for decision in path_decisions],
            "scope_payload": self._safe_grant_scope_payload(
                scope=scope,
                reason=reason,
                path_decisions=path_decisions,
            ),
        }


async def get_mcp_hub_path_enforcement_service() -> McpHubPathEnforcementService:
    """Create a path enforcement service backed by the current sandbox scope resolver."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    workspace_root_resolver = McpHubWorkspaceRootResolver(repo=repo)
    return McpHubPathEnforcementService(
        path_scope_service=McpHubPathScopeService(
            workspace_root_resolver=workspace_root_resolver,
        ),
        multi_root_path_service=McpHubMultiRootPathService(
            workspace_root_resolver=workspace_root_resolver,
        ),
    )
