from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

from .config import CodeGraphSettings
from .models import WorkspaceResolution


class WorkspaceRootResolver(Protocol):
    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        ...


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


class CodeGraphWorkspaceResolver:
    """Resolve trusted workspace roots and tldw-managed index paths."""

    def __init__(
        self,
        workspace_root_resolver: WorkspaceRootResolver | None = None,
        settings: CodeGraphSettings | None = None,
    ) -> None:
        if workspace_root_resolver is None:
            from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
                McpHubWorkspaceRootResolver,
            )

            self._workspace_root_resolver: WorkspaceRootResolver = McpHubWorkspaceRootResolver()
        else:
            self._workspace_root_resolver = workspace_root_resolver
        self._settings = settings or CodeGraphSettings.from_mapping({})

    async def resolve(self, context: Any | None) -> WorkspaceResolution:
        metadata = getattr(context, "metadata", None)
        metadata_map = dict(metadata) if isinstance(metadata, dict) else {}
        session_id = _first_nonempty(
            getattr(context, "session_id", None),
            metadata_map.get("session_id"),
        )
        user_id = _first_nonempty(
            getattr(context, "user_id", None),
            metadata_map.get("user_id"),
        )
        workspace_trust_source = _first_nonempty(
            metadata_map.get("workspace_trust_source"),
            metadata_map.get("selected_workspace_trust_source"),
        )
        if session_id and not user_id and workspace_trust_source != "shared_registry":
            raise PermissionError("workspace_root_unavailable")

        workspace_id = _first_nonempty(metadata_map.get("workspace_id"))
        resolution = await self._workspace_root_resolver.resolve_for_context(
            session_id=session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            workspace_trust_source=workspace_trust_source,
            owner_scope_type=_first_nonempty(
                metadata_map.get("owner_scope_type"),
                metadata_map.get("selected_workspace_scope_type"),
            ),
            owner_scope_id=_first_nonempty(
                metadata_map.get("owner_scope_id"),
                metadata_map.get("selected_workspace_scope_id"),
            ),
        )
        workspace_root_raw = str(resolution.get("workspace_root") or "").strip()
        if not workspace_root_raw:
            reason = str(resolution.get("reason") or "workspace_root_unavailable")
            raise PermissionError(reason)

        workspace_root = Path(workspace_root_raw).expanduser().resolve(strict=False)
        resolved_workspace_id = _first_nonempty(resolution.get("workspace_id"), workspace_id)
        source = _first_nonempty(resolution.get("source"), workspace_trust_source)
        workspace_key = self._workspace_key(
            user_id=user_id,
            workspace_id=resolved_workspace_id,
            trust_source=source,
            workspace_root=workspace_root,
        )
        index_base_dir = self._settings.index_base_dir.expanduser().resolve(strict=False)
        index_db_path = index_base_dir / workspace_key / "codegraph.db"
        return WorkspaceResolution(
            workspace_root=workspace_root,
            workspace_key=workspace_key,
            index_db_path=index_db_path,
            workspace_id=resolved_workspace_id,
            source=source,
        )

    @staticmethod
    def _workspace_key(
        *,
        user_id: str | None,
        workspace_id: str | None,
        trust_source: str | None,
        workspace_root: Path,
    ) -> str:
        identity = "\n".join(
            (
                f"user={user_id or ''}",
                f"workspace={workspace_id or ''}",
                f"trust={trust_source or ''}",
                f"root={workspace_root.as_posix()}",
            )
        )
        return f"ws_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"
