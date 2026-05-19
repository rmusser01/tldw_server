"""Service for checking and persisting user permission decisions.

Part of the unified decision hierarchy (step 4): after MCPHub snapshot
checks and before admin ACP permission policies.  Allows users to
"remember" their approve/deny choices so they are auto-applied on
subsequent tool calls.
"""
from __future__ import annotations

import fnmatch
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB


class PermissionDecisionService:
    """Thin service layer over the ``permission_decisions`` table."""

    def __init__(self, db: ACPSessionsDB) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def check(
        self,
        user_id: int,
        tool_name: str,
        session_id: str | None = None,
    ) -> str | None:
        """Return persisted decision (``'allow'`` | ``'deny'``) if one matches, else ``None``.

        Checks global decisions first, then session-scoped if *session_id*
        is provided.  Uses :func:`fnmatch.fnmatch` for pattern matching.
        """
        decisions = self._db.list_permission_decisions(user_id=user_id)
        # Global scope first, then session scope
        for d in decisions:
            if fnmatch.fnmatch(tool_name, d["tool_pattern"]) and d["scope"] == "global":
                return d["decision"]
        if session_id is not None:
            for d in decisions:
                if (
                    fnmatch.fnmatch(tool_name, d["tool_pattern"])
                    and d["scope"] == "session"
                    and d.get("session_id") == session_id
                ):
                    return d["decision"]
        return None

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def persist(
        self,
        user_id: int,
        tool_pattern: str,
        decision: str,
        scope: str = "session",
        session_id: str | None = None,
        persona_id: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Persist a permission decision. Returns the decision ID."""
        decision_id = str(uuid.uuid4())
        self._db.insert_permission_decision(
            id=decision_id,
            user_id=user_id,
            tool_pattern=tool_pattern,
            decision=decision,
            scope=scope,
            session_id=session_id,
            persona_id=persona_id,
            reason=reason,
        )
        return decision_id

    # ------------------------------------------------------------------
    # List / Revoke
    # ------------------------------------------------------------------

    def list_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Return all non-expired persisted decisions for *user_id*."""
        return self._db.list_permission_decisions(user_id=user_id)

    def revoke(self, decision_id: str) -> bool:
        """Delete a persisted decision by ID. Returns ``True`` if removed."""
        return self._db.delete_permission_decision(decision_id)
