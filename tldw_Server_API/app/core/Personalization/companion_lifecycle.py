from __future__ import annotations

"""Scoped purge and rebuild helpers for companion-derived state."""

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Personalization.companion_derivations import derive_companion_knowledge_cards
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_existing_companion_storage_user_id,
)


COMPANION_LIFECYCLE_SCOPES = frozenset(
    {
        "knowledge",
        "reflections",
        "derived_goals",
        "goal_progress",
    }
)


def _normalize_scope(scope: str) -> str:
    """Normalize and validate a requested companion lifecycle scope.

    Raises:
        ValueError: If the caller passes an unsupported lifecycle scope.
    """
    normalized = str(scope or "").strip().lower()
    if normalized not in COMPANION_LIFECYCLE_SCOPES:
        raise ValueError(f"Unsupported companion lifecycle scope: {scope}")
    return normalized


def _resolve_personalization_db(
    user_id: str,
    personalization_db: PersonalizationDB | None,
) -> PersonalizationDB:
    """Return the provided personalization DB or open the default DB for the user."""
    if personalization_db is not None:
        return personalization_db
    storage_user_id = resolve_existing_companion_storage_user_id(user_id)
    return PersonalizationDB.for_user(storage_user_id)


def _resolve_collections_db(
    user_id: str,
    collections_db: CollectionsDatabase | None,
) -> CollectionsDatabase:
    """Return the provided collections DB or open the default DB for the user."""
    if collections_db is not None:
        return collections_db
    return CollectionsDatabase.for_user(user_id=user_id)


def _empty_counts() -> dict[str, int]:
    """Return the standard zeroed count structure used by lifecycle responses."""
    return {
        "knowledge": 0,
        "reflections": 0,
        "notifications": 0,
        "derived_goals": 0,
        "goal_progress": 0,
    }


def purge_companion_scope(
    *,
    user_id: str | int,
    scope: str,
    personalization_db: PersonalizationDB | None = None,
    collections_db: CollectionsDatabase | None = None,
) -> dict[str, Any]:
    """Purge one scoped slice of companion state while preserving raw activity by default."""
    normalized_user_id = str(user_id)
    normalized_scope = _normalize_scope(scope)
    db = _resolve_personalization_db(normalized_user_id, personalization_db)
    deleted_counts = _empty_counts()

    if normalized_scope == "knowledge":
        deleted_counts["knowledge"] = db.delete_companion_knowledge_cards(normalized_user_id)
    elif normalized_scope == "reflections":
        reflection_ids, deleted = db.delete_companion_reflection_activity_events(normalized_user_id)
        deleted_counts["reflections"] = deleted
        if reflection_ids:
            cdb = _resolve_collections_db(normalized_user_id, collections_db)
            deleted_counts["notifications"] = cdb.delete_user_notifications_by_link(
                link_type="companion_reflection",
                link_ids=reflection_ids,
            )
    elif normalized_scope == "derived_goals":
        deleted_counts["derived_goals"] = db.delete_companion_goals_by_origin_kind(
            normalized_user_id,
            "derived",
        )
    elif normalized_scope == "goal_progress":
        deleted_counts["goal_progress"] = db.reset_companion_goal_progress(
            normalized_user_id,
            progress_mode="computed",
        )

    return {
        "status": "completed",
        "scope": normalized_scope,
        "deleted_counts": deleted_counts,
        "rebuilt_counts": _empty_counts(),
    }


def rebuild_companion_scope(
    *,
    user_id: str | int,
    scope: str,
    personalization_db: PersonalizationDB | None = None,
    collections_db: CollectionsDatabase | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Rebuild one scoped slice of derived companion state from preserved user activity."""
    normalized_user_id = str(user_id)
    normalized_scope = _normalize_scope(scope)
    db = _resolve_personalization_db(normalized_user_id, personalization_db)
    deleted_counts = _empty_counts()
    rebuilt_counts = _empty_counts()

    if normalized_scope == "knowledge":
        deleted_counts["knowledge"] = db.delete_companion_knowledge_cards(normalized_user_id)
        cards = derive_companion_knowledge_cards(db, user_id=normalized_user_id)
        for card in cards:
            db.upsert_companion_knowledge_card(user_id=normalized_user_id, **card)
        rebuilt_counts["knowledge"] = len(cards)
    elif normalized_scope == "reflections":
        from tldw_Server_API.app.core.Personalization.companion_reflection_jobs import (
            run_companion_reflection_job,
        )

        purge_result = purge_companion_scope(
            user_id=normalized_user_id,
            scope="reflections",
            personalization_db=db,
            collections_db=collections_db,
        )
        deleted_counts.update(purge_result["deleted_counts"])
        cdb = _resolve_collections_db(normalized_user_id, collections_db)
        current_time = now or datetime.now(timezone.utc)
        for cadence in ("daily", "weekly"):
            result = run_companion_reflection_job(
                user_id=normalized_user_id,
                cadence=cadence,
                now=current_time,
                personalization_db=db,
                collections_db=cdb,
            )
            if result.get("status") != "completed":
                continue
            rebuilt_counts["reflections"] += 1
            if result.get("notification_id"):
                rebuilt_counts["notifications"] += 1
    elif normalized_scope == "derived_goals":
        # Derived goal regeneration is not implemented yet; keep existing goal rows intact.
        rebuilt_counts["derived_goals"] = 0
    elif normalized_scope == "goal_progress":
        # Computed progress regeneration is not implemented yet; do not clear user-visible progress.
        rebuilt_counts["goal_progress"] = 0

    return {
        "status": "completed",
        "scope": normalized_scope,
        "deleted_counts": deleted_counts,
        "rebuilt_counts": rebuilt_counts,
    }


__all__ = [
    "COMPANION_LIFECYCLE_SCOPES",
    "purge_companion_scope",
    "rebuild_companion_scope",
]
