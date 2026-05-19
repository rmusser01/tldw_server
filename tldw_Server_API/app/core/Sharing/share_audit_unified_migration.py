"""Explicit migration of legacy Sharing audit rows into unified audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo
from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter


class ShareAuditMigrationError(RuntimeError):
    """Raised when legacy Sharing audit rows cannot be migrated safely."""


@dataclass(frozen=True)
class ShareAuditMigrationReport:
    scanned: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    max_legacy_id: int = 0


async def _iter_legacy_batches(
    repo: SharedWorkspaceRepo,
    *,
    batch_size: int,
) -> AsyncIterator[list[dict]]:
    """Yield legacy audit rows one batch at a time to cap memory usage."""
    after_id = 0
    while True:
        batch = await repo.list_legacy_audit_events_for_migration(
            after_id=after_id,
            limit=batch_size,
        )
        if not batch:
            return
        yield batch
        after_id = int(batch[-1]["id"])


async def migrate_share_audit_log_to_unified_audit(
    *,
    repo: SharedWorkspaceRepo,
    shared_audit_db_path: str | Path | None = None,
    batch_size: int = 500,
) -> ShareAuditMigrationReport:
    writer = UnifiedShareAuditWriter(
        db_path=str(shared_audit_db_path) if shared_audit_db_path is not None else None
    )
    await writer.initialize()
    try:
        state = await writer.get_identity_state()
        scanned = 0
        inserted = 0
        skipped_existing = 0
        max_legacy_id = 0
        floor_bumped = False

        async for batch in _iter_legacy_batches(repo, batch_size=batch_size):
            batch_max_id = max(int(row["id"]) for row in batch)
            max_legacy_id = max(max_legacy_id, batch_max_id)
            scanned += len(batch)

            # Check for conflicting ids in this batch
            conflicting_ids = sorted(
                int(row["id"])
                for row in batch
                if int(row["id"]) not in state.legacy_ids
                and int(row["id"]) in state.compatibility_ids
            )
            if conflicting_ids:
                raise ShareAuditMigrationError(
                    "Legacy Sharing audit migration cannot proceed because compatibility id "
                    f"{conflicting_ids[0]} is already occupied by a non-legacy unified event"
                )

            # Bump compatibility floor once we know the full max_legacy_id so far
            if not floor_bumped:
                await writer.bump_compatibility_floor(max(max_legacy_id, state.max_compatibility_id))
                floor_bumped = True

            # Process rows in this batch immediately
            for row in batch:
                legacy_id = int(row["id"])
                if legacy_id in state.legacy_ids:
                    skipped_existing += 1
                    continue
                did_insert = await writer.import_legacy_event(
                    legacy_audit_id=legacy_id,
                    event_type=str(row.get("event_type") or "share.unknown"),
                    resource_type=str(row.get("resource_type") or "workspace"),
                    resource_id=str(row.get("resource_id") or ""),
                    owner_user_id=int(row["owner_user_id"]),
                    actor_user_id=row.get("actor_user_id"),
                    share_id=row.get("share_id"),
                    token_id=row.get("token_id"),
                    metadata=row.get("metadata"),
                    ip_address=row.get("ip_address"),
                    user_agent=row.get("user_agent"),
                    created_at=row.get("created_at"),
                )
                if did_insert:
                    inserted += 1
                else:
                    skipped_existing += 1

        # Final floor bump in case later batches had higher ids
        if max_legacy_id > 0:
            await writer.bump_compatibility_floor(max(max_legacy_id, state.max_compatibility_id))

        return ShareAuditMigrationReport(
            scanned=scanned,
            inserted=inserted,
            skipped_existing=skipped_existing,
            max_legacy_id=max_legacy_id,
        )
    finally:
        await writer.stop()
