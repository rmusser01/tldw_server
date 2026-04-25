"""Tests for migrating legacy share audit rows into unified audit."""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import ValidationError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_share_audit_backfill_is_idempotent(repo, sharing_db, tmp_path):
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        actor_user_id=2,
        share_id=5,
        metadata={"scope_type": "team"},
    )
    legacy_id, legacy_created_at = sharing_db.execute(
        "SELECT id, created_at FROM share_audit_log ORDER BY id ASC LIMIT 1"
    ).fetchone()

    from tldw_Server_API.app.core.Sharing.share_audit_unified_migration import (
        migrate_share_audit_log_to_unified_audit,
    )
    from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

    shared_audit_path = tmp_path / "audit_shared.db"
    report1 = await migrate_share_audit_log_to_unified_audit(
        repo=repo,
        shared_audit_db_path=shared_audit_path,
    )
    report2 = await migrate_share_audit_log_to_unified_audit(
        repo=repo,
        shared_audit_db_path=shared_audit_path,
    )

    writer = UnifiedShareAuditWriter(db_path=str(shared_audit_path))
    await writer.initialize()
    try:
        rows = await writer.query_events(owner_user_id=1)
        compatibility_ids = [row["id"] for row in rows]

        await writer.log_event(
            event_type="share.updated",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
            actor_user_id=2,
            share_id=5,
            metadata={"scope_type": "team"},
        )
        after_rows = await writer.query_events(owner_user_id=1)
    finally:
        await writer.stop()

    assert report1.inserted == 1
    assert report1.max_legacy_id == legacy_id
    assert report2.inserted == 0
    assert compatibility_ids == [legacy_id]
    assert rows[0]["event_id"] == f"share-audit-legacy-{legacy_id}"
    assert rows[0]["created_at"] == legacy_created_at
    assert rows[0]["metadata"]["legacy_share_audit_id"] == legacy_id
    assert sorted(row["id"] for row in after_rows) == [legacy_id, legacy_id + 1]


@pytest.mark.asyncio
async def test_share_audit_legacy_import_replay_is_skipped(tmp_path):
    from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

    shared_audit_path = tmp_path / "audit_shared.db"
    writer = UnifiedShareAuditWriter(db_path=str(shared_audit_path))
    await writer.initialize()
    try:
        inserted_id = await writer.import_legacy_event(
            legacy_share_audit_id=7,
            event_type="share.created",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
            actor_user_id=2,
            share_id=5,
            metadata={"scope_type": "team"},
            created_at="2026-04-09T19:00:00+00:00",
        )
        replay_id = await writer.import_legacy_event(
            legacy_share_audit_id=7,
            event_type="share.created",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
            actor_user_id=2,
            share_id=5,
            metadata={"scope_type": "team"},
            created_at="2026-04-09T19:00:00+00:00",
        )
        rows = await writer.query_events(owner_user_id=1)
    finally:
        await writer.stop()

    assert inserted_id == 7
    assert replay_id is None
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_share_audit_legacy_import_uses_validation_errors(tmp_path):
    from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

    shared_audit_path = tmp_path / "audit_shared.db"
    writer = UnifiedShareAuditWriter(db_path=str(shared_audit_path))
    await writer.initialize()
    try:
        with pytest.raises(ValidationError, match="must match"):
            await writer.import_legacy_event(
                legacy_share_audit_id=7,
                legacy_audit_id=8,
                event_type="share.created",
                resource_type="workspace",
                resource_id="ws-1",
                owner_user_id=1,
            )

        with pytest.raises(ValidationError, match="required"):
            await writer.import_legacy_event(
                event_type="share.created",
                resource_type="workspace",
                resource_id="ws-1",
                owner_user_id=1,
            )
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_share_audit_backfill_fails_when_legacy_ids_are_already_occupied(repo, tmp_path):
    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="ws-1",
        owner_user_id=1,
        actor_user_id=2,
        share_id=5,
    )

    from tldw_Server_API.app.core.Sharing.share_audit_unified_migration import (
        ShareAuditMigrationError,
        migrate_share_audit_log_to_unified_audit,
    )
    from tldw_Server_API.app.core.Sharing.unified_share_audit import UnifiedShareAuditWriter

    shared_audit_path = tmp_path / "audit_shared.db"
    writer = UnifiedShareAuditWriter(db_path=str(shared_audit_path))
    await writer.initialize()
    try:
        compatibility_id = await writer.log_event(
            event_type="share.updated",
            resource_type="workspace",
            resource_id="ws-1",
            owner_user_id=1,
            actor_user_id=2,
            share_id=5,
        )
    finally:
        await writer.stop()

    assert compatibility_id == 1

    with pytest.raises(ShareAuditMigrationError, match="compatibility id"):
        await migrate_share_audit_log_to_unified_audit(
            repo=repo,
            shared_audit_db_path=shared_audit_path,
        )
