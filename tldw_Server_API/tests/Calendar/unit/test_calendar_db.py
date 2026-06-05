from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Calendar.errors import CalendarReadOnlyError, CalendarValidationError


@pytest.fixture
def calendar_db(tmp_path):
    db = CalendarDatabase(db_path=tmp_path / "calendar.db")
    db.ensure_schema()
    return db


def _create_calendar(db: CalendarDatabase):
    return db.create_calendar(
        tenant_id="default",
        owner_user_id=1,
        org_id=None,
        name="Research",
        timezone="America/Los_Angeles",
        color="#2563eb",
    )


def test_calendar_db_creates_personal_calendar(calendar_db):
    calendar = _create_calendar(calendar_db)

    assert calendar.name == "Research"
    assert calendar.owner_user_id == 1
    assert calendar.archived_at is None


def test_calendar_db_creates_owner_membership_automatically(calendar_db):
    calendar = _create_calendar(calendar_db)

    memberships = calendar_db.list_memberships(calendar.id)

    assert len(memberships) == 1
    assert memberships[0].calendar_id == calendar.id
    assert memberships[0].principal_type == "user"
    assert memberships[0].principal_id == "1"
    assert memberships[0].role == "owner"


def test_create_item_rejects_provider_owned_local_creates(calendar_db):
    calendar = _create_calendar(calendar_db)
    start_at = datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc).isoformat()
    end_at = datetime(2026, 6, 5, 18, 0, tzinfo=timezone.utc).isoformat()

    with pytest.raises(CalendarReadOnlyError):
        calendar_db.create_item(
            calendar_id=calendar.id,
            kind="event",
            title="Provider meeting",
            source_owner="provider",
            provider_owned=True,
            start_at=start_at,
            end_at=end_at,
        )


def test_remote_deleted_provider_tombstones_are_hidden_from_window_queries(calendar_db):
    calendar = _create_calendar(calendar_db)
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref="calendar-secret-1",
        account_metadata_json='{"principal": "user@example.com"}',
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
        remote_display_name="Remote Calendar",
    )
    start = datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    item = calendar_db.upsert_provider_item(
        calendar_id=calendar.id,
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        title="Imported meeting",
        start_at=start.isoformat(),
        end_at=end.isoformat(),
        provider_payload_json='{"uid": "remote-event-1"}',
    )

    calendar_db.mark_provider_item_remote_deleted(
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        remote_deleted_at=(end + timedelta(days=1)).isoformat(),
    )

    assert calendar_db.get_item(item.id, include_deleted=True).remote_deleted_at is not None
    visible_items = calendar_db.list_items_window(
        calendar_ids=[calendar.id],
        window_start=(start - timedelta(days=1)).isoformat(),
        window_end=(end + timedelta(days=1)).isoformat(),
    )
    assert visible_items == []


def test_external_account_rows_expose_secret_ref_not_credential_payload(calendar_db):
    secret_ref = calendar_db.create_secret_ref(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        encrypted_payload="encrypted-token-payload",
    )

    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=secret_ref,
        account_metadata_json='{"principal": "user@example.com"}',
    )

    fetched = calendar_db.get_external_account(account.id)

    assert fetched.secret_ref == secret_ref
    assert "encrypted-token-payload" not in repr(fetched)
    assert calendar_db.resolve_secret_ref(secret_ref) == "encrypted-token-payload"


@pytest.mark.parametrize("account_cleanup_method", ["revoke", "delete"])
def test_external_account_cleanup_wipes_secret_payload(calendar_db, account_cleanup_method):
    secret_ref = calendar_db.create_secret_ref(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        encrypted_payload="encrypted-token-payload",
    )
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=secret_ref,
    )

    if account_cleanup_method == "revoke":
        calendar_db.revoke_external_account(account.id)
    else:
        calendar_db.delete_external_account(account.id)

    with pytest.raises(CalendarValidationError):
        calendar_db.resolve_secret_ref(secret_ref)
    with calendar_db.connection() as conn:
        row = conn.execute(
            """
            SELECT encrypted_payload, deleted_at
            FROM calendar_external_account_secrets
            WHERE secret_ref = ?
            """,
            (secret_ref,),
        ).fetchone()
    assert row is not None
    assert row["encrypted_payload"] == ""
    assert row["deleted_at"] is not None


def test_external_binding_stores_window_and_provider_capabilities(calendar_db):
    calendar = _create_calendar(calendar_db)
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref="calendar-secret-1",
    )

    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
        remote_display_name="Remote Calendar",
        lookback_days=30,
        lookahead_days=60,
        provider_capabilities_json='{"read_only": true, "supports_vevent": true}',
    )

    assert binding.lookback_days == 30
    assert binding.lookahead_days == 60
    assert binding.provider_capabilities_json == '{"read_only": true, "supports_vevent": true}'


def test_destructive_account_cleanup_preserves_copied_tldw_item(calendar_db):
    calendar = _create_calendar(calendar_db)
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref="calendar-secret-1",
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
    )
    start = datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc)
    provider_item = calendar_db.upsert_provider_item(
        calendar_id=calendar.id,
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        title="Imported meeting",
        start_at=start.isoformat(),
        end_at=(start + timedelta(hours=1)).isoformat(),
    )
    copied_item = calendar_db.create_item(
        calendar_id=calendar.id,
        kind="event",
        title="Copied meeting",
        start_at=start.isoformat(),
        end_at=(start + timedelta(hours=1)).isoformat(),
        copied_from_item_id=provider_item.id,
    )

    calendar_db.delete_external_account(
        account.id,
        destructive_imported_record_cleanup=True,
    )

    copied_after_cleanup = calendar_db.get_item(copied_item.id, include_deleted=True)
    assert copied_after_cleanup.source_owner == "tldw"
    assert copied_after_cleanup.copied_from_item_id is None


def test_remote_tombstone_cleanup_preserves_copied_tldw_item(calendar_db):
    calendar = _create_calendar(calendar_db)
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref="calendar-secret-1",
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
    )
    start = datetime(2026, 6, 5, 17, 0, tzinfo=timezone.utc)
    provider_item = calendar_db.upsert_provider_item(
        calendar_id=calendar.id,
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        title="Imported meeting",
        start_at=start.isoformat(),
        end_at=(start + timedelta(hours=1)).isoformat(),
    )
    copied_item = calendar_db.create_item(
        calendar_id=calendar.id,
        kind="event",
        title="Copied meeting",
        start_at=start.isoformat(),
        end_at=(start + timedelta(hours=1)).isoformat(),
        copied_from_item_id=provider_item.id,
    )
    remote_deleted_at = (start + timedelta(days=1)).isoformat()
    calendar_db.mark_provider_item_remote_deleted(
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        remote_deleted_at=remote_deleted_at,
    )

    deleted_count = calendar_db.delete_remote_tombstones_eligible_for_cleanup(
        before_iso=(start + timedelta(days=2)).isoformat()
    )

    copied_after_cleanup = calendar_db.get_item(copied_item.id, include_deleted=True)
    assert deleted_count == 1
    assert copied_after_cleanup.source_owner == "tldw"
    assert copied_after_cleanup.copied_from_item_id is None
