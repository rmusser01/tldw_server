from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks.crypto import ProtectedValue
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
)
from tldw_Server_API.app.core.Admin_Webhooks.repository import (
    AdminWebhookRepository,
    IdempotencyLookupKind,
    RegistrationInsert,
    RegistrationPatch,
    RegistrationTarget,
    WebhookRepositoryError,
    WebhookRepositoryErrorCode,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@dataclass
class SQLiteRepositoryFixture:
    repository: AdminWebhookRepository
    pool: DatabasePool
    path: Path


@pytest_asyncio.fixture
async def sqlite_repo(tmp_path: Path) -> SQLiteRepositoryFixture:
    path = tmp_path / "admin-webhooks.db"
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{path}",
        )
    )
    await pool.initialize()
    fixture = SQLiteRepositoryFixture(
        repository=AdminWebhookRepository(pool),
        pool=pool,
        path=path,
    )
    try:
        yield fixture
    finally:
        await pool.close()


def _protected(label: str) -> ProtectedValue:
    return ProtectedValue(
        ciphertext_json=f'{{"ciphertext":"opaque-{label}"}}',
        key_id="key-2026-08",
    )


def _registration_insert(
    webhook_id: int,
    *,
    description: str = "Example",
    event_types: tuple[str, ...] = ("user.created",),
    active: bool = False,
    now: datetime = NOW,
) -> RegistrationInsert:
    return RegistrationInsert(
        id=webhook_id,
        description=description,
        target=RegistrationTarget(
            protected=_protected(f"target-{webhook_id}"),
            hostname="hooks.example.com",
            display="https://hooks.example.com",
        ),
        event_types=event_types,
        active=active,
        timeout_seconds=10,
        secret=_protected(f"secret-{webhook_id}"),
        secret_rotation_required=False,
        actor_user_id=7,
        now=now,
    )


async def _seed_registration(
    fixture: SQLiteRepositoryFixture,
    *,
    webhook_id: int | None = None,
    event_types: tuple[str, ...] = ("user.created",),
    active: bool = False,
    now: datetime = NOW,
):
    async with fixture.repository.transaction() as tx:
        allocated = webhook_id or await tx.allocate_registration_id()
        return await tx.insert_registration(
            _registration_insert(
                allocated,
                event_types=event_types,
                active=active,
                now=now,
            )
        )


def _idempotency_values(
    *,
    key: str = "0123456789abcdef0123456789abcdef",
    route: str = "/admin/webhooks",
    body: dict[str, object] | None = None,
) -> tuple[object, str, str]:
    scope = build_idempotency_scope(
        actor_id=7,
        operation="create",
        route=route,
    )
    return (
        scope,
        idempotency_lookup_digest(key, scope),
        canonical_request_hash(
            key,
            scope=scope,
            body=body or {"description": "Example"},
            conditional_version=None,
        ),
    )


async def test_create_commits_before_connection_close(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    created = await _seed_registration(sqlite_repo)

    reopened_pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{sqlite_repo.path}",
        )
    )
    await reopened_pool.initialize()
    try:
        reopened = AdminWebhookRepository(reopened_pool)
        assert await reopened.get_registration(created.id) == created
    finally:
        await reopened_pool.close()


async def test_sequence_allocation_is_unique_under_concurrency(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    async def allocate() -> int:
        async with sqlite_repo.repository.transaction() as tx:
            return await tx.allocate_registration_id()

    allocated = await asyncio.gather(*(allocate() for _ in range(12)))

    assert sorted(allocated) == list(range(1, 13))


async def test_insert_read_list_event_roundtrip_and_tombstone_exclusion(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    first = await _seed_registration(
        sqlite_repo,
        event_types=("user.deleted", "user.created"),
    )
    second = await _seed_registration(sqlite_repo)
    third = await _seed_registration(sqlite_repo)

    page = await sqlite_repo.repository.list_registrations(limit=2)
    assert [item.id for item in page] == [third.id, second.id]
    assert (
        await sqlite_repo.repository.list_registrations(
            limit=2,
            before_id=second.id,
        )
    ) == [first]
    assert first.event_types == ("user.deleted", "user.created")

    async with sqlite_repo.repository.transaction() as tx:
        deleted = await tx.soft_delete_registration(
            second.id,
            expected_revision=second.revision,
            actor_user_id=9,
            at=NOW + timedelta(minutes=1),
        )

    assert deleted.deleted_at == NOW + timedelta(minutes=1)
    assert await sqlite_repo.repository.get_registration(second.id) is None
    assert (
        await sqlite_repo.repository.get_registration(
            second.id,
            include_deleted=True,
        )
    ) == deleted
    assert [item.id for item in await sqlite_repo.repository.list_registrations(limit=10)] == [
        third.id,
        first.id,
    ]


async def test_patch_versions_follow_effective_field_changes(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    original = await _seed_registration(sqlite_repo)

    async with sqlite_repo.repository.transaction() as tx:
        no_op = await tx.patch_registration(
            original.id,
            expected_revision=original.revision,
            patch=RegistrationPatch(description=original.description),
            actor_user_id=8,
            at=NOW + timedelta(minutes=1),
        )
    assert no_op.changed is False
    assert no_op.registration == original

    async with sqlite_repo.repository.transaction() as tx:
        description = await tx.patch_registration(
            original.id,
            expected_revision=original.revision,
            patch=RegistrationPatch(description="Renamed"),
            actor_user_id=8,
            at=NOW + timedelta(minutes=2),
        )
    assert description.changed is True
    assert description.registration.revision == 2
    assert description.registration.delivery_config_version == 1
    assert description.registration.target_version == 1
    assert description.registration.secret_version == 1

    async with sqlite_repo.repository.transaction() as tx:
        configured = await tx.patch_registration(
            original.id,
            expected_revision=2,
            patch=RegistrationPatch(
                event_types=("user.updated",),
                timeout_seconds=20,
                active=True,
            ),
            actor_user_id=8,
            at=NOW + timedelta(minutes=3),
        )
    assert configured.registration.revision == 3
    assert configured.registration.delivery_config_version == 2
    assert configured.registration.target_version == 1
    assert configured.registration.secret_version == 1

    async with sqlite_repo.repository.transaction() as tx:
        target_changed = await tx.patch_registration(
            original.id,
            expected_revision=3,
            patch=RegistrationPatch(
                target=RegistrationTarget(
                    protected=_protected("target-replacement"),
                    hostname="receiver.example.net",
                    display="https://receiver.example.net",
                )
            ),
            actor_user_id=8,
            at=NOW + timedelta(minutes=4),
        )
    assert target_changed.registration.revision == 4
    assert target_changed.registration.delivery_config_version == 3
    assert target_changed.registration.target_version == 2
    assert target_changed.registration.secret_version == 1

    async with sqlite_repo.repository.transaction() as tx:
        secret_changed = await tx.patch_registration(
            original.id,
            expected_revision=4,
            patch=RegistrationPatch(
                secret=_protected("secret-replacement"),
                secret_rotation_required=False,
            ),
            actor_user_id=8,
            at=NOW + timedelta(minutes=5),
        )
    assert secret_changed.registration.revision == 5
    assert secret_changed.registration.delivery_config_version == 4
    assert secret_changed.registration.target_version == 2
    assert secret_changed.registration.secret_version == 2

    stored = await sqlite_repo.repository.get_protected_registration(original.id)
    assert stored is not None
    assert stored.target == _protected("target-replacement")
    assert stored.secret == _protected("secret-replacement")


async def test_secret_rotation_requirement_cannot_be_cleared_without_new_secret(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    async with sqlite_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        imported = await tx.insert_registration(
            replace(
                _registration_insert(webhook_id),
                secret_rotation_required=True,
            )
        )
        with pytest.raises(ValueError, match="without a new secret"):
            await tx.patch_registration(
                webhook_id,
                expected_revision=imported.revision,
                patch=RegistrationPatch(secret_rotation_required=False),
                actor_user_id=8,
                at=NOW + timedelta(minutes=1),
            )

        rotated = await tx.patch_registration(
            webhook_id,
            expected_revision=imported.revision,
            patch=RegistrationPatch(
                secret=_protected("rotated-import-secret"),
                secret_rotation_required=False,
            ),
            actor_user_id=8,
            at=NOW + timedelta(minutes=1),
        )
    assert rotated.registration.secret_rotation_required is False
    assert rotated.registration.secret_version == 2


async def test_stale_revision_counts_and_over_limit_state(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    active = await _seed_registration(sqlite_repo, active=True)
    await _seed_registration(sqlite_repo)

    async with sqlite_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as stale:
            await tx.patch_registration(
                active.id,
                expected_revision=99,
                patch=RegistrationPatch(description="stale"),
                actor_user_id=8,
                at=NOW,
            )
    assert stale.value.code is WebhookRepositoryErrorCode.STALE_REVISION

    assert await sqlite_repo.repository.count_registrations() == 2
    assert await sqlite_repo.repository.count_active_registrations() == 1
    assert await sqlite_repo.repository.count_secret_rotation_required() == 0
    async with sqlite_repo.repository.transaction() as tx:
        marked = await tx.patch_registration(
            active.id,
            expected_revision=active.revision,
            patch=RegistrationPatch(secret_rotation_required=True),
            actor_user_id=8,
            at=NOW + timedelta(minutes=1),
        )
    assert marked.registration.secret_rotation_required is True
    assert await sqlite_repo.repository.count_secret_rotation_required() == 1
    state = await sqlite_repo.repository.registration_limit_state(limit=1)
    assert state.current == 2
    assert state.limit == 1
    assert state.at_limit is True
    assert state.over_limit is True
    active_state = await sqlite_repo.repository.active_registration_limit_state(limit=1)
    assert active_state.current == 1
    assert active_state.at_limit is True
    assert active_state.over_limit is False

    async with sqlite_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as full:
            await tx.enforce_registration_limit(limit=2)
    assert full.value.code is WebhookRepositoryErrorCode.REGISTRATION_LIMIT

    async with sqlite_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as active_full:
            await tx.enforce_active_registration_limit(limit=1)
    assert active_full.value.code is WebhookRepositoryErrorCode.ACTIVE_LIMIT


async def test_purge_eligibility_honors_all_blockers(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    deleted_at = NOW - timedelta(days=31)
    eligible = await _seed_registration(sqlite_repo)
    delivery_blocked = await _seed_registration(sqlite_repo)
    idempotency_blocked = await _seed_registration(sqlite_repo)
    migration_blocked = await _seed_registration(sqlite_repo)
    too_new = await _seed_registration(sqlite_repo)

    for item in (eligible, delivery_blocked, idempotency_blocked, migration_blocked):
        async with sqlite_repo.repository.transaction() as tx:
            await tx.soft_delete_registration(
                item.id,
                expected_revision=item.revision,
                actor_user_id=9,
                at=deleted_at,
            )
    async with sqlite_repo.repository.transaction() as tx:
        await tx.soft_delete_registration(
            too_new.id,
            expected_revision=too_new.revision,
            actor_user_id=9,
            at=NOW - timedelta(days=29),
        )

    await sqlite_repo.pool.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, source_command_id,
            source_component, body_ciphertext_json, body_key_id, body_size_bytes
        ) VALUES (?, ?, ?, 'command', ?, ?, ?, ?, ?)
        """,
        "event-purge-blocker",
        "user.created",
        "2026-07-01",
        "command-purge-blocker",
        "authnz",
        '{"ciphertext":"opaque-body"}',
        "key-2026-08",
        12,
    )
    await sqlite_repo.pool.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES (?, ?, ?, 'automatic', 1, 1, 'pending', ?)
        """,
        "delivery-purge-blocker",
        "event-purge-blocker",
        delivery_blocked.id,
        NOW + timedelta(days=1),
    )

    scope, digest, fingerprint = _idempotency_values(route="/admin/webhooks/purge-check")
    scope = build_idempotency_scope(
        actor_id=scope.actor_id,
        operation=scope.operation,
        route=scope.route,
        webhook_id=idempotency_blocked.id,
    )
    digest = idempotency_lookup_digest("fedcba9876543210fedcba9876543210", scope)
    fingerprint = canonical_request_hash(
        "fedcba9876543210fedcba9876543210",
        scope=scope,
        body={},
        conditional_version=1,
    )
    async with sqlite_repo.repository.transaction() as tx:
        await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )

    state = await sqlite_repo.repository.get_migration_state()
    async with sqlite_repo.repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=state.state_revision,
            updates={"source_mapping_json": {"legacy-row": migration_blocked.id}},
            at=NOW,
        )

    assert await sqlite_repo.repository.find_purge_eligible_registration_ids(
        now=NOW,
        limit=100,
    ) == [eligible.id]


async def test_idempotency_new_replay_conflict_in_progress_expiry_and_supersession(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    registration = await _seed_registration(sqlite_repo)
    scope, digest, fingerprint = _idempotency_values()

    async with sqlite_repo.repository.transaction() as tx:
        claimed = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
        assert claimed.kind is IdempotencyLookupKind.NEW
        completed = await tx.complete_idempotency(
            lookup_digest=digest,
            request_fingerprint=fingerprint,
            resource_id=registration.id,
            resource_version=registration.revision,
            secret_version=registration.secret_version,
            replay_secret=_protected("replay-secret"),
            response_status=201,
            response_metadata={"result_kind": "registration"},
            at=NOW,
        )
        assert completed.kind is IdempotencyLookupKind.REPLAY

    async with sqlite_repo.repository.transaction() as tx:
        replay = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW + timedelta(minutes=1),
            expires_at=NOW + timedelta(days=1),
        )
    assert replay.kind is IdempotencyLookupKind.REPLAY
    assert replay.replay_secret == _protected("replay-secret")
    assert replay.resource_superseded is False

    conflicting_fingerprint = canonical_request_hash(
        "0123456789abcdef0123456789abcdef",
        scope=scope,
        body={"description": "Different"},
        conditional_version=None,
    )
    async with sqlite_repo.repository.transaction() as tx:
        conflict = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=conflicting_fingerprint,
            now=NOW + timedelta(minutes=2),
            expires_at=NOW + timedelta(days=1),
        )
    assert conflict.kind is IdempotencyLookupKind.CONFLICT

    pending_scope, pending_digest, pending_fingerprint = _idempotency_values(
        key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        route="/admin/webhooks/pending",
    )
    async with sqlite_repo.repository.transaction() as tx:
        await tx.claim_idempotency(
            lookup_digest=pending_digest,
            scope=pending_scope,
            request_fingerprint=pending_fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(minutes=1),
        )
    async with sqlite_repo.repository.transaction() as tx:
        pending = await tx.claim_idempotency(
            lookup_digest=pending_digest,
            scope=pending_scope,
            request_fingerprint=pending_fingerprint,
            now=NOW + timedelta(seconds=30),
            expires_at=NOW + timedelta(days=1),
        )
    assert pending.kind is IdempotencyLookupKind.IN_PROGRESS

    async with sqlite_repo.repository.transaction() as tx:
        expired_reclaimed = await tx.claim_idempotency(
            lookup_digest=pending_digest,
            scope=pending_scope,
            request_fingerprint=pending_fingerprint,
            now=NOW + timedelta(minutes=2),
            expires_at=NOW + timedelta(days=1),
        )
    assert expired_reclaimed.kind is IdempotencyLookupKind.NEW

    async with sqlite_repo.repository.transaction() as tx:
        await tx.patch_registration(
            registration.id,
            expected_revision=registration.revision,
            patch=RegistrationPatch(description="Superseded"),
            actor_user_id=8,
            at=NOW + timedelta(minutes=3),
        )
    async with sqlite_repo.repository.transaction() as tx:
        still_replayable = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW + timedelta(minutes=4),
            expires_at=NOW + timedelta(days=1),
        )
    assert still_replayable.kind is IdempotencyLookupKind.REPLAY
    assert still_replayable.resource_superseded is False

    async with sqlite_repo.repository.transaction() as tx:
        await tx.patch_registration(
            registration.id,
            expected_revision=registration.revision + 1,
            patch=RegistrationPatch(secret=_protected("newer-secret")),
            actor_user_id=8,
            at=NOW + timedelta(minutes=5),
        )
    async with sqlite_repo.repository.transaction() as tx:
        superseded = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW + timedelta(minutes=6),
            expires_at=NOW + timedelta(days=1),
        )
    assert superseded.kind is IdempotencyLookupKind.REPLAY
    assert superseded.resource_superseded is True


async def test_idempotency_is_route_scoped_and_persists_no_sensitive_inputs(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    raw_key = "raw-key-canary-0123456789abcdef"
    canonical_query = "https://hooks.example.com/private?token=url-query-canary"
    first_scope, first_digest, first_fingerprint = _idempotency_values(
        key=raw_key,
        route="/admin/webhooks",
        body={"url": canonical_query},
    )
    second_scope, second_digest, second_fingerprint = _idempotency_values(
        key=raw_key,
        route="/admin/webhooks/other",
        body={"url": canonical_query},
    )

    async with sqlite_repo.repository.transaction() as tx:
        first = await tx.claim_idempotency(
            lookup_digest=first_digest,
            scope=first_scope,
            request_fingerprint=first_fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
        second = await tx.claim_idempotency(
            lookup_digest=second_digest,
            scope=second_scope,
            request_fingerprint=second_fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
    assert first.kind is second.kind is IdempotencyLookupKind.NEW

    database_bytes = b"".join(
        path.read_bytes() for path in sqlite_repo.path.parent.glob(f"{sqlite_repo.path.name}*") if path.is_file()
    )
    assert raw_key.encode() not in database_bytes
    assert canonical_query.encode() not in database_bytes
    assert b"url-query-canary" not in database_bytes


async def test_migration_state_cas_and_activity_marker_persistence(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    initial = await sqlite_repo.repository.get_migration_state()
    async with sqlite_repo.repository.transaction() as tx:
        updated = await tx.compare_and_set_migration_state(
            expected_revision=initial.state_revision,
            updates={"source_mapping_json": {"system-ops:0": 41}},
            at=NOW,
        )
    assert updated.state_revision == initial.state_revision + 1
    assert updated.source_mapping == {"system-ops:0": 41}

    async with sqlite_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as stale:
            await tx.compare_and_set_migration_state(
                expected_revision=initial.state_revision,
                updates={"source_mapping_json": {}},
                at=NOW,
            )
    assert stale.value.code is WebhookRepositoryErrorCode.STALE_MIGRATION_STATE

    async with sqlite_repo.repository.transaction() as tx:
        first = await tx.mark_first_canonical_activity(
            "registration_mutation",
            NOW,
        )
    async with sqlite_repo.repository.transaction() as tx:
        retained = await tx.mark_first_canonical_activity(
            "event_capture",
            NOW + timedelta(minutes=1),
        )
    assert first.first_canonical_activity_at == NOW
    assert retained.first_canonical_activity_at == NOW
    assert retained.first_canonical_activity_kind == "registration_mutation"


async def test_migration_state_persists_key_rotation_cursor(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    initial = await sqlite_repo.repository.get_migration_state()
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with sqlite_repo.repository.transaction() as tx:
        completed = await tx.compare_and_set_migration_state(
            expected_revision=initial.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 7,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": "fingerprint-key",
                "active_primary_key_id": "source-key",
                "system_ops_webhook_fingerprint": fingerprint,
                "legacy_table_fingerprint": fingerprint,
                "redacted_report_digest": digest,
                "completed_at": NOW,
                "active_report_path": "/srv/tldw/report.json",
                "staging_report_path": "/srv/tldw/report.json.staging",
                "report_owner_id": 1000,
                "report_group_id": 1000,
                "report_mode": 384,
                "report_file_identity": "1048576:42",
            },
            at=NOW,
        )
        rotating = await tx.compare_and_set_migration_state(
            expected_revision=completed.state_revision,
            updates={
                "rotation_operation_id": "rotation-2026-08",
                "rotation_source_key_id": "source-key",
                "rotation_target_key_id": "target-key",
                "rotation_phase": "rewriting",
                "rotation_table_cursor": "admin_webhook_registrations",
                "rotation_key_cursor": "41:target",
                "rotation_processed_count": 41,
                "rotation_started_at": NOW + timedelta(minutes=1),
            },
            at=NOW + timedelta(minutes=1),
        )

    reread = await sqlite_repo.repository.get_migration_state()
    assert reread == rotating
    assert reread.rotation_table_cursor == "admin_webhook_registrations"
    assert reread.rotation_key_cursor == "41:target"
    assert reread.rotation_processed_count == 41


async def test_transaction_rolls_back_after_injected_exception(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    with pytest.raises(TransactionError, match="injected"):
        async with sqlite_repo.repository.transaction() as tx:
            webhook_id = await tx.allocate_registration_id()
            await tx.insert_registration(_registration_insert(webhook_id))
            raise RuntimeError("injected")

    assert await sqlite_repo.repository.count_registrations() == 0
    async with sqlite_repo.repository.transaction() as tx:
        assert await tx.allocate_registration_id() == 1


async def test_invalid_migration_activity_kind_is_rejected(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    async with sqlite_repo.repository.transaction() as tx:
        with pytest.raises(ValueError, match="activity kind"):
            await tx.mark_first_canonical_activity("arbitrary", NOW)


async def test_database_contains_only_redacted_registration_target(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    registration = await _seed_registration(sqlite_repo)
    with sqlite3.connect(sqlite_repo.path) as conn:
        row = conn.execute(
            """
            SELECT target_ciphertext_json, target_hostname, target_display,
                   secret_ciphertext_json
            FROM admin_webhook_registrations WHERE id = ?
            """,
            (registration.id,),
        ).fetchone()
    assert row == (
        _protected(f"target-{registration.id}").ciphertext_json,
        "hooks.example.com",
        "https://hooks.example.com",
        _protected(f"secret-{registration.id}").ciphertext_json,
    )


async def test_repository_rejects_unredacted_target_metadata(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    async with sqlite_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        registration = _registration_insert(webhook_id)
        with pytest.raises(ValueError, match="redacted origin"):
            await tx.insert_registration(
                replace(
                    registration,
                    target=replace(
                        registration.target,
                        display="https://hooks.example.com/private?token=canary",
                    ),
                )
            )

    assert await sqlite_repo.repository.count_registrations() == 0
