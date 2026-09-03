from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks import control_plane
from tldw_Server_API.app.core.Admin_Webhooks.audit import MutationAudit
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    AdminWebhookControlPlane,
    CreateRegistrationCommand,
    UnavailableDeliveryCapability,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    ProtectedValue,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    ValidatedWebhookTarget,
    WebhookError,
    WebhookErrorCode,
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseLockError, TransactionError
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    DATABASE_PROTECTED_TABLE_ORDER,
    AdminWebhookRepository,
    IdempotencyLookupKind,
    RegistrationInsert,
    RegistrationPatch,
    RegistrationTarget,
    WebhookRepositoryError,
    WebhookRepositoryErrorCode,
)

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)
pytestmark = pytest.mark.postgres

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@dataclass
class PostgreSQLRepositoryFixture:
    repository: AdminWebhookRepository
    pool: object


async def _execute_test_ddl(test_db_pool, query: str) -> None:
    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(query)
    finally:
        await connection.close()


@pytest_asyncio.fixture
async def pg_repo(test_db_pool) -> PostgreSQLRepositoryFixture:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _execute_test_ddl(
        test_db_pool,
        """
        TRUNCATE TABLE
            admin_webhook_delivery_attempts,
            admin_webhook_deliveries,
            admin_webhook_events,
            admin_webhook_idempotency,
            admin_webhook_registrations,
            admin_webhook_sequences,
            admin_webhook_migration_state
        RESTART IDENTITY CASCADE
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_sequences (name, next_value)
        VALUES ('registration', 1)
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_migration_state (
            singleton_id, schema_version, phase
        ) VALUES (1, 1, 'migration_pending')
        """
    )
    yield PostgreSQLRepositoryFixture(
        repository=AdminWebhookRepository(test_db_pool),
        pool=test_db_pool,
    )


def _protected(
    label: str,
    *,
    key_id: str = "key-2026-08",
) -> ProtectedValue:
    return ProtectedValue(
        ciphertext_json=f'{{"ciphertext":"opaque-{label}"}}',
        key_id=key_id,
    )


async def _complete_migration(
    repository: AdminWebhookRepository,
    *,
    primary_key_id: str = "key-2026-08",
) -> None:
    state = await repository.get_migration_state()
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=state.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 7,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": primary_key_id,
                "active_primary_key_id": primary_key_id,
                "system_ops_webhook_fingerprint": fingerprint,
                "legacy_table_fingerprint": fingerprint,
                "redacted_report_digest": digest,
                "completed_at": NOW,
                "active_report_path": "/srv/tldw/webhook-report.json",
                "staging_report_path": "/srv/tldw/webhook-report.json.staging",
                "report_owner_id": 1000,
                "report_group_id": 1000,
                "report_mode": 384,
                "report_file_identity": "1048576:41",
            },
            at=NOW,
        )


def _registration_insert(webhook_id: int, *, now: datetime = NOW) -> RegistrationInsert:
    return RegistrationInsert(
        id=webhook_id,
        description="PostgreSQL",
        target=RegistrationTarget(
            protected=_protected(f"target-{webhook_id}"),
            hostname="hooks.example.com",
            display="https://hooks.example.com",
        ),
        event_types=("user.created",),
        active=False,
        timeout_seconds=10,
        secret=_protected(f"secret-{webhook_id}"),
        secret_rotation_required=False,
        actor_user_id=7,
        now=now,
    )


def _idempotency(
    key: str,
    *,
    body: dict[str, object],
) -> tuple[object, str, str]:
    scope = build_idempotency_scope(
        actor_id=7,
        operation="create",
        route="/admin/webhooks",
    )
    return (
        scope,
        idempotency_lookup_digest(key, scope),
        canonical_request_hash(
            key,
            scope=scope,
            body=body,
            conditional_version=None,
        ),
    )


@pytest.mark.integration
async def test_postgres_commit_sequence_and_readback(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async with pg_repo.repository.transaction() as tx:
        first_id = await tx.allocate_registration_id()
        created = await tx.insert_registration(_registration_insert(first_id))

    assert await pg_repo.repository.get_registration(first_id) == created

    async def allocate() -> int:
        async with pg_repo.repository.transaction() as tx:
            return await tx.allocate_registration_id()

    allocated = await asyncio.gather(*(allocate() for _ in range(10)))
    assert sorted(allocated) == list(range(2, 12))


@pytest.mark.integration
async def test_postgres_registration_list_supports_bounded_offset(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async with pg_repo.repository.transaction() as tx:
        for expected_id in range(1, 4):
            webhook_id = await tx.allocate_registration_id()
            assert webhook_id == expected_id
            await tx.insert_registration(_registration_insert(webhook_id))

    first_page = await pg_repo.repository.list_registrations(limit=2)
    offset_page = await pg_repo.repository.list_registrations(limit=2, offset=1)

    assert [item.id for item in first_page] == [3, 2]
    assert [item.id for item in offset_page] == [2, 1]
    with pytest.raises(ValueError, match="offset must be between 0 and 1000"):
        await pg_repo.repository.list_registrations(limit=2, offset=1_001)


@pytest.mark.integration
async def test_postgres_counts_secret_rotation_required(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async with pg_repo.repository.transaction() as tx:
        first_id = await tx.allocate_registration_id()
        first = await tx.insert_registration(_registration_insert(first_id))
        second_id = await tx.allocate_registration_id()
        await tx.insert_registration(_registration_insert(second_id))
        marked = await tx.patch_registration(
            first.id,
            expected_revision=first.revision,
            patch=RegistrationPatch(secret_rotation_required=True),
            actor_user_id=8,
            at=NOW + timedelta(minutes=1),
        )

    assert marked.registration.secret_rotation_required is True
    assert await pg_repo.repository.count_secret_rotation_required() == 1
    async with pg_repo.repository.transaction() as tx:
        await tx.soft_delete_registration(
            first.id,
            expected_revision=marked.registration.revision,
            actor_user_id=8,
            at=NOW + timedelta(minutes=2),
        )
    assert await pg_repo.repository.count_secret_rotation_required() == 0


@pytest.mark.integration
async def test_postgres_pages_and_replaces_every_protected_inventory(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async with pg_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        await tx.insert_registration(_registration_insert(webhook_id))

    await pg_repo.pool.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, source_command_id,
            source_component, body_ciphertext_json, body_key_id, body_size_bytes,
            created_at
        ) VALUES (?, ?, ?, 'command', ?, ?, ?, ?, ?, ?)
        """,
        "event-protected-1",
        "incident.created",
        "2026-07-01",
        "command-protected-1",
        "admin",
        _protected("event-body").ciphertext_json,
        "key-2026-08",
        12,
        NOW,
    )
    scope, digest, fingerprint = _idempotency(
        "0123456789abcdef0123456789abcdef",
        body={"description": "protected inventory"},
    )
    async with pg_repo.repository.transaction() as tx:
        await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW,
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        await tx.complete_idempotency(
            lookup_digest=digest,
            request_fingerprint=fingerprint,
            resource_id=webhook_id,
            resource_version=1,
            secret_version=1,
            replay_secret=_protected("replay-secret"),
            response_status=201,
            response_metadata={"result_kind": "created"},
            at=NOW,
        )

    expected_fields = {
        "registration_targets": "target",
        "registration_secrets": "secret",
        "event_bodies": "body",
        "idempotency_replay_secrets": "replay_secret",
    }
    for table in DATABASE_PROTECTED_TABLE_ORDER:
        rows = await pg_repo.repository.page_protected_rows(
            table=table,
            after=None,
            limit=1,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.field == expected_fields[table]
        replacement = _protected(f"rotated-{table}", key_id="key-2026-09")
        async with pg_repo.repository.transaction() as tx:
            assert await tx.replace_protected_value(
                row,
                expected_ciphertext=row.protected.ciphertext_json,
                replacement=replacement,
            )
        readback = await pg_repo.repository.page_protected_rows(
            table=table,
            after=None,
            limit=1,
        )
        assert readback[0].protected == replacement
        assert (
            await pg_repo.repository.page_protected_rows(
                table=table,
                after=readback[0].row_identity,
                limit=1,
            )
            == []
        )

    expired_at = datetime.now(timezone.utc) - timedelta(days=1)
    await pg_repo.pool.execute(
        "UPDATE admin_webhook_idempotency SET expires_at = ?",
        expired_at,
    )
    assert (
        await pg_repo.repository.page_protected_rows(
            table="idempotency_replay_secrets",
            after=None,
            limit=1,
        )
        == []
    )
    snapshot_rows = await pg_repo.repository.page_protected_rows(
        table="idempotency_replay_secrets",
        after=None,
        limit=1,
        inventory_at=expired_at - timedelta(minutes=1),
    )
    assert len(snapshot_rows) == 1
    snapshot_row = snapshot_rows[0]
    async with pg_repo.repository.transaction() as tx:
        assert await tx.replace_protected_value(
            snapshot_row,
            expected_ciphertext=snapshot_row.protected.ciphertext_json,
            replacement=_protected("snapshot-replay", key_id="key-2026-10"),
        )


@pytest.mark.integration
async def test_postgres_protected_rewrite_and_rotation_cursor_are_atomic(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    await _complete_migration(pg_repo.repository)
    async with pg_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        await tx.insert_registration(_registration_insert(webhook_id))
        initial = await tx.lock_migration_state()
        started = await tx.compare_and_set_migration_state(
            expected_revision=initial.state_revision,
            updates={
                "rotation_operation_id": "rotation-postgres-1",
                "rotation_source_key_id": "key-2026-08",
                "rotation_target_key_id": "key-2026-09",
                "rotation_phase": "rewriting",
                "rotation_table_cursor": "registration_targets",
                "rotation_started_at": NOW,
            },
            at=NOW,
        )
    row = (
        await pg_repo.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    replacement = _protected("atomic-target", key_id="key-2026-09")

    with pytest.raises(TransactionError, match="PostgreSQL transaction"):
        async with pg_repo.repository.transaction() as tx:
            assert await tx.replace_protected_value(
                row,
                expected_ciphertext=row.protected.ciphertext_json,
                replacement=replacement,
            )
            await tx.compare_and_set_migration_state(
                expected_revision=started.state_revision,
                updates={
                    "rotation_key_cursor": row.row_identity,
                    "rotation_processed_count": 1,
                },
                at=NOW + timedelta(minutes=1),
            )
            raise RuntimeError("rollback protected batch")

    rolled_back_row = (
        await pg_repo.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    rolled_back_state = await pg_repo.repository.get_migration_state()
    assert rolled_back_row.protected == row.protected
    assert rolled_back_state.rotation_key_cursor is None
    assert rolled_back_state.rotation_processed_count == 0

    async with pg_repo.repository.transaction() as tx:
        assert await tx.replace_protected_value(
            row,
            expected_ciphertext=row.protected.ciphertext_json,
            replacement=replacement,
        )
        committed = await tx.compare_and_set_migration_state(
            expected_revision=rolled_back_state.state_revision,
            updates={
                "rotation_key_cursor": row.row_identity,
                "rotation_processed_count": 1,
            },
            at=NOW + timedelta(minutes=2),
        )
    assert committed.rotation_key_cursor == row.row_identity
    assert committed.rotation_processed_count == 1

    async with pg_repo.repository.transaction() as tx:
        assert (
            await tx.replace_protected_value(
                row,
                expected_ciphertext=row.protected.ciphertext_json,
                replacement=_protected("stale", key_id="key-2026-10"),
            )
            is False
        )


@pytest.mark.integration
async def test_postgres_revision_noop_versions_soft_delete_and_limits(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async with pg_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        original = await tx.insert_registration(_registration_insert(webhook_id))
        no_op = await tx.patch_registration(
            webhook_id,
            expected_revision=1,
            patch=RegistrationPatch(description=original.description),
            actor_user_id=8,
            at=NOW + timedelta(minutes=1),
        )
    assert no_op.changed is False
    assert no_op.registration.revision == 1

    async with pg_repo.repository.transaction() as tx:
        changed = await tx.patch_registration(
            webhook_id,
            expected_revision=1,
            patch=RegistrationPatch(
                event_types=("user.updated",),
                target=RegistrationTarget(
                    protected=_protected("changed-target"),
                    hostname="changed.example.net",
                    display="https://changed.example.net",
                ),
            ),
            actor_user_id=8,
            at=NOW + timedelta(minutes=2),
        )
    assert changed.registration.revision == 2
    assert changed.registration.delivery_config_version == 2
    assert changed.registration.target_version == 2
    assert changed.registration.secret_version == 1

    async with pg_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as stale:
            await tx.patch_registration(
                webhook_id,
                expected_revision=1,
                patch=RegistrationPatch(description="stale"),
                actor_user_id=8,
                at=NOW,
            )
    assert stale.value.code is WebhookRepositoryErrorCode.STALE_REVISION

    async with pg_repo.repository.transaction() as tx:
        deleted = await tx.soft_delete_registration(
            webhook_id,
            expected_revision=2,
            actor_user_id=9,
            at=NOW + timedelta(minutes=3),
        )
    assert deleted.deleted_at == NOW + timedelta(minutes=3)
    assert await pg_repo.repository.get_registration(webhook_id) is None
    assert await pg_repo.repository.count_registrations() == 0
    assert (await pg_repo.repository.registration_limit_state(limit=1)).current == 0
    snapshot = await pg_repo.repository.get_legacy_import_snapshot()
    assert snapshot.canonical_registration_ids == (webhook_id,)
    assert snapshot.canonical_non_deleted_count == 0


@pytest.mark.integration
async def test_fail_once_audit_error_is_preserved_across_postgres_transaction(
    pg_repo: PostgreSQLRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _complete_migration(pg_repo.repository, primary_key_id="primary")
    monkeypatch.setattr(
        control_plane,
        "validate_webhook_target",
        lambda url, *, allow_http_dev, allow_e2e_loopback: ValidatedWebhookTarget(
            url=url,
            hostname="hooks.example.com",
            target_display="https://hooks.example.com",
        ),
    )
    ring = WebhookKeyRing(
        {"primary": base64.b64encode(b"p" * 32).decode("ascii")},
        primary_id="primary",
    )
    service = AdminWebhookControlPlane(
        repository=pg_repo.repository,
        settings=AdminWebhookSettings(
            mode=AdminWebhookMode.ON,
            registration_limit=100,
            active_limit=25,
            allow_http_dev=False,
            idempotency_ttl_seconds=86_400,
            rollback_window_days=7,
        ),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    calls = 0

    async def fail_once(_record: MutationAudit) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MandatoryAuditWriteError("audit unavailable") from DatabaseLockError()

    with pytest.raises(WebhookError) as exc_info:
        await service.create(
            CreateRegistrationCommand(
                actor_id=7,
                idempotency_key="0123456789abcdef0123456789abcdef",
                url="https://hooks.example.com/private",
                event_types=("user.created",),
                description="PostgreSQL audit boundary",
                timeout_seconds=10,
                request_id="postgres-audit-fail-once",
                now=NOW,
            ),
            audit_sink=fail_once,
        )

    assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert calls == 1
    assert await pg_repo.repository.count_registrations() == 0


async def _run_idempotent_create(
    fixture: PostgreSQLRepositoryFixture,
    *,
    scope,
    digest: str,
    fingerprint: str,
) -> IdempotencyLookupKind:
    async with fixture.repository.transaction() as tx:
        lookup = await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW,
            expires_at=NOW + timedelta(days=1),
        )
        if lookup.kind is not IdempotencyLookupKind.NEW:
            return lookup.kind
        webhook_id = await tx.allocate_registration_id()
        registration = await tx.insert_registration(_registration_insert(webhook_id))
        await asyncio.sleep(0.05)
        await tx.complete_idempotency(
            lookup_digest=digest,
            request_fingerprint=fingerprint,
            resource_id=registration.id,
            resource_version=registration.revision,
            secret_version=registration.secret_version,
            replay_secret=_protected(f"replay-{webhook_id}"),
            response_status=201,
            response_metadata={"result_kind": "registration"},
            at=NOW,
        )
        return IdempotencyLookupKind.NEW


@pytest.mark.integration
async def test_postgres_identical_idempotency_race_has_one_winner_and_replay(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    scope, digest, fingerprint = _idempotency(
        "11111111111111111111111111111111",
        body={"description": "same"},
    )
    outcomes = await asyncio.gather(
        _run_idempotent_create(
            pg_repo,
            scope=scope,
            digest=digest,
            fingerprint=fingerprint,
        ),
        _run_idempotent_create(
            pg_repo,
            scope=scope,
            digest=digest,
            fingerprint=fingerprint,
        ),
    )
    assert sorted(item.value for item in outcomes) == ["new", "replay"]
    assert await pg_repo.repository.count_registrations() == 1


@pytest.mark.integration
async def test_postgres_registration_admission_is_atomic_under_concurrency(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    async def create_if_capacity() -> int:
        async with pg_repo.repository.transaction() as tx:
            await tx.enforce_registration_limit(limit=1)
            webhook_id = await tx.allocate_registration_id()
            await tx.insert_registration(_registration_insert(webhook_id))
            await asyncio.sleep(0.05)
            return webhook_id

    outcomes = await asyncio.gather(
        create_if_capacity(),
        create_if_capacity(),
        return_exceptions=True,
    )
    created = [item for item in outcomes if isinstance(item, int)]
    rejected = [item for item in outcomes if isinstance(item, WebhookRepositoryError)]
    assert len(created) == 1
    assert len(rejected) == 1
    assert rejected[0].code is WebhookRepositoryErrorCode.REGISTRATION_LIMIT
    assert await pg_repo.repository.count_registrations() == 1


@pytest.mark.integration
async def test_postgres_conflicting_idempotency_race_has_one_winner_and_conflict(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    key = "22222222222222222222222222222222"
    scope, digest, first_fingerprint = _idempotency(
        key,
        body={"description": "first"},
    )
    _, _, second_fingerprint = _idempotency(
        key,
        body={"description": "second"},
    )
    outcomes = await asyncio.gather(
        _run_idempotent_create(
            pg_repo,
            scope=scope,
            digest=digest,
            fingerprint=first_fingerprint,
        ),
        _run_idempotent_create(
            pg_repo,
            scope=scope,
            digest=digest,
            fingerprint=second_fingerprint,
        ),
    )
    assert sorted(item.value for item in outcomes) == ["conflict", "new"]
    assert await pg_repo.repository.count_registrations() == 1


@pytest.mark.integration
async def test_postgres_migration_and_activity_compare_and_set(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    initial = await pg_repo.repository.get_migration_state()
    async with pg_repo.repository.transaction() as tx:
        updated = await tx.compare_and_set_migration_state(
            expected_revision=initial.state_revision,
            updates={"source_mapping_json": {"legacy": 17}},
            at=NOW,
        )
        marked = await tx.mark_first_canonical_activity("event_capture", NOW)
    assert updated.state_revision == 2
    assert marked.state_revision == 3
    assert marked.first_canonical_activity_kind == "event_capture"

    async with pg_repo.repository.transaction() as tx:
        with pytest.raises(WebhookRepositoryError) as stale:
            await tx.compare_and_set_migration_state(
                expected_revision=1,
                updates={"source_mapping_json": {}},
                at=NOW,
            )
    assert stale.value.code is WebhookRepositoryErrorCode.STALE_MIGRATION_STATE


@pytest.mark.integration
async def test_postgres_transaction_rolls_back_resource_and_sequence(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    with pytest.raises(TransactionError, match="PostgreSQL transaction"):
        async with pg_repo.repository.transaction() as tx:
            webhook_id = await tx.allocate_registration_id()
            await tx.insert_registration(_registration_insert(webhook_id))
            raise RuntimeError("injected")

    assert await pg_repo.repository.count_registrations() == 0
    async with pg_repo.repository.transaction() as tx:
        assert await tx.allocate_registration_id() == 1


@pytest.mark.integration
async def test_postgres_lock_timeout_maps_to_database_busy_without_claim(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    raw_pool = pg_repo.pool.pool
    assert raw_pool is not None
    timed_repository = AdminWebhookRepository(
        pg_repo.pool,
        postgres_lock_timeout_ms=50,
        postgres_statement_timeout_ms=500,
    )
    async with raw_pool.acquire() as blocker, blocker.transaction():
        await blocker.execute(
            """
            UPDATE admin_webhook_sequences
            SET next_value = next_value
            WHERE name = 'registration'
            """
        )
        with pytest.raises(WebhookRepositoryError) as busy:
            async with timed_repository.transaction() as tx:
                await tx.allocate_registration_id()
    assert busy.value.code is WebhookRepositoryErrorCode.DATABASE_BUSY
    assert await pg_repo.pool.fetchval("SELECT COUNT(*) FROM admin_webhook_idempotency") == 0


@pytest.mark.integration
async def test_postgres_statement_timeout_maps_to_database_busy(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    timed_repository = AdminWebhookRepository(
        pg_repo.pool,
        postgres_lock_timeout_ms=500,
        postgres_statement_timeout_ms=50,
    )
    with pytest.raises(WebhookRepositoryError) as busy:
        async with timed_repository.transaction() as tx:
            await tx._fetchrow("SELECT pg_sleep(?) AS slept", (0.2,))
    assert busy.value.code is WebhookRepositoryErrorCode.DATABASE_BUSY
