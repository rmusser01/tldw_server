from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)
pytestmark = [pytest.mark.postgres, pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@dataclass
class PostgreSQLRepositoryFixture:
    repository: AdminWebhookRepository
    pool: object


@pytest_asyncio.fixture
async def pg_repo(test_db_pool) -> PostgreSQLRepositoryFixture:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await test_db_pool.execute(
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


def _protected(label: str) -> ProtectedValue:
    return ProtectedValue(
        ciphertext_json=f'{{"ciphertext":"opaque-{label}"}}',
        key_id="key-2026-08",
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


async def test_postgres_transaction_rolls_back_resource_and_sequence(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    with pytest.raises(TransactionError, match="injected"):
        async with pg_repo.repository.transaction() as tx:
            webhook_id = await tx.allocate_registration_id()
            await tx.insert_registration(_registration_insert(webhook_id))
            raise RuntimeError("injected")

    assert await pg_repo.repository.count_registrations() == 0
    async with pg_repo.repository.transaction() as tx:
        assert await tx.allocate_registration_id() == 1


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
