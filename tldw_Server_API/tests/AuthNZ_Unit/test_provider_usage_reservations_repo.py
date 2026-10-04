"""Real-backend reservation contracts, also reused by PostgreSQL parity tests."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import shutil
import sqlite3
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.repos import _dual_backend

MAX_INT = 2**63 - 1
MODULE = "tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo"
pytestmark = pytest.mark.unit


def api():
    assert importlib.util.find_spec(MODULE) is not None, "reservation repository missing"
    return importlib.import_module(MODULE)


async def execute(pool, conn, sql, *args):
    return await _dual_backend.execute(conn, pool.pool is not None, sql, args)


async def rows(pool, conn, sql, *args):
    return await _dual_backend.fetch_all(conn, pool.pool is not None, sql, args, ())


def reservation(**overrides):
    mod = api()
    fields = {
        "execution_id": str(uuid4()),
        "user_id": 1,
        "active_team_id": None,
        "active_organization_id": None,
        "billing_scope": mod.BillingScope("user", 1),
        "provider": "openai",
        "model": "fixed-model",
        "reserved_input_tokens": 10,
        "reserved_output_tokens": 20,
        "reserved_cost_units": 50,
    }
    fields.update(overrides)
    return mod.ProviderUsageReservation(**fields)


async def unlimited(_conn):
    return api().ReservationQuotaSnapshot(0, None, 0, None)


def writer(pool, *, fail=False, entered=None, release=None):
    async def write(conn, item, actuals):
        await execute(
            pool,
            conn,
            """INSERT INTO llm_usage_log
               (user_id, operation, provider, model, request_id, billing_org_id,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd)
               VALUES (?, 'mcp_model_completion', ?, ?, ?, ?, ?, ?, ?, ?)""",
            item.user_id,
            item.provider,
            item.model,
            item.execution_id,
            item.active_organization_id,
            actuals.input_tokens,
            actuals.output_tokens,
            actuals.input_tokens + actuals.output_tokens,
            actuals.cost_units / 1_000_000_000,
        )
        if entered is not None:
            entered.set()
            await release.wait()
        if fail:
            raise RuntimeError("credential-secret prompt-secret")

    return write


@pytest.fixture(scope="session")
def reservation_schema(tmp_path_factory):
    path = tmp_path_factory.mktemp("provider-reservations") / "schema.db"
    migrations.apply_authnz_migrations(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash) "
            "VALUES (1, 'reservation-user', 'reservation@example.com', 'hash')"
        )
    return path


@pytest_asyncio.fixture
async def reservation_pool(tmp_path, reservation_schema):
    path = tmp_path / "reservations.db"
    shutil.copyfile(reservation_schema, path)
    pool = DatabasePool()
    pool.db_path = str(path)
    pool._sqlite_uri = False
    pool._initialized = True
    yield pool
    await pool.close()


def test_migration_100_registered():
    assert migrations.get_authnz_migrations()[-1].version == 100


def test_published_user_usage_index_precedes_mcp_reservation_upgrade(tmp_path):
    path = tmp_path / "published-099.db"
    migrations.apply_authnz_migrations(path, target_version=99)
    with sqlite3.connect(path) as conn:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(llm_usage_log)")}
        assert "idx_llm_usage_log_user_ts" in indexes
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='provider_usage_reservations'"
            ).fetchone()
            is None
        )
    migrations.apply_authnz_migrations(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 100
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='provider_usage_reservations'"
            ).fetchone()[0]
            == "provider_usage_reservations"
        )


def test_repository_contract_exists():
    assert api().ProviderUsageReservationsRepo


def test_migration_additive_idempotent_and_minimal(tmp_path):
    path = tmp_path / "legacy.db"
    migrations.apply_authnz_migrations(path, target_version=98)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO llm_usage_log (operation, request_id) VALUES ('chat', ?)",
            [("legacy-id",), ("legacy-id",)],
        )
    migrations.apply_authnz_migrations(path)
    with sqlite3.connect(path) as conn:
        migrations.migration_100_create_provider_usage_reservations(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(provider_usage_reservations)")}
        assert columns == {
            "execution_id",
            "user_id",
            "active_team_id",
            "active_organization_id",
            "billing_scope_type",
            "billing_scope_id",
            "provider",
            "model",
            "reserved_input_tokens",
            "reserved_output_tokens",
            "reserved_cost_units",
            "actual_input_tokens",
            "actual_output_tokens",
            "actual_cost_units",
            "state",
            "created_at",
            "updated_at",
            "dispatched_at",
            "resolved_at",
        }
        assert conn.execute("SELECT COUNT(*) FROM llm_usage_log").fetchone()[0] == 2
        assert "billing_org_id" in {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)")}


def test_migration_fails_for_duplicate_canonical_execution_ids(tmp_path):
    path = tmp_path / "duplicate.db"
    migrations.apply_authnz_migrations(path, target_version=98)
    with sqlite3.connect(path) as conn:
        conn.executemany(
            "INSERT INTO llm_usage_log (operation, request_id) " "VALUES ('mcp_model_completion', ?)",
            [("duplicate",), ("duplicate",)],
        )
    with pytest.raises(sqlite3.IntegrityError):
        migrations.apply_authnz_migrations(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 99
        assert "billing_org_id" not in {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)")}


class ReservationContract:
    """The same state, arithmetic, rollback and race assertions on both backends."""

    @pytest.mark.asyncio
    async def test_committed_restart_and_scope_totals(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        assert (await repo.reserve(item, snapshot_reader=unlimited))["state"] == "reserved"
        was_postgres, path = pool.pool is not None, pool.db_path
        await pool.close()
        reopened = DatabasePool()
        if was_postgres:
            await reopened.initialize()
        else:
            reopened.db_path = path
            reopened._sqlite_uri = False
            reopened._initialized = True
        try:
            restarted = mod.ProviderUsageReservationsRepo(reopened)
            assert (await restarted.get(item.execution_id))["provider"] == "openai"
            assert await restarted.outstanding(item.billing_scope) == {
                "tokens": 30,
                "cost_units": 50,
            }
            assert await restarted.get(str(uuid4())) is None
        finally:
            await reopened.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kind,team,org,value",
        [
            ("user", None, None, 1),
            ("team", 5, None, 5),
            ("org", 5, 7, 7),
        ],
    )
    async def test_explicit_billing_scope(self, reservation_pool, kind, team, org, value):
        mod = api()
        item = reservation(active_team_id=team, active_organization_id=org, billing_scope=mod.BillingScope(kind, value))
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        await repo.reserve(item, snapshot_reader=unlimited)
        assert (await repo.outstanding(item.billing_scope))["tokens"] == 30
        assert (await repo.outstanding(mod.BillingScope("user", 99)))["tokens"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_admission_rejected(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(item, snapshot_reader=unlimited)
        assert error.value.code == "duplicate_execution"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "field",
        [
            "reserved_input_tokens",
            "reserved_output_tokens",
            "reserved_cost_units",
            "user_id",
            "active_team_id",
            "active_organization_id",
        ],
    )
    @pytest.mark.parametrize("value", [True, -1, 1.0, MAX_INT + 1])
    async def test_exact_integer_validation(self, reservation_pool, field, value):
        mod = api()
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            item = reservation(**{field: value})
            await mod.ProviderUsageReservationsRepo(reservation_pool).reserve(
                item,
                snapshot_reader=unlimited,
            )
        assert error.value.code in {"invalid_reservation", "integer_overflow"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "execution_id",
        [
            "caller-id",
            str(uuid4()).upper(),
            "00000000-0000-1000-8000-000000000000",
            "00000000-0000-4000-0000-000000000000",
        ],
    )
    async def test_execution_id_is_exact_uuid4(self, reservation_pool, execution_id):
        mod = api()
        with pytest.raises(mod.ProviderUsageReservationError):
            await mod.ProviderUsageReservationsRepo(reservation_pool).reserve(
                reservation(execution_id=execution_id),
                snapshot_reader=unlimited,
            )

    @pytest.mark.asyncio
    async def test_mismatched_scope_rejected(self, reservation_pool):
        mod = api()
        with pytest.raises(mod.ProviderUsageReservationError):
            await mod.ProviderUsageReservationsRepo(reservation_pool).reserve(
                reservation(active_organization_id=7),
                snapshot_reader=unlimited,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("snapshot", [(0, 29, 0, None), (0, None, 1, 50), (1, 30, 0, None)])
    async def test_quota_denial(self, reservation_pool, snapshot):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()

        async def read(_conn):
            return mod.ReservationQuotaSnapshot(*snapshot)

        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(item, snapshot_reader=read)
        assert error.value.code == "quota_exceeded"
        assert await repo.get(item.execution_id) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["reserved", "dispatched", "ambiguous"])
    async def test_all_unresolved_rows_carry_forward(self, reservation_pool, state):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        if state != "reserved":
            await repo.mark_dispatched(item.execution_id)
        if state == "ambiguous":
            await repo.retain_ambiguous(item.execution_id)
        async with pool.transaction() as conn:
            await execute(
                pool, conn, "UPDATE provider_usage_reservations " "SET created_at = '2000-01-01T00:00:00+00:00'"
            )

        async def read(_conn):
            return mod.ReservationQuotaSnapshot(0, 59, 0, None)

        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(reservation(), snapshot_reader=read)
        assert error.value.code == "quota_exceeded"

    @pytest.mark.asyncio
    async def test_snapshot_failure_sanitized_and_rollback(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()

        async def fail(_conn):
            raise RuntimeError("credential-secret prompt-secret")

        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(item, snapshot_reader=fail)
        assert error.value.code == "snapshot_unavailable"
        assert error.value.__context__ is None and error.value.__cause__ is None
        assert "secret" not in str(error.value)
        assert await repo.get(item.execution_id) is None

    @pytest.mark.asyncio
    async def test_cancellation_preserved(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()

        async def cancel(_conn):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await repo.reserve(item, snapshot_reader=cancel)
        assert await repo.get(item.execution_id) is None

    @pytest.mark.asyncio
    async def test_release_idempotent_and_terminal(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        first = await repo.release_before_dispatch(item.execution_id)
        assert first == await repo.release_before_dispatch(item.execution_id)
        assert first["state"] == "released"
        assert (await repo.outstanding(item.billing_scope))["tokens"] == 0
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.mark_dispatched(item.execution_id)

    @pytest.mark.asyncio
    async def test_dispatched_and_ambiguous_idempotent(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        first = await repo.mark_dispatched(item.execution_id)
        assert first == await repo.mark_dispatched(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.release_before_dispatch(item.execution_id)
        ambiguous = await repo.retain_ambiguous(item.execution_id)
        assert ambiguous == await repo.retain_ambiguous(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.mark_dispatched(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.reconcile(
                item.execution_id, mod.ReservationActuals(1, 1, 1), usage_writer=writer(reservation_pool)
            )

    @pytest.mark.asyncio
    async def test_reconcile_atomic_idempotent_and_terminal(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        actuals = mod.ReservationActuals(2, 3, 4)
        first = await repo.reconcile(item.execution_id, actuals, usage_writer=writer(pool))
        assert first == await repo.reconcile(item.execution_id, actuals, usage_writer=writer(pool))
        assert first["actual_cost_units"] == 4 and first["state"] == "reconciled"
        assert (await repo.outstanding(item.billing_scope))["tokens"] == 0
        async with pool.transaction() as conn:
            assert len(await rows(pool, conn, "SELECT * FROM llm_usage_log")) == 1
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reconcile(item.execution_id, mod.ReservationActuals(2, 3, 5), usage_writer=writer(pool))
        assert error.value.code == "conflicting_replay"
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.retain_ambiguous(item.execution_id)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "actuals", [(11, 0, 0), (0, 21, 0), (0, 0, 51), (True, 0, 0), (-1, 0, 0), (0, 0, MAX_INT + 1)]
    )
    async def test_actuals_individual_ceiling(self, reservation_pool, actuals):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError):
            await repo.reconcile(
                item.execution_id, mod.ReservationActuals(*actuals), usage_writer=writer(reservation_pool)
            )
        assert (await repo.get(item.execution_id))["state"] == "dispatched"

    @pytest.mark.asyncio
    async def test_usage_writer_failure_rolls_back_both(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reconcile(
                item.execution_id, mod.ReservationActuals(1, 1, 1), usage_writer=writer(pool, fail=True)
            )
        assert error.value.code == "usage_write_failed"
        assert error.value.__context__ is None and error.value.__cause__ is None
        assert (await repo.get(item.execution_id))["state"] == "dispatched"
        async with pool.transaction() as conn:
            assert not await rows(pool, conn, "SELECT * FROM llm_usage_log")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["mark_dispatched", "release_before_dispatch", "retain_ambiguous"])
    async def test_missing_transitions_reject(self, reservation_pool, method):
        mod = api()
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await getattr(mod.ProviderUsageReservationsRepo(reservation_pool), method)(str(uuid4()))
        assert error.value.code == "reservation_not_found"

    @pytest.mark.asyncio
    async def test_overflow_checked_even_unlimited(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        await repo.reserve(
            reservation(reserved_input_tokens=MAX_INT, reserved_output_tokens=0), snapshot_reader=unlimited
        )
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(reservation(), snapshot_reader=unlimited)
        assert error.value.code == "integer_overflow"

    @pytest.mark.asyncio
    async def test_concurrent_admission_one_winner(self, reservation_pool):
        mod = api()

        async def read(_conn):
            await asyncio.sleep(0.02)
            return mod.ReservationQuotaSnapshot(0, 30, 0, 50)

        async def admit():
            return await mod.ProviderUsageReservationsRepo(reservation_pool).reserve(
                reservation(),
                snapshot_reader=read,
            )

        results = await asyncio.gather(admit(), admit(), return_exceptions=True)
        assert sum(isinstance(result, dict) for result in results) == 1
        errors = [result for result in results if isinstance(result, Exception)]
        assert len(errors) == 1 and errors[0].code == "quota_exceeded"

    @pytest.mark.asyncio
    async def test_settlement_lock_precedes_snapshot(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        entered, release, snapshot_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(conn):
            snapshot_entered.set()
            usage = await rows(pool, conn, "SELECT total_tokens FROM llm_usage_log")
            return mod.ReservationQuotaSnapshot(sum(row["total_tokens"] for row in usage), 30, 0, None)

        settlement = asyncio.create_task(
            repo.reconcile(
                item.execution_id,
                mod.ReservationActuals(10, 20, 50),
                usage_writer=writer(pool, entered=entered, release=release),
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        admission = asyncio.create_task(repo.reserve(reservation(), snapshot_reader=read))
        try:
            await asyncio.sleep(0.05)
            assert not snapshot_entered.is_set()
        finally:
            release.set()
            await settlement
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await admission
        assert error.value.code == "quota_exceeded"

    @pytest.mark.asyncio
    async def test_partial_usage_uniqueness(self, reservation_pool):
        pool = reservation_pool
        async with pool.transaction() as conn:
            for operation, request_id in [
                ("chat", "legacy"),
                ("chat", "legacy"),
                ("mcp_model_completion", None),
                ("mcp_model_completion", None),
                ("mcp_model_completion", "canonical"),
            ]:
                await execute(
                    pool, conn, "INSERT INTO llm_usage_log (operation, request_id) VALUES (?, ?)", operation, request_id
                )
        with pytest.raises(TransactionError):
            async with pool.transaction() as conn:
                await execute(
                    pool,
                    conn,
                    "INSERT INTO llm_usage_log (operation, request_id) VALUES (?, ?)",
                    "mcp_model_completion",
                    "canonical",
                )

    @pytest.mark.asyncio
    async def test_settlement_callback_receives_immutable_values(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item, actuals = reservation(), mod.ReservationActuals(0, 0, 0)
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)

        async def write(conn, received, counts):
            assert received == item and type(received) is mod.ProviderUsageReservation
            assert counts is actuals
            with pytest.raises(FrozenInstanceError):
                received.model = "mutated"
            await writer(pool)(conn, received, counts)

        await repo.reconcile(item.execution_id, actuals, usage_writer=write)

    @pytest.mark.asyncio
    async def test_settlement_cancellation_rolls_back_usage(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)

        async def cancel(conn, received, counts):
            await writer(pool)(conn, received, counts)
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await repo.reconcile(item.execution_id, mod.ReservationActuals(1, 1, 1), usage_writer=cancel)
        assert (await repo.get(item.execution_id))["state"] == "dispatched"
        async with pool.transaction() as conn:
            assert not await rows(pool, conn, "SELECT * FROM llm_usage_log")

    @pytest.mark.asyncio
    async def test_admission_lock_blocks_settlement_before_usage_write(self, reservation_pool):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        await repo.mark_dispatched(item.execution_id)
        snapshot_entered, release_snapshot, writer_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(_conn):
            snapshot_entered.set()
            await release_snapshot.wait()
            return mod.ReservationQuotaSnapshot(0, 59, 0, None)

        async def write(conn, received, counts):
            writer_entered.set()
            await writer(pool)(conn, received, counts)

        admission = asyncio.create_task(repo.reserve(reservation(), snapshot_reader=read))
        await asyncio.wait_for(snapshot_entered.wait(), 5)
        settlement = asyncio.create_task(
            repo.reconcile(item.execution_id, mod.ReservationActuals(1, 1, 1), usage_writer=write)
        )
        try:
            await asyncio.sleep(0.05)
            assert not writer_entered.is_set()
        finally:
            release_snapshot.set()
            with pytest.raises(mod.ProviderUsageReservationError) as error:
                await admission
            await settlement
        assert error.value.code == "quota_exceeded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["reserved", "released", "ambiguous"])
    async def test_reconcile_requires_dispatched_state(self, reservation_pool, state):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        item = reservation()
        await repo.reserve(item, snapshot_reader=unlimited)
        if state == "released":
            await repo.release_before_dispatch(item.execution_id)
        elif state == "ambiguous":
            await repo.mark_dispatched(item.execution_id)
            await repo.retain_ambiguous(item.execution_id)
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reconcile(
                item.execution_id, mod.ReservationActuals(0, 0, 0), usage_writer=writer(reservation_pool)
            )
        assert error.value.code == "invalid_transition"

    @pytest.mark.asyncio
    async def test_snapshot_required_and_wrong_type_fails_closed(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)

        async def wrong(_conn):
            return {"used_tokens": 0}

        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.reserve(reservation(), snapshot_reader=wrong)
        assert error.value.code == "snapshot_unavailable"
        with pytest.raises(TypeError):
            # Signature binding remains available to callers via functools.wraps.
            import inspect

            inspect.signature(repo.reserve).bind(reservation())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("snapshot", [(MAX_INT, None, 0, None), (0, None, MAX_INT, None)])
    async def test_snapshot_plus_reservation_overflow(self, reservation_pool, snapshot):
        mod = api()

        async def read(_conn):
            return mod.ReservationQuotaSnapshot(*snapshot)

        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await mod.ProviderUsageReservationsRepo(reservation_pool).reserve(reservation(), snapshot_reader=read)
        assert error.value.code == "integer_overflow"

    @pytest.mark.asyncio
    async def test_store_failure_has_no_backend_exception_graph(self, reservation_pool, monkeypatch):
        mod, pool = api(), reservation_pool
        repo = mod.ProviderUsageReservationsRepo(pool)

        async def unavailable(conn, _execution_id):
            return await repo._rows(conn, "SELECT * FROM unavailable_provider_usage_reservations")

        monkeypatch.setattr(repo, "_get", unavailable)
        with pytest.raises(mod.ProviderUsageReservationError) as error:
            await repo.get(str(uuid4()))
        assert error.value.code == "reservation_store_unavailable"
        assert error.value.__context__ is None and error.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_duplicate_execution_across_scopes(self, reservation_pool):
        mod = api()
        repo = mod.ProviderUsageReservationsRepo(reservation_pool)
        first = reservation()
        second = reservation(
            execution_id=first.execution_id, active_team_id=5, billing_scope=mod.BillingScope("team", 5)
        )

        async def read(_conn):
            await asyncio.sleep(0.05)
            return mod.ReservationQuotaSnapshot(0, None, 0, None)

        results = await asyncio.gather(
            repo.reserve(first, snapshot_reader=read),
            repo.reserve(second, snapshot_reader=read),
            return_exceptions=True,
        )
        assert sum(isinstance(result, dict) for result in results) == 1
        errors = [result for result in results if isinstance(result, Exception)]
        assert len(errors) == 1 and errors[0].code == "duplicate_execution"


class TestSQLiteReservations(ReservationContract):
    pass


def test_contracts_frozen():
    mod = api()
    for item in [
        mod.BillingScope("user", 1),
        reservation(),
        mod.ReservationActuals(0, 0, 0),
        mod.ReservationQuotaSnapshot(0, None, 0, None),
    ]:
        field = next(iter(item.__dataclass_fields__))
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, None)
