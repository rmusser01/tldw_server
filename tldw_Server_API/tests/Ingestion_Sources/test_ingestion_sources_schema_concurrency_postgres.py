"""Sources status reads preserve sync progress and cold/partial PostgreSQL upgrades."""

import asyncio
import secrets
from urllib.parse import quote

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.Ingestion_Sources_DB import postgres_source_schema_is_ready
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    _service_connection,
    create_source,
    ensure_ingestion_sources_schema,
    get_source_by_id,
    list_source_items,
    start_source_sync_job,
    upsert_source_item,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def source_pool(request, pg_restricted_backend, monkeypatch):
    cfg = pg_restricted_backend.config
    dsn = (
        "postgresql://"
        + quote(cfg.pg_user, safe="")
        + ":"
        + quote(cfg.pg_password, safe="")
        + "@"
        + cfg.pg_host
        + ":"
        + str(cfg.pg_port)
        + "/"
        + cfg.pg_database
    )
    monkeypatch.setenv("TLDW_USER_DB_BACKEND", "postgresql")
    settings = Settings(
        AUTH_MODE=getattr(request, "param", "single_user"),
        DATABASE_URL=dsn,
        JWT_SECRET_KEY=secrets.token_urlsafe(48),
        DATABASE_POOL_MIN_SIZE=2,
        DATABASE_POOL_MAX_SIZE=5,
    )
    pool = DatabasePool(settings)
    await pool.initialize()
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def seeded_source(source_pool, tmp_path):
    async with source_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        return await create_source(
            db,
            user_id=341,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": str(tmp_path)},
            },
        )


@pytest.mark.parametrize("source_pool", ["single_user", "multi_user"], indirect=True)
async def test_warm_status_read_does_not_block_source_sync_writer(source_pool, seeded_source):
    pool, source = source_pool, seeded_source
    pending = None
    entered = asyncio.Event()
    completed_before_reader_exit = False

    async def sync_state_writer():
        async with pool.transaction() as db:
            entered.set()
            await ensure_ingestion_sources_schema(db)
            return await start_source_sync_job(db, source_id=source["id"], job_id="41")

    try:
        try:
            async with pool.transaction() as reader:
                await ensure_ingestion_sources_schema(reader)
                found = await get_source_by_id(reader, source_id=source["id"], user_id=341)
                assert found["id"] == source["id"]  # nosec B101 -- Verify the real owned status read.
                pending = asyncio.create_task(sync_state_writer())
                await asyncio.wait_for(entered.wait(), 5)
                try:
                    await asyncio.wait_for(asyncio.shield(pending), 1)
                except TimeoutError:
                    pass
                else:
                    completed_before_reader_exit = True
        finally:
            if pending is not None:
                # Release the reader transaction before draining the blocked writer.
                result = await asyncio.wait_for(pending, 5)
        assert completed_before_reader_exit  # nosec B101 -- A status read must not lock out sync progress.
        assert result["active_job_id"] == "41"  # nosec B101 -- Require actual committed source state.
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_cold_schema_and_missing_index_use_normal_ddl(source_pool, pg_restricted_backend):
    async with source_pool.transaction() as db:
        assert not await postgres_source_schema_is_ready(_service_connection(db))  # nosec B101 -- Cold catalogs cannot enter the read-only fast path.
        await ensure_ingestion_sources_schema(db)
        assert await postgres_source_schema_is_ready(_service_connection(db))  # nosec B101 -- Normal cold setup installs every required relation.
    pg_restricted_backend.execute("DROP INDEX idx_ingestion_sources_scheduler")
    async with source_pool.transaction() as db:
        assert not await postgres_source_schema_is_ready(_service_connection(db))  # nosec B101 -- A partial catalog must run its normal upgrade.
        await ensure_ingestion_sources_schema(db)
        assert await postgres_source_schema_is_ready(_service_connection(db))  # nosec B101 -- Missing indexes are restored.


async def test_legacy_presence_upgrade_preserves_complete_existing_rows(
    source_pool, seeded_source, pg_restricted_backend
):
    async with source_pool.transaction() as db:
        await upsert_source_item(
            db,
            source_id=seeded_source["id"],
            normalized_relative_path="retained.txt",
            content_hash="retained-content-hash",
            sync_status="active",
            binding={"media_id": 17},
            present_in_source=True,
        )
        before = await list_source_items(db, source_id=seeded_source["id"], include_absent=True)
    pg_restricted_backend.execute("ALTER TABLE ingestion_source_items DROP COLUMN present_in_source")
    async with source_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        after = await list_source_items(db, source_id=seeded_source["id"], include_absent=True)
    assert after == before  # nosec B101 -- Upgrade restores presence without losing any existing item field.


async def test_relations_in_another_schema_cannot_fill_missing_catalog(
    source_pool, seeded_source, pg_restricted_backend
):
    pg_restricted_backend.execute("CREATE SCHEMA source_catalog_other")
    pg_restricted_backend.execute(
        "CREATE TABLE source_catalog_other.ingestion_source_artifacts "
        "(id BIGINT, source_id BIGINT, artifact_kind TEXT, status TEXT)"
    )
    pg_restricted_backend.execute(
        "CREATE INDEX idx_ingestion_source_artifacts_source_kind_status "
        "ON source_catalog_other.ingestion_source_artifacts(source_id, artifact_kind, status, id DESC)"
    )
    async with source_pool.transaction() as db:
        source_before = await get_source_by_id(db, source_id=seeded_source["id"], user_id=341)
    pg_restricted_backend.execute("DROP TABLE public.ingestion_source_artifacts")
    async with source_pool.transaction() as db:
        before = await postgres_source_schema_is_ready(_service_connection(db))
        await ensure_ingestion_sources_schema(db)
        after = await postgres_source_schema_is_ready(_service_connection(db))
        source_after = await get_source_by_id(db, source_id=seeded_source["id"], user_id=341)
    assert before is False  # nosec B101 -- Other-schema names cannot qualify missing target relations.
    assert after is True  # nosec B101 -- Normal DDL restores the target namespace's own relations.
    assert source_after == source_before  # nosec B101 -- Compare complete GET snapshots; create has a smaller response.


@pytest.mark.parametrize("variant", ["wrong_index_parent", "wrong_table_kind"])
async def test_relation_names_alone_do_not_qualify_catalog(source_pool, seeded_source, pg_restricted_backend, variant):
    if variant == "wrong_index_parent":
        pg_restricted_backend.execute("DROP INDEX idx_ingestion_sources_scheduler")
        pg_restricted_backend.execute(
            "CREATE INDEX idx_ingestion_sources_scheduler ON ingestion_source_state(source_id)"
        )
    else:
        pg_restricted_backend.execute("DROP TABLE ingestion_source_artifacts")
        pg_restricted_backend.execute("CREATE VIEW ingestion_source_artifacts AS SELECT 1 AS id")
    async with source_pool.transaction() as db:
        assert not await postgres_source_schema_is_ready(_service_connection(db))  # nosec B101 -- Wrong relation kinds or index parents cannot skip schema setup.
