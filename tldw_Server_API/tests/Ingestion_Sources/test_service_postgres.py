from __future__ import annotations

import uuid

import pytest

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)
pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_ingestion_sources_schema_initializes_twice_in_postgres(test_db_pool) -> None:
    """The service schema must initialize idempotently on guarded PostgreSQL."""
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        ensure_ingestion_sources_schema,
    )

    async with test_db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        await ensure_ingestion_sources_schema(db)


@pytest.mark.asyncio
async def test_ingestion_sources_upgrades_legacy_item_presence_in_postgres(test_db_pool) -> None:
    """An existing catalog gains presence state without losing its item bindings."""
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        ensure_ingestion_sources_schema,
        list_source_items,
        upsert_source_item,
    )

    async with test_db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        source = await create_source(
            db,
            user_id=7,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/legacy"},
            },
        )
        item = await upsert_source_item(
            db,
            source_id=int(source["id"]),
            normalized_relative_path="legacy.md",
            content_hash="original",
            sync_status="completed",
            binding={"media_id": 17},
            present_in_source=True,
        )
        # Model a pre-column catalog in the official fixture's isolated database.
        await db.execute("ALTER TABLE ingestion_source_items DROP COLUMN present_in_source")

        await ensure_ingestion_sources_schema(db)
        await ensure_ingestion_sources_schema(db)

        column = await db.fetchrow(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'ingestion_source_items' AND column_name = 'present_in_source'"
        )
        if column is None:
            pytest.fail("Existing PostgreSQL source catalog was not upgraded with present_in_source")

        rows = await list_source_items(db, source_id=int(source["id"]))
        assert len(rows) == 1
        assert rows[0]["id"] == item["id"]
        assert rows[0]["binding"] == {"media_id": 17}
        assert rows[0]["present_in_source"] is True

        updated = await upsert_source_item(
            db,
            source_id=int(source["id"]),
            normalized_relative_path="legacy.md",
            content_hash="changed",
            sync_status="completed",
            binding={"media_id": 17},
            present_in_source=False,
        )
        assert updated["id"] == item["id"]
        assert updated["present_in_source"] is False
        assert await list_source_items(db, source_id=int(source["id"]), include_absent=False) == []


@pytest.mark.asyncio
async def test_current_source_catalog_initialization_does_not_lock_out_readers(test_db_pool) -> None:
    """Per-request schema checks must not acquire an exclusive lock after upgrade."""
    from tldw_Server_API.app.core.Ingestion_Sources.service import ensure_ingestion_sources_schema

    async with test_db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
    async with test_db_pool.transaction() as reader:
        await reader.fetch("SELECT id FROM ingestion_source_items")
        async with test_db_pool.transaction() as initializer:
            await initializer.execute("SET LOCAL lock_timeout = '500ms'")
            await ensure_ingestion_sources_schema(initializer)


@pytest.mark.asyncio
async def test_ingestion_sources_service_round_trip_is_owner_isolated_in_postgres(
    test_db_pool,
) -> None:
    """Guarded PostgreSQL supports source writes, IDs, and owner-only readbacks."""
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.Ingestion_Sources.service import (
        create_source,
        create_source_artifact,
        create_source_snapshot,
        delete_source_artifact,
        delete_source_snapshot,
        ensure_ingestion_sources_schema,
        finish_source_sync_job,
        get_source_by_id,
        list_source_artifacts,
        list_source_items,
        list_source_snapshots,
        list_sources_by_user,
        list_sources_for_retention_cleanup,
        list_sources_for_scheduler,
        record_ingestion_item_event,
        start_source_sync_job,
        update_source,
        update_source_artifact,
        update_source_item_state,
        update_source_snapshot,
        upsert_source_item,
    )

    users_db = UsersDB(test_db_pool)
    await users_db.initialize()
    unique_suffix = uuid.uuid4().hex
    owner_a = int((await users_db.create_user(
        username=f"ingestion-source-owner-a-{unique_suffix}",
        email=f"ingestion-source-owner-a-{unique_suffix}@example.test",
        password_hash=uuid.uuid4().hex,
    ))["id"])
    owner_b_id = int((await users_db.create_user(
        username=f"ingestion-source-owner-b-{unique_suffix}",
        email=f"ingestion-source-owner-b-{unique_suffix}@example.test",
        password_hash=uuid.uuid4().hex,
    ))["id"])

    async with test_db_pool.transaction() as db:
        await ensure_ingestion_sources_schema(db)
        await ensure_ingestion_sources_schema(db)
        source_a = await create_source(
            db,
            user_id=owner_a,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/owner-a"},
            },
        )
        source_b = await create_source(
            db,
            user_id=owner_b_id,
            payload={
                "source_type": "local_directory",
                "sink_type": "media",
                "policy": "canonical",
                "config": {"path": "/allowed/owner-b"},
            },
        )
        source_a_id = int(source_a["id"])
        source_b_id = int(source_b["id"])

        assert source_a_id > 0
        assert source_b_id > 0
        assert await get_source_by_id(db, source_id=source_a_id, user_id=owner_b_id) == {}
        assert [row["id"] for row in await list_sources_by_user(db, user_id=owner_a)] == [source_a_id]
        assert [row["id"] for row in await list_sources_by_user(db, user_id=owner_b_id)] == [source_b_id]

        snapshot = await create_source_snapshot(
            db,
            source_id=source_a_id,
            snapshot_kind="initial",
            status="success",
            summary={"items": 1},
        )
        artifact = await create_source_artifact(
            db,
            source_id=source_a_id,
            snapshot_id=int(snapshot["id"]),
            artifact_kind="manifest",
            status="ready",
            storage_path="storage/manifest.json",
            metadata={"format": "json"},
        )
        first_item = await upsert_source_item(
            db,
            source_id=source_a_id,
            normalized_relative_path="notes/a.md",
            content_hash="first",
            sync_status="pending",
            binding={"media_id": 1},
            present_in_source=True,
        )
        updated_item = await upsert_source_item(
            db,
            source_id=source_a_id,
            normalized_relative_path="notes/a.md",
            content_hash="second",
            sync_status="completed",
            binding={"media_id": 2},
            present_in_source=False,
        )
        event = await record_ingestion_item_event(
            db,
            source_id=source_a_id,
            item_path="notes/a.md",
            event_type="updated",
            payload={"revision": 2},
        )

        assert int(snapshot["id"]) > 0
        assert int(artifact["id"]) > 0
        assert int(first_item["id"]) == int(updated_item["id"])
        assert int(event["id"]) > 0
        items = await list_source_items(db, source_id=source_a_id, include_absent=True)
        assert len(items) == 1
        assert items[0]["content_hash"] == "second"
        assert items[0]["present_in_source"] is False

        scheduled = await create_source(
            db,
            user_id=owner_a,
            payload={
                "source_type": "archive_snapshot",
                "sink_type": "media",
                "policy": "canonical",
                "schedule": {"interval_minutes": 30},
                "config": {"path": "/allowed/scheduled"},
            },
        )
        assert [row["id"] for row in await list_sources_for_scheduler(db)] == [
            int(scheduled["id"])
        ]
        assert [row["id"] for row in await list_sources_for_retention_cleanup(db)] == [
            int(scheduled["id"])
        ]

        updated_source = await update_source(
            db,
            source_id=source_a_id,
            user_id=owner_a,
            patch={"enabled": False},
        )
        assert updated_source["enabled"] is False
        started_state = await start_source_sync_job(
            db,
            source_id=source_a_id,
            job_id="source-a-job",
        )
        assert started_state["active_job_id"] == "source-a-job"
        finished_state = await finish_source_sync_job(
            db,
            source_id=source_a_id,
            job_id="source-a-job",
            outcome="success",
            snapshot_id=int(snapshot["id"]),
        )
        assert finished_state["last_successful_snapshot_id"] == int(snapshot["id"])
        with pytest.raises(RuntimeError, match="active sync job"):
            await finish_source_sync_job(
                db,
                source_id=source_a_id,
                job_id="wrong-job",
                outcome="success",
                snapshot_id=int(snapshot["id"]),
            )

        updated_snapshot = await update_source_snapshot(
            db,
            snapshot_id=int(snapshot["id"]),
            summary={"items": 2},
        )
        assert updated_snapshot["summary"] == {"items": 2}
        updated_artifact = await update_source_artifact(
            db,
            artifact_id=int(artifact["id"]),
            status="archived",
            metadata={"retained": True},
        )
        assert updated_artifact["status"] == "archived"
        assert updated_artifact["metadata"] == {"format": "json", "retained": True}
        assert len(await list_source_artifacts(db, source_id=source_a_id)) == 1

        updated_item_state = await update_source_item_state(
            db,
            item_id=int(updated_item["id"]),
            sync_status="indexed",
            binding={"media_id": 3},
            present_in_source=False,
        )
        assert updated_item_state["sync_status"] == "indexed"
        assert await list_source_items(db, source_id=source_a_id, include_absent=False) == []

        await delete_source_artifact(db, artifact_id=int(artifact["id"]))
        await delete_source_snapshot(db, snapshot_id=int(snapshot["id"]))
        assert await list_source_artifacts(db, source_id=source_a_id) == []
        assert await list_source_snapshots(db, source_id=source_a_id) == []
