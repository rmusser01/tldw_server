from __future__ import annotations

import base64
from pathlib import Path

import aiosqlite
import pytest

from tldw_Server_API.app.core.External_Sources import connectors_service as svc


@pytest.fixture
async def sqlite_db(tmp_path: Path):
    db = await aiosqlite.connect(tmp_path / "reference_manager.sqlite3")
    db.row_factory = aiosqlite.Row
    db._is_sqlite = True
    try:
        yield db
    finally:
        await db.close()


async def _create_reference_manager_source(db: aiosqlite.Connection) -> dict:
    account = await svc.create_account(
        db,
        user_id=7,
        provider="zotero",
        display_name="Zotero",
        email="researcher@example.com",
        tokens={"access_token": "token"},
    )
    return await svc.create_source(
        db,
        account_id=account["id"],
        provider="zotero",
        remote_id="COLL1234",
        type_="collection",
        path=None,
        options={},
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_reference_item_binding_round_trips_provider_and_collection_identity(
    sqlite_db: aiosqlite.Connection,
) -> None:
    source = await _create_reference_manager_source(sqlite_db)

    first = await svc.upsert_reference_item_binding(
        sqlite_db,
        source_id=source["id"],
        provider="zotero",
        provider_item_key="ABCD1234",
        provider_library_id="123456",
        collection_key="COLL1234",
        collection_name="Language Models",
        provider_version="1",
        provider_updated_at="2026-03-01T00:00:00Z",
        media_id=99,
        dedupe_match_reason="doi",
        raw_reference_metadata={
            "provider": "zotero",
            "collection_name": "Language Models",
            "title": "Attention Is All You Need",
        },
    )
    second = await svc.upsert_reference_item_binding(
        sqlite_db,
        source_id=source["id"],
        provider="zotero",
        provider_item_key="ABCD1234",
        provider_library_id="123456",
        collection_key="COLL1234",
        collection_name="Language Models",
        provider_version="2",
        provider_updated_at="2026-03-02T00:00:00Z",
        media_id=99,
        dedupe_match_reason="title",
        raw_reference_metadata={
            "provider": "zotero",
            "collection_name": "Language Models",
            "title": "Attention Is All You Need",
        },
    )

    pragma = await sqlite_db.execute("PRAGMA table_info(external_reference_items)")
    columns = {row["name"] for row in await pragma.fetchall()}

    assert first["provider_item_key"] == "ABCD1234"
    assert first["provider_library_id"] == "123456"
    assert first["collection_key"] == "COLL1234"
    assert first["collection_name"] == "Language Models"
    assert first["dedupe_match_reason"] == "doi"
    assert first["first_imported_at"] is not None
    assert first["last_imported_at"] is not None
    assert second["dedupe_match_reason"] == "title"
    assert second["first_imported_at"] == first["first_imported_at"]
    assert second["last_imported_at"] >= first["last_imported_at"]
    assert second["raw_reference_metadata"]["collection_name"] == "Language Models"
    assert {"provider_item_key", "collection_key", "dedupe_match_reason", "first_imported_at", "last_imported_at"} <= columns


@pytest.mark.asyncio
@pytest.mark.unit
async def test_reference_item_binding_clears_metadata_only_reason_when_item_later_imports(
    sqlite_db: aiosqlite.Connection,
) -> None:
    source = await _create_reference_manager_source(sqlite_db)

    first = await svc.upsert_reference_item_binding(
        sqlite_db,
        source_id=source["id"],
        provider="zotero",
        provider_item_key="META1234",
        provider_library_id="123456",
        collection_key="COLL1234",
        collection_name="Language Models",
        provider_version="1",
        provider_updated_at="2026-03-01T00:00:00Z",
        media_id=None,
        dedupe_match_reason="metadata_only",
        raw_reference_metadata={"title": "Attachment Pending"},
    )
    second = await svc.upsert_reference_item_binding(
        sqlite_db,
        source_id=source["id"],
        provider="zotero",
        provider_item_key="META1234",
        provider_library_id="123456",
        collection_key="COLL1234",
        collection_name="Language Models",
        provider_version="2",
        provider_updated_at="2026-03-02T00:00:00Z",
        media_id=55,
        dedupe_match_reason=None,
        raw_reference_metadata={"title": "Attachment Imported"},
    )

    assert first["media_id"] is None
    assert first["dedupe_match_reason"] == "metadata_only"
    assert second["media_id"] == 55
    assert second["dedupe_match_reason"] is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_oauth_state_metadata_is_encrypted_at_rest_when_crypto_enabled(
    sqlite_db: aiosqlite.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", base64.b64encode(b"r" * 32).decode("ascii"))

    await svc.create_oauth_state(
        sqlite_db,
        user_id=7,
        provider="zotero",
        state="state-123",
        metadata={
            "oauth_token": "temp-token",
            "oauth_token_secret": "temp-secret",
        },
    )

    cur = await sqlite_db.execute(
        "SELECT metadata FROM external_oauth_state WHERE state = ? AND user_id = ?",
        ("state-123", 7),
    )
    row = await cur.fetchone()
    assert row is not None
    assert isinstance(row["metadata"], str)
    assert "temp-secret" not in row["metadata"]

    consumed = await svc.consume_oauth_state(
        sqlite_db,
        user_id=7,
        provider="zotero",
        state="state-123",
    )

    assert consumed is not False
    assert consumed["oauth_token"] == "temp-token"
    assert consumed["oauth_token_secret"] == "temp-secret"
