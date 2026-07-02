from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


def test_migrate_adds_source_tables_and_document_lifecycle_columns(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()

    with closing(store.connect()) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")
        }
        document_columns = {row["name"] for row in conn.execute("PRAGMA table_info(docs_documents)")}

    assert {"docs_sources", "docs_source_documents", "docs_sync_runs"}.issubset(tables)  # nosec B101
    assert "lifecycle_status" in document_columns  # nosec B101
    assert "preserve_on_source_tombstone" in document_columns  # nosec B101


def test_store_upserts_and_lists_sources_by_scope(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    source_id = store.upsert_source(
        scope=scope_a,
        source_type="local_file",
        canonical_uri="file:///docs/a.md",
        display_name="a.md",
        source_path="/docs/a.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["local"]},
    )
    second_id = store.upsert_source(
        scope=scope_a,
        source_type="local_file",
        canonical_uri="file:///docs/a.md",
        display_name="a.md updated",
        source_path="/docs/a.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["local"]},
    )

    assert second_id == source_id  # nosec B101
    assert [source["canonical_uri"] for source in store.list_sources(scope=scope_a)] == ["file:///docs/a.md"]  # nosec B101
    assert store.list_sources(scope=scope_b) == []  # nosec B101


def test_source_document_count_ignores_out_of_scope_links(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    source_id = store.upsert_source(
        scope=scope_a,
        source_type="local_file",
        canonical_uri="file:///docs/a.md",
        display_name="a.md",
        source_path="/docs/a.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    out_of_scope_document_id = store.upsert_document(
        scope=scope_b,
        title="Other Owner",
        document_type="text",
        canonical_uri="file:///other-owner.txt",
        source_path="/other-owner.txt",
        source_url=None,
        text="other owner sqlite body",
        sections=[],
        chunks=[{"text": "other owner sqlite body", "citation": "other-owner.txt"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    with closing(store.connect()) as conn:
        conn.execute(
            """
            INSERT INTO docs_source_documents (source_id, document_id, source_item_uri)
            VALUES (?, ?, ?)
            """,
            (source_id, out_of_scope_document_id, "file:///other-owner.txt"),
        )
        conn.commit()

    assert store.get_source(scope=scope_a, source_id=source_id)["document_count"] == 0  # nosec B101
    assert store.get_source(scope=scope_a, canonical_uri="file:///docs/a.md")["document_count"] == 0  # nosec B101
    assert store.list_sources(scope=scope_a)[0]["document_count"] == 0  # nosec B101


def test_upsert_document_for_sync_merges_existing_organization(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    document_id = store.upsert_document(
        scope=scope,
        title="Guide",
        document_type="text",
        canonical_uri="file:///guide.txt",
        source_path="/guide.txt",
        source_url=None,
        text="old sqlite body",
        sections=[],
        chunks=[{"text": "old sqlite body", "citation": "guide.txt"}],
        keywords=("manual",),
        collection_names=("Manual",),
        metadata={"importer": "local"},
    )

    updated_id = store.upsert_document_for_sync(
        scope=scope,
        title="Guide",
        document_type="text",
        canonical_uri="file:///guide.txt",
        source_path="/guide.txt",
        source_url=None,
        text="new sqlite body",
        sections=[],
        chunks=[{"text": "new sqlite body", "citation": "guide.txt"}],
        source_default_keywords=("source-default",),
        source_default_collections=("Source Defaults",),
        metadata={"importer": "local", "sync": True},
    )

    assert updated_id == document_id  # nosec B101
    assert store.search_chunks(scope, "new", limit=10)[0]["title"] == "Guide"  # nosec B101
    assert {item["keyword"] for item in store.list_keywords(scope)} == {"manual", "source-default"}  # nosec B101
    assert {item["name"] for item in store.list_collections(scope)} == {"Manual", "Source Defaults"}  # nosec B101
