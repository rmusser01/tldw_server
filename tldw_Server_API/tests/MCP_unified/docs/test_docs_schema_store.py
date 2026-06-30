from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def test_store_migrates_and_reports_status(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()

    status = store.status()

    assert status["schema_version"] == 1  # nosec B101
    assert status["fts_available"] is True  # nosec B101
    assert status["counts"]["documents"] == 0  # nosec B101


def test_document_without_collection_is_searchable(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    document_id = store.upsert_document(
        scope=scope,
        title="Install Guide",
        document_type="markdown",
        canonical_uri="file:///docs/install.md",
        source_path="/docs/install.md",
        source_url=None,
        text="Install the server with sqlite fts enabled.",
        sections=[{"heading": "Install", "level": 1, "start_char": 0, "end_char": 43}],
        chunks=[{"text": "Install the server with sqlite fts enabled.", "citation": "install.md:1"}],
        keywords=("setup",),
        collection_names=(),
        metadata={"source": "unit"},
    )

    results = store.search_chunks(scope=scope, query="sqlite", limit=10)

    assert document_id > 0  # nosec B101
    assert len(results) == 1  # nosec B101
    assert results[0]["title"] == "Install Guide"  # nosec B101


@pytest.mark.parametrize("query", ("sqlite:", "C++", "foo/bar", "foo -bar"))
def test_search_chunks_treats_punctuated_user_queries_as_terms(tmp_path: Path, query: str) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_document(
        scope=scope,
        title="Punctuation Guide",
        document_type="text",
        canonical_uri="file:///punctuation.txt",
        source_path="/punctuation.txt",
        source_url=None,
        text="sqlite C++ foo/bar foo bar material",
        sections=[],
        chunks=[{"text": "sqlite C++ foo/bar foo bar material", "citation": "punctuation.txt"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    results = store.search_chunks(scope=scope, query=query, limit=10)

    assert len(results) == 1  # nosec B101
    assert results[0]["title"] == "Punctuation Guide"  # nosec B101


def test_search_chunks_applies_supported_filters(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_document(
        scope=scope,
        title="Filtered Doc",
        document_type="markdown",
        canonical_uri="file:///docs/filtered.md",
        source_path="/docs/filtered.md",
        source_url=None,
        text="fts filter material",
        sections=[],
        chunks=[{"text": "fts filter material", "citation": "filtered.md"}],
        keywords=("setup", "sqlite"),
        collection_names=("Reference",),
        metadata={"package": "sqlite", "version": "3.45"},
    )
    store.upsert_document(
        scope=scope,
        title="Other Doc",
        document_type="html",
        canonical_uri="file:///other/filtered.html",
        source_path="/other/filtered.html",
        source_url=None,
        text="fts filter material",
        sections=[],
        chunks=[{"text": "fts filter material", "citation": "filtered.html"}],
        keywords=("misc",),
        collection_names=("Other",),
        metadata={"package": "other", "version": "1.0"},
    )

    filtered_results = store.search_chunks(
        scope=scope,
        query="fts",
        limit=10,
        filters={
            "collection": "Reference",
            "keywords": ("setup", "sqlite"),
            "document_type": "markdown",
            "uri_prefix": "file:///docs/",
            "package": "sqlite",
            "version": "3.45",
        },
    )
    mismatched_results = store.search_chunks(
        scope=scope,
        query="fts",
        limit=10,
        filters={
            "collection": "Reference",
            "keywords": ("missing",),
            "document_type": "markdown",
            "uri_prefix": "file:///docs/",
            "package": "sqlite",
            "version": "3.45",
        },
    )

    assert [result["title"] for result in filtered_results] == ["Filtered Doc"]  # nosec B101
    assert mismatched_results == []  # nosec B101


def test_store_enforces_owner_and_profile_scope(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    store.upsert_document(
        scope=scope_a,
        title="Private Doc",
        document_type="text",
        canonical_uri="file:///private.txt",
        source_path="/private.txt",
        source_url=None,
        text="private sqlite material",
        sections=[],
        chunks=[{"text": "private sqlite material", "citation": "private.txt"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    assert store.search_chunks(scope=scope_a, query="sqlite", limit=10)  # nosec B101
    assert store.search_chunks(scope=scope_b, query="sqlite", limit=10) == []  # nosec B101


def test_default_scope_upsert_replaces_document_without_duplicates(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()

    first_id = store.upsert_document(
        scope=scope,
        title="Old Title",
        document_type="text",
        canonical_uri="file:///same.txt",
        source_path="/same.txt",
        source_url=None,
        text="old sqlite text",
        sections=[],
        chunks=[{"text": "old sqlite text", "citation": "same.txt:1"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )
    second_id = store.upsert_document(
        scope=scope,
        title="Updated Title",
        document_type="text",
        canonical_uri="file:///same.txt",
        source_path="/same.txt",
        source_url=None,
        text="updated sqlite text",
        sections=[],
        chunks=[{"text": "updated sqlite text", "citation": "same.txt:1"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    results = store.search_chunks(scope=scope, query="sqlite", limit=10)
    documents = store.list_documents(scope=scope, limit=10, offset=0)

    assert second_id == first_id  # nosec B101
    assert len(documents) == 1  # nosec B101
    assert len(results) == 1  # nosec B101
    assert results[0]["title"] == "Updated Title"  # nosec B101


def test_migrate_backfills_legacy_null_default_scope_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "docs.db"
    with closing(DocsCatalogStore(db_path).connect()) as conn:
        conn.execute(
            """
            CREATE TABLE docs_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_scope TEXT,
                profile_scope TEXT,
                title TEXT NOT NULL,
                document_type TEXT NOT NULL,
                canonical_uri TEXT NOT NULL,
                source_path TEXT,
                source_url TEXT,
                content_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                package_name TEXT,
                package_version TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor = conn.execute(
            """
            INSERT INTO docs_documents (
                owner_scope,
                profile_scope,
                title,
                document_type,
                canonical_uri,
                source_path,
                source_url,
                content_hash,
                text,
                metadata_json
            )
            VALUES (NULL, NULL, ?, ?, ?, ?, NULL, ?, ?, '{}')
            """,
            (
                "Legacy Doc",
                "text",
                "file:///legacy.txt",
                "/legacy.txt",
                "legacy-hash",
                "legacy sqlite text",
            ),
        )
        legacy_id = int(cursor.lastrowid)
        conn.commit()

    store = DocsCatalogStore(db_path)
    store.migrate()
    updated_id = store.upsert_document(
        scope=AccessScope(),
        title="Updated Legacy Doc",
        document_type="text",
        canonical_uri="file:///legacy.txt",
        source_path="/legacy.txt",
        source_url=None,
        text="updated legacy sqlite text",
        sections=[],
        chunks=[{"text": "updated legacy sqlite text", "citation": "legacy.txt"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    documents = store.list_documents(scope=AccessScope(), limit=10)

    assert updated_id == legacy_id  # nosec B101
    assert [document["title"] for document in documents] == ["Updated Legacy Doc"]  # nosec B101


def test_list_keywords_includes_keyword_field(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    store.upsert_document(
        scope=scope,
        title="Keyword Doc",
        document_type="text",
        canonical_uri="file:///keyword.txt",
        source_path="/keyword.txt",
        source_url=None,
        text="keyword sqlite material",
        sections=[],
        chunks=[{"text": "keyword sqlite material", "citation": "keyword.txt"}],
        keywords=("setup",),
        collection_names=(),
        metadata={},
    )

    keywords = store.list_keywords(scope)

    assert keywords == [{"id": 1, "name": "setup", "keyword": "setup", "document_count": 1}]  # nosec B101


def test_resolve_name_returns_scoped_document_collection_and_keyword_matches(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    document_id = store.upsert_document(
        scope=scope_a,
        title="Private Manual",
        document_type="markdown",
        canonical_uri="file:///private-manual.md",
        source_path="/private-manual.md",
        source_url=None,
        text="private manual sqlite material",
        sections=[],
        chunks=[{"text": "private manual sqlite material", "citation": "private-manual.md"}],
        keywords=("private-keyword",),
        collection_names=("Private Collection",),
        metadata={},
    )

    matches = store.resolve_name(scope_a, "Private")
    by_type = {match["target_type"]: match for match in matches}

    assert by_type["document"] == {  # nosec B101
        "target_type": "document",
        "id": str(document_id),
        "title": "Private Manual",
        "uri": "file:///private-manual.md",
    }
    assert by_type["collection"] == {  # nosec B101
        "target_type": "collection",
        "id": "Private Collection",
        "title": "Private Collection",
        "metadata": {"document_count": 1, "description": ""},
    }
    assert by_type["keyword"] == {  # nosec B101
        "target_type": "keyword",
        "id": "private-keyword",
        "title": "private-keyword",
    }
    assert store.resolve_name(scope_b, "Private") == []  # nosec B101


def test_list_collections_includes_description_and_schema_supports_column(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    store.upsert_document(
        scope=scope,
        title="Collection Doc",
        document_type="text",
        canonical_uri="file:///collection.txt",
        source_path="/collection.txt",
        source_url=None,
        text="collection sqlite material",
        sections=[],
        chunks=[{"text": "collection sqlite material", "citation": "collection.txt"}],
        keywords=(),
        collection_names=("Reference",),
        metadata={},
    )

    with closing(store.connect()) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(docs_collections)").fetchall()}
        assert "description" in columns  # nosec B101
        conn.execute(
            """
            UPDATE docs_collections
            SET description = ?
            WHERE name = ?
            """,
            ("Curated reference docs", "Reference"),
        )
        conn.commit()

    collections = store.list_collections(scope)

    assert collections == [  # nosec B101
        {"id": 1, "name": "Reference", "description": "Curated reference docs", "document_count": 1}
    ]
