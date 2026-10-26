"""Case-insensitive Media DB keyword identity (TASK-13950.5, slice 2).

Keywords are case-insensitive product-wide with display casing preserved: the
first-seen casing is stored, and every lookup, dedupe and link compares
``normalize_keyword_identity`` values (NFC plus casefold).

SQLite gets ASCII case-insensitivity from ``Keywords.keyword COLLATE NOCASE``.
Postgres strips NOCASE, so it gets a unique ``LOWER(keyword)`` index over active
rows, created after active case-variant duplicates are merged. Neither folds
the German sharp s, accented capitals (SQLite) or NFC forms, so
``KeywordsRepository.find`` falls back to comparing helper identities in Python.
Known limit: duplicates that only the helper folds (ß/SS, NFC) and that already
exist are not merged; lookups resolve them to the earliest row.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.api import fetch_keywords_for_media
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

NON_ASCII_VARIANTS = [
    pytest.param("Straße", "STRASSE", id="sharp-s"),
    pytest.param("Éclair", "éclair", id="accented-capital"),
    pytest.param("éclair", "éclair", id="nfc-composition"),
]


@pytest.fixture(
    params=[
        pytest.param("sqlite", marks=pytest.mark.unit),
        pytest.param("postgres", marks=pytest.mark.integration),
    ]
)
def db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = MediaDatabase(db_path=str(tmp_path / "media_keywords.db"), client_id="913950", backend=backend)
    yield database
    database.close_connection()


def _active_keywords(db: MediaDatabase) -> list[str]:
    return db.fetch_all_keywords()


def _add_media(db: MediaDatabase, keywords: list[str], title: str = "Keyword identity") -> int:
    media_id, _, _ = db.add_media_with_keywords(title=title, media_type="text", content=title, keywords=keywords)
    assert media_id is not None
    return int(media_id)


def _link_count(db: MediaDatabase, media_id: int) -> int:
    rows = db.execute_query("SELECT COUNT(*) AS n FROM MediaKeywords WHERE media_id = ?", (media_id,)).fetchall()
    return int(rows[0]["n"])


def test_adding_foo_then_foo_keeps_one_keyword_found_in_either_case(db) -> None:
    keyword_id, _ = db.add_keyword("Foo")

    assert db.add_keyword("foo")[0] == keyword_id
    assert _active_keywords(db) == ["Foo"]

    media_id = _add_media(db, ["FOO"])
    assert fetch_keywords_for_media(db, media_id) == ["Foo"]
    for probe in ("Foo", "foo", "FOO"):
        found = db.fetch_media_for_keywords([probe])
        assert [item["id"] for item in found["foo"]] == [media_id], probe


def test_linking_case_variants_to_media_with_foo_keeps_one_link(db) -> None:
    media_id = _add_media(db, ["foo"])

    db.update_keywords_for_media(media_id, ["Foo", "foo"])

    assert fetch_keywords_for_media(db, media_id) == ["foo"]
    assert _link_count(db, media_id) == 1


def test_media_keywords_keep_their_first_seen_display_casing(db) -> None:
    media_id = _add_media(db, ["Alpha", "alpha", "BETA"])

    assert fetch_keywords_for_media(db, media_id) == ["Alpha", "BETA"]
    assert _active_keywords(db) == ["Alpha", "BETA"]


def test_soft_delete_in_another_case_then_readd_restores_the_same_keyword(db) -> None:
    keyword_id, _ = db.add_keyword("Foo")

    assert db.soft_delete_keyword("FOO") is True
    assert _active_keywords(db) == []
    assert db.add_keyword("foo")[0] == keyword_id
    assert _active_keywords(db) == ["Foo"]


@pytest.mark.parametrize(("stored", "probe"), NON_ASCII_VARIANTS)
def test_non_ascii_case_variants_share_one_keyword(db, stored: str, probe: str) -> None:
    keyword_id, _ = db.add_keyword(stored)
    media_id = _add_media(db, [probe])

    assert db.add_keyword(probe)[0] == keyword_id
    assert fetch_keywords_for_media(db, media_id) == [stored]
    assert len(_active_keywords(db)) == 1
    found = db.fetch_media_for_keywords([stored])
    assert [item["id"] for item in next(iter(found.values()))] == [media_id]


# --- Postgres: merge existing case-variant duplicates, then enforce LOWER uniqueness ---


def _keyword_rows(backend) -> list[dict]:
    return [
        dict(row)
        for row in backend.execute(
            "SELECT id, uuid, keyword, deleted, version, merge_parent_uuid FROM keywords ORDER BY id"
        ).rows
    ]


def _links(backend) -> set[tuple[int, int]]:
    rows = backend.execute("SELECT media_id, keyword_id FROM mediakeywords").rows
    return {(int(row["media_id"]), int(row["keyword_id"])) for row in rows}


@pytest.mark.integration
def test_postgres_merges_case_variant_duplicates_before_enforcing_lower_uniqueness(pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="913950", backend=backend)
    try:
        both = _add_media(db, ["seed"], title="both")
        only_loser = _add_media(db, ["seed"], title="only loser")
        # Pre-index state: NOCASE is stripped on Postgres, so case variants could coexist.
        backend.execute("DROP INDEX IF EXISTS idx_keywords_keyword_identity")
        with backend.transaction() as conn:
            for text, deleted in (("Foo", False), ("foo", False), ("FOO", False), ("fOo", True), ("Bar", False)):
                backend.execute(
                    "INSERT INTO keywords (keyword, uuid, last_modified, version, client_id, deleted) "
                    "VALUES (%s, %s, '2026-10-01T00:00:00Z', 1, 'seed', %s)",
                    (text, f"uuid-{text}", deleted),
                    connection=conn,
                )
        ids = {row["keyword"]: int(row["id"]) for row in _keyword_rows(backend)}
        with backend.transaction() as conn:
            for media_id, keyword in ((both, "Foo"), (both, "foo"), (only_loser, "FOO")):
                backend.execute(
                    "INSERT INTO mediakeywords (media_id, keyword_id) VALUES (%s, %s)",
                    (media_id, ids[keyword]),
                    connection=conn,
                )

        db._initialize_schema()

        rows = {row["keyword"]: row for row in _keyword_rows(backend)}
        assert [text for text, row in rows.items() if not row["deleted"]] == ["seed", "Foo", "Bar"]
        for loser in ("foo", "FOO"):
            assert rows[loser]["deleted"] and rows[loser]["merge_parent_uuid"] == "uuid-Foo"
            assert rows[loser]["version"] == 2
        assert rows["fOo"]["merge_parent_uuid"] is None and rows["fOo"]["version"] == 1
        assert _links(backend) == {
            (both, ids["seed"]),
            (only_loser, ids["seed"]),
            (both, ids["Foo"]),
            (only_loser, ids["Foo"]),
        }
        assert fetch_keywords_for_media(db, only_loser) == ["Foo", "seed"]

        before = _keyword_rows(backend)
        db._initialize_schema()
        assert _keyword_rows(backend) == before
        index = backend.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_keywords_keyword_identity'"
        ).rows
        assert "UNIQUE" in index[0]["indexdef"] and "lower(keyword)" in index[0]["indexdef"]
        with pytest.raises(BackendDatabaseError):
            backend.execute(
                "INSERT INTO keywords (keyword, uuid, last_modified, version, client_id) "
                "VALUES ('BAR', 'uuid-BAR', '2026-10-01T00:00:00Z', 1, 'seed')"
            )
    finally:
        db.close_connection()


@pytest.mark.integration
def test_postgres_keyword_identity_ensure_is_a_no_op_once_the_index_exists(pg_database_config, monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
        ensure_postgres_keyword_identity,
    )

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="913950", backend=backend)
    try:
        statements: list[str] = []
        real_execute = backend.execute

        def recording_execute(query, *args, **kwargs):
            statements.append(query)
            return real_execute(query, *args, **kwargs)

        with backend.transaction() as conn:
            monkeypatch.setattr(backend, "execute", recording_execute)
            ensure_postgres_keyword_identity(db, conn)
            monkeypatch.undo()
        assert len(statements) == 1 and "pg_indexes" in statements[0]
    finally:
        db.close_connection()
