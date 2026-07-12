from __future__ import annotations

import json
import sqlite3

from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
    normalize_media_dedupe_url,
)


def test_normalize_media_dedupe_url_http_rules() -> None:
    raw = "HTTPS://Example.COM:443/path//to///doc/?utm_source=abc&b=2&a=1#section"
    normalized = normalize_media_dedupe_url(raw)
    assert normalized == "https://example.com/path/to/doc?a=1&b=2"


def test_media_dedupe_url_candidates_include_raw_for_legacy_rows() -> None:
    raw = "https://Example.com/article/?utm_source=alpha&b=2&a=1"
    candidates = media_dedupe_url_candidates(raw)
    assert candidates[0] == "https://example.com/article?a=1&b=2"
    assert raw in candidates


def test_add_media_with_keywords_dedupes_url_variants(memory_db_factory) -> None:
    db = memory_db_factory("dedupe-url-client")
    content = "Same content body for canonical URL dedupe test."

    first_url = "https://Example.com/article/?utm_source=alpha&b=2&a=1"
    second_url = "https://example.com/article?a=1&b=2"

    media_id_1, media_uuid_1, msg_1 = db.add_media_with_keywords(
        url=first_url,
        title="Canonical URL Seed",
        media_type="document",
        content=content,
        keywords=None,
        transcription_model="whisper-test",
    )

    media_id_2, media_uuid_2, msg_2 = db.add_media_with_keywords(
        url=second_url,
        title="Canonical URL Variant",
        media_type="document",
        content=content,
        keywords=None,
        transcription_model="whisper-test",
        overwrite=False,
    )

    assert media_id_1 == media_id_2
    assert media_uuid_1 == media_uuid_2
    assert msg_1 == "Media 'Canonical URL Seed' added."
    assert "already exists" in msg_2

    row = db.execute_query("SELECT url FROM Media WHERE id = ?", (media_id_1,)).fetchone()
    assert row["url"] == "https://example.com/article?a=1&b=2"


def test_get_media_by_url_matches_variant_forms(memory_db_factory) -> None:
    db = memory_db_factory("get-media-by-url-client")

    media_id, _, _ = db.add_media_with_keywords(
        url="https://example.com/path?a=1&b=2",
        title="Lookup Variant",
        media_type="document",
        content="lookup-content",
        keywords=None,
    )

    fetched = db.get_media_by_url("https://EXAMPLE.com/path/?utm_source=x&b=2&a=1#frag")
    assert fetched is not None
    assert fetched["id"] == media_id


def test_get_media_by_urls_normalizes_candidates_in_one_query(memory_db_factory, monkeypatch) -> None:
    db = memory_db_factory("get-media-by-urls-client")
    first_id, _, _ = db.add_media_with_keywords(
        url="https://example.com/first?a=1&b=2",
        title="First batch URL",
        media_type="document",
        content="first batch body",
        keywords=None,
    )
    second_id, _, _ = db.add_media_with_keywords(
        url="https://example.com/second",
        title="Second batch URL",
        media_type="document",
        content="second batch body",
        keywords=None,
    )
    original_execute = db.execute_query
    queries: list[tuple[str, tuple]] = []

    def counting_execute(query, params=()):
        queries.append((query, tuple(params)))
        return original_execute(query, params)

    monkeypatch.setattr(db, "execute_query", counting_execute)

    fetched = db.get_media_by_urls(
        [
            "HTTPS://EXAMPLE.COM:443/first/?utm_source=x&b=2&a=1#fragment",
            "https://example.com/second/",
            "https://example.com/missing",
        ]
    )

    assert {row["id"] for row in fetched} == {first_id, second_id}
    assert len(queries) == 1
    query, params = queries[0]
    assert len(params) == 1
    assert "https://example.com/first?a=1&b=2" in json.loads(params[0])


def test_get_media_by_urls_handles_500_variant_inputs_with_one_parameter(
    memory_db_factory,
    monkeypatch,
) -> None:
    db = memory_db_factory("get-media-by-urls-limit-client")
    connection = db.get_connection()
    previous_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    original_execute = db.execute_query
    queries: list[tuple[str, tuple]] = []

    def counting_execute(query, params=()):
        queries.append((query, tuple(params)))
        return original_execute(query, params)

    monkeypatch.setattr(db, "execute_query", counting_execute)
    urls = [f"https://EXAMPLE.com/video/{index}?utm_source=batch" for index in range(500)]
    try:
        assert db.get_media_by_urls(urls) == []
    finally:
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert len(queries) == 1
    assert len(queries[0][1]) == 1
    assert len(json.loads(queries[0][1][0])) == 1_000


def test_get_media_by_urls_empty_input_avoids_query(memory_db_factory, monkeypatch) -> None:
    db = memory_db_factory("get-media-by-urls-empty-client")
    calls = 0

    def unexpected_execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("empty batch must not execute SQL")

    monkeypatch.setattr(db, "execute_query", unexpected_execute)

    assert db.get_media_by_urls([]) == []
    assert calls == 0


def test_add_media_with_keywords_identical_content_different_urls_dedupes_by_hash(
    memory_db_factory,
) -> None:
    db = memory_db_factory("dedupe-hash-client")
    content = "Identical content hash should dedupe regardless of differing non-canonical URL forms."

    first_id, first_uuid, _ = db.add_media_with_keywords(
        url="https://example.com/first",
        title="First URL",
        media_type="document",
        content=content,
        keywords=None,
    )

    second_id, second_uuid, msg = db.add_media_with_keywords(
        url="https://example.com/completely-different",
        title="Second URL",
        media_type="document",
        content=content,
        keywords=None,
        overwrite=False,
    )

    assert second_id == first_id
    assert second_uuid == first_uuid
    assert "already exists" in msg
