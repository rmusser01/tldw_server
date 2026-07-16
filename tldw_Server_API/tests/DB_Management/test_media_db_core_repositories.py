from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.legacy_reads import (
    get_latest_transcription,
    get_media_prompts,
    get_media_transcripts,
    get_specific_prompt,
    get_specific_transcript,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_state import (
    check_media_exists,
    get_unprocessed_media,
    mark_media_as_processed,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    upsert_transcript,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers import (
    get_document_version,
    import_obsidian_note_to_db,
    ingest_article_to_db_new,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.chunks_repository import (
    ChunksRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.document_versions_repository import (
    DocumentVersionsRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_lookup_repository import (
    MediaLookupRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import (
    MediaRepository,
)


@pytest.mark.unit
def test_media_database_add_media_with_keywords_delegates_to_media_repository(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-delegate")
    sentinel = (321, "repo-uuid", "delegated")
    captured: dict[str, object] = {}

    def fake_add_media_with_keywords(self, **kwargs):
        captured["session"] = self.session
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        MediaRepository,
        "add_media_with_keywords",
        fake_add_media_with_keywords,
        raising=False,
    )

    try:
        result = db.add_media_with_keywords(
            title="Delegated doc",
            content="delegate me",
            media_type="text",
            keywords=["alpha", "beta"],
            visibility="personal",
        )

        assert result == sentinel
        assert captured["session"] is db
        assert captured["kwargs"] == {
            "url": None,
            "title": "Delegated doc",
            "media_type": "text",
            "content": "delegate me",
            "keywords": ["alpha", "beta"],
            "prompt": None,
            "analysis_content": None,
            "safe_metadata": None,
            "source_hash": None,
            "transcription_model": None,
            "author": None,
            "ingestion_date": None,
            "overwrite": False,
            "chunk_options": None,
            "chunks": None,
            "visibility": "personal",
            "owner_user_id": None,
        }
    finally:
        db.close_connection()


@pytest.mark.unit
def test_ingest_article_wrapper_uses_media_repository(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="article-wrapper")
    sentinel = (98, "article-uuid", "article delegated")
    captured: dict[str, object] = {}

    def fake_add_media_with_keywords(self, **kwargs):
        captured["session"] = self.session
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        MediaRepository,
        "add_media_with_keywords",
        fake_add_media_with_keywords,
        raising=False,
    )
    monkeypatch.setattr(
        db,
        "add_media_with_keywords",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy shim should not be used")),
    )

    try:
        result = ingest_article_to_db_new(
            db,
            url="https://example.com/article",
            title="Example Article",
            content="Article body",
            author="Author",
            keywords=["alpha"],
            summary="Summary",
            ingestion_date="2024-01-02T03:04:05Z",
            custom_prompt="Prompt",
            overwrite=True,
        )

        assert result == sentinel
        assert captured["session"] is db
        assert captured["kwargs"] == {
            "url": "https://example.com/article",
            "title": "Example Article",
            "media_type": "article",
            "content": "Article body",
            "keywords": ["alpha"],
            "prompt": "Prompt",
            "analysis_content": "Summary",
            "author": "Author",
            "ingestion_date": "2024-01-02T03:04:05Z",
            "overwrite": True,
        }
    finally:
        db.close_connection()


@pytest.mark.unit
def test_import_obsidian_note_wrapper_uses_media_repository(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="obsidian-wrapper")
    sentinel = (77, "obsidian-uuid", "obsidian delegated")
    captured: dict[str, object] = {}

    def fake_add_media_with_keywords(self, **kwargs):
        captured["session"] = self.session
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        MediaRepository,
        "add_media_with_keywords",
        fake_add_media_with_keywords,
        raising=False,
    )
    monkeypatch.setattr(
        db,
        "add_media_with_keywords",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy shim should not be used")),
    )

    try:
        result = import_obsidian_note_to_db(
            db,
            {
                "title": "Daily Note",
                "content": "Note body",
                "tags": ["tag-a", 7, None],
                "frontmatter": {"author": "Jane", "status": "draft"},
                "file_created_date": "2024-01-02T03:04:05Z",
                "overwrite": True,
            },
        )

        assert result == sentinel
        assert captured["session"] is db
        assert captured["kwargs"]["url"] == "obsidian://note/Daily Note"
        assert captured["kwargs"]["title"] == "Daily Note"
        assert captured["kwargs"]["media_type"] == "obsidian_note"
        assert captured["kwargs"]["content"] == "Note body"
        assert captured["kwargs"]["keywords"] == ["tag-a", "7"]
        assert captured["kwargs"]["author"] == "Jane"
        assert captured["kwargs"]["prompt"] == "Obsidian Frontmatter"
        assert "author: Jane" in str(captured["kwargs"]["analysis_content"])
        assert captured["kwargs"]["ingestion_date"] == "2024-01-02T03:04:05Z"
        assert captured["kwargs"]["overwrite"] is True
    finally:
        db.close_connection()


@pytest.mark.unit
def test_get_document_version_wrapper_uses_document_versions_repository(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="version-wrapper")
    sentinel = {"media_id": 11, "version_number": 2, "content": "delegated"}
    captured: dict[str, object] = {}

    def fake_get(self, *, media_id, version_number=None, include_content=True):
        captured["session"] = self.session
        captured["kwargs"] = {
            "media_id": media_id,
            "version_number": version_number,
            "include_content": include_content,
        }
        return sentinel

    monkeypatch.setattr(
        DocumentVersionsRepository,
        "get",
        fake_get,
        raising=False,
    )

    try:
        result = get_document_version(
            db,
            media_id=11,
            version_number=2,
            include_content=False,
        )

        assert result == sentinel
        assert captured["session"] is db
        assert captured["kwargs"] == {
            "media_id": 11,
            "version_number": 2,
            "include_content": False,
        }
    finally:
        db.close_connection()


@pytest.mark.integration
def test_legacy_read_wrappers_round_trip_transcripts_and_prompts() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-read-wrappers")
    media_repo = MediaRepository.from_legacy_db(db)
    versions_repo = DocumentVersionsRepository.from_legacy_db(db)
    try:
        media_id, _media_uuid, _msg = media_repo.add_text_media(
            title="Prompted doc",
            content="v1",
            media_type="text",
        )
        version = versions_repo.create(
            media_id=media_id,
            content="v2",
            prompt="Prompt 2",
            analysis_content="Analysis 2",
        )
        transcript = upsert_transcript(
            db,
            media_id=media_id,
            transcription='{"text": "Transcript text"}',
            whisper_model="base",
        )

        prompts = get_media_prompts(db, media_id)
        transcripts = get_media_transcripts(db, media_id)
        latest_transcript = get_latest_transcription(db, media_id)
        specific_prompt = get_specific_prompt(db, version["uuid"])
        specific_transcript = get_specific_transcript(db, transcript["uuid"])

        assert [item["content"] for item in prompts] == ["Prompt 2"]
        assert len(transcripts) == 1
        assert transcripts[0]["uuid"] == transcript["uuid"]
        assert latest_transcript == "Transcript text"
        assert specific_prompt == "Prompt 2"
        assert specific_transcript is not None
        assert specific_transcript["uuid"] == transcript["uuid"]
    finally:
        db.close_connection()


@pytest.mark.integration
def test_legacy_state_wrappers_round_trip_exists_and_processing_flags() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="legacy-state-wrappers")
    media_repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, media_uuid, _msg = media_repo.add_text_media(
            title="Stateful doc",
            content="needs vectors",
            media_type="text",
            url="https://example.com/stateful",
        )

        by_id = check_media_exists(db, media_id=media_id)
        by_url = check_media_exists(db, url="https://example.com/stateful/")
        unprocessed_before = get_unprocessed_media(db)

        mark_media_as_processed(db, media_id)

        unprocessed_after = get_unprocessed_media(db)
        media_row = db.execute_query(
            "SELECT vector_processing, chunking_status, version, client_id FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()
        sync_row = db.execute_query(
            """
            SELECT operation, version, client_id
            FROM sync_log
            WHERE entity = 'Media' AND entity_uuid = ?
            ORDER BY change_id DESC
            LIMIT 1
            """,
            (media_uuid,),
        ).fetchone()

        assert by_id == media_id
        assert by_url == media_id
        assert any(row["id"] == media_id and row["uuid"] == media_uuid for row in unprocessed_before)
        assert all(row["id"] != media_id for row in unprocessed_after)
        assert media_row is not None
        assert media_row["vector_processing"] == 1
        assert media_row["chunking_status"] == "completed"
        assert media_row["version"] == 2
        assert media_row["client_id"] == "legacy-state-wrappers"
        assert sync_row is not None
        assert sync_row["operation"] == "update"
        assert sync_row["version"] == 2
        assert sync_row["client_id"] == "legacy-state-wrappers"
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_repository_add_text_media_creates_row() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-repo")
    repo = MediaRepository.from_legacy_db(db)
    try:
        media_id, media_uuid, message = repo.add_text_media(
            title="Repo doc",
            content="hello",
            media_type="text",
            keywords=["alpha"],
        )

        assert isinstance(media_id, int)
        assert isinstance(media_uuid, str)
        assert "added" in message.lower()
    finally:
        db.close_connection()


@pytest.mark.integration
def test_document_versions_repository_returns_latest_version() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="doc-repo")
    media_repo = MediaRepository.from_legacy_db(db)
    versions_repo = DocumentVersionsRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Versioned doc",
            content="v1",
            media_type="text",
        )
        versions_repo.create(media_id=media_id, content="v2", prompt="p2", analysis_content="a2")

        latest = versions_repo.get(media_id=media_id, version_number=None, include_content=True)

        assert latest is not None
        assert latest["version_number"] == 2
        assert latest["content"] == "v2"
    finally:
        db.close_connection()


@pytest.mark.integration
def test_document_versions_repository_create_uses_transaction_context(monkeypatch) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="doc-repo-transaction")
    media_repo = MediaRepository.from_legacy_db(db)
    versions_repo = DocumentVersionsRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Versioned doc",
            content="v1",
            media_type="text",
        )
        calls: dict[str, object] = {"count": 0}
        original_transaction = db.transaction

        @contextmanager
        def _tracking_transaction():
            calls["count"] = int(calls["count"]) + 1
            with original_transaction() as conn:
                calls["conn"] = conn
                yield conn

        monkeypatch.setattr(db, "transaction", _tracking_transaction)
        monkeypatch.setattr(
            db,
            "get_connection",
            lambda: (_ for _ in ()).throw(AssertionError("create() should not call get_connection")),
        )

        version = versions_repo.create(media_id=media_id, content="v2")

        assert version["version_number"] == 2
        assert calls["count"] == 1
        assert calls["conn"] is not None
    finally:
        db.close_connection()


@pytest.mark.integration
def test_chunks_repository_batch_insert_generates_unique_chunk_ids() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="chunk-repo")
    media_repo = MediaRepository.from_legacy_db(db)
    chunks_repo = ChunksRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Chunked doc",
            content="chunk source",
            media_type="text",
        )

        inserted = chunks_repo.batch_insert(
            media_id,
            [
                {"text": "chunk-1", "metadata": {"start_index": 0, "end_index": 5}},
                {"text": "chunk-2", "metadata": {"start_index": 6, "end_index": 11}},
            ],
        )

        assert inserted == 2

        rows = db.execute_query(
            "SELECT chunk_id FROM MediaChunks WHERE media_id = ? ORDER BY id",
            (media_id,),
        ).fetchall()
        chunk_ids = [row["chunk_id"] for row in rows]
        assert len(chunk_ids) == 2
        assert len(set(chunk_ids)) == 2
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_source_projection_is_explicit_bounded_and_uses_current_precedence(
    monkeypatch,
) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-projection")
    media_repo = MediaRepository.from_legacy_db(db)
    versions_repo = DocumentVersionsRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Projection title",
            content="media fallback body",
            media_type="text",
        )
        versions_repo.create(
            media_id=media_id,
            content="document fallback body",
        )
        normalized_transcript = '{"text":"short transcript","segments":["' + ("x" * 50_000) + '"]}'
        upsert_transcript(
            db,
            media_id=media_id,
            transcription=normalized_transcript,
            whisper_model="base",
        )

        queries: list[tuple[str, tuple | None]] = []
        original_execute = db.execute_query

        def recording_execute(query, params=None, **kwargs):
            queries.append((str(query), params))
            return original_execute(query, params, **kwargs)

        monkeypatch.setattr(db, "execute_query", recording_execute)

        row = lookup_repo.source_projection_by_id(media_id, max_chars=20)

        assert row == {
            "id": media_id,
            "source_text": "short transcript",
            "source_invalid": False,
        }
        assert len(row["source_text"]) <= 21
        assert len(queries) == 1
        sql = " ".join(queries[0][0].split()).lower()
        assert "select *" not in sql
        assert "vector_embedding" not in sql
        assert "visualdocuments" not in sql
        assert "mediafiles" not in sql
        assert "image" not in sql
        assert "substr" in sql
        assert "limit 1" in sql
        assert "dv.deleted = 0" in sql
        assert "t.deleted = 0" in sql
    finally:
        db.close_connection()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("transcription", "legacy_text", "projected_text"),
    [
        ('{"segments":[{"text":"metadata only"}]}', "", "document fallback body"),
        ('{"text":null,"segments":[]}', "", "document fallback body"),
        ('{"text":123}', "123", "123"),
        ('{"text":true}', "True", "True"),
        ('{"text":', '{"text":', '{"text":'),
        ('["segment"]', '["segment"]', '["segment"]'),
    ],
)
def test_sqlite_media_source_projection_matches_legacy_transcript_normalization(
    transcription: str,
    legacy_text: str,
    projected_text: str,
) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-normalization")
    media_repo = MediaRepository.from_legacy_db(db)
    versions_repo = DocumentVersionsRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Normalized transcript precedence",
            content="media fallback body",
            media_type="text",
        )
        versions_repo.create(
            media_id=media_id,
            content="document fallback body",
        )
        upsert_transcript(
            db,
            media_id=media_id,
            transcription=transcription,
            whisper_model="base",
        )

        assert get_latest_transcription(db, media_id) == legacy_text
        assert lookup_repo.source_projection_by_id(media_id, max_chars=100) == {
            "id": media_id,
            "source_text": projected_text,
            "source_invalid": False,
        }
    finally:
        db.close_connection()


@pytest.mark.unit
def test_postgres_media_source_projection_extracts_normalized_text_before_bound() -> None:
    captured: dict[str, object] = {}

    class _Cursor:
        @staticmethod
        def fetchone():
            return {
                "id": 9,
                "source_text": "normalized text",
                "source_invalid": False,
            }

    class _PostgresSession:
        backend_type = SimpleNamespace(name="POSTGRESQL")

        @staticmethod
        def execute_query(query, params, **kwargs):
            captured["query"] = str(query)
            captured["params"] = params
            captured["kwargs"] = kwargs
            return _Cursor()

    row = MediaLookupRepository(_PostgresSession()).source_projection_by_id(
        9,
        max_chars=20,
        owner_user_id="owner-1",
    )

    assert row == {"id": 9, "source_text": "normalized text", "source_invalid": False}
    sql = " ".join(str(captured["query"]).split()).lower()
    assert "left(ltrim(t.transcription), 1) = '{'" in sql
    assert "public.tldw_try_extract_normalized_transcript_text" in sql
    assert "( t.transcription )" in sql
    assert "cast(t.transcription as jsonb)" not in sql
    assert "substr( coalesce(" in sql
    assert captured["params"] == (21, 9, "owner-1")
    assert captured["kwargs"] == {"log_errors": False}


@pytest.mark.unit
def test_postgres_media_source_projection_requires_owner_scope() -> None:
    class _PostgresSession:
        backend_type = SimpleNamespace(name="POSTGRESQL")

        @staticmethod
        def execute_query(*_args, **_kwargs):
            raise AssertionError("unscoped PostgreSQL projection must not query")

    with pytest.raises(InputError, match="owner_user_id"):
        MediaLookupRepository(_PostgresSession()).source_projection_by_id(
            9,
            max_chars=20,
        )


@pytest.mark.integration
def test_sqlite_media_source_projection_marks_nul_text_invalid() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-nul")
    media_repo = MediaRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="NUL source",
            content="valid",
            media_type="text",
        )
        db.execute_query(
            """
            UPDATE DocumentVersions
            SET content = ?, version = version + 1
            WHERE media_id = ?
            """,
            ("prefix\0" + ("secret" * 1000), media_id),
            commit=True,
        )

        projection = lookup_repo.source_projection_by_id(media_id, max_chars=20)

        assert projection is not None
        assert projection["source_invalid"] is True
    finally:
        db.close_connection()


@pytest.mark.integration
def test_sqlite_media_source_projection_marks_missing_text_valid() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-empty")
    media_repo = MediaRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Empty source",
            content="placeholder",
            media_type="text",
        )
        db.execute_query(
            "DELETE FROM DocumentVersions WHERE media_id = ?",
            (media_id,),
            commit=True,
        )
        db.execute_query(
            "UPDATE Media SET content = NULL, version = version + 1 WHERE id = ?",
            (media_id,),
            commit=True,
        )

        assert lookup_repo.source_projection_by_id(media_id, max_chars=20) == {
            "id": media_id,
            "source_text": None,
            "source_invalid": False,
        }
    finally:
        db.close_connection()


@pytest.mark.integration
def test_sqlite_media_source_projection_decode_failure_never_logs_source() -> None:
    secret = "SECRET"
    messages: list[str] = []
    db = MediaDatabase(db_path=":memory:", client_id="media-source-invalid-utf8")
    media_repo = MediaRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Invalid UTF-8 source",
            content="placeholder",
            media_type="text",
        )
        db.execute_query(
            "DELETE FROM DocumentVersions WHERE media_id = ?",
            (media_id,),
            commit=True,
        )
        db.execute_query(
            """
            UPDATE Media
            SET content = CAST(X'666F6F805345435245545F4D454449415F465241474D454E54' AS TEXT),
                version = version + 1
            WHERE id = ?
            """,
            (media_id,),
            commit=True,
        )

        sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
        try:
            with pytest.raises(DatabaseError) as exc_info:
                lookup_repo.source_projection_by_id(media_id, max_chars=20)
        finally:
            logger.remove(sink_id)

        assert str(exc_info.value) == "Media source projection failed."
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert secret not in repr(exc_info.value)
        assert secret not in "\n".join(messages)
    finally:
        db.close_connection()


@pytest.mark.unit
def test_media_source_projection_repository_failure_is_fixed_and_redacted() -> None:
    secret = "PRIVATE_TRANSCRIPT_FRAGMENT"
    messages: list[str] = []

    class _FailingSession:
        backend_type = SimpleNamespace(name="POSTGRESQL")

        @staticmethod
        def execute_query(_query, _params, **kwargs):
            assert kwargs == {"log_errors": False}
            raise RuntimeError(secret)

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(DatabaseError) as exc_info:
            MediaLookupRepository(_FailingSession()).source_projection_by_id(
                9,
                max_chars=20,
                owner_user_id="owner-1",
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Media source projection failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)
    assert secret not in "\n".join(messages)


@pytest.mark.integration
def test_media_source_projection_hides_deleted_and_trash_parent_rows() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-active-only")
    media_repo = MediaRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Active only",
            content="body",
            media_type="text",
        )
        assert lookup_repo.source_projection_by_id(media_id, max_chars=20) is not None

        db.execute_query(
            "UPDATE Media SET is_trash = 1, version = version + 1 WHERE id = ?",
            (media_id,),
            commit=True,
        )
        assert lookup_repo.source_projection_by_id(media_id, max_chars=20) is None

        db.execute_query(
            "UPDATE Media SET is_trash = 0, deleted = 1, version = version + 1 WHERE id = ?",
            (media_id,),
            commit=True,
        )
        assert lookup_repo.source_projection_by_id(media_id, max_chars=20) is None
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_source_projection_enforces_explicit_owner_scope() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="42")
    media_repo = MediaRepository.from_legacy_db(db)
    lookup_repo = MediaLookupRepository.from_legacy_db(db)
    try:
        media_id, _, _ = media_repo.add_text_media(
            title="Owner local",
            content="private body",
            media_type="text",
        )

        assert (
            lookup_repo.source_projection_by_id(
                media_id,
                max_chars=20,
                owner_user_id="42",
            )
            is not None
        )
        assert (
            lookup_repo.source_projection_by_id(
                media_id,
                max_chars=20,
                owner_user_id="7",
            )
            is None
        )
    finally:
        db.close_connection()


@pytest.mark.parametrize("max_chars", [True, 0, -1, "20"])
def test_media_source_projection_rejects_invalid_character_budget(max_chars) -> None:
    db = MediaDatabase(db_path=":memory:", client_id="media-source-invalid-budget")
    try:
        with pytest.raises(InputError):
            MediaLookupRepository.from_legacy_db(db).source_projection_by_id(
                1,
                max_chars=max_chars,
            )
    finally:
        db.close_connection()
