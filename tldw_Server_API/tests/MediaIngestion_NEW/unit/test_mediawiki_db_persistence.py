from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki import Media_Wiki


def test_process_single_item_uses_media_repository_for_media_db_sessions(monkeypatch):
    class _MediaDb:
        backend = object()
        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 71, "wiki-uuid", "stored"

    media_db = _MediaDb()
    fake_repo = _FakeRepo()
    managed_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield media_db
        finally:
            media_db.close_connection()

    monkeypatch.setattr(Media_Wiki, "optimized_chunking", lambda content, options: [{"text": content, "metadata": {}}])
    monkeypatch.setattr(Media_Wiki, "managed_media_database", _fake_managed_media_database, raising=False)
    monkeypatch.setattr(
        Media_Wiki,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(Media_Wiki, "get_media_repository", lambda db: fake_repo, raising=False)

    result = Media_Wiki.process_single_item(
        content="Wiki body",
        title="Test Page",
        wiki_name="ExampleWiki",
        chunk_options={},
        item={
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "namespace": 0,
            "page_id": 123,
            "revision_id": 456,
        },
        store_to_db=True,
        store_to_vector_db=False,
    )

    assert result["status"] == "Success"
    assert result["media_id"] == 71
    assert media_db.closed is True
    assert managed_calls == [
        {
            "client_id": "mediawiki_import",
            "initialize": True,
            "kwargs": {},
        }
    ]
    assert fake_repo.calls == [
        {
            "url": "mediawiki:ExampleWiki:Test%20Page",
            "title": "Test Page",
            "media_type": "mediawiki_page",
            "content": "Wiki body",
            "keywords": ["mediawiki", "ExampleWiki", "page"],
            "prompt": "",
            "analysis_content": "",
            "transcription_model": "N/A",
            "author": "MediaWiki",
            "ingestion_date": "2024-01-02",
        }
    ]


def test_import_mediawiki_dump_reuses_single_managed_media_database(
    monkeypatch,
    tmp_path,
):
    class _MediaDb:
        backend = object()

        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 71, "wiki-uuid", "stored"

    media_db = _MediaDb()
    fake_repo = _FakeRepo()
    managed_calls: list[dict[str, object]] = []
    checkpoint_saves: list[int] = []
    dump_path = tmp_path / "dump.xml"
    dump_path.write_text("<mediawiki />")

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield media_db
        finally:
            media_db.close_connection()

    items = [
        {
            "title": "Page One",
            "content": "Body one",
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "namespace": 0,
            "page_id": 123,
            "revision_id": 456,
        },
        {
            "title": "Page Two",
            "content": "Body two",
            "timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc),
            "namespace": 0,
            "page_id": 124,
            "revision_id": 457,
        },
    ]

    monkeypatch.setattr(Media_Wiki, "sanitize_wiki_name", lambda name: name)
    monkeypatch.setattr(Media_Wiki, "validate_file_path", lambda *args, **kwargs: dump_path)
    monkeypatch.setattr(Media_Wiki, "count_pages", lambda *args, **kwargs: len(items))
    monkeypatch.setattr(Media_Wiki, "parse_mediawiki_dump", lambda *args, **kwargs: iter(items))
    monkeypatch.setattr(Media_Wiki, "optimized_chunking", lambda content, options: [{"text": content, "metadata": {}}])
    monkeypatch.setattr(Media_Wiki, "load_checkpoint", lambda path: 0)
    monkeypatch.setattr(Media_Wiki, "save_checkpoint", lambda path, page_id: checkpoint_saves.append(page_id))
    monkeypatch.setattr(Media_Wiki, "get_safe_checkpoint_path", lambda wiki_name: tmp_path / f"{wiki_name}.json")
    monkeypatch.setattr(Media_Wiki, "managed_media_database", _fake_managed_media_database, raising=False)
    monkeypatch.setattr(
        Media_Wiki,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(Media_Wiki, "get_media_repository", lambda db: fake_repo, raising=False)

    results = list(
        Media_Wiki.import_mediawiki_dump(
            file_path=str(dump_path),
            wiki_name="ExampleWiki",
            store_to_db=True,
            store_to_vector_db=False,
            allowed_dir=tmp_path,
        )
    )

    assert results[-1]["type"] == "summary"
    assert len(managed_calls) == 1
    assert managed_calls[0] == {
        "client_id": "mediawiki_import",
        "initialize": True,
        "kwargs": {},
    }
    assert len(fake_repo.calls) == 2
    assert checkpoint_saves == [123, 124]
    assert media_db.closed is True


@pytest.mark.parametrize(
    ("checkpoint_identity_scope", "expected_checkpoint_scope"),
    [
        pytest.param(None, "request-user-42", id="legacy-user-fallback"),
        pytest.param(
            '{"org_id":7,"team_id":9,"user_id":"request-user-42"}',
            '{"org_id":7,"team_id":9,"user_id":"request-user-42"}',
            id="request-tenant-scope",
        ),
    ],
)
def test_import_mediawiki_dump_uses_injected_writer_without_managed_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint_identity_scope: str | None,
    expected_checkpoint_scope: str,
) -> None:
    """Use request persistence and prefer tenant scope for checkpoint state."""
    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 72, "wiki-uuid", "stored"

    fake_repo = _FakeRepo()
    checkpoint_saves: list[int] = []
    checkpoint_scopes: list[tuple[str, str | None]] = []
    dump_path = tmp_path / "dump.xml"
    dump_path.write_text("<mediawiki />")
    items = [
        {
            "title": "Request Page",
            "content": "Scoped body",
            "timestamp": datetime(2024, 1, 4, tzinfo=timezone.utc),
            "namespace": 0,
            "page_id": 125,
            "revision_id": 458,
        },
    ]

    monkeypatch.setattr(Media_Wiki, "sanitize_wiki_name", lambda name: name)
    monkeypatch.setattr(Media_Wiki, "validate_file_path", lambda *args, **kwargs: dump_path)
    monkeypatch.setattr(Media_Wiki, "count_pages", lambda *args, **kwargs: len(items))
    monkeypatch.setattr(
        Media_Wiki,
        "parse_mediawiki_dump",
        lambda *args, **kwargs: iter(items),
    )
    monkeypatch.setattr(
        Media_Wiki,
        "optimized_chunking",
        lambda content, options: [{"text": content, "metadata": {}}],
    )
    monkeypatch.setattr(Media_Wiki, "load_checkpoint", lambda path: 0)
    monkeypatch.setattr(
        Media_Wiki,
        "save_checkpoint",
        lambda path, page_id: checkpoint_saves.append(page_id),
    )
    def _checkpoint_path(
        wiki_name: str,
        checkpoint_dir: Path | None = None,
        *,
        identity_scope: str | None = None,
    ) -> Path:
        checkpoint_scopes.append((wiki_name, identity_scope))
        return tmp_path / f"{wiki_name}.json"

    monkeypatch.setattr(Media_Wiki, "get_safe_checkpoint_path", _checkpoint_path)
    monkeypatch.setattr(
        Media_Wiki,
        "managed_media_database",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("request-scoped importer must not open fallback media DB")
        ),
        raising=False,
    )

    results = list(
        Media_Wiki.import_mediawiki_dump(
            file_path=str(dump_path),
            wiki_name="ExampleWiki",
            store_to_db=True,
            store_to_vector_db=False,
            allowed_dir=tmp_path,
            media_writer=fake_repo,
            vector_user_id="request-user-42",
            checkpoint_identity_scope=checkpoint_identity_scope,
        )
    )

    assert results[-1]["type"] == "summary"
    assert fake_repo.calls[0]["title"] == "Request Page"
    assert checkpoint_saves == [125]
    assert checkpoint_scopes == [("ExampleWiki", expected_checkpoint_scope)]


def test_checkpoint_paths_are_isolated_by_request_identity(tmp_path: Path) -> None:
    """Hash user and tenant identities into distinct checkpoint paths."""
    legacy = Media_Wiki.get_safe_checkpoint_path("Example Wiki", checkpoint_dir=tmp_path)
    first = Media_Wiki.get_safe_checkpoint_path(
        "Example Wiki",
        checkpoint_dir=tmp_path,
        identity_scope="request-user-42",
    )
    second = Media_Wiki.get_safe_checkpoint_path(
        "Example Wiki",
        checkpoint_dir=tmp_path,
        identity_scope="request-user-99",
    )
    first_tenant = Media_Wiki.get_safe_checkpoint_path(
        "Example Wiki",
        checkpoint_dir=tmp_path,
        identity_scope='{"org_id":7,"team_id":9,"user_id":"request-user-42"}',
    )
    second_tenant = Media_Wiki.get_safe_checkpoint_path(
        "Example Wiki",
        checkpoint_dir=tmp_path,
        identity_scope='{"org_id":8,"team_id":9,"user_id":"request-user-42"}',
    )

    assert legacy.name == "Example_Wiki_import_checkpoint.json"
    assert first != legacy
    assert second != legacy
    assert first != second
    assert first_tenant != second_tenant
    assert "request-user-42" not in first.name
    assert "request-user-99" not in second.name


def test_import_mediawiki_dump_sanitizes_unexpected_import_error(monkeypatch, tmp_path):
    dump_path = tmp_path / "dump.xml"
    dump_path.write_text("<mediawiki />")

    monkeypatch.setattr(Media_Wiki, "sanitize_wiki_name", lambda name: name)
    monkeypatch.setattr(Media_Wiki, "validate_file_path", lambda *args, **kwargs: dump_path)
    monkeypatch.setattr(Media_Wiki, "get_mediawiki_import_config", lambda: {"chunking": {}})
    monkeypatch.setattr(
        Media_Wiki,
        "count_pages",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("mediawiki parser failed at /private/wiki/dump.xml")
        ),
    )

    results = list(
        Media_Wiki.import_mediawiki_dump(
            file_path=str(dump_path),
            wiki_name="ExampleWiki",
            store_to_db=False,
            store_to_vector_db=False,
            allowed_dir=tmp_path,
        )
    )

    assert results == [{"type": "error", "message": "Error during import"}]
    assert "mediawiki parser failed" not in str(results)
    assert "/private/wiki/dump.xml" not in str(results)


def test_process_single_item_sanitizes_unexpected_processing_error(monkeypatch):
    monkeypatch.setattr(
        Media_Wiki,
        "optimized_chunking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("chunker failed at /private/wiki/chunks")
        ),
    )

    result = Media_Wiki.process_single_item(
        content="Wiki body",
        title="Broken Page",
        wiki_name="ExampleWiki",
        chunk_options={},
        item={
            "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "namespace": 0,
            "page_id": 123,
            "revision_id": 456,
        },
        store_to_db=False,
        store_to_vector_db=False,
    )

    assert result["status"] == "Error"
    assert result["error_message"] == "MediaWiki item processing failed"
    assert result["message"] == "Failed to process MediaWiki item"
    assert "chunker failed" not in str(result)
    assert "/private/wiki/chunks" not in str(result)
