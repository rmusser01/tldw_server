from __future__ import annotations

import contextlib

import pytest

from tldw_Server_API.app.services import document_processing_service as dps


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_documents_store_in_db_uses_media_repository_api(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("Alpha document body", encoding="utf-8")

    class _FakeDb:
        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 42, "doc-repo-uuid", "stored"

    fake_db = _FakeDb()
    fake_repo = _FakeRepo()
    fake_chunks = [{"text": "Alpha document body", "chunk_type": "text"}]

    monkeypatch.setattr(dps, "_ensure_placeholder_enabled", lambda: None)
    managed_calls = []

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, *, db_path=None, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "db_path": db_path,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(dps, "managed_media_database", _fake_managed_media_database, raising=False)
    monkeypatch.setattr(dps, "get_user_media_db_path", lambda _user_id: str(tmp_path / "media.db"))
    monkeypatch.setattr(dps, "build_plaintext_chunks", lambda *args, **kwargs: fake_chunks)
    monkeypatch.setattr(dps, "get_media_repository", lambda db: fake_repo, raising=False)

    result = await dps.process_documents(
        doc_urls=None,
        doc_files=[str(document_path)],
        api_name=None,
        api_key=None,
        custom_prompt_input=None,
        system_prompt_input=None,
        use_cookies=False,
        cookies=None,
        keep_original=True,
        custom_keywords=["alpha", "beta"],
        chunk_method="sentences",
        max_chunk_size=256,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language="en",
        store_in_db=True,
        overwrite_existing=False,
        custom_title="Stored doc",
    )

    assert result["status"] == "success"
    assert result["results"][0]["db_id"] == 42
    assert fake_db.closed is True
    assert managed_calls == [
        {
            "client_id": "document_processing_service",
            "db_path": str(tmp_path / "media.db"),
            "initialize": False,
            "kwargs": {},
        }
    ]
    assert fake_repo.calls == [
        {
            "url": str(document_path),
            "title": "Stored doc",
            "media_type": "document",
            "content": "Alpha document body",
            "keywords": ["alpha", "beta"],
            "prompt": None,
            "analysis_content": "",
            "safe_metadata": '{"title": "Stored doc", "source": "document", "url": "'
            + str(document_path)
            + '"}',
            "transcription_model": "document-import",
            "author": None,
            "ingestion_date": None,
            "overwrite": False,
            "chunks": fake_chunks,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_documents_file_failure_progress_does_not_leak_raw_exception(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("Alpha document body", encoding="utf-8")
    leak = "database exploded /tmp/doc-secret-token"

    monkeypatch.setattr(dps, "_ensure_placeholder_enabled", lambda: None)
    monkeypatch.setattr(
        dps,
        "_store_document_in_db",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(leak)),
    )

    result = await dps.process_documents(
        doc_urls=None,
        doc_files=[str(document_path)],
        api_name=None,
        api_key=None,
        custom_prompt_input=None,
        system_prompt_input=None,
        use_cookies=False,
        cookies=None,
        keep_original=True,
        custom_keywords=[],
        chunk_method="sentences",
        max_chunk_size=256,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language="en",
        store_in_db=True,
    )

    assert result["status"] == "partial"
    assert result["results"][0]["error"] == leak
    rendered_progress = "\n".join(result["progress"])
    assert "Failed to process file 1" in rendered_progress
    assert "database exploded" not in rendered_progress
    assert "/tmp/doc-secret-token" not in rendered_progress


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_documents_summarization_progress_does_not_leak_raw_exception(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("Alpha document body", encoding="utf-8")
    leak = "summarizer exploded /tmp/summarizer-secret-token"

    monkeypatch.setattr(dps, "_ensure_placeholder_enabled", lambda: None)
    monkeypatch.setattr(dps, "load_prompt", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        dps,
        "improved_chunking_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(leak)),
    )

    result = await dps.process_documents(
        doc_urls=None,
        doc_files=[str(document_path)],
        api_name="mock-provider",
        api_key=None,
        custom_prompt_input=None,
        system_prompt_input=None,
        use_cookies=False,
        cookies=None,
        keep_original=True,
        custom_keywords=[],
        chunk_method="sentences",
        max_chunk_size=256,
        chunk_overlap=0,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        chunk_language="en",
        store_in_db=False,
    )

    assert result["status"] == "success"
    assert result["results"][0]["summary"] == "Summary generation failed"
    rendered_progress = "\n".join(result["progress"])
    assert "Summarization failed" in rendered_progress
    assert "summarizer exploded" not in rendered_progress
    assert "/tmp/summarizer-secret-token" not in rendered_progress
