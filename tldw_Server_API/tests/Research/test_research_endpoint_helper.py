from __future__ import annotations

import contextlib

import pytest

from tldw_Server_API.app.api.v1.endpoints import research


@pytest.mark.unit
def test_process_and_ingest_arxiv_paper_uses_media_repository_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            return 51, "arxiv-uuid", "stored"

    fake_db = _FakeDb()
    fake_repo = _FakeRepo()

    monkeypatch.setattr(research, "fetch_arxiv_xml", lambda paper_id: f"<xml>{paper_id}</xml>")
    monkeypatch.setattr(
        research,
        "convert_xml_to_markdown",
        lambda xml: ("# Paper\n\nBody", "Paper Title", ["Ada Lovelace", "Alan Turing"], ["cs.AI", "cs.IR"]),
    )
    managed_calls = []

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(
        research,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(research, "get_media_repository", lambda db: fake_repo, raising=False)

    message = research.process_and_ingest_arxiv_paper("1234.5678", "ml,rag")

    assert message == "arXiv paper 'Paper Title' ingested successfully."
    assert fake_db.closed is True
    assert managed_calls == [
        {
            "client_id": "research_ingest",
            "initialize": False,
            "kwargs": {},
        }
    ]
    assert len(fake_repo.calls) == 1
    payload = dict(fake_repo.calls[0])
    ingestion_date = payload.pop("ingestion_date")
    assert payload == {
        "url": "https://arxiv.org/abs/1234.5678",
        "title": "Paper Title",
        "media_type": "document",
        "content": "# Paper\n\nBody",
        "keywords": ["arxiv", "cs.AI", "cs.IR", "ml", "rag"],
        "prompt": "No prompt for arXiv papers",
        "analysis_content": "arXiv paper ingested from XML",
        "transcription_model": "None",
        "author": "Ada Lovelace, Alan Turing",
    }
    assert isinstance(ingestion_date, str)


@pytest.mark.unit
def test_process_and_ingest_arxiv_paper_sanitizes_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_fetch(_paper_id):
        raise RuntimeError("arxiv fetch exploded at /private/research/cache.xml")

    monkeypatch.setattr(research, "fetch_arxiv_xml", _fail_fetch)

    message = research.process_and_ingest_arxiv_paper("1234.5678", "ml,rag")

    assert message == "Error processing arXiv paper"
    assert "arxiv fetch exploded" not in message
    assert "/private/research/cache.xml" not in message
