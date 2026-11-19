from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import paper_search


@pytest.mark.unit
def test_ingest_paper_search_media_uses_media_repository_for_media_db_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MediaDb:
        backend = object()

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 31, "paper-uuid", "stored"

    fake_repo = _FakeRepo()
    media_db = _MediaDb()

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository", lambda db: fake_repo, raising=False)

    result = paper_search._ingest_paper_search_media(
        media_db=media_db,
        url="arxiv:1706.03762",
        title="Attention Is All You Need",
        media_type="pdf",
        content="paper body",
        keywords=["transformers"],
        safe_metadata='{"source":"arxiv"}',
        overwrite=False,
    )

    assert result == (31, "paper-uuid", "stored")
    assert fake_repo.calls == [
        {
            "url": "arxiv:1706.03762",
            "title": "Attention Is All You Need",
            "media_type": "pdf",
            "content": "paper body",
            "keywords": ["transformers"],
            "safe_metadata": '{"source":"arxiv"}',
            "overwrite": False,
        }
    ]


@pytest.mark.unit
def test_ingest_paper_search_media_falls_back_to_direct_writer_when_repository_wrap_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DirectWriter:
        backend = object()

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 41, "direct-uuid", "stored"

    direct_writer = _DirectWriter()

    def _raise_not_a_db(_db):
        raise TypeError("already a writer")

    monkeypatch.setattr("tldw_Server_API.app.core.DB_Management.media_db.api.get_media_repository", _raise_not_a_db, raising=False)

    result = paper_search._ingest_paper_search_media(
        media_db=direct_writer,
        url="doi:10.1000/example",
        title="Direct writer",
        media_type="pdf",
        content="body",
        keywords=["science"],
        overwrite=False,
    )

    assert result == (41, "direct-uuid", "stored")
    assert direct_writer.calls == [
        {
            "url": "doi:10.1000/example",
            "title": "Direct writer",
            "media_type": "pdf",
            "content": "body",
            "keywords": ["science"],
            "overwrite": False,
        }
    ]
