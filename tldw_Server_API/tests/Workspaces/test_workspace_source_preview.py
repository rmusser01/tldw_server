"""Tests for reusable bounded workspace source previews."""
from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import MediaRepository
from tldw_Server_API.app.core.Workspaces.source_preview import (
    build_workspace_source_preview,
)

pytestmark = pytest.mark.unit


class _MediaDB:
    def __init__(self) -> None:
        self.range_calls: list[tuple[int, int, int]] = []
        self.content = "abcdefghijklmnopqrstuvwxyz"
        self.chunks = [
            {
                "chunk_index": index,
                "uuid": f"chunk-{index}",
                "chunk_text": f"Chunk {index}",
                "start_char": index * 10,
                "end_char": (index * 10) + 9,
                "chunk_type": "text",
                "deleted": 0,
            }
            for index in range(20)
        ]
        self.chunks[10]["deleted"] = 1

    def get_media_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        _ = (include_deleted, include_trash)
        if media_id != 5:
            return None
        return {"id": media_id, "content": self.content}

    def get_unvectorized_chunks_in_range(
        self,
        media_id: int,
        start_index: int,
        end_index: int,
    ) -> list[dict[str, Any]]:
        self.range_calls.append((media_id, start_index, end_index))
        return [
            dict(chunk)
            for chunk in self.chunks
            if not chunk["deleted"]
            and start_index <= int(chunk["chunk_index"]) <= end_index
        ]


def _source(**overrides: Any) -> dict[str, Any]:
    source = {
        "id": "source-1",
        "media_id": 5,
        "title": "Evidence",
        "source_type": "pdf",
        "url": "https://example.test/evidence.pdf",
    }
    source.update(overrides)
    return source


def _status(**overrides: Any) -> dict[str, Any]:
    status = {
        "state": "queryable",
        "status_reason": "source_queryable",
        "readiness": {"citation_ready": True},
    }
    status.update(overrides)
    return status


def test_preview_preserves_local_response_shape_and_bounds() -> None:
    media_db = _MediaDB()

    preview = build_workspace_source_preview(
        workspace_id="workspace-alpha",
        source=_source(),
        source_status=_status(),
        media_db=media_db,
        max_chars=8,
        chunk_limit=2,
    )

    assert preview.keys() == {
        "workspace_id",
        "source_id",
        "media_id",
        "title",
        "source_type",
        "url",
        "state",
        "status_reason",
        "readiness",
        "content_available",
        "preview_mode",
        "unavailable_reason",
        "text_preview",
        "text_total_chars",
        "text_truncated",
        "snippets",
        "generated_at",
    }
    assert preview["workspace_id"] == "workspace-alpha"
    assert preview["source_id"] == "source-1"
    assert preview["media_id"] == 5
    assert preview["text_preview"] == "abcdefgh"
    assert preview["text_total_chars"] == 26
    assert preview["text_truncated"] is True
    assert preview["snippets"][0]["kind"] == "content_excerpt"
    assert [item["chunk_index"] for item in preview["snippets"][1:]] == [0, 1]
    assert media_db.range_calls == [(5, 0, 1)]


@pytest.mark.parametrize(
    ("focus_chunk_index", "chunk_limit", "expected_range", "expected_indexes"),
    [
        (10, 3, (9, 11), [9, 11]),
        (1, 5, (0, 4), [0, 1, 2, 3, 4]),
        (19, 4, (17, 20), [17, 18, 19]),
    ],
)
def test_focus_preview_fetches_centered_active_chunk_window(
    focus_chunk_index: int,
    chunk_limit: int,
    expected_range: tuple[int, int],
    expected_indexes: list[int],
) -> None:
    media_db = _MediaDB()

    preview = build_workspace_source_preview(
        workspace_id="workspace-alpha",
        source=_source(),
        source_status=_status(),
        media_db=media_db,
        max_chars=12,
        chunk_limit=chunk_limit,
        focus_chunk_index=focus_chunk_index,
    )

    assert media_db.range_calls == [(5, *expected_range)]
    assert [item["chunk_index"] for item in preview["snippets"][1:]] == expected_indexes
    assert len(preview["snippets"][1:]) <= chunk_limit


def test_negative_focus_is_rejected_before_media_access() -> None:
    media_db = _MediaDB()

    with pytest.raises(ValueError, match="focus_chunk_index"):
        build_workspace_source_preview(
            workspace_id="workspace-alpha",
            source=_source(),
            source_status=_status(),
            media_db=media_db,
            max_chars=12,
            chunk_limit=3,
            focus_chunk_index=-1,
        )

    assert media_db.range_calls == []


@pytest.mark.parametrize(
    ("max_chars", "chunk_limit", "match"),
    [
        (0, 3, "max_chars"),
        (12001, 3, "max_chars"),
        (10, -1, "chunk_limit"),
        (10, 11, "chunk_limit"),
    ],
)
def test_preview_rejects_values_outside_existing_endpoint_bounds(
    max_chars: int,
    chunk_limit: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_workspace_source_preview(
            workspace_id="workspace-alpha",
            source=_source(),
            source_status=_status(),
            media_db=_MediaDB(),
            max_chars=max_chars,
            chunk_limit=chunk_limit,
        )


def test_unavailable_preview_preserves_neutral_local_payload() -> None:
    preview = build_workspace_source_preview(
        workspace_id="workspace-alpha",
        source=_source(media_id=99),
        source_status=_status(
            state="missing_media",
            status_reason="media_not_found",
            readiness={"citation_ready": False},
        ),
        media_db=_MediaDB(),
        max_chars=3000,
        chunk_limit=3,
    )

    assert preview["content_available"] is False
    assert preview["preview_mode"] == "missing_media"
    assert preview["unavailable_reason"] == "media_not_found"
    assert preview["text_preview"] is None
    assert preview["snippets"] == []


class _PinnedMediaDB(_MediaDB):
    def __init__(self):
        super().__init__()
        self.version_calls = []
        self.version = {"version_number": 1, "content": "old accepted snapshot"}
        self.content = "new current head"

    def get_document_version(self, media_id, version_number=None, include_content=True):
        self.version_calls.append((media_id, version_number, include_content))
        return self.version if version_number == 1 else None


def test_pinned_preview_reads_exact_old_version_without_current_chunks():
    db = _PinnedMediaDB()
    preview = build_workspace_source_preview(
        workspace_id="owned",
        source=_source(),
        source_status=_status(),
        media_db=db,
        max_chars=12,
        chunk_limit=3,
        version_number=1,
    )
    assert preview["document_version_number"] == 1
    assert preview["text_preview"] == "old accepted"
    assert preview["text_total_chars"] == len("old accepted snapshot")
    assert all(item["kind"] != "chunk" for item in preview["snippets"])
    assert db.range_calls == []
    assert db.version_calls == [(5, 1, True)]


@pytest.mark.parametrize("missing_media", [False, True])
def test_missing_pin_never_falls_back_to_current_content(missing_media):
    db = _PinnedMediaDB()
    preview = build_workspace_source_preview(
        workspace_id="owned",
        source=_source(media_id=99 if missing_media else 5),
        source_status=_status(),
        media_db=db,
        max_chars=3000,
        chunk_limit=3,
        version_number=7,
    )
    assert preview["content_available"] is False
    assert preview["text_preview"] is None
    assert preview["snippets"] == []
    assert db.range_calls == []
    assert db.version_calls == ([] if missing_media else [(5, 7, True)])


def test_preview_rejects_nonpositive_version_before_media_access():
    db = _PinnedMediaDB()
    with pytest.raises(ValueError, match="version_number"):
        build_workspace_source_preview(
            workspace_id="owned",
            source=_source(),
            source_status=_status(),
            media_db=db,
            max_chars=3000,
            chunk_limit=3,
            version_number=0,
        )
    assert db.version_calls == []


def test_real_media_pin_reads_old_active_version_and_rejects_deleted_pin(tmp_path):
    db = MediaDatabase(db_path=str(tmp_path / "versions.db"), client_id="1")
    try:
        repository = MediaRepository.from_legacy_db(db)
        media_id, _, _ = repository.add_text_media(
            url="https://example.com/snapshot",
            title="Snapshot",
            media_type="article",
            content="old accepted text",
            owner_user_id=1,
        )
        old = media_db_api.get_document_version(db, media_id)
        repository.add_text_media(
            url="https://example.com/snapshot",
            title="Snapshot",
            media_type="article",
            content="new current text",
            owner_user_id=1,
            overwrite=True,
        )
        assert media_db_api.get_media_by_id(db, media_id)["content"] == "new current text"
        params = {
            "workspace_id": "owned",
            "source": _source(media_id=media_id),
            "source_status": _status(),
            "media_db": db,
            "max_chars": 3000,
            "chunk_limit": 3,
            "version_number": old["version_number"],
        }
        preview = build_workspace_source_preview(**params)
        assert preview["text_preview"] == "old accepted text"
        assert all(item["kind"] != "chunk" for item in preview["snippets"])
        assert media_db_api.soft_delete_document_version(db, old["uuid"])
        missing = build_workspace_source_preview(**params)
        assert missing["content_available"] is False
        assert missing["text_preview"] is None
        assert missing["snippets"] == []
    finally:
        db.close_connection()
