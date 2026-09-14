"""Tests for reusable bounded workspace source previews."""
from __future__ import annotations

from typing import Any

import pytest

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
