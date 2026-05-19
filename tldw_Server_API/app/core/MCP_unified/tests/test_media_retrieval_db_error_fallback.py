from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import MediaModule


def _expect_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        pytest.fail(f"{message}: expected {expected!r}, got {actual!r}")


class _BrokenPrechunkedDb:
    def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
        return {
            "id": media_id,
            "title": "Chunked doc",
            "content": "ABCDEFGHIJ",
            "type": "text",
            "url": None,
            "ingestion_date": None,
            "last_modified": None,
            "version": 1,
            "owner_user_id": 1,
        }

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        raise DatabaseError("prechunk lookup failed")


def test_media_get_normalized_falls_back_to_runtime_chunking_on_database_error() -> None:
    module = MediaModule(ModuleConfig(name="media"))
    module.db = _BrokenPrechunkedDb()
    context = SimpleNamespace(user_id=1, metadata={})
    chars_per_token_key = "_".join(("chars", "per", "token"))

    result = module._media_get_normalized_sync(
        media_id=1,
        retrieval={
            "mode": "chunk",
            chars_per_token_key: 1,
            "chunk_size_tokens": 5,
            "loc": {"approx_offset": 7},
        },
        context=context,
    )

    _expect_equal(result["content"], "FGHIJ", "expected on-the-fly chunk fallback content")
    _expect_equal(
        result["meta"]["loc"],
        {"chunk_index": 1, "approx_offset": 5},
        "expected fallback chunk locator",
    )
