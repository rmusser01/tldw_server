from __future__ import annotations

import hashlib
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import rag_unified


class _LoggerStub:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        try:
            rendered = message.format(*args, **kwargs)
        except (IndexError, KeyError, ValueError):
            rendered = f"{message} args={args!r} kwargs={kwargs!r}"
        self.infos.append(rendered)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "metadata"),
    [
        ("Unified RAG search", {"user": "alice"}),
        ("Advanced search", {}),
    ],
)
def test_rag_search_request_log_omits_raw_query(monkeypatch, label, metadata):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(rag_unified, "logger", logger_stub)
    query = "summarize secret token sk-live-private and /private/customer.db"

    rag_unified._log_rag_search_request(label, query, **metadata)

    rendered = "\n".join(logger_stub.infos)
    expected_hash = hashlib.md5(
        query.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:8]

    assert query not in rendered
    assert "sk-live-private" not in rendered
    assert "/private/customer.db" not in rendered
    assert f"query_hash={expected_hash}" in rendered
    assert f"len={len(query)}" in rendered
    for key, value in metadata.items():
        assert f"{key}={value}" in rendered


def test_rag_search_request_log_handles_non_string_query(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(rag_unified, "logger", logger_stub)

    rag_unified._log_rag_search_request("Unified RAG search", 42)

    assert logger_stub.infos == ["Unified RAG search: query_hash=a1d0c6e8 len=2"]
