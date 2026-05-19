from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.RAG.rag_service.rewrite_cache import (
    RewriteCache,
    _is_relative_to,
    _normalize_query,
    _safe_path,
)

pytestmark = pytest.mark.unit


def test_rewrite_cache_put_get(tmp_path, monkeypatch):


    p = tmp_path / "rc.jsonl"
    monkeypatch.setenv("RAG_REWRITE_CACHE_PATH", str(p))
    rc = RewriteCache()

    q = "What is CUDA?"
    rewrites = ["compute unified device architecture", "nvidia cuda"]
    rc.put(q, rewrites, intent="FACTUAL", corpus="ml")

    out = rc.get(q, intent="FACTUAL", corpus="ml")
    assert out is not None
    assert any("compute unified" in r for r in out)


def test_rewrite_cache_user_id_path_is_sandboxed(tmp_path, monkeypatch):


    base_dir = tmp_path / "Databases" / "user_databases"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.chdir(tmp_path)
    rc = RewriteCache(user_id="../evil")
    cache_path = Path(rc.path).resolve()
    assert _is_relative_to(cache_path, base_dir)
    assert cache_path.name == "rewrite_cache.jsonl"


def test_rewrite_cache_user_id_preserves_safe_segment(tmp_path, monkeypatch):


    base_dir = tmp_path / "Databases" / "user_databases"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.chdir(tmp_path)
    rc = RewriteCache(user_id="user_123")
    cache_path = Path(rc.path).resolve()
    assert cache_path.parent.name == "Rewrite_Cache"
    assert cache_path.parent.parent.name == "user_123"


def test_rewrite_cache_user_id_with_special_chars(tmp_path, monkeypatch):


    base_dir = tmp_path / "Databases" / "user_databases"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.chdir(tmp_path)
    rc = RewriteCache(user_id="user_abc-123")
    cache_path = Path(rc.path).resolve()
    assert cache_path.parent.parent.name == "user_abc-123"


def test_rewrite_cache_unsafe_user_id_is_hashed(tmp_path, monkeypatch):


    base_dir = tmp_path / "Databases" / "user_databases"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.chdir(tmp_path)
    rc = RewriteCache(user_id="../../etc/passwd")
    cache_path = Path(rc.path).resolve()
    assert cache_path.parent.parent.name.startswith("user_")
    assert cache_path.parent.parent.name != "../../etc/passwd"
    rc2 = RewriteCache(user_id="../../etc/passwd")
    cache_path2 = Path(rc2.path).resolve()
    assert cache_path.parent.parent.name == cache_path2.parent.parent.name


def test_rewrite_cache_corpus_scopes_entries(tmp_path, monkeypatch):
    p = tmp_path / "rc.jsonl"
    monkeypatch.setenv("RAG_REWRITE_CACHE_PATH", str(p))
    rc = RewriteCache()

    q = "shared query"
    rc.put(q, ["tenant-a rewrite"], intent="FACTUAL", corpus="tenant-a")
    rc.put(q, ["tenant-b rewrite"], intent="FACTUAL", corpus="tenant-b")

    out_a = rc.get(q, intent="FACTUAL", corpus="tenant-a")
    out_b = rc.get(q, intent="FACTUAL", corpus="tenant-b")

    assert out_a is not None
    assert out_b is not None
    assert any("tenant-a rewrite" == r for r in out_a)
    assert any("tenant-b rewrite" == r for r in out_b)
    assert all("tenant-b rewrite" != r for r in out_a)
    assert all("tenant-a rewrite" != r for r in out_b)


def test_safe_path_resolution_log_omits_raw_exception_details(monkeypatch):
    secret_path = "/tmp/rewrite-cache/secret-token-123/cache.jsonl"
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    def fail_path_resolution(user_id):
        raise OSError(f"cannot access {secret_path}")

    monkeypatch.delenv("RAG_REWRITE_CACHE_PATH", raising=False)
    monkeypatch.setattr(DatabasePaths, "get_user_rewrite_cache_path", fail_path_resolution)

    try:
        with pytest.raises(OSError, match="secret-token-123"):
            _safe_path("user-with-secret-token-123")
    finally:
        logger.remove(sink_id)

    logged = "\n".join(messages)
    assert "Rewrite cache: failed to resolve cache path" in logged
    assert "secret-token-123" not in logged
    assert secret_path not in logged


def test_normalize_query_fallback_log_omits_raw_exception_details():
    secret_token = "normalizer-secret-token-456"
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    class BadQuery:
        def strip(self):
            raise AttributeError(f"cannot strip {secret_token}")

        def __bool__(self):
            return True

    query = BadQuery()
    try:
        assert _normalize_query(query) is query
    finally:
        logger.remove(sink_id)

    logged = "\n".join(messages)
    assert "Rewrite cache: failed to normalize query; returning fallback" in logged
    assert secret_token not in logged


def test_put_persist_failure_log_omits_raw_exception_details(tmp_path, monkeypatch):
    secret_path = tmp_path / "secret-token-789" / "rc.jsonl"
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    original_open = Path.open

    def fail_open(self, *args, **kwargs):
        if self == secret_path:
            raise OSError(f"cannot write {secret_path}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setenv("RAG_REWRITE_CACHE_PATH", str(secret_path))
    monkeypatch.setattr(Path, "open", fail_open)
    rc = RewriteCache()

    try:
        rc.put("What is CUDA?", ["nvidia cuda"], intent="FACTUAL", corpus="ml")
    finally:
        logger.remove(sink_id)

    assert rc.get("What is CUDA?", intent="FACTUAL", corpus="ml") == ["nvidia cuda"]
    logged = "\n".join(messages)
    assert "Failed to persist rewrite cache" in logged
    assert "secret-token-789" not in logged
    assert str(secret_path) not in logged
