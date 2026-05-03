from __future__ import annotations

import hashlib
import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import (
    MediaDatabase,
)
from tldw_Server_API.app.core.DB_Management.scope_context import ScopeContext


pytestmark = pytest.mark.unit


_RUNTIME_MODULE_NAME = (
    "tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_search_ops"
)


def _load_runtime_module():
    try:
        return importlib.import_module(_RUNTIME_MODULE_NAME)
    except ModuleNotFoundError:
        return None


def _load_helper(name: str):
    module = _load_runtime_module()
    if module is None:
        return None
    return getattr(module, name, None)


def _make_db(tmp_path: Path, name: str) -> MediaDatabase:
    db = MediaDatabase(db_path=str(tmp_path / name), client_id="1")
    db.initialize_db()
    return db


def _seed_media(
    db: MediaDatabase,
    *,
    title: str,
    content: str,
    owner_user_id: int,
    visibility: str = "personal",
    team_id: int | None = None,
    org_id: int | None = None,
) -> int:
    media_id, _, _ = db.add_media_with_keywords(
        title=title,
        media_type="text",
        content=content,
        keywords=None,
        owner_user_id=owner_user_id,
        visibility=visibility,
    )
    if team_id is not None or org_id is not None:
        now = db._get_current_utc_timestamp_str()
        db.execute_query(
            "UPDATE Media SET team_id = ?, org_id = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE id = ?",
            (team_id, org_id, now, db.client_id, int(media_id)),
        )
    return int(media_id)


def _seed_claim(
    db: MediaDatabase,
    *,
    media_id: int,
    chunk_index: int,
    claim_text: str,
) -> int:
    chunk_hash = hashlib.sha256(f"{media_id}:{chunk_index}:{claim_text}".encode("utf-8")).hexdigest()
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": chunk_index,
                "span_start": None,
                "span_end": None,
                "claim_text": claim_text,
                "confidence": 0.8,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": chunk_hash,
            }
        ]
    )
    row = db.execute_query(
        "SELECT id FROM Claims WHERE media_id = ? ORDER BY id DESC LIMIT 1",
        (media_id,),
    ).fetchone()
    return int(row["id"])


def test_claims_search_helper_rebinds_and_preserves_sqlite_search_scope_and_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _make_db(tmp_path, "claims-search.db")
    try:
        assert db.search_claims.__func__ is _load_helper("search_claims")

        visible_media = _seed_media(
            db,
            title="Visible",
            content="python visible",
            owner_user_id=1,
            visibility="personal",
        )
        hidden_media = _seed_media(
            db,
            title="Hidden",
            content="python hidden",
            owner_user_id=2,
            visibility="personal",
        )
        visible_claim_id = _seed_claim(
            db,
            media_id=visible_media,
            chunk_index=0,
            claim_text="Python visible claim",
        )
        _seed_claim(
            db,
            media_id=hidden_media,
            chunk_index=0,
            claim_text="Python hidden claim",
        )

        scope = ScopeContext(
            user_id=1,
            org_ids=[],
            team_ids=[],
            active_org_id=None,
            active_team_id=None,
            is_admin=False,
            session_role=None,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_search_ops.get_scope",
            lambda: scope,
        )

        rows = db.search_claims("Python", limit="oops")

        assert [int(row["id"]) for row in rows] == [visible_claim_id]
        assert isinstance(float(rows[0]["relevance_score"]), float)
    finally:
        db.close_connection()


def test_claims_search_helper_can_disable_or_use_fallback_like(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _make_db(tmp_path, "claims-search-fallback.db")
    try:
        media_id = _seed_media(
            db,
            title="Fallback",
            content="alpha beta",
            owner_user_id=1,
            visibility="personal",
        )
        claim_id = _seed_claim(
            db,
            media_id=media_id,
            chunk_index=0,
            claim_text="Alpha fallback marker",
        )

        runtime_module = _load_runtime_module()
        assert runtime_module is not None

        original_fetchall = db._fetchall_with_connection
        calls: list[str] = []

        def fake_fetchall(connection, sql, params):
            calls.append(sql)
            if "claims_fts MATCH ?" in sql:
                return []
            if "claim_text LIKE ?" in sql:
                return [
                    {
                        "id": claim_id,
                        "media_id": media_id,
                        "chunk_index": 0,
                        "claim_text": "Alpha fallback marker",
                        "claim_cluster_id": None,
                    }
                ]
            return original_fetchall(connection, sql, params)

        monkeypatch.setattr(db, "_fetchall_with_connection", fake_fetchall)
        monkeypatch.setattr(runtime_module, "get_scope", lambda: None)

        no_fallback = db.search_claims("Alpha", fallback_to_like=False)
        with_fallback = db.search_claims("Alpha", fallback_to_like=True)

        assert no_fallback == []
        assert [int(row["id"]) for row in with_fallback] == [claim_id]
        assert float(with_fallback[0]["relevance_score"]) == 0.0
        assert any("claims_fts MATCH ?" in sql for sql in calls)
        assert any("claim_text LIKE ?" in sql for sql in calls)
    finally:
        db.close_connection()


def test_claims_search_helper_postgres_branch_uses_tsquery_and_ilike_fallback(
    monkeypatch,
) -> None:
    runtime_module = _load_runtime_module()
    assert runtime_module is not None

    fetchall_calls: list[tuple[str, tuple[object, ...]]] = []

    @contextmanager
    def fake_transaction():
        yield object()

    def fake_fetchall(_connection, sql, params):
        fetchall_calls.append((sql, tuple(params)))
        if "ILIKE" in sql:
            return [
                {
                    "id": 9,
                    "media_id": 5,
                    "chunk_index": 0,
                    "claim_text": "alpha beta",
                    "claim_cluster_id": 12,
                }
            ]
        return []

    fake_db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        transaction=fake_transaction,
        _fetchall_with_connection=fake_fetchall,
    )

    monkeypatch.setattr(runtime_module, "get_scope", lambda: None)
    normalize_calls: list[tuple[str, str]] = []

    def fake_normalize(query: str, backend: str) -> str:
        normalize_calls.append((query, backend))
        return "alpha & beta"

    monkeypatch.setattr(runtime_module.FTSQueryTranslator, "normalize_query", fake_normalize)

    rows = runtime_module.search_claims(fake_db, "alpha beta")

    assert normalize_calls == [("alpha beta", "postgresql")]
    assert len(fetchall_calls) == 2
    fts_sql, fts_params = fetchall_calls[0]
    fallback_sql, fallback_params = fetchall_calls[1]
    assert "to_tsquery('english', ?)" in fts_sql
    assert "ILIKE" in fallback_sql
    assert fts_params == ("alpha & beta", "alpha & beta", 20, 0)
    assert fallback_params == ("%alpha beta%", 20, 0)
    assert rows == [
        {
            "id": 9,
            "media_id": 5,
            "chunk_index": 0,
            "claim_text": "alpha beta",
            "claim_cluster_id": 12,
            "relevance_score": 0.0,
        }
    ]
