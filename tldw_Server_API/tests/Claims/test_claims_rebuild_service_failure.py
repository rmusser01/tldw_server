import time
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_rebuild_service
from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import ClaimsRebuildService, ClaimsRebuildTask
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit


def test_rebuild_claims_for_media_returns_skipped_for_missing_media(tmp_path):
    from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import (
        rebuild_claims_for_media,
    )

    db_path = tmp_path / "missing-media.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test")
    db.initialize_db()
    db.close_connection()

    result = rebuild_claims_for_media(db_path=str(db_path), media_id=404)

    assert result == {"outcome": "skipped", "reason": "media_missing", "media_id": 404}


def test_claims_rebuild_service_worker_handles_failure(monkeypatch):
    svc = ClaimsRebuildService(worker_threads=1)

    # Monkeypatch _process_task to raise an error
    def _boom(task: ClaimsRebuildTask):  # noqa: ARG001
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "_process_task", _boom)
    svc.start()
    try:
        svc.submit(media_id=123, db_path=":memory:")
        # Give worker a moment to process
        time.sleep(0.2)
        stats = svc.get_stats()
        assert stats.get("enqueued", 0) >= 1
        # Should have recorded a failure and not crash
        assert stats.get("failed", 0) >= 1
        # processed should remain 0 because _process_task failed
        assert stats.get("processed", 0) == 0
    finally:
        svc.stop()


def test_claims_rebuild_service_process_task_delegates_to_rebuild_helper(monkeypatch, tmp_path):
    svc = ClaimsRebuildService(worker_threads=1)
    db_path = str(tmp_path / "claims-task.db")
    rebuild_calls: list[dict[str, object]] = []

    def _fake_rebuild_claims_for_media(*, db_path: str, media_id: int) -> dict[str, object]:
        rebuild_calls.append({"db_path": db_path, "media_id": media_id})
        return {"outcome": "skipped", "reason": "media_missing", "media_id": media_id}

    def _unexpected_managed_media_database(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("_process_task should delegate to rebuild_claims_for_media")

    monkeypatch.setattr(
        claims_rebuild_service,
        "rebuild_claims_for_media",
        _fake_rebuild_claims_for_media,
        raising=False,
    )
    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _unexpected_managed_media_database)

    result = svc._process_task(ClaimsRebuildTask(media_id=42, db_path=db_path))

    assert result is None
    assert rebuild_calls == [{"db_path": db_path, "media_id": 42}]


def test_claims_rebuild_service_persist_health_uses_managed_media_database(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.health_calls: list[dict[str, object]] = []

        def upsert_claims_monitoring_health(self, **kwargs) -> None:
            self.health_calls.append(kwargs)

        def close_connection(self) -> None:
            pass

    svc = ClaimsRebuildService(worker_threads=1)
    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []
    db_path = str(tmp_path / "claims-health.db")

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        yield fake_db

    monkeypatch.setattr(claims_rebuild_service, "get_user_media_db_path", lambda _user_id: db_path)
    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _fake_managed_media_database)

    svc._persist_health(force=True)

    assert svc._health_db_initialized is True
    assert len(fake_db.health_calls) == 1
    assert managed_calls == [
        {
            "client_id": claims_rebuild_service.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": True,
            "kwargs": {
                "db_path": db_path,
                "suppress_close_exceptions": claims_rebuild_service._CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]


def test_claims_rebuild_service_process_task_uses_managed_media_database(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.deleted_media_ids: list[int] = []

        def get_media_by_id(self, media_id, include_deleted=False, include_trash=False):
            assert media_id == 7
            assert include_deleted is False
            assert include_trash is False
            return {
                "id": media_id,
                "title": "Doc",
                "content": "First. Second.",
            }

        def soft_delete_claims_for_media(self, media_id):
            self.deleted_media_ids.append(media_id)
            return 1

        @contextmanager
        def transaction(self):
            yield self

        def close_connection(self) -> None:
            pass

    svc = ClaimsRebuildService(worker_threads=1)
    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []
    store_calls: list[dict[str, object]] = []
    db_path = str(tmp_path / "claims-task.db")

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        yield fake_db

    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_rebuild_service,
        "chunk_for_embedding",
        lambda content, file_name: [
            {
                "text": content,
                "metadata": {"chunk_index": 0},
            }
        ],
    )
    monkeypatch.setattr(claims_rebuild_service, "resolve_claims_job_budget", lambda settings: "budget")
    monkeypatch.setattr(
        claims_rebuild_service,
        "extract_claims_for_chunks",
        lambda chunks, extractor_mode, max_per_chunk, budget: [
            {
                "chunk_index": 0,
                "claim_text": "First.",
            }
        ],
    )

    def _fake_store_claims(db, media_id, chunk_texts_by_index, claims):
        store_calls.append(
            {
                "db": db,
                "media_id": media_id,
                "chunk_texts_by_index": chunk_texts_by_index,
                "claims": claims,
            }
        )
        return 1

    monkeypatch.setattr(claims_rebuild_service, "store_claims", _fake_store_claims)

    svc._process_task(ClaimsRebuildTask(media_id=7, db_path=db_path))

    assert fake_db.deleted_media_ids == [7]
    assert len(store_calls) == 1
    assert managed_calls == [
        {
            "client_id": claims_rebuild_service.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": False,
            "kwargs": {
                "db_path": db_path,
                "suppress_close_exceptions": claims_rebuild_service._CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]


def test_rebuild_claims_for_media_returns_success_result(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.deleted_media_ids: list[int] = []

        def get_media_by_id(self, media_id, include_deleted=False, include_trash=False):
            assert media_id == 7
            assert include_deleted is False
            assert include_trash is False
            return {
                "id": media_id,
                "title": "Doc",
                "content": "First. Second.",
            }

        def soft_delete_claims_for_media(self, media_id):
            self.deleted_media_ids.append(media_id)
            return 1

        @contextmanager
        def transaction(self):
            yield self

        def close_connection(self) -> None:
            pass

    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []
    store_calls: list[dict[str, object]] = []
    db_path = str(tmp_path / "claims-helper-success.db")

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        yield fake_db

    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_rebuild_service,
        "chunk_for_embedding",
        lambda content, file_name: [
            {
                "text": content,
                "metadata": {"chunk_index": 0},
            }
        ],
    )
    monkeypatch.setattr(claims_rebuild_service, "resolve_claims_job_budget", lambda settings: "budget")
    monkeypatch.setattr(
        claims_rebuild_service,
        "extract_claims_for_chunks",
        lambda chunks, extractor_mode, max_per_chunk, budget: [
            {
                "chunk_index": 0,
                "claim_text": "First.",
            }
        ],
    )

    def _fake_store_claims(db, media_id, chunk_texts_by_index, claims):
        store_calls.append(
            {
                "db": db,
                "media_id": media_id,
                "chunk_texts_by_index": chunk_texts_by_index,
                "claims": claims,
            }
        )
        return 1

    monkeypatch.setattr(claims_rebuild_service, "store_claims", _fake_store_claims)

    result = claims_rebuild_service.rebuild_claims_for_media(db_path=db_path, media_id=7)

    assert result == {"outcome": "ok", "media_id": 7, "deleted": 1, "inserted": 1}
    assert fake_db.deleted_media_ids == [7]
    assert len(store_calls) == 1
    assert managed_calls == [
        {
            "client_id": claims_rebuild_service.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": False,
            "kwargs": {
                "db_path": db_path,
                "suppress_close_exceptions": claims_rebuild_service._CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]


def test_claims_rebuild_process_task_rolls_back_soft_delete_when_store_returns_zero(monkeypatch, tmp_path):
    db_path = str(tmp_path / "claims-rebuild-rollback.db")
    seed_db = MediaDatabase(db_path=db_path, client_id="1")
    seed_db.initialize_db()
    media_id, _, _ = seed_db.add_media_with_keywords(
        title="Doc",
        media_type="text",
        content="Original claim. Replacement claim.",
        keywords=None,
    )
    seed_db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Original claim.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "oldhash",
            }
        ]
    )
    original = seed_db.execute_query(
        "SELECT id FROM Claims WHERE media_id = ? AND deleted = 0",
        (media_id,),
    ).fetchone()
    original_claim_id = int(original["id"])
    seed_db.close_connection()

    monkeypatch.setattr(
        claims_rebuild_service,
        "chunk_for_embedding",
        lambda content, file_name: [
            {
                "text": content,
                "metadata": {"chunk_index": 0},
            }
        ],
    )
    monkeypatch.setattr(claims_rebuild_service, "resolve_claims_job_budget", lambda settings: None)
    monkeypatch.setattr(
        claims_rebuild_service,
        "extract_claims_for_chunks",
        lambda chunks, extractor_mode, max_per_chunk, budget: [
            {
                "chunk_index": 0,
                "claim_text": "Replacement claim.",
            }
        ],
    )
    monkeypatch.setattr(claims_rebuild_service, "store_claims", lambda *args, **kwargs: 0)

    svc = ClaimsRebuildService(worker_threads=1)
    with pytest.raises(RuntimeError, match="zero replacement claims"):
        svc._process_task(ClaimsRebuildTask(media_id=media_id, db_path=db_path))

    verify_db = MediaDatabase(db_path=db_path, client_id="1")
    try:
        row = verify_db.execute_query(
            "SELECT deleted FROM Claims WHERE id = ?",
            (original_claim_id,),
        ).fetchone()
        assert int(row["deleted"]) == 0
    finally:
        verify_db.close_connection()
