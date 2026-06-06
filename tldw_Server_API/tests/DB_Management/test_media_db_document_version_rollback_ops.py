from __future__ import annotations

from contextlib import nullcontext
import importlib
import sys
import types


def _rollback_module():
    return importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.document_version_rollback_ops"
    )


def test_rollback_to_version_rebinds_to_runtime_wrapper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import MediaDatabase

    assert MediaDatabase.__dict__["rollback_to_version"].__globals__["__name__"] == (
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.document_version_rollback_ops"
    )


def test_rollback_to_version_returns_error_when_target_version_missing(monkeypatch) -> None:
    rollback_ops = _rollback_module()

    class _Db:
        client_id = "client-1"
        backend = object()

        def transaction(self):
            return nullcontext(object())

        def _get_current_utc_timestamp_str(self):
            return "2026-03-22T00:00:00Z"

        def _fetchone_with_connection(self, _conn, query, _params=None):
            if "FROM Media WHERE id = ?" in query:
                return {
                    "uuid": "media-uuid",
                    "version": 1,
                    "title": "Rollback Doc",
                    "content": "current",
                }
            return None

    monkeypatch.setattr(rollback_ops, "get_document_version", lambda *_args, **_kwargs: None)

    result = rollback_ops.rollback_to_version(_Db(), 7, 5)

    assert result == {"error": "Rollback target version 5 not found or inactive."}


def test_rollback_to_version_returns_error_when_target_is_latest(monkeypatch) -> None:
    rollback_ops = _rollback_module()

    class _Db:
        client_id = "client-1"
        backend = object()

        def transaction(self):
            return nullcontext(object())

        def _get_current_utc_timestamp_str(self):
            return "2026-03-22T00:00:00Z"

        def _fetchone_with_connection(self, _conn, query, _params=None):
            if "FROM Media WHERE id = ?" in query:
                return {
                    "uuid": "media-uuid",
                    "version": 1,
                    "title": "Rollback Doc",
                    "content": "current",
                }
            if "SELECT MAX(version_number)" in query:
                return {"latest_vn": 2}
            return None

    monkeypatch.setattr(
        rollback_ops,
        "get_document_version",
        lambda *_args, **_kwargs: {
            "content": "rolled-back",
            "prompt": "prompt",
            "analysis_content": "analysis",
        },
    )

    result = rollback_ops.rollback_to_version(_Db(), 7, 2)

    assert result == {"error": "Cannot rollback to the current latest version number."}


def test_rollback_to_version_success_keeps_hook_failures_non_blocking(monkeypatch) -> None:
    rollback_ops = _rollback_module()
    sync_events: list[tuple[str, str, int, dict[str, object] | None]] = []
    fts_updates: list[tuple[int, str, str]] = []

    class _Db:
        client_id = "1"
        backend = object()

        def transaction(self):
            return nullcontext(object())

        def _get_current_utc_timestamp_str(self):
            return "2026-03-22T00:00:00Z"

        def _fetchone_with_connection(self, _conn, query, _params=None):
            if "SELECT * FROM Media WHERE id = ?" in query:
                return {"uuid": "media-uuid", "content": "rolled-back"}
            if "FROM Media WHERE id = ?" in query:
                return {
                    "uuid": "media-uuid",
                    "version": 1,
                    "title": "Rollback Doc",
                    "content": "current",
                }
            if "SELECT MAX(version_number)" in query:
                return {"latest_vn": 2}
            return None

        def create_document_version(self, **_kwargs):
            return {"version_number": 3, "uuid": "new-doc-uuid"}

        def _execute_with_connection(self, _conn, _query, _params=None):
            return types.SimpleNamespace(rowcount=1)

        def _log_sync_event(self, _conn, entity, entity_uuid, operation, version, payload=None):
            sync_events.append((entity, entity_uuid, version, payload))

        def _update_fts_media(self, _conn, media_id, title, content, **_kwargs):
            fts_updates.append((media_id, title, content))

    class _FailingCollectionsDb:
        @classmethod
        def from_backend(cls, **_kwargs):
            return cls()

        def mark_highlights_stale_if_content_changed(self, *_args, **_kwargs):
            raise RuntimeError("stale-mark failed")

    monkeypatch.setattr(
        rollback_ops,
        "get_document_version",
        lambda *_args, **_kwargs: {
            "content": "rolled-back",
            "prompt": "prompt-v1",
            "analysis_content": "analysis-v1",
        },
    )
    monkeypatch.setattr(rollback_ops, "_CollectionsDB", _FailingCollectionsDb)

    agentic_module = types.ModuleType(
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker"
    )
    agentic_module.invalidate_intra_doc_vectors = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("invalidate failed")
    )
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker",
        agentic_module,
    )

    result = rollback_ops.rollback_to_version(_Db(), 7, 1)

    assert result == {
        "success": "Rolled back to version 1. State saved as new version 3.",
        "new_document_version_number": 3,
        "new_document_version_uuid": "new-doc-uuid",
        "new_media_version": 2,
    }
    assert sync_events[-1][0] == "Media"
    assert sync_events[-1][3]["rolled_back_to_doc_ver_uuid"] == "new-doc-uuid"
    assert sync_events[-1][3]["rolled_back_to_doc_ver_num"] == 3
    assert fts_updates == [(7, "Rollback Doc", "rolled-back")]
