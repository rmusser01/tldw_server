from __future__ import annotations

import json
from pathlib import Path

import tldw_Server_API.app.core.RAG.rag_service.user_personalization_store as personalization_store
from tldw_Server_API.app.core.RAG.rag_service.user_personalization_store import UserPersonalizationStore


def test_personalization_store_persists_event_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    store = UserPersonalizationStore("user-1")
    store.record_event(
        event_type="dwell_time",
        doc_id="doc-1",
        chunk_ids=["chunk-1"],
        rank=2,
        session_id="sess-1",
        conversation_id="conv-1",
        message_id="msg-1",
        dwell_ms=3000,
        query="reset auth",
        impression=["doc-1", "doc-2"],
        corpus="media_db",
    )

    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data.get("event_log")
    entry = data["event_log"][-1]
    assert entry["event_type"] == "dwell_time"
    assert entry["doc_id"] == "doc-1"
    assert entry["chunk_ids"] == ["chunk-1"]
    assert entry["rank"] == 2
    assert entry["dwell_ms"] == 3000
    assert entry["session_id"] == "sess-1"
    assert entry["conversation_id"] == "conv-1"
    assert entry["message_id"] == "msg-1"
    assert entry["query"] == "reset auth"
    assert entry["impression_list"] == ["doc-1", "doc-2"]


def _capture_personalization_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = personalization_store.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
    )
    return messages, sink_id


def test_personalization_store_load_failure_log_is_sanitized(tmp_path, monkeypatch) -> None:
    secret_path = tmp_path / "private-user-dir" / "rag_personalization.json"
    secret_path.parent.mkdir()
    secret_path.write_text("{}", encoding="utf-8")
    messages, sink_id = _capture_personalization_logs("WARNING")
    original_open = Path.open

    def _raise_open(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        _ = (args, kwargs)
        if self == secret_path:
            raise OSError(f"cannot open {secret_path}?token=secret-token")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(
        personalization_store.DatabasePaths,
        "get_user_rag_personalization_path",
        lambda _user_id: secret_path,
    )
    monkeypatch.setattr(Path, "open", _raise_open)

    try:
        store = UserPersonalizationStore("user-1")
    finally:
        personalization_store.logger.remove(sink_id)

    assert store._data == {"priors": {}, "events": {}, "pairs": {}, "event_log": []}
    joined = "\n".join(messages)
    assert "Failed loading personalization data" in joined
    assert "private-user-dir" not in joined
    assert "secret-token" not in joined
    assert str(secret_path) not in joined


def test_personalization_store_save_failure_log_is_sanitized(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    store = UserPersonalizationStore("user-1")
    secret_path = store.path
    messages, sink_id = _capture_personalization_logs("DEBUG")
    original_open = Path.open

    def _raise_open(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        _ = (args, kwargs)
        if self == secret_path:
            raise OSError(f"cannot write {secret_path}?token=secret-token")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raise_open)

    try:
        store._save()
    finally:
        personalization_store.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed saving personalization" in joined
    assert "user-1" not in joined
    assert "secret-token" not in joined
    assert str(secret_path) not in joined
