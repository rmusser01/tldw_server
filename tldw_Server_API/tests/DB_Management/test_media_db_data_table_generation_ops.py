from __future__ import annotations

import contextlib
import importlib
import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError

pytestmark = pytest.mark.unit


def _build_db_with_generation_state():
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    calls: list[tuple[str, object, tuple[object, ...], dict[str, object]]] = []

    class _FakeCursor:
        def __init__(self, rowcount: int = 1):
            self.rowcount = rowcount

        def fetchone(self):
            return {"client_id": "77"}

    class _FakeTransaction:
        def __init__(self, state):
            self.state = state

        def __enter__(self):
            calls.append(("transaction_enter", None, (), {}))
            return object()

        def __exit__(self, exc_type, exc, tb):
            calls.append(("transaction_exit", exc_type, (), {}))
            return False

    class _FakeDB:
        backend_type = BackendType.SQLITE

        def __init__(self):
            self.state = {
                "table": {
                    "id": 9,
                    "uuid": "table-9",
                    "status": "queued",
                    "row_count": 0,
                    "generation_model": "existing-model",
                },
                "columns": [{"column_id": "col_a", "name": "Name", "type": "text"}],
                "rows": [{"row_id": "row_a", "row_index": 0, "row_json": json.dumps({"col_a": "old"})}],
                "sources": [{"source_type": "chat", "source_id": "chat_existing"}],
            }

        def _resolve_data_table_write_client_id(self, table_id, owner_user_id=None):
            calls.append(("resolve_write_client", table_id, (), {"owner_user_id": owner_user_id}))
            return "77"

        def _get_current_utc_timestamp_str(self):
            calls.append(("get_now", None, (), {}))
            return "2026-03-21T00:00:00.000Z"

        def _generate_uuid(self):
            calls.append(("generate_uuid", None, (), {}))
            return f"generated-{len([item for item in calls if item[0] == 'generate_uuid'])}"

        def _normalize_data_table_row_json(self, row_json, *, column_ids=None, validate_keys=False):
            calls.append(
                (
                    "normalize_row_json",
                    row_json,
                    (),
                    {
                        "column_ids": column_ids,
                        "validate_keys": validate_keys,
                    },
                )
            )
            if isinstance(row_json, str):
                return row_json
            return json.dumps(row_json)

        def _get_data_table_owner_client_id(self, conn, table_id):
            calls.append(("get_owner", conn, (table_id,), {}))
            return self.state["table"].get("client_id", "77")

        def transaction(self):
            calls.append(("transaction", None, (), {}))
            return _FakeTransaction(self.state)

        def _execute_with_connection(self, conn, query, params):
            calls.append(("execute_with_connection", query, (conn, params), {}))
            normalized_query = " ".join(query.split())
            if normalized_query.startswith("UPDATE data_tables SET"):
                self.state["table"]["status"] = params[0]
                self.state["table"]["row_count"] = params[1]
                self.state["table"]["last_error"] = params[2]
                self.state["table"]["updated_at"] = params[3]
                self.state["table"]["last_modified"] = params[4]
                if "generation_model = ?" in normalized_query:
                    self.state["table"]["generation_model"] = params[5]
            if normalized_query.startswith("UPDATE data_table_sources SET"):
                self.state["sources"] = []
            return _FakeCursor()

        def execute_many(self, query, rows, *, commit=False, connection=None):
            calls.append(("execute_many", query, (tuple(rows),), {"commit": commit, "connection": connection}))
            return _FakeCursor()

        def get_data_table(self, table_id, *, include_deleted=False, owner_user_id=None):
            calls.append(
                (
                    "get_data_table",
                    table_id,
                    (),
                    {"include_deleted": include_deleted, "owner_user_id": owner_user_id},
                )
            )
            return self.state["table"].copy()

    db = _FakeDB()
    return db, calls, data_table_generation_ops_module


def test_persist_data_table_generation_rejects_blank_owner_user_id() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db = SimpleNamespace()

    with pytest.raises(InputError, match="owner_user_id is required"):
        data_table_generation_ops_module.persist_data_table_generation(
            db,
            9,
            columns=[{"name": "Name", "type": "text"}],
            rows=[{"row_json": {"col_a": "value"}}],
            owner_user_id="   ",
        )


def test_persist_data_table_generation_rejects_owner_mismatch() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db = SimpleNamespace(
        _resolve_data_table_write_client_id=lambda *_args, **_kwargs: "77",
        _get_current_utc_timestamp_str=lambda: "2026-03-21T00:00:00.000Z",
        _generate_uuid=lambda: "generated-1",
        _normalize_data_table_row_json=lambda row_json, **_kwargs: json.dumps(row_json),
        transaction=lambda: contextlib.nullcontext(object()),
        _get_data_table_owner_client_id=lambda conn, table_id: "88",
        _execute_with_connection=lambda *_args, **_kwargs: None,
        execute_many=lambda *_args, **_kwargs: None,
        get_data_table=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InputError, match="data_table_owner_mismatch"):
        data_table_generation_ops_module.persist_data_table_generation(
            db,
            9,
            columns=[{"name": "Name", "type": "text"}],
            rows=[{"row_json": {"col_a": "value"}}],
            owner_user_id="77",
        )


def test_persist_data_table_generation_preserves_existing_sources_when_sources_is_none() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db, calls, _ = _build_db_with_generation_state()

    result = data_table_generation_ops_module.persist_data_table_generation(
        db,
        9,
        columns=[{"name": "Name", "type": "text"}],
        rows=[{"row_json": {"col_a": "value"}}],
        sources=None,
        status="ready",
        row_count=1,
        generation_model=None,
        owner_user_id=None,
    )

    assert result["status"] == "ready"
    assert result["generation_model"] == "existing-model"
    assert db.state["sources"] == [{"source_type": "chat", "source_id": "chat_existing"}]
    assert not any(
        call[0] == "execute_with_connection" and "data_table_sources" in str(call[1])
        for call in calls
    )


def test_persist_data_table_generation_clears_existing_sources_when_sources_is_empty_list() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db, calls, _ = _build_db_with_generation_state()

    result = data_table_generation_ops_module.persist_data_table_generation(
        db,
        9,
        columns=[{"name": "Name", "type": "text"}],
        rows=[{"row_json": {"col_a": "value"}}],
        sources=[],
        status="ready",
        row_count=1,
        generation_model=None,
        owner_user_id=None,
    )

    assert result["status"] == "ready"
    assert db.state["sources"] == []
    assert any(
        call[0] == "execute_with_connection" and "data_table_sources" in str(call[1])
        for call in calls
    )


def test_persist_data_table_generation_preserves_generation_model_when_none() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db, _, _ = _build_db_with_generation_state()

    result = data_table_generation_ops_module.persist_data_table_generation(
        db,
        9,
        columns=[{"name": "Name", "type": "text"}],
        rows=[{"row_json": {"col_a": "value"}}],
        sources=None,
        status="ready",
        row_count=1,
        generation_model=None,
        owner_user_id=None,
    )

    assert result["generation_model"] == "existing-model"
    assert result["status"] == "ready"


def test_persist_data_table_generation_returns_refreshed_table_row() -> None:
    data_table_generation_ops_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.data_table_generation_ops"
    )

    db, _, _ = _build_db_with_generation_state()

    result = data_table_generation_ops_module.persist_data_table_generation(
        db,
        9,
        columns=[{"name": "Name", "type": "text"}],
        rows=[{"row_json": {"col_a": "value"}}],
        sources=None,
        status="ready",
        row_count=1,
        generation_model="new-model",
        owner_user_id=None,
    )

    assert result["uuid"] == "table-9"
    assert result["status"] == "ready"
    assert result["row_count"] == 1
    assert result["generation_model"] == "new-model"
