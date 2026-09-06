import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

pytestmark = pytest.mark.unit


def test_build_tts_history_filters_preserves_condition_and_param_order() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        tts_history_ops as tts_history_ops_module,
    )

    def append_case_insensitive_like(conditions, params, column: str, pattern: str) -> None:
        conditions.append(f"{column} LIKE ?")
        params.append(pattern)

    db = SimpleNamespace(_append_case_insensitive_like=append_case_insensitive_like)

    conditions, params = tts_history_ops_module._build_tts_history_filters(
        db,
        user_id="1",
        q="hello",
        text_hash="hash123",
        favorite=True,
        provider="openai",
        model="tts-1",
        voice_id="alloy",
        voice_name="Alloy",
        created_from="2026-01-01T00:00:00.000Z",
        created_to="2026-01-31T00:00:00.000Z",
        cursor_created_at="2026-02-01T00:00:00.000Z",
        cursor_id=42,
        include_deleted=False,
    )

    assert conditions == [
        "user_id = ?",
        "deleted = 0",
        "favorite = ?",
        "provider = ?",
        "model = ?",
        "voice_id = ?",
        "voice_name = ?",
        "text_hash = ?",
        "created_at >= ?",
        "created_at <= ?",
        "text LIKE ?",
        "(created_at < ? OR (created_at = ? AND id < ?))",
    ]
    assert params == [
        "1",
        1,
        "openai",
        "tts-1",
        "alloy",
        "Alloy",
        "hash123",
        "2026-01-01T00:00:00.000Z",
        "2026-01-31T00:00:00.000Z",
        "%hello%",
        "2026-02-01T00:00:00.000Z",
        "2026-02-01T00:00:00.000Z",
        42,
    ]


def test_mark_tts_history_artifacts_deleted_for_file_id_matches_rows_and_ignores_malformed_json() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        tts_history_ops as tts_history_ops_module,
    )

    rows = [
        {"id": 1, "artifact_ids": json.dumps([10, 20])},
        {"id": 2, "artifact_ids": "not json"},
        {"id": 3, "artifact_ids": json.dumps([30])},
        {"id": 4, "artifact_ids": None},
        {"id": 5, "artifact_ids": json.dumps([20, 40])},
    ]
    execute_calls: list[tuple[str, tuple[object, ...] | None, bool]] = []

    class FakeCursor:
        def __init__(self, *, rows=None, rowcount: int = 0) -> None:
            self._rows = rows or []
            self.rowcount = rowcount

        def fetchall(self):
            return self._rows

    def execute_query(query: str, params=None, commit: bool = False):
        execute_calls.append((query, params, commit))
        if query.startswith("SELECT id, artifact_ids FROM tts_history"):
            return FakeCursor(rows=rows)
        return FakeCursor(rowcount=2)

    db = SimpleNamespace(execute_query=execute_query)

    removed = tts_history_ops_module.mark_tts_history_artifacts_deleted_for_file_id(
        db,
        user_id="1",
        file_id=20,
        deleted_at="2026-03-21T00:00:00.000Z",
    )

    assert removed == 2
    assert execute_calls == [
        (
            "SELECT id, artifact_ids FROM tts_history "
            "WHERE user_id = ? AND artifact_ids IS NOT NULL AND deleted = 0",
            ("1",),
            False,
        ),
        (
            "UPDATE tts_history "
            "SET artifact_deleted_at = ?, output_id = NULL, artifact_ids = NULL "
            "WHERE user_id = ? AND id IN (?,?) AND deleted = 0",
            ("2026-03-21T00:00:00.000Z", "1", 1, 5),
            True,
        ),
    ]


def test_purge_tts_history_for_user_applies_retention_then_row_cap() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        tts_history_ops as tts_history_ops_module,
    )

    execute_calls: list[tuple[str, tuple[object, ...] | None, bool]] = []

    class FakeCursor:
        def __init__(self, *, rowcount: int = 0, row=None) -> None:
            self.rowcount = rowcount
            self._row = row

        def fetchone(self):
            return self._row

    def execute_query(query: str, params=None, commit: bool = False):
        execute_calls.append((query, params, commit))
        if query.startswith("DELETE FROM tts_history WHERE user_id = ? AND created_at < ?"):
            return FakeCursor(rowcount=3)
        if query.startswith("SELECT COUNT(*) AS count FROM tts_history"):
            return FakeCursor(row={"count": 105})
        return FakeCursor(rowcount=5)

    db = SimpleNamespace(execute_query=execute_query)

    removed = tts_history_ops_module.purge_tts_history_for_user(
        db,
        user_id="1",
        retention_days=30,
        max_rows=100,
    )

    assert removed == 8
    assert execute_calls[0][0] == "DELETE FROM tts_history WHERE user_id = ? AND created_at < ?"
    assert execute_calls[0][1][0] == "1"
    assert execute_calls[0][1][1].endswith("Z")
    assert execute_calls[1] == (
        "SELECT COUNT(*) AS count FROM tts_history WHERE user_id = ?",
        ("1",),
        False,
    )
    assert execute_calls[2] == (
        "DELETE FROM tts_history WHERE user_id = ? AND id IN ("
        "SELECT id FROM tts_history WHERE user_id = ? "
        "ORDER BY created_at ASC, id ASC LIMIT ?"
        ")",
        ("1", "1", 5),
        True,
    )


def test_create_tts_history_entry_returns_postgres_returning_id_and_serializes_payloads() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        tts_history_ops as tts_history_ops_module,
    )

    execute_calls: list[tuple[str, tuple[object, ...], bool, object | None]] = []

    class FakeCursor:
        def fetchone(self):
            return {"id": 17}

    def execute_query(query: str, params: tuple[object, ...], commit: bool = False, connection=None):
        execute_calls.append((query, params, commit, connection))
        return FakeCursor()

    db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        execute_query=execute_query,
        transaction=lambda: nullcontext(None),
        _get_current_utc_timestamp_str=lambda: "2026-03-21T00:00:00.000Z",
    )

    result = tts_history_ops_module.create_tts_history_entry(
        db,
        user_id="1",
        text_hash="hash123",
        text="hello",
        text_length=5,
        provider="openai",
        model="tts-1",
        voice_id="alloy",
        voice_name="Alloy",
        voice_info={"lang": "en"},
        format="mp3",
        duration_ms=100,
        generation_time_ms=50,
        params_json={"speed": 1.0},
        status="success",
        segments_json={"segments": [1]},
        favorite=True,
        job_id=9,
        output_id=11,
        artifact_ids=[21, 22],
        error_message=None,
    )

    assert result == 17
    assert execute_calls == [
        (
            "INSERT INTO tts_history "
            "(user_id, created_at, text, text_hash, text_length, provider, model, voice_id, voice_name, "
            "voice_info, format, duration_ms, generation_time_ms, params_json, status, segments_json, "
            "favorite, job_id, output_id, artifact_ids, artifact_deleted_at, error_message, deleted, deleted_at, output_incarnation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (
                "1",
                "2026-03-21T00:00:00.000Z",
                "hello",
                "hash123",
                5,
                "openai",
                "tts-1",
                "alloy",
                "Alloy",
                '{"lang":"en"}',
                "mp3",
                100,
                50,
                '{"speed":1.0}',
                "success",
                '{"segments":[1]}',
                1,
                9,
                11,
                "[21,22]",
                None,
                None,
                0,
                None,
                None,
            ),
            False,
            None,
        )
    ]


def test_gateway_history_identity_uses_existing_provider_and_params_json_without_schema_change(
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    db = MediaDatabase(db_path=str(tmp_path / "history.sqlite3"), client_id="gateway-history")
    try:
        history_id = db.create_tts_history_entry(
            user_id="1",
            text_hash="gateway-hash",
            provider="gateway:backup",
            model="Vendor/Backup-TTS",
            params_json={
                "requested_backend": "gateway:company",
                "fallback_used": True,
                "conversion_used": False,
            },
            status="success",
        )
        row = db.get_tts_history_entry(user_id="1", history_id=int(history_id))
        columns = {
            item["name"]
            for item in db.execute_query("PRAGMA table_info(tts_history)").fetchall()
        }
    finally:
        db.close_connection()

    assert row is not None
    assert row["provider"] == "gateway:backup"
    assert json.loads(row["params_json"]) == {
        "requested_backend": "gateway:company",
        "fallback_used": True,
        "conversion_used": False,
    }
    assert "requested_backend" not in columns
    assert "fallback_used" not in columns
    assert "conversion_used" not in columns
