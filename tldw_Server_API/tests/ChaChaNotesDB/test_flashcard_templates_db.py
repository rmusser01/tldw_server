import os
import tempfile
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    BackendDatabaseError,
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)


def test_flashcard_template_create_update_delete_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")

        template_id = db.add_flashcard_template(
            name="Vocabulary Definition",
            model_type="basic",
            front_template="What does {{term}} mean?",
            back_template="{{definition}}",
            placeholder_definitions=[
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        )

        template = db.get_flashcard_template(template_id)
        assert template is not None
        assert template["name"] == "Vocabulary Definition"
        assert template["model_type"] == "basic"
        assert template["placeholder_definitions"][0]["key"] == "term"

        updated = db.update_flashcard_template(
            template_id,
            {
                "name": "Vocabulary Definition v2",
                "front_template": "Define {{term}}",
                "back_template": "{{definition}}",
            },
            expected_version=template["version"],
        )
        assert updated is True

        renamed = db.get_flashcard_template(template_id)
        assert renamed is not None
        assert renamed["name"] == "Vocabulary Definition v2"
        assert renamed["version"] == template["version"] + 1

        deleted = db.soft_delete_flashcard_template(template_id, expected_version=renamed["version"])
        assert deleted is True
        assert db.get_flashcard_template(template_id) is None
        assert db.soft_delete_flashcard_template(template_id, expected_version=renamed["version"] + 1) is False


def test_flashcard_template_validation_requires_matching_placeholders():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")

        with pytest.raises(InputError):
            db.add_flashcard_template(
                name="Broken",
                model_type="basic",
                front_template="What does {{term}} mean?",
                back_template="{{definition}}",
                placeholder_definitions=[
                    {
                        "key": "definition",
                        "label": "Definition",
                        "required": True,
                        "targets": ["back_template"],
                    }
                ],
            )


def test_flashcard_template_queries_use_postgres_safe_deleted_clause(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        captured: list[tuple[str, object]] = []

        class _Cursor:
            def __init__(self, *, row=None, rows=None):
                self._row = row
                self._rows = rows or []

            def fetchone(self):
                return self._row

            def fetchall(self):
                return self._rows

        def _fake_execute_query(query, params=None):
            captured.append((str(query), params))
            if "COUNT(*)" in str(query):
                return _Cursor(row={"cnt": 0})
            if "WHERE id = ?" in str(query):
                return _Cursor(row=None)
            return _Cursor(rows=[])

        monkeypatch.setattr(db, "backend_type", BackendType.POSTGRESQL)
        monkeypatch.setattr(db, "execute_query", _fake_execute_query)

        assert db.count_flashcard_templates() == 0
        assert db.list_flashcard_templates() == []
        assert db.get_flashcard_template(17) is None

        assert all("deleted = ?" in query for query, _params in captured)
        assert captured[0][1] == (False,)
        assert captured[1][1] == (False, 100, 0)
        assert captured[2][1] == (17, False)
        assert not any("deleted = 0" in query or "deleted = 1" in query for query, _params in captured)


def test_flashcard_template_update_returns_false_for_deleted_template():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")

        template_id = db.add_flashcard_template(
            name="Delete then update",
            model_type="basic",
            front_template="What does {{term}} mean?",
            back_template="{{definition}}",
            placeholder_definitions=[
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        )
        template = db.get_flashcard_template(template_id)
        assert template is not None
        assert db.soft_delete_flashcard_template(template_id, expected_version=template["version"]) is True

        assert (
            db.update_flashcard_template(
                template_id,
                {"name": "Should not update"},
                expected_version=template["version"] + 1,
            )
            is False
        )


def test_flashcard_template_empty_update_is_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")

        template_id = db.add_flashcard_template(
            name="No-op update",
            model_type="basic",
            front_template="What does {{term}} mean?",
            back_template="{{definition}}",
            placeholder_definitions=[
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        )
        before = db.get_flashcard_template(template_id)
        assert before is not None

        assert db.update_flashcard_template(template_id, {}, expected_version=before["version"]) is True

        after = db.get_flashcard_template(template_id)
        assert after is not None
        assert after["version"] == before["version"]
        assert after["last_modified"] == before["last_modified"]


def test_flashcard_template_serialize_logs_invalid_placeholder_json(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        warnings: list[str] = []

        def _capture_warning(message, *args):
            warnings.append(str(message).format(*args))

        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.logger.warning",
            _capture_warning,
        )

        serialized = db._serialize_flashcard_template_row(
            {
                "id": 17,
                "name": "Broken JSON",
                "placeholder_definitions_json": "{not valid json}",
            }
        )

        assert serialized["placeholder_definitions"] == []
        assert warnings
        assert any("_serialize_flashcard_template_row" in warning for warning in warnings)
        assert any("17" in warning for warning in warnings)


def test_flashcard_template_sync_log_tracks_create_update_and_delete():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        start_change_id = db.get_latest_sync_log_change_id()

        template_id = db.add_flashcard_template(
            name="Sync template",
            model_type="basic",
            front_template="What does {{term}} mean?",
            back_template="{{definition}}",
            placeholder_definitions=[
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        )
        created = db.get_flashcard_template(template_id)
        assert created is not None
        assert db.update_flashcard_template(
            template_id,
            {"name": "Sync template updated"},
            expected_version=created["version"],
        )
        updated = db.get_flashcard_template(template_id)
        assert updated is not None
        assert db.soft_delete_flashcard_template(template_id, expected_version=updated["version"])

        entries = db.get_sync_log_entries(since_change_id=start_change_id)
        template_entries = [
            entry
            for entry in entries
            if entry["entity"] == "flashcard_templates" and entry["entity_id"] == str(template_id)
        ]

        assert [entry["operation"] for entry in template_entries] == ["create", "update", "delete"]
        assert template_entries[0]["payload"]["name"] == "Sync template"
        assert template_entries[1]["payload"]["name"] == "Sync template updated"


def test_flashcard_template_schema_postgres_registers_sync_log_trigger(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        executed: list[str] = []

        class _Backend:
            def execute(self, query, params=None, connection=None):
                executed.append(str(query))
                return None

        monkeypatch.setattr(db, "backend_type", BackendType.POSTGRESQL)
        monkeypatch.setattr(db, "backend", _Backend())

        db._ensure_flashcard_template_schema_postgres(conn=object())

        assert any("flashcard_templates_sync_log_fn" in query for query in executed)
        assert any("DROP TRIGGER IF EXISTS flashcard_templates_sync_log" in query for query in executed)
        assert any("CREATE TRIGGER flashcard_templates_sync_log" in query for query in executed)


def test_flashcard_template_delete_uses_version_in_update_query(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        executed: list[tuple[str, tuple[object, ...]]] = []

        class _Cursor:
            def __init__(self, row=None, rowcount: int = 0):
                self._row = row
                self.rowcount = rowcount

            def fetchone(self):
                return self._row

        class _Conn:
            def execute(self, query: str, params: tuple[object, ...]):
                normalized_query = query.strip()
                executed.append((normalized_query, params))
                if normalized_query.startswith("SELECT version, deleted FROM flashcard_templates"):
                    return _Cursor((4, 0))
                if normalized_query.startswith("UPDATE flashcard_templates"):
                    return _Cursor(rowcount=1)
                raise AssertionError(f"Unexpected query: {query}")

        @contextmanager
        def _fake_transaction():
            yield _Conn()

        monkeypatch.setattr(db, "transaction", _fake_transaction)

        assert db.soft_delete_flashcard_template(17, expected_version=4) is True

        update_query, update_params = next(
            (query, params)
            for query, params in executed
            if query.startswith("UPDATE flashcard_templates")
        )
        assert "version = ?" in update_query
        assert update_params[-2:] == (4, 0)


def test_flashcard_template_restore_prefers_most_recent_deleted_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        placeholder_definitions = [
            {
                "key": "term",
                "label": "Term",
                "required": True,
                "targets": ["front_template"],
            },
            {
                "key": "definition",
                "label": "Definition",
                "required": True,
                "targets": ["back_template"],
            },
        ]

        first_id = db.add_flashcard_template(
            name="Duplicate tombstone",
            model_type="basic",
            front_template="What does {{term}} mean?",
            back_template="{{definition}}",
            placeholder_definitions=placeholder_definitions,
        )
        first = db.get_flashcard_template(first_id)
        assert first is not None
        assert db.soft_delete_flashcard_template(first_id, expected_version=first["version"]) is True

        now = db._get_current_utc_timestamp_iso()
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO flashcard_templates(
                    name, model_type, front_template, back_template, notes_template, extra_template,
                    placeholder_definitions_json, created_at, last_modified, deleted, client_id, version
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Duplicate tombstone",
                    "basic",
                    "Older tombstone {{term}}",
                    "{{definition}}",
                    None,
                    None,
                    '[{\"key\":\"term\",\"label\":\"Term\",\"required\":true,\"targets\":[\"front_template\"]},'
                    '{"key":"definition","label":"Definition","required":true,"targets":["back_template"]}]',
                    now,
                    now,
                    1,
                    "legacy-test",
                    7,
                ),
            )
            second_id = int(cursor.lastrowid)

        restored_id = db.add_flashcard_template(
            name="Duplicate tombstone",
            model_type="basic",
            front_template="Newest {{term}}",
            back_template="{{definition}}",
            placeholder_definitions=placeholder_definitions,
        )
        restored = db.get_flashcard_template(restored_id)

        assert restored_id == second_id
        assert restored is not None
        assert restored["front_template"] == "Newest {{term}}"


def test_flashcard_template_delete_wraps_backend_database_errors(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")

        @contextmanager
        def _fake_transaction():
            raise BackendDatabaseError("pg down")
            yield

        monkeypatch.setattr(db, "transaction", _fake_transaction)

        with pytest.raises(CharactersRAGDBError, match="Failed to delete flashcard template: pg down"):
            db.soft_delete_flashcard_template(17, expected_version=1)
