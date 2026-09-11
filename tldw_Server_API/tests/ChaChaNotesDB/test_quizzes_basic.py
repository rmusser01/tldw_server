import contextlib
import gc
import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    reset_managed_sqlite_backends,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

_TEMP_CLEANUP_ATTEMPTS = 50


def _rmtree_retrying_transient_locks(path: str | Path) -> None:
    for attempt in range(_TEMP_CLEANUP_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == _TEMP_CLEANUP_ATTEMPTS - 1:
                raise
            gc.collect()
            time.sleep(min(0.1 * (attempt + 1), 0.5))


@contextmanager
def _temporary_directory():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        _rmtree_retrying_transient_locks(tmpdir)


def _close_temp_chacha_db(db: CharactersRAGDB, db_path: str) -> None:
    db.close_all_connections()
    with contextlib.suppress(Exception):
        reset_managed_sqlite_backends(
            sqlite_targets=[db_path, str(Path(db_path).resolve())],
            mode="hard",
        )


@contextmanager
def _temp_chacha_db(client_id: str = "test"):
    with _temporary_directory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id=client_id)
        try:
            yield db
        finally:
            _close_temp_chacha_db(db, db_path)


def test_temporary_directory_retries_transient_cleanup_locks(monkeypatch):
    real_mkdtemp = tempfile.mkdtemp
    real_rmtree = shutil.rmtree
    cleanup_attempts: list[Path] = []
    cleanup_sleeps: list[float] = []

    def fake_mkdtemp():
        return real_mkdtemp()

    def flaky_rmtree(path):
        cleanup_attempts.append(Path(path))
        if len(cleanup_attempts) == 1:
            raise PermissionError("temporary SQLite file is still locked")
        real_rmtree(path)

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(time, "sleep", cleanup_sleeps.append)

    with _temporary_directory() as tmpdir:
        temp_path = Path(tmpdir)
        (temp_path / "ChaChaNotes.db").write_text("locked once", encoding="utf-8")

    assert cleanup_attempts == [temp_path, temp_path]
    assert cleanup_sleeps == [0.1]
    assert not temp_path.exists()


def test_quizzes_basic_flow():
    with _temp_chacha_db() as db:

        quiz_id = db.create_quiz(name="Quiz One", description="desc", media_id=None)
        assert isinstance(quiz_id, int)

        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="What is 2+2?",
            options=["1", "2", "4", "5"],
            correct_answer=2,
            explanation="2+2=4",
            points=1,
            order_index=0,
        )
        assert isinstance(question_id, int)

        quiz = db.get_quiz(quiz_id)
        assert quiz is not None
        assert quiz["total_questions"] == 1

        public_payload = db.list_questions(quiz_id, include_answers=False, limit=10, offset=0)
        public_questions = public_payload["items"]
        assert public_questions
        assert "correct_answer" not in public_questions[0]

        admin_payload = db.list_questions(quiz_id, include_answers=True, limit=10, offset=0)
        admin_questions = admin_payload["items"]
        assert admin_questions[0]["correct_answer"] == 2

        attempt = db.start_attempt(quiz_id)
        assert attempt["total_possible"] == 1
        assert attempt["questions"]
        assert "correct_answer" not in attempt["questions"][0]

        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": 2, "time_spent_ms": 250}],
        )
        assert result["score"] == 1
        assert result["total_possible"] == 1
        assert result["answers"][0]["is_correct"] is True

        attempts = db.list_attempts(quiz_id=quiz_id, limit=10, offset=0)
        assert attempts["count"] == 1


def test_quiz_workspace_id_persists_and_can_move_between_scopes():
    with _temp_chacha_db() as db:
        db.upsert_workspace("ws-1", "Workspace One")

        quiz_id = db.create_quiz(
            name="Scoped Quiz",
            workspace_id="ws-1",
            workspace_tag="workspace:legacy",
        )
        quiz = db.get_quiz(quiz_id)

        assert quiz is not None
        assert quiz["workspace_id"] == "ws-1"
        assert quiz["workspace_tag"] == "workspace:legacy"

        default_items = db.list_quizzes(limit=20, offset=0)["items"]
        assert all(item["id"] != quiz_id for item in default_items)

        workspace_items = db.list_quizzes(workspace_id="ws-1", limit=20, offset=0)["items"]
        assert [item["id"] for item in workspace_items] == [quiz_id]

        all_items = db.list_quizzes(include_workspace_items=True, limit=20, offset=0)["items"]
        assert any(item["id"] == quiz_id for item in all_items)

        assert db.update_quiz(
            quiz_id,
            {
                "workspace_id": None,
                "workspace_tag": None,
                "expected_version": quiz["version"],
            },
        ) is True
        moved_to_general = db.get_quiz(quiz_id)
        assert moved_to_general is not None
        assert moved_to_general["workspace_id"] is None
        assert moved_to_general["workspace_tag"] is None

        general_items = db.list_quizzes(limit=20, offset=0)["items"]
        assert any(item["id"] == quiz_id for item in general_items)

        assert db.update_quiz(
            quiz_id,
            {
                "workspace_id": "ws-1",
                "workspace_tag": "workspace:ws-1",
                "expected_version": moved_to_general["version"],
            },
        ) is True
        moved_back = db.get_quiz(quiz_id)
        assert moved_back is not None
        assert moved_back["workspace_id"] == "ws-1"
        assert moved_back["workspace_tag"] == "workspace:ws-1"


def test_fill_blank_accepts_delimited_alternates():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Fill blank alternates")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="Spell color/colour",
            correct_answer="color || colour",
            points=1,
            order_index=0,
        )

        attempt = db.start_attempt(quiz_id)
        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": "colour", "time_spent_ms": 100}],
        )
        assert result["score"] == 1
        assert result["answers"][0]["is_correct"] is True


def test_quiz_activity_fields_roundtrip_and_legacy_create_defaults_to_questions():
    with _temp_chacha_db() as db:
        questions_id = db.create_quiz(name="Questions")
        osce_id = db.create_quiz(
            name="OSCE",
            activity_type="osce",
            generation_profile="osce_scenario",
        )

        questions = db.get_quiz(questions_id)
        osce = db.get_quiz(osce_id)
        assert questions is not None
        assert questions["activity_type"] == "questions"
        assert questions["generation_profile"] is None
        assert questions["total_stations"] == 0
        assert osce is not None
        assert osce["activity_type"] == "osce"
        assert osce["generation_profile"] == "osce_scenario"
        assert osce["total_questions"] == 0
        assert osce["total_stations"] == 0


@pytest.mark.parametrize("setting", ["passing_score", "time_limit_seconds"])
def test_osce_quiz_rejects_question_only_settings(setting: str):
    with _temp_chacha_db() as db:
        with pytest.raises(InputError, match=setting):
            db.create_quiz(name="OSCE", activity_type="osce", **{setting: 10})

        quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
        with pytest.raises(InputError, match=setting):
            db.update_quiz(quiz_id, {setting: 10})


def test_quiz_activity_change_requires_empty_quiz_and_preserves_optimistic_locking():
    with _temp_chacha_db() as db:
        empty_id = db.create_quiz(name="Empty")
        empty = db.get_quiz(empty_id)
        assert empty is not None
        assert db.update_quiz(
            empty_id,
            {"activity_type": "osce", "expected_version": empty["version"]},
        ) is True
        switched = db.get_quiz(empty_id)
        assert switched is not None
        assert switched["activity_type"] == "osce"
        assert switched["total_questions"] == 0

        question_quiz_id = db.create_quiz(name="Has questions")
        db.create_question(
            quiz_id=question_quiz_id,
            question_type="true_false",
            question_text="True?",
            correct_answer="true",
        )
        with pytest.raises(ConflictError, match="activity type"):
            db.update_quiz(question_quiz_id, {"activity_type": "osce"})

        station_quiz_id = db.create_quiz(name="Has stations", activity_type="osce")
        with db.transaction() as conn:
            conn.execute("UPDATE quizzes SET total_stations = 1 WHERE id = ?", (station_quiz_id,))
        with pytest.raises(ConflictError, match="activity type"):
            db.update_quiz(station_quiz_id, {"activity_type": "questions"})


def test_quiz_size_sort_uses_activity_relevant_count():
    with _temp_chacha_db() as db:
        questions_id = db.create_quiz(name="Questions")
        osce_id = db.create_quiz(name="OSCE", activity_type="osce")
        with db.transaction() as conn:
            conn.execute("UPDATE quizzes SET total_questions = 2 WHERE id = ?", (questions_id,))
            conn.execute(
                "UPDATE quizzes SET total_questions = 99, total_stations = 4 WHERE id = ?",
                (osce_id,),
            )

        items = db.list_quizzes(
            include_workspace_items=True,
            sort_by="size",
            sort_order="desc",
        )["items"]
        assert [item["id"] for item in items] == [osce_id, questions_id]
        assert items[0]["total_questions"] == 0


def test_question_paths_reject_osce_quizzes():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.create_question(
                quiz_id=quiz_id,
                question_type="true_false",
                question_text="True?",
                correct_answer="true",
            )
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.list_questions(quiz_id)

        now = db._get_current_utc_timestamp_iso()
        with db.transaction() as conn:
            question_id = conn.execute(
                """
                INSERT INTO quiz_questions(
                    quiz_id, question_type, question_text, correct_answer,
                    created_at, last_modified
                ) VALUES (?, 'true_false', 'Injected question', 'true', ?, ?)
                """,
                (quiz_id, now, now),
            ).lastrowid

        with pytest.raises(ConflictError, match="Quiz not found"):
            db.get_question(question_id)
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.update_question(question_id, {"question_text": "Updated"})
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.delete_question(question_id)


def test_normal_attempt_start_and_submit_reject_osce_quizzes():
    with _temp_chacha_db() as db:
        osce_id = db.create_quiz(name="OSCE", activity_type="osce")
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.start_attempt(osce_id)

        questions_id = db.create_quiz(name="Questions")
        question_id = db.create_question(
            quiz_id=questions_id,
            question_type="true_false",
            question_text="True?",
            correct_answer="true",
        )
        attempt = db.start_attempt(questions_id)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE quizzes SET activity_type = 'osce', total_questions = 0 WHERE id = ?",
                (questions_id,),
            )
        with pytest.raises(ConflictError, match="Quiz not found"):
            db.submit_attempt(attempt["id"], [{"question_id": question_id, "user_answer": "true"}])


def test_question_attempt_can_be_submitted_after_quiz_soft_delete():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Questions")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="true_false",
            question_text="True?",
            correct_answer="true",
        )
        attempt = db.start_attempt(quiz_id)

        assert db.delete_quiz(quiz_id) is True
        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": "true"}],
        )

        assert result["score"] == 1
        assert result["answers"][0]["is_correct"] is True


def test_soft_quiz_delete_hides_osce_stations():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
        now = db._get_current_utc_timestamp_iso()
        with db.transaction() as conn:
            station_id = conn.execute(
                """
                INSERT INTO osce_stations(
                    quiz_id, schema_version, content_json, order_index, origin,
                    verification_state, created_at, updated_at
                ) VALUES (?, 'osce.station.v1', '{}', 0, 'manual', 'manually_authored', ?, ?)
                """,
                (quiz_id, now, now),
            ).lastrowid

        assert db.delete_quiz(quiz_id) is True
        row = db.get_connection().execute(
            "SELECT deleted FROM osce_stations WHERE id = ?",
            (station_id,),
        ).fetchone()
        assert row is not None
        assert row["deleted"] == 1


def test_hint_penalty_applies_only_when_hint_used_on_correct_answer():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Hint penalty quiz")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Capital of France?",
            options=["Berlin", "Paris", "Rome"],
            correct_answer=1,
            hint="It's called the city of lights.",
            hint_penalty_points=2,
            points=5,
            order_index=0,
        )

        attempt_with_hint = db.start_attempt(quiz_id)
        with_hint_result = db.submit_attempt(
            attempt_with_hint["id"],
            [{"question_id": question_id, "user_answer": 1, "hint_used": True}],
        )
        assert with_hint_result["score"] == 3
        assert with_hint_result["answers"][0]["is_correct"] is True
        assert with_hint_result["answers"][0]["points_awarded"] == 3
        assert with_hint_result["answers"][0]["hint_used"] is True
        assert with_hint_result["answers"][0]["hint_penalty_points"] == 2

        attempt_without_hint = db.start_attempt(quiz_id)
        without_hint_result = db.submit_attempt(
            attempt_without_hint["id"],
            [{"question_id": question_id, "user_answer": 1, "hint_used": False}],
        )
        assert without_hint_result["score"] == 5
        assert without_hint_result["answers"][0]["points_awarded"] == 5
        assert without_hint_result["answers"][0]["hint_used"] is False

        attempt_wrong_answer = db.start_attempt(quiz_id)
        wrong_answer_result = db.submit_attempt(
            attempt_wrong_answer["id"],
            [{"question_id": question_id, "user_answer": 0, "hint_used": True}],
        )
        assert wrong_answer_result["score"] == 0
        assert wrong_answer_result["answers"][0]["is_correct"] is False
        assert wrong_answer_result["answers"][0]["points_awarded"] == 0


def test_source_citations_roundtrip_through_attempt_snapshot_and_results():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Citation integrity quiz", media_id=501)
        citations = [
            {
                "label": "Transcript chunk",
                "quote": "ATP powers many cellular processes.",
                "media_id": 501,
                "chunk_id": "chunk-12",
                "timestamp_seconds": 87.4,
            }
        ]
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="What molecule carries cellular energy?",
            options=["ATP", "DNA", "RNA"],
            correct_answer=0,
            explanation="ATP stores and transfers usable energy.",
            source_citations=citations,
            points=2,
            order_index=0,
        )

        questions_payload = db.list_questions(quiz_id, include_answers=True, limit=10, offset=0)
        assert questions_payload["items"][0]["source_citations"] == citations

        attempt = db.start_attempt(quiz_id)
        assert attempt["questions"][0]["source_citations"] == citations

        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": 0, "time_spent_ms": 200}],
        )
        assert result["answers"][0]["is_correct"] is True
        assert result["answers"][0]["source_citations"] == citations

        stored_attempt = db.get_attempt(attempt["id"], include_questions=True, include_answers=True)
        assert stored_attempt is not None
        assert stored_attempt["questions"][0]["source_citations"] == citations
        assert stored_attempt["answers"][0]["source_citations"] == citations


def test_fill_blank_supports_fuzzy_threshold_tokens():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Fill blank fuzzy token")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="Organelle",
            correct_answer="~0.93:mitochondrion",
            points=1,
            order_index=0,
        )

        attempt = db.start_attempt(quiz_id)
        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": "mitocondrion", "time_spent_ms": 100}],
        )
        assert result["score"] == 0
        assert result["answers"][0]["is_correct"] is False


def test_fill_blank_supports_json_rule_config():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Fill blank JSON config")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="fill_blank",
            question_text="Variant spelling",
            correct_answer='{"accepted_answers":["sulfur","sulphur"],"fuzzy":true,"fuzzy_threshold":0.85}',
            points=1,
            order_index=0,
        )

        attempt = db.start_attempt(quiz_id)
        result = db.submit_attempt(
            attempt["id"],
            [{"question_id": question_id, "user_answer": "sulphor", "time_spent_ms": 100}],
        )
        assert result["score"] == 1
        assert result["answers"][0]["is_correct"] is True


def test_multi_select_grades_set_equality():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Multi select quiz")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="multi_select",
            question_text="Select prime numbers",
            options=["2", "4", "5", "6"],
            correct_answer=[0, 2],
            points=2,
            order_index=0,
        )

        attempt_correct = db.start_attempt(quiz_id)
        result_correct = db.submit_attempt(
            attempt_correct["id"],
            [{"question_id": question_id, "user_answer": [2, 0], "time_spent_ms": 120}],
        )
        assert result_correct["score"] == 2
        assert result_correct["answers"][0]["is_correct"] is True

        attempt_incorrect = db.start_attempt(quiz_id)
        result_incorrect = db.submit_attempt(
            attempt_incorrect["id"],
            [{"question_id": question_id, "user_answer": [0], "time_spent_ms": 120}],
        )
        assert result_incorrect["score"] == 0
        assert result_incorrect["answers"][0]["is_correct"] is False


def test_matching_grades_key_value_pairs_case_insensitively():
    with _temp_chacha_db() as db:
        quiz_id = db.create_quiz(name="Matching quiz")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="matching",
            question_text="Match each term to its definition",
            options=["ATP", "Mitochondria"],
            correct_answer={"ATP": "Energy currency", "Mitochondria": "Powerhouse"},
            points=2,
            order_index=0,
        )

        attempt_correct = db.start_attempt(quiz_id)
        result_correct = db.submit_attempt(
            attempt_correct["id"],
            [{
                "question_id": question_id,
                "user_answer": {"atp": "energy currency", "mitochondria": "powerhouse"},
                "time_spent_ms": 120
            }],
        )
        assert result_correct["score"] == 2
        assert result_correct["answers"][0]["is_correct"] is True

        attempt_incorrect = db.start_attempt(quiz_id)
        result_incorrect = db.submit_attempt(
            attempt_incorrect["id"],
            [{"question_id": question_id, "user_answer": {"ATP": "Ribosome"}, "time_spent_ms": 120}],
        )
        assert result_incorrect["score"] == 0
        assert result_incorrect["answers"][0]["is_correct"] is False


def test_quiz_schema_migration_v23_to_v24_supports_matching():
    with _temporary_directory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        seed_db = CharactersRAGDB(db_path, client_id="seed")
        seed_quiz_id = seed_db.create_quiz(name="Seed quiz")
        _ = seed_db.create_question(
            quiz_id=seed_quiz_id,
            question_type="multi_select",
            question_text="Seed question",
            options=["A", "B", "C"],
            correct_answer=[0, 2],
            points=1,
            order_index=0,
        )
        _close_temp_chacha_db(seed_db, db_path)

        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                DROP TRIGGER IF EXISTS quiz_questions_ai;
                DROP TRIGGER IF EXISTS quiz_questions_au;
                DROP TRIGGER IF EXISTS quiz_questions_ad;
                DROP TABLE IF EXISTS quiz_questions_fts;
                ALTER TABLE quiz_questions RENAME TO quiz_questions_legacy;
                CREATE TABLE quiz_questions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
                  question_type TEXT NOT NULL CHECK(question_type IN ('multiple_choice', 'multi_select', 'true_false', 'fill_blank')),
                  question_text TEXT NOT NULL,
                  options TEXT,
                  correct_answer TEXT NOT NULL,
                  explanation TEXT,
                  points INTEGER NOT NULL DEFAULT 1,
                  order_index INTEGER NOT NULL DEFAULT 0,
                  tags_json TEXT,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO quiz_questions(
                  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
                  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
                )
                SELECT
                  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
                  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
                FROM quiz_questions_legacy;
                DROP TABLE quiz_questions_legacy;
                UPDATE db_schema_version
                   SET version = 23
                 WHERE schema_name = 'rag_char_chat_schema';
                """
            )
            conn.commit()

        migrator = object.__new__(CharactersRAGDB)
        migrator.db_path_str = db_path
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            migrator._migrate_from_v23_to_v24(conn)
            version_row = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()
            assert version_row is not None
            assert int(version_row["version"]) == 24

            table_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'quiz_questions'"
            ).fetchone()
            assert table_sql_row is not None
            assert "'matching'" in str(table_sql_row["sql"])
            conn.execute(
                """
                INSERT INTO quiz_questions(
                    quiz_id, question_type, question_text, correct_answer
                ) VALUES (?, 'matching', 'Match terms', ?)
                """,
                (seed_quiz_id, json.dumps({"CPU": "Processor"})),
            )


def test_quiz_schema_migration_v24_to_v25_supports_hint_metadata():
    with _temporary_directory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        seed_db = CharactersRAGDB(db_path, client_id="seed")
        seed_quiz_id = seed_db.create_quiz(name="Seed quiz")
        _ = seed_db.create_question(
            quiz_id=seed_quiz_id,
            question_type="matching",
            question_text="Seed question",
            options=["A", "B"],
            correct_answer={"A": "Alpha", "B": "Beta"},
            points=1,
            order_index=0,
        )
        _close_temp_chacha_db(seed_db, db_path)

        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                DROP TRIGGER IF EXISTS quiz_questions_ai;
                DROP TRIGGER IF EXISTS quiz_questions_au;
                DROP TRIGGER IF EXISTS quiz_questions_ad;
                DROP TABLE IF EXISTS quiz_questions_fts;
                ALTER TABLE quiz_questions RENAME TO quiz_questions_legacy;
                CREATE TABLE quiz_questions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
                  question_type TEXT NOT NULL CHECK(question_type IN ('multiple_choice', 'multi_select', 'matching', 'true_false', 'fill_blank')),
                  question_text TEXT NOT NULL,
                  options TEXT,
                  correct_answer TEXT NOT NULL,
                  explanation TEXT,
                  points INTEGER NOT NULL DEFAULT 1,
                  order_index INTEGER NOT NULL DEFAULT 0,
                  tags_json TEXT,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO quiz_questions(
                  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
                  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
                )
                SELECT
                  id, quiz_id, question_type, question_text, options, correct_answer, explanation,
                  points, order_index, tags_json, deleted, client_id, version, created_at, last_modified
                FROM quiz_questions_legacy;
                DROP TABLE quiz_questions_legacy;
                CREATE VIRTUAL TABLE IF NOT EXISTS quiz_questions_fts
                USING fts5(
                  question_text, explanation,
                  content='quiz_questions',
                  content_rowid='id'
                );

                CREATE TRIGGER quiz_questions_ai
                AFTER INSERT ON quiz_questions BEGIN
                  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
                  SELECT new.id, new.question_text, new.explanation
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER quiz_questions_au
                AFTER UPDATE ON quiz_questions BEGIN
                  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
                  VALUES('delete',old.id,old.question_text,old.explanation);

                  INSERT INTO quiz_questions_fts(rowid, question_text, explanation)
                  SELECT new.id, new.question_text, new.explanation
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER quiz_questions_ad
                AFTER DELETE ON quiz_questions BEGIN
                  INSERT INTO quiz_questions_fts(quiz_questions_fts,rowid,question_text,explanation)
                  VALUES('delete',old.id,old.question_text,old.explanation);
                END;

                INSERT INTO quiz_questions_fts(quiz_questions_fts) VALUES('rebuild');
                UPDATE db_schema_version
                   SET version = 24
                 WHERE schema_name = 'rag_char_chat_schema';
                """
            )
            conn.commit()

        migrator = object.__new__(CharactersRAGDB)
        migrator.db_path_str = db_path
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            migrator._migrate_from_v24_to_v25(conn)
            version_row = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()
            assert version_row is not None
            assert int(version_row["version"]) == 25

            columns = {
                row["name"] if isinstance(row, sqlite3.Row) else row[1]
                for row in conn.execute("PRAGMA table_info('quiz_questions')").fetchall()
            }
            assert "hint" in columns
            assert "hint_penalty_points" in columns
            conn.execute(
                """
                INSERT INTO quiz_questions(
                    quiz_id, question_type, question_text, correct_answer,
                    hint, hint_penalty_points
                ) VALUES (?, 'true_false', 'Migrated hint?', 'true', ?, ?)
                """,
                (seed_quiz_id, "Hint text", 2),
            )


def test_sqlite_quiz_client_id_defaults_to_instance_and_preserves_explicit_values():
    with _temp_chacha_db(client_id="sqlite-owner") as db:
        default_id = db.create_quiz(name="Default owner")
        explicit_id = db.create_quiz(name="Explicit owner", client_id="sqlite-explicit")

        assert db.get_quiz(default_id)["client_id"] == "sqlite-owner"
        assert db.get_quiz(explicit_id)["client_id"] == "sqlite-explicit"

        assert db.update_quiz(
            explicit_id,
            {"name": "Changed explicit owner"},
            client_id="sqlite-updater",
        )
        assert db.get_quiz(explicit_id)["client_id"] == "sqlite-updater"


@pytest.mark.timeout(90)
def test_postgres_quiz_domain_operations_are_owner_scoped(
    pg_database_config: DatabaseConfig,
):
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="quiz-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="quiz-attacker", backend=attacker_backend
    )
    try:
        noop_id = owner.create_quiz(name="No-op target")
        update_id = owner.create_quiz(name="Update target")
        soft_delete_id = owner.create_quiz(name="Soft-delete target")
        hard_delete_id = owner.create_quiz(name="Hard-delete target")
        legacy_id = owner.create_quiz(name="Quarantined legacy")
        owner.execute_query(
            "UPDATE quizzes SET client_id = ? WHERE id = ?",
            ("unknown", legacy_id),
        )

        assert attacker.list_quizzes(include_workspace_items=True) == {
            "items": [],
            "count": 0,
        }
        assert not attacker.update_quiz(noop_id, {"expected_version": 1})
        assert not attacker.update_quiz(update_id, {"name": "Captured"})
        assert not attacker.delete_quiz(soft_delete_id, expected_version=1)
        assert not attacker.delete_quiz(hard_delete_id, hard_delete=True)

        assert owner.update_quiz(
            update_id,
            {"name": "Owner update"},
            client_id="attempted-owner-rewrite",
        )

        rows = owner.execute_query(
            "SELECT id, name, deleted, client_id FROM quizzes "
            "WHERE id IN (?, ?, ?, ?, ?) ORDER BY id",
            (noop_id, update_id, soft_delete_id, hard_delete_id, legacy_id),
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        assert by_id[noop_id]["name"] == "No-op target"
        assert by_id[update_id]["name"] == "Owner update"
        assert by_id[update_id]["client_id"] == owner.client_id
        assert not bool(by_id[soft_delete_id]["deleted"])
        assert not bool(by_id[hard_delete_id]["deleted"])
        assert by_id[soft_delete_id]["client_id"] == owner.client_id
        assert by_id[hard_delete_id]["client_id"] == owner.client_id
        assert by_id[legacy_id]["client_id"] == "unknown"
        assert owner.get_quiz(legacy_id, include_deleted=True) is None
        assert legacy_id not in {
            item["id"]
            for item in owner.list_quizzes(
                include_workspace_items=True,
                include_deleted=True,
            )["items"]
        }
    finally:
        attacker.close_all_connections()
        owner.close_all_connections()
