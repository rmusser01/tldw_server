import contextlib
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _rmtree_retrying_transient_locks(path: str | Path) -> None:
    for attempt in range(10):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


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

        migrated_db = CharactersRAGDB(db_path, client_id="migration-check")
        try:
            conn = migrated_db.get_connection()
            version_row = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()
            assert version_row is not None
            assert int(version_row["version"]) == CharactersRAGDB._CURRENT_SCHEMA_VERSION

            table_sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'quiz_questions'"
            ).fetchone()
            assert table_sql_row is not None
            table_sql = str(table_sql_row["sql"])
            assert "'matching'" in table_sql

            quiz_id = migrated_db.create_quiz(name="Migrated matching quiz")
            question_id = migrated_db.create_question(
                quiz_id=quiz_id,
                question_type="matching",
                question_text="Match terms",
                options=["CPU", "RAM"],
                correct_answer={"CPU": "Processor", "RAM": "Memory"},
                points=2,
                order_index=0,
            )
            attempt = migrated_db.start_attempt(quiz_id)
            result = migrated_db.submit_attempt(
                attempt["id"],
                [{"question_id": question_id, "user_answer": {"cpu": "processor", "ram": "memory"}}],
            )
            assert result["score"] == 2
            assert result["answers"][0]["is_correct"] is True
        finally:
            _close_temp_chacha_db(migrated_db, db_path)


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

        migrated_db = CharactersRAGDB(db_path, client_id="migration-check")
        try:
            conn = migrated_db.get_connection()
            version_row = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()
            assert version_row is not None
            assert int(version_row["version"]) == CharactersRAGDB._CURRENT_SCHEMA_VERSION

            columns = {
                row["name"] if isinstance(row, sqlite3.Row) else row[1]
                for row in conn.execute("PRAGMA table_info('quiz_questions')").fetchall()
            }
            assert "hint" in columns
            assert "hint_penalty_points" in columns

            quiz_id = migrated_db.create_quiz(name="Migrated hint quiz")
            question_id = migrated_db.create_question(
                quiz_id=quiz_id,
                question_type="multiple_choice",
                question_text="Capital of France?",
                options=["Berlin", "Paris", "Rome"],
                correct_answer=1,
                hint="Think Eiffel Tower.",
                hint_penalty_points=2,
                points=4,
                order_index=0,
            )
            attempt = migrated_db.start_attempt(quiz_id)
            result = migrated_db.submit_attempt(
                attempt["id"],
                [{"question_id": question_id, "user_answer": 1, "hint_used": True}],
            )
            assert result["score"] == 2
            assert result["answers"][0]["hint_used"] is True
            assert result["answers"][0]["hint_penalty_points"] == 2
        finally:
            _close_temp_chacha_db(migrated_db, db_path)
