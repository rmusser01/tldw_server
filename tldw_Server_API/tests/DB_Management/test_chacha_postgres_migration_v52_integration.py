"""PostgreSQL migration coverage from ChaChaNotes schema v52 to current."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]


def test_postgres_v52_to_current_preserves_questions_and_is_rerunnable(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="emq-v53-postgres", backend=backend)

    try:
        quiz_id = db.create_quiz(name="Preserved quiz")
        question_id = db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Preserved question",
            correct_answer=1,
            options=["A", "B"],
            explanation="Preserved explanation",
        )

        with backend.transaction() as conn:
            backend.execute(
                "ALTER TABLE quiz_questions DROP COLUMN IF EXISTS group_id",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE quiz_questions DROP COLUMN IF EXISTS group_prompt",
                connection=conn,
            )
            backend.execute(
                "UPDATE db_schema_version SET version = %s WHERE schema_name = %s",
                (52, CharactersRAGDB._SCHEMA_NAME),
                connection=conn,
            )

        db._initialize_schema_postgres()

        columns = {
            row["column_name"]
            for row in backend.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = 'quiz_questions'
                   AND column_name IN ('group_id', 'group_prompt')
                """
            ).rows
        }
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        question = db.get_question(question_id)

        assert columns == {"group_id", "group_prompt"}
        assert int(version) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert question is not None
        assert question["question_text"] == "Preserved question"
        assert question["options"] == ["A", "B"]
        assert question["correct_answer"] == 1
        assert question["explanation"] == "Preserved explanation"
        assert question["group_id"] is None
        assert question["group_prompt"] is None

        db.close_connection()
        db._initialize_schema_postgres()

        rerun_version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        rerun_question = db.get_question(question_id)
        assert int(rerun_version) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert rerun_question is not None
        assert rerun_question["question_text"] == "Preserved question"
        assert rerun_question["group_id"] is None
        assert rerun_question["group_prompt"] is None
    finally:
        db.close_all_connections()
