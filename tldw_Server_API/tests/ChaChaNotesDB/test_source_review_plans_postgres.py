from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_source_review_schema_and_sync_triggers(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="source-review-pg-test", backend=backend)

    try:
        expected_columns = {
            "source_review_plans": {
                "id",
                "title",
                "starts_on",
                "timezone",
                "source_bundle_json",
                "created_at",
                "last_modified",
                "deleted",
                "client_id",
                "version",
            },
            "source_review_occurrences": {
                "id",
                "plan_id",
                "offset_value",
                "offset_unit",
                "activity_type",
                "due_at",
                "status",
                "launch_state_json",
                "started_at",
                "completed_at",
                "completion_source",
                "created_at",
                "last_modified",
                "deleted",
                "client_id",
                "version",
            },
        }
        for table_name, columns in expected_columns.items():
            assert backend.table_exists(table_name)  # nosec B101
            rows = list(
                backend.execute(
                    """
                    SELECT column_name
                      FROM information_schema.columns
                     WHERE table_schema = current_schema()
                       AND table_name = ?
                    """,
                    (table_name,),
                )
            )
            assert {row["column_name"] for row in rows} == columns  # nosec B101

        index_rows = list(
            backend.execute(
                """
                SELECT indexname
                  FROM pg_indexes
                 WHERE schemaname = current_schema()
                   AND tablename IN ('source_review_plans', 'source_review_occurrences')
                """
            )
        )
        assert {row["indexname"] for row in index_rows}.issuperset(  # nosec B101
            {
                "idx_source_review_plans_deleted",
                "idx_source_review_plans_list",
                "idx_source_review_occurrences_plan_id",
                "idx_source_review_occurrences_due_status",
                "idx_source_review_occurrences_deleted",
                "idx_source_review_occurrences_due_list",
            }
        )

        trigger_rows = list(
            backend.execute(
                """
                SELECT event_object_table, trigger_name
                  FROM information_schema.triggers
                 WHERE trigger_schema = current_schema()
                   AND event_object_table IN ('source_review_plans', 'source_review_occurrences')
                """
            )
        )
        assert {(row["event_object_table"], row["trigger_name"]) for row in trigger_rows}.issuperset(  # nosec B101
            {
                ("source_review_plans", "source_review_plans_sync_create"),
                ("source_review_plans", "source_review_plans_sync_update"),
                ("source_review_plans", "source_review_plans_sync_delete"),
                ("source_review_occurrences", "source_review_occurrences_sync_create"),
                ("source_review_occurrences", "source_review_occurrences_sync_update"),
                ("source_review_occurrences", "source_review_occurrences_sync_delete"),
            }
        )

        sequence_rows = list(
            backend.execute(
                """
                SELECT pg_get_serial_sequence('source_review_plans', 'id') AS plan_sequence,
                       pg_get_serial_sequence('source_review_occurrences', 'id') AS occurrence_sequence
                """
            )
        )
        assert sequence_rows[0]["plan_sequence"]  # nosec B101
        assert sequence_rows[0]["occurrence_sequence"]  # nosec B101

        plan_id = db.create_source_review_plan(
            title="PostgreSQL source review",
            starts_on="2026-07-09",
            timezone_name="UTC",
            source_bundle_json={"items": [{"source_type": "note", "source_id": "pg-note-1", "label": "PG note"}]},
            schedule=[
                {
                    "offset_value": 1,
                    "offset_unit": "day",
                    "activity_type": "quiz",
                    "due_at": "2026-07-10T00:00:00Z",
                }
            ],
        )
        occurrence_id = int(
            list(
                backend.execute(
                    "SELECT id FROM source_review_occurrences WHERE plan_id = ?",
                    (plan_id,),
                )
            )[
                0
            ]["id"]
        )
        db.start_source_review_occurrence(occurrence_id)
        assert db.soft_delete_source_review_plan(plan_id) is True  # nosec B101

        sync_rows = list(
            backend.execute(
                """
                SELECT entity, operation
                  FROM sync_log
                 WHERE entity IN ('source_review_plans', 'source_review_occurrences')
                 ORDER BY change_id
                """
            )
        )
        assert {(row["entity"], row["operation"]) for row in sync_rows}.issuperset(  # nosec B101
            {
                ("source_review_plans", "create"),
                ("source_review_plans", "delete"),
                ("source_review_occurrences", "create"),
                ("source_review_occurrences", "update"),
                ("source_review_occurrences", "delete"),
            }
        )
    finally:
        db.close_connection()
