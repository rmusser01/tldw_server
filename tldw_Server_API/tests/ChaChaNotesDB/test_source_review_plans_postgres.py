from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

SOURCE_REVIEW_RLS_ROLE = "tldw_source_review_rls_tester"


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
                SELECT indexname, indexdef
                  FROM pg_indexes
                 WHERE schemaname = current_schema()
                   AND tablename IN ('source_review_plans', 'source_review_occurrences')
                """
            )
        )
        indexes = {row["indexname"]: row["indexdef"] for row in index_rows}
        assert set(indexes).issuperset(  # nosec B101
            {
                "idx_source_review_plans_deleted",
                "idx_source_review_plans_list",
                "idx_source_review_occurrences_plan_id",
                "idx_source_review_occurrences_due_status",
                "idx_source_review_occurrences_deleted",
                "idx_source_review_occurrences_due_list",
            }
        )
        due_list_index = "".join(indexes["idx_source_review_occurrences_due_list"].split()).lower()
        assert "(deleted,due_at,id,status)" in due_list_index  # nosec B101

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

        policy_rows = list(
            backend.execute(
                """
                SELECT tablename, policyname, qual, with_check
                  FROM pg_policies
                 WHERE schemaname = current_schema()
                   AND tablename IN ('source_review_plans', 'source_review_occurrences')
                """
            )
        )
        policies = {row["tablename"]: row for row in policy_rows}
        assert set(policies) == {"source_review_plans", "source_review_occurrences"}  # nosec B101
        assert all(row["qual"] and row["with_check"] for row in policies.values())  # nosec B101

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
        plans, plan_total = db.list_source_review_plans()
        due_rows, due_total = db.list_due_source_review_occurrences(now_utc="2026-07-10T00:00:00Z")
        started = db.start_source_review_occurrence(occurrence_id)
        resumed = db.start_source_review_occurrence(occurrence_id)

        assert plan_total == 1 and plans[0]["id"] == plan_id  # nosec B101
        assert due_total == 1 and due_rows[0]["id"] == occurrence_id  # nosec B101
        assert started["status"] == "in_progress"  # nosec B101
        assert resumed["version"] == started["version"]  # nosec B101
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


def test_postgres_source_review_rls_isolates_two_principals(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner_db = CharactersRAGDB(db_path=":memory:", client_id="101", backend=backend)
    other_db = CharactersRAGDB(db_path=":memory:", client_id="202", backend=backend)
    ident = backend.escape_identifier  # type: ignore[attr-defined]

    try:
        owner_plan_id = owner_db.create_source_review_plan(
            title="Owner source review",
            starts_on="2026-07-09",
            timezone_name="UTC",
            source_bundle_json={
                "items": [
                    {
                        "source_type": "note",
                        "source_id": "owner-note",
                        "label": "Private owner note",
                    }
                ]
            },
            schedule=[
                {
                    "offset_value": 1,
                    "offset_unit": "day",
                    "activity_type": "reread",
                    "due_at": "2026-07-10T00:00:00Z",
                }
            ],
        )
        other_plan_id = other_db.create_source_review_plan(
            title="Other source review",
            starts_on="2026-07-09",
            timezone_name="UTC",
            source_bundle_json={"items": [{"source_type": "note", "source_id": "other-note"}]},
            schedule=[
                {
                    "offset_value": 1,
                    "offset_unit": "day",
                    "activity_type": "reread",
                    "due_at": "2026-07-10T00:00:00Z",
                }
            ],
        )

        with backend.transaction() as conn:
            role_exists = backend.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = ? LIMIT 1",
                (SOURCE_REVIEW_RLS_ROLE,),
                connection=conn,
            ).scalar is not None
            if not role_exists:
                backend.execute(
                    f"CREATE ROLE {ident(SOURCE_REVIEW_RLS_ROLE)} NOLOGIN",
                    connection=conn,
                )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(SOURCE_REVIEW_RLS_ROLE)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON "
                f"source_review_plans, source_review_occurrences TO {ident(SOURCE_REVIEW_RLS_ROLE)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public "
                f"TO {ident(SOURCE_REVIEW_RLS_ROLE)}",
                connection=conn,
            )
            backend.execute(
                f"GRANT {ident(SOURCE_REVIEW_RLS_ROLE)} TO CURRENT_USER",
                connection=conn,
            )

        with backend.transaction() as conn:
            backend.execute(
                f"SET LOCAL ROLE {ident(SOURCE_REVIEW_RLS_ROLE)}",
                connection=conn,
            )
            backend.execute(
                "SELECT set_config('app.current_user_id', ?, true)",
                ("202",),
                connection=conn,
            )
            visible_ids = {
                int(row["id"])
                for row in backend.execute(
                    "SELECT id FROM source_review_plans ORDER BY id",
                    connection=conn,
                )
            }
            visible_occurrences = {
                int(row["plan_id"])
                for row in backend.execute(
                    "SELECT plan_id FROM source_review_occurrences ORDER BY id",
                    connection=conn,
                )
            }
            cross_tenant_update = backend.execute(
                "UPDATE source_review_plans SET title = ? WHERE id = ?",
                ("Cross-tenant edit", owner_plan_id),
                connection=conn,
            )
            cross_tenant_occurrence_update = backend.execute(
                "UPDATE source_review_occurrences SET status = ? WHERE plan_id = ?",
                ("skipped", owner_plan_id),
                connection=conn,
            )

        assert visible_ids == {other_plan_id}  # nosec B101
        assert visible_occurrences == {other_plan_id}  # nosec B101
        assert cross_tenant_update.rowcount == 0  # nosec B101
        assert cross_tenant_occurrence_update.rowcount == 0  # nosec B101
    finally:
        owner_db.close_connection()
        other_db.close_connection()
