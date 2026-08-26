"""Package-owned PostgreSQL row-level security helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging

    logger = logging.getLogger("media_db_postgres_rls")


def _postgres_policy_exists(db: Any, conn: Any, table: str, policy: str) -> bool:
    """Check whether a named RLS policy exists for the given table."""

    try:
        result = db.backend.execute(
            "SELECT 1 FROM pg_policies WHERE schemaname = current_schema() AND tablename = %s AND policyname = %s",
            (table, policy),
            connection=conn,
        )
        rows = getattr(result, "rows", None)
        return bool(rows)
    except BackendDatabaseError as exc:
        logger.warning(
            f"Failed to inspect Postgres RLS policy '{policy}' on table '{table}': {exc}"
        )
        return False


def _ensure_postgres_rls(db: Any, conn: Any) -> None:
    """Ensure row-level security policies exist for shared content tables."""

    backend = db.backend
    ident = backend.escape_identifier

    org_array = "COALESCE(string_to_array(NULLIF(current_setting('app.org_ids', true), ''), ',')::BIGINT[], ARRAY[]::BIGINT[])"
    team_array = "COALESCE(string_to_array(NULLIF(current_setting('app.team_ids', true), ''), ',')::BIGINT[], ARRAY[]::BIGINT[])"
    current_user = "current_setting('app.current_user_id', true)"
    is_admin = "COALESCE(current_setting('app.is_admin', true), '0') = '1'"
    not_deleted_predicate = f"COALESCE({ident('media')}.deleted, FALSE) = FALSE"

    personal_predicate = (
        f"(COALESCE({ident('media')}.visibility, 'personal') = 'personal' "
        f"AND (COALESCE({ident('media')}.owner_user_id::TEXT, {ident('media')}.client_id) = {current_user}))"
    )
    team_predicate = (
        f"({ident('media')}.visibility = 'team' "
        f"AND {ident('media')}.team_id IS NOT NULL "
        f"AND {ident('media')}.team_id = ANY({team_array}))"
    )
    org_predicate = (
        f"({ident('media')}.visibility = 'org' "
        f"AND {ident('media')}.org_id IS NOT NULL "
        f"AND {ident('media')}.org_id = ANY({org_array}))"
    )
    marked_clone_predicate = (
        f"({ident('media')}.system_operation_kind = 'shared_workspace_clone' "
        f"AND {ident('media')}.system_operation_id IS NOT NULL "
        f"AND length({ident('media')}.system_operation_id) BETWEEN 1 AND 255 "
        f"AND {ident('media')}.system_source_identity IS NOT NULL "
        f"AND length({ident('media')}.system_source_identity) BETWEEN 1 AND 255 "
        f"AND {ident('media')}.system_content_hash ~ '^[0-9a-f]{{64}}$')"
    )
    media_access_predicate = (
        f"({is_admin} OR "
        f"({not_deleted_predicate} AND "
        f"({personal_predicate} OR {team_predicate} OR {org_predicate})) OR "
        f"({personal_predicate} AND {marked_clone_predicate}))"
    )
    pending_keyword_predicate = (
        "(EXISTS ("
        f"SELECT 1 FROM {ident('media')} AS owned_media "  # nosec B608
        f"WHERE owned_media.id = {ident('operationownedclonekeywords')}.media_id "
        f"AND COALESCE(owned_media.owner_user_id::TEXT, owned_media.client_id) = {current_user} "
        "AND owned_media.system_operation_kind = 'shared_workspace_clone' "
        "AND owned_media.system_operation_id IS NOT NULL "
        "AND owned_media.system_source_identity IS NOT NULL "
        "AND owned_media.system_content_hash IS NOT NULL))"
    )

    policy_sets = {
        "media": [
            ("media_visibility_access", media_access_predicate),
        ],
        "sync_log": [
            ("sync_scope_admin", is_admin),
            ("sync_scope_personal", f"{ident('sync_log')}.client_id = {current_user}"),
            (
                "sync_scope_org",
                f"{ident('sync_log')}.org_id IS NOT NULL AND {ident('sync_log')}.org_id = ANY({org_array})",
            ),
            (
                "sync_scope_team",
                f"{ident('sync_log')}.team_id IS NOT NULL AND {ident('sync_log')}.team_id = ANY({team_array})",
            ),
        ],
        "operationownedclonekeywords": [
            ("owned_clone_pending_keyword_access", pending_keyword_predicate),
        ],
    }

    old_media_policies = [
        "media_scope_admin",
        "media_scope_personal",
        "media_scope_org",
        "media_scope_team",
    ]
    for old_policy in old_media_policies:
        try:
            if db._postgres_policy_exists(conn, "media", old_policy):
                backend.execute(
                    f"DROP POLICY IF EXISTS {backend.escape_identifier(old_policy)} ON {ident('media')}",
                    connection=conn,
                )
                logger.debug(f"Dropped old media policy: {old_policy}")
        except BackendDatabaseError as exc:
            logger.warning(f"Could not drop old media policy '{old_policy}': {exc}")

    try:
        backend.execute(
            f"ALTER TABLE {ident('media')} ENABLE ROW LEVEL SECURITY",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('media')} FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning(f"Could not enable RLS for media table: {exc}")

    for policy_name, predicate in policy_sets["media"]:
        try:
            try:
                backend.execute(
                    f"DROP POLICY IF EXISTS {backend.escape_identifier(policy_name)} ON {ident('media')}",
                    connection=conn,
                )
            except BackendDatabaseError as exc:
                logger.warning(
                    f"Could not drop existing media policy '{policy_name}': {exc}"
                )
            backend.execute(
                f"""
                CREATE POLICY {backend.escape_identifier(policy_name)} ON {ident('media')}
                FOR ALL
                USING ({predicate})
                WITH CHECK ({predicate})
                """,
                connection=conn,
            )
        except BackendDatabaseError as exc:
            logger.warning(f"Skipping creation of media policy '{policy_name}': {exc}")

    pending_table = ident("operationownedclonekeywords")
    try:
        backend.execute(
            f"ALTER TABLE {pending_table} ENABLE ROW LEVEL SECURITY",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {pending_table} FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning(f"Could not enable RLS for pending clone keywords: {exc}")

    for policy_name, predicate in policy_sets["operationownedclonekeywords"]:
        try:
            backend.execute(
                f"DROP POLICY IF EXISTS {backend.escape_identifier(policy_name)} "
                f"ON {pending_table}",
                connection=conn,
            )
            backend.execute(
                f"""
                CREATE POLICY {backend.escape_identifier(policy_name)} ON {pending_table}
                FOR ALL
                USING ({predicate})
                WITH CHECK ({predicate})
                """,
                connection=conn,
            )
        except BackendDatabaseError as exc:
            logger.warning(
                f"Skipping creation of pending clone keyword policy '{policy_name}': {exc}"
            )

    try:
        backend.execute(
            f"ALTER TABLE {ident('sync_log')} ENABLE ROW LEVEL SECURITY",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('sync_log')} FORCE ROW LEVEL SECURITY",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning(f"Could not enable RLS for sync_log table: {exc}")

    for policy_name, predicate in policy_sets["sync_log"]:
        try:
            if not db._postgres_policy_exists(conn, "sync_log", policy_name):
                backend.execute(
                    f"""
                    CREATE POLICY {backend.escape_identifier(policy_name)} ON {ident('sync_log')}
                    FOR ALL
                    USING ({predicate})
                    WITH CHECK ({predicate})
                    """,
                    connection=conn,
                )
        except BackendDatabaseError as exc:
            logger.warning(f"Skipping creation of sync_log policy '{policy_name}': {exc}")


__all__ = [
    "_ensure_postgres_rls",
    "_postgres_policy_exists",
]
