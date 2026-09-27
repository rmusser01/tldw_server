"""Applies client sync-log changes to the server's Media database (sync v1).

Moved out of the sync endpoint module (TASK-13317): it is the component that runs the
SQL, so it belongs with core. The endpoint re-exports it under its old import path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.Sync.sync_contract import (
    ALLOWED_SYNC_OPERATIONS,
    ALLOWED_SYNC_SEND_ENTITIES,
)


class ServerSyncProcessor:
    """Apply legacy client sync changes to a user's media database.

    The legacy `/sync/send` endpoint now returns 410 in favor of Sync v2, but
    this processor remains importable for compatibility tests and direct
    callers that still exercise the old batch-application semantics.
    """

    def __init__(self, db: Any, user_id: str, requesting_client_id: str) -> None:
        self.db = db
        self.user_id = user_id
        self.requesting_client_id = requesting_client_id
        self._server_column_cache: dict[str, list[str]] = {}

    @staticmethod
    def _validate_entity_operation(entity: str, operation: str) -> Optional[str]:
        if entity not in ALLOWED_SYNC_SEND_ENTITIES:
            return f"Unsupported sync entity '{entity}'"
        if operation not in ALLOWED_SYNC_OPERATIONS:
            return f"Unsupported sync operation '{operation}'"
        if entity == "MediaKeywords" and operation not in {"link", "unlink"}:
            return "MediaKeywords only supports 'link'/'unlink' operations"
        if entity != "MediaKeywords" and operation in {"link", "unlink"}:
            return f"Operation '{operation}' is only valid for MediaKeywords"
        return None

    @staticmethod
    def _parse_iso8601_timestamp_utc(timestamp_str: str, *, label: str) -> datetime:
        if not isinstance(timestamp_str, str) or not timestamp_str.strip():
            raise ValueError(f"{label} timestamp is empty or invalid")

        parse_error: ValueError | None = None
        for parse_format in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
        ):
            try:
                parsed_dt = datetime.strptime(timestamp_str.strip(), parse_format)
                if parse_format.endswith("Z") or parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                return parsed_dt.astimezone(timezone.utc)
            except ValueError as exc:
                parse_error = exc

        raise ValueError(
            f"Could not parse {label} timestamp '{timestamp_str}' as ISO-8601 (Z or offset)."
        ) from parse_error

    def apply_client_changes_batch(self, changes: list[dict]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        server_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                for change in sorted(changes, key=lambda item: item["change_id"]):
                    success, error_msg = self._apply_single_client_change_sync(
                        cursor,
                        change,
                        server_timestamp,
                    )
                    if success:
                        continue
                    full_error = f"Failed Change ID {change.get('change_id', 'N/A')}: {error_msg}"
                    errors.append(full_error)
                    raise DatabaseError(
                        f"Failed to apply change {change.get('change_id', 'N/A')}, rolling back batch."
                    )
        except (DatabaseError, sqlite3.Error, ConflictError) as exc:
            if not errors:
                errors.append(f"Transaction rolled back: {exc}")
            return False, errors
        except Exception as exc:
            logger.exception(
                "Unexpected error applying legacy sync batch for user {} from client {}",
                self.user_id,
                self.requesting_client_id,
            )
            errors.append(f"Unexpected server error: {exc}")
            return False, errors

        return True, errors

    def _apply_single_client_change_sync(
        self,
        cursor: sqlite3.Cursor,
        change: dict,
        current_server_time_str: str,
    ) -> tuple[bool, Optional[str]]:
        entity = str(change["entity"])
        entity_uuid = str(change["entity_uuid"])
        operation = str(change["operation"])
        client_version = int(change["version"])
        originating_client_id = str(change["client_id"])
        payload_str = change.get("payload")

        validation_error = self._validate_entity_operation(entity, operation)
        if validation_error:
            return False, validation_error

        if not payload_str and not (entity == "MediaKeywords" and operation in {"link", "unlink"}):
            return False, f"Missing payload for incoming change: {change.get('change_id', 'N/A')}"

        try:
            payload = json.loads(payload_str) if payload_str else {}
        except json.JSONDecodeError as exc:
            return False, f"Failed to decode payload for change ID {change.get('change_id', 'N/A')}: {exc}"

        if entity == "MediaKeywords":
            try:
                self._execute_server_change_sql_sync(
                    cursor,
                    entity,
                    operation,
                    payload,
                    entity_uuid,
                    client_version,
                    originating_client_id,
                    current_server_time_str,
                )
                return True, None
            except (DatabaseError, sqlite3.Error, KeyError, ValueError) as exc:
                return False, f"Error applying MediaKeywords change for {entity_uuid}: {exc}"

        try:
            cursor.execute(
                f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?",  # nosec B608
                (entity_uuid,),
            )
            server_record_info = cursor.fetchone()
            server_version = int(server_record_info[0]) if server_record_info else 0
            server_client_id = server_record_info[1] if server_record_info else None

            if client_version <= server_version:
                if client_version < server_version or originating_client_id == server_client_id:
                    return True, None
                return self._resolve_server_conflict_sync(
                    cursor,
                    change,
                    server_record_info,
                    current_server_time_str,
                )

            self._execute_server_change_sql_sync(
                cursor,
                entity,
                operation,
                payload,
                entity_uuid,
                client_version,
                originating_client_id,
                current_server_time_str,
            )
            return True, None
        except ConflictError:
            cursor.execute(
                f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?",  # nosec B608
                (entity_uuid,),
            )
            return self._resolve_server_conflict_sync(
                cursor,
                change,
                cursor.fetchone(),
                current_server_time_str,
            )
        except (DatabaseError, sqlite3.Error, json.JSONDecodeError, KeyError, InputError, ValueError) as exc:
            return False, f"Error applying single change for {entity} {entity_uuid}: {exc}"

    def _resolve_server_conflict_sync(
        self,
        cursor: sqlite3.Cursor,
        client_change: dict,
        server_record_info: Optional[tuple],
        current_server_time_str: str,
    ) -> tuple[bool, Optional[str]]:
        if server_record_info is None:
            return False, "Cannot resolve conflict without server record state."

        entity = str(client_change["entity"])
        entity_uuid = str(client_change["entity_uuid"])
        operation = str(client_change["operation"])
        client_version = int(client_change["version"])
        originating_client_id = str(client_change["client_id"])
        payload = json.loads(client_change.get("payload") or "{}")
        server_last_modified = (
            server_record_info[2]
            if len(server_record_info) > 2 and isinstance(server_record_info[2], str)
            else "1970-01-01T00:00:00Z"
        )

        try:
            server_dt = self._parse_iso8601_timestamp_utc(server_last_modified, label="server")
            authoritative_dt = self._parse_iso8601_timestamp_utc(current_server_time_str, label="authoritative")
        except ValueError as exc:
            return False, f"Timestamp parsing error during conflict resolution: {exc}"

        if server_dt >= authoritative_dt:
            return True, None

        try:
            self._execute_server_change_sql_sync(
                cursor,
                entity,
                operation,
                payload,
                entity_uuid,
                client_version,
                originating_client_id,
                current_server_time_str,
                force_apply=True,
            )
        except (DatabaseError, sqlite3.Error, ValueError) as exc:
            return False, f"Failed to force apply winning change: {exc}"
        return True, None

    def _execute_server_change_sql_sync(
        self,
        cursor: sqlite3.Cursor,
        entity: str,
        operation: str,
        payload: dict,
        uuid: str,
        client_version: int,
        originating_client_id: str,
        server_timestamp: str,
        force_apply: bool = False,
    ) -> None:
        if entity == "MediaKeywords":
            self._execute_server_media_keyword_sql_sync(cursor, operation, payload)
            return

        table_columns = self._get_table_columns_sync(cursor, entity)
        if not table_columns:
            raise DatabaseError(f"Cannot proceed: Failed to get columns for entity '{entity}'.")

        cursor.execute(f"SELECT version FROM `{entity}` WHERE uuid = ?", (uuid,))  # nosec B608
        current_rec = cursor.fetchone()
        current_server_version = int(current_rec[0]) if current_rec else 0
        target_sql_version = current_server_version + 1 if force_apply else client_version

        optimistic_lock_sql = ""
        optimistic_lock_param: list[int] = []
        expected_base_version = client_version - 1
        if not force_apply and operation in {"update", "delete"} and expected_base_version > 0:
            optimistic_lock_sql = " AND version = ?"
            optimistic_lock_param = [expected_base_version]

        if operation == "create":
            target_sql_version = 1
            all_data = {
                **payload,
                "uuid": uuid,
                "last_modified": server_timestamp,
                "version": target_sql_version,
                "client_id": originating_client_id,
                "deleted": 1 if payload.get("deleted", False) else 0,
            }
            cols_sql: list[str] = []
            placeholders_sql: list[str] = []
            params_list: list[Any] = []
            for col in table_columns:
                if col == "id" or col not in all_data:
                    continue
                value = all_data[col]
                cols_sql.append(f"`{col}`")
                placeholders_sql.append("?")
                params_list.append(1 if isinstance(value, bool) and value else 0 if isinstance(value, bool) else value)
            if not cols_sql:
                raise ValueError(f"Cannot build INSERT for {entity} {uuid}: No columns")
            sql = f"INSERT OR IGNORE INTO `{entity}` ({', '.join(cols_sql)}) VALUES ({', '.join(placeholders_sql)})"
            params_tuple = tuple(params_list)
        elif operation == "update":
            set_clauses: list[str] = []
            params_list = []
            for col in table_columns:
                if col in payload and col not in {"id", "uuid"}:
                    value = payload[col]
                    set_clauses.append(f"`{col}` = ?")
                    params_list.append(1 if isinstance(value, bool) and value else 0 if isinstance(value, bool) else value)
            set_clauses.extend(["`last_modified` = ?", "`version` = ?", "`client_id` = ?", "`deleted` = ?"])
            params_list.extend(
                [
                    server_timestamp,
                    target_sql_version,
                    originating_client_id,
                    1 if payload.get("deleted", False) else 0,
                ]
            )
            where_clause = " WHERE uuid = ?" + optimistic_lock_sql
            sql = f"UPDATE `{entity}` SET {', '.join(set_clauses)}{where_clause}"  # nosec B608
            params_tuple = tuple(params_list + [uuid] + optimistic_lock_param)
        elif operation == "delete":
            where_clause = " WHERE uuid = ?" + optimistic_lock_sql
            sql = (
                f"UPDATE `{entity}` SET deleted = 1, last_modified = ?, "  # nosec B608
                f"version = ?, client_id = ?{where_clause}"
            )
            params_tuple = tuple([server_timestamp, target_sql_version, originating_client_id, uuid] + optimistic_lock_param)
        else:
            raise ValueError(f"Unsupported operation '{operation}' from client for entity '{entity}'")

        try:
            cursor.execute(sql, params_tuple)
            if not force_apply and operation in {"update", "delete"} and optimistic_lock_sql and cursor.rowcount == 0:
                raise ConflictError("Optimistic lock failed applying client change.", entity=entity, identifier=uuid)

            conn = getattr(cursor, "connection", None)
            if conn is None:
                raise DatabaseError("Sync write completed but cursor has no connection for FTS refresh.")
            self.db.sync_refresh_fts_for_entity(
                conn,
                entity=entity,
                entity_uuid=uuid,
                operation=operation,
                payload=payload,
            )
        except sqlite3.IntegrityError as exc:
            raise DatabaseError(f"Integrity error applying client change: {exc}") from exc

    def _get_table_columns_sync(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[list[str]]:
        if table_name in self._server_column_cache:
            return self._server_column_cache[table_name]
        if not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table_name}")

        cursor.execute(f"PRAGMA table_info(`{table_name}`)")  # nosec B608
        columns = [row[1] for row in cursor.fetchall()]
        if not columns:
            return None
        self._server_column_cache[table_name] = columns
        return columns

    def _execute_server_media_keyword_sql_sync(
        self,
        cursor: sqlite3.Cursor,
        operation: str,
        payload: dict,
    ) -> None:
        media_uuid = payload.get("media_uuid")
        keyword_uuid = payload.get("keyword_uuid")
        if not media_uuid or not keyword_uuid:
            raise ValueError(f"Missing UUIDs for MediaKeywords {operation}")

        cursor.execute("SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
        media_rec = cursor.fetchone()
        cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (keyword_uuid,))
        keyword_rec = cursor.fetchone()
        if not media_rec or not keyword_rec:
            return

        if operation == "link":
            cursor.execute(
                "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                (media_rec[0], keyword_rec[0]),
            )
        elif operation == "unlink":
            cursor.execute(
                "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?",
                (media_rec[0], keyword_rec[0]),
            )
        else:
            raise ValueError(f"Unsupported server operation '{operation}' for MediaKeywords entity")
