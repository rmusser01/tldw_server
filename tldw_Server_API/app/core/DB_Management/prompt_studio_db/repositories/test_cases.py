"""Prompt Studio test cases: inputs and expected outputs a prompt is evaluated against."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from contextlib import suppress
from typing import Any, Optional, Union

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS
from tldw_Server_API.app.core.DB_Management.Prompts_DB import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

_JSON_FIELDS = frozenset({"inputs", "expected_outputs", "actual_outputs"})
_BOOL_FIELDS = frozenset({"is_golden", "is_generated"})
# Other keys in `updates` are ignored, as before; column names are interpolated.
_UPDATABLE_FIELDS = _JSON_FIELDS | _BOOL_FIELDS | {"name", "description", "tags", "signature_id"}

_INSERT_SQL = """
    INSERT INTO prompt_studio_test_cases (
        uuid, project_id, signature_id, name, description,
        inputs, expected_outputs, actual_outputs, tags,
        is_golden, is_generated, client_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    RETURNING *
"""


def _serialise_tags(tags: Optional[Union[str, Iterable[str]]]) -> Optional[str]:
    """Tags are stored as one comma-separated string."""
    if tags is None:
        return None
    if isinstance(tags, str):
        return tags
    try:
        return ",".join(str(tag).strip() for tag in tags if str(tag).strip()) or None
    except TypeError:
        return None


def _parse_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return []
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    return []


def _format_record(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Tags as a list, flags as bools and JSON decoded, whichever backend produced the row."""
    if record is None:
        return None
    normalised = dict(record)
    normalised["tags"] = _parse_tags(normalised.get("tags"))
    for field in ("is_golden", "is_generated", "deleted"):
        if normalised.get(field) is not None:
            normalised[field] = bool(normalised[field])
    for field in _JSON_FIELDS:
        value = normalised.get(field)
        if isinstance(value, str):
            with suppress(json.JSONDecodeError, TypeError):
                normalised[field] = json.loads(value)
    return normalised


def _json_unless_empty(value: Any) -> Optional[str]:
    """Callers treat empty outputs as absent (see test_case_generator/test_case_io)."""
    return json.dumps(value) if value else None


class TestCasesRepository:
    __test__ = False  # not a pytest class

    def __init__(self, session: Any):
        self.session = session

    # --- helpers -------------------------------------------------------------

    def _format(self, cursor: Any, row: Any) -> Optional[dict[str, Any]]:
        return _format_record(self.session._row_to_dict(cursor, row)) if row else None

    def _rows(self, query: str, params: Any) -> list[dict[str, Any]]:
        db = self.session

        def _read() -> list[dict[str, Any]]:
            cursor = db._execute(query, params)
            return [self._format(cursor, row) for row in cursor.fetchall() if row]

        return run_with_contention_retry(_read)

    def _insert(self, conn: Any, project_id: int, case: dict[str, Any], client_id: Optional[str]) -> dict[str, Any]:
        db = self.session
        cursor = db._cursor_exec(
            conn,
            _INSERT_SQL,
            (
                str(uuid.uuid4()),
                project_id,
                case.get("signature_id"),
                str(case.get("name") or "").strip(),
                case.get("description"),
                json.dumps(case.get("inputs") or {}),
                _json_unless_empty(case.get("expected_outputs")),
                _json_unless_empty(case.get("actual_outputs")),
                _serialise_tags(case.get("tags")),
                bool(case.get("is_golden", False)),
                bool(case.get("is_generated", False)),
                client_id or case.get("client_id") or db.client_id,
            ),
        )
        return self._format(cursor, cursor.fetchone()) or {}

    # --- writes --------------------------------------------------------------

    def create(
        self,
        project_id: int,
        name: str,
        *,
        inputs: dict[str, Any],
        description: Optional[str] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        actual_outputs: Optional[dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        is_golden: bool = False,
        is_generated: bool = False,
        signature_id: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not name.strip():
            raise InputError("Test case name cannot be empty")  # noqa: TRY003
        db = self.session
        case = {
            "name": name, "description": description, "inputs": inputs, "expected_outputs": expected_outputs,
            "actual_outputs": actual_outputs, "tags": tags, "is_golden": is_golden, "is_generated": is_generated,
            "signature_id": signature_id,
        }

        def _create() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                # No database constraint backs this; single creates have always enforced it.
                duplicate = db._cursor_exec(
                    conn,
                    "SELECT 1 FROM prompt_studio_test_cases WHERE project_id = ? AND name = ? AND deleted = FALSE",
                    (project_id, name.strip()),
                ).fetchone()
                if duplicate:
                    raise ConflictError(f"Test case with name '{name}' already exists")  # noqa: TRY003
                return self._insert(conn, project_id, case, client_id)

        try:
            return run_with_contention_retry(_create)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create test case: {exc}") from exc  # noqa: TRY003

    def create_bulk(
        self,
        project_id: int,
        test_cases: list[dict[str, Any]],
        *,
        signature_id: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """All or nothing. Unlike create(), names are not checked for duplicates, as before."""
        db = self.session

        def _create_all() -> list[dict[str, Any]]:
            with db._write_lock, db.transaction() as conn:
                return [
                    self._insert(
                        conn,
                        project_id,
                        {**case, "signature_id": signature_id or case.get("signature_id")},
                        client_id,
                    )
                    for case in test_cases
                ]

        try:
            return run_with_contention_retry(_create_all)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create test cases in bulk: {exc}") from exc  # noqa: TRY003

    def update(self, test_case_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        applied = {key: value for key, value in updates.items() if key in _UPDATABLE_FIELDS}
        if not applied:
            existing = self.get(test_case_id)
            if existing is None:
                raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
            return existing

        db = self.session
        params: list[Any] = []
        for field, value in applied.items():
            if field in _JSON_FIELDS and value is not None:
                value = json.dumps(value)
            elif field in _BOOL_FIELDS and value is not None:
                value = bool(value)  # PostgreSQL rejects an integer for a boolean column
            elif field == "tags":
                value = _serialise_tags(value)
            params.append(value)
        params.append(test_case_id)
        update_sql = (
            "UPDATE prompt_studio_test_cases SET "  # nosec B608 - allowlisted columns
            + ", ".join(f"{field} = ?" for field in applied)
            + ", updated_at = CURRENT_TIMESTAMP WHERE id = ? AND deleted = FALSE RETURNING *"
        )

        def _update() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
                return self._format(cursor, row)

        try:
            return run_with_contention_retry(_update)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to update test case {test_case_id}: {exc}") from exc  # noqa: TRY003

    def delete(self, test_case_id: int, *, hard_delete: bool = False) -> bool:
        db = self.session
        if hard_delete:
            sql = "DELETE FROM prompt_studio_test_cases WHERE id = ? RETURNING id"
        else:
            sql = (
                "UPDATE prompt_studio_test_cases SET deleted = TRUE, deleted_at = CURRENT_TIMESTAMP"
                " WHERE id = ? AND deleted = FALSE RETURNING id"
            )

        def _delete() -> bool:
            with db._write_lock, db.transaction() as conn:
                return db._cursor_exec(conn, sql, (test_case_id,)).fetchone() is not None

        try:
            return run_with_contention_retry(_delete)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to delete test case {test_case_id}: {exc}") from exc  # noqa: TRY003

    # --- reads ---------------------------------------------------------------

    def get(self, test_case_id: int, *, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        query = "SELECT * FROM prompt_studio_test_cases WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = FALSE"
        try:
            rows = self._rows(query, (test_case_id,))
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch test case {test_case_id}: {exc}") from exc  # noqa: TRY003
        return rows[0] if rows else None

    def get_by_ids(self, test_case_ids: Iterable[int], *, include_deleted: bool = False) -> list[dict[str, Any]]:
        identifiers = list(dict.fromkeys(test_case_ids))
        if not identifiers:
            return []
        query = f"SELECT * FROM prompt_studio_test_cases WHERE id IN ({','.join('?' * len(identifiers))})"  # nosec B608
        if not include_deleted:
            query += " AND deleted = FALSE"
        try:
            return self._rows(query, identifiers)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch test cases: {exc}") from exc  # noqa: TRY003

    def list(
        self,
        project_id: int,
        *,
        signature_id: Optional[int] = None,
        is_golden: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        search: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        db = self.session
        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if not include_deleted:
            conditions.append("deleted = FALSE")
        if signature_id is not None:
            conditions.append("signature_id = ?")
            params.append(signature_id)
        if is_golden is not None:
            conditions.append("is_golden = ?")
            params.append(bool(is_golden))
        if tags:
            conditions.append("(" + " OR ".join("tags LIKE ?" for _ in tags) + ")")
            params.extend(f"%{tag}%" for tag in tags)
        if search:
            comparator = "ILIKE" if db.backend_type == BackendType.POSTGRESQL else "LIKE"
            conditions.append(f"(name {comparator} ? OR description {comparator} ?)")
            params.extend([f"%{search}%"] * 2)
        where_clause = " WHERE " + " AND ".join(conditions)
        list_sql = (
            f"SELECT * FROM prompt_studio_test_cases{where_clause}"  # nosec B608
            " ORDER BY is_golden DESC, created_at DESC, id DESC LIMIT ? OFFSET ?"
        )

        def _read() -> tuple[int, list[dict[str, Any]]]:
            count_row = db._execute(f"SELECT COUNT(*) FROM prompt_studio_test_cases{where_clause}", params).fetchone()  # nosec B608
            cursor = db._execute(list_sql, [*params, per_page, max(page - 1, 0) * per_page])
            records = [self._format(cursor, row) for row in cursor.fetchall() if row]
            return (int(count_row[0]) if count_row else 0), records

        try:
            total, records = run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list test cases for project {project_id}: {exc}") from exc  # noqa: TRY003
        if not return_pagination:
            return records
        return {
            "test_cases": records,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page else 0,
            },
        }

    def search(self, project_id: int, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        db = self.session
        if db.backend_type == BackendType.POSTGRESQL:
            fts_query = FTSQueryTranslator.normalize_query(query, "postgresql") or query
            fts_column = db.get_fts_column("prompt_studio_test_cases") or "prompt_studio_test_cases_tsv"
            search_sql = f"""
                SELECT tc.*, ts_rank({fts_column}, to_tsquery('english', ?)) AS rank
                FROM prompt_studio_test_cases tc
                WHERE tc.project_id = ? AND tc.deleted = FALSE
                  AND {fts_column} @@ to_tsquery('english', ?)
                ORDER BY rank DESC
                LIMIT ?
            """  # nosec B608 - column name from the schema config
            params: list[Any] = [fts_query, project_id, fts_query, limit]
        else:
            search_sql = """
                SELECT tc.*
                FROM prompt_studio_test_cases tc
                JOIN prompt_studio_test_cases_fts ON tc.id = prompt_studio_test_cases_fts.rowid
                WHERE tc.project_id = ? AND tc.deleted = FALSE
                  AND prompt_studio_test_cases_fts MATCH ?
                ORDER BY bm25(prompt_studio_test_cases_fts)
                LIMIT ?
            """
            params = [project_id, query, limit]
        try:
            return [{k: v for k, v in row.items() if k != "rank"} for row in self._rows(search_sql, params)]
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to search test cases in project {project_id}: {exc}") from exc  # noqa: TRY003

    def get_by_signature(self, signature_id: int) -> list[dict[str, Any]]:
        try:
            return self._rows(
                "SELECT * FROM prompt_studio_test_cases WHERE signature_id = ? AND deleted = FALSE"
                " ORDER BY is_golden DESC, created_at DESC, id DESC",
                (signature_id,),
            )
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch test cases for signature {signature_id}: {exc}") from exc  # noqa: TRY003

    def get_golden(self, project_id: int, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        try:
            return self._rows(
                "SELECT * FROM prompt_studio_test_cases WHERE project_id = ? AND is_golden = TRUE AND deleted = FALSE"
                " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (project_id, limit, offset),
            )
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to fetch golden test cases for project {project_id}: {exc}") from exc  # noqa: TRY003

    def stats(self, project_id: int) -> dict[str, Any]:
        db = self.session
        live = "FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = FALSE"

        def _count(extra: str) -> int:
            row = db._execute(f"SELECT COUNT(*) {live}{extra}", (project_id,)).fetchone()  # nosec B608
            return int(row[0]) if row else 0

        def _read() -> dict[str, Any]:
            by_signature = db._execute(
                f"SELECT signature_id, COUNT(*) {live} AND signature_id IS NOT NULL GROUP BY signature_id",  # nosec B608
                (project_id,),
            ).fetchall()
            tag_counts: dict[str, int] = {}
            for row in db._execute(f"SELECT tags {live} AND tags IS NOT NULL", (project_id,)).fetchall():  # nosec B608
                for tag in _parse_tags(row[0]):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            return {
                "total": _count(""),
                "golden": _count(" AND is_golden = TRUE"),
                "generated": _count(" AND is_generated = TRUE"),
                "with_expected": _count(" AND expected_outputs IS NOT NULL"),
                "by_signature": {row[0]: row[1] for row in by_signature if row and row[0] is not None},
                # Ties by name, so the order does not depend on the backend's row order.
                "top_tags": sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[:10],
            }

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to compute test case stats for project {project_id}: {exc}") from exc  # noqa: TRY003
