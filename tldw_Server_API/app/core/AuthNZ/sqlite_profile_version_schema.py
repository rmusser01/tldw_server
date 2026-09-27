"""Canonical SQLite schema ownership for ``users.profile_version``.

This leaf module is shared by AuthNZ migration 091 and synchronous embedded
AuthNZ bootstrap. It has no dependency on either database owner.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import Tokenizer, TokenType

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _classify_sql

SQLITE_PROFILE_VERSION_COLUMN_SQL = (
    "profile_version TEXT NOT NULL DEFAULT "
    "(STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))"
)

_LEGACY_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_PROFILE_VERSION_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)
_PROFILE_VERSION_DEFAULT_PATTERN = re.compile(
    r"^\s*\(?\s*STRFTIME\s*\(\s*'%Y-%m-%dT%H:%M:%f000Z'\s*,\s*'now'\s*\)"
    r"\s*\)?\s*$",
    re.IGNORECASE,
)
_REBUILD_TABLE = "__authnz_users_profile_version_v91"
_CONNECTION_INVALID_ATTRIBUTE = (
    "_authnz_sqlite_profile_version_connection_invalid"
)


def sqlite_profile_version_connection_invalid(failure: BaseException) -> bool:
    """Return whether remediation left its SQLite connection unsafe to reuse."""
    return getattr(failure, _CONNECTION_INVALID_ATTRIBUTE, False) is True


def _record_cleanup_failure(
    primary: BaseException,
    *,
    phase: str,
    cleanup: BaseException,
) -> None:
    try:
        setattr(primary, _CONNECTION_INVALID_ATTRIBUTE, True)
    except Exception:  # noqa: BLE001 - error annotation is best effort
        return
    try:
        primary.add_note(
            "AuthNZ profile_version remediation "
            f"{phase} cleanup failed ({type(cleanup).__name__})"
        )
    except Exception:  # noqa: BLE001 - error annotation is best effort
        return


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _reject_sqlite_profile_shadow_relations(conn: sqlite3.Connection) -> None:
    placeholders = ", ".join("?" for _ in ("users", _REBUILD_TABLE))
    row = conn.execute(
        f"SELECT name FROM temp.sqlite_master "  # nosec B608 - fixed placeholders
        f"WHERE type IN ('table', 'view') AND name IN ({placeholders}) LIMIT 1",
        ("users", _REBUILD_TABLE),
    ).fetchone()
    if row is not None:
        raise RuntimeError(
            "AuthNZ profile_version found an unsafe temporary users relation"
        )


def normalize_sqlite_profile_version(value: Any) -> str:
    """Normalize one legacy SQLite timestamp as canonical UTC RFC3339."""
    if not isinstance(value, str):
        raise RuntimeError("AuthNZ profile_version migration found a null timestamp")
    candidate = value.strip()
    if not _LEGACY_TIMESTAMP_PATTERN.fullmatch(candidate):
        raise RuntimeError("AuthNZ profile_version migration found an invalid timestamp")
    normalized: str | None = None
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        normalized = parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except (OverflowError, ValueError):
        pass
    if normalized is None:
        raise RuntimeError(
            "AuthNZ profile_version migration found an invalid timestamp"
        )
    return normalized


def is_canonical_sqlite_profile_version(value: Any) -> bool:
    if not isinstance(value, str) or not _PROFILE_VERSION_PATTERN.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return False
    return True


def _table_definition_spans(
    create_sql: str,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Locate one CREATE TABLE body and its top-level comma-delimited elements."""
    quote: str | None = None
    in_brackets = False
    in_line_comment = False
    in_block_comment = False
    body_open: int | None = None
    element_start: int | None = None
    depth = 0
    spans: list[tuple[int, int]] = []
    index = 0

    while index < len(create_sql):
        char = create_sql[index]
        next_char = create_sql[index + 1] if index + 1 < len(create_sql) else ""

        if in_line_comment:
            if char in "\r\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if char == quote:
                if next_char == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if in_brackets:
            if char == "]":
                in_brackets = False
            index += 1
            continue

        if char == "-" and next_char == "-":
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "[":
            in_brackets = True
            index += 1
            continue
        if char == "(":
            if body_open is None:
                body_open = index
                element_start = index + 1
                depth = 1
            else:
                depth += 1
            index += 1
            continue
        if char == ")" and body_open is not None:
            depth -= 1
            if depth == 0:
                if element_start is None:
                    raise RuntimeError(
                        "AuthNZ profile_version migration found invalid users schema"
                    )
                spans.append((element_start, index))
                return body_open, index, spans
            index += 1
            continue
        if char == "," and body_open is not None and depth == 1:
            if element_start is None:
                raise RuntimeError(
                    "AuthNZ profile_version migration found invalid users schema"
                )
            spans.append((element_start, index))
            element_start = index + 1
        index += 1

    raise RuntimeError("AuthNZ profile_version migration found invalid users schema")


def _leading_schema_identifier(definition: str) -> str | None:
    candidate = definition.lstrip()
    if not candidate:
        return None
    if candidate[0] in {'"', "`"}:
        quote = candidate[0]
        index = 1
        value: list[str] = []
        while index < len(candidate):
            if candidate[index] == quote:
                if index + 1 < len(candidate) and candidate[index + 1] == quote:
                    value.append(quote)
                    index += 2
                    continue
                return "".join(value)
            value.append(candidate[index])
            index += 1
        return None
    if candidate[0] == "[":
        end = candidate.find("]", 1)
        return candidate[1:end] if end >= 0 else None
    match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", candidate)
    return match.group(0) if match else None


def _starts_with_table_constraint(definition: str) -> bool:
    candidate = definition
    while True:
        candidate = candidate.lstrip()
        if candidate.startswith("--"):
            newline = re.search(r"[\r\n]", candidate[2:])
            if newline is None:
                return False
            candidate = candidate[newline.end() + 1 :]
            continue
        if candidate.startswith("/*"):
            comment_end = candidate.find("*/", 2)
            if comment_end < 0:
                return False
            candidate = candidate[comment_end + 2 :]
            continue
        break
    if not candidate or candidate[0] in {'"', "`", "["}:
        return False
    match = re.match(r"[A-Za-z_][A-Za-z0-9_$]*", candidate)
    return bool(
        match
        and match.group(0).upper()
        in {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"}
    )


def _users_rebuild_sql(create_sql: str, *, profile_exists: bool) -> str:
    body_open, body_close, spans = _table_definition_spans(create_sql)
    body = create_sql[body_open + 1 : body_close]

    if profile_exists:
        matching_spans = [
            (start, end)
            for start, end in spans
            if (_leading_schema_identifier(create_sql[start:end]) or "").casefold()
            == "profile_version"
        ]
        if len(matching_spans) != 1:
            raise RuntimeError(
                "AuthNZ profile_version migration found invalid users schema metadata"
            )
        start, end = matching_spans[0]
        definition = create_sql[start:end]
        leading_whitespace = definition[: len(definition) - len(definition.lstrip())]
        relative_start = start - body_open - 1
        relative_end = end - body_open - 1
        body = (
            body[:relative_start]
            + leading_whitespace
            + SQLITE_PROFILE_VERSION_COLUMN_SQL
            + body[relative_end:]
        )
    else:
        constraint_starts = [
            start
            for start, end in spans
            if _starts_with_table_constraint(create_sql[start:end])
        ]
        if constraint_starts:
            relative_start = constraint_starts[0] - body_open - 1
            body = (
                body[:relative_start]
                + "\n    "
                + SQLITE_PROFILE_VERSION_COLUMN_SQL
                + ","
                + body[relative_start:]
            )
        else:
            body = body + ",\n    " + SQLITE_PROFILE_VERSION_COLUMN_SQL

    return (
        f'CREATE TABLE main."{_REBUILD_TABLE}" ('
        + body
        + ")"
        + create_sql[body_close + 1 :]
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _validate_sqlite_profile_triggers(conn: sqlite3.Connection) -> None:
    trigger_rows = conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'trigger' AND sql IS NOT NULL "
        "UNION ALL "
        "SELECT sql FROM temp.sqlite_master "
        "WHERE type = 'trigger' AND sql IS NOT NULL"
    ).fetchall()
    for (trigger_sql,) in trigger_rows:
        if not isinstance(trigger_sql, str):
            raise RuntimeError("AuthNZ profile_version found an unsafe SQLite trigger")
        try:
            tokens = Tokenizer(dialect="sqlite").tokenize(trigger_sql)
            begin = next(token for token in tokens if token.token_type is TokenType.BEGIN)
            end = next(
                token
                for token in reversed(tokens)
                if token.token_type is TokenType.END
            )
            if begin.end >= end.start:
                raise ValueError
            body = trigger_sql[begin.end + 1 : end.start]
            statements = sqlglot.parse(
                body,
                read="sqlite",
                error_level="RAISE",
                error_message_context=0,
            )
            if not statements or any(statement is None for statement in statements):
                raise ValueError
            for statement in statements:
                classification = _classify_sql(statement.sql(dialect="sqlite"), "sqlite")
                if classification.protected:
                    raise RuntimeError(
                        "AuthNZ profile_version found an unsafe SQLite trigger"
                    )
        except RuntimeError:
            raise
        except (ParseError, TokenError, StopIteration, TypeError, ValueError):
            raise RuntimeError(
                "AuthNZ profile_version found an unsafe SQLite trigger"
            ) from None


def rebuild_sqlite_users_with_profile_version(conn: sqlite3.Connection) -> None:
    """Rebuild ``users`` with canonical metadata inside the caller transaction."""
    _reject_sqlite_profile_shadow_relations(conn)
    if conn.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise RuntimeError(
            "AuthNZ profile_version migration requires foreign keys disabled "
            "for its atomic users-table rebuild"
        )
    if _table_exists(conn, _REBUILD_TABLE):
        raise RuntimeError(
            "AuthNZ profile_version migration found an unexpected rebuild table"
        )

    schema_row = conn.execute(
        "SELECT sql FROM main.sqlite_master "
        "WHERE type = 'table' AND name = 'users'"
    ).fetchone()
    if schema_row is None or not isinstance(schema_row[0], str):
        raise RuntimeError("AuthNZ profile_version migration found invalid users schema")
    create_sql = schema_row[0]

    table_xinfo = conn.execute("PRAGMA main.table_xinfo(users)").fetchall()
    ordinary_columns = [row[1] for row in table_xinfo if len(row) < 7 or row[6] == 0]
    profile_exists = "profile_version" in ordinary_columns
    needs_updated_at = not profile_exists
    if profile_exists:
        needs_updated_at = conn.execute(
            "SELECT 1 FROM main.users WHERE profile_version IS NULL LIMIT 1"
        ).fetchone() is not None
    if needs_updated_at and "updated_at" not in ordinary_columns:
        raise RuntimeError(
            "AuthNZ users table is missing required columns: updated_at, profile_version"
        )
    target_columns = (
        ordinary_columns if profile_exists else [*ordinary_columns, "profile_version"]
    )
    updated_at_index = (
        ordinary_columns.index("updated_at") if needs_updated_at else None
    )
    profile_index = ordinary_columns.index("profile_version") if profile_exists else None

    schema_objects = conn.execute(
        """
        SELECT type, name, sql
        FROM main.sqlite_master
        WHERE tbl_name = 'users'
          AND type IN ('index', 'trigger')
          AND sql IS NOT NULL
        ORDER BY CASE type WHEN 'index' THEN 0 ELSE 1 END, name
        """
    ).fetchall()
    sequence_value: int | None = None
    if _table_exists(conn, "sqlite_sequence"):
        sequence_row = conn.execute(
            "SELECT seq FROM main.sqlite_sequence WHERE name = 'users'"
        ).fetchone()
        if sequence_row is not None:
            sequence_value = int(sequence_row[0])

    conn.execute(_users_rebuild_sql(create_sql, profile_exists=profile_exists))

    quoted_source_columns = ", ".join(
        _quote_identifier(column) for column in ordinary_columns
    )
    quoted_target_columns = ", ".join(
        _quote_identifier(column) for column in target_columns
    )
    placeholders = ", ".join("?" for _ in target_columns)
    select_sql = f"SELECT {quoted_source_columns} FROM main.users"  # nosec B608
    insert_sql = (
        f'INSERT INTO main."{_REBUILD_TABLE}" '  # nosec B608
        f"({quoted_target_columns}) VALUES ({placeholders})"  # nosec B608
    )

    source_cursor = conn.execute(select_sql)
    while batch := source_cursor.fetchmany(500):
        transformed: list[tuple[Any, ...]] = []
        for row in batch:
            values = list(row)
            if profile_index is None:
                if updated_at_index is None:
                    raise RuntimeError(
                        "AuthNZ users table is missing required columns: updated_at"
                    )
                values.append(normalize_sqlite_profile_version(row[updated_at_index]))
            elif values[profile_index] is None:
                if updated_at_index is None:
                    raise RuntimeError(
                        "AuthNZ users table is missing required columns: updated_at"
                    )
                values[profile_index] = normalize_sqlite_profile_version(
                    row[updated_at_index]
                )
            elif not is_canonical_sqlite_profile_version(values[profile_index]):
                raise RuntimeError(
                    "AuthNZ profile_version migration found an invalid existing value"
                )
            transformed.append(tuple(values))
        conn.executemany(insert_sql, transformed)

    conn.execute("DROP TABLE main.users")
    conn.execute(f'ALTER TABLE main."{_REBUILD_TABLE}" RENAME TO users')
    if sequence_value is not None:
        conn.execute(
            "DELETE FROM main.sqlite_sequence WHERE name IN (?, ?)",
            ("users", _REBUILD_TABLE),
        )
        conn.execute(
            "INSERT INTO main.sqlite_sequence(name, seq) VALUES ('users', ?)",
            (sequence_value,),
        )
    for _object_type, _name, object_sql in schema_objects:
        conn.execute(object_sql)


def validate_sqlite_profile_version_readiness(conn: sqlite3.Connection) -> None:
    """Fail closed unless profile-version schema and values are canonical."""
    _reject_sqlite_profile_shadow_relations(conn)
    if not _table_exists(conn, "users"):
        raise RuntimeError(
            "AuthNZ users table is missing required columns: profile_version"
        )
    columns = {
        row[1]: row
        for row in conn.execute("PRAGMA main.table_info(users)").fetchall()
    }
    profile_column = columns.get("profile_version")
    if profile_column is None:
        raise RuntimeError(
            "AuthNZ users table is missing required columns: profile_version"
        )
    column_type = str(profile_column[2] or "").strip().upper()
    not_null = bool(profile_column[3])
    default = profile_column[4]
    if (
        column_type != "TEXT"
        or not not_null
        or not isinstance(default, str)
        or not _PROFILE_VERSION_DEFAULT_PATTERN.fullmatch(default)
    ):
        raise RuntimeError(
            "AuthNZ profile_version readiness validation failed for schema metadata"
        )
    invalid_count = 0
    for (profile_version,) in conn.execute(
        "SELECT profile_version FROM main.users"
    ):
        if not is_canonical_sqlite_profile_version(profile_version):
            invalid_count += 1
    if invalid_count:
        raise RuntimeError(
            "AuthNZ profile_version readiness validation failed for "
            f"{invalid_count} user row(s)"
        )
    _validate_sqlite_profile_triggers(conn)


def remediate_sqlite_profile_version_schema(conn: sqlite3.Connection) -> None:
    """Atomically repair and validate an embedded SQLite users schema."""
    if conn.in_transaction:
        raise RuntimeError(
            "AuthNZ profile_version remediation requires no active transaction"
        )
    _reject_sqlite_profile_shadow_relations(conn)
    _validate_sqlite_profile_triggers(conn)
    try:
        validate_sqlite_profile_version_readiness(conn)
    except RuntimeError:
        pass
    else:
        return

    foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    failure: BaseException | None = None
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        if conn.execute("PRAGMA foreign_keys").fetchone()[0]:
            raise RuntimeError(
                "AuthNZ profile_version remediation could not disable foreign keys"
            )
        conn.execute("BEGIN IMMEDIATE")
        rebuild_sqlite_users_with_profile_version(conn)
        validate_sqlite_profile_version_readiness(conn)
        conn.commit()
    except BaseException as exc:  # noqa: BLE001 - cleanup must survive interruption
        failure = exc
        if conn.in_transaction:
            try:
                conn.rollback()
            except BaseException as cleanup:  # noqa: BLE001 - cleanup must survive interruption
                _record_cleanup_failure(
                    failure,
                    phase="rollback",
                    cleanup=cleanup,
                )
        if conn.in_transaction:
            _record_cleanup_failure(
                failure,
                phase="rollback verification",
                cleanup=RuntimeError("transaction remains active"),
            )

    try:
        conn.execute(f"PRAGMA foreign_keys = {foreign_keys}")
        if int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) != foreign_keys:
            raise RuntimeError("foreign key state was not restored")
    except BaseException as cleanup:  # noqa: BLE001 - cleanup must survive interruption
        if failure is None:
            failure = RuntimeError(
                "AuthNZ profile_version remediation could not restore connection state"
            )
        _record_cleanup_failure(
            failure,
            phase="foreign-key restoration",
            cleanup=cleanup,
        )

    if failure is not None:
        raise failure.with_traceback(failure.__traceback__) from None


def validate_sqlite_profile_version_database(db_path: str | Path) -> None:
    """Validate one file-backed database without assuming a serving owner."""
    with sqlite3.connect(db_path) as conn:
        validate_sqlite_profile_version_readiness(conn)
