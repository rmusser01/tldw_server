"""Shared backend SQL helper utilities.

These helpers consolidate placeholder conversion, parameter normalisation, and
SQLite → PostgreSQL query rewrites so individual database adapters do not need
bespoke implementations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Optional, Union

from .base import BackendType

ParamsType = Optional[Union[tuple[Any, ...], list[Any], dict[str, Any], Any]]


def normalise_params(params: ParamsType) -> tuple[Any, ...] | dict[str, Any] | None:
    """Normalize parameter containers for backend execution.

    Rules:
    - None stays None
    - tuple/list coerced to tuple (positional placeholders)
    - dict preserved as-is (named placeholders like %(name)s)
    - scalars wrapped in a single-item tuple
    """
    if params is None:
        return None
    if isinstance(params, dict):
        return params
    if isinstance(params, tuple):
        return params
    if isinstance(params, list):
        return tuple(params)
    return (params,)


def convert_sqlite_placeholders_to_postgres(query: str) -> str:
    """Convert SQLite positional placeholders (`?`) to PostgreSQL (`%s`)."""
    if "?" not in query:
        return query

    result: list[str] = []
    in_single = False
    in_double = False
    i = 0
    length = len(query)

    def _prev_non_space(idx: int) -> str:
        j = idx - 1
        while j >= 0 and query[j].isspace():
            j -= 1
        return query[j] if j >= 0 else ""

    def _next_non_space(idx: int) -> str:
        j = idx + 1
        while j < length and query[j].isspace():
            j += 1
        return query[j] if j < length else ""

    def _prev_token(idx: int) -> str:
        j = idx - 1
        while j >= 0 and query[j].isspace():
            j -= 1
        end = j
        while j >= 0 and (query[j].isalnum() or query[j] == "_"):
            j -= 1
        return query[j + 1 : end + 1].upper()

    def _next_token(idx: int) -> str:
        j = idx + 1
        while j < length and query[j].isspace():
            j += 1
        if j >= length:
            return ""
        ch = query[j]
        if ch in ("'", '"'):
            return ch
        start = j
        while j < length and (query[j].isalnum() or query[j] == "_"):
            j += 1
        return query[start:j].upper()

    def _is_jsonb_operator(idx: int) -> bool:
        # Preserve Postgres JSONB operators: ?, ?|, ?&
        if idx + 1 < length and query[idx + 1] in ("|", "&", "?"):
            return True
        next_ch = _next_non_space(idx)
        if next_ch == "?":
            return True
        prev_token = _prev_token(idx)
        next_token = _next_token(idx)
        # CASE introduces expressions on the left; END closes one on the right.
        # Keep `CASE ... END ? 'key'` and `payload ? CASE ... END` as JSONB.
        if prev_token in {"CASE", "WHEN", "THEN", "ELSE"} or next_token in {
            "WHEN", "THEN", "ELSE", "END",
        }:
            return False
        if prev_token in {
            "LIMIT",
            "OFFSET",
            "WHERE",
            "AND",
            "OR",
            "IN",
            "VALUES",
            "SET",
            "SELECT",
            "UPDATE",
            "INSERT",
            "DELETE",
            "ORDER",
            "BY",
            "GROUP",
            "HAVING",
            "RETURNING",
        }:
            return False
        if next_token in {
            "LIMIT",
            "OFFSET",
            "WHERE",
            "AND",
            "OR",
            "IN",
            "VALUES",
            "SET",
            "SELECT",
            "UPDATE",
            "INSERT",
            "DELETE",
            "ORDER",
            "BY",
            "GROUP",
            "HAVING",
            "RETURNING",
        }:
            return False
        if next_token in {"'", '"'}:
            return True
        prev_ch = _prev_non_space(idx)
        if not prev_ch or not next_ch:
            return False
        prev_is_expr = prev_ch.isalnum() or prev_ch in "_)]}\"'"
        next_is_expr = next_ch.isalpha() or next_ch in "_'\"?"
        return prev_is_expr and next_is_expr

    while i < length:
        ch = query[i]

        if ch == "'" and not in_double:
            if in_single:
                if i + 1 < length and query[i + 1] == "'":
                    result.append("''")
                    i += 2
                    continue
                in_single = False
                result.append(ch)
                i += 1
                continue
            in_single = True
            result.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            if in_double:
                if i + 1 < length and query[i + 1] == '"':
                    result.append('""')
                    i += 2
                    continue
                in_double = False
                result.append(ch)
                i += 1
                continue
            in_double = True
            result.append(ch)
            i += 1
            continue

        if ch == "?" and not in_single and not in_double:
            if _is_jsonb_operator(i):
                result.append(ch)
                i += 1
                continue
            result.append("%s")
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def replace_insert_or_ignore(query: str) -> str:
    """Translate SQLite `INSERT OR IGNORE` statements into Postgres-compatible form."""
    if 'INSERT OR IGNORE' not in query.upper():
        return query

    pattern = re.compile(r'INSERT\s+OR\s+IGNORE\s+INTO', re.IGNORECASE)
    replaced = pattern.sub('INSERT INTO', query)
    stripped = replaced.rstrip()
    suffix = ''
    if stripped.endswith(';'):
        stripped = stripped[:-1]
        suffix = ';'
    if 'ON CONFLICT' in stripped.upper():
        return stripped + suffix
    return f"{stripped} ON CONFLICT DO NOTHING{suffix}"


def replace_collate_nocase(query: str) -> str:
    """Remove SQLite-specific `COLLATE NOCASE` directives."""
    return re.sub(r'COLLATE\s+NOCASE', '', query, flags=re.IGNORECASE)


_RANDOMBLOB_PATTERN = re.compile(r"lower\s*\(\s*hex\s*\(\s*randomblob\s*\(\s*(\d+)\s*\)\s*\)\s*\)", re.IGNORECASE)
_HEX_RANDOMBLOB_PATTERN = re.compile(r"hex\s*\(\s*randomblob\s*\(\s*(\d+)\s*\)\s*\)", re.IGNORECASE)
_JSON_EXTRACT_PATTERN = re.compile(
    r"json_extract\s*\(\s*([A-Za-z0-9_\.]+)\s*,\s*'\$\.([A-Za-z0-9_]+)'\s*\)",
    re.IGNORECASE,
)
_BOOLEAN_EQ_FALSE_PATTERN = re.compile(
    r"\b((?:is_[A-Za-z0-9_]+)|(?:has_[A-Za-z0-9_]+)|(?:deleted)|(?:enabled))\s*=\s*0\b",
    re.IGNORECASE,
)
_BOOLEAN_EQ_TRUE_PATTERN = re.compile(
    r"\b((?:is_[A-Za-z0-9_]+)|(?:has_[A-Za-z0-9_]+)|(?:deleted)|(?:enabled))\s*=\s*1\b",
    re.IGNORECASE,
)
_RETURNING_PATTERN = re.compile(r"\bRETURNING\b", re.IGNORECASE)

# Heuristic for columns that are booleans in Postgres schema
_LIKELY_BOOLEAN_COLUMN = re.compile(
    r"^(?:is_[A-Za-z0-9_]+|has_[A-Za-z0-9_]+|deleted|enabled)$",
    re.IGNORECASE,
)


def _replace_randomblob_calls(query: str) -> str:
    """Translate SQLite randomblob-based UUID helpers to PostgreSQL."""

    def _lower_hex_sub(match: re.Match[str]) -> str:
        length = match.group(1)
        return f"lower(encode(gen_random_bytes({length}), 'hex'))"

    def _hex_sub(match: re.Match[str]) -> str:
        length = match.group(1)
        return f"encode(gen_random_bytes({length}), 'hex')"

    query = _RANDOMBLOB_PATTERN.sub(_lower_hex_sub, query)
    return _HEX_RANDOMBLOB_PATTERN.sub(_hex_sub, query)


def _replace_json_extract_calls(query: str) -> str:
    """Replace SQLite json_extract usages with PostgreSQL jsonb accessors."""

    def _json_extract_sub(match: re.Match[str]) -> str:
        column = match.group(1)
        path = match.group(2)
        return f"({column} ->> '{path}')"

    return _JSON_EXTRACT_PATTERN.sub(_json_extract_sub, query)


def _ensure_returning_id(query: str) -> str:
    """Append a RETURNING clause to INSERT statements when missing.

    Defaults to ``RETURNING id`` for general-purpose tables. For known tables
    that do not use a numeric ``id`` primary key (e.g., workflows tables that
    rely on natural keys), this falls back to ``RETURNING *``.
    """

    if _RETURNING_PATTERN.search(query):
        return query

    match = re.match(r"\s*INSERT\s+INTO\s+", query, flags=re.IGNORECASE)
    if not match:
        return query

    trailing_semicolon = ''
    stripped = query.rstrip()
    if stripped.endswith(';'):
        trailing_semicolon = ';'
        stripped = stripped[:-1].rstrip()

    # Try to detect table name for special-case handling
    try:
        m = re.match(r'\s*INSERT\s+INTO\s+([\w\."]+)', stripped, flags=re.IGNORECASE)
        table_token = (m.group(1) if m else "")
        # Unquote simple identifiers
        table_name = table_token.strip('"').split('.')[-1].lower() if table_token else ""
    except Exception:
        table_name = ""

    special_tables = {
        'workflow_runs', 'workflow_step_runs', 'workflow_events', 'workflow_artifacts'
    }
    if table_name in special_tables:
        return f"{stripped} RETURNING *{trailing_semicolon}"
    return f"{stripped} RETURNING id{trailing_semicolon}"

def _split_csv_ignoring_quotes_and_parens(text: str) -> list[str]:
    """Split a comma-separated list, ignoring commas in quotes/parentheses."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'" and not in_double:
            if in_single:
                # doubled single quote inside string
                if i + 1 < n and text[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_single = False
                buf.append(ch)
                i += 1
                continue
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            if in_double:
                if i + 1 < n and text[i + 1] == '"':
                    buf.append('""')
                    i += 2
                    continue
                in_double = False
                buf.append(ch)
                i += 1
                continue
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if not in_single and not in_double:
            if ch == '(':
                depth += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ')':
                if depth > 0:
                    depth -= 1
                buf.append(ch)
                i += 1
                continue
            if ch == ',' and depth == 0:
                parts.append(''.join(buf).strip())
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append(''.join(buf).strip())
    return parts

def _convert_insert_boolean_literals(query: str) -> str:
    """Convert 0/1 literals to FALSE/TRUE for boolean-like columns in INSERTs.

    Only applies when an explicit column list and VALUES(...) are present, and
    rewrites values where the corresponding column name matches the heuristic
    boolean column pattern.
    """
    upper = query.upper()
    if 'INSERT' not in upper or 'VALUES' not in upper:
        return query
    try:
        # Find start of column list: after "INSERT INTO ... ("
        m = re.search(r"\bINSERT\s+INTO\b[^()]*\(", query, flags=re.IGNORECASE)
        if not m:
            return query
        cols_open = m.end() - 1  # position of '('

        # Find matching ')' for columns
        depth = 1
        i = cols_open + 1
        while i < len(query) and depth > 0:
            ch = query[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            elif ch in ("'", '"'):
                quote = ch
                i += 1
                while i < len(query):
                    c = query[i]
                    if c == quote:
                        if i + 1 < len(query) and query[i + 1] == quote:
                            i += 2
                            continue
                        break
                    i += 1
            i += 1
        if depth != 0:
            return query
        cols_close = i
        columns_block = query[cols_open + 1:cols_close]
        column_names = [c.strip().strip('"') for c in _split_csv_ignoring_quotes_and_parens(columns_block)]

        # Find VALUES keyword, then parse one or more VALUES tuples.
        tail = query[cols_close + 1:]
        m2 = re.search(r"\bVALUES\b", tail, flags=re.IGNORECASE)
        if not m2:
            return query
        values_kw_end = cols_close + 1 + m2.end()
        i = values_kw_end
        while i < len(query) and query[i].isspace():
            i += 1

        first_tuple_start = i
        converted_tuples: list[str] = []
        changed_any = False

        while i < len(query):
            while i < len(query) and query[i].isspace():
                i += 1
            if i >= len(query) or query[i] != '(':
                break

            tuple_open = i
            depth = 1
            i += 1
            in_single = False
            in_double = False
            while i < len(query) and depth > 0:
                ch = query[i]
                nxt = query[i + 1] if i + 1 < len(query) else ""
                if ch == "'" and not in_double:
                    if in_single and nxt == "'":
                        i += 2
                        continue
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    if in_double and nxt == '"':
                        i += 2
                        continue
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            break
                i += 1

            if depth != 0:
                return query

            tuple_close = i
            tuple_block = query[tuple_open + 1:tuple_close]
            values = _split_csv_ignoring_quotes_and_parens(tuple_block)
            if len(column_names) != len(values):
                return query

            tuple_changed = False
            for idx, (col, val) in enumerate(zip(column_names, values)):
                if not _LIKELY_BOOLEAN_COLUMN.match(col):
                    continue
                v = val.strip()
                if re.fullmatch(r"0", v):
                    values[idx] = "FALSE"
                    tuple_changed = True
                elif re.fullmatch(r"1", v):
                    values[idx] = "TRUE"
                    tuple_changed = True

            changed_any = changed_any or tuple_changed
            converted_tuples.append(f"({', '.join(values)})")

            i = tuple_close + 1
            while i < len(query) and query[i].isspace():
                i += 1
            if i < len(query) and query[i] == ',':
                i += 1
                continue
            break

        if not changed_any or not converted_tuples:
            return query

        suffix = query[i:]
        return query[:first_tuple_start] + ", ".join(converted_tuples) + suffix
    except Exception:
        # On any parsing error, return original query unchanged
        return query


def _replace_boolean_comparisons(query: str) -> str:
    """Convert common boolean equality checks to TRUE/FALSE literals.

    Skips replacements inside quoted strings and SQL comments.
    """

    def _false_sub(match: re.Match[str]) -> str:
        column = match.group(1)
        return f"{column} = FALSE"

    def _true_sub(match: re.Match[str]) -> str:
        column = match.group(1)
        return f"{column} = TRUE"

    def _apply_rewrites(segment: str) -> str:
        segment = _BOOLEAN_EQ_FALSE_PATTERN.sub(_false_sub, segment)
        return _BOOLEAN_EQ_TRUE_PATTERN.sub(_true_sub, segment)

    if "=" not in query:
        return query

    result: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    length = len(query)

    def _flush_buf() -> None:
        if buf:
            result.append(_apply_rewrites("".join(buf)))
            buf.clear()

    while i < length:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < length else ""

        if in_line_comment:
            result.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            result.append(ch)
            if ch == "*" and nxt == "/":
                result.append(nxt)
                i += 2
                in_block_comment = False
                continue
            i += 1
            continue

        if in_single:
            result.append(ch)
            if ch == "'" and nxt == "'":
                result.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            result.append(ch)
            if ch == '"' and nxt == '"':
                result.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        # code context
        if ch == "-" and nxt == "-":
            _flush_buf()
            result.append("--")
            i += 2
            in_line_comment = True
            continue

        if ch == "/" and nxt == "*":
            _flush_buf()
            result.append("/*")
            i += 2
            in_block_comment = True
            continue

        if ch == "'":
            _flush_buf()
            result.append(ch)
            in_single = True
            i += 1
            continue

        if ch == '"':
            _flush_buf()
            result.append(ch)
            in_double = True
            i += 1
            continue

        buf.append(ch)
        i += 1

    _flush_buf()
    return "".join(result)


def transform_sqlite_query_for_postgres(
    query: str,
    *,
    replace_insert: bool = True,
    replace_collate: bool = True,
    ensure_returning: bool = False,
) -> str:
    """Apply common SQLite→Postgres rewrites expected across adapters."""
    transformed = query
    if replace_insert:
        transformed = replace_insert_or_ignore(transformed)
    if replace_collate:
        transformed = replace_collate_nocase(transformed)
    transformed = _replace_randomblob_calls(transformed)
    transformed = _replace_json_extract_calls(transformed)
    transformed = _replace_boolean_comparisons(transformed)
    transformed = _convert_insert_boolean_literals(transformed)
    if ensure_returning:
        transformed = _ensure_returning_id(transformed)
    return transformed


def prepare_backend_statement(
    backend_type: BackendType,
    query: str,
    params: ParamsType = None,
    *,
    transformer: Any | None = None,
    apply_default_transform: bool = False,
    ensure_returning: bool = False,
) -> tuple[str, tuple[Any, ...] | dict[str, Any] | None]:
    """Prepare a query/params pair for execution on the configured backend."""
    if backend_type != BackendType.POSTGRESQL:
        return query, params

    returning_requested = ensure_returning
    if transformer is not None:
        query = transformer(query)
        if returning_requested:
            query = _ensure_returning_id(query)
    elif apply_default_transform:
        query = transform_sqlite_query_for_postgres(
            query,
            ensure_returning=returning_requested,
        )
    elif returning_requested:
        query = _ensure_returning_id(query)

    converted_query = convert_sqlite_placeholders_to_postgres(query)
    prepared_params = normalise_params(params)
    return converted_query, prepared_params


def prepare_backend_many_statement(
    backend_type: BackendType,
    query: str,
    params_list: Sequence[ParamsType],
    *,
    transformer: Any | None = None,
    apply_default_transform: bool = False,
    ensure_returning: bool = False,
) -> tuple[str, list[tuple[Any, ...] | dict[str, Any] | None]]:
    """Prepare a batch query/params list for execution on the configured backend."""
    if backend_type != BackendType.POSTGRESQL:
        return query, list(params_list)

    if transformer is not None:
        query = transformer(query)
    elif apply_default_transform:
        query = transform_sqlite_query_for_postgres(
            query,
            ensure_returning=ensure_returning,
        )
    elif ensure_returning:
        query = _ensure_returning_id(query)

    converted_query = convert_sqlite_placeholders_to_postgres(query)
    prepared_params = [normalise_params(params) for params in params_list]
    return converted_query, prepared_params
