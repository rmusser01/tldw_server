from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _uses_postgres(pool: Any) -> bool:
    backend_type = str(getattr(pool, "backend_type", "") or "").lower()
    return backend_type == "postgres" or getattr(pool, "pool", None) is not None


def _question_marks_to_dollar_params(query: str, param_count: int) -> str:
    if "?" not in query:
        return query

    rebuilt: list[str] = []
    placeholder_index = 0
    index = 0
    length = len(query)

    while index < length:
        char = query[index]

        if char == "'":
            start = index
            index += 1
            while index < length:
                if query[index] == "'":
                    if index + 1 < length and query[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            rebuilt.append(query[start:index])
            continue

        if char == '"':
            start = index
            index += 1
            while index < length:
                if query[index] == '"':
                    if index + 1 < length and query[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            rebuilt.append(query[start:index])
            continue

        if char == "-" and index + 1 < length and query[index + 1] == "-":
            start = index
            index += 2
            while index < length and query[index] not in "\r\n":
                index += 1
            rebuilt.append(query[start:index])
            continue

        if char == "/" and index + 1 < length and query[index + 1] == "*":
            start = index
            index += 2
            while index + 1 < length:
                if query[index] == "*" and query[index + 1] == "/":
                    index += 2
                    break
                index += 1
            rebuilt.append(query[start:index])
            continue

        if char == "$" and index + 1 < length and query[index + 1].isdigit():
            return query

        if char == "?":
            placeholder_index += 1
            rebuilt.append(f"${placeholder_index}")
            index += 1
            continue

        rebuilt.append(char)
        index += 1

    if placeholder_index != param_count:
        return query
    return "".join(rebuilt)


async def execute_transaction_sql(
    pool: Any,
    conn: Any,
    query: str,
    params: Sequence[Any] | None = None,
) -> Any:
    """Execute SQL on a raw transaction connection with backend-aware placeholders."""
    if params is None:
        return await conn.execute(query)

    param_tuple = tuple(params)
    if _uses_postgres(pool):
        query = _question_marks_to_dollar_params(query, len(param_tuple))
        return await conn.execute(query, *param_tuple)
    return await conn.execute(query, param_tuple)
