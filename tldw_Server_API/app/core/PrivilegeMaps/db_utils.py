from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _uses_postgres(pool: Any) -> bool:
    backend_type = str(getattr(pool, "backend_type", "") or "").lower()
    return backend_type == "postgres" or getattr(pool, "pool", None) is not None


def _question_marks_to_dollar_params(query: str, param_count: int) -> str:
    if "?" not in query or "$" in query:
        return query
    if query.count("?") != param_count:
        return query
    parts = query.split("?")
    rebuilt: list[str] = []
    for index, part in enumerate(parts[:-1], start=1):
        rebuilt.append(part)
        rebuilt.append(f"${index}")
    rebuilt.append(parts[-1])
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
