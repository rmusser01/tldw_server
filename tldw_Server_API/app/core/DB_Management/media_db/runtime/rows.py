"""Row and cursor adapters for extracted media DB runtime helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult


class RowAdapter:
    """Row adapter that supports both key and index access."""

    def __init__(self, row_dict: dict[str, Any], description: Sequence[Sequence[Any]] | None = None):
        self._data = row_dict
        self._cols = [entry[0] for entry in (description or []) if entry and len(entry) > 0]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if self._cols and 0 <= key < len(self._cols):
                return self._data.get(self._cols[key])
            try:
                return list(self._data.values())[key]
            except (IndexError, KeyError, TypeError) as exc:
                raise KeyError(key) from exc
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self):
        if self._cols:
            return iter(self._data.get(column) for column in self._cols)
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"RowAdapter({self._data!r})"


class BackendCursorAdapter:
    """Adapter exposing QueryResult via a sqlite3-like cursor interface."""

    def __init__(self, result: QueryResult):
        self._result = result
        self._index = 0
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description

    def _wrap(self, row: dict[str, Any]) -> RowAdapter:
        return RowAdapter(row, self.description)

    def fetchall(self):
        rows = self._result.rows[self._index:]
        self._index = len(self._result.rows)
        return [self._wrap(row) for row in rows]

    def fetchone(self):
        if self._index >= len(self._result.rows):
            return None
        row = self._result.rows[self._index]
        self._index += 1
        return self._wrap(row)

    def fetchmany(self, size: int | None = None):
        if size is None or size <= 0:
            size = len(self._result.rows) - self._index
        end = min(self._index + size, len(self._result.rows))
        rows = self._result.rows[self._index:end]
        self._index = end
        return [self._wrap(row) for row in rows]

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def close(self) -> None:
        self._result = QueryResult(rows=[], rowcount=0)
        self.rowcount = 0
        self.lastrowid = None
        self.description = None
