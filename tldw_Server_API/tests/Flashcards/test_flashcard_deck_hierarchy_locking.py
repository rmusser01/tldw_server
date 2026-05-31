from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


class _FakeCursor:
    def __init__(self, row: dict[str, Any] | None):
        self._row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self._row


class _RecordingConnection:
    def __init__(self, rows: list[dict[str, Any] | None]):
        self._rows = rows
        self.queries: list[str] = []

    def execute(self, query: str, params: tuple[Any, ...]) -> _FakeCursor:
        self.queries.append(query)
        if not self._rows:
            raise AssertionError(f"Unexpected query: {query}")
        return _FakeCursor(self._rows.pop(0))


@pytest.mark.parametrize(
    ("backend_type", "expects_lock"),
    [
        (BackendType.POSTGRESQL, True),
        (BackendType.SQLITE, False),
    ],
)
def test_deck_parent_cycle_check_locks_parent_chain_for_postgres(
    monkeypatch: pytest.MonkeyPatch,
    backend_type: BackendType,
    expects_lock: bool,
) -> None:
    monkeypatch.setattr(
        CharactersRAGDB,
        "backend_type",
        property(lambda _self: backend_type),
    )
    db = object.__new__(CharactersRAGDB)
    conn = _RecordingConnection(
        [
            {"id": 2, "parent_deck_id": 3},
            {"parent_deck_id": 3},
            {"parent_deck_id": None},
        ]
    )

    assert db._validate_deck_parent_locked(conn, deck_id=1, parent_deck_id=2) == 2

    assert len(conn.queries) == 3
    if expects_lock:
        assert all(query.endswith(" FOR UPDATE") for query in conn.queries)
    else:
        assert all("FOR UPDATE" not in query for query in conn.queries)
