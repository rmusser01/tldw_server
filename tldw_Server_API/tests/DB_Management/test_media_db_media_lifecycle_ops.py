from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError


def _load_media_lifecycle_ops():
    return import_module(
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_lifecycle_ops"
    )


class _Cursor:
    def __init__(self, row: dict[str, object] | None = None, rowcount: int = 1) -> None:
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class _Txn:
    def __init__(self, conn: object) -> None:
        self.conn = conn
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.exited += 1
        return False


def test_soft_delete_media_cascade_unlinks_keywords_soft_deletes_children_and_uses_instance_delete_fts_seam() -> None:
    ops = _load_media_lifecycle_ops()
    conn = object()
    txn = _Txn(conn)
    delete_calls: list[tuple[object, int]] = []
    sync_calls: list[tuple[object, str, str, str, int, dict[str, object]]] = []
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    fetchall_calls: list[tuple[str, tuple[object, ...]]] = []

    execute_results = iter(
        [_Cursor(rowcount=1), _Cursor(rowcount=1), _Cursor(rowcount=1), _Cursor(rowcount=1)]
    )

    def _fetchone_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        assert query == (
            "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0 "
            "AND system_operation_id IS NULL"
        )
        assert params == (37,)
        return {"uuid": "media-uuid", "version": 4}

    def _fetchall_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        fetchall_calls.append((query, params))
        if query.startswith("SELECT mk.keyword_id"):
            return [
                {"keyword_id": 3, "keyword_uuid": "kw-a"},
                {"keyword_id": 8, "keyword_uuid": "kw-b"},
            ]
        if "FROM Transcripts" in query:
            return [
                {"id": 101, "uuid": "txn-a", "version": 2},
            ]
        if "FROM MediaChunks" in query:
            return [
                {"id": 202, "uuid": "chunk-a", "version": 5},
            ]
        if "FROM UnvectorizedMediaChunks" in query or "FROM DocumentVersions" in query:
            return []
        raise AssertionError(f"Unexpected query: {query}")

    def _execute_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        execute_calls.append((query, params))
        return next(execute_results)

    def _log_sync_event(_conn, entity: str, uuid: str, action: str, version: int, payload: dict[str, object]):
        assert _conn is conn
        sync_calls.append((_conn, entity, uuid, action, version, payload))

    def _delete_fts_media(_conn, media_id: int):
        delete_calls.append((_conn, media_id))

    db = SimpleNamespace(
        client_id="client-1",
        transaction=lambda: txn,
        _get_current_utc_timestamp_str=lambda: "2026-03-21T12:00:00Z",
        _fetchone_with_connection=_fetchone_with_connection,
        _fetchall_with_connection=_fetchall_with_connection,
        _execute_with_connection=_execute_with_connection,
        _log_sync_event=_log_sync_event,
        _delete_fts_media=_delete_fts_media,
    )

    result = ops.soft_delete_media(db, media_id=37, cascade=True)

    assert result is True
    assert txn.entered == 1
    assert txn.exited == 1
    assert delete_calls == [(conn, 37)]
    assert any(
        "DELETE FROM MediaKeywords" in query and params == (37, 3, 8)
        for query, params in execute_calls
    )
    assert any("UPDATE Transcripts SET deleted = 1" in query for query, _params in execute_calls)
    assert any("UPDATE MediaChunks SET deleted = 1" in query for query, _params in execute_calls)
    assert {entry[1] for entry in sync_calls} >= {"Media", "MediaKeywords", "Transcripts", "MediaChunks"}
    assert fetchall_calls[0][0].startswith("SELECT mk.keyword_id")


@pytest.mark.parametrize(
    ("visibility", "org_id", "team_id", "match"),
    [
        ("bogus", None, None, "Invalid visibility"),
        ("team", 42, None, "team_id is required"),
        ("org", None, 7, "org_id is required"),
    ],
)
def test_share_media_rejects_invalid_scope_combinations(
    visibility: str,
    org_id: int | None,
    team_id: int | None,
    match: str,
) -> None:
    ops = _load_media_lifecycle_ops()
    db = SimpleNamespace(client_id="client-1")

    with pytest.raises(InputError, match=match):
        ops.share_media(db, media_id=11, visibility=visibility, org_id=org_id, team_id=team_id)


@pytest.mark.parametrize(
    ("visibility", "org_id", "team_id", "expected_org_id", "expected_team_id"),
    [
        ("personal", None, None, None, None),
        ("team", 42, 7, 42, 7),
        ("org", 42, None, 42, None),
    ],
)
def test_share_media_writes_expected_visibility_scope_values(
    visibility: str,
    org_id: int | None,
    team_id: int | None,
    expected_org_id: int | None,
    expected_team_id: int | None,
) -> None:
    ops = _load_media_lifecycle_ops()
    conn = object()
    txn = _Txn(conn)
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    sync_calls: list[tuple[object, str, str, str, int, dict[str, object]]] = []

    def _fetchone_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        assert query == (
            "SELECT id, uuid, version, visibility, org_id, team_id FROM Media "
            "WHERE id = ? AND deleted = 0 AND system_operation_id IS NULL"
        )
        assert params == (12,)
        return {
            "id": 12,
            "uuid": "media-uuid",
            "version": 1,
            "visibility": "personal",
            "org_id": None,
            "team_id": None,
        }

    def _execute_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        execute_calls.append((query, params))
        return _Cursor(rowcount=1)

    def _log_sync_event(_conn, entity: str, uuid: str, action: str, version: int, payload: dict[str, object]):
        sync_calls.append((_conn, entity, uuid, action, version, payload))

    db = SimpleNamespace(
        client_id="client-1",
        transaction=lambda: txn,
        _get_current_utc_timestamp_str=lambda: "2026-03-21T12:00:00Z",
        _fetchone_with_connection=_fetchone_with_connection,
        _execute_with_connection=_execute_with_connection,
        _log_sync_event=_log_sync_event,
    )

    result = ops.share_media(
        db,
        media_id=12,
        visibility=visibility,
        org_id=org_id,
        team_id=team_id,
    )

    assert result is True
    assert txn.entered == 1
    assert txn.exited == 1
    assert len(execute_calls) == 1
    _, params = execute_calls[0]
    assert params == (
        visibility,
        expected_org_id,
        expected_team_id,
        2,
        "2026-03-21T12:00:00Z",
        "client-1",
        12,
        1,
    )
    assert sync_calls == [
        (
            conn,
            "Media",
            "media-uuid",
            "update",
            2,
            {
                "visibility": visibility,
                "org_id": expected_org_id,
                "team_id": expected_team_id,
                "version": 2,
                "last_modified": "2026-03-21T12:00:00Z",
            },
        )
    ]


def test_unshare_media_routes_through_share_path_to_restore_personal_visibility() -> None:
    ops = _load_media_lifecycle_ops()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _share_media(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    db = SimpleNamespace(share_media=_share_media)

    result = ops.unshare_media(db, media_id=88)

    assert result is True
    assert calls == [((88,), {"visibility": "personal"})]


@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [
        (
            {
                "visibility": "team",
                "org_id": 42,
                "team_id": 7,
                "owner_user_id": 9,
                "client_id": "client-1",
            },
            {
                "visibility": "team",
                "org_id": 42,
                "team_id": 7,
                "owner_user_id": 9,
                "client_id": "client-1",
            },
        ),
        (None, None),
    ],
)
def test_get_media_visibility_returns_payload_and_none_when_media_row_is_missing(
    fetch_result: dict[str, object] | None,
    expected: dict[str, object] | None,
) -> None:
    ops = _load_media_lifecycle_ops()
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _QueryCursor:
        def fetchone(self):
            return fetch_result

    def _execute_query(query: str, params: tuple[object, ...]):
        calls.append((query, params))
        return _QueryCursor()

    db = SimpleNamespace(
        execute_query=_execute_query,
    )

    result = ops.get_media_visibility(db, media_id=51)

    assert calls == [
        (
            "SELECT visibility, org_id, team_id, owner_user_id, client_id "
            "FROM Media WHERE id = ? AND deleted = 0 AND system_operation_id IS NULL",
            (51,),
        )
    ]
    assert result == expected


@pytest.mark.parametrize(
    ("method_name", "initial_row", "update_sql_fragment", "expected_is_trash", "expected_trash_date"),
    [
        (
            "mark_as_trash",
            {"uuid": "media-uuid", "version": 4, "is_trash": 0},
            "UPDATE Media SET is_trash=1, trash_date=?, last_modified=?, version=?, "
            "client_id=? WHERE id=? AND version=? AND system_operation_id IS NULL",
            1,
            "2026-03-21T12:00:00Z",
        ),
        (
            "restore_from_trash",
            {"uuid": "media-uuid", "version": 5, "is_trash": 1},
            "UPDATE Media SET is_trash=0, trash_date=NULL, last_modified=?, version=?, "
            "client_id=? WHERE id=? AND version=? AND system_operation_id IS NULL",
            0,
            None,
        ),
    ],
)
def test_trash_helpers_preserve_transaction_and_sync_update_behavior(
    method_name: str,
    initial_row: dict[str, object],
    update_sql_fragment: str,
    expected_is_trash: int,
    expected_trash_date: str | None,
) -> None:
    ops = _load_media_lifecycle_ops()
    conn = object()
    txn = _Txn(conn)
    fetch_rows = iter(
        [
            initial_row,
            {
                "id": 73,
                "uuid": "media-uuid",
                "version": int(initial_row["version"]) + 1,
                "is_trash": expected_is_trash,
                "trash_date": expected_trash_date,
                "last_modified": "2026-03-21T12:00:00Z",
                "client_id": "client-1",
            },
        ]
    )
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    sync_calls: list[tuple[object, str, str, str, int, dict[str, object]]] = []

    def _fetchone_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        assert params == (73,)
        return next(fetch_rows)

    def _execute_with_connection(_conn, query: str, params: tuple[object, ...]):
        assert _conn is conn
        execute_calls.append((query, params))
        return _Cursor(rowcount=1)

    def _log_sync_event(_conn, entity: str, uuid: str, action: str, version: int, payload: dict[str, object]):
        sync_calls.append((_conn, entity, uuid, action, version, payload))

    db = SimpleNamespace(
        client_id="client-1",
        transaction=lambda: txn,
        _get_current_utc_timestamp_str=lambda: "2026-03-21T12:00:00Z",
        _fetchone_with_connection=_fetchone_with_connection,
        _execute_with_connection=_execute_with_connection,
        _log_sync_event=_log_sync_event,
    )

    result = getattr(ops, method_name)(db, media_id=73)

    assert result is True
    assert txn.entered == 1
    assert txn.exited == 1
    assert execute_calls[0][0] == update_sql_fragment
    assert sync_calls == [
        (
            conn,
            "Media",
            "media-uuid",
            "update",
            int(initial_row["version"]) + 1,
            {
                "id": 73,
                "uuid": "media-uuid",
                "version": int(initial_row["version"]) + 1,
                "is_trash": expected_is_trash,
                "trash_date": expected_trash_date,
                "last_modified": "2026-03-21T12:00:00Z",
                "client_id": "client-1",
            },
        )
    ]
