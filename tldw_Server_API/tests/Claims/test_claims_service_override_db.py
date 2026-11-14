from contextlib import contextmanager
from types import SimpleNamespace

from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="admin",
        roles=["admin"],
        permissions=[CLAIMS_ADMIN],
        is_admin=True,
    )


def test_resolve_media_db_uses_override_helper_for_sqlite_admin_override(monkeypatch):
    fake_override_db = object()
    helper_calls: list[int] = []

    @contextmanager
    def _fake_override_helper(user_id: int):
        helper_calls.append(user_id)
        yield fake_override_db, f"/tmp/user-{user_id}.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _fake_override_helper, raising=False)
    monkeypatch.setattr(
        claims_service,
        "MediaDatabase",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw override should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda user_id: f"/tmp/user-{user_id}.db")

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    current_user = SimpleNamespace(id=1, role="admin", roles=["admin"], permissions=[])

    with claims_service._resolve_media_db(
        db=db,
        current_user=current_user,
        user_id=7,
        admin_required=True,
        owner_filter=False,
    ) as (target_db, owner_filter):
        assert target_db is fake_override_db
        assert owner_filter is None

    assert helper_calls == [7]


def test_rebuild_claim_clusters_uses_override_helper_for_sqlite_admin_override(monkeypatch):
    class _FakeOverrideDb:
        def rebuild_claim_clusters_exact(self, *, user_id: str, min_size: int):
            return {"status": "ok", "user_id": user_id, "min_size": min_size}

    helper_calls: list[int] = []
    watchlist_calls: list[tuple[object, str]] = []

    @contextmanager
    def _fake_override_helper(user_id: int):
        helper_calls.append(user_id)
        yield _FakeOverrideDb(), f"/tmp/user-{user_id}.db"

    def _fake_watchlist_notifications(db: object, user_id: str) -> dict[str, object]:
        watchlist_calls.append((db, user_id))
        return {"status": "skipped", "reason": "no_subscriptions"}

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _fake_override_helper, raising=False)
    monkeypatch.setattr(
        claims_service,
        "_evaluate_watchlist_cluster_notifications",
        _fake_watchlist_notifications,
        raising=False,
    )
    monkeypatch.setattr(
        claims_service,
        "MediaDatabase",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw override should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda user_id: f"/tmp/user-{user_id}.db")

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    current_user = SimpleNamespace(id=1, role="admin", roles=["admin"], permissions=[])

    result = claims_service.rebuild_claim_clusters(
        min_size=2,
        user_id=7,
        method="exact",
        similarity_threshold=None,
        principal=_admin_principal(),
        current_user=current_user,
        db=db,
    )

    assert result == {
        "status": "ok",
        "user_id": "7",
        "min_size": 2,
        "watchlist_notifications": {"status": "skipped", "reason": "no_subscriptions"},
    }
    assert helper_calls == [7]
    assert len(watchlist_calls) == 1
    assert watchlist_calls[0][1] == "7"


def test_rebuild_claim_clusters_override_path_preserves_watchlist_notifications(monkeypatch):
    class _FakeOverrideDb:
        def rebuild_claim_clusters_exact(self, *, user_id: str, min_size: int):
            assert user_id == "7"
            assert min_size == 2
            return {"status": "ok"}

    fake_override_db = _FakeOverrideDb()
    helper_calls: list[int] = []
    watchlist_calls: list[tuple[object, str]] = []

    @contextmanager
    def _fake_override_helper(user_id: int):
        helper_calls.append(user_id)
        yield fake_override_db, f"/tmp/user-{user_id}.db"

    def _fake_watchlist_notifications(db: object, user_id: str) -> dict[str, object]:
        watchlist_calls.append((db, user_id))
        return {"status": "ok", "inserted": 2}

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _fake_override_helper, raising=False)
    monkeypatch.setattr(
        claims_service,
        "_evaluate_watchlist_cluster_notifications",
        _fake_watchlist_notifications,
        raising=False,
    )
    monkeypatch.setattr(
        claims_service,
        "MediaDatabase",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw override should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda user_id: f"/tmp/user-{user_id}.db")

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    current_user = SimpleNamespace(id=1, role="admin", roles=["admin"], permissions=[])

    result = claims_service.rebuild_claim_clusters(
        min_size=2,
        user_id=7,
        method="exact",
        similarity_threshold=None,
        principal=_admin_principal(),
        current_user=current_user,
        db=db,
    )

    assert result == {"status": "ok", "watchlist_notifications": {"status": "ok", "inserted": 2}}
    assert helper_calls == [7]
    assert watchlist_calls == [(fake_override_db, "7")]


def test_rebuild_all_media_uses_override_helper_for_sqlite_admin_override(monkeypatch):
    class _FakeOverrideDb:
        def execute_query(self, sql: str):
            assert "SELECT id FROM Media" in sql
            return _RowsResult([{"id": 5}, {"id": 8}])

    class _FakeService:
        def __init__(self) -> None:
            self.submissions: list[tuple[int, str, object]] = []

        def submit(self, *, media_id: int, db_path: str, owner_user_id: object = None) -> None:
            self.submissions.append((media_id, db_path, owner_user_id))

    helper_calls: list[int] = []

    @contextmanager
    def _fake_override_helper(user_id: int):
        helper_calls.append(user_id)
        yield _FakeOverrideDb(), f"/tmp/user-{user_id}.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _fake_override_helper, raising=False)
    monkeypatch.setattr(
        claims_service,
        "MediaDatabase",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw override should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda user_id: f"/tmp/user-{user_id}.db")

    svc = _FakeService()
    db = SimpleNamespace(db_path_str="/tmp/base.db")
    current_user = SimpleNamespace(id=1, role="admin", roles=["admin"], permissions=[])

    result = claims_service.rebuild_all_media(
        policy="all",
        user_id=7,
        current_user=current_user,
        db=db,
        rebuild_service=svc,
    )

    assert result == {"status": "accepted", "enqueued": 2, "policy": "all"}
    assert svc.submissions == [(5, "/tmp/user-7.db", "7"), (8, "/tmp/user-7.db", "7")]
    assert helper_calls == [7]


def test_rebuild_claims_fts_uses_override_helper_for_sqlite_admin_override(monkeypatch):
    class _FakeOverrideDb:
        def rebuild_claims_fts(self) -> int:
            return 12

    helper_calls: list[int] = []

    @contextmanager
    def _fake_override_helper(user_id: int):
        helper_calls.append(user_id)
        yield _FakeOverrideDb(), f"/tmp/user-{user_id}.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _fake_override_helper, raising=False)
    monkeypatch.setattr(
        claims_service,
        "MediaDatabase",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw override should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_service, "get_user_media_db_path", lambda user_id: f"/tmp/user-{user_id}.db")

    db = SimpleNamespace()
    current_user = SimpleNamespace(id=1, role="admin", roles=["admin"], permissions=[])

    result = claims_service.rebuild_claims_fts(
        user_id=7,
        current_user=current_user,
        db=db,
    )

    assert result == {"status": "ok", "indexed": 12}
    assert helper_calls == [7]


def test_list_claims_rebuild_media_ids_supports_stale_days_policy() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _FakeDb:
        def execute_query(self, sql: str, params: tuple[object, ...] | None = None):
            calls.append((sql, params or ()))
            return _RowsResult([{"id": 3}, {"id": 9}])

    result = claims_service.list_claims_rebuild_media_ids(
        _FakeDb(),
        policy="stale",
        stale_days=14,
        compare_media_last_modified=False,
        limit=25,
    )

    assert result == [3, 9]
    assert len(calls) == 1
    assert "julianday('now') - julianday(c.lastc) >= ?" in calls[0][0]
    assert calls[0][1] == (14, 25)
