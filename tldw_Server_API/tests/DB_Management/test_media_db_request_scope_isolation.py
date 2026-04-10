from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.backends import factory as backend_factory
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import MediaDbFactory
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError
from tldw_Server_API.app.core.DB_Management.media_db.runtime import session as media_db_session
from tldw_Server_API.app.core.DB_Management.media_db.runtime import validation as media_db_validation


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.warning_calls: list[str] = []
        self.debug_calls: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        try:
            self.info_calls.append(message.format(*args))
        except Exception:
            self.info_calls.append(str(message))

    def warning(self, message, *args, **kwargs) -> None:
        try:
            self.warning_calls.append(message.format(*args))
        except Exception:
            self.warning_calls.append(str(message))

    def debug(self, message, *args, **kwargs) -> None:
        try:
            self.debug_calls.append(message.format(*args))
        except Exception:
            self.debug_calls.append(str(message))


def test_factory_returns_distinct_sessions_for_distinct_scopes() -> None:
    factory = MediaDbFactory.for_sqlite_path(":memory:", client_id="scope-test")

    first = factory.for_request(org_id=10, team_id=20)
    second = factory.for_request(org_id=11, team_id=21)

    assert first is not second
    assert (first.org_id, first.team_id) == (10, 20)
    assert (second.org_id, second.team_id) == (11, 21)


def test_media_db_session_runtime_module_no_longer_mentions_media_db_v2_in_source() -> None:
    source = Path(media_db_session.__file__).read_text()
    assert "Media_DB_v2" not in source


def test_cached_factory_does_not_mutate_existing_session_scope() -> None:
    factory = MediaDbFactory.for_sqlite_path(":memory:", client_id="scope-test")

    first = factory.for_request(org_id=1, team_id=2)
    _ = factory.for_request(org_id=3, team_id=4)

    assert (first.org_id, first.team_id) == (1, 2)


def test_for_sqlite_path_provisions_shared_backend_for_request_scopes(monkeypatch) -> None:
    sentinel_backend = object()
    backend_calls: list[tuple[str, str]] = []
    database_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        media_db_session,
        "_create_sqlite_backend",
        lambda db_path, client_id: backend_calls.append((db_path, client_id)) or sentinel_backend,
    )

    def _fake_database_factory(**kwargs):
        database_calls.append(kwargs)
        return SimpleNamespace(
            default_org_id=None,
            default_team_id=None,
            release_context_connection=lambda: None,
        )

    factory = media_db_session.MediaDbFactory.for_sqlite_path(
        "/tmp/scope-test.db",
        client_id="scope-test",
    )
    factory.database_factory = _fake_database_factory

    first = factory.for_request(org_id=10, team_id=20)
    second = factory.for_request(org_id=11, team_id=21)

    assert backend_calls == [("/tmp/scope-test.db", "scope-test")]
    assert factory.backend is sentinel_backend
    assert database_calls == [
        {
            "db_path": "/tmp/scope-test.db",
            "client_id": "scope-test",
            "backend": sentinel_backend,
        },
        {
            "db_path": "/tmp/scope-test.db",
            "client_id": "scope-test",
            "backend": sentinel_backend,
        },
    ]
    assert (first.org_id, first.team_id) == (10, 20)
    assert (second.org_id, second.team_id) == (11, 21)


def test_get_or_create_media_db_factory_logs_one_info_on_cache_miss(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(deps, "logger", recorder, raising=True)
    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=False)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)

    class FakeFactory:
        pass

    monkeypatch.setattr(
        deps.MediaDbFactory,
        "for_sqlite_path",
        classmethod(lambda cls, db_path, client_id: FakeFactory()),
        raising=True,
    )

    first = deps._get_or_create_media_db_factory(_make_user())
    second = deps._get_or_create_media_db_factory(_make_user())

    assert first is second
    expected_target = str(Path("/tmp/1.db").resolve())
    assert recorder.info_calls == [
        f"Initializing MediaDbFactory user_id=1 backend=sqlite target={expected_target}"
    ]


def _make_user(user_id: int = 1) -> User:
    return User(
        id=user_id,
        username=f"user-{user_id}",
        roles=["user"],
        permissions=[],
        is_admin=False,
    )


def test_get_or_create_media_db_factory_caches_factory_per_user(monkeypatch) -> None:
    created: list[object] = []

    class FakeFactory:
        def __init__(self, *, db_path: str, client_id: str):
            self.db_path = db_path
            self.client_id = client_id
            created.append(self)

    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=False)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.MediaDbFactory,
        "for_sqlite_path",
        classmethod(lambda cls, db_path, client_id: FakeFactory(db_path=db_path, client_id=client_id)),
        raising=True,
    )

    first = deps._get_or_create_media_db_factory(_make_user())
    second = deps._get_or_create_media_db_factory(_make_user())

    assert first is second
    assert len(created) == 1


def test_resolve_media_db_for_user_reuses_shared_sqlite_backend_per_user(monkeypatch) -> None:
    sentinel_backend = object()
    backend_calls: list[tuple[str, str]] = []
    database_calls: list[dict[str, object]] = []

    class _FakeDatabase:
        def __init__(self, *, db_path: str, client_id: str, backend=None):
            self.db_path = db_path
            self.client_id = client_id
            self.backend = backend
            self.default_org_id = None
            self.default_team_id = None
            database_calls.append(
                {
                    "db_path": db_path,
                    "client_id": client_id,
                    "backend": backend,
                }
            )

        def release_context_connection(self) -> None:
            return None

    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=False)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(deps, "get_scope", lambda: None, raising=True)
    monkeypatch.setattr(
        media_db_session,
        "_create_sqlite_backend",
        lambda db_path, client_id: backend_calls.append((db_path, client_id)) or sentinel_backend,
        raising=True,
    )
    monkeypatch.setattr(
        media_db_session,
        "_load_default_media_database_factory",
        lambda: _FakeDatabase,
        raising=True,
    )

    first = deps._resolve_media_db_for_user(_make_user())
    second = deps._resolve_media_db_for_user(_make_user())

    assert first is not second
    assert backend_calls == [("/tmp/1.db", "1")]
    assert database_calls == [
        {
            "db_path": "/tmp/1.db",
            "client_id": "1",
            "backend": sentinel_backend,
        },
        {
            "db_path": "/tmp/1.db",
            "client_id": "1",
            "backend": sentinel_backend,
        },
    ]


def test_get_or_create_media_db_factory_accepts_string_user_ids_via_id_int(monkeypatch) -> None:
    created: list[object] = []

    class FakeFactory:
        def __init__(self, *, db_path: str, client_id: str):
            self.db_path = db_path
            self.client_id = client_id
            created.append(self)

    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=False)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.MediaDbFactory,
        "for_sqlite_path",
        classmethod(lambda cls, db_path, client_id: FakeFactory(db_path=db_path, client_id=client_id)),
        raising=True,
    )

    string_user = _make_user()
    string_user.id = "7"

    factory = deps._get_or_create_media_db_factory(string_user)

    assert factory is created[0]
    assert factory.db_path == "/tmp/7.db"


def test_resolve_media_db_for_user_returns_fresh_scoped_session_from_cached_factory(monkeypatch) -> None:
    sessions: list[object] = []

    class FakeFactory:
        def for_request(self, *, org_id=None, team_id=None):
            session = SimpleNamespace(
                org_id=org_id,
                team_id=team_id,
                release_context_connection=lambda: None,
            )
            sessions.append(session)
            return session

    scopes = [
        SimpleNamespace(effective_org_id=10, effective_team_id=20),
        SimpleNamespace(effective_org_id=11, effective_team_id=21),
    ]

    monkeypatch.setattr(deps, "_get_or_create_media_db_factory", lambda current_user: FakeFactory(), raising=False)
    monkeypatch.setattr(deps, "get_scope", lambda: scopes.pop(0), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)

    class FakeLegacyDb:
        def __init__(self, db_path: str, client_id: str, backend=None):
            self.db_path = db_path
            self.client_id = client_id
            self.default_org_id = None
            self.default_team_id = None

        def release_context_connection(self) -> None:
            return None

    monkeypatch.setattr(deps, "MediaDatabase", FakeLegacyDb, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", {}, raising=True)

    first = deps._resolve_media_db_for_user(_make_user())
    second = deps._resolve_media_db_for_user(_make_user())

    assert first is not second
    assert (first.org_id, first.team_id) == (10, 20)
    assert (second.org_id, second.team_id) == (11, 21)
    assert len(sessions) == 2


def test_reset_media_db_cache_closes_cached_factories(monkeypatch) -> None:
    closed: list[str] = []

    class _FakeFactory:
        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr(deps, "_media_db_factories", {1: _FakeFactory()}, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", {}, raising=True)

    deps.reset_media_db_cache()

    assert closed == ["closed"]
    assert deps._media_db_factories == {}


def test_reset_media_db_cache_logs_cleanup_failures(monkeypatch) -> None:
    recorder = _RecordingLogger()

    class _BrokenDb:
        @property
        def backend(self):
            raise RuntimeError("db backend unavailable")

        def close_connection(self) -> None:
            raise RuntimeError("db close failed")

    class _BrokenFactory:
        @property
        def backend(self):
            raise RuntimeError("factory backend unavailable")

        def close(self) -> None:
            raise RuntimeError("factory close failed")

    monkeypatch.setattr(deps, "logger", recorder, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", {1: _BrokenDb()}, raising=True)
    monkeypatch.setattr(deps, "_media_db_factories", {2: _BrokenFactory()}, raising=True)
    monkeypatch.setattr(deps, "reset_managed_sqlite_backends", lambda **kwargs: None, raising=True)

    deps.reset_media_db_cache()

    assert deps._user_db_instances == {}
    assert deps._media_db_factories == {}
    assert any("legacy DB backend during media DB cache reset" in call for call in recorder.warning_calls)
    assert any("legacy DB connection cleanup during media DB cache reset" in call for call in recorder.warning_calls)
    assert any("factory backend during media DB cache reset" in call for call in recorder.warning_calls)
    assert any("factory close during media DB cache reset" in call for call in recorder.warning_calls)


def test_reset_media_db_cache_propagates_managed_backend_reset_failures(monkeypatch) -> None:
    monkeypatch.setattr(deps, "_user_db_instances", {}, raising=True)
    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=True)
    monkeypatch.setattr(
        deps,
        "reset_managed_sqlite_backends",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("reset failed")),
        raising=True,
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        deps.reset_media_db_cache()

    assert deps._user_db_instances == {}
    assert deps._media_db_factories == {}


def test_reset_media_db_cache_only_evicts_cached_media_managed_sqlite_backends(
    monkeypatch,
    tmp_path,
) -> None:
    media_factory = media_db_session.MediaDbFactory.for_sqlite_path(
        str(tmp_path / "media-reset.db"),
        client_id="media-reset",
    )
    media_backend = media_factory.backend
    assert media_backend is not None

    unrelated_backend = backend_factory.DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "unrelated-shared.db"),
            client_id="unrelated",
        )
    )

    monkeypatch.setattr(deps, "_media_db_factories", {1: media_factory}, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", {}, raising=True)

    deps.reset_media_db_cache()

    recreated_media_backend = backend_factory.DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "media-reset.db"),
            client_id="media-reset-recreated",
        )
    )

    assert deps._media_db_factories == {}
    assert backend_factory.is_factory_managed_backend(media_backend) is False
    assert backend_factory.is_factory_managed_backend(unrelated_backend) is True
    assert recreated_media_backend is not media_backend
    assert unrelated_backend.get_pool().get_connection() is not None

    backend_factory.close_all_backends()


def test_managed_media_db_for_owner_releases_session_on_exit(monkeypatch) -> None:
    released: list[str] = []
    fake_session = SimpleNamespace(
        release_context_connection=lambda: released.append("released"),
    )

    monkeypatch.setattr(deps, "get_media_db_for_owner", lambda owner_user_id: fake_session, raising=True)

    with deps.managed_media_db_for_owner(123) as session:
        assert session is fake_session

    assert released == ["released"]


def test_get_media_db_path_for_rag_omits_postgres_memory_placeholder() -> None:
    media_db = SimpleNamespace(
        db_path=":memory:",
        backend_type=BackendType.POSTGRESQL,
    )

    assert deps.get_media_db_path_for_rag(media_db) is None


def test_get_media_db_path_for_rag_keeps_real_sqlite_path() -> None:
    media_db = SimpleNamespace(
        db_path="/tmp/media.db",
        backend_type=BackendType.SQLITE,
    )

    assert deps.get_media_db_path_for_rag(media_db) == "/tmp/media.db"


def test_conflict_error_str_includes_zero_identifier() -> None:
    exc = ConflictError(entity="Media", identifier=0)

    assert "ID: 0" in str(exc)


def test_media_db_validation_requires_full_legacy_helper_contract() -> None:
    incomplete = SimpleNamespace(
        execute_query=lambda *args, **kwargs: None,
        transaction=lambda: None,
        client_id="scope-test",
        db_path_str=":memory:",
    )

    with pytest.raises(TypeError):
        media_db_validation.require_media_database_like(
            incomplete,
            error_message="expected media db",
        )


def test_media_db_session_release_context_connection_keeps_wrapper_local_cleanup_without_pool_shutdown() -> None:
    events: list[str] = []

    class _FakePool:
        def close_all(self) -> None:
            events.append("close_all")

    class _FakeBackend:
        def __init__(self) -> None:
            self.pool = _FakePool()

        def get_pool(self) -> _FakePool:
            return self.pool

    class _FakeDatabase:
        def __init__(self) -> None:
            self.backend = _FakeBackend()

        def release_context_connection(self) -> None:
            events.append("release_context_connection")

        def close_connection(self) -> None:
            events.append("close_connection")

    session = media_db_session.MediaDbSession(
        db_path="/tmp/request-owned.db",
        client_id="scope-test",
        database=_FakeDatabase(),
        owns_backend_resources=True,
    )

    session.release_context_connection()

    assert events == [
        "release_context_connection",
        "close_connection",
    ]


def test_media_db_factory_close_does_not_close_factory_managed_shared_pool(monkeypatch) -> None:
    closed: list[str] = []
    released: list[object] = []

    class _Pool:
        def close_all(self) -> None:
            closed.append("closed")

    class _Backend:
        backend_type = BackendType.SQLITE

        def get_pool(self) -> _Pool:
            return _Pool()

    backend = _Backend()
    factory = media_db_session.MediaDbFactory(
        db_path="/tmp/shared.db",
        client_id="1",
        backend=backend,
    )

    monkeypatch.setattr(
        media_db_session,
        "is_factory_managed_backend",
        lambda candidate: candidate is backend,
        raising=False,
    )
    monkeypatch.setattr(
        media_db_session,
        "release_managed_backend",
        lambda candidate: released.append(candidate),
        raising=False,
    )

    factory.close()

    assert released == [backend]
    assert closed == []


def test_media_db_factory_close_releases_managed_backend_at_most_once(
    monkeypatch,
) -> None:
    released: list[object] = []

    class _Backend:
        backend_type = BackendType.SQLITE

        def get_pool(self):
            raise AssertionError("managed backend close should not touch the pool")

    backend = _Backend()
    factory = media_db_session.MediaDbFactory(
        db_path="/tmp/shared.db",
        client_id="1",
        backend=backend,
    )

    monkeypatch.setattr(
        media_db_session,
        "is_factory_managed_backend",
        lambda candidate: candidate is backend,
        raising=False,
    )
    monkeypatch.setattr(
        media_db_session,
        "release_managed_backend",
        lambda candidate: released.append(candidate),
        raising=False,
    )

    factory.close()
    factory.close()

    assert factory.backend is None
    assert released == [backend]


def test_media_db_factory_close_with_real_managed_sqlite_backend_avoids_pool_shutdown(tmp_path) -> None:
    factory = media_db_session.MediaDbFactory.for_sqlite_path(
        str(tmp_path / "factory-close-real-managed.db"),
        client_id="real-managed",
    )
    backend = factory.backend
    assert backend is not None
    assert media_db_session.is_factory_managed_backend(backend) is True

    pool = backend.get_pool()
    closed: list[str] = []
    original_close_all = pool.close_all

    def _record_close_all() -> None:
        closed.append("closed")
        original_close_all()

    pool.close_all = _record_close_all
    try:
        factory.close()
    finally:
        pool.close_all = original_close_all

    assert closed == []
    assert media_db_session.is_factory_managed_backend(backend) is True


def test_media_db_factory_close_skips_sqlite_release_for_postgres_backend(monkeypatch) -> None:
    closed: list[str] = []
    released: list[object] = []

    class _Pool:
        def close_all(self) -> None:
            closed.append("closed")

    class _Backend:
        backend_type = BackendType.POSTGRESQL

        def get_pool(self) -> _Pool:
            return _Pool()

    backend = _Backend()
    factory = media_db_session.MediaDbFactory(
        db_path="/tmp/postgres-proxy.db",
        client_id="1",
        backend=backend,
    )

    monkeypatch.setattr(
        media_db_session,
        "is_factory_managed_backend",
        lambda candidate: True,
        raising=False,
    )
    monkeypatch.setattr(
        media_db_session,
        "release_managed_backend",
        lambda candidate: released.append(candidate),
        raising=False,
    )

    factory.close()

    assert released == []
    assert closed == ["closed"]
