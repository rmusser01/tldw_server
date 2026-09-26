import configparser
from contextlib import contextmanager
import inspect
from types import ModuleType

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, QueryResult
from tldw_Server_API.app.core.DB_Management.media_db.runtime import collections as runtime_collections
from tldw_Server_API.app.core.DB_Management.media_db.runtime import factory as runtime_factory


def _runtime_config(
    *,
    postgres_content_mode: bool,
    backend_loader=None,
    default_db_path: str = "media.db",
):
    cfg = configparser.ConfigParser()
    return runtime_factory.MediaDbRuntimeConfig(
        default_db_path=default_db_path,
        default_config=cfg,
        postgres_content_mode=postgres_content_mode,
        backend_loader=backend_loader or (lambda: None),
    )


def test_runtime_factory_no_longer_mentions_media_db_v2_in_source():
    assert "Media_DB_v2" not in inspect.getsource(runtime_factory)


def test_runtime_media_class_no_longer_mentions_legacy_identifiers_in_source():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import media_class

    source = inspect.getsource(media_class)

    assert "legacy_identifiers" not in source
    assert "NATIVE_MEDIA_DB_MODULE" in source


def test_load_media_database_cls_resolves_native_class_module():
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_class import (
        load_media_database_cls,
    )
    from tldw_Server_API.app.core.DB_Management.media_db import media_database
    from tldw_Server_API.app.core.DB_Management.media_db import native_class

    cls = load_media_database_cls()

    assert cls is native_class.MediaDatabase
    assert cls is media_database.MediaDatabase


def test_runtime_media_database_cls_resolves_direct_native_owner() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.media_class import (
        load_media_database_cls,
    )

    cls = load_media_database_cls()

    assert cls.__module__.endswith("media_database_impl")


def test_get_current_media_schema_version_matches_canonical_runtime_class():
    from tldw_Server_API.app.core.DB_Management.media_db import media_database

    assert (
        runtime_factory.get_current_media_schema_version()
        == media_database.MediaDatabase._CURRENT_SCHEMA_VERSION
    )


def test_create_media_database_sqlite_uses_default_path_and_no_backend(monkeypatch):
    class StubMediaDatabase:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.calls.append(kwargs)

    runtime = _runtime_config(postgres_content_mode=False)
    monkeypatch.setattr(runtime_factory, "_load_media_database_cls", lambda: StubMediaDatabase)

    db = runtime_factory.create_media_database(
        "client-1",
        runtime=runtime,
    )

    assert isinstance(db, StubMediaDatabase)
    assert StubMediaDatabase.calls == [
        {
            "db_path": "media.db",
            "client_id": "client-1",
            "backend": None,
            "config": runtime.default_config,
        }
    ]


def test_create_media_database_sqlite_preserves_explicit_backend(monkeypatch):
    explicit_backend = object()

    class StubMediaDatabase:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.calls.append(kwargs)

    runtime = _runtime_config(postgres_content_mode=False)
    monkeypatch.setattr(runtime_factory, "_load_media_database_cls", lambda: StubMediaDatabase)

    runtime_factory.create_media_database(
        "client-2",
        db_path="explicit.db",
        backend=explicit_backend,
        runtime=runtime,
    )

    assert StubMediaDatabase.calls == [
        {
            "db_path": "explicit.db",
            "client_id": "client-2",
            "backend": explicit_backend,
            "config": runtime.default_config,
        }
    ]


def test_create_media_database_postgres_requires_backend():
    runtime = _runtime_config(postgres_content_mode=True, backend_loader=lambda: None)

    with pytest.raises(RuntimeError) as excinfo:
        runtime_factory.create_media_database(
            "client-2",
            runtime=runtime,
        )

    assert "PostgreSQL content backend configured" in str(excinfo.value)


def test_validate_postgres_content_backend_uses_factory_validator(monkeypatch):
    expected_version = 7

    class StubBackend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self):
            self.queries = []

        @contextmanager
        def transaction(self):
            yield object()

        def execute(self, query, params=None, connection=None):
            self.queries.append(query)
            if "schema_version" in query:
                return QueryResult(rows=[{"version": expected_version}], rowcount=1)
            return QueryResult(rows=[{"ok": 1}], rowcount=1)

    class StubMediaDatabase:
        _CURRENT_SCHEMA_VERSION = expected_version
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.checked_policies = []
            self.closed = False
            self.__class__.instances.append(self)

        def _postgres_policy_exists(self, conn, table, policy):
            self.checked_policies.append((table, policy))
            return True

        def close_connection(self):
            self.closed = True

    runtime = _runtime_config(
        postgres_content_mode=True,
        backend_loader=lambda: None,
    )
    stub_backend = StubBackend()
    monkeypatch.setattr(runtime_factory, "_load_media_database_cls", lambda: StubMediaDatabase)

    runtime_factory.validate_postgres_content_backend(
        runtime=runtime,
        get_content_backend_instance=lambda: stub_backend,
    )

    assert any("schema_version" in query for query in stub_backend.queries)
    assert StubMediaDatabase.instances
    assert StubMediaDatabase.instances[-1].checked_policies == [
        ("media", "media_visibility_access"),
        ("sync_log", "sync_scope_admin"),
        ("sync_log", "sync_scope_personal"),
        ("sync_log", "sync_scope_org"),
        ("sync_log", "sync_scope_team"),
    ]
    assert StubMediaDatabase.instances[-1].closed is True


def test_validate_postgres_content_backend_reports_schema_version_probe_failure(monkeypatch):
    class StubBackend:
        backend_type = BackendType.POSTGRESQL

        @contextmanager
        def transaction(self):
            yield object()

        def execute(self, query, params=None, connection=None):
            if "schema_version" in query:
                raise RuntimeError("relation schema_version does not exist")
            return QueryResult(rows=[{"ok": 1}], rowcount=1)

    class StubMediaDatabase:
        _CURRENT_SCHEMA_VERSION = 7

        def __init__(self, **kwargs):
            self.closed = False

        def _postgres_policy_exists(self, conn, table, policy):
            return True

        def close_connection(self):
            self.closed = True

    runtime = _runtime_config(
        postgres_content_mode=True,
        backend_loader=lambda: None,
    )
    monkeypatch.setattr(runtime_factory, "_load_media_database_cls", lambda: StubMediaDatabase)

    with pytest.raises(RuntimeError, match="schema validation failed while reading schema_version"):
        runtime_factory.validate_postgres_content_backend(
            runtime=runtime,
            get_content_backend_instance=lambda: StubBackend(),
        )


def test_validate_postgres_content_backend_reports_unreadable_policy_as_validation_failure(monkeypatch):
    class StubBackend:
        backend_type = BackendType.POSTGRESQL

        @contextmanager
        def transaction(self):
            yield object()

        def execute(self, query, params=None, connection=None):
            if "schema_version" in query:
                return QueryResult(rows=[{"version": 7}], rowcount=1)
            return QueryResult(rows=[{"ok": 1}], rowcount=1)

    class StubMediaDatabase:
        _CURRENT_SCHEMA_VERSION = 7

        def __init__(self, **kwargs):
            self.closed = False

        def _postgres_policy_exists(self, conn, table, policy):
            return False

        def close_connection(self):
            self.closed = True

    runtime = _runtime_config(
        postgres_content_mode=True,
        backend_loader=lambda: None,
    )
    monkeypatch.setattr(runtime_factory, "_load_media_database_cls", lambda: StubMediaDatabase)

    with pytest.raises(RuntimeError, match="media_visibility_access.*is missing or could not be inspected"):
        runtime_factory.validate_postgres_content_backend(
            runtime=runtime,
            get_content_backend_instance=lambda: StubBackend(),
        )


def test_load_collections_database_cls_returns_none_for_missing_optional_module(monkeypatch):
    def _raise_missing(_module_path: str):
        raise ModuleNotFoundError("missing optional module", name="tldw_Server_API.app.core.DB_Management.Collections_DB")

    monkeypatch.setattr(runtime_collections, "import_module", _raise_missing)

    assert runtime_collections.load_collections_database_cls() is None


def test_load_collections_database_cls_reraises_internal_import_errors(monkeypatch):
    def _raise_internal(_module_path: str):
        raise ModuleNotFoundError("missing dependency", name="loguru")

    monkeypatch.setattr(runtime_collections, "import_module", _raise_internal)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        runtime_collections.load_collections_database_cls()

    assert excinfo.value.name == "loguru"


def test_load_collections_database_cls_returns_only_class_objects(monkeypatch):
    monkeypatch.setattr(
        runtime_collections,
        "import_module",
        lambda _module_path: ModuleType("collections_stub"),
    )

    assert runtime_collections.load_collections_database_cls() is None
