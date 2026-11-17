import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

pytestmark = pytest.mark.unit


def _load_postgres_rls_module():
    module_name = "tldw_Server_API.app.core.DB_Management.media_db.schema.features.postgres_rls"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Expected schema helper module spec for {module_name}"
    return importlib.import_module(module_name)


def test_postgres_rls_helper_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    postgres_rls_module = _load_postgres_rls_module()

    assert MediaDatabase.__dict__["_postgres_policy_exists"].__globals__["__name__"] == (postgres_rls_module.__name__)
    assert MediaDatabase.__dict__["_ensure_postgres_rls"].__globals__["__name__"] == (postgres_rls_module.__name__)


def test_postgres_policy_exists_returns_true_when_rows_are_present() -> None:
    postgres_rls_module = _load_postgres_rls_module()

    backend = SimpleNamespace(
        execute=lambda query, params, connection=None: SimpleNamespace(rows=[{"ok": 1}]),
    )
    db = SimpleNamespace(backend=backend)

    assert postgres_rls_module._postgres_policy_exists(db, object(), "media", "policy_a") is True


def test_postgres_policy_exists_returns_false_on_backend_error() -> None:
    postgres_rls_module = _load_postgres_rls_module()

    def _raise_backend_error(query, params, connection=None):
        raise BackendDatabaseError("probe failed")

    backend = SimpleNamespace(execute=_raise_backend_error)
    db = SimpleNamespace(backend=backend)

    assert postgres_rls_module._postgres_policy_exists(db, object(), "media", "policy_a") is False


def test_ensure_postgres_rls_refreshes_existing_and_creates_missing_sync_log_policies() -> None:
    postgres_rls_module = _load_postgres_rls_module()

    conn = object()
    calls: list[str] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, params=None, connection=None):
            calls.append(" ".join(query.split()))
            return SimpleNamespace(rows=[{"ok": 1}])

    existing_policies = {
        ("media", "media_scope_admin"): True,
        ("media", "media_scope_personal"): False,
        ("media", "media_scope_org"): False,
        ("media", "media_scope_team"): True,
        ("sync_log", "sync_scope_admin"): True,
        ("sync_log", "sync_scope_personal"): False,
        ("sync_log", "sync_scope_org"): False,
        ("sync_log", "sync_scope_team"): False,
    }
    checked: list[tuple[str, str]] = []

    def _policy_exists(connection, table: str, policy: str) -> bool:
        checked.append((table, policy))
        return existing_policies.get((table, policy), False)

    db = SimpleNamespace(
        backend=FakeBackend(),
        _postgres_policy_exists=_policy_exists,
    )

    postgres_rls_module._ensure_postgres_rls(db, conn)

    assert ("media", "media_scope_admin") in checked
    assert ("media", "media_scope_team") in checked
    assert ("sync_log", "sync_scope_admin") in checked
    assert ("sync_log", "sync_scope_team") in checked

    assert any('DROP POLICY IF EXISTS "media_scope_admin" ON "media"' in call for call in calls)
    assert any('DROP POLICY IF EXISTS "media_scope_team" ON "media"' in call for call in calls)
    assert any('ALTER TABLE "media" ENABLE ROW LEVEL SECURITY' in call for call in calls)
    assert any('ALTER TABLE "media" FORCE ROW LEVEL SECURITY' in call for call in calls)
    assert any('DROP POLICY IF EXISTS "media_visibility_access" ON "media"' in call for call in calls)
    assert any('CREATE POLICY "media_visibility_access" ON "media"' in call for call in calls)
    assert any('ALTER TABLE "sync_log" ENABLE ROW LEVEL SECURITY' in call for call in calls)
    assert any('ALTER TABLE "sync_log" FORCE ROW LEVEL SECURITY' in call for call in calls)
    assert not any('CREATE POLICY "sync_scope_admin" ON "sync_log"' in call for call in calls)
    assert any('ALTER POLICY "sync_scope_admin" ON "sync_log" USING (' in call for call in calls)  # nosec B101 # owner refresh regression assertion.
    assert not any("DROP POLICY" in call and 'ON "sync_log"' in call for call in calls)  # nosec B101 # preserve existing roles and command.
    assert any('CREATE POLICY "sync_scope_personal" ON "sync_log"' in call for call in calls)
    assert any('CREATE POLICY "sync_scope_org" ON "sync_log"' in call for call in calls)
    assert any('CREATE POLICY "sync_scope_team" ON "sync_log"' in call for call in calls)
