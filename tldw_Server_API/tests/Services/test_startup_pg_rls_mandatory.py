"""PostgreSQL content mode must not start without tenant isolation policies.

RLS installation used to be gated behind RAG_ENSURE_PG_RLS, which defaults to
off and appears in no compose file, and a failure to apply was a warning the
server started through. On PostgreSQL every account's rows share one table, so
that combination meant a default deployment ran with no database-level
isolation at all.

SQLite is unaffected: per-user database files are the boundary there, and there
is nothing to install.
"""

import pytest

from tldw_Server_API.app.services import startup_infra_services as infra

pytestmark = pytest.mark.unit


def _patch_backend(monkeypatch):
    import tldw_Server_API.app.core.DB_Management.backends.base as base
    import tldw_Server_API.app.core.DB_Management.backends.factory as factory

    monkeypatch.setattr(base.DatabaseConfig, "from_env", classmethod(lambda cls: object()))
    monkeypatch.setattr(
        factory.DatabaseBackendFactory, "create_backend", staticmethod(lambda cfg: object())
    )


async def test_postgres_mode_aborts_startup_when_policies_cannot_be_applied(monkeypatch):
    """The regression: this used to log a warning and serve anyway."""
    monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: True)
    _patch_backend(monkeypatch)

    def _boom(_backend):
        raise RuntimeError("policy install exploded")

    with pytest.raises(RuntimeError, match="Refusing to start"):
        await infra._maybe_ensure_pg_rls(_boom)


async def test_postgres_mode_applies_policies_without_any_env_flag(monkeypatch):
    """No RAG_ENSURE_PG_RLS set, and the policies still get installed."""
    monkeypatch.delenv("RAG_ENSURE_PG_RLS", raising=False)
    monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: True)
    _patch_backend(monkeypatch)

    applied = []
    await infra._maybe_ensure_pg_rls(lambda backend: applied.append(backend))

    assert len(applied) == 1


async def test_sqlite_mode_skips_without_touching_the_backend(monkeypatch):
    monkeypatch.delenv("RAG_ENSURE_PG_RLS", raising=False)
    monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: False)

    called = []
    await infra._maybe_ensure_pg_rls(lambda backend: called.append(backend))

    assert called == []


async def test_sqlite_mode_with_the_flag_on_still_only_warns(monkeypatch):
    """Opt-in installs on SQLite stay best-effort; there is no tenant risk."""
    monkeypatch.setenv("RAG_ENSURE_PG_RLS", "true")
    monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: False)
    _patch_backend(monkeypatch)

    def _boom(_backend):
        raise RuntimeError("policy install exploded")

    await infra._maybe_ensure_pg_rls(_boom)  # must not raise
