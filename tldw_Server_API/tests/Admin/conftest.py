"""Local conftest for Admin tests.

Ensures the AuthNZ schema fixtures are available in environments where
pytest might not auto-load plugins declared in pyproject.toml (e.g., some CI
configurations). Prefer importing from the shared plugin; fall back to local
definitions if unavailable.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def backup_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Own an isolated backup workspace and clear cached settings at both boundaries."""
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path / "backups"))
    monkeypatch.setenv("TLDW_DB_ALLOWED_BASE_DIRS", str(tmp_path))
    monkeypatch.delenv("TLDW_POSTGRES_BACKUP_DSN", raising=False)
    reset_settings()
    try:
        yield tmp_path
    finally:
        reset_settings()


try:
    # Prefer the shared plugin implementations
    from tldw_Server_API.tests._plugins.authnz_fixtures import (
        authnz_schema_ready_sync,  # noqa: F401
        authnz_schema_ready,       # noqa: F401
    )
except Exception:
    # Fallback: define lightweight local fixtures matching the shared ones

    def _run_async(coro):

        import asyncio as _asyncio
        try:
            loop = _asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(coro)  # type: ignore[misc]
        except RuntimeError:
            _ = None
        return _asyncio.run(coro)

    @pytest.fixture
    def authnz_schema_ready_sync():  # noqa: F401
        """Sync-friendly variant to ensure AuthNZ schema for SQLite tests."""
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import (
                reset_settings as _reset_settings,
            )
            from tldw_Server_API.app.core.AuthNZ.database import (
                reset_db_pool as _reset_db_pool,
            )
            _run_async(_reset_db_pool())
            _reset_settings()
        except Exception:
            _ = None
        try:
            from tldw_Server_API.app.core.AuthNZ.initialize import (
                ensure_authnz_schema_ready_once as _ensure_once,
            )
            _run_async(_ensure_once())
        except Exception:
            _ = None
        return None

    try:
        import pytest_asyncio  # type: ignore

        @pytest_asyncio.fixture
        async def authnz_schema_ready():  # noqa: F401
            try:
                from tldw_Server_API.app.core.AuthNZ.settings import (
                    reset_settings as _reset_settings,
                )
                from tldw_Server_API.app.core.AuthNZ.database import (
                    reset_db_pool as _reset_db_pool,
                )
                await _reset_db_pool()
                _reset_settings()
            except Exception:
                _ = None
            try:
                from tldw_Server_API.app.core.AuthNZ.initialize import (
                    ensure_authnz_schema_ready_once as _ensure_once,
                )
                await _ensure_once()
            except Exception:
                _ = None
            return None
    except Exception:
        # If pytest-asyncio isn't available, only the sync fixture will exist.
        _ = None
