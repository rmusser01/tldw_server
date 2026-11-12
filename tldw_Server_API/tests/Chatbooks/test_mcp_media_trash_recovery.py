"""MCP user deletion must retain the same recoverable Trash boundary as REST."""

from collections import OrderedDict
from contextlib import ExitStack
from threading import Lock
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import media_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import MediaModule

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def owned_media(request, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-mcp-trash-boundary-" * 4)
    reset_settings()
    try:
        with ExitStack() as cleanup:
            path = str(tmp_path / "mcp-media.db")
            if request.param == "postgres":
                backend = request.getfixturevalue("pg_restricted_backend")
            else:
                backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=path))
                cleanup.callback(backend.get_pool().close_all)
            with scoped_context(user_id=101, is_admin=False):
                db = MediaDatabase(db_path=path, client_id="101", backend=backend)
                cleanup.callback(db.close_connection)
                media_id, _, _ = db.add_media_with_keywords(
                    url="https://example.test/s02/mcp-trash",
                    title="S02 recoverable Rowan source",
                    media_type="document",
                    content="The telescope is Selene. Preserve this source through Trash and restore.",
                    keywords=["rowan"],
                    owner_user_id=101,
                )
                module = MediaModule(ModuleConfig(name="media", settings={"db_path": path}))
                mcp_db = MediaDatabase(db_path=path, client_id="mcp_media_media", backend=backend)
                cleanup.callback(mcp_db.close_connection)
                module.db = module._module_db_owner = mcp_db
                module._media_cache = {"prior": {"media_id": media_id}}
                module._user_db_cache_lock = Lock()
                module._user_db_cache = OrderedDict()
                create = media_module.create_media_database

                def create_with_fixture_backend(client_id, **kwargs):
                    handle = create(client_id, backend=backend, **kwargs)
                    cleanup.callback(handle.close_connection)
                    return handle

                monkeypatch.setattr(media_module, "create_media_database", create_with_fixture_backend)
                context = SimpleNamespace(user_id="101", metadata={"roles": ["user"]}, db_paths={"media": path})
                yield module, db, int(media_id), context
    finally:
        reset_settings()


async def test_mcp_user_delete_retains_recoverable_source(owned_media):
    module, db, media_id, context = owned_media
    before = db.get_media_by_id(media_id)
    result = await module.execute_tool("delete_media", {"media_id": media_id}, context=context)
    assert result == {"media_id": media_id, "action": "soft_deleted", "success": True}  # nosec B101 -- Public tool contract.
    assert module._media_cache == {}  # nosec B101 -- Successful writes invalidate stale retrievals.
    assert db.get_media_by_id(media_id) is None  # nosec B101 -- Trashed source leaves ordinary retrieval.
    trashed = db.get_media_by_id(media_id, include_deleted=True, include_trash=True)
    assert trashed is not None and trashed["is_trash"] and not trashed["deleted"]  # nosec B101 -- User deletion remains reversible.
    assert trashed["uuid"] == before["uuid"] and trashed["content"] == before["content"]  # nosec B101 -- Preserve source identity and body.
    assert db.restore_from_trash(media_id)  # nosec B101 -- Actual ordinary-user recovery.
    restored = db.get_media_by_id(media_id)
    assert restored["uuid"] == before["uuid"] and restored["content"] == before["content"]  # nosec B101 -- Roundtrip data assertion.


async def test_mcp_foreign_delete_cannot_change_source(owned_media):
    module, db, media_id, _context = owned_media
    before = db.get_media_by_id(media_id)
    context = SimpleNamespace(user_id="202", metadata={"roles": ["user"]}, db_paths={"media": db.db_path_str})
    with scoped_context(user_id=202, is_admin=False):
        with pytest.raises((PermissionError, ValueError)):
            await module.execute_tool("delete_media", {"media_id": media_id}, context=context)
    assert db.get_media_by_id(media_id) == before  # nosec B101 -- Foreign failure performs no mutation.
    assert "prior" in module._media_cache  # nosec B101 -- Failed writes do not report cache invalidation.


async def test_mcp_permanent_delete_still_requires_admin(owned_media):
    module, db, media_id, context = owned_media
    before = db.get_media_by_id(media_id)
    with pytest.raises(PermissionError, match="Admin role required"):
        await module.execute_tool("delete_media", {"media_id": media_id, "permanent": True}, context=context)
    assert db.get_media_by_id(media_id) == before  # nosec B101 -- No permanent write for ordinary users.
    assert "prior" in module._media_cache  # nosec B101 -- Denial retains the original retrieval state.


async def test_mcp_media_handles_are_bound_to_principal_even_at_same_path(owned_media):
    module, db, _media_id, context = owned_media
    own = module._open_media_db(context)
    foreign_context = SimpleNamespace(user_id="202", metadata={"roles": ["user"]}, db_paths={"media": db.db_path_str})
    foreign = module._open_media_db(foreign_context)
    assert own.client_id == "101" and foreign.client_id == "202"  # nosec B101 -- Authenticated sync identity follows the caller.
    assert own is not foreign  # nosec B101 -- Shared paths cannot reuse another principal's handle.
    assert module._open_media_db(context) is own  # nosec B101 -- Same principal/path cache remains effective.
    module.db = module._module_db_owner = db
    assert module._open_media_db(context) is db  # nosec B101 -- Correctly bound startup handle remains reusable.
    assert module._open_media_db(foreign_context) is foreign  # nosec B101 -- Matching path alone cannot grant reuse.


async def test_mcp_uncached_media_handles_use_principal(owned_media):
    module, db, _media_id, context = owned_media
    module._user_db_cache_lock = None
    own = module._open_media_db(context)
    foreign_context = SimpleNamespace(user_id="202", metadata={"roles": ["user"]}, db_paths={"media": db.db_path_str})
    foreign = module._open_media_db(foreign_context)
    assert own.client_id == "101" and foreign.client_id == "202"  # nosec B101 -- No-lock fallback retains authenticated identity.
    assert own is not foreign and module._user_db_cache == {}  # nosec B101 -- Fallback does not introduce shared-principal state.
