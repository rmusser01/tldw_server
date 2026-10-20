"""CharactersRAGDB.close_all_connections() on PostgreSQL (TASK-13861 AC3)."""

import time
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.flashcards_module import FlashcardsModule

pytestmark = pytest.mark.integration


async def test_an_mcp_tool_call_leaves_the_shared_content_pool_open(pg_database_config, tmp_path, monkeypatch):
    """Every handle built without a backend shares one cached content pool in PostgreSQL mode."""
    shared = DatabaseBackendFactory.create_backend(pg_database_config)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: shared)
    cached = CharactersRAGDB(tmp_path / "cached.db", client_id="1")
    try:
        assert cached.backend is shared
        module = FlashcardsModule(ModuleConfig(name="flashcards"))
        context = SimpleNamespace(db_paths={"chacha": str(tmp_path / "mcp.db")}, metadata={})

        # The module opens a handle per call and closes it with close_all_connections().
        listed = await module.execute_tool("flashcards.decks.list", {"limit": 5}, context=context)

        assert listed["total"] == 0
        assert cached.get_character_card_by_name("nobody") is None
    finally:
        cached.close_connection()
        shared.get_pool().close_all()


def test_close_all_connections_closes_its_checkout_before_closing_the_callers_pool(
    pg_database_config, tmp_path, monkeypatch
):
    """A checkout handed to the pool's reset worker must not outlive the pool unreset and open."""
    from psycopg_pool import pool as psycopg_pool_module

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    if not backend.get_pool()._use_psycopg_pool:
        pytest.skip("the fallback pool rolls a returned checkout back synchronously")
    db = CharactersRAGDB(tmp_path / "own.db", client_id="1", backend=backend)
    db.execute_query("SELECT count(*) AS n FROM character_cards")
    raw = db._local.conn
    assert raw.info.transaction_status.name == "INTRANS"
    original_run = psycopg_pool_module.ReturnConnection.run

    def late_run(task):
        time.sleep(0.5)  # a busy pool worker: the pool closes before the queued reset starts
        original_run(task)

    monkeypatch.setattr(psycopg_pool_module.ReturnConnection, "run", late_run)

    db.close_all_connections()

    assert raw.closed, f"the returned checkout was left open ({raw.info.transaction_status.name})"
