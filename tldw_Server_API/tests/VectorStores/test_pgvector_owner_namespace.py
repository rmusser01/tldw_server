"""pgvector stores must be namespaced per account.

Unlike the ChromaDB adapter, which isolates accounts by directory, pgvector
puts every account's stores in one PostgreSQL database. The table name was
derived from the collection name alone, so a store id was sufficient to read,
write or drop another account's vectors -- and list_collections scanned
pg_tables for every table named vs_%, handing those ids to whoever asked.

VectorStoreConfig has documented the intended "user_{user_id}_" convention all
along; this adapter simply never applied it.

Pure string and stub assertions, so no database is required.
"""

import pytest

from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.pgvector_adapter import (
    PGVectorAdapter,
)

pytestmark = pytest.mark.unit


def _adapter(user_id: str) -> PGVectorAdapter:
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={},
        embedding_dim=8,
        user_id=user_id,
    )
    return PGVectorAdapter(cfg)


async def _collections_for(user_id: str, tables: list[str], monkeypatch) -> list[str]:
    """List collections an account can see, given what exists in the database."""
    adapter = _adapter(user_id)

    async def _fake_query(sql, params):
        prefix = params[0].rstrip("%")
        return [(t,) for t in tables if t.startswith(prefix)]

    monkeypatch.setattr(adapter, "_query", _fake_query)
    return await adapter.list_collections()


async def test_one_account_cannot_see_anothers_store(monkeypatch):
    """The regression, stated in observable terms.

    Both accounts previously resolved the same store name to the same table and
    list_collections returned every account's stores to whoever asked.
    """
    existing = ["vs_u11_shared_name", "vs_u22_shared_name", "vs_u22_secret"]

    alice = await _collections_for("11", existing, monkeypatch)
    bob = await _collections_for("22", existing, monkeypatch)

    assert alice == ["shared_name"], "must see only its own store"
    assert sorted(bob) == ["secret", "shared_name"]
    assert "secret" not in alice, "bob's private store must be invisible to alice"


def test_the_same_store_name_is_a_different_table_per_account():
    """Characterises the naming scheme the isolation above relies on."""
    alice = _adapter("11")._sanitize_collection("vs_abc123")
    bob = _adapter("22")._sanitize_collection("vs_abc123")

    assert alice != bob


def test_table_names_stay_safe_identifiers_for_hostile_input():
    for user_id, name in (("11", "weird-name!; DROP TABLE x"), ("../../etc; DROP", "s")):
        table = _adapter(user_id)._sanitize_collection(name)
        assert table.replace("_", "").isalnum()


async def test_list_collections_only_returns_this_accounts_stores(monkeypatch):
    alice = _adapter("11")
    seen_params = {}

    async def _fake_query(sql, params):
        seen_params["params"] = params
        # The server would filter by the LIKE pattern; emulate that faithfully.
        prefix = params[0].rstrip("%")
        rows = [("vs_u11_store_a",), ("vs_u11_store_b",), ("vs_u22_secret",)]
        return [r for r in rows if r[0].startswith(prefix)]

    monkeypatch.setattr(alice, "_query", _fake_query)

    collections = await alice.list_collections()

    assert collections == ["store_a", "store_b"]
    assert "vs_u22_secret" not in collections
    assert seen_params["params"][0] == "vs_u11_%", "must scope the LIKE to one account"
