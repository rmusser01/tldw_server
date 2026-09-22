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


def test_same_store_name_maps_to_different_tables_per_account():
    """The regression: both accounts previously resolved to the same table."""
    alice = _adapter("11")._sanitize_collection("vs_abc123")
    bob = _adapter("22")._sanitize_collection("vs_abc123")

    assert alice != bob
    assert "11" in alice
    assert "22" in bob


def test_table_name_is_still_a_safe_identifier():
    name = _adapter("11")._sanitize_collection("weird-name!; DROP TABLE x")

    assert name.replace("_", "").isalnum()


def test_owner_prefix_survives_a_hostile_user_id():
    """A user id is not assumed to be numeric or clean."""
    name = _adapter("../../etc; DROP")._sanitize_collection("s")

    assert name.replace("_", "").isalnum()


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
