import asyncio
import json
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import InvalidMetadataOrderKeyError
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.pgvector_adapter import PGVectorAdapter


@pytest.mark.unit
def test_pg_list_vectors_paginated_builds_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify paginated vector listing applies metadata filters."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=8,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)

    captured: list[tuple[str, Any]] = []

    async def fake_query(sql: str, params: Any = None) -> list[tuple[Any, ...]]:
        captured.append((sql, params))
        # Return two rows
        if 'COUNT(*)' in sql:
            return [(2,)]
        return [("a", "doc a", {"genre": "a"}), ("b", "doc b", {"genre": "b"})]

    monkeypatch.setattr(adapter, '_query', fake_query)

    # Invoke with a metadata filter
    res = asyncio.run(
        adapter.list_vectors_paginated('store', limit=10, offset=0, filter={'genre': 'a'})
    )
    assert res['total'] == 2
    assert isinstance(res['items'], list)
    # Ensure WHERE clause was used
    assert any('WHERE metadata @> %s' in sql for (sql, _p) in captured)


@pytest.mark.unit
def test_pg_list_vectors_with_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify paginated vector listing includes embedding vectors."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=4,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)

    async def fake_query(sql: str, params: Any = None) -> list[tuple[Any, ...]]:
        if 'COUNT(*)' in sql:
            return [(1,)]
        return [("id1", "doc1", {"k": 1}, [0.1, 0.2, 0.3, 0.4])]

    monkeypatch.setattr(adapter, '_query', fake_query)

    res = asyncio.run(
        adapter.list_vectors_with_embeddings_paginated('store', limit=1, offset=0)
    )
    assert res['total'] == 1
    assert res['items'][0]['id'] == 'id1'
    assert isinstance(res['items'][0]['vector'], list)


@pytest.mark.unit
def test_pg_list_vectors_paginated_binds_metadata_order_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify metadata order keys are bound as query parameters."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=8,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)
    captured: list[tuple[str, Any]] = []

    async def fake_query(sql: str, params: Any = None) -> list[tuple[Any, ...]]:
        captured.append((sql, params))
        if 'COUNT(*)' in sql:
            return [(1,)]
        return [("a", "doc a", {"score": "0.9"})]

    monkeypatch.setattr(adapter, '_query', fake_query)

    asyncio.run(
        adapter.list_vectors_paginated(
            'store',
            limit=10,
            offset=0,
            order_by='metadata.score',
        )
    )

    select_sql, select_params = captured[0]
    assert "metadata->>'score'" not in select_sql
    assert "metadata->>%s" in select_sql
    assert select_params == ("score", 10, 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    ["list_vectors_paginated", "list_vectors_with_embeddings_paginated"],
)
def test_pg_list_vectors_rejects_unsafe_metadata_order_key(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    """Verify unsafe metadata order keys are rejected before query execution."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=8,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)

    async def fake_query(sql: str, params: Any = None) -> list[tuple[Any, ...]]:
        if 'COUNT(*)' in sql:
            return [(0,)]
        return []

    monkeypatch.setattr(adapter, '_query', fake_query)

    with pytest.raises(InvalidMetadataOrderKeyError, match="metadata order key"):
        asyncio.run(
            getattr(adapter, method_name)(
                'store',
                limit=10,
                offset=0,
                order_by="metadata.score') DESC; SELECT pg_sleep(10); --",
            )
        )


@pytest.mark.unit
def test_pg_multi_search_normalizes_patterns_to_sanitized_collection_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify namespace-style glob patterns match sanitized collection names."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=4,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)
    searched_collections: list[str] = []

    async def fake_list_collections() -> list[str]:
        return ["tenant_alpha", "tenant_beta", "tenantXalpha"]

    async def fake_search(
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        searched_collections.append(collection_name)
        return []

    monkeypatch.setattr(adapter, 'list_collections', fake_list_collections)
    monkeypatch.setattr(adapter, 'search', fake_search)

    asyncio.run(adapter.multi_search(["tenant.*"], [0.1, 0.2, 0.3, 0.4], k=5))

    assert searched_collections == ["tenant_alpha", "tenant_beta"]


@pytest.mark.unit
def test_pg_multi_search_treats_unsupported_glob_tokens_as_literals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collection patterns only support the intended star wildcard."""
    cfg = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        connection_params={'dsn': 'postgresql://u:p@localhost:5432/db'},
        embedding_dim=4,
        user_id='1'
    )
    adapter = PGVectorAdapter(cfg)
    searched_collections: list[str] = []

    async def fake_list_collections() -> list[str]:
        return ["tenant_alpha", "tenant_beta", "tenant_ab_gamma"]

    async def fake_search(
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        searched_collections.append(collection_name)
        return []

    monkeypatch.setattr(adapter, 'list_collections', fake_list_collections)
    monkeypatch.setattr(adapter, 'search', fake_search)

    asyncio.run(adapter.multi_search(["tenant[ab]*"], [0.1, 0.2, 0.3, 0.4], k=5))

    assert searched_collections == ["tenant_ab_gamma"]
