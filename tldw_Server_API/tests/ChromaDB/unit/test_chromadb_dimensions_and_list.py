import numpy as np
import pytest
from unittest.mock import MagicMock
from chromadb.errors import InternalError

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from unittest.mock import patch


def _make_manager_with_mock(mock_client, tmp_path):


    user_cfg = {
        "USER_DB_BASE_DIR": str(tmp_path),
        "embedding_config": {"default_model_id": "text-embedding-3-large", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    # Use constructor injection to provide the mock client
    mgr = ChromaDBManager(user_id="test_user", user_embedding_config=user_cfg, client=mock_client)
    return mgr


def test_dimension_metadata_mismatch_recreates_collection(tmp_path):


    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_coll.name = "dim_meta"
    # Collection has metadata dimension 256
    mock_coll.metadata = {"embedding_dimension": 256}
    mock_client.get_or_create_collection.return_value = mock_coll
    mock_client.create_collection.return_value = mock_coll

    mgr = _make_manager_with_mock(mock_client, tmp_path)

    # New embeddings have dimension 512
    texts = ["a", "b"]
    embeddings = np.random.rand(2, 512).astype(float).tolist()
    ids = ["1", "2"]
    metas = [{"source": "t1"}, {"source": "t2"}]

    mgr.store_in_chroma("dim_meta", texts, embeddings, ids, metas, embedding_model_id_for_dim_check="text-embedding-3-large")

    # Expect deletion and recreation with new dimension metadata
    mock_client.delete_collection.assert_called_with(name="dim_meta")
    mock_client.create_collection.assert_called()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs.get("name") == "dim_meta"
    assert kwargs.get("metadata")["embedding_dimension"] == 512
    assert kwargs.get("metadata")["source_model_id"] == "text-embedding-3-large"
    # Upsert invoked on the (current) collection
    assert mock_coll.upsert.called


def test_dimension_sample_mismatch_recreates_collection(tmp_path):


    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_coll.name = "dim_sample"
    # No metadata; has items
    mock_coll.metadata = {}
    mock_coll.count.return_value = 1
    # Return a single stored embedding of size 128
    mock_coll.get.return_value = {"embeddings": [[0.0] * 128]}
    mock_client.get_or_create_collection.return_value = mock_coll
    mock_client.create_collection.return_value = mock_coll

    mgr = _make_manager_with_mock(mock_client, tmp_path)

    texts = ["x"]
    embeddings = np.random.rand(1, 256).astype(float).tolist()
    ids = ["id-x"]
    metas = [{"source": "unit"}]

    mgr.store_in_chroma("dim_sample", texts, embeddings, ids, metas, embedding_model_id_for_dim_check="text-embedding-3-large")

    # Expect recreation due to sampled dim mismatch
    mock_client.delete_collection.assert_called_with(name="dim_sample")
    mock_client.create_collection.assert_called()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs.get("name") == "dim_sample"
    assert kwargs.get("metadata")["embedding_dimension"] == 256
    assert kwargs.get("metadata")["source_model_id"] == "text-embedding-3-large"
    assert mock_coll.upsert.called


def test_list_collections_propagates(mock_chroma_client, tmp_path):


     # Reuse fixture mock client from conftest to ensure typical shape
    mgr = _make_manager_with_mock(mock_chroma_client, tmp_path)
    # Simulate two collections
    c1 = MagicMock(); c1.name = "c1"
    c2 = MagicMock(); c2.name = "c2"
    mock_chroma_client.list_collections.return_value = [c1, c2]

    cols = mgr.list_collections()
    assert [c.name for c in cols] == ["c1", "c2"]


def test_delete_collection_calls_client(mock_chroma_client, tmp_path):


    mgr = _make_manager_with_mock(mock_chroma_client, tmp_path)
    mgr.delete_collection("to_delete")
    mock_chroma_client.delete_collection.assert_called_with(name="to_delete")


def test_precomputed_query_retries_transient_hnsw_segment_error(mock_chroma_client, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

    first_collection = MagicMock()
    first_collection.name = "retry_collection"
    first_collection.query.side_effect = InternalError(
        "Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk"
    )
    retry_collection = MagicMock()
    retry_collection.name = "retry_collection"
    retry_result = {"ids": [["doc_1"]], "documents": [["hello"]], "metadatas": [[{"i": 1}]], "distances": [[0.0]]}
    retry_collection.query.return_value = retry_result
    mock_chroma_client.get_or_create_collection.side_effect = [first_collection, retry_collection]
    monkeypatch.setattr(cdl.time, "sleep", lambda _seconds: None)

    mgr = _make_manager_with_mock(mock_chroma_client, tmp_path)

    result = mgr.query_collection_with_precomputed_embeddings(
        collection_name="retry_collection",
        query_embeddings=[[0.1, 0.2, 0.3]],
    )

    assert result == retry_result
    assert mock_chroma_client.get_or_create_collection.call_count == 2
    first_collection.query.assert_called_once()
    retry_collection.query.assert_called_once()


def test_precomputed_query_retries_repeated_transient_hnsw_segment_errors(mock_chroma_client, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

    transient_error = InternalError(
        "Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk"
    )
    first_collection = MagicMock()
    first_collection.name = "retry_collection"
    first_collection.query.side_effect = transient_error
    second_collection = MagicMock()
    second_collection.name = "retry_collection"
    second_collection.query.side_effect = transient_error
    third_collection = MagicMock()
    third_collection.name = "retry_collection"
    retry_result = {"ids": [["doc_1"]], "documents": [["hello"]], "metadatas": [[{"i": 1}]], "distances": [[0.0]]}
    third_collection.query.return_value = retry_result
    mock_chroma_client.get_or_create_collection.side_effect = [
        first_collection,
        second_collection,
        third_collection,
    ]
    monkeypatch.setattr(cdl.time, "sleep", lambda _seconds: None)

    mgr = _make_manager_with_mock(mock_chroma_client, tmp_path)

    result = mgr.query_collection_with_precomputed_embeddings(
        collection_name="retry_collection",
        query_embeddings=[[0.1, 0.2, 0.3]],
    )

    assert result == retry_result
    assert mock_chroma_client.get_or_create_collection.call_count == 3
    first_collection.query.assert_called_once()
    second_collection.query.assert_called_once()
    third_collection.query.assert_called_once()


def test_store_retries_transient_hnsw_segment_error(mock_chroma_client, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

    first_collection = MagicMock()
    first_collection.name = "retry_store_collection"
    first_collection.metadata = {}
    first_collection.count.return_value = 0
    first_collection.upsert.side_effect = InternalError(
        "Error executing plan: Internal error: Error creating hnsw segment reader: Nothing found on disk"
    )
    retry_collection = MagicMock()
    retry_collection.name = "retry_store_collection"
    mock_chroma_client.get_or_create_collection.side_effect = [first_collection, retry_collection]
    monkeypatch.setattr(cdl.time, "sleep", lambda _seconds: None)

    mgr = _make_manager_with_mock(mock_chroma_client, tmp_path)

    result = mgr.store_in_chroma(
        collection_name="retry_store_collection",
        texts=["hello"],
        embeddings=[[0.1, 0.2, 0.3]],
        ids=["doc_1"],
        metadatas=[{"i": 1}],
    )

    assert result is retry_collection
    assert mock_chroma_client.get_or_create_collection.call_count == 2
    first_collection.upsert.assert_called_once()
    retry_collection.upsert.assert_called_once_with(
        documents=["hello"],
        embeddings=[[0.1, 0.2, 0.3]],
        ids=["doc_1"],
        metadatas=[{"i": 1}],
    )


@pytest.mark.unit
def test_minimal_integration_with_real_persistent_client(temp_chroma_path):
    """Lightweight integration: real PersistentClient in temp dir, basic lifecycle."""
    # Build manager without patching client
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="itest", user_embedding_config=user_cfg)

    # Upsert two vectors and verify count
    coll = "itest_coll"
    texts = ["hello", "world"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["a", "b"]
    metas = [{"source": "t"}, {"source": "t"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")
    assert mgr.count_items_in_collection(coll) == 2

    # Delete one and re-count
    mgr.delete_from_collection(["a"], coll)
    assert mgr.count_items_in_collection(coll) == 1

    # List should contain the collection we used
    names = [c.name for c in mgr.list_collections()]
    assert coll in names
    mgr.close()


@pytest.mark.unit
def test_vector_search_with_mocked_query_embedding(temp_chroma_path, monkeypatch):
    """Vector search smoke with real PersistentClient and mocked query embedding."""
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="vsearch", user_embedding_config=user_cfg)

    coll = "vsearch_coll"
    texts = ["hello", "world"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["a", "b"]
    metas = [{"source": "t"}, {"source": "t"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")

    # Patch create_embedding used by vector_search to match our 3-dim space
    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
    monkeypatch.setattr(cdl, "create_embedding", lambda text, user_app_config, model_id_override: [0.1, 0.2, 0.3])

    results = mgr.vector_search(
        query="hello",
        collection_name=coll,
        k=1,
        embedding_model_id_override="manual",
        include_fields=["documents", "metadatas", "distances"],
    )
    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert first.get("id") in {"a", "b"}
    assert "content" in first
    mgr.close()


@pytest.mark.unit
def test_vector_search_passes_user_app_config(temp_chroma_path, monkeypatch):
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="vsearch_cfg", user_embedding_config=user_cfg)

    coll = "vsearch_cfg_coll"
    texts = ["hello", "world"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["a", "b"]
    metas = [{"source": "t"}, {"source": "t"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")

    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl

    captured = {}

    def _fake_create_embedding(*args, **kwargs):
        captured["kwargs"] = kwargs
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(cdl, "create_embedding", _fake_create_embedding)

    mgr.vector_search(
        query="hello",
        collection_name=coll,
        k=1,
        embedding_model_id_override="manual",
        include_fields=["documents", "metadatas", "distances"],
    )

    assert "user_app_config" in captured["kwargs"]
    assert captured["kwargs"]["user_app_config"] == user_cfg
    assert "user_embedding_config" not in captured["kwargs"]
    mgr.close()


@pytest.mark.unit
def test_vector_search_k2_ids(temp_chroma_path, monkeypatch):
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="vsearch2", user_embedding_config=user_cfg)

    coll = "vsearch_coll2"
    texts = ["doc-a", "doc-b"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["a", "b"]
    metas = [{"source": "t"}, {"source": "t"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")

    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
    monkeypatch.setattr(cdl, "create_embedding", lambda text, user_app_config, model_id_override: [0.1, 0.2, 0.3])

    results = mgr.vector_search(
        query="doc-a",
        collection_name=coll,
        k=2,
        embedding_model_id_override="manual",
        include_fields=["documents", "metadatas", "distances"],
    )
    assert isinstance(results, list) and len(results) == 2
    got_ids = {r["id"] for r in results}
    assert got_ids == {"a", "b"}
    mgr.close()


@pytest.mark.unit
def test_vector_search_where_filter(temp_chroma_path, monkeypatch):
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="vfilter", user_embedding_config=user_cfg)

    coll = "vfilter_coll"
    texts = ["keep", "skip"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["ka", "kb"]
    metas = [{"source": "keep"}, {"source": "skip"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")

    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
    monkeypatch.setattr(cdl, "create_embedding", lambda text, user_app_config, model_id_override: [0.1, 0.2, 0.3])

    results = mgr.vector_search(
        query="keep",
        collection_name=coll,
        k=2,
        embedding_model_id_override="manual",
        where_filter={"source": "keep"},
        include_fields=["documents", "metadatas", "distances"],
    )
    # Should only return entries with metadata.source == 'keep'
    assert isinstance(results, list) and len(results) >= 1
    for r in results:
        assert r.get("metadata", {}).get("source") == "keep"
    mgr.close()


@pytest.mark.unit
def test_vector_search_include_embeddings_returns_embeddings(temp_chroma_path, monkeypatch):
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="vemb", user_embedding_config=user_cfg)

    coll = "vemb_coll"
    texts = ["emb1", "emb2"]
    embeddings = [[0.11, 0.22, 0.33], [0.21, 0.12, 0.44]]
    ids = ["e1", "e2"]
    metas = [{"source": "x"}, {"source": "y"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")

    from tldw_Server_API.app.core.Embeddings import ChromaDB_Library as cdl
    monkeypatch.setattr(cdl, "create_embedding", lambda text, user_app_config, model_id_override: [0.11, 0.22, 0.33])

    results = mgr.vector_search(
        query="emb1",
        collection_name=coll,
        k=1,
        embedding_model_id_override="manual",
        include_fields=["documents", "metadatas", "distances", "embeddings"],
    )
    assert isinstance(results, list) and len(results) >= 1
    r0 = results[0]
    assert "embedding" in r0
    assert isinstance(r0["embedding"], list) and len(r0["embedding"]) == 3
    mgr.close()


@pytest.mark.unit
def test_reset_collection_clears_items_count(temp_chroma_path):
    user_cfg = {
        "USER_DB_BASE_DIR": str(temp_chroma_path),
        "embedding_config": {"default_model_id": "unused", "models": {}},
        "chroma_client_settings": {"anonymized_telemetry": False, "allow_reset": True},
    }
    mgr = ChromaDBManager(user_id="rcol", user_embedding_config=user_cfg)

    coll = "reset_me"
    texts = ["one", "two"]
    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    ids = ["1", "2"]
    metas = [{"source": "t"}, {"source": "t"}]
    mgr.store_in_chroma(coll, texts, embeddings, ids, metas, embedding_model_id_for_dim_check="manual")
    assert mgr.count_items_in_collection(coll) == 2

    mgr.reset_chroma_collection(coll)
    assert mgr.count_items_in_collection(coll) == 0
    mgr.close()
