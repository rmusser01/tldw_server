from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Dict, List

from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki import Media_Wiki as mediawiki


def test_process_single_item_stores_vectors(monkeypatch):
    class DummyDB:
        def add_media_with_keywords(self, **_kwargs: Any):
            return 123, "ok"

    class FakeManager:
        last_call: Dict[str, Any] | None = None
        last_user_id: str | None = None

        def __init__(self, user_id: str, user_embedding_config: Dict[str, Any]):
            FakeManager.last_user_id = user_id
            self.user_id = user_id
            self.user_embedding_config = user_embedding_config

        def store_in_chroma(
            self,
            collection_name: str,
            texts: List[str],
            embeddings: List[List[float]],
            ids: List[str],
            metadatas: List[Dict[str, Any]],
            embedding_model_id_for_dim_check: str | None = None,
        ) -> None:
            FakeManager.last_call = {
                "collection_name": collection_name,
                "texts": texts,
                "embeddings": embeddings,
                "ids": ids,
                "metadatas": metadatas,
                "model_id": embedding_model_id_for_dim_check,
            }

    monkeypatch.setattr(mediawiki, "ChromaDBManager", FakeManager)

    def fake_create_embeddings_batch(
        *,
        texts: List[str],
        user_app_config: Dict[str, Any],
        model_id_override: str | None = None,
    ) -> List[List[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(mediawiki, "create_embeddings_batch", fake_create_embeddings_batch)

    item = {
        "timestamp": datetime.now(timezone.utc),
        "page_id": 42,
        "revision_id": 7,
        "namespace": 0,
    }

    kwargs = {
        "content": "Intro\n== Section ==\nMore text",
        "title": "Test Page",
        "wiki_name": "enwiki",
        "chunk_options": {"max_size": 50},
        "item": item,
        "store_to_db": True,
        "store_to_vector_db": True,
        "api_name_vector_db": "openai:text-embedding-3-small",
        "api_key_vector_db": "test-key",
        "media_writer": DummyDB(),
    }
    if "vector_user_id" in inspect.signature(mediawiki.process_single_item).parameters:
        kwargs["vector_user_id"] = 42

    result = mediawiki.process_single_item(**kwargs)

    assert result["media_id"] == 123
    assert FakeManager.last_user_id == "42"
    assert FakeManager.last_call is not None
    assert FakeManager.last_call["collection_name"].startswith("mediawiki_")
    assert FakeManager.last_call["metadatas"][0]["media_id"] == str(result["media_id"])
