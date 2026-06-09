import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import RerankingStrategy
from tldw_Server_API.tests.LLamaCpp.llamacpp_test_app import llamacpp_test_client


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    with llamacpp_test_client(headers=headers) as c:
        yield c


@patch("tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker")
def test_backend_transformers_selected(mock_factory, client: TestClient):
    mock_rer = type("_R", (), {"rerank": lambda self, q, d: []})()
    mock_factory.return_value = mock_rer
    payload = {
        "backend": "transformers",
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 1,
    }
    resp = client.post("/v1/reranking", json=payload)
    assert resp.status_code == 200
    # Assert create_reranker was asked for CROSS_ENCODER
    called_args, _ = mock_factory.call_args
    assert called_args[0] == RerankingStrategy.CROSS_ENCODER


@patch("tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker")
def test_backend_llamacpp_selected(mock_factory, client: TestClient):
    mock_rer = type("_R", (), {"rerank": lambda self, q, d: []})()
    mock_factory.return_value = mock_rer
    payload = {
        "backend": "llamacpp",
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 1,
    }
    resp = client.post("/v1/reranking", json=payload)
    assert resp.status_code == 200
    called_args, _ = mock_factory.call_args
    assert called_args[0] == RerankingStrategy.LLAMA_CPP
