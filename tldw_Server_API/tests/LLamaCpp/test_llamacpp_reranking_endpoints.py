import json
import builtins
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


@pytest.fixture(scope="module")
def client():
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as c:
        yield c


class DummyReranker:
    async def rerank(self, query, documents, original_scores=None):
        # Return decreasing scores to test ordering and top limits
        out = []
        for i, d in enumerate(documents):
            score = 1.0 - (i * 0.1)
            out.append(type("_SD", (), {"document": d, "rerank_score": score}))
        return out


class ExplodingReranker:
    async def rerank(self, query, documents, original_scores=None):
        raise RuntimeError("reranker backend exploded")


@patch("tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker", autospec=True)
def test_public_reranking_basic(mock_factory, client: TestClient):
    mock_factory.return_value = DummyReranker()

    payload = {
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
        "query": "What is panda?",
        "top_n": 2,
        "documents": ["hi", "it is a bear", "The giant panda (Ailuropoda melanoleuca) ..."],
    }
    resp = client.post("/v1/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    # Should be in decreasing order of score
    scores = [r["score"] for r in data["results"]]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


@patch("tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker", autospec=True)
def test_llamacpp_reranking_passages(mock_factory, client: TestClient):
    mock_factory.return_value = DummyReranker()

    payload = {
        "query": "What do llamas eat?",
        "top_k": 2,
        "passages": [
            {"id": "a", "text": "Llamas eat bananas"},
            {"id": "b", "text": "Llamas in pyjamas"},
            {"id": "c", "text": "A bowl of fruit salad"},
        ],
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
    }
    resp = client.post("/api/v1/llamacpp/reranking", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 2
    # The first result should be highest score
    assert data["results"][0]["score"] >= data["results"][1]["score"]


def test_llamacpp_reranking_sanitizes_module_import_failure(client: TestClient, monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "tldw_Server_API.app.core.RAG.rag_service.advanced_reranking",
            "tldw_Server_API.app.core.RAG.rag_service.types",
        }:
            raise ImportError("reranking modules exploded")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    payload = {
        "query": "What do llamas eat?",
        "top_k": 2,
        "passages": [{"id": "a", "text": "Llamas eat grass"}],
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
    }
    response = client.post("/api/v1/llamacpp/reranking", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to initialize reranking"


@patch("tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker", autospec=True)
def test_llamacpp_reranking_sanitizes_backend_failure(mock_factory, client: TestClient):
    mock_factory.return_value = ExplodingReranker()

    payload = {
        "query": "What do llamas eat?",
        "top_k": 2,
        "passages": [{"id": "a", "text": "Llamas eat grass"}],
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
    }
    response = client.post("/api/v1/llamacpp/reranking", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to rerank passages"


def test_public_reranking_sanitizes_backend_failure(client: TestClient, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import llamacpp as mod

    async def _fake_run_public_rerank(**_kwargs):
        raise RuntimeError("public reranker backend exploded")

    monkeypatch.setattr(mod, "_run_public_rerank", _fake_run_public_rerank)

    payload = {
        "model": "/models/Qwen3-Embedding-0.6B_f16.gguf",
        "query": "What is panda?",
        "top_n": 2,
        "documents": ["hi", "it is a bear"],
    }
    response = client.post("/v1/reranking", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to rerank documents"
