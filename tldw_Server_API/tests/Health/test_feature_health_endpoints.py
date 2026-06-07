import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks, llm_providers, notes, prompts


@pytest.fixture()
def client():
    test_app = FastAPI()
    test_app.include_router(notes.router, prefix="/api/v1/notes")
    test_app.include_router(prompts.router, prefix="/api/v1/prompts")
    test_app.include_router(chatbooks.router, prefix="/api/v1")
    test_app.include_router(llm_providers.router, prefix="/api/v1")

    with TestClient(test_app) as c:
        yield c


def test_notes_health_endpoint(client):
    resp = client.get("/api/v1/notes/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "notes"
    assert "status" in data


def test_prompts_health_endpoint(client):
    resp = client.get("/api/v1/prompts/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "prompts"
    assert "status" in data


def test_chatbooks_health_endpoint(client):
    resp = client.get("/api/v1/chatbooks/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "chatbooks"
    assert "status" in data


def test_llm_health_endpoint(client):
    resp = client.get("/api/v1/llm/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("service") == "llm_inference"
    assert "status" in data
