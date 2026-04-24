"""Tests for archetype API endpoints (list / detail / preview)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import router
from tldw_Server_API.app.core.Persona.archetype_loader import (
    _CACHE,
    load_archetypes_from_directory,
)

pytestmark = pytest.mark.integration

# Minimal app with just the archetype router
_app = FastAPI()
_app.include_router(router, prefix="/api/v1/persona/archetypes")

# Path to the real archetype YAML files checked into the repo
_ARCHETYPES_DIR = Path(__file__).resolve().parents[2] / "Config_Files" / "persona_archetypes"


@pytest.fixture(autouse=True)
def _load_real_archetypes():
    """Load archetypes from the repo config directory before each test."""
    load_archetypes_from_directory(_ARCHETYPES_DIR)
    yield
    _CACHE.clear()


@pytest.fixture()
def client():
    with TestClient(_app) as c:
        yield c


# -- List endpoint -----------------------------------------------------------


class TestListArchetypes:
    def test_returns_200_with_list(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes")
        assert r.status_code == 200
        payload = r.json()
        assert isinstance(payload, list)
        assert len(payload) > 0

    def test_contains_expected_keys(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes")
        keys = {item["key"] for item in r.json()}
        assert "research_assistant" in keys
        assert "study_buddy" in keys
        assert "writing_coach" in keys

    def test_summary_shape(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes")
        for item in r.json():
            assert "key" in item
            assert "label" in item
            assert "tagline" in item
            assert "icon" in item


# -- Detail endpoint ---------------------------------------------------------


class TestGetArchetype:
    def test_returns_200_for_known_key(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/research_assistant")
        assert r.status_code == 200
        body = r.json()
        assert body["key"] == "research_assistant"
        assert body["label"] == "Research Assistant"
        assert "persona" in body
        assert body["persona"]["name"] == "Research Assistant"

    def test_returns_404_for_unknown_key(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/nonexistent")
        assert r.status_code == 404
        assert "nonexistent" in r.json()["detail"]

    def test_full_template_has_expected_sections(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/research_assistant")
        body = r.json()
        assert "mcp_modules" in body
        assert "policy" in body
        assert "voice_defaults" in body
        assert "buddy" in body
        assert "starter_commands" in body


# -- Preview endpoint --------------------------------------------------------


class TestArchetypePreview:
    def test_returns_200_for_known_key(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/research_assistant/preview")
        assert r.status_code == 200

    def test_returns_404_for_unknown_key(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/nonexistent/preview")
        assert r.status_code == 404

    def test_preview_shape(self, client: TestClient):
        r = client.get("/api/v1/persona/archetypes/research_assistant/preview")
        body = r.json()
        assert body["name"] == "Research Assistant"
        assert isinstance(body["system_prompt"], str)
        assert body["archetype_key"] == "research_assistant"
        assert isinstance(body["voice_defaults"], dict)
        assert body["setup"] == {"status": "not_started", "current_step": "archetype"}
