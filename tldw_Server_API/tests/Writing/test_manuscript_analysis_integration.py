"""Integration tests for manuscript AI analysis endpoints (mocked LLM)."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration

PREFIX = "/api/v1/writing/manuscripts"

# Module path for patching the LLM call
_LLM_MODULE = "tldw_Server_API.app.core.Chat.chat_service"


def _mock_llm_response(content: str):
    return {"choices": [{"message": {"content": content}}]}


def _pacing_json(**overrides):
    data = {"pacing": 0.6, "tension": 0.4, "atmosphere": 0.5,
            "engagement": 0.7, "assessment": "Well-paced", "beats": ["intro", "climax"]}
    data.update(overrides)
    return json.dumps(data)


def _plot_holes_json():
    return json.dumps({
        "plot_holes": [{"title": "Gap", "description": "Missing info", "severity": "high", "location_hint": "ch1"}],
        "inconsistencies": ["Name changes"],
    })


def _consistency_json():
    return json.dumps({
        "character_issues": [], "world_issues": [],
        "timeline_issues": [], "overall_score": 0.92,
    })


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "analysis_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="analysis_test_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    class _NoopRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return True, {}

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db():
        return db

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_DISABLE", "media,audio")
    monkeypatch.setenv("SKIP_AUDIO_ROUTERS_IN_TESTS", "1")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db
    fastapi_app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()


def _create_project_chapter_scene(client: TestClient) -> tuple[str, str, str]:
    """Helper to create a project, chapter, and scene. Returns (project_id, chapter_id, scene_id)."""
    resp = client.post(f"{PREFIX}/projects", json={"title": "Analysis Novel"})
    assert resp.status_code == 201, resp.text
    project_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter One"},
    )
    assert resp.status_code == 201, resp.text
    chapter_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={
            "title": "Opening Scene",
            "content_plain": "The rain poured down. Alice stepped into the dark alley.",
        },
    )
    assert resp.status_code == 201, resp.text
    scene_id = resp.json()["id"]

    return project_id, chapter_id, scene_id


def _seed_project_and_chapter_analyses(client: TestClient, project_id: str, chapter_id: str) -> None:
    """Create chapter and project analyses so later scene mutations can stale-mark them."""
    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json()),
    ):
        resp = client.post(f"{PREFIX}/chapters/{chapter_id}/analyze", json={"analysis_types": ["pacing"]})
    assert resp.status_code == 200, resp.text

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_consistency_json()),
    ):
        resp = client.post(f"{PREFIX}/projects/{project_id}/analyze/consistency", json={})
    assert resp.status_code == 200, resp.text


def _seed_scene_analysis(client: TestClient, scene_id: str) -> None:
    """Create a scene analysis so delete flows can stale-mark it."""
    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json()),
    ):
        resp = client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json={"analysis_types": ["pacing"]})
    assert resp.status_code == 200, resp.text


def _assert_project_and_chapter_analyses_stale(
    client: TestClient,
    project_id: str,
    *,
    expected_scopes: set[str] | None = None,
) -> None:
    """Assert the cached analyses for a project are marked stale."""
    expected_scopes = expected_scopes or {"chapter", "project"}
    resp = client.get(
        f"{PREFIX}/projects/{project_id}/analyses",
        params={"include_stale": True},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == len(expected_scopes)
    assert {analysis["scope_type"] for analysis in data["analyses"]} == expected_scopes
    assert all(analysis["stale"] is True for analysis in data["analyses"])


def test_analyze_scene(client: TestClient):
    """Create a scene, run pacing analysis, verify response structure and cached in DB."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json()),
    ):
        resp = client.post(
            f"{PREFIX}/scenes/{scene_id}/analyze",
            json={"analysis_types": ["pacing"]},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    analysis = data[0]
    assert analysis["analysis_type"] == "pacing"
    assert analysis["scope_type"] == "scene"
    assert analysis["scope_id"] == scene_id
    assert analysis["result"]["pacing"] == 0.6
    assert analysis["score"] == 0.6
    assert analysis["stale"] is False
    assert "id" in analysis
    assert "created_at" in analysis

    # Verify cached in DB via list endpoint
    resp = client.get(f"{PREFIX}/projects/{project_id}/analyses")
    assert resp.status_code == 200
    list_data = resp.json()
    assert list_data["total"] == 1
    assert list_data["analyses"][0]["id"] == analysis["id"]


def test_analyze_chapter(client: TestClient):
    """Create chapter with 2 scenes, analyze chapter, verify."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    # Add a second scene
    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={
            "title": "Second Scene",
            "content_plain": "The sun rose. Bob stretched and yawned.",
            "sort_order": 2.0,
        },
    )
    assert resp.status_code == 201

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json(pacing=0.8)),
    ):
        resp = client.post(
            f"{PREFIX}/chapters/{chapter_id}/analyze",
            json={"analysis_types": ["pacing"]},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    assert data[0]["scope_type"] == "chapter"
    assert data[0]["scope_id"] == chapter_id
    assert data[0]["score"] == 0.8


def test_analyze_scene_not_found(client: TestClient):
    """Analyze a non-existent scene returns 404."""
    resp = client.post(
        f"{PREFIX}/scenes/nonexistent-id/analyze",
        json={"analysis_types": ["pacing"]},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("payload", "expected_fragment"),
    [
        ({"analysis_types": []}, "analysis_types"),
        ({"analysis_types": ["invalid"]}, "analysis_types"),
    ],
)
def test_analyze_scene_rejects_invalid_analysis_types(
    client: TestClient,
    payload: dict[str, object],
    expected_fragment: str,
):
    """Invalid or empty analysis types should fail request validation."""
    _, _, scene_id = _create_project_chapter_scene(client)

    resp = client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json=payload)

    assert resp.status_code == 422, resp.text
    assert expected_fragment in resp.text


@pytest.mark.parametrize("scope", ["scene", "chapter"])
def test_analysis_endpoints_enforce_runtime_rate_limit(client: TestClient, scope: str):
    """Scene/chapter analysis endpoints should surface 429 when runtime limits deny the call."""
    _project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    class _RejectingRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return False, {"retry_after": 13}

    client.app.dependency_overrides[get_rate_limiter_dep] = lambda: _RejectingRateLimiter()
    target_id = scene_id if scope == "scene" else chapter_id
    resp = client.post(
        f"{PREFIX}/{scope}s/{target_id}/analyze",
        json={"analysis_types": ["pacing"]},
    )

    assert resp.status_code == 429, resp.text
    assert resp.headers["Retry-After"] == "13"


def test_list_analyses(client: TestClient):
    """Create multiple analyses, list them, verify count."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json()),
    ):
        client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json={"analysis_types": ["pacing"]})

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_consistency_json()),
    ):
        client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json={"analysis_types": ["consistency"]})

    resp = client.get(f"{PREFIX}/projects/{project_id}/analyses")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    types = {a["analysis_type"] for a in data["analyses"]}
    assert types == {"pacing", "consistency"}


def test_list_analyses_filter_by_type(client: TestClient):
    """Filter analyses by analysis_type."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_pacing_json()),
    ):
        client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json={"analysis_types": ["pacing"]})

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_consistency_json()),
    ):
        client.post(f"{PREFIX}/scenes/{scene_id}/analyze", json={"analysis_types": ["consistency"]})

    resp = client.get(f"{PREFIX}/projects/{project_id}/analyses", params={"analysis_type": "pacing"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["analyses"][0]["analysis_type"] == "pacing"


def test_stale_after_scene_create(client: TestClient):
    """Creating a new scene should stale chapter and project analyses."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)
    _seed_project_and_chapter_analyses(client, project_id, chapter_id)

    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={
            "title": "Inserted Scene",
            "content_plain": "A new scene arrives and changes the chapter.",
        },
    )
    assert resp.status_code == 201, resp.text

    _assert_project_and_chapter_analyses_stale(client, project_id)


def test_stale_after_scene_update(client: TestClient):
    """Updating scene body content should stale chapter and project analyses."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)
    _seed_project_and_chapter_analyses(client, project_id, chapter_id)

    # Get scene version for optimistic locking
    resp = client.get(f"{PREFIX}/scenes/{scene_id}")
    assert resp.status_code == 200
    scene_version = resp.json()["version"]

    # Update the scene content (this should mark analyses stale)
    resp = client.patch(
        f"{PREFIX}/scenes/{scene_id}",
        json={"content_plain": "Updated scene content with changes."},
        headers={"expected-version": str(scene_version)},
    )
    assert resp.status_code == 200, resp.text

    _assert_project_and_chapter_analyses_stale(client, project_id)

    resp = client.get(f"{PREFIX}/projects/{project_id}/analyses")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


def test_stale_after_scene_delete(client: TestClient):
    """Deleting a scene should stale chapter and project analyses."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)
    _seed_project_and_chapter_analyses(client, project_id, chapter_id)
    _seed_scene_analysis(client, scene_id)

    resp = client.get(f"{PREFIX}/scenes/{scene_id}")
    assert resp.status_code == 200
    scene_version = resp.json()["version"]

    resp = client.delete(
        f"{PREFIX}/scenes/{scene_id}",
        headers={"expected-version": str(scene_version)},
    )
    assert resp.status_code == 204, resp.text

    _assert_project_and_chapter_analyses_stale(
        client,
        project_id,
        expected_scopes={"scene", "chapter", "project"},
    )

    resp = client.get(f"{PREFIX}/projects/{project_id}/analyses")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


def test_analyze_scene_rejects_unknown_analysis_type(client: TestClient):
    """Invalid analysis types should be rejected at request validation time."""
    _project_id, _chapter_id, scene_id = _create_project_chapter_scene(client)
    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/analyze",
        json={"analysis_types": ["made_up_analysis"]},
    )

    assert resp.status_code == 422, resp.text


def test_analyze_scene_rejects_unknown_provider_override(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Provider overrides must be validated against the configured allowlist."""
    _project_id, _chapter_id, scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(writing_endpoint, "is_model_known_for_provider", lambda *_args, **_kwargs: True)

    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/analyze",
        json={"analysis_types": ["pacing"], "provider": "totally-invalid", "model": "gpt-4o-mini"},
    )

    assert resp.status_code == 400, resp.text
    assert "provider" in resp.json()["detail"].lower()


def test_analyze_scene_rejects_unknown_model_override(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """Model overrides must be validated for the chosen provider."""
    _project_id, _chapter_id, scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(
        writing_endpoint,
        "is_model_known_for_provider",
        lambda provider, model: False if provider == "openai" and model == "bad-model" else True,
    )

    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/analyze",
        json={"analysis_types": ["pacing"], "provider": "openai", "model": "bad-model"},
    )

    assert resp.status_code == 400, resp.text
    assert "model" in resp.json()["detail"].lower()


@pytest.mark.parametrize("analysis_route", ["plot-holes", "consistency"])
def test_project_analysis_enforces_runtime_rate_limit(client: TestClient, analysis_route: str):
    """Project analysis endpoints should surface 429 when runtime limits deny the call."""
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)

    class _RejectingRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return False, {"retry_after": 13}

    client.app.dependency_overrides[get_rate_limiter_dep] = lambda: _RejectingRateLimiter()
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/analyze/{analysis_route}",
        json={},
    )

    assert resp.status_code == 429, resp.text
    assert resp.headers["Retry-After"] == "13"


@pytest.mark.parametrize("analysis_route", ["plot-holes", "consistency"])
def test_project_analysis_rejects_unknown_provider_override(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    analysis_route: str,
):
    """Project analysis overrides must be validated against the configured allowlist."""
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(writing_endpoint, "is_model_known_for_provider", lambda *_args, **_kwargs: True)

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/analyze/{analysis_route}",
        json={"provider": "totally-invalid", "model": "gpt-4o-mini"},
    )

    assert resp.status_code == 400, resp.text
    assert "provider" in resp.json()["detail"].lower()


@pytest.mark.parametrize("analysis_route", ["plot-holes", "consistency"])
def test_project_analysis_rejects_unknown_model_override(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    analysis_route: str,
):
    """Project analysis overrides must validate the selected provider/model pair."""
    project_id, _chapter_id, _scene_id = _create_project_chapter_scene(client)
    import tldw_Server_API.app.api.v1.endpoints.writing_manuscripts as writing_endpoint

    monkeypatch.setattr(
        writing_endpoint,
        "get_provider_manager",
        lambda: SimpleNamespace(providers=["openai"], primary_provider="openai"),
    )
    monkeypatch.setattr(
        writing_endpoint,
        "is_model_known_for_provider",
        lambda provider, model: False if provider == "openai" and model == "bad-model" else True,
    )

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/analyze/{analysis_route}",
        json={"provider": "openai", "model": "bad-model"},
    )

    assert resp.status_code == 400, resp.text
    assert "model" in resp.json()["detail"].lower()


def test_analyze_plot_holes_project(client: TestClient):
    """Run project-level plot hole analysis."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_plot_holes_json()),
    ):
        resp = client.post(
            f"{PREFIX}/projects/{project_id}/analyze/plot-holes",
            json={},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    assert data[0]["analysis_type"] == "plot_holes"
    assert data[0]["scope_type"] == "project"
    assert len(data[0]["result"]["plot_holes"]) == 1


def test_analyze_consistency_project(client: TestClient):
    """Run project-level consistency analysis."""
    project_id, chapter_id, scene_id = _create_project_chapter_scene(client)

    with patch(
        f"{_LLM_MODULE}.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=_mock_llm_response(_consistency_json()),
    ):
        resp = client.post(
            f"{PREFIX}/projects/{project_id}/analyze/consistency",
            json={},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data) == 1
    assert data[0]["analysis_type"] == "consistency"
    assert data[0]["result"]["overall_score"] == 0.92
    assert data[0]["score"] == 0.92
