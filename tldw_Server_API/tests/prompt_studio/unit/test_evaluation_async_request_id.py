import os
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.main import app


@pytest.fixture
def test_db():
    # Create DB under repository workspace to satisfy sandbox
    # Ensure the tmp directory exists on clean environments
    os.makedirs("tmp", exist_ok=True)
    db_path = os.path.join("tmp", "pstest_eval_bg.db")
    # Ensure clean slate
    try:
        if os.path.exists(db_path):
            os.unlink(db_path)
    except Exception:
        _ = None
    db = PromptStudioDatabase(db_path, "test-client")
    # Seed minimal project/prompt to satisfy FK constraints for evaluations
    try:
        proj = db.create_project(name="Test Project", description="seed", status="active", user_id="test-user")
        pid = int(proj.get("id", 1)) if isinstance(proj, dict) else 1
        db.create_prompt(project_id=pid, name="Seed Prompt", system_prompt="", user_prompt="Hello")
    except Exception:
        _ = None
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            try:
                db.close_connection()
            except Exception:
                _ = None
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
        except Exception:
            _ = None


@pytest.fixture
def override_db_dependency(test_db):
    async def _override_db():
        return test_db

    app.dependency_overrides[deps.get_prompt_studio_db] = _override_db
    yield
    app.dependency_overrides.pop(deps.get_prompt_studio_db, None)


def test_create_evaluation_async_schedules_with_request_id(monkeypatch, override_db_dependency):


    called = {"request_id": None}

    # Ensure code path schedules background task (unset pytest env marker)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("TEST_MODE", "false")

    # Patch the async runner to capture propagated request_id
    import tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_evaluations as eval_ep

    async def fake_run_evaluation_async(evaluation_id, db, *, request_id=None, traceparent="", **_kwargs):
        called["request_id"] = request_id

    monkeypatch.setattr(eval_ep, "run_evaluation_async", fake_run_evaluation_async, raising=True)

    class _ResolvedRuntime:
        async def resolve(self, provider, *, model=None):
            return SimpleNamespace(
                provider=provider,
                api_key="test-openai-key",
                app_config={},
                credentials_resolved=True,
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        eval_ep,
        "ProviderCredentialRuntime",
        lambda **_kwargs: _ResolvedRuntime(),
        raising=True,
    )

    client = TestClient(app)
    api_key = get_settings().SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-REPLACE-ME")
    # Ensure the app sees the expected API key in environment for single-user auth
    monkeypatch.setenv("SINGLE_USER_API_KEY", api_key)

    resp = client.post(
        "/api/v1/prompt-studio/evaluations",
        json={
            "project_id": 1,
            "prompt_id": 1,
            "name": "Async Eval",
            "test_case_ids": [],
            "config": {"model_name": "gpt-4o-mini"},
            "run_async": True,
        },
        headers={
            "X-Request-ID": "req-ps-eval-001",
            "X-API-KEY": api_key,
            "Authorization": f"Bearer {api_key}",
        },
    )
    assert resp.status_code == 200, resp.text

    # BackgroundTasks executes after response generation in TestClient; give a tiny nudge if needed
    if called["request_id"] is None:
        time.sleep(0.01)
    assert called["request_id"] == "req-ps-eval-001"
