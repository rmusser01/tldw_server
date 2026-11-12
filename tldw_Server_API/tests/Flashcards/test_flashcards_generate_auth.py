"""Flashcard generation must not be reachable without authentication.

`POST /api/v1/flashcards/generate` calls an LLM provider using the operator's
configured credentials. It was rate limited but anonymous, and it was the only
one of the router's 51 routes that resolved no user.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps


@pytest.fixture(autouse=True)
def flashcards_generate_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep background workers out of this suite."""
    monkeypatch.setenv("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
    monkeypatch.setenv("READING_DIGEST_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("TEST_MODE", "1")


def _app_without_a_caller() -> FastAPI:
    """Serve the flashcards router with the rate limiter stubbed but no user."""
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

    async def _fake_check_rate_limit() -> None:
        return

    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    return app


@pytest.mark.unit
def test_flashcard_generation_requires_authentication() -> None:
    """An anonymous caller must not be able to spend the operator's LLM credits."""
    with TestClient(_app_without_a_caller()) as client:
        response = client.post(
            "/api/v1/flashcards/generate",
            json={"text": "photosynthesis converts light into chemical energy",
                  "num_cards": 1},
        )

    assert response.status_code in (401, 403), response.text
