"""Document Insights owner prompts through HTTP, real storage and cache encoding."""

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.endpoints import service_prompts
from tldw_Server_API.app.api.v1.endpoints.media import document_insights as endpoint
from tldw_Server_API.app.api.v1.utils import cache
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter

pytestmark = pytest.mark.integration
PROMPT_ID = "media.document.insights"
DEFAULT_SYSTEM = """You are a research analyst. Analyze the following document and extract structured insights.
For each category, provide a concise title and detailed content.

Categories to analyze:
- research_gap: What problem or gap does this work address?
- research_question: What is the main research question?
- motivation: Why is this research important?
- methods: What methods or approaches were used?
- key_findings: What are the main results or findings?
- limitations: What are the limitations or caveats?
- future_work: What future work is suggested?
- summary: A brief 2-3 sentence summary

Return JSON with this structure:
{"insights": [{"category": "...", "title": "...", "content": "..."}]}

Important:
- Only include categories that are relevant to the document
- Keep titles short (5-10 words)
- Keep content concise but informative (1-3 sentences)
- If the document is not a research paper, adapt the categories as appropriate
- For non-academic documents, focus on: summary, key_findings, and any applicable categories
- Return ONLY valid JSON, no other text
"""


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Replace external storage transport/model calls, not prompt or cache logic."""
    state = SimpleNamespace(
        owner=1,
        calls=[],
        reads=[],
        cache={},
        content="Document {literal} facts.",
        databases={owner: PromptsDatabase(tmp_path / f"{owner}.sqlite", "insights-test") for owner in (1, 2)},
        output={"insights": [{"category": "summary", "title": "Result", "content": "Facts."}]},
        during_model=None,
    )
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/media")
    app.include_router(service_prompts.router, prefix="/api/v1")

    async def user() -> User:
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def database(request: Request, owner: User) -> PromptsDatabase:
        state.reads.append(owner.id)
        return state.databases[owner.id]

    class MediaStore:
        """Stand in for the existing media read boundary, keeping scope explicit."""

        db_path_str = str(tmp_path / "media.db")

        def get_media_by_id(self, media_id: int, **kwargs: Any) -> dict[str, Any]:
            return {"id": media_id, "content": state.content, "type": "pdf"}

    class CacheTransport:
        """In-memory Redis transport; production cache encoding/lookup stays real."""

        def get(self, key: str) -> str | None:
            return state.cache.get(key)

        def setex(self, key: str, ttl: int, value: str) -> None:
            state.cache[key] = value

        def sadd(self, *args: Any) -> None:
            pass

        def expire(self, *args: Any) -> None:
            pass

    def model(self: OpenAIAdapter, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        state.calls.append(payload)
        if state.during_model:
            state.during_model()
        return {
            "choices": [
                {"message": {"role": "assistant", "content": json.dumps(state.output)}, "finish_reason": "stop"}
            ]
        }

    async def no_rate_limit() -> None:
        pass

    for route in app.routes:
        for dep in getattr(getattr(route, "dependant", None), "dependencies", []):
            if getattr(dep.call, "_tldw_rate_limit_resource", None):
                app.dependency_overrides[dep.call] = no_rate_limit
    app.dependency_overrides[get_request_user] = user
    app.dependency_overrides[get_media_db_for_user] = MediaStore
    app.dependency_overrides[get_prompts_db_for_user] = lambda: state.databases[state.owner]
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=state.owner,
        username=f"owner-{state.owner}",
        subject=f"user:{state.owner}",
        token_type="access",
    )
    monkeypatch.setattr(endpoint, "get_prompts_db_for_user", database, raising=False)
    monkeypatch.setattr(endpoint, "DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(endpoint, "resolve_provider_api_key", lambda *a, **kw: ("test-key", None))
    monkeypatch.setattr(endpoint, "load_and_log_configs", lambda: {"openai_api": {"model": "test-model"}})
    monkeypatch.setattr(OpenAIAdapter, "chat", model)
    monkeypatch.setattr(cache, "get_cache_client", CacheTransport)
    with TestClient(app, raise_server_exceptions=False) as client:
        state.client = client
        yield state
    for db in state.databases.values():
        db.close_connection()


def save(
    context: SimpleNamespace,
    analysis: str = "Analyze {literally} in French.",
    presentation: str = "Use detailed paragraphs.",
) -> None:
    """Write a real owner override, including its optimistic revision."""
    db = context.databases[context.owner]
    row = db.get_service_prompt_override(PROMPT_ID)
    db.save_service_prompt_override(
        PROMPT_ID,
        {
            "analysis_guidance": analysis,
            "presentation_guidance": presentation,
        },
        row.revision if row else None,
    )


def generate(context: SimpleNamespace, **options: Any) -> dict[str, Any]:
    """Request real insights and retain failures as useful assertions."""
    response = context.client.post("/api/v1/media/7/insights", json=options)
    assert response.status_code == 200, response.text
    return response.json()


def test_default_messages_and_provider_controls_are_unchanged(context: SimpleNamespace) -> None:
    generate(context, categories=["methods"], model="chosen-model", max_content_length=500)
    payload = context.calls[0]
    assert payload["messages"] == [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {
            "role": "user",
            "content": "Analyze this document and extract insights:\n\n---\nDocument {literal} facts.\n---\n\n\nOnly generate insights for these categories: methods",
        },
    ]
    assert {key: payload[key] for key in ("model", "temperature", "max_tokens", "response_format")} == {
        "model": "chosen-model",
        "temperature": 0.3,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }


def test_saved_guidance_changes_model_message_but_not_json_contract(context: SimpleNamespace) -> None:
    save(context)
    generate(context)
    assert context.calls[0]["messages"][0]["content"] == (
        "Analyze {literally} in French.\n\nReturn JSON with this structure:\n"
        '{"insights": [{"category": "...", "title": "...", "content": "..."}]}\n\n'
        "Use detailed paragraphs.\n- Return ONLY valid JSON, no other text\n"
    )
    assert context.reads == [1]


def test_prompt_edits_miss_cache_and_unchanged_prompts_hit(context: SimpleNamespace) -> None:
    assert generate(context)["cached"] is False
    assert generate(context)["cached"] is True
    save(context)
    assert generate(context)["cached"] is False
    assert generate(context)["cached"] is True
    save(context, presentation="Use a short sentence.")
    assert generate(context)["cached"] is False
    assert len(context.calls) == 3
    assert all("French" not in key for key in context.cache)


def test_owner_overrides_and_cache_entries_are_isolated(context: SimpleNamespace) -> None:
    save(context, "Owner one", "One style")
    generate(context)
    context.owner = 2
    save(context, "Owner two", "Two style")
    assert generate(context)["cached"] is False
    assert context.calls[-1]["messages"][0]["content"].startswith("Owner two\n")
    context.owner = 1
    assert generate(context)["cached"] is True
    assert context.calls[0]["messages"][0]["content"].startswith("Owner one\n")


def test_mid_request_edit_does_not_poison_new_prompt_cache(context: SimpleNamespace) -> None:
    save(context, "Old instructions", "Old style")
    context.during_model = lambda: save(context, "New instructions", "New style")
    generate(context)
    context.during_model = None
    assert generate(context)["cached"] is False
    assert context.calls[0]["messages"][0]["content"].startswith("Old instructions\n")
    assert context.calls[1]["messages"][0]["content"].startswith("New instructions\n")


def test_custom_guidance_does_not_disable_output_normalization(context: SimpleNamespace) -> None:
    save(context, "Invent categories", "Use any output shape")
    context.output = {
        "insights": [
            None,
            {"category": "invented", "title": "Bad", "content": "Bad"},
            {"category": "summary", "title": "", "content": "Bad"},
            {"category": "methods", "title": "Good", "content": "Kept", "confidence": 2},
        ]
    }
    assert generate(context)["insights"] == [
        {"category": "methods", "title": "Good", "content": "Kept", "confidence": 1.0}
    ]


def test_settings_save_and_reset_change_generation_and_cache(context: SimpleNamespace) -> None:
    path = f"/api/v1/service-prompts/{PROMPT_ID}"
    detail = context.client.get(path)
    assert detail.status_code == 200
    assert set(detail.json()["effective_parts"]) == {"analysis_guidance", "presentation_guidance"}
    saved = context.client.put(
        path,
        json={
            "parts": {
                "analysis_guidance": "Find practical applications.",
                "presentation_guidance": "Explain simply.",
            },
            "expected_revision": None,
        },
    )
    assert saved.status_code == 200, saved.text
    assert generate(context)["cached"] is False
    assert context.calls[-1]["messages"][0]["content"].startswith("Find practical applications.\n")
    reset = context.client.delete(path, params={"expected_revision": saved.json()["revision"]})
    assert reset.status_code == 200, reset.text
    assert reset.json()["source"] == "packaged"
    assert generate(context)["cached"] is False
    assert context.calls[-1]["messages"][0]["content"] == DEFAULT_SYSTEM


def test_invalid_saved_guidance_does_not_return_previously_cached_insights(context: SimpleNamespace) -> None:
    generate(context)
    db = context.databases[1]
    db.save_service_prompt_override(PROMPT_ID, {"unexpected": "private prompt text"}, None)
    response = context.client.post("/api/v1/media/7/insights", json={})
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to load document insights guidance"}
    assert len(context.calls) == 1


@pytest.mark.parametrize("fail_read", [False, True])
def test_lookup_connection_is_closed_on_its_worker(
    context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    fail_read: bool,
) -> None:
    db = context.databases[1]
    events = []
    original_read = db.get_service_prompt_override
    original_close = db.close_connection

    def read(definition_id: str) -> Any:
        events.append(("read", threading.get_ident()))
        if fail_read:
            raise RuntimeError("private database path")
        return original_read(definition_id)

    def close() -> None:
        events.append(("close", threading.get_ident()))
        original_close()

    monkeypatch.setattr(db, "get_service_prompt_override", read)
    monkeypatch.setattr(db, "close_connection", close)
    response = context.client.post("/api/v1/media/7/insights", json={})
    assert response.status_code == (500 if fail_read else 200)
    assert [event for event, _ in events] == ["read", "close"]
    assert events[0][1] == events[1][1]
    assert events[0][1] != threading.get_ident()


def test_truncation_and_force_bypass_remain_effective(context: SimpleNamespace) -> None:
    save(context)
    context.content = "x" * 501
    generate(context, max_content_length=500)
    assert generate(context, max_content_length=500)["cached"] is True
    assert generate(context, max_content_length=500, force=True)["cached"] is False
    assert context.calls[-1]["messages"][1]["content"] == (
        "Analyze this document and extract insights:\n\n---\n"
        + "x" * 500
        + "\n\n[Content truncated for analysis...]\n---\n"
    )
