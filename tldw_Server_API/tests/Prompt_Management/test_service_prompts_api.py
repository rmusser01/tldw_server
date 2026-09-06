from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.api.v1.endpoints import service_prompts as service_prompts_module
from tldw_Server_API.app.api.v1.schemas.service_prompt_schemas import (
    ServicePromptCatalogItemResponse,
    ServicePromptDetailResponse,
    ServicePromptUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    DatabaseError,
    PromptsDatabase,
)
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

pytestmark = pytest.mark.integration

TRANSLATION_ID = "media.text.translation"
TRANSLATION_PATH = f"/api/v1/service-prompts/{TRANSLATION_ID}"
CUSTOM_PARTS = {
    "system": "Translate faithfully.",
    "user_template": "Translate {text} into {target_language}.",
}
TITLE_ID = "chat.title.generation"
TITLE_PATH = f"/api/v1/service-prompts/{TITLE_ID}"
TITLE_CUSTOM_PARTS = {"user_template": "Name this request: {query}"}
NOTES_TITLE_ID = "notes.title.generate"
NOTES_TITLE_PATH = f"/api/v1/service-prompts/{NOTES_TITLE_ID}"
NOTES_TITLE_CUSTOM_PARTS = {
    "system": "Write note titles in the account style.",
    "title_instruction": "Create a specific note title",
}


def _principal(*, api_key_id: int | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="api_key" if api_key_id is not None else "user",
        user_id=1,
        api_key_id=api_key_id,
        username="service-prompt-test",
        subject="user:1",
        token_type="api_key" if api_key_id is not None else "access",
    )


@pytest.fixture
def api_context(tmp_path: Path):
    db = PromptsDatabase(tmp_path / "service-prompts.sqlite", "service-prompts-test")
    app = FastAPI()
    app.state.db = db
    app.state.principal = _principal()
    app.state.api_key_scope = None

    async def override_principal(request: Request) -> AuthPrincipal:
        if app.state.api_key_scope is not None:
            request.state._api_key_scope = app.state.api_key_scope
        return app.state.principal

    def override_db() -> PromptsDatabase:
        return app.state.db

    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[get_prompts_db_for_user] = override_db
    app.include_router(service_prompts_module.router, prefix="/api/v1")
    with TestClient(app) as local_client:
        yield SimpleNamespace(app=app, client=local_client, db=db)
    db.close_connection()


def test_service_prompt_catalog_returns_exact_metadata_without_prompt_bodies(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/service-prompts")

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == [
        {
            "id": "chat.rag.answer",
            "label": "RAG answer",
            "description": ("Controls how retrieved context and the current question are presented to the model."),
            "parts": [
                {
                    "key": "template",
                    "label": "Template",
                    "mode": "template",
                    "required_variables": ["context", "question"],
                }
            ],
            "affected_workflows": [
                {"id": "chat.main.rag", "label": "Main chat RAG"},
                {"id": "chat.tab.rag", "label": "Tab chat RAG"},
                {"id": "chat.document.rag", "label": "Document chat RAG"},
                {"id": "chat.sidepanel.rag", "label": "Sidepanel RAG"},
            ],
        },
        {
            "id": "chat.rag.question_rewrite",
            "label": "RAG follow-up rewrite",
            "description": ("Controls how a conversational follow-up is rewritten into a standalone retrieval query."),
            "parts": [
                {
                    "key": "template",
                    "label": "Template",
                    "mode": "template",
                    "required_variables": ["chat_history", "question"],
                }
            ],
            "affected_workflows": [
                {"id": "chat.main.rag", "label": "Main chat RAG"},
                {"id": "chat.document.rag", "label": "Document chat RAG"},
                {"id": "chat.sidepanel.rag", "label": "Sidepanel RAG"},
            ],
        },
        {
            "id": "chat.web_search.answer",
            "label": "Web-search answer",
            "description": ("Controls how normalized web-search results are presented for the final answer."),
            "parts": [
                {
                    "key": "template",
                    "label": "Template",
                    "mode": "template",
                    "required_variables": ["current_date_time", "search_results"],
                }
            ],
            "affected_workflows": [
                {"id": "chat.main.web_search", "label": "Main chat web search"},
                {"id": "chat.compare.web_search", "label": "Compare web search"},
            ],
        },
        {
            "id": "chat.title.generation",
            "label": "Conversation title",
            "description": ("Controls the instruction used to generate automatic conversation titles."),
            "parts": [
                {
                    "key": "user_template",
                    "label": "User template",
                    "mode": "template",
                    "required_variables": ["query"],
                }
            ],
            "affected_workflows": [{"id": "chat.title.generation", "label": "Automatic conversation titles"}],
        },
        {
            "id": "image.prompt.refinement",
            "label": "Image prompt refinement",
            "description": ("Controls the semantic instructions used to refine image-generation prompt drafts."),
            "parts": [
                {
                    "key": "system_semantics",
                    "label": "Refinement guidance",
                    "mode": "literal",
                    "required_variables": [],
                },
                {
                    "key": "rewrite_semantics",
                    "label": "Rewrite guidance",
                    "mode": "literal",
                    "required_variables": [],
                },
            ],
            "affected_workflows": [{"id": "image.prompt.refinement", "label": "Image prompt refinement"}],
        },
        {
            "id": "media.document.summarization",
            "label": "Document summarization",
            "description": "Controls system instructions for synchronous document analysis. Without a saved override, server defaults apply.",
            "parts": [{"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []}],
            "affected_workflows": [{"id": "media.document.summarization", "label": "Synchronous document analysis"}],
        },
        {
            "id": "media.pdf.summarization",
            "label": "PDF summarization",
            "description": "Controls system instructions for synchronous PDF analysis. Without a saved override, server defaults apply.",
            "parts": [{"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []}],
            "affected_workflows": [{"id": "media.pdf.summarization", "label": "Synchronous PDF analysis"}],
        },
        {
            "id": "media.ebook.summarization",
            "label": "EPUB summarization",
            "description": "Controls system instructions for synchronous EPUB analysis. Without a saved override, server defaults apply.",
            "parts": [{"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []}],
            "affected_workflows": [{"id": "media.ebook.summarization", "label": "Synchronous EPUB analysis"}],
        },
        {
            "id": "media.email.summarization",
            "label": "Email summarization",
            "description": "Controls system instructions for synchronous email analysis. Without a saved override, server defaults apply.",
            "parts": [{"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []}],
            "affected_workflows": [{"id": "media.email.summarization", "label": "Synchronous email analysis"}],
        },
        {
            "id": "media.audio.analysis",
            "label": "Audio summarization",
            "description": "Controls system and user instructions for synchronous audio analysis. Without a saved override, server defaults apply.",
            "parts": [
                {"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []},
                {"key": "user", "label": "User instructions", "mode": "literal", "required_variables": []},
            ],
            "affected_workflows": [{"id": "media.audio.analysis", "label": "Synchronous audio analysis"}],
        },
        {
            "id": "media.video.summarization",
            "label": "Video summarization",
            "description": "Controls system instructions and recursive final-summary instructions for synchronous video analysis. Without a saved override, server defaults apply.",
            "parts": [
                {"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []},
                {
                    "key": "final_summary",
                    "label": "Final-summary instructions",
                    "mode": "literal",
                    "required_variables": [],
                },
            ],
            "affected_workflows": [{"id": "media.video.summarization", "label": "Synchronous video analysis"}],
        },
        {
            "id": "media.web.summarization",
            "label": "Web article summarization",
            "description": "Controls summary instructions for synchronous web scraping. Reset restores each scraping engine's existing defaults; the displayed defaults are the deployed web-article prompts.",
            "parts": [
                {"key": "system", "label": "System instructions", "mode": "literal", "required_variables": []},
                {"key": "user", "label": "User instructions", "mode": "literal", "required_variables": []},
            ],
            "affected_workflows": [{"id": "media.web.summarization", "label": "Synchronous web scraping"}],
        },
        {
            "id": "media.text.translation",
            "label": "Text translation",
            "description": ("Controls the visible instructions used by synchronous text translation."),
            "parts": [
                {
                    "key": "system",
                    "label": "System instructions",
                    "mode": "literal",
                    "required_variables": [],
                },
                {
                    "key": "user_template",
                    "label": "User template",
                    "mode": "template",
                    "required_variables": ["target_language", "text"],
                },
            ],
            "affected_workflows": [{"id": "media.text.translation", "label": "Text translation"}],
        },
        {
            "id": "notes.title.generate",
            "label": "Notes title",
            "description": "Controls the wording used by LLM-backed automatic Notes titles.",
            "parts": [
                {
                    "key": "system",
                    "label": "System instructions",
                    "mode": "literal",
                    "required_variables": [],
                },
                {
                    "key": "title_instruction",
                    "label": "Title instruction",
                    "mode": "literal",
                    "required_variables": [],
                },
            ],
            "affected_workflows": [{"id": "notes.title.generate", "label": "Automatic Notes titles"}],
        },
    ]
    assert "You are a helpful AI assistant" not in response.text


def test_service_prompt_detail_returns_packaged_state_without_caching(api_context) -> None:
    response = api_context.client.get(TRANSLATION_PATH)

    assert response.status_code == 200
    body = response.json()
    assert response.headers["cache-control"] == "no-store"
    assert body["source"] == "packaged"
    assert body["revision"] is None
    assert body["saved_parts"] is None
    assert body["effective_parts"] == body["default_parts"]
    assert set(body["default_parts"]) == {"system", "user_template"}


def test_web_settings_save_pair_atomically_and_reset_deployment_defaults(
    api_context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Web instructions share atomic updates and expose the deployed article defaults."""
    from tldw_Server_API.app.core.Utils import prompt_loader

    (tmp_path / "webscraping.prompts.yaml").write_text(
        "article_summary_system: Deployment system\narticle_summary_user: Deployment summary\n"
    )
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(prompt_loader, "get_global_context_integrity_resolver", lambda: None)
    path = "/api/v1/service-prompts/media.web.summarization"
    defaults = {"system": "Deployment system", "user": "Deployment summary"}
    assert api_context.client.get(path).json()["effective_parts"] == defaults
    parts = {"system": "System {literal}", "user": "Summary {literal}"}
    saved = api_context.client.put(path, json={"parts": parts, "expected_revision": None})
    assert saved.status_code == 200
    revision = saved.json()["revision"]
    partial = api_context.client.put(path, json={"parts": {"user": "Incomplete"}, "expected_revision": revision})
    assert partial.status_code == 422
    assert api_context.client.get(path).json()["effective_parts"] == parts
    reset = api_context.client.delete(path, params={"expected_revision": revision})
    assert reset.status_code == 200
    assert reset.json()["effective_parts"] == defaults


def test_video_settings_save_pair_atomically_and_reset_defaults(
    api_context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two video stages remain one atomic Settings value with effective defaults."""
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summary

    monkeypatch.setattr(summary, "_resolve_default_system_prompt", lambda: "Deployment system")
    path = "/api/v1/service-prompts/media.video.summarization"
    defaults = {
        "system": "Deployment system",
        "final_summary": "Summarize the key points from the preceding text sections.",
    }
    detail = api_context.client.get(path)
    assert detail.status_code == 200
    assert detail.json()["effective_parts"] == defaults
    parts = {"system": "System {literal}", "final_summary": "Final {literal}"}
    saved = api_context.client.put(path, json={"parts": parts, "expected_revision": None})
    assert saved.status_code == 200
    revision = saved.json()["revision"]
    partial = api_context.client.put(path, json={"parts": {"system": "Incomplete"}, "expected_revision": revision})
    assert partial.status_code == 422
    assert api_context.client.get(path).json()["effective_parts"] == parts
    reset = api_context.client.delete(path, params={"expected_revision": revision})
    assert reset.status_code == 200
    assert reset.json()["effective_parts"] == defaults


def test_audio_settings_save_pair_atomically_and_reset_deployment_defaults(
    api_context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Partial writes cannot replace either half; reset returns effective audio file defaults."""
    from tldw_Server_API.app.core.Utils import prompt_loader

    (tmp_path / "audio.prompts.yaml").write_text(
        "system_prompt: Deployment system\ntranscription_analysis_summary: Deployment user\n"
    )
    monkeypatch.setattr(prompt_loader, "_prompts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(prompt_loader, "get_global_context_integrity_resolver", lambda: None)
    path = "/api/v1/service-prompts/media.audio.analysis"
    defaults = {"system": "Deployment system", "user": "Deployment user"}
    detail = api_context.client.get(path)
    assert detail.status_code == 200
    assert detail.json()["effective_parts"] == defaults
    parts = {"system": "System {literal}", "user": "User {literal}"}
    saved = api_context.client.put(path, json={"parts": parts, "expected_revision": None})
    assert saved.status_code == 200
    assert saved.json()["effective_parts"] == parts
    revision = saved.json()["revision"]
    partial = api_context.client.put(path, json={"parts": {"system": "Incomplete"}, "expected_revision": revision})
    assert partial.status_code == 422
    assert api_context.client.get(path).json()["effective_parts"] == parts
    reset = api_context.client.delete(path, params={"expected_revision": revision})
    assert reset.status_code == 200
    assert reset.json()["effective_parts"] == defaults


@pytest.mark.parametrize(
    "prompt_id",
    [
        "media.document.summarization",
        "media.pdf.summarization",
        "media.ebook.summarization",
        "media.email.summarization",
    ],
)
def test_summary_prompt_can_be_saved_and_reset(api_context: SimpleNamespace, prompt_id: str) -> None:
    """Exercise independent media guidance through the generic save/reset API."""
    path = f"/api/v1/service-prompts/{prompt_id}"
    parts = {"system": "Summarize in French. Preserve literal {braces}."}
    saved = api_context.client.put(path, json={"parts": parts, "expected_revision": None})
    assert saved.status_code == 200
    assert saved.json()["effective_parts"] == parts
    reset = api_context.client.delete(path, params={"expected_revision": saved.json()["revision"]})
    assert reset.status_code == 200
    assert reset.json()["source"] == "packaged"


@pytest.mark.parametrize(
    "prompt_id",
    [
        "media.document.summarization",
        "media.pdf.summarization",
        "media.ebook.summarization",
        "media.email.summarization",
    ],
)
def test_summary_settings_uses_deployment_default_for_detail_and_reset(
    api_context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, prompt_id: str
) -> None:
    """Saving an unchanged Settings draft must preserve the actual runtime default."""
    monkeypatch.setattr(summarization, "load_prompt", lambda *args: "Deployment summary guidance.")
    path = f"/api/v1/service-prompts/{prompt_id}"
    detail = api_context.client.get(path)
    assert detail.status_code == 200
    assert detail.json()["effective_parts"] == {"system": "Deployment summary guidance."}
    saved = api_context.client.put(path, json={"parts": detail.json()["effective_parts"], "expected_revision": None})
    assert saved.status_code == 200
    assert saved.json()["effective_parts"] == {"system": "Deployment summary guidance."}
    reset = api_context.client.delete(path, params={"expected_revision": saved.json()["revision"]})
    assert reset.status_code == 200
    assert reset.json()["effective_parts"] == {"system": "Deployment summary guidance."}
    assert reset.json()["saved_parts"] is None


def test_title_prompt_can_be_saved_and_reset_through_generic_api(api_context) -> None:
    saved = api_context.client.put(
        TITLE_PATH,
        json={"parts": TITLE_CUSTOM_PARTS, "expected_revision": None},
    )
    assert saved.status_code == 200
    assert saved.json()["effective_parts"] == TITLE_CUSTOM_PARTS

    reset = api_context.client.delete(
        TITLE_PATH,
        params={"expected_revision": saved.json()["revision"]},
    )
    assert reset.status_code == 200
    assert reset.json()["source"] == "packaged"
    assert reset.json()["effective_parts"] == reset.json()["default_parts"]


def test_notes_title_prompt_can_be_saved_and_reset_through_generic_api(api_context) -> None:
    saved = api_context.client.put(
        NOTES_TITLE_PATH,
        json={"parts": NOTES_TITLE_CUSTOM_PARTS, "expected_revision": None},
    )
    assert saved.status_code == 200
    assert saved.json()["effective_parts"] == NOTES_TITLE_CUSTOM_PARTS

    reset = api_context.client.delete(
        NOTES_TITLE_PATH,
        params={"expected_revision": saved.json()["revision"]},
    )
    assert reset.status_code == 200
    assert reset.json()["source"] == "packaged"
    assert reset.json()["effective_parts"] == reset.json()["default_parts"]


def test_service_prompt_put_activates_immediately_and_identical_retry_keeps_revision(
    api_context,
) -> None:
    first = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )

    assert first.status_code == 200, first.text
    first_body = first.json()
    UUID(first_body["revision"])
    assert first.headers["cache-control"] == "no-store"
    assert first_body["source"] == "user"
    assert first_body["saved_parts"] == CUSTOM_PARTS
    assert first_body["effective_parts"] == CUSTOM_PARTS
    assert api_context.client.get(TRANSLATION_PATH).json()["effective_parts"] == CUSTOM_PARTS

    retry = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )

    assert retry.status_code == 200
    assert retry.json()["revision"] == first_body["revision"]


@pytest.mark.parametrize("method", ["get", "put", "delete"])
def test_service_prompt_detail_rejects_changed_expected_user_without_mutation(
    api_context,
    method: str,
) -> None:
    expected_source = "packaged"
    if method == "delete":
        seeded = api_context.client.put(
            TRANSLATION_PATH,
            json={"parts": CUSTOM_PARTS, "expected_revision": None},
        )
        assert seeded.status_code == 200
        expected_source = "user"
    kwargs: dict[str, object] = {
        "headers": {"X-TLDW-Expected-User-ID": "999"},
    }
    if method == "put":
        kwargs["json"] = {"parts": CUSTOM_PARTS, "expected_revision": None}

    response = getattr(api_context.client, method)(TRANSLATION_PATH, **kwargs)

    assert response.status_code == 412
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "detail": {
            "code": "request_config_scope_changed",
            "message": "The server or authenticated account changed before the request was sent.",
        }
    }
    assert api_context.client.get(TRANSLATION_PATH).json()["source"] == expected_source


def test_service_prompt_detail_accepts_matching_expected_user(api_context) -> None:
    response = api_context.client.put(
        TRANSLATION_PATH,
        headers={"X-TLDW-Expected-User-ID": "1"},
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )

    assert response.status_code == 200
    assert response.json()["source"] == "user"


def test_service_prompt_catalog_rejects_changed_expected_user(api_context) -> None:
    response = api_context.client.get(
        "/api/v1/service-prompts",
        headers={"X-TLDW-Expected-User-ID": "999"},
    )

    assert response.status_code == 412
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "request_config_scope_changed"


def test_service_prompt_detail_openapi_declares_optional_expected_user_header(
    api_context,
) -> None:
    paths = api_context.app.openapi()["paths"]
    collection_parameter = next(
        item
        for item in paths["/api/v1/service-prompts"]["get"]["parameters"]
        if item["in"] == "header" and item["name"] == "X-TLDW-Expected-User-ID"
    )
    assert collection_parameter["required"] is False

    path = paths["/api/v1/service-prompts/{definition_id}"]

    for method in ("get", "put", "delete"):
        parameter = next(
            item
            for item in path[method]["parameters"]
            if item["in"] == "header" and item["name"] == "X-TLDW-Expected-User-ID"
        )
        assert parameter["required"] is False


def test_service_prompt_delete_resets_and_is_idempotent(api_context) -> None:
    saved = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    ).json()

    reset = api_context.client.delete(
        TRANSLATION_PATH,
        params={"expected_revision": saved["revision"]},
    )
    repeated = api_context.client.delete(TRANSLATION_PATH)

    assert reset.status_code == 200
    assert reset.headers["cache-control"] == "no-store"
    assert reset.json()["source"] == "packaged"
    assert reset.json()["saved_parts"] is None
    assert reset.json()["revision"] is None
    assert repeated.status_code == 200
    assert repeated.json() == reset.json()


def test_service_prompt_stale_put_and_delete_return_current_revision(api_context) -> None:
    saved = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    ).json()
    stale_revision = str(uuid4())
    changed_parts = {**CUSTOM_PARTS, "system": "Different instructions."}

    put_response = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": changed_parts, "expected_revision": stale_revision},
    )
    delete_response = api_context.client.delete(
        TRANSLATION_PATH,
        params={"expected_revision": stale_revision},
    )

    expected = {
        "detail": {
            "code": "service_prompt_revision_conflict",
            "message": "Service Prompt override changed since it was loaded.",
            "current_revision": saved["revision"],
        }
    }
    assert put_response.status_code == 409
    assert put_response.json() == expected
    assert delete_response.status_code == 409
    assert delete_response.json() == expected


def test_service_prompt_delete_can_reset_corrupt_override(api_context) -> None:
    revision = str(uuid4())
    connection = api_context.db.get_connection()
    connection.execute(
        "INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision) VALUES (?, ?, ?)",
        (TRANSLATION_ID, "not-json", revision),
    )
    connection.commit()

    corrupt = api_context.client.get(TRANSLATION_PATH)
    reset = api_context.client.delete(
        TRANSLATION_PATH,
        params={"expected_revision": revision},
    )

    assert corrupt.status_code == 500
    assert corrupt.json() == {
        "detail": {
            "code": "service_prompt_corrupt_override",
            "message": "The saved Service Prompt override is corrupt and can be reset.",
            "revision": revision,
            "can_reset": True,
        }
    }
    assert reset.status_code == 200
    assert reset.json()["source"] == "packaged"


def test_service_prompt_delete_returns_packaged_detail_without_rereading_parts(
    api_context,
) -> None:
    class ResetOnlyDatabase:
        def reset_service_prompt_override(
            self,
            definition_id: str,
            expected_revision: str | None,
        ) -> None:
            assert definition_id == TRANSLATION_ID
            assert expected_revision is None

        def get_service_prompt_override(self, _definition_id: str):
            pytest.fail("DELETE must not reread stored prompt content after reset")

    api_context.app.state.db = ResetOnlyDatabase()
    response = api_context.client.delete(TRANSLATION_PATH)

    assert response.status_code == 200
    assert response.json()["source"] == "packaged"
    assert response.json()["saved_parts"] is None


def test_service_prompt_unknown_definition_uses_locked_domain_envelope(api_context) -> None:
    response = api_context.client.get("/api/v1/service-prompts/not.registered")

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "detail": {
            "code": "service_prompt_unknown_definition",
            "message": "Service Prompt definition was not found.",
        }
    }


def test_service_prompt_semantic_validation_is_safe_and_field_keyed(api_context) -> None:
    secret = "BODY-MUST-NOT-LEAK"
    captured_logs: list[str] = []
    sink_id = logger.add(captured_logs.append, format="{message}")
    try:
        response = api_context.client.put(
            TRANSLATION_PATH,
            json={
                "parts": {
                    "system": secret,
                    "user_template": "Translate {unknown} into {target_language}.",
                },
                "expected_revision": None,
            },
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "service_prompt_validation_failed",
            "message": "Service Prompt validation failed.",
            "field_errors": {"user_template": "Template variables must match the registered variables exactly once."},
        }
    }
    assert secret not in response.text
    assert secret not in "".join(captured_logs)


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("put", TRANSLATION_PATH, {"json": {"parts": CUSTOM_PARTS}}),
        (
            "put",
            TRANSLATION_PATH,
            {
                "json": {
                    "parts": CUSTOM_PARTS,
                    "expected_revision": "not-a-uuid",
                }
            },
        ),
        (
            "put",
            TRANSLATION_PATH,
            {
                "json": {
                    "parts": CUSTOM_PARTS,
                    "expected_revision": None,
                    "database_path": "/private/forbidden",
                }
            },
        ),
        ("delete", f"{TRANSLATION_PATH}?expected_revision=not-a-uuid", {}),
    ],
)
def test_service_prompt_structural_validation_stays_fastapi_native(
    api_context,
    method: str,
    path: str,
    kwargs: dict[str, object],
) -> None:
    response = getattr(api_context.client, method)(path, **kwargs)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert isinstance(errors, list)
    assert errors
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)


def test_service_prompt_structural_validation_hides_authored_parts(api_context) -> None:
    secret = "STRUCTURAL-BODY-MUST-NOT-LEAK"
    response = api_context.client.put(
        TRANSLATION_PATH,
        json={
            "parts": {
                "system": secret,
                "user_template": "Translate {text} into {target_language}.",
            }
        },
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert secret not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_service_prompt_structural_validation_hides_authored_field_names(
    api_context,
) -> None:
    sentinel = "STRUCTURAL-FIELD-NAME-MUST-NOT-LEAK"
    response = api_context.client.put(
        TRANSLATION_PATH,
        json={
            "parts": CUSTOM_PARTS,
            "expected_revision": None,
            sentinel: "attacker-controlled value",
        },
    )

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors
    assert all(set(error) == {"type", "loc", "msg"} for error in errors)
    assert sentinel not in response.text


def test_service_prompt_validation_location_preserves_only_public_segments() -> None:
    sentinel = "STRUCTURAL-LOCATION-MUST-NOT-LEAK"

    assert service_prompts_module._sanitize_validation_location(("body", "parts", sentinel, 3)) == [
        "body",
        "parts",
        "field",
        3,
    ]


def test_service_prompt_store_failure_is_content_free(api_context) -> None:
    class FailingDatabase:
        def get_service_prompt_override(self, _definition_id: str):
            raise DatabaseError("BODY-MUST-NOT-LEAK")

        def close_connection(self) -> None:
            raise RuntimeError("CLOSE-MUST-NOT-LEAK")

    api_context.app.state.db = FailingDatabase()
    response = api_context.client.get(TRANSLATION_PATH)

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "service_prompt_store_failed",
            "message": "Service Prompt storage operation failed.",
        }
    }
    assert "BODY-MUST-NOT-LEAK" not in response.text
    assert "CLOSE-MUST-NOT-LEAK" not in response.text


def test_service_prompt_dependency_and_auth_errors_keep_native_shapes(api_context) -> None:
    def unavailable_db():
        raise HTTPException(status_code=503, detail="dependency unavailable")

    api_context.app.dependency_overrides[get_prompts_db_for_user] = unavailable_db
    dependency_response = api_context.client.get(TRANSLATION_PATH)

    async def rejected_principal(_request: Request):
        raise HTTPException(status_code=401, detail="authentication required")

    api_context.app.dependency_overrides[get_auth_principal] = rejected_principal
    auth_response = api_context.client.get("/api/v1/service-prompts")

    assert dependency_response.status_code == 503
    assert dependency_response.json() == {"detail": "dependency unavailable"}
    assert auth_response.status_code == 401
    assert auth_response.json() == {"detail": "authentication required"}


def test_service_prompt_api_key_scopes_and_jwt_bypass(api_context) -> None:
    api_context.app.state.principal = _principal(api_key_id=10)
    api_context.app.state.api_key_scope = "read"

    read_response = api_context.client.get(TRANSLATION_PATH)
    denied_put = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )
    denied_delete = api_context.client.delete(TRANSLATION_PATH)

    assert read_response.status_code == 200
    assert denied_put.status_code == 403
    assert denied_delete.status_code == 403

    api_context.app.state.api_key_scope = "write"
    saved = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )
    assert saved.status_code == 200
    assert (
        api_context.client.delete(
            TRANSLATION_PATH,
            params={"expected_revision": saved.json()["revision"]},
        ).status_code
        == 200
    )

    api_context.app.state.principal = _principal()
    api_context.app.state.api_key_scope = None
    jwt_save = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )
    assert jwt_save.status_code == 200


def test_service_prompt_database_calls_run_off_the_event_loop(
    api_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_threads: set[int] = set()
    database_threads: set[int] = set()
    connection_close_threads: list[int] = []

    async def override_db() -> PromptsDatabase:
        event_loop_threads.add(threading.get_ident())
        return api_context.db

    api_context.app.dependency_overrides[get_prompts_db_for_user] = override_db
    for method_name in (
        "get_service_prompt_override",
        "save_service_prompt_override",
        "reset_service_prompt_override",
    ):
        original = getattr(api_context.db, method_name)

        def tracked(*args, _original=original, **kwargs):
            database_threads.add(threading.get_ident())
            return _original(*args, **kwargs)

        monkeypatch.setattr(api_context.db, method_name, tracked)

    original_close_connection = api_context.db.close_connection

    def tracked_close_connection() -> None:
        connection_close_threads.append(threading.get_ident())
        original_close_connection()

    monkeypatch.setattr(
        api_context.db,
        "close_connection",
        tracked_close_connection,
    )

    get_response = api_context.client.get(TRANSLATION_PATH)
    put_response = api_context.client.put(
        TRANSLATION_PATH,
        json={"parts": CUSTOM_PARTS, "expected_revision": None},
    )
    delete_response = api_context.client.delete(
        TRANSLATION_PATH,
        params={"expected_revision": put_response.json()["revision"]},
    )

    def fail_read(*_args, **_kwargs):
        database_threads.add(threading.get_ident())
        raise DatabaseError("forced read failure")

    monkeypatch.setattr(api_context.db, "get_service_prompt_override", fail_read)
    failed_response = api_context.client.get(TRANSLATION_PATH)

    assert [
        get_response.status_code,
        put_response.status_code,
        delete_response.status_code,
        failed_response.status_code,
    ] == [
        200,
        200,
        200,
        500,
    ]
    assert event_loop_threads
    assert database_threads.isdisjoint(event_loop_threads)
    assert len(connection_close_threads) == 4
    assert database_threads <= set(connection_close_threads)


def test_service_prompt_databases_are_isolated_by_dependency(api_context, tmp_path: Path) -> None:
    second_db = PromptsDatabase(tmp_path / "other-user.sqlite", "other-user")
    try:
        saved = api_context.client.put(
            TRANSLATION_PATH,
            json={"parts": CUSTOM_PARTS, "expected_revision": None},
        ).json()

        api_context.app.state.db = second_db
        other_detail = api_context.client.get(TRANSLATION_PATH)
        other_reset = api_context.client.delete(
            TRANSLATION_PATH,
            params={"expected_revision": saved["revision"]},
        )

        assert other_detail.json()["source"] == "packaged"
        assert other_reset.status_code == 409
        assert other_reset.json()["detail"]["current_revision"] is None

        api_context.app.state.db = api_context.db
        assert api_context.client.get(TRANSLATION_PATH).json()["revision"] == saved["revision"]
    finally:
        second_db.close_connection()


def test_service_prompt_wire_schemas_never_accept_user_or_database_targeting() -> None:
    schema_models = (
        ServicePromptCatalogItemResponse,
        ServicePromptDetailResponse,
        ServicePromptUpdateRequest,
    )

    for model in schema_models:
        assert "user_id" not in model.model_fields
        assert "database_path" not in model.model_fields
