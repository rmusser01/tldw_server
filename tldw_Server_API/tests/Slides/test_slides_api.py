import io
import json
import zipfile

import pytest
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Response
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import get_slides_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import slides as slides_ep
from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Slides.slides_db import ConflictError as SlidesConflictError
from tldw_Server_API.app.core.Slides.slides_db import InputError as SlidesInputError
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.slides_export import SlidesExportError, SlidesExportInputError
from tldw_Server_API.app.core.Slides.slides_generator import SlidesGenerationError
from tldw_Server_API.app.core.Slides.slides_templates import SlidesTemplateInvalidError
from tldw_Server_API.app.core.Slides.visual_styles import list_builtin_visual_styles

_SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9XgU1b0AAAAASUVORK5CYII="
)


def _build_assets(tmp_path):
    assets_dir = tmp_path / "revealjs"
    (assets_dir / "plugin" / "notes").mkdir(parents=True)
    (assets_dir / "theme").mkdir(parents=True)
    (assets_dir / "reveal.js").write_text("// reveal.js", encoding="utf-8")
    (assets_dir / "reveal.css").write_text("/* reveal.css */", encoding="utf-8")
    (assets_dir / "plugin" / "notes" / "notes.js").write_text("// notes", encoding="utf-8")
    (assets_dir / "theme" / "black.css").write_text("/* theme */", encoding="utf-8")
    (assets_dir / "LICENSE.revealjs.txt").write_text("license", encoding="utf-8")
    return assets_dir


def _write_templates(tmp_path):
    templates = {
        "templates": [
            {
                "id": "template-1",
                "name": "Template One",
                "theme": "white",
                "marp_theme": "gaia",
                "settings": {"controls": False, "progress": False},
                "default_slides": [
                    {
                        "order": 0,
                        "layout": "title",
                        "title": "Template Title",
                        "content": "",
                        "speaker_notes": None,
                        "metadata": {},
                    },
                    {
                        "order": 1,
                        "layout": "content",
                        "title": "Template Slide",
                        "content": "- Item 1\n- Item 2",
                        "speaker_notes": None,
                        "metadata": {},
                    },
                ],
                "custom_css": ".reveal { font-size: 36px; }",
            }
        ]
    }
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(templates), encoding="utf-8")
    return path


class FakeNotesDB:
    def __init__(self) -> None:
        self.conversations = {"conv_1": {"id": "conv_1", "title": "Conversation"}}
        self.messages = {
            "conv_1": [
                {"sender": "user", "content": "Hello"},
                {"sender": "assistant", "content": "Summary points"},
            ]
        }
        self.notes = {
            "note_1": {"id": "note_1", "title": "Note 1", "content": "Content 1"},
            "note_2": {"id": "note_2", "title": "Note 2", "content": "Content 2"},
        }

    def get_conversation_by_id(self, conversation_id: str):
        return self.conversations.get(conversation_id)

    def get_messages_for_conversation(self, conversation_id: str, *args, **kwargs):
        return self.messages.get(conversation_id, [])

    def get_note_by_id(self, note_id: str):
        return self.notes.get(note_id)


class FakeMediaDB:
    def __init__(self) -> None:
        self.media = {1: {"id": 1, "title": "Media"}}

    def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
        return self.media.get(media_id)


class FakeCollectionsDB:
    def get_output_artifact(self, output_id: int):
        return {"id": output_id}

    def resolve_output_storage_path(self, path_value):
        return str(path_value)


@pytest.fixture()
def slides_client(tmp_path):
    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])

    async def _override_user():
        return User(id=1, username="tester", email=None, is_active=True, is_admin=True)

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - auth principal stub token type, not a credential
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    async def _override_db():
        db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
        try:
            yield db
        finally:
            db.close_connection()

    async def _override_collections_db():
        return FakeCollectionsDB()

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db
    app.dependency_overrides[get_collections_db_for_user] = _override_collections_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def slides_client_with_sources(tmp_path):
    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])
    fake_notes = FakeNotesDB()
    fake_media = FakeMediaDB()

    async def _override_user():
        return User(id=1, username="tester", email=None, is_active=True, is_admin=True)

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - auth principal stub token type, not a credential
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    async def _override_db():
        db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
        try:
            yield db
        finally:
            db.close_connection()

    async def _override_notes_db():
        return fake_notes

    async def _override_media_db():
        return fake_media

    async def _override_collections_db():
        return FakeCollectionsDB()

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db
    app.dependency_overrides[get_chacha_db_for_user] = _override_notes_db
    app.dependency_overrides[get_media_db_for_user] = _override_media_db
    app.dependency_overrides[get_collections_db_for_user] = _override_collections_db

    with TestClient(app) as client:
        yield client, fake_notes, fake_media

    app.dependency_overrides.clear()


def _build_llm_stub(title: str):
    payload = {
        "title": title,
        "slides": [
            {
                "order": 0,
                "layout": "title",
                "title": title,
                "content": "",
                "speaker_notes": None,
                "metadata": {},
            }
        ],
    }

    def _stub(api_provider=None, messages=None, **kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    return _stub


def _setup_chat_too_large(monkeypatch, fake_notes, fake_media):
    if fake_notes is None:
        return
    fake_notes.messages["conv_1"] = [{"sender": "user", "content": "x" * 50}]


def _setup_notes_too_large(monkeypatch, fake_notes, fake_media):
    if fake_notes is None:
        return
    fake_notes.notes["note_1"]["content"] = "x" * 50


def _setup_media_too_large(monkeypatch, fake_notes, fake_media):
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.get_latest_transcription",
        lambda _db, _media_id: "x" * 50,
    )


def _setup_rag_too_large(monkeypatch, fake_notes, fake_media):
    async def stub_rag_pipeline(*args, **kwargs):
        doc = SimpleNamespace(content="x" * 50, metadata={"title": "Doc"})
        return SimpleNamespace(documents=[doc], generated_answer=None)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.unified_rag_pipeline",
        stub_rag_pipeline,
    )


def _assert_compact_builtin_snapshot(snapshot: dict) -> None:
    assert snapshot["id"] == "notebooklm-blueprint"
    assert snapshot["scope"] == "builtin"
    assert snapshot["resolution"]["base_theme"] == "night"
    assert snapshot["resolution"]["resolved_theme"] == "night"
    assert snapshot["resolution"]["style_pack"] == "technical_grid"
    assert snapshot["resolution"]["style_pack_version"] == 1
    assert "custom_css" not in snapshot
    assert "custom_css" not in snapshot["resolution"]


def _create_presentation(slides_client: TestClient):
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
            {"order": 1, "layout": "content", "title": "Slide", "content": "- A\n- B", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    response = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert response.status_code == 201, response.text
    return response


def _presentation_update_payload() -> dict:
    return {
        "title": "Deck Updated",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {
                "order": 0,
                "layout": "title",
                "title": "Deck Updated",
                "content": "",
                "speaker_notes": None,
                "metadata": {},
            },
            {
                "order": 1,
                "layout": "content",
                "title": "Slide",
                "content": "- A\n- B",
                "speaker_notes": None,
                "metadata": {},
            },
        ],
        "custom_css": None,
    }


def _assert_precondition_failed(response) -> None:
    assert response.status_code == 412
    assert response.json()["detail"] == "precondition_failed"


def test_resolve_template_sanitizes_invalid_template_error(monkeypatch):
    def _raise_invalid(template_id):
        _ = template_id
        raise SlidesTemplateInvalidError("template parse exploded")

    monkeypatch.setattr(slides_ep, "get_slide_template", _raise_invalid)

    with pytest.raises(HTTPException) as exc_info:
        slides_ep._resolve_template("broken-template")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to resolve slide template"


def test_generate_presentation_sanitizes_generation_error(monkeypatch):
    class _RaisingGenerator:
        def generate_from_text(self, **kwargs):
            _ = kwargs
            raise SlidesGenerationError("llm backend exploded")

    monkeypatch.setattr(slides_ep, "_resolve_provider", lambda provider: "stub-provider")
    monkeypatch.setattr(slides_ep, "SlidesGenerator", _RaisingGenerator)
    monkeypatch.setattr(slides_ep, "get_metrics_registry", lambda: None)

    request = SimpleNamespace(
        visual_style_id=None,
        visual_style_scope=None,
        template_id=None,
        theme="black",
        marp_theme=None,
        settings={"controls": True},
        provider="stub-provider",
        model=None,
        title_hint="Deck",
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=True,
        chunk_size_tokens=None,
        summary_tokens=None,
        custom_css=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        slides_ep._generate_presentation(
            response=Response(),
            db=SimpleNamespace(client_id="1"),
            request=request,
            source_text="Source text",
            source_type="prompt",
            source_ref=None,
            source_query=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate presentation"


_TOO_LARGE_CASES = [
    {
        "id": "prompt",
        "client": "slides_client",
        "path": "/api/v1/slides/generate",
        "payload": {
            "prompt": "x" * 50,
            "title_hint": "Big",
            "theme": "black",
            "max_source_chars": 10,
            "enable_chunking": False,
        },
        "setup": None,
    },
    {
        "id": "chat",
        "client": "slides_client_with_sources",
        "path": "/api/v1/slides/generate/from-chat",
        "payload": {
            "conversation_id": "conv_1",
            "title_hint": "Big Chat",
            "theme": "black",
            "max_source_chars": 10,
            "enable_chunking": False,
        },
        "setup": _setup_chat_too_large,
    },
    {
        "id": "notes",
        "client": "slides_client_with_sources",
        "path": "/api/v1/slides/generate/from-notes",
        "payload": {
            "note_ids": ["note_1"],
            "title_hint": "Big Notes",
            "theme": "black",
            "max_source_chars": 10,
            "enable_chunking": False,
        },
        "setup": _setup_notes_too_large,
    },
    {
        "id": "media",
        "client": "slides_client_with_sources",
        "path": "/api/v1/slides/generate/from-media",
        "payload": {
            "media_id": 1,
            "title_hint": "Big Media",
            "theme": "black",
            "max_source_chars": 10,
            "enable_chunking": False,
        },
        "setup": _setup_media_too_large,
    },
    {
        "id": "rag",
        "client": "slides_client",
        "path": "/api/v1/slides/generate/from-rag",
        "payload": {
            "query": "Big query",
            "title_hint": "Big RAG",
            "theme": "black",
            "max_source_chars": 10,
            "enable_chunking": False,
        },
        "setup": _setup_rag_too_large,
    },
]


def test_slides_create_and_export_json(slides_client):
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "studio_data": {
            "origin": "blank",
            "default_voice": {"provider": "openai", "voice": "alloy"},
        },
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
            {"order": 1, "layout": "content", "title": "Slide", "content": "- A\n- B", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Deck"
    assert data["studio_data"] == payload["studio_data"]
    presentation_id = data["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=json")
    assert export_resp.status_code == 200
    exported = export_resp.json()
    assert exported["id"] == presentation_id
    assert exported["studio_data"] == payload["studio_data"]


def test_slides_create_canonicalizes_presentation_studio_slide_metadata(slides_client):
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "studio_data": {
            "origin": "blank",
        },
        "slides": [
            {
                "order": 0,
                "layout": "title",
                "title": "Deck",
                "content": "",
                "speaker_notes": "",
                "metadata": {
                    "studio": {
                        "slideId": "slide-1",
                        "audio": {"status": "missing"},
                        "image": {"status": "missing"},
                    }
                },
            }
        ],
        "custom_css": None,
    }

    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    studio = data["slides"][0]["metadata"]["studio"]
    assert studio["slideId"] == "slide-1"
    assert studio["transition"] == "fade"
    assert studio["timing_mode"] == "auto"
    assert studio["manual_duration_ms"] is None
    assert studio["audio"]["status"] == "missing"
    assert studio["image"]["status"] == "missing"

    export_resp = slides_client.get(f"/api/v1/slides/presentations/{data['id']}/export?format=json")
    assert export_resp.status_code == 200
    exported = export_resp.json()
    exported_studio = exported["slides"][0]["metadata"]["studio"]
    assert exported_studio["transition"] == "fade"
    assert exported_studio["timing_mode"] == "auto"
    assert exported_studio["manual_duration_ms"] is None


def test_slides_create_rejects_invalid_image(slides_client):
    payload = {
        "title": "Deck",
        "theme": "black",
        "slides": [
            {
                "order": 0,
                "layout": "content",
                "title": "Slide",
                "content": "",
                "speaker_notes": None,
                "metadata": {"images": [{"mime": "image/png", "data_b64": "not-base64"}]},
            }
        ],
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "image_data_b64_invalid"


def test_slides_create_and_export_markdown_with_image_asset_ref(slides_client, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.resolve_slide_asset",
        lambda asset_ref, **kwargs: {
            "asset_ref": asset_ref,
            "mime": "image/png",
            "data_b64": _SAMPLE_PNG_B64,
            "alt": "Cover",
        },
    )
    payload = {
        "title": "Deck",
        "theme": "black",
        "slides": [
            {
                "order": 0,
                "layout": "content",
                "title": "Slide",
                "content": "Hello",
                "speaker_notes": None,
                "metadata": {"images": [{"asset_ref": "output:123", "alt": "Cover"}]},
            }
        ],
    }

    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["slides"][0]["metadata"]["images"][0]["asset_ref"] == "output:123"

    export_resp = slides_client.get(f"/api/v1/slides/presentations/{data['id']}/export?format=markdown")
    assert export_resp.status_code == 200
    assert "![Cover](data:image/png;base64," in export_resp.text


def test_slides_export_offloads_blocking_markdown_and_reveal_generation(
    slides_client, monkeypatch, tmp_path
):
    assets_dir = _build_assets(tmp_path)
    monkeypatch.setenv("SLIDES_REVEALJS_ASSETS_DIR", str(assets_dir))
    offloaded_calls: list[str] = []

    async def _fake_to_thread(func, *args, **kwargs):
        offloaded_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.asyncio.to_thread", _fake_to_thread)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.resolve_slide_asset",
        lambda asset_ref, **kwargs: {
            "asset_ref": asset_ref,
            "mime": "image/png",
            "data_b64": _SAMPLE_PNG_B64,
            "alt": "Cover",
        },
    )
    payload = {
        "title": "Deck",
        "theme": "black",
        "slides": [
            {
                "order": 0,
                "layout": "content",
                "title": "Slide",
                "content": "Hello",
                "speaker_notes": "Narration",
                "metadata": {"images": [{"asset_ref": "output:123", "alt": "Cover"}]},
            }
        ],
    }

    create_resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]

    markdown_resp = slides_client.get(
        f"/api/v1/slides/presentations/{presentation_id}/export?format=markdown"
    )
    assert markdown_resp.status_code == 200, markdown_resp.text

    reveal_resp = slides_client.get(
        f"/api/v1/slides/presentations/{presentation_id}/export?format=revealjs"
    )
    assert reveal_resp.status_code == 200, reveal_resp.text

    assert "export_presentation_markdown" in offloaded_calls
    assert "export_presentation_bundle" in offloaded_calls


def test_slides_export_reveal(slides_client, tmp_path, monkeypatch):
    assets_dir = _build_assets(tmp_path)
    monkeypatch.setenv("SLIDES_REVEALJS_ASSETS_DIR", str(assets_dir))
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {
                "order": 0,
                "layout": "title",
                "title": "Deck",
                "content": "",
                "speaker_notes": None,
                "metadata": {
                    "images": [
                        {
                            "mime": "image/png",
                            "data_b64": _SAMPLE_PNG_B64,
                            "alt": "Logo",
                            "width": 16,
                            "height": 16,
                        }
                    ]
                },
            },
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=revealjs")
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(export_resp.content)) as zf:
        assert "index.html" in zf.namelist()
        index_html = zf.read("index.html").decode("utf-8")
        assert "data:image/png;base64," in index_html
        assert "alt=\"Logo\"" in index_html


def test_slides_export_reveal_passes_visual_style_snapshot(slides_client, monkeypatch):
    captured = {}

    def _stub_export(**kwargs):
        captured["visual_style_snapshot"] = kwargs.get("visual_style_snapshot")
        return b"PK\x03\x04stub"

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_bundle",
        _stub_export,
    )
    payload = {
        "title": "Styled Deck",
        "description": None,
        "theme": "black",
        "visual_style_id": "notebooklm-blueprint",
        "visual_style_scope": "builtin",
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201, resp.text

    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(
        f"/api/v1/slides/presentations/{presentation_id}/export?format=revealjs"
    )

    assert export_resp.status_code == 200, export_resp.text
    assert captured["visual_style_snapshot"]["id"] == "notebooklm-blueprint"
    assert captured["visual_style_snapshot"]["resolution"]["style_pack"] == "technical_grid"


def test_slides_export_markdown_marp_override(slides_client):
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "marp_theme": "gaia",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=markdown")
    assert export_resp.status_code == 200
    assert "theme: gaia" in export_resp.text


def test_slides_export_pdf(slides_client, monkeypatch):
    captured = {}

    def _stub_export(**kwargs):
        captured["options"] = kwargs.get("pdf_options")
        return b"%PDF-1.4\n%stub"

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_pdf", _stub_export)
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(
        f"/api/v1/slides/presentations/{presentation_id}/export?format=pdf&pdf_format=Letter&pdf_landscape=true&pdf_margin_top=0.2in"
    )
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"].startswith("application/pdf")
    assert export_resp.content.startswith(b"%PDF")
    options = captured.get("options") or {}
    assert options.get("format") == "Letter"
    assert options.get("landscape") is True
    assert (options.get("margin") or {}).get("top") == "0.2in"


def test_slides_export_pdf_passes_visual_style_snapshot(slides_client, monkeypatch):
    captured = {}

    def _stub_export(**kwargs):
        captured["visual_style_snapshot"] = kwargs.get("visual_style_snapshot")
        return b"%PDF-1.4\n%stub"

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_pdf", _stub_export)
    payload = {
        "title": "Styled Deck",
        "description": None,
        "theme": "black",
        "visual_style_id": "notebooklm-blueprint",
        "visual_style_scope": "builtin",
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201, resp.text

    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(
        f"/api/v1/slides/presentations/{presentation_id}/export?format=pdf"
    )

    assert export_resp.status_code == 200, export_resp.text
    assert captured["visual_style_snapshot"]["id"] == "notebooklm-blueprint"
    assert captured["visual_style_snapshot"]["resolution"]["style_pack"] == "technical_grid"


def test_slides_export_pdf_input_error(slides_client, monkeypatch):
    def _stub_export(**kwargs):
        raise SlidesExportInputError("pdf_format_invalid")

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_pdf", _stub_export)
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=pdf")
    assert export_resp.status_code == 422
    assert export_resp.json()["detail"] == "pdf_format_invalid"


def test_slides_export_pdf_failure(slides_client, monkeypatch):
    def _stub_export(**kwargs):
        raise SlidesExportError("pdf_render_failed")

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_pdf", _stub_export)
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=pdf")
    assert export_resp.status_code == 500
    assert export_resp.json()["detail"] == "Failed to export presentation as pdf"


def test_slides_export_markdown_failure(slides_client, monkeypatch):
    def _stub_export(**kwargs):
        raise SlidesExportError("markdown_render_failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_markdown",
        _stub_export,
    )
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=markdown")
    assert export_resp.status_code == 500
    assert export_resp.json()["detail"] == "Failed to export presentation as markdown"


def test_slides_export_reveal_failure(slides_client, monkeypatch):
    def _stub_export(**kwargs):
        raise SlidesExportError("reveal_bundle_failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.export_presentation_bundle",
        _stub_export,
    )
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    presentation_id = resp.json()["id"]
    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=revealjs")
    assert export_resp.status_code == 500
    assert export_resp.json()["detail"] == "Failed to export presentation as revealjs"


def test_slides_templates_list_and_get(slides_client, tmp_path, monkeypatch):
    templates_path = _write_templates(tmp_path)
    monkeypatch.setenv("SLIDES_TEMPLATES_PATH", str(templates_path))
    list_resp = slides_client.get("/api/v1/slides/templates")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["templates"][0]["id"] == "template-1"

    get_resp = slides_client.get("/api/v1/slides/templates/template-1")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Template One"


def test_slides_styles_list_returns_builtin_and_user_styles(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True, "bullet_bias": "high"},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    created_id = create_resp.json()["id"]

    list_resp = slides_client.get("/api/v1/slides/styles")
    assert list_resp.status_code == 200
    payload = list_resp.json()
    styles = payload["styles"]

    builtin = next(item for item in styles if item["id"] == "notebooklm-blueprint")
    assert builtin["scope"] == "builtin"
    assert builtin["category"] == "technical"
    assert builtin["guide_number"] == 7
    assert builtin["tags"] == ["technical", "technical_grid", "technical_precision"]
    assert builtin["best_for"] == ["systems explanation", "architecture walkthrough"]
    assert builtin["appearance_defaults"]["theme"] == "night"
    assert "custom_css" not in builtin["appearance_defaults"]
    assert any(item["scope"] == "user" and item["id"] == created_id for item in styles)
    assert payload["total_count"] >= len(styles)
    assert payload["limit"] == 50
    assert payload["offset"] == 0
    assert payload["pagination"]["mode"] == "offset"
    assert payload["pagination"]["limit"] == 50
    assert payload["pagination"]["offset"] == 0
    assert payload["pagination"]["total"] == payload["total_count"]
    assert payload["pagination"]["has_more"] is False
    assert payload["pagination"]["next_offset"] is None
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


def test_slides_builtin_style_detail_exposes_catalog_metadata_and_compact_defaults(slides_client):
    resp = slides_client.get("/api/v1/slides/styles/notebooklm-blueprint")
    assert resp.status_code == 200, resp.text
    style = resp.json()
    assert style["scope"] == "builtin"
    assert style["category"] == "technical"
    assert style["guide_number"] == 7
    assert style["tags"] == ["technical", "technical_grid", "technical_precision"]
    assert style["best_for"] == ["systems explanation", "architecture walkthrough"]
    assert style["appearance_defaults"]["theme"] == "night"
    assert "custom_css" not in style["appearance_defaults"]


def test_slides_builtin_style_list_skips_custom_css_rendering(slides_client, monkeypatch):
    from tldw_Server_API.app.core.Slides.visual_style_resolver import (
        resolve_builtin_visual_style as real_resolve_builtin_visual_style,
    )

    captured: list[bool] = []

    def _recorded_resolve(style_id: str, *, include_custom_css: bool):
        captured.append(include_custom_css)
        return real_resolve_builtin_visual_style(style_id, include_custom_css=include_custom_css)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.resolve_builtin_visual_style",
        _recorded_resolve,
    )

    resp = slides_client.get("/api/v1/slides/styles")

    assert resp.status_code == 200, resp.text
    assert captured
    assert set(captured) == {False}


def test_slides_builtin_style_detail_skips_custom_css_rendering(slides_client, monkeypatch):
    from tldw_Server_API.app.core.Slides.visual_style_resolver import (
        resolve_builtin_visual_style as real_resolve_builtin_visual_style,
    )

    captured: list[bool] = []

    def _recorded_resolve(style_id: str, *, include_custom_css: bool):
        captured.append(include_custom_css)
        return real_resolve_builtin_visual_style(style_id, include_custom_css=include_custom_css)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.resolve_builtin_visual_style",
        _recorded_resolve,
    )

    resp = slides_client.get("/api/v1/slides/styles/notebooklm-blueprint")

    assert resp.status_code == 200, resp.text
    assert captured == [False]


def test_slides_styles_list_supports_pagination(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "User Style",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_resp.status_code == 201, create_resp.text

    builtin_count = len(list_builtin_visual_styles())
    list_resp = slides_client.get(f"/api/v1/slides/styles?limit=1&offset={builtin_count}")
    assert list_resp.status_code == 200
    payload = list_resp.json()

    assert payload["limit"] == 1
    assert payload["offset"] == builtin_count
    assert len(payload["styles"]) == 1
    assert payload["styles"][0]["scope"] == "user"
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": builtin_count,
        "total": payload["total_count"],
        "has_more": False,
        "next_offset": None,
    }
    assert payload["has_more"] is False
    assert payload["next_offset"] is None


def test_slides_presentations_list_and_search_include_pagination_metadata(slides_client):
    first_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={"title": "Alpha Deck", "slides": []},
    )
    assert first_resp.status_code == 201, first_resp.text

    second_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={"title": "Beta Deck", "slides": []},
    )
    assert second_resp.status_code == 201, second_resp.text

    list_resp = slides_client.get("/api/v1/slides/presentations?limit=1&offset=0")
    assert list_resp.status_code == 200, list_resp.text
    list_payload = list_resp.json()
    assert len(list_payload["presentations"]) == 1
    assert list_payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert list_payload["has_more"] is True
    assert list_payload["next_offset"] == 1

    search_resp = slides_client.get("/api/v1/slides/presentations/search?q=Deck&limit=1&offset=1")
    assert search_resp.status_code == 200, search_resp.text
    search_payload = search_resp.json()
    assert len(search_payload["presentations"]) == 1
    assert search_payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 1,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert search_payload["has_more"] is False
    assert search_payload["next_offset"] is None


def test_slides_styles_crud_for_user_styles(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True, "bullet_bias": "high"},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    created = create_resp.json()
    style_id = created["id"]
    assert created["scope"] == "user"
    assert created["generation_rules"]["exam_focus"] is True

    get_resp = slides_client.get(f"/api/v1/slides/styles/{style_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Exam Sprint"

    patch_resp = slides_client.patch(
        f"/api/v1/slides/styles/{style_id}",
        json={
            "name": "Exam Sprint Updated",
            "generation_rules": {"exam_focus": True, "bullet_bias": "medium"},
            "artifact_preferences": ["comparison_matrix"],
        },
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["name"] == "Exam Sprint Updated"
    assert patched["generation_rules"]["bullet_bias"] == "medium"
    assert patched["artifact_preferences"] == ["comparison_matrix"]

    delete_resp = slides_client.delete(f"/api/v1/slides/styles/{style_id}")
    assert delete_resp.status_code == 204

    missing_resp = slides_client.get(f"/api/v1/slides/styles/{style_id}")
    assert missing_resp.status_code == 404
    assert missing_resp.json()["detail"] == "visual_style_not_found"


def test_slides_styles_patch_can_clear_description(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    style_id = create_resp.json()["id"]

    patch_resp = slides_client.patch(
        f"/api/v1/slides/styles/{style_id}",
        json={"description": None},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    assert patched["description"] is None
    assert patched["generation_rules"] == {"exam_focus": True}


def test_slides_styles_reject_builtin_mutation(slides_client):
    patch_resp = slides_client.patch(
        "/api/v1/slides/styles/timeline",
        json={"name": "Rewritten Timeline"},
    )
    assert patch_resp.status_code == 403
    assert patch_resp.json()["detail"] == "builtin_visual_style_read_only"


def test_slides_styles_reject_non_string_custom_css(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Broken CSS Style",
            "description": "Should fail validation",
            "generation_rules": {},
            "artifact_preferences": [],
            "appearance_defaults": {
                "theme": "white",
                "custom_css": {"selector": ".reveal"},
            },
            "fallback_policy": {"mode": "outline"},
        },
    )
    assert create_resp.status_code == 422
    assert create_resp.json()["detail"] == "invalid_visual_style_custom_css"


def test_slides_create_rejects_resolved_style_with_non_string_custom_css(slides_client, monkeypatch):
    def _broken_style(_self, style_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=style_id,
            scope="user",
            name="Broken CSS Style",
            style_payload=json.dumps(
                {
                    "description": "Broken style payload",
                    "generation_rules": {},
                    "artifact_preferences": [],
                    "appearance_defaults": {
                        "theme": "white",
                        "custom_css": {"selector": ".reveal"},
                    },
                    "fallback_policy": {"mode": "outline"},
                }
            ),
            created_at="2026-03-17T00:00:00Z",
            updated_at="2026-03-17T00:00:00Z",
        )

    monkeypatch.setattr(SlidesDatabase, "get_visual_style_by_id", _broken_style)

    resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Broken Style Deck",
            "visual_style_id": "broken-css-style",
            "visual_style_scope": "user",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "Broken",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_visual_style_custom_css"


def test_slides_styles_patch_accepts_null_theme_in_appearance_defaults(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    style_id = create_resp.json()["id"]

    patch_resp = slides_client.patch(
        f"/api/v1/slides/styles/{style_id}",
        json={"appearance_defaults": {"theme": None}},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    assert patch_resp.json()["appearance_defaults"]["theme"] is None


def test_slides_styles_patch_maps_conflict_error(slides_client, monkeypatch):
    create_style_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_style_resp.status_code == 201, create_style_resp.text
    style_id = create_style_resp.json()["id"]

    def _raise_conflict_error(
        self,
        *,
        style_id: str,
        name: str,
        style_payload: str,
        expected_updated_at: str,
    ):
        raise SlidesConflictError("visual style update conflicted with a newer revision")

    monkeypatch.setattr(SlidesDatabase, "update_visual_style", _raise_conflict_error)

    patch_resp = slides_client.patch(
        f"/api/v1/slides/styles/{style_id}",
        json={"description": "Updated"},
    )

    assert patch_resp.status_code == 409
    assert patch_resp.json()["detail"] == "visual_style_version_conflict"


def test_slides_styles_delete_rejects_styles_in_use(slides_client):
    create_style_resp = slides_client.post(
        "/api/v1/slides/styles",
        json={
            "name": "Exam Sprint",
            "description": "Recall-first deck",
            "generation_rules": {"exam_focus": True},
            "artifact_preferences": ["stat_group"],
            "appearance_defaults": {"theme": "white"},
            "fallback_policy": {"mode": "key-points"},
        },
    )
    assert create_style_resp.status_code == 201, create_style_resp.text
    style_id = create_style_resp.json()["id"]

    create_presentation_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Exam Deck",
            "visual_style_id": style_id,
            "visual_style_scope": "user",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "Exam Deck",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert create_presentation_resp.status_code == 201, create_presentation_resp.text

    delete_resp = slides_client.delete(f"/api/v1/slides/styles/{style_id}")
    assert delete_resp.status_code == 409
    assert delete_resp.json()["detail"] == "visual_style_in_use"


def test_slides_create_with_template_defaults(slides_client, tmp_path, monkeypatch):
    templates_path = _write_templates(tmp_path)
    monkeypatch.setenv("SLIDES_TEMPLATES_PATH", str(templates_path))
    resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Deck",
            "template_id": "template-1",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["theme"] == "white"
    assert data["marp_theme"] == "gaia"
    assert data["template_id"] == "template-1"
    assert data["custom_css"] == ".reveal { font-size: 36px; }"
    assert data["slides"][0]["title"] == "Template Title"


def test_slides_create_persists_visual_style_snapshot(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    payload = resp.json()
    assert payload["visual_style_id"] == "notebooklm-blueprint"
    assert payload["visual_style_scope"] == "builtin"
    assert payload["visual_style_name"] == "Blueprint"
    assert payload["visual_style_version"] == 1
    assert payload["theme"] == "night"
    assert payload["settings"] == {"controls": True, "progress": True}
    assert payload["custom_css"]
    _assert_compact_builtin_snapshot(payload["visual_style_snapshot"])
    assert payload["visual_style_snapshot"]["resolution"]["resolved_settings"] == payload["settings"]


def test_slides_create_with_builtin_style_preserves_explicit_appearance_overrides(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "theme": "white",
            "marp_theme": "gaia",
            "settings": {"controls": False, "progress": False},
            "custom_css": ".reveal { font-size: 42px; }",
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert resp.status_code == 201, resp.text
    payload = resp.json()
    assert payload["theme"] == "white"
    assert payload["marp_theme"] == "gaia"
    assert payload["settings"] == {"controls": False, "progress": False}
    assert payload["custom_css"] == ".reveal { font-size: 42px; }"
    _assert_compact_builtin_snapshot(payload["visual_style_snapshot"])
    assert payload["visual_style_snapshot"]["resolution"]["resolved_theme"] == "night"
    assert payload["visual_style_snapshot"]["resolution"]["resolved_settings"] == {
        "controls": True,
        "progress": True,
    }


def test_slides_create_with_builtin_style_rejects_invalid_theme(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "theme": "ultraviolet",
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid_theme"


def test_slides_put_applies_builtin_visual_style_atomically(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]

    put_resp = slides_client.put(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "title": "History Deck Updated",
            "description": None,
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
        headers={"If-Match": create_resp.headers["ETag"]},
    )
    assert put_resp.status_code == 200, put_resp.text
    payload = put_resp.json()
    assert payload["title"] == "History Deck Updated"
    assert payload["visual_style_id"] == "notebooklm-blueprint"
    assert payload["theme"] == "night"
    assert payload["settings"] == {"controls": True, "progress": True}
    assert payload["custom_css"]
    _assert_compact_builtin_snapshot(payload["visual_style_snapshot"])


def test_slides_patch_updates_visual_style_snapshot(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]
    patch_resp = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "theme": "white",
            "marp_theme": "gaia",
            "settings": {"controls": False, "progress": False},
            "custom_css": ".reveal { font-size: 42px; }",
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
        },
        headers={"If-Match": create_resp.headers["ETag"]},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    payload = patch_resp.json()
    assert payload["visual_style_id"] == "notebooklm-blueprint"
    assert payload["visual_style_scope"] == "builtin"
    assert payload["theme"] == "white"
    assert payload["marp_theme"] == "gaia"
    assert payload["settings"] == {"controls": False, "progress": False}
    assert payload["custom_css"] == ".reveal { font-size: 42px; }"
    _assert_compact_builtin_snapshot(payload["visual_style_snapshot"])
    assert payload["visual_style_snapshot"]["resolution"]["resolved_theme"] == "night"
    assert payload["visual_style_snapshot"]["resolution"]["resolved_settings"] == {
        "controls": True,
        "progress": True,
    }


def test_slides_patch_with_builtin_style_preserves_explicit_null_appearance_overrides(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]
    patch_resp = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
            "marp_theme": None,
            "settings": None,
            "custom_css": None,
        },
        headers={"If-Match": create_resp.headers["ETag"]},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    payload = patch_resp.json()
    assert payload["visual_style_id"] == "notebooklm-blueprint"
    assert payload["visual_style_scope"] == "builtin"
    assert payload["theme"] == "night"
    assert payload["marp_theme"] is None
    assert payload["settings"] is None
    assert payload["custom_css"] is None
    _assert_compact_builtin_snapshot(payload["visual_style_snapshot"])
    assert payload["visual_style_snapshot"]["resolution"]["resolved_theme"] == "night"
    assert payload["visual_style_snapshot"]["resolution"]["resolved_settings"] == {
        "controls": True,
        "progress": True,
    }


def test_slides_patch_rejects_explicit_null_theme(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "History Deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "History",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]

    patch_resp = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "theme": None,
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
        },
        headers={"If-Match": create_resp.headers["ETag"]},
    )

    assert patch_resp.status_code == 422
    assert patch_resp.json()["detail"] == "invalid_theme"


def test_slides_generate_with_template_defaults(slides_client, tmp_path, monkeypatch):
    templates_path = _write_templates(tmp_path)
    monkeypatch.setenv("SLIDES_TEMPLATES_PATH", str(templates_path))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Generated Deck"),
    )
    resp = slides_client.post(
        "/api/v1/slides/generate",
        json={
            "title_hint": "Generated Deck",
            "prompt": "Summarize key findings.",
            "template_id": "template-1",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme"] == "white"
    assert data["marp_theme"] == "gaia"
    assert data["template_id"] == "template-1"
    assert data["custom_css"] == ".reveal { font-size: 36px; }"


def test_slides_generate_with_visual_style_snapshot(slides_client, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Generated History Deck"),
    )
    resp = slides_client.post(
        "/api/v1/slides/generate",
        json={
            "title_hint": "Generated History Deck",
            "prompt": "Summarize key history milestones.",
            "visual_style_id": "notebooklm-blueprint",
            "visual_style_scope": "builtin",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["visual_style_id"] == "notebooklm-blueprint"
    assert data["visual_style_scope"] == "builtin"
    assert data["theme"] == "night"
    assert data["settings"] == {"controls": True, "progress": True}
    assert data["custom_css"]
    _assert_compact_builtin_snapshot(data["visual_style_snapshot"])


def test_slides_reorder(slides_client):
    resp = _create_presentation(slides_client)
    presentation_id = resp.json()["id"]
    etag = resp.headers["ETag"]
    reorder_resp = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/reorder",
        json={"order": [1, 0]},
        headers={"If-Match": etag},
    )
    assert reorder_resp.status_code == 200
    reordered = reorder_resp.json()
    assert reordered["slides"][0]["title"] == "Slide"


def test_slides_patch_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_input_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesInputError("patch_invalid")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_input_error)

    response = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={"title": "Updated"},
        headers={"If-Match": etag},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "patch_invalid"


def test_slides_reorder_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_input_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesInputError("reorder_invalid")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_input_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/reorder",
        json={"order": [1, 0]},
        headers={"If-Match": etag},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "reorder_invalid"


def test_slides_delete_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_input_error(self, presentation_id: str, expected_version: int | None):
        raise SlidesInputError("delete_invalid")

    monkeypatch.setattr(SlidesDatabase, "soft_delete_presentation", _raise_input_error)

    response = slides_client.delete(
        f"/api/v1/slides/presentations/{presentation_id}",
        headers={"If-Match": etag},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "delete_invalid"


def test_slides_restore_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]
    delete_resp = slides_client.delete(
        f"/api/v1/slides/presentations/{presentation_id}",
        headers={"If-Match": etag},
    )
    assert delete_resp.status_code == 200

    def _raise_input_error(self, presentation_id: str, expected_version: int | None):
        raise SlidesInputError("restore_invalid")

    monkeypatch.setattr(SlidesDatabase, "restore_presentation", _raise_input_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/restore",
        headers={"If-Match": delete_resp.headers["ETag"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "restore_invalid"


def test_slides_put_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_input_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesInputError("put_invalid")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_input_error)

    response = slides_client.put(
        f"/api/v1/slides/presentations/{presentation_id}",
        json=_presentation_update_payload(),
        headers={"If-Match": etag},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "put_invalid"


def test_slides_put_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_conflict_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_conflict_error)

    response = slides_client.put(
        f"/api/v1/slides/presentations/{presentation_id}",
        json=_presentation_update_payload(),
        headers={"If-Match": etag},
    )

    _assert_precondition_failed(response)


def test_slides_patch_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_conflict_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_conflict_error)

    response = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={"title": "Updated"},
        headers={"If-Match": etag},
    )

    _assert_precondition_failed(response)


def test_slides_reorder_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_conflict_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_conflict_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/reorder",
        json={"order": [1, 0]},
        headers={"If-Match": etag},
    )

    _assert_precondition_failed(response)


def test_slides_delete_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    def _raise_conflict_error(self, presentation_id: str, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "soft_delete_presentation", _raise_conflict_error)

    response = slides_client.delete(
        f"/api/v1/slides/presentations/{presentation_id}",
        headers={"If-Match": etag},
    )

    _assert_precondition_failed(response)


def test_slides_restore_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]
    delete_resp = slides_client.delete(
        f"/api/v1/slides/presentations/{presentation_id}",
        headers={"If-Match": etag},
    )
    assert delete_resp.status_code == 200

    def _raise_conflict_error(self, presentation_id: str, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "restore_presentation", _raise_conflict_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/restore",
        headers={"If-Match": delete_resp.headers["ETag"]},
    )

    _assert_precondition_failed(response)


def test_slides_restore_version_maps_conflict_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]

    def _raise_conflict_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesConflictError("version_conflict")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_conflict_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": create_resp.headers["ETag"]},
    )

    _assert_precondition_failed(response)


def test_slides_restore_version_maps_input_error(slides_client, monkeypatch):
    create_resp = _create_presentation(slides_client)
    presentation_id = create_resp.json()["id"]

    def _raise_input_error(self, presentation_id: str, update_fields: dict, expected_version: int | None):
        raise SlidesInputError("restore_version_invalid")

    monkeypatch.setattr(SlidesDatabase, "update_presentation", _raise_input_error)

    response = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": create_resp.headers["ETag"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "restore_version_invalid"


def test_slides_versions_and_restore(slides_client):
    payload = {
        "title": "Deck",
        "description": None,
        "theme": "black",
        "settings": {"controls": True},
        "studio_data": {
            "origin": "blank",
            "default_voice": {"provider": "openai", "voice": "alloy"},
        },
        "slides": [
            {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        ],
        "custom_css": None,
    }
    resp = slides_client.post("/api/v1/slides/presentations", json=payload)
    assert resp.status_code == 201
    presentation_id = resp.json()["id"]
    etag = resp.headers["ETag"]

    update_resp = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "title": "Updated",
            "studio_data": {
                "origin": "extension_capture",
                "default_voice": {"provider": "openai", "voice": "verse"},
            },
        },
        headers={"If-Match": etag},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["studio_data"]["origin"] == "extension_capture"
    new_etag = update_resp.headers["ETag"]

    versions_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/versions")
    assert versions_resp.status_code == 200
    versions_data = versions_resp.json()
    assert versions_data["total"] == 2
    assert versions_data["versions"][0]["version"] == 2
    assert versions_data["pagination"] == {
        "mode": "offset",
        "limit": 50,
        "offset": 0,
        "total": 2,
        "has_more": False,
        "next_offset": None,
    }
    assert versions_data["has_more"] is False
    assert versions_data["next_offset"] is None

    version_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/versions/1")
    assert version_resp.status_code == 200
    assert version_resp.json()["title"] == "Deck"
    assert version_resp.json()["studio_data"] == payload["studio_data"]

    restore_resp = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": new_etag},
    )
    assert restore_resp.status_code == 200
    assert restore_resp.json()["title"] == "Deck"
    assert restore_resp.json()["studio_data"] == payload["studio_data"]


def test_slides_patch_persists_presentation_studio_timing_and_transition_metadata(slides_client):
    create_resp = slides_client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Deck",
            "description": None,
            "theme": "black",
            "studio_data": {"origin": "blank"},
            "slides": [
                {
                    "order": 0,
                    "layout": "content",
                    "title": "Slide",
                    "content": "Body",
                    "speaker_notes": "Narration",
                    "metadata": {
                        "studio": {
                            "slideId": "slide-1",
                            "audio": {"status": "ready", "duration_ms": 12000},
                            "image": {"status": "ready"},
                        }
                    },
                }
            ],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    presentation_id = create_resp.json()["id"]
    etag = create_resp.headers["ETag"]

    patch_resp = slides_client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={
            "slides": [
                {
                    "order": 0,
                    "layout": "content",
                    "title": "Slide",
                    "content": "Body",
                    "speaker_notes": "Narration",
                    "metadata": {
                        "studio": {
                            "slideId": "slide-1",
                            "transition": "wipe",
                            "timing_mode": "manual",
                            "manual_duration_ms": 45000,
                            "audio": {"status": "ready", "duration_ms": 12000},
                            "image": {"status": "ready"},
                        }
                    },
                }
            ]
        },
        headers={"If-Match": etag},
    )
    assert patch_resp.status_code == 200, patch_resp.text
    patched = patch_resp.json()
    patched_studio = patched["slides"][0]["metadata"]["studio"]
    assert patched_studio["transition"] == "wipe"
    assert patched_studio["timing_mode"] == "manual"
    assert patched_studio["manual_duration_ms"] == 45000
    new_etag = patch_resp.headers["ETag"]

    version_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/versions/1")
    assert version_resp.status_code == 200
    version_studio = version_resp.json()["slides"][0]["metadata"]["studio"]
    assert version_studio["transition"] == "fade"
    assert version_studio["timing_mode"] == "auto"
    assert version_studio["manual_duration_ms"] is None

    restore_resp = slides_client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": new_etag},
    )
    assert restore_resp.status_code == 200
    restored_studio = restore_resp.json()["slides"][0]["metadata"]["studio"]
    assert restored_studio["transition"] == "fade"
    assert restored_studio["timing_mode"] == "auto"
    assert restored_studio["manual_duration_ms"] is None

    export_resp = slides_client.get(f"/api/v1/slides/presentations/{presentation_id}/export?format=json")
    assert export_resp.status_code == 200
    exported_studio = export_resp.json()["slides"][0]["metadata"]["studio"]
    assert exported_studio["transition"] == "fade"
    assert exported_studio["timing_mode"] == "auto"
    assert exported_studio["manual_duration_ms"] is None

def test_slides_generate_from_prompt_uses_stubbed_llm(slides_client, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Generated Deck"),
    )

    resp = slides_client.post(
        "/api/v1/slides/generate",
        json={
            "title_hint": "Generated Deck",
            "prompt": "Summarize key findings.",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Generated Deck"
    assert data["source_type"] == "prompt"


def test_slides_generate_from_rag_explicit_sources(slides_client, monkeypatch):
    sources_seen = {}

    async def stub_rag_pipeline(*args, **kwargs):
        sources_seen["sources"] = kwargs.get("sources")
        doc = SimpleNamespace(content="Doc content", metadata={"title": "Doc"})
        return SimpleNamespace(documents=[doc], generated_answer=None)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.unified_rag_pipeline",
        stub_rag_pipeline,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("RAG Deck"),
    )

    resp = slides_client.post(
        "/api/v1/slides/generate/from-rag",
        json={
            "query": "Risks in roadmap",
            "title_hint": "RAG Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    assert sources_seen["sources"] == ["media_db", "notes", "chats"]


def test_slides_generate_from_chat_uses_stubbed_llm(slides_client_with_sources, monkeypatch):
    client, _, _ = slides_client_with_sources
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Chat Deck"),
    )
    resp = client.post(
        "/api/v1/slides/generate/from-chat",
        json={
            "conversation_id": "conv_1",
            "title_hint": "Chat Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Chat Deck"
    assert data["source_type"] == "chat"
    assert data["source_ref"] == "conv_1"


def test_slides_generate_from_notes_uses_stubbed_llm(slides_client_with_sources, monkeypatch):
    client, _, _ = slides_client_with_sources
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Notes Deck"),
    )
    resp = client.post(
        "/api/v1/slides/generate/from-notes",
        json={
            "note_ids": ["note_1", "note_2"],
            "title_hint": "Notes Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Notes Deck"
    assert data["source_type"] == "notes"
    assert data["source_ref"] == ["note_1", "note_2"]


def test_slides_generate_from_media_uses_stubbed_llm(slides_client_with_sources, monkeypatch):
    client, _, _ = slides_client_with_sources
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.get_latest_transcription",
        lambda _db, _media_id: "Transcript content",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Media Deck"),
    )
    resp = client.post(
        "/api/v1/slides/generate/from-media",
        json={
            "media_id": 1,
            "title_hint": "Media Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Media Deck"
    assert data["source_type"] == "media"
    assert data["source_ref"] == 1


def test_slides_generate_from_document_media_uses_document_content(
    slides_client_with_sources, monkeypatch
):
    client, _, fake_media = slides_client_with_sources
    fake_media.media[1] = {"id": 1, "title": "Document Media", "type": "document"}
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.get_latest_transcription",
        lambda _db, _media_id: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.get_document_version",
        lambda db_instance, media_id, version_number=None, include_content=True: {
            "content": "Document body content for slide generation."
        },
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.perform_chat_api_call",
        _build_llm_stub("Document Deck"),
    )
    resp = client.post(
        "/api/v1/slides/generate/from-media",
        json={
            "media_id": 1,
            "title_hint": "Document Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Document Deck"
    assert data["source_type"] == "media"
    assert data["source_ref"] == 1


def test_slides_generate_from_chat_missing_conversation(slides_client_with_sources):
    client, fake_notes, _ = slides_client_with_sources
    fake_notes.conversations.pop("conv_1", None)
    resp = client.post(
        "/api/v1/slides/generate/from-chat",
        json={
            "conversation_id": "conv_1",
            "title_hint": "Chat Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "conversation_not_found"


def test_slides_generate_from_notes_missing_note(slides_client_with_sources):
    client, fake_notes, _ = slides_client_with_sources
    fake_notes.notes.pop("note_2", None)
    resp = client.post(
        "/api/v1/slides/generate/from-notes",
        json={
            "note_ids": ["note_1", "note_2"],
            "title_hint": "Notes Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["missing_note_ids"] == ["note_2"]


def test_slides_generate_from_media_missing_media(slides_client_with_sources):
    client, _, fake_media = slides_client_with_sources
    fake_media.media.pop(1, None)
    resp = client.post(
        "/api/v1/slides/generate/from-media",
        json={
            "media_id": 1,
            "title_hint": "Media Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "media_not_found"


def test_slides_generate_from_media_missing_transcript(slides_client_with_sources, monkeypatch):
    client, _, _ = slides_client_with_sources
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.slides.get_latest_transcription",
        lambda _db, _media_id: None,
    )
    resp = client.post(
        "/api/v1/slides/generate/from-media",
        json={
            "media_id": 1,
            "title_hint": "Media Deck",
            "theme": "black",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "media_transcript_not_found"


def test_slides_generate_invalid_prompt(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/generate",
        json={
            "prompt": "   ",
            "title_hint": "Bad",
            "theme": "black",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "prompt_required"


def test_slides_generate_invalid_conversation_id(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/generate/from-chat",
        json={
            "conversation_id": "   ",
            "title_hint": "Bad",
            "theme": "black",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "conversation_id_required"


def test_slides_generate_invalid_note_ids(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/generate/from-notes",
        json={
            "note_ids": [],
            "title_hint": "Bad",
            "theme": "black",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "note_ids_required"


def test_slides_generate_invalid_media_id(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/generate/from-media",
        json={
            "media_id": "not-a-number",
            "title_hint": "Bad",
            "theme": "black",
        },
    )
    assert resp.status_code == 422


def test_slides_generate_invalid_rag_query(slides_client):
    resp = slides_client.post(
        "/api/v1/slides/generate/from-rag",
        json={
            "query": "   ",
            "title_hint": "Bad",
            "theme": "black",
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "query_required"


@pytest.mark.parametrize("case", _TOO_LARGE_CASES, ids=[c["id"] for c in _TOO_LARGE_CASES])
def test_slides_generate_too_large(case, request, monkeypatch):
    fake_notes = None
    fake_media = None
    if case["client"] == "slides_client_with_sources":
        client, fake_notes, fake_media = request.getfixturevalue("slides_client_with_sources")
    else:
        client = request.getfixturevalue("slides_client")
    setup = case.get("setup")
    if setup:
        setup(monkeypatch, fake_notes, fake_media)
    resp = client.post(case["path"], json=case["payload"])
    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert detail["code"] == "input_too_large"
    assert detail["max_source_chars"] == 10
