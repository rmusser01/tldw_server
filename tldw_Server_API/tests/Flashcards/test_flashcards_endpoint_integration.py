import json
import uuid
import io
import zipfile
import sqlite3
import os
from datetime import datetime
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger
from PIL import Image

# Keep this module self-contained and deterministic by disabling optional
# reading-digest startup paths that pull heavyweight STT deps during app import.
os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router
from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Flashcards.asset_refs import extract_flashcard_asset_uuids
from tldw_Server_API.app.core.Flashcards.apkg_exporter import export_apkg_from_rows
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import SchedulerSettingsError
from tldw_Server_API.tests.test_config import TestConfig

# Explicit auth headers for single-user mode (required by get_request_user)
AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}


def _build_test_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG_1X1_BYTES = _build_test_png_bytes()


def _build_flashcards_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(config_info_router, prefix="/api/v1")
    app.include_router(flashcards_router, prefix="/api/v1")
    return app


fastapi_app = _build_flashcards_test_app()


@pytest.fixture(scope="function")
def flashcards_db(tmp_path):
    db_path = tmp_path / "flashcards.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture
def client_with_flashcards_db(flashcards_db: CharactersRAGDB):
    TestConfig.setup_test_environment()
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    fastapi_app = _build_flashcards_test_app()

    def override_get_db():
        logger.info("[TEST] override get_chacha_db_for_user -> flashcards_db")
        try:
            yield flashcards_db
        finally:
            pass
    # Also bypass AuthNZ in tests by returning a fixed user
    async def override_user():
        # Use an admin-capable user so tests can exercise admin-only query caps
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    # Apply dependency overrides
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    fastapi_app.dependency_overrides[get_request_user] = override_user

    # Provide a TestClient with default X-API-KEY header for all requests
    default_headers = {"X-API-KEY": TestConfig.TEST_API_KEY}
    with TestClient(fastapi_app, headers=default_headers) as c:
        yield c
    fastapi_app.dependency_overrides.clear()
    TestConfig.reset_settings()


def test_export_apkg_basic_integration(client_with_flashcards_db: TestClient):
    # Create deck
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckOne", "description": "d1"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    deck = r.json()
    deck_id = deck["id"]

    # Create basic card
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "What is 2+2?",
        "back": "4",
        "tags": ["math"]
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Create reverse card
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "USA capital?",
        "back": "Washington, D.C.",
        "model_type": "basic_reverse"
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Create cloze card with two clozes and embedded data URI image in Extra
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    extra_html = f"<img src='data:image/png;base64,{img_b64}'/>"
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "The capital of {{c1::France}} is {{c2::Paris}}.",
        "back": "",
        "model_type": "cloze",
        "extra": extra_html,
        "tags": ["geo"]
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Export APKG
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("application/apkg")
    apkg = r.content
    # open and verify
    zf = zipfile.ZipFile(io.BytesIO(apkg))
    try:
        assert 'collection.anki2' in zf.namelist()
        media_json = zf.read('media').decode('utf-8')
        media_map = json.loads(media_json)
        # At least one media from cloze extra
        assert len(media_map) >= 1
        first_index = sorted(media_map.keys())[0]
        assert first_index in zf.namelist()

        # Load notes/cards to check counts
        with zf.open('collection.anki2') as f:
            data = f.read()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
                cards = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
                # 3 notes: basic, reverse, cloze
                assert notes == 3
                # Cards: basic=1, basic_reverse=2, cloze with c1,c2 adds 2 => total 5
                assert cards == 5
            finally:
                conn.close()
    finally:
        zf.close()


def test_upload_flashcard_asset_returns_markdown_snippet_and_content(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("slide.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["asset_uuid"]
    assert payload["reference"] == f"flashcard-asset://{payload['asset_uuid']}"
    assert payload["reference"] in payload["markdown_snippet"]
    assert payload["mime_type"] == "image/png"
    assert payload["byte_size"] == len(PNG_1X1_BYTES)
    assert payload["width"] == 1
    assert payload["height"] == 1

    stored = flashcards_db.get_flashcard_asset(payload["asset_uuid"])
    assert stored is not None
    assert stored["card_uuid"] is None

    content = client_with_flashcards_db.get(
        f"/api/v1/flashcards/assets/{payload['asset_uuid']}/content",
        headers=AUTH_HEADERS,
    )
    assert content.status_code == 200
    assert content.headers["content-type"] == "image/png"
    assert content.content == PNG_1X1_BYTES


def test_deck_workspace_id_contract_and_scope_filters(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    flashcards_db.upsert_workspace("ws-1", "Workspace One")

    create_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Scoped Deck", "workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["workspace_id"] == "ws-1"
    deck_id = created["id"]

    default_list = client_with_flashcards_db.get("/api/v1/flashcards/decks", headers=AUTH_HEADERS)
    assert default_list.status_code == 200
    assert all(item["id"] != deck_id for item in default_list.json())

    workspace_list = client_with_flashcards_db.get(
        "/api/v1/flashcards/decks",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_list.status_code == 200
    assert [item["id"] for item in workspace_list.json()] == [deck_id]

    all_list = client_with_flashcards_db.get(
        "/api/v1/flashcards/decks",
        params={"include_workspace_items": True},
        headers=AUTH_HEADERS,
    )
    assert all_list.status_code == 200
    assert any(item["id"] == deck_id for item in all_list.json())

    move_to_general = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/decks/{deck_id}",
        json={"workspace_id": None, "expected_version": created["version"]},
        headers=AUTH_HEADERS,
    )
    assert move_to_general.status_code == 200
    moved = move_to_general.json()
    assert moved["workspace_id"] is None

    general_list = client_with_flashcards_db.get("/api/v1/flashcards/decks", headers=AUTH_HEADERS)
    assert general_list.status_code == 200
    assert any(item["id"] == deck_id for item in general_list.json())


def test_create_deck_returns_review_prompt_side(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Biology Reverse Recall",
            "review_prompt_side": "back",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["review_prompt_side"] == "back"


def test_create_deck_rejects_null_review_prompt_side(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Null Orientation Deck",
            "review_prompt_side": None,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_create_deck_undelete_preserves_review_prompt_side(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    create_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Restorable Deck",
            "review_prompt_side": "back",
        },
        headers=AUTH_HEADERS,
    )

    assert create_response.status_code == 200
    deck_id = create_response.json()["id"]

    flashcards_db.soft_delete_deck_by_id(deck_id)

    resurrected_id = flashcards_db.add_deck("Restorable Deck")
    assert resurrected_id == deck_id

    resurrected = flashcards_db.get_deck(deck_id)
    assert resurrected is not None
    assert resurrected["deleted"] == 0
    assert resurrected["review_prompt_side"] == "back"


def test_create_deck_undelete_preserves_orientation_unless_explicit_front_override(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    create_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Orientation API Deck",
            "review_prompt_side": "back",
        },
        headers=AUTH_HEADERS,
    )

    assert create_response.status_code == 200
    deck_id = create_response.json()["id"]
    flashcards_db.soft_delete_deck_by_id(deck_id)

    omitted_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Orientation API Deck"},
        headers=AUTH_HEADERS,
    )
    assert omitted_response.status_code == 200
    assert omitted_response.json()["review_prompt_side"] == "back"

    flashcards_db.soft_delete_deck_by_id(deck_id)

    explicit_front_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Orientation API Deck",
            "review_prompt_side": "front",
        },
        headers=AUTH_HEADERS,
    )
    assert explicit_front_response.status_code == 200
    assert explicit_front_response.json()["review_prompt_side"] == "front"


def test_update_deck_rejects_null_review_prompt_side(client_with_flashcards_db: TestClient):
    create_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Patch Null Deck"},
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200

    update_response = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/decks/{create_response.json()['id']}",
        json={
            "review_prompt_side": None,
            "expected_version": create_response.json()["version"],
        },
        headers=AUTH_HEADERS,
    )

    assert update_response.status_code == 422


def test_deck_endpoints_reject_unknown_workspace_ids(
    client_with_flashcards_db: TestClient,
):
    create_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Bad Deck", "workspace_id": "missing-ws"},
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 404
    assert "missing-ws" in create_response.json()["detail"]

    valid_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Good Deck"},
        headers=AUTH_HEADERS,
    )
    assert valid_deck.status_code == 200
    deck_id = valid_deck.json()["id"]

    update_response = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/decks/{deck_id}",
        json={"workspace_id": "missing-ws", "expected_version": valid_deck.json()["version"]},
        headers=AUTH_HEADERS,
    )
    assert update_response.status_code == 404
    assert "missing-ws" in update_response.json()["detail"]


def test_flashcard_visibility_endpoints_respect_default_general_only_and_explicit_scope(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    flashcards_db.upsert_workspace("ws-1", "Workspace One")

    workspace_deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Workspace Deck", "workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_deck_response.status_code == 200
    workspace_deck_id = workspace_deck_response.json()["id"]

    general_deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "General Deck"},
        headers=AUTH_HEADERS,
    )
    assert general_deck_response.status_code == 200
    general_deck_id = general_deck_response.json()["id"]

    workspace_card_response = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": workspace_deck_id,
            "front": "Workspace Front",
            "back": "Workspace Back",
        },
        headers=AUTH_HEADERS,
    )
    assert workspace_card_response.status_code == 200
    workspace_card = workspace_card_response.json()

    general_card_response = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": general_deck_id,
            "front": "General Front",
            "back": "General Back",
        },
        headers=AUTH_HEADERS,
    )
    assert general_card_response.status_code == 200
    general_card = general_card_response.json()

    default_list = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert default_list.status_code == 200
    default_items = default_list.json()["items"]
    assert default_list.json()["total"] == 1
    assert any(item["uuid"] == general_card["uuid"] for item in default_items)
    assert all(item["deck_id"] != workspace_deck_id for item in default_items)

    workspace_list = client_with_flashcards_db.get(
        "/api/v1/flashcards",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_list.status_code == 200
    assert workspace_list.json()["total"] == 1
    assert {item["uuid"] for item in workspace_list.json()["items"]} == {workspace_card["uuid"]}

    all_list = client_with_flashcards_db.get(
        "/api/v1/flashcards",
        params={"include_workspace_items": True},
        headers=AUTH_HEADERS,
    )
    assert all_list.status_code == 200
    assert all_list.json()["total"] == 2
    assert {item["uuid"] for item in all_list.json()["items"]} == {general_card["uuid"], workspace_card["uuid"]}

    deck_list = client_with_flashcards_db.get(
        "/api/v1/flashcards",
        params={"deck_id": workspace_deck_id},
        headers=AUTH_HEADERS,
    )
    assert deck_list.status_code == 200
    assert deck_list.json()["total"] == 1
    assert {item["uuid"] for item in deck_list.json()["items"]} == {workspace_card["uuid"]}

    default_next = client_with_flashcards_db.get("/api/v1/flashcards/review/next", headers=AUTH_HEADERS)
    assert default_next.status_code == 200
    assert default_next.json()["card"]["uuid"] == general_card["uuid"]

    workspace_next = client_with_flashcards_db.get(
        "/api/v1/flashcards/review/next",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_next.status_code == 200
    assert workspace_next.json()["card"]["uuid"] == workspace_card["uuid"]

    deck_next = client_with_flashcards_db.get(
        "/api/v1/flashcards/review/next",
        params={"deck_id": workspace_deck_id},
        headers=AUTH_HEADERS,
    )
    assert deck_next.status_code == 200
    assert deck_next.json()["card"]["uuid"] == workspace_card["uuid"]

    default_export = client_with_flashcards_db.get("/api/v1/flashcards/export", headers=AUTH_HEADERS)
    assert default_export.status_code == 200
    default_export_text = default_export.content.decode("utf-8")
    assert "General Front" in default_export_text
    assert "Workspace Front" not in default_export_text

    workspace_export = client_with_flashcards_db.get(
        "/api/v1/flashcards/export",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_export.status_code == 200
    workspace_export_text = workspace_export.content.decode("utf-8")
    assert "Workspace Front" in workspace_export_text
    assert "General Front" not in workspace_export_text

    deck_export = client_with_flashcards_db.get(
        "/api/v1/flashcards/export",
        params={"deck_id": workspace_deck_id},
        headers=AUTH_HEADERS,
    )
    assert deck_export.status_code == 200
    deck_export_text = deck_export.content.decode("utf-8")
    assert "Workspace Front" in deck_export_text
    assert "General Front" not in deck_export_text


def test_flashcard_analytics_summary_endpoint_respects_workspace_visibility(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    flashcards_db.upsert_workspace("ws-1", "Workspace One")

    general_deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "General Analytics Deck"},
        headers=AUTH_HEADERS,
    )
    assert general_deck_response.status_code == 200
    general_deck_id = general_deck_response.json()["id"]

    workspace_deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Workspace Analytics Deck", "workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_deck_response.status_code == 200
    workspace_deck_id = workspace_deck_response.json()["id"]

    review_card_response = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": workspace_deck_id,
            "front": "Workspace Review Front",
            "back": "Workspace Review Back",
        },
        headers=AUTH_HEADERS,
    )
    assert review_card_response.status_code == 200
    workspace_card = review_card_response.json()

    review_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": workspace_card["uuid"], "rating": 4},
        headers=AUTH_HEADERS,
    )
    assert review_response.status_code == 200

    default_summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        headers=AUTH_HEADERS,
    )
    assert default_summary.status_code == 200
    default_payload = default_summary.json()
    assert default_payload["reviewed_today"] == 0
    assert all(deck["deck_id"] != workspace_deck_id for deck in default_payload["decks"])

    workspace_summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_summary.status_code == 200
    workspace_payload = workspace_summary.json()
    assert workspace_payload["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in workspace_payload["decks"]} == {workspace_deck_id}

    all_summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"include_workspace_items": True},
        headers=AUTH_HEADERS,
    )
    assert all_summary.status_code == 200
    all_payload = all_summary.json()
    assert all_payload["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in all_payload["decks"]} == {general_deck_id, workspace_deck_id}

    deck_summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"deck_id": workspace_deck_id},
        headers=AUTH_HEADERS,
    )
    assert deck_summary.status_code == 200
    deck_payload = deck_summary.json()
    assert deck_payload["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in deck_payload["decks"]} == {workspace_deck_id}


def test_flashcard_analytics_summary_endpoint_keeps_empty_targeted_deck_visible(
    client_with_flashcards_db: TestClient,
):
    empty_deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Empty Analytics Deck"},
        headers=AUTH_HEADERS,
    )
    assert empty_deck_response.status_code == 200
    empty_deck_id = empty_deck_response.json()["id"]

    summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"deck_id": empty_deck_id},
        headers=AUTH_HEADERS,
    )

    assert summary.status_code == 200
    payload = summary.json()
    assert payload["reviewed_today"] == 0
    assert len(payload["decks"]) == 1
    assert payload["decks"][0]["deck_id"] == empty_deck_id
    assert payload["decks"][0]["total"] == 0
    assert payload["decks"][0]["new"] == 0
    assert payload["decks"][0]["learning"] == 0
    assert payload["decks"][0]["due"] == 0
    assert payload["decks"][0]["mature"] == 0


def test_flashcard_analytics_summary_endpoint_excludes_deleted_deck_reviews(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Deleted Analytics Deck"},
        headers=AUTH_HEADERS,
    )
    assert deck_response.status_code == 200
    deck_id = deck_response.json()["id"]

    card_response = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": deck_id,
            "front": "Front",
            "back": "Back",
        },
        headers=AUTH_HEADERS,
    )
    assert card_response.status_code == 200
    card_uuid = card_response.json()["uuid"]

    review_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_uuid, "rating": 4},
        headers=AUTH_HEADERS,
    )
    assert review_response.status_code == 200

    conn = flashcards_db.get_connection()
    conn.execute("UPDATE decks SET deleted = 1 WHERE id = ?", (deck_id,))
    conn.commit()

    summary = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"include_workspace_items": True},
        headers=AUTH_HEADERS,
    )
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["reviewed_today"] == 0
    assert all(deck["deck_id"] != deck_id for deck in payload["decks"])


def test_upload_flashcard_asset_rejects_invalid_or_oversized_upload(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    invalid = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
        headers=AUTH_HEADERS,
    )
    assert invalid.status_code == 400

    monkeypatch.setenv("FLASHCARD_ASSET_MAX_BYTES", "8")
    oversized = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("slide.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )
    assert oversized.status_code == 400


def test_create_flashcard_attaches_uploaded_asset_and_rejects_missing_refs(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    upload = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("histology.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )
    assert upload.status_code == 200
    markdown_snippet = upload.json()["markdown_snippet"]
    asset_uuid = upload.json()["asset_uuid"]

    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "front": f"Question {markdown_snippet}",
            "back": "Answer",
            "notes": "With image context",
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]
    asset = flashcards_db.get_flashcard_asset(asset_uuid)
    assert asset is not None
    assert asset["card_uuid"] == card_uuid

    missing = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "front": "Bad ![Missing](flashcard-asset://00000000-0000-0000-0000-000000000000)",
            "back": "Answer",
        },
        headers=AUTH_HEADERS,
    )
    assert missing.status_code == 400


def test_patch_and_bulk_patch_reject_invalid_asset_refs_without_rolling_back_siblings(
    client_with_flashcards_db: TestClient,
):
    first = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "First", "back": "Back"},
        headers=AUTH_HEADERS,
    )
    second = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "Second", "back": "Back"},
        headers=AUTH_HEADERS,
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_card = first.json()
    second_card = second.json()

    invalid_patch = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/{first_card['uuid']}",
        json={
            "front": "Broken ![Missing](flashcard-asset://00000000-0000-0000-0000-000000000000)",
            "expected_version": first_card["version"],
        },
        headers=AUTH_HEADERS,
    )
    assert invalid_patch.status_code == 400

    bulk = client_with_flashcards_db.patch(
        "/api/v1/flashcards/bulk",
        json=[
            {
                "uuid": first_card["uuid"],
                "back": "Saved sibling",
                "expected_version": first_card["version"],
            },
            {
                "uuid": second_card["uuid"],
                "front": "Broken ![Missing](flashcard-asset://00000000-0000-0000-0000-000000000000)",
                "expected_version": second_card["version"],
            },
        ],
        headers=AUTH_HEADERS,
    )
    assert bulk.status_code == 200
    payload = bulk.json()
    assert payload["results"][0]["status"] == "updated"
    assert payload["results"][0]["flashcard"]["back"] == "Saved sibling"
    assert payload["results"][1]["status"] == "validation_error"
    assert payload["results"][1]["error"]["invalid_fields"] == ["front"]


def test_generate_flashcards_endpoint_returns_generated_cards(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    async def fake_generate_adapter(config, context):
        assert config.get("text") == "Cell respiration summary"
        assert config.get("num_cards") == 2
        return {
            "flashcards": [
                {"front": "ATP stands for?", "back": "Adenosine triphosphate", "tags": ["bio"]},
                {"front": "Where does glycolysis occur?", "back": "Cytoplasm", "tags": "biology metabolism"},
            ],
            "count": 2,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Cell respiration summary",
            "num_cards": 2,
            "card_type": "basic",
            "difficulty": "medium",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["flashcards"][0]["front"] == "ATP stands for?"
    assert payload["flashcards"][1]["tags"] == ["biology", "metabolism"]


def test_generate_flashcards_endpoint_uses_default_provider_when_omitted(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import flashcards as flashcards_endpoint

    monkeypatch.setattr(
        flashcards_endpoint,
        "loaded_config_data",
        {"API": {"default_api": "anthropic"}},
    )

    async def fake_generate_adapter(config, context):
        assert config.get("provider") == "anthropic"
        assert config.get("text") == "Provider fallback"
        return {
            "flashcards": [
                {"front": "Q1", "back": "A1", "tags": []},
            ],
            "count": 1,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Provider fallback",
            "num_cards": 1,
            "card_type": "basic",
            "difficulty": "medium",
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["flashcards"][0]["front"] == "Q1"


def test_export_apkg_include_reverse_flag_generates_reverse(client_with_flashcards_db: TestClient):
    # Create deck and a plain basic card (no reverse/model_type)
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckTwo"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Capital of Japan?",
        "back": "Tokyo"
    }, headers=AUTH_HEADERS)
    card = r.json()
    assert card["model_type"] == "basic"
    assert card["reverse"] is False

    # Export with include_reverse=true
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg", "include_reverse": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        with zf.open('collection.anki2') as f:
            data = f.read()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                # There should be two cards for the single basic note
                notes = conn.execute("SELECT id FROM notes").fetchall()
                assert len(notes) == 1
                nid = notes[0][0]
                cards = conn.execute("SELECT nid, ord FROM cards").fetchall()
                assert len(cards) == 2
                assert all(c[0] == nid for c in cards)
                ords = sorted(c[1] for c in cards)
                assert ords == [0, 1]
            finally:
                conn.close()
    finally:
        zf.close()


def test_export_apkg_basic_reverse_without_include_reverse(client_with_flashcards_db: TestClient):
    # Create deck and a basic_reverse card (no include_reverse flag)
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckBasicRev"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Front BR",
        "back": "Back BR",
        "model_type": "basic_reverse"
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        with zf.open('collection.anki2') as f:
            data = f.read()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                notes = conn.execute("SELECT id FROM notes").fetchall()
                assert len(notes) == 1
                nid = notes[0][0]
                cards = conn.execute("SELECT nid, ord FROM cards").fetchall()
                assert len(cards) == 2
                assert all(c[0] == nid for c in cards)
                assert sorted(c[1] for c in cards) == [0, 1]
            finally:
                conn.close()
    finally:
        zf.close()


def test_export_csv_shape_and_content(client_with_flashcards_db: TestClient):
    # Create deck and two cards with tags and notes
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "CSVDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Q1",
        "back": "A1",
        "notes": "N1",
        "tags": ["alpha", "beta"]
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Q2",
        "back": "A2",
        "notes": "N2",
        "tags": ["gamma"]
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Export CSV (TSV)
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", headers=AUTH_HEADERS)
    assert r.status_code == 200
    text = r.content.decode('utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        cols = ln.split('\t')
        # Expect 5 columns: Deck, Front, Back, Tags, Notes
        assert len(cols) == 5
        assert cols[0] == "CSVDeck"
    # Verify tags and notes appear for first card
    assert any("alpha beta" in ln and "N1" in ln for ln in lines)
    assert any("gamma" in ln and "N2" in ln for ln in lines)


def test_set_tags_and_linkage(client_with_flashcards_db: TestClient):
    # Create deck and a card
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "TagDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Tagged Q",
        "back": "Tagged A"
    }, headers=AUTH_HEADERS)
    card = r.json()
    uuid = card["uuid"]

    # Set tags via endpoint
    r = client_with_flashcards_db.put(f"/api/v1/flashcards/{uuid}/tags", json={"tags": ["alpha", "beta"]}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    updated = r.json()
    assert updated.get("tags") == ["alpha", "beta"]
    # tags_json returns JSON string
    tags_json = updated.get("tags_json")
    assert tags_json
    tags = json.loads(tags_json)
    assert set(tags) == {"alpha", "beta"}

    # Verify keyword linkage
    r = client_with_flashcards_db.get(f"/api/v1/flashcards/{uuid}/tags", headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    items = data.get("items", [])
    kw_texts = {kw.get("keyword") for kw in items}
    assert {"alpha", "beta"}.issubset(kw_texts)


def test_get_flashcard_alias_path_returns_card(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "AliasDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Alias Q",
        "back": "Alias A",
        "tags": ["t1"]
    }, headers=AUTH_HEADERS)
    card = r.json()
    uuid = card["uuid"]

    r = client_with_flashcards_db.get(f"/api/v1/flashcards/{uuid}", headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["uuid"] == uuid
    assert data.get("tags") == ["t1"]


def test_source_attribution_fields_present_in_flashcard_responses(client_with_flashcards_db: TestClient):
    deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "SourceDeck"},
        headers=AUTH_HEADERS,
    )
    assert deck_response.status_code == 200
    deck_id = deck_response.json()["id"]

    created_response = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": deck_id,
            "front": "Source Q",
            "back": "Source A",
            "source_ref_type": "media",
            "source_ref_id": "42",
        },
        headers=AUTH_HEADERS,
    )
    assert created_response.status_code == 200
    created = created_response.json()
    card_uuid = created["uuid"]

    assert created["source_ref_type"] == "media"
    assert created["source_ref_id"] == "42"
    assert "conversation_id" in created
    assert "message_id" in created
    assert created["conversation_id"] is None
    assert created["message_id"] is None

    list_response = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert list_response.status_code == 200
    listed_items = list_response.json().get("items", [])
    listed = next((item for item in listed_items if item.get("uuid") == card_uuid), None)
    assert listed is not None
    assert listed["source_ref_type"] == "media"
    assert listed["source_ref_id"] == "42"
    assert "conversation_id" in listed
    assert "message_id" in listed

    get_response = client_with_flashcards_db.get(f"/api/v1/flashcards/{card_uuid}", headers=AUTH_HEADERS)
    assert get_response.status_code == 200
    fetched = get_response.json()
    assert fetched["source_ref_type"] == "media"
    assert fetched["source_ref_id"] == "42"
    assert "conversation_id" in fetched
    assert "message_id" in fetched


def test_openapi_flashcard_response_includes_source_fields():
    openapi = fastapi_app.openapi()
    flashcard_schema = openapi["components"]["schemas"]["Flashcard"]
    properties = flashcard_schema.get("properties", {})

    assert "source_ref_type" in properties
    assert "source_ref_id" in properties
    assert "conversation_id" in properties
    assert "message_id" in properties


def test_create_flashcard_normalizes_model_fields(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "NormalizeDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Cloze {{c1::x}}",
        "back": "",
        "is_cloze": True
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    card = r.json()
    assert card["model_type"] == "cloze"
    assert card["is_cloze"] is True
    assert card["reverse"] is False


def test_bulk_create_rejects_invalid_cloze(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "BulkDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards/bulk", json=[{
        "deck_id": deck_id,
        "front": "No cloze pattern here",
        "back": "",
        "model_type": "cloze"
    }], headers=AUTH_HEADERS)
    assert r.status_code == 400


def test_review_missing_flashcard_returns_404(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/review", json={
        "card_uuid": str(uuid.uuid4()),
        "rating": 3
    }, headers=AUTH_HEADERS)
    assert r.status_code == 404


def test_deck_endpoints_expose_and_patch_scheduler_settings(
    client_with_flashcards_db: TestClient,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Scheduler Deck", "description": "defaults"},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    deck = created.json()
    assert deck["scheduler_type"] == "sm2_plus"
    assert deck["scheduler_settings"]["sm2_plus"]["new_steps_minutes"] == [1, 10]
    assert deck["scheduler_settings"]["sm2_plus"]["enable_fuzz"] is False
    assert deck["scheduler_settings"]["fsrs"]["target_retention"] == pytest.approx(0.9)

    updated = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/decks/{deck['id']}",
        json={
            "description": "updated",
            "scheduler_type": "fsrs",
            "scheduler_settings": {
                "sm2_plus": {
                    "new_steps_minutes": [2, 20],
                    "relearn_steps_minutes": [15],
                    "easy_bonus": 1.5,
                },
                "fsrs": {
                    "target_retention": 0.88,
                    "maximum_interval_days": 7300,
                    "enable_fuzz": True,
                },
            },
            "expected_version": deck["version"],
        },
        headers=AUTH_HEADERS,
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["description"] == "updated"
    assert payload["scheduler_type"] == "fsrs"
    assert payload["scheduler_settings"]["sm2_plus"]["new_steps_minutes"] == [2, 20]
    assert payload["scheduler_settings"]["sm2_plus"]["relearn_steps_minutes"] == [15]
    assert payload["scheduler_settings"]["sm2_plus"]["easy_bonus"] == pytest.approx(1.5)
    assert payload["scheduler_settings"]["sm2_plus"]["graduating_interval_days"] == 1
    assert payload["scheduler_settings"]["fsrs"]["target_retention"] == pytest.approx(0.88)
    assert payload["scheduler_settings"]["fsrs"]["maximum_interval_days"] == 7300
    assert payload["scheduler_settings"]["fsrs"]["enable_fuzz"] is True

    conflict = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/decks/{deck['id']}",
        json={"expected_version": deck["version"], "description": "stale"},
        headers=AUTH_HEADERS,
    )
    assert conflict.status_code == 409


def test_review_next_prefers_due_learning_before_due_review_and_new_cards(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    deck_id = flashcards_db.add_deck("Review ordering")

    review_due_uuid = flashcards_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Review due",
            "back": "Answer",
            "queue_state": "review",
            "interval_days": 7,
            "repetitions": 4,
            "lapses": 0,
            "last_reviewed_at": "2026-03-10T10:00:00Z",
            "due_at": "2026-03-12T09:00:00Z",
        }
    )
    learning_due_uuid = flashcards_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Learning due",
            "back": "Answer",
            "queue_state": "learning",
            "step_index": 0,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "last_reviewed_at": "2026-03-12T09:55:00Z",
            "due_at": "2026-03-12T09:59:00Z",
        }
    )
    new_uuid = flashcards_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Brand new",
            "back": "Answer",
            "queue_state": "new",
        }
    )

    response = client_with_flashcards_db.get(
        "/api/v1/flashcards/review/next",
        params={"deck_id": deck_id},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selection_reason"] == "learning_due"
    assert payload["card"]["uuid"] == learning_due_uuid
    assert payload["card"]["queue_state"] == "learning"
    assert payload["card"]["scheduler_type"] == "sm2_plus"
    assert payload["card"]["next_intervals"]["again"] == "1 min"
    assert payload["card"]["next_intervals"]["easy"] == "4 days"
    assert payload["card"]["uuid"] != review_due_uuid
    assert payload["card"]["uuid"] != new_uuid


def test_fsrs_review_bootstraps_scheduler_state_and_returns_scheduler_type(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    created_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "FSRS Deck",
            "scheduler_type": "fsrs",
            "scheduler_settings": {
                "fsrs": {
                    "target_retention": 0.88,
                    "maximum_interval_days": 7300,
                    "enable_fuzz": False,
                }
            },
        },
        headers=AUTH_HEADERS,
    )
    assert created_deck.status_code == 200
    deck_id = created_deck.json()["id"]

    card_uuid = flashcards_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Existing review card",
            "back": "Answer",
            "queue_state": "review",
            "interval_days": 12,
            "repetitions": 7,
            "lapses": 1,
            "last_reviewed_at": "2026-03-01T00:00:00Z",
            "due_at": "2026-03-13T00:00:00Z",
        }
    )

    reviewed = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_uuid, "rating": 3, "answer_time_ms": 1800},
        headers=AUTH_HEADERS,
    )
    assert reviewed.status_code == 200
    payload = reviewed.json()

    assert payload["scheduler_type"] == "fsrs"
    assert payload["next_intervals"]["good"]

    persisted = flashcards_db.get_flashcard(card_uuid)
    assert persisted is not None
    assert persisted["scheduler_state_json"] != "{}"

    review_row = flashcards_db.execute_query(
        "SELECT scheduler_type FROM flashcard_reviews fr JOIN flashcards f ON f.id = fr.card_id WHERE f.uuid = ? ORDER BY fr.id DESC LIMIT 1",
        (card_uuid,),
    ).fetchone()
    assert review_row is not None
    assert review_row["scheduler_type"] == "fsrs"


def test_fsrs_review_returns_400_for_invalid_deck_settings(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
    monkeypatch,
):
    deck_id = flashcards_db.add_deck("Broken FSRS Deck", scheduler_type="fsrs")
    card_uuid = flashcards_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Broken review card",
            "back": "Answer",
            "queue_state": "review",
            "interval_days": 12,
            "repetitions": 7,
            "lapses": 1,
            "last_reviewed_at": "2026-03-01T00:00:00Z",
            "due_at": "2026-03-13T00:00:00Z",
        }
    )

    def raise_invalid_settings(raw):
        raise SchedulerSettingsError("target_retention must be between 0 and 1")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB._normalize_scheduler_settings_envelope",
        raise_invalid_settings,
    )

    reviewed = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_uuid, "rating": 3, "answer_time_ms": 1800},
        headers=AUTH_HEADERS,
    )

    assert reviewed.status_code == 400
    assert "target_retention" in reviewed.json()["detail"]


def test_get_flashcard_assistant_returns_thread_messages_and_context(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "What is a nephron?", "back": "Kidney functional unit", "notes": "renal physiology"},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]

    thread = flashcards_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=card_uuid,
    )
    flashcards_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="user",
        action_type="follow_up",
        input_modality="text",
        content="What does it do first?",
        structured_payload={"kind": "question"},
        context_snapshot={"flashcard_uuid": card_uuid},
    )

    response = client_with_flashcards_db.get(
        f"/api/v1/flashcards/{card_uuid}/assistant",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread"]["id"] == thread["id"]
    assert payload["messages"][0]["content"] == "What does it do first?"
    assert payload["context_snapshot"]["flashcard"]["uuid"] == card_uuid
    assert "fact_check" in payload["available_actions"]


def test_flashcard_assistant_respond_persists_user_and_assistant_messages(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "What is the glomerulus?", "back": "It filters blood."},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]

    async def fake_generate_reply(*, action, context, message=None, provider=None, model=None):
        assert action == "explain"
        assert context["flashcard"]["uuid"] == card_uuid
        return {
            "assistant_text": "The glomerulus is the filtration tuft in the nephron.",
            "structured_payload": {},
            "provider": provider or "openai",
            "model": model or "gpt-test",
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.generate_study_assistant_reply",
        fake_generate_reply,
    )

    response = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/assistant/respond",
        json={"action": "explain"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_message"]["role"] == "user"
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["assistant_message"]["content"] == "The glomerulus is the filtration tuft in the nephron."
    assert payload["thread"]["message_count"] == 2


def test_flashcard_assistant_respond_returns_409_for_stale_thread_version(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
    monkeypatch,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "What is the glomerulus?", "back": "It filters blood."},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]

    context_response = client_with_flashcards_db.get(
        f"/api/v1/flashcards/{card_uuid}/assistant",
        headers=AUTH_HEADERS,
    )
    assert context_response.status_code == 200
    stale_version = int(context_response.json()["thread"]["version"])
    thread_id = int(context_response.json()["thread"]["id"])

    flashcards_db.append_study_assistant_message(
        thread_id=thread_id,
        role="user",
        action_type="follow_up",
        input_modality="text",
        content="Concurrent message",
    )

    call_count = 0

    async def fake_generate_reply(*, action, context, message=None, provider=None, model=None):
        nonlocal call_count
        call_count += 1
        return {
            "assistant_text": "The glomerulus is the filtration tuft in the nephron.",
            "structured_payload": {},
            "provider": provider or "openai",
            "model": model or "gpt-test",
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.generate_study_assistant_reply",
        fake_generate_reply,
    )

    response = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/assistant/respond",
        json={"action": "explain", "expected_thread_version": stale_version},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Study assistant thread version mismatch"
    assert call_count == 0


def test_flashcard_assistant_missing_card_returns_404(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.get(
        f"/api/v1/flashcards/{uuid.uuid4()}/assistant",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 404


def test_flashcard_assistant_invalid_action_returns_422(
    client_with_flashcards_db: TestClient,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "Card", "back": "Answer"},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]

    response = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/assistant/respond",
        json={"action": "invalid_action"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 422


def test_flashcard_assistant_fact_check_returns_required_payload_keys(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "What filters blood?", "back": "The glomerulus filters blood."},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card_uuid = created.json()["uuid"]

    async def fake_generate_reply(*, action, context, message=None, provider=None, model=None):
        assert action == "fact_check"
        return {
            "assistant_text": "Filtration occurs in the glomerulus.",
            "structured_payload": {
                "verdict": "incorrect",
                "corrections": ["Filtration occurs in the glomerulus."],
                "missing_points": ["The collecting duct does not filter blood."],
                "next_prompt": "Explain where filtration starts.",
            },
            "provider": "openai",
            "model": "gpt-test",
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.generate_study_assistant_reply",
        fake_generate_reply,
    )

    response = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/assistant/respond",
        json={
            "action": "fact_check",
            "message": "I think the collecting duct filters blood.",
            "input_modality": "voice_transcript",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["structured_payload"]["verdict"] == "incorrect"
    assert payload["structured_payload"]["corrections"] == ["Filtration occurs in the glomerulus."]
    assert payload["structured_payload"]["missing_points"] == ["The collecting duct does not filter blood."]
    assert payload["structured_payload"]["next_prompt"] == "Explain where filtration starts."
    assert payload["user_message"]["input_modality"] == "voice_transcript"


def test_analytics_summary_returns_daily_metrics_and_deck_progress(client_with_flashcards_db: TestClient):
    # Seed one deck and two cards
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "AnalyticsDeck"},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    deck_id = r.json()["id"]

    first = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deck_id, "front": "Q1", "back": "A1"},
        headers=AUTH_HEADERS,
    )
    second = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deck_id, "front": "Q2", "back": "A2"},
        headers=AUTH_HEADERS,
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_uuid = first.json()["uuid"]
    second_uuid = second.json()["uuid"]

    # One successful recall + one lapse to verify retention/lapse calculations
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": first_uuid, "rating": 3, "answer_time_ms": 2500},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": second_uuid, "rating": 1, "answer_time_ms": 4500},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    # Fetch analytics summary
    r = client_with_flashcards_db.get("/api/v1/flashcards/analytics/summary", headers=AUTH_HEADERS)
    assert r.status_code == 200
    payload = r.json()

    assert payload["reviewed_today"] == 2
    assert payload["study_streak_days"] >= 1
    assert payload["avg_answer_time_ms_today"] == pytest.approx(3500.0)
    assert payload["retention_rate_today"] == pytest.approx(50.0)
    assert payload["lapse_rate_today"] == pytest.approx(50.0)
    assert payload.get("generated_at")

    deck = next((d for d in payload["decks"] if d["deck_id"] == deck_id), None)
    assert deck is not None
    assert deck["deck_name"] == "AnalyticsDeck"
    assert deck["total"] == 2
    assert deck["new"] == 0


def test_analytics_summary_honors_deck_filter(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "DeckA"},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    deck_a = r.json()["id"]

    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "DeckB"},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    deck_b = r.json()["id"]

    card_a = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deck_a, "front": "A1", "back": "A1"},
        headers=AUTH_HEADERS,
    )
    card_b = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deck_b, "front": "B1", "back": "B1"},
        headers=AUTH_HEADERS,
    )
    assert card_a.status_code == 200
    assert card_b.status_code == 200

    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_a.json()["uuid"], "rating": 3, "answer_time_ms": 1200},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_b.json()["uuid"], "rating": 1, "answer_time_ms": 3600},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200

    scoped = client_with_flashcards_db.get(
        "/api/v1/flashcards/analytics/summary",
        params={"deck_id": deck_a},
        headers=AUTH_HEADERS,
    )
    assert scoped.status_code == 200
    payload = scoped.json()

    assert payload["reviewed_today"] == 1
    assert payload["avg_answer_time_ms_today"] == pytest.approx(1200.0)
    assert payload["retention_rate_today"] == pytest.approx(100.0)
    assert payload["lapse_rate_today"] == pytest.approx(0.0)
    assert len(payload["decks"]) == 1
    assert payload["decks"][0]["deck_id"] == deck_a
    assert payload["decks"][0]["deck_name"] == "DeckA"


def test_reset_scheduling_resets_card_to_new_defaults(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "ResetDeck", "scheduler_type": "fsrs"},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    deck_id = r.json()["id"]

    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deck_id, "front": "Reset Q", "back": "Reset A"},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    card = created.json()
    card_uuid = card["uuid"]
    with flashcards_db.transaction() as conn:
        conn.execute(
            """
            UPDATE flashcards
               SET queue_state = 'review',
                   interval_days = 12,
                   repetitions = 7,
                   lapses = 1,
                   due_at = '2026-03-13T00:00:00Z',
                   last_reviewed_at = '2026-03-01T00:00:00Z',
                   scheduler_state_json = ?
             WHERE uuid = ?
            """,
            ('{"stability": 12.5, "difficulty": 4.2, "retrievability": 0.86}', card_uuid),
        )
    assert flashcards_db.get_flashcard(card_uuid)["scheduler_state_json"] != "{}"

    current = client_with_flashcards_db.get(f"/api/v1/flashcards/id/{card_uuid}", headers=AUTH_HEADERS)
    assert current.status_code == 200
    current_version = current.json()["version"]

    reset = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/reset-scheduling",
        json={"expected_version": current_version},
        headers=AUTH_HEADERS,
    )
    assert reset.status_code == 200
    payload = reset.json()

    assert payload["ef"] == pytest.approx(2.5)
    assert payload["interval_days"] == 0
    assert payload["repetitions"] == 0
    assert payload["lapses"] == 0
    assert payload["last_reviewed_at"] is None
    assert payload["due_at"] is not None
    persisted = flashcards_db.get_flashcard(card_uuid)
    assert persisted is not None
    assert persisted["scheduler_state_json"] == "{}"

    conflict = client_with_flashcards_db.post(
        f"/api/v1/flashcards/{card_uuid}/reset-scheduling",
        json={"expected_version": current_version},
        headers=AUTH_HEADERS,
    )
    assert conflict.status_code == 409


def test_patch_partial_update_keeps_required_fields(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "PatchDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Qpatch",
        "back": "Apatch",
        "notes": "N1",
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    uuid = r.json()["uuid"]

    r = client_with_flashcards_db.patch(f"/api/v1/flashcards/{uuid}", json={"notes": "N2"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    updated = r.json()
    assert updated["front"] == "Qpatch"
    assert updated["back"] == "Apatch"
    assert updated["notes"] == "N2"


def test_patch_tags_conflict_does_not_update(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "PatchTags"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Qtag",
        "back": "Atag",
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    card = r.json()
    uuid = card["uuid"]
    version = card["version"]

    r = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/{uuid}",
        json={"tags": ["alpha"], "expected_version": version + 1},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 409

    r = client_with_flashcards_db.get(f"/api/v1/flashcards/{uuid}/tags", headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json().get("count") == 0

    r = client_with_flashcards_db.get(f"/api/v1/flashcards/id/{uuid}", headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json().get("tags_json") in (None, "[]")


def test_patch_deck_id_rejects_invalid_or_deleted(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckValid"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Qdeck",
        "back": "Adeck",
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    uuid = r.json()["uuid"]

    r = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/{uuid}",
        json={"deck_id": 999999},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 400
    det = r.json().get("detail") or {}
    assert det.get("error") == "Deck not found"

    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckDeleted"}, headers=AUTH_HEADERS)
    del_id = r.json()["id"]
    with flashcards_db.transaction() as conn:
        conn.execute("UPDATE decks SET deleted = 1 WHERE id = ?", (del_id,))

    r = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/{uuid}",
        json={"deck_id": del_id},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 400
    det = r.json().get("detail") or {}
    assert det.get("error") == "Deck not found"


def test_bulk_patch_returns_mixed_results(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "BulkPatchDeck"},
        headers=AUTH_HEADERS,
    )
    deck_id = r.json()["id"]

    first = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": deck_id,
            "front": "Original first",
            "back": "Original back",
        },
        headers=AUTH_HEADERS,
    )
    second = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": deck_id,
            "front": "Plain front",
            "back": "Plain back",
        },
        headers=AUTH_HEADERS,
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_card = first.json()
    second_card = second.json()

    response = client_with_flashcards_db.patch(
        "/api/v1/flashcards/bulk",
        json=[
            {
                "uuid": first_card["uuid"],
                "front": "Updated front",
                "expected_version": first_card["version"],
            },
            {
                "uuid": second_card["uuid"],
                "model_type": "cloze",
                "front": "Not a cloze",
                "expected_version": second_card["version"],
            },
        ],
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["status"] == "updated"
    assert data["results"][0]["flashcard"]["front"] == "Updated front"
    assert data["results"][1]["status"] == "validation_error"
    assert data["results"][1]["error"]["invalid_fields"] == ["front"]


def test_bulk_patch_reports_conflict_without_rolling_back_siblings(
    client_with_flashcards_db: TestClient,
):
    first = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "Sibling front", "back": "Sibling back"},
        headers=AUTH_HEADERS,
    )
    second = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "Conflict front", "back": "Conflict back"},
        headers=AUTH_HEADERS,
    )
    assert first.status_code == 200
    assert second.status_code == 200
    first_card = first.json()
    second_card = second.json()

    concurrent = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/{second_card['uuid']}",
        json={
            "front": "Other update",
            "expected_version": second_card["version"],
        },
        headers=AUTH_HEADERS,
    )
    assert concurrent.status_code == 200

    response = client_with_flashcards_db.patch(
        "/api/v1/flashcards/bulk",
        json=[
            {
                "uuid": first_card["uuid"],
                "back": "Saved sibling",
                "expected_version": first_card["version"],
            },
            {
                "uuid": second_card["uuid"],
                "back": "Conflicted edit",
                "expected_version": second_card["version"],
            },
        ],
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["status"] == "updated"
    assert data["results"][0]["flashcard"]["back"] == "Saved sibling"
    assert data["results"][1]["status"] == "conflict"

    current = client_with_flashcards_db.get(
        f"/api/v1/flashcards/id/{first_card['uuid']}",
        headers=AUTH_HEADERS,
    )
    assert current.status_code == 200
    assert current.json()["back"] == "Saved sibling"


def test_bulk_patch_classifies_deleted_deck_and_missing_card(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    live_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "LiveBulkPatchDeck"},
        headers=AUTH_HEADERS,
    )
    deleted_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "DeletedBulkPatchDeck"},
        headers=AUTH_HEADERS,
    )
    assert live_deck.status_code == 200
    assert deleted_deck.status_code == 200
    deleted_deck_id = deleted_deck.json()["id"]
    with flashcards_db.transaction() as conn:
        conn.execute("UPDATE decks SET deleted = 1 WHERE id = ?", (deleted_deck_id,))

    live_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": live_deck.json()["id"],
            "front": "Live card",
            "back": "Live back",
        },
        headers=AUTH_HEADERS,
    )
    assert live_card.status_code == 200
    card = live_card.json()

    response = client_with_flashcards_db.patch(
        "/api/v1/flashcards/bulk",
        json=[
            {
                "uuid": card["uuid"],
                "deck_id": deleted_deck_id,
                "expected_version": card["version"],
            },
            {
                "uuid": str(uuid.uuid4()),
                "front": "No card",
                "expected_version": 1,
            },
        ],
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["status"] == "validation_error"
    assert data["results"][0]["error"]["invalid_deck_ids"] == [deleted_deck_id]
    assert data["results"][1]["status"] == "not_found"


def test_export_apkg_include_reverse_no_duplication_for_basic_reverse(client_with_flashcards_db: TestClient):
    # Create deck and a basic_reverse card
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DeckThree"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Alpha",
        "back": "Omega",
        "model_type": "basic_reverse"
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    # Export with include_reverse=true; still should be only two cards (ord 0 and 1)
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg", "include_reverse": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        with zf.open('collection.anki2') as f:
            data = f.read()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                notes = conn.execute("SELECT id FROM notes").fetchall()
                assert len(notes) == 1
                nid = notes[0][0]
                cards = conn.execute("SELECT nid, ord FROM cards").fetchall()
                assert len(cards) == 2
                assert all(c[0] == nid for c in cards)
                assert sorted(c[1] for c in cards) == [0, 1]
            finally:
                conn.close()
    finally:
        zf.close()


def test_export_csv_escapes_specials_and_multiline(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "SpecDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    special_front = "Line1\nLine\t2\nEmoji: 😀"
    special_back = "Tab\tSeparated\nNewline"
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": special_front,
        "back": special_back,
        "notes": "Note\nWith\tTabs",
        "tags": ["spec"]
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200

    r = client_with_flashcards_db.get("/api/v1/flashcards/export", headers=AUTH_HEADERS)
    assert r.status_code == 200
    text = r.content.decode('utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    cols = lines[0].split('\t')
    assert len(cols) == 5
    # No embedded tabs/newlines in any column (other than separators)
    for c in cols:
        assert '\n' not in c and '\r' not in c and '\t' not in c


def test_import_tsv_creates_cards_and_decks(client_with_flashcards_db: TestClient):
    # Build TSV content with two lines
    content = (
        "DeckA\tFront A\tBack A\talpha beta\tNote A\n"
        "DeckB\tFront B\tBack B\tgamma\tNote B\n"
    )
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 2
    # List flashcards and verify deck names and tags present in list output
    r = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert r.status_code == 200
    items = r.json().get('items', [])
    # We expect at least two entries from import
    assert any(it.get('deck_name') == 'DeckA' and 'alpha' in (json.loads(it.get('tags_json') or '[]')) for it in items)
    assert any(it.get('deck_name') == 'DeckB' and 'gamma' in (json.loads(it.get('tags_json') or '[]')) for it in items)


def test_import_tsv_with_header_modeltype_extra(client_with_flashcards_db: TestClient):
    header = "Deck\tFront\tBack\tTags\tNotes\tExtra\tModelType\tReverse\n"
    rows = (
        "DeckC\tF1\tB1\talpha\tN1\tE1\tbasic_reverse\ttrue\n"
        "DeckC\tCloze {{c1::x}}\t\tctag\tCN\tCE\tcloze\tfalse\n"
    )
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    # List back
    r = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    items = r.json().get('items', [])
    # Find basic_reverse card
    br = next(it for it in items if it.get('front') == 'F1')
    assert br['model_type'] == 'basic_reverse'
    assert br['reverse'] is True
    assert br.get('extra') == 'E1'
    # Find cloze
    cz = next(it for it in items if it.get('model_type') == 'cloze')
    assert '{{c1::x}}' in cz.get('front')
    assert cz.get('extra') == 'CE'


def test_import_tsv_with_deckdescription_and_iscloze(client_with_flashcards_db: TestClient):
    header = "Deck\tFront\tBack\tTags\tNotes\tDeckDescription\tIsCloze\n"
    rows = (
        "DeckD\tCloze {{c1::zz}}\t\tztag\tZN\tThis is a described deck\ttrue\n"
    )
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    # Verify deck description persisted
    r = client_with_flashcards_db.get("/api/v1/flashcards/decks", headers=AUTH_HEADERS)
    assert r.status_code == 200
    decks = r.json()
    deck = next(d for d in decks if d.get('name') == 'DeckD')
    assert deck.get('description') == 'This is a described deck'
    # Verify imported card is cloze
    r = client_with_flashcards_db.get("/api/v1/flashcards")
    items = r.json().get('items', [])
    card = next(it for it in items if it.get('deck_name') == 'DeckD')
    assert card.get('is_cloze') is True
    assert card.get('model_type') == 'cloze'


def test_export_csv_with_header_and_custom_delimiter(client_with_flashcards_db: TestClient):
    # Create deck/card
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DelimDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "F",
        "back": "B",
        "notes": "N",
        "tags": ["t1", "t2"]
    })
    assert r.status_code == 200
    # Export with delimiter ';' and header
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"delimiter": ";", "include_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.headers.get('content-type', '').startswith('text/csv')
    text = r.content.decode('utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 2
    header = lines[0]
    assert header == 'Deck;Front;Back;Tags;Notes'
    cols = lines[1].split(';')
    assert len(cols) == 5
    assert cols[0] == 'DelimDeck'
    # Ensure no tabs present
    assert '\t' not in lines[0] and '\t' not in lines[1]


def test_export_csv_extended_header_and_values(client_with_flashcards_db: TestClient):
    # Create deck/card with extra + reverse
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "ExtDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Fext",
        "back": "Bext",
        "extra": "Xtra",
        "model_type": "basic_reverse"
    })
    assert r.status_code == 200
    # Export with extended header
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"delimiter": ";", "include_header": True, "extended_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    text = r.content.decode('utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) >= 2
    header = lines[0]
    assert header == 'Deck;Front;Back;Tags;Notes;Extra;Reverse'
    # Check row contains extra and reverse column values
    cols = lines[1].split(';')
    assert len(cols) == 7
    assert cols[0] == 'ExtDeck'
    assert cols[5] == 'Xtra'
    assert cols[6] in ('true', 'false')


def test_import_minimal_header_front_back_default_deck(client_with_flashcards_db: TestClient):
    header = "Front\tBack\n"
    rows = "Qmin\tAmin\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    # Default deck should exist and card should belong to it
    r = client_with_flashcards_db.get("/api/v1/flashcards/decks", headers=AUTH_HEADERS)
    decks = r.json()
    assert any(d.get('name') == 'Default' for d in decks)
    r = client_with_flashcards_db.get("/api/v1/flashcards")
    items = r.json().get('items', [])
    found = next((it for it in items if it.get('front') == 'Qmin' and it.get('back') == 'Amin'), None)
    assert found is not None
    assert found.get('deck_name') == 'Default'


def test_round_trip_import_export_extra_reverse(client_with_flashcards_db: TestClient):
    # Import TSV with Extra and Reverse
    header = "Deck\tFront\tBack\tTags\tNotes\tExtra\tReverse\n"
    rows = "DeckR\tFr\tBk\ttr\tnt\tXVal\ttrue\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    # Export CSV extended
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"delimiter": ";", "include_header": True, "extended_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    txt = r.content.decode('utf-8')
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    # Find the DeckR row and assert Extra/Reverse values
    row = next(ln for ln in lines[1:] if ln.startswith('DeckR;'))
    cols = row.split(';')
    assert cols[0] == 'DeckR' and cols[1] == 'Fr' and cols[2] == 'Bk'
    assert cols[5] == 'XVal' and cols[6] in ('true', 'false')
    # Export APKG and assert two cards (reverse)
    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    import io, sqlite3, zipfile, tempfile, os
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        with zf.open('collection.anki2') as f:
            data = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                notes = conn.execute("SELECT id FROM notes").fetchall()
                # At least one note for DeckR
                assert len(notes) >= 1
                # cards: ensure at least one note has two cards
                nids = [n[0] for n in notes]
                has_two = False
                for nid in nids:
                    cnt = conn.execute("SELECT COUNT(*) FROM cards WHERE nid=?", (nid,)).fetchone()[0]
                    if cnt == 2:
                        has_two = True
                        break
                assert has_two
            finally:
                conn.close()
    finally:
        zf.close()


def test_export_apkg_uses_due_at_for_review_cards(client_with_flashcards_db: TestClient, flashcards_db: CharactersRAGDB):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "DueDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Due Q",
        "back": "Due A"
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    card = r.json()
    card_uuid = card["uuid"]

    # Force a review state with a specific due_at
    due_at = "2030-01-15T00:00:00Z"
    with flashcards_db.transaction() as conn:
        conn.execute(
            "UPDATE flashcards SET repetitions = 3, interval_days = 10, due_at = ? WHERE uuid = ?",
            (due_at, card_uuid),
        )

    r = client_with_flashcards_db.get("/api/v1/flashcards/export", params={"format": "apkg"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        with zf.open('collection.anki2') as f:
            data = f.read()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'col.anki2')
            with open(p, 'wb') as fh:
                fh.write(data)
            conn = sqlite3.connect(p)
            try:
                col = conn.execute("SELECT crt FROM col").fetchone()
                assert col
                crt = int(col[0])
                due_dt = datetime.fromisoformat(due_at.replace('Z', '+00:00'))
                expected_due = max(1, int((due_dt.timestamp() - crt) / 86400.0))
                note_id = conn.execute("SELECT id FROM notes WHERE flds LIKE ?", ("Due Q%",)).fetchone()[0]
                due = conn.execute("SELECT due FROM cards WHERE nid = ?", (note_id,)).fetchone()[0]
                assert due == expected_due
            finally:
                conn.close()
    finally:
        zf.close()


def test_import_malformed_rows_reported(client_with_flashcards_db: TestClient):
    # Missing Front field entirely
    header = "Deck\tBack\tTags\tNotes\n"
    rows = "BadDeck\tOnlyBack\talpha\tSomeNote\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    # Should import 0 and report an error
    assert data.get('imported') == 0
    errs = data.get('errors', [])
    assert len(errs) == 1
    assert 'Missing required field' in errs[0].get('error', '')
    # No cards created
    r = client_with_flashcards_db.get("/api/v1/flashcards")
    assert r.status_code == 200
    assert len(r.json().get('items', [])) == 0


def test_import_malformed_missing_deck_with_header(client_with_flashcards_db: TestClient):
    # Header includes Deck, but row has empty Deck
    header = "Deck\tFront\tBack\tTags\tNotes\n"
    rows = "\tFrontOnly\tB\ta\tn\n"  # first field empty deck
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 0
    errs = data.get('errors', [])
    assert any('Missing required field: Deck' in e.get('error', '') for e in errs)


def test_import_malformed_invalid_cloze(client_with_flashcards_db: TestClient):
    # Header declares cloze model, but front lacks cN pattern
    header = "Deck\tFront\tBack\tTags\tNotes\tModelType\n"
    rows = "CDeck\tNot a cloze\t\t\t\tcloze\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 0
    errs = data.get('errors', [])
    assert any('Invalid cloze' in e.get('error', '') for e in errs)


def test_import_oversize_field_rejected(client_with_flashcards_db: TestClient):
    # Front exceeds MAX_FIELD_LENGTH (8192)
    long_front = 'F' * 9000
    header = "Deck\tFront\tBack\tTags\tNotes\n"
    rows = f"LD\t{long_front}\tB\tT\tN\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 0
    errs = data.get('errors', [])
    assert any('Field too long' in e.get('error', '') for e in errs)


def test_import_oversize_line_rejected(client_with_flashcards_db: TestClient):
    # Construct a very long single line (> 32768)
    big = 'A' * 33000
    header = "Deck\tFront\tBack\tTags\tNotes\n"
    rows = f"LD\t{big}\tB\tT\tN\n"
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 0
    errs = data.get('errors', [])
    assert any('Line too long' in e.get('error', '') for e in errs)


def test_import_respects_max_lines_cap(client_with_flashcards_db: TestClient, monkeypatch):
    # Limit to 3 lines via env override
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_LINES', '3')
    header = "Deck\tFront\tBack\tTags\tNotes\n"
    rows = "".join([f"D\tF{i}\tB\tT\tN\n" for i in range(1, 6)])  # 5 rows
    content = header + rows
    r = client_with_flashcards_db.post("/api/v1/flashcards/import", json={"content": content, "has_header": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 3
    errs = data.get('errors', [])
    # Should include a cap message
    assert any('Maximum import line limit' in (e.get('error') or '') for e in errs)


def test_import_respects_query_param_caps(client_with_flashcards_db: TestClient, monkeypatch):
    # Env cap high, query lowers to 2
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_LINES', '10')
    header = "Deck\tFront\tBack\tTags\tNotes\n"
    rows = "".join([f"D\tF{i}\tB\tT\tN\n" for i in range(1, 5)])  # 4 rows
    content = header + rows
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards/import",
        params={"max_lines": 2},
        json={"content": content, "has_header": True},
        headers=AUTH_HEADERS
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 2
    errs = data.get('errors', [])
    assert any('Maximum import line limit' in (e.get('error') or '') for e in errs)


def test_config_endpoint_flashcards_import_limits(client_with_flashcards_db: TestClient, monkeypatch):
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_LINES', '999')
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_LINE_LENGTH', '12345')
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_FIELD_LENGTH', '2345')
    r = client_with_flashcards_db.get("/api/v1/config/flashcards-import-limits", headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('max_lines') == 999
    assert data.get('max_line_length') == 12345
    assert data.get('max_field_length') == 2345
    assert 'query_params' in data.get('overrides', {})


def test_structured_preview_endpoint_returns_drafts(client_with_flashcards_db: TestClient):
    payload = {
        "content": "Q: What is ATP?\nA: Primary energy currency.\n"
    }

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/import/structured/preview",
        json=payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_format"] == "qa_labels"
    assert data["drafts"][0]["front"] == "What is ATP?"
    assert data["drafts"][0]["back"] == "Primary energy currency."
    assert data["errors"] == []


def test_structured_preview_respects_line_caps(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    monkeypatch.setenv("FLASHCARDS_IMPORT_MAX_LINES", "2")

    payload = {
        "content": "Q: One\nA: First\nQ: Two\nA: Second\n"
    }

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/import/structured/preview",
        json=payload,
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["drafts"]) == 1
    assert any(
        "Maximum preview line limit" in error["error"]
        for error in data["errors"]
    )


def test_bulk_create_rejects_fields_longer_than_import_cap(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    monkeypatch.setenv("FLASHCARDS_IMPORT_MAX_FIELD_LENGTH", "10")

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/bulk",
        json=[
            {
                "front": "Front value that is too long",
                "back": "Short",
            }
        ],
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "Flashcard field too long"
    assert detail["invalid_fields"] == ["front"]
    assert "10 bytes" in detail["message"]


def test_import_json_file_basic(client_with_flashcards_db: TestClient):
    import json as _json
    payload = [
        {"deck": "JDeck", "front": "JF1", "back": "JB1", "tags": ["jt1", "jt2"], "extra": "JE1", "reverse": True},
        {"deck": "JDeck", "front": "Cloze {{c1::xx}}", "model_type": "cloze", "extra": "JCE"}
    ]
    files = {
        'file': ('cards.json', _json.dumps(payload), 'application/json')
    }
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/json", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 2
    # Verify created
    r = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    items = r.json().get('items', [])
    assert any(it.get('deck_name') == 'JDeck' and it.get('front') == 'JF1' for it in items)
    assert any(it.get('model_type') == 'cloze' and '{{c1::' in it.get('front') for it in items)


def test_import_json_reuses_workspace_owned_deck_by_name(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    flashcards_db.upsert_workspace("ws-1", "Workspace One")
    workspace_deck_id = flashcards_db.add_deck("Workspace Deck", workspace_id="ws-1")

    payload = [
        {"deck": "Workspace Deck", "front": "Front", "back": "Back"},
    ]
    files = {"file": ("cards.json", json.dumps(payload), "application/json")}
    response = client_with_flashcards_db.post("/api/v1/flashcards/import/json", files=files, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["imported"] == 1

    cards_response = client_with_flashcards_db.get(
        "/api/v1/flashcards",
        params={"deck_id": workspace_deck_id},
        headers=AUTH_HEADERS,
    )
    assert cards_response.status_code == 200
    cards = cards_response.json()["items"]
    assert len(cards) == 1
    assert cards[0]["deck_id"] == workspace_deck_id

    decks = flashcards_db.list_decks(limit=100, offset=0, include_workspace_items=True)
    same_name_decks = [deck for deck in decks if deck["name"] == "Workspace Deck"]
    assert len(same_name_decks) == 1
    assert same_name_decks[0]["id"] == workspace_deck_id


def test_import_json_caps_and_errors(client_with_flashcards_db: TestClient, monkeypatch):
    import json as _json
    # Set env caps low
    monkeypatch.setenv('FLASHCARDS_IMPORT_MAX_LINES', '1')
    payload = [
        {"deck": "J1", "front": "F1", "back": "B1"},
        {"deck": "J2", "front": "F2", "back": "B2"}
    ]
    files = {'file': ('cards.json', _json.dumps(payload), 'application/json')}
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/json", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get('imported') == 1
    errs = data.get('errors', [])
    assert any('Maximum import item limit' in (e.get('error') or '') for e in errs)
    # Invalid cloze error
    payload2 = [{"front": "not cloze", "model_type": "cloze"}]
    files2 = {'file': ('cards.json', _json.dumps(payload2), 'application/json')}
    r2 = client_with_flashcards_db.post("/api/v1/flashcards/import/json", files=files2, headers=AUTH_HEADERS)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get('imported') == 0
    errs2 = data2.get('errors', [])
    assert any('Invalid cloze' in (e.get('error') or '') for e in errs2)


def test_import_json_unicode_preserved(client_with_flashcards_db: TestClient):
    import json as _json
    payload = [
        {"deck": "JUnicode", "front": "Hello 😀", "back": "World"}
    ]
    files = {'file': ('cards.json', _json.dumps(payload), 'application/json')}
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/json", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 200
    r2 = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert r2.status_code == 200
    items = r2.json().get('items', [])
    assert any('😀' in (it.get('front') or '') for it in items)


def test_import_apkg_file_basic(client_with_flashcards_db: TestClient):
    payload_rows = [
        {
            "deck_name": "APKG Deck",
            "model_type": "basic",
            "front": "APKG Q1",
            "back": "APKG A1",
            "extra": "extra 1",
            "tags_json": json.dumps(["apkg", "basic"]),
        },
        {
            "deck_name": "APKG Deck",
            "model_type": "basic_reverse",
            "front": "APKG Q2",
            "back": "APKG A2",
            "extra": "extra 2",
            "tags_json": json.dumps(["apkg", "reverse"]),
            "reverse": True,
        },
        {
            "deck_name": "APKG Deck",
            "model_type": "cloze",
            "front": "APKG {{c1::cloze}}",
            "back": "",
            "extra": "cloze extra",
            "tags_json": json.dumps(["apkg", "cloze"]),
        },
    ]
    apkg = export_apkg_from_rows(payload_rows)
    files = {
        "file": ("flashcards.apkg", apkg, "application/apkg"),
    }
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/apkg", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get("imported") == 3
    assert data.get("errors") == []

    r2 = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert r2.status_code == 200
    items = r2.json().get("items", [])
    by_front = {item.get("front"): item for item in items}
    assert "APKG Q1" in by_front
    assert "APKG Q2" in by_front
    assert "APKG {{c1::cloze}}" in by_front
    assert by_front["APKG Q2"].get("model_type") == "basic_reverse"
    assert by_front["APKG Q2"].get("reverse") is True
    assert by_front["APKG {{c1::cloze}}"].get("model_type") == "cloze"
    assert by_front["APKG {{c1::cloze}}"].get("is_cloze") is True


def test_import_apkg_preserves_scheduling_fields(client_with_flashcards_db: TestClient):
    payload_rows = [
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "New Card",
            "back": "A0",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
        },
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Review Card",
            "back": "A1",
            "ef": 2.1,
            "interval_days": 14,
            "repetitions": 5,
            "lapses": 2,
            "due_at": None,
        },
    ]
    apkg = export_apkg_from_rows(payload_rows)
    files = {
        "file": ("sched.apkg", apkg, "application/apkg"),
    }
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/apkg", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 200
    assert r.json().get("imported") == 2

    r2 = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert r2.status_code == 200
    by_front = {item.get("front"): item for item in r2.json().get("items", [])}
    assert by_front["New Card"].get("repetitions") == 0
    assert by_front["New Card"].get("interval_days") == 0
    assert by_front["New Card"].get("due_at") is None

    review = by_front["Review Card"]
    assert review.get("repetitions") == 5
    assert review.get("interval_days") == 14
    assert review.get("lapses") == 2
    assert pytest.approx(review.get("ef"), rel=1e-3) == 2.1
    assert review.get("due_at") is not None


def test_import_apkg_invalid_archive_returns_400(client_with_flashcards_db: TestClient):
    files = {
        "file": ("bad.apkg", b"not a zip archive", "application/apkg"),
    }
    r = client_with_flashcards_db.post("/api/v1/flashcards/import/apkg", files=files, headers=AUTH_HEADERS)
    assert r.status_code == 400
    assert "Invalid APKG archive" in r.json().get("detail", "")


def test_export_apkg_packages_managed_assets_and_preserves_notes(
    client_with_flashcards_db: TestClient,
):
    upload_front = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("front.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )
    assert upload_front.status_code == 200
    upload_notes = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("notes.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )
    assert upload_notes.status_code == 200

    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "front": f"Question {upload_front.json()['markdown_snippet']}",
            "back": "Answer",
            "notes": f"Notes {upload_notes.json()['markdown_snippet']}",
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200

    exported = client_with_flashcards_db.get(
        "/api/v1/flashcards/export",
        params={"format": "apkg"},
        headers=AUTH_HEADERS,
    )
    assert exported.status_code == 200

    zf = zipfile.ZipFile(io.BytesIO(exported.content))
    try:
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert len(media_map) == 2

        with zf.open("collection.anki2") as collection_file:
            collection_bytes = collection_file.read()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            col_path = os.path.join(tmp, "collection.anki2")
            with open(col_path, "wb") as fh:
                fh.write(collection_bytes)
            conn = sqlite3.connect(col_path)
            try:
                flds = conn.execute("SELECT flds FROM notes").fetchone()[0].split("\x1f")
                assert len(flds) == 4
                assert "<img" in flds[0]
                assert "<img" in flds[3]
                assert "flashcard-asset://" not in flds[0]
                assert "flashcard-asset://" not in flds[3]
            finally:
                conn.close()
    finally:
        zf.close()


def test_import_apkg_creates_managed_assets_from_media_and_preserves_notes(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    payload_rows = [
        {
            "deck_name": "APKG Managed",
            "model_type": "basic",
            "front": "Front ![Slide](flashcard-asset://asset-front)",
            "back": "Back",
            "notes": "Notes ![Notes](flashcard-asset://asset-notes)",
        }
    ]

    def asset_loader(asset_uuid: str):
        return {
            "content": PNG_1X1_BYTES,
            "mime_type": "image/png",
            "original_filename": f"{asset_uuid}.png",
        }

    apkg = export_apkg_from_rows(payload_rows, asset_loader=asset_loader)
    imported = client_with_flashcards_db.post(
        "/api/v1/flashcards/import/apkg",
        files={"file": ("managed.apkg", apkg, "application/apkg")},
        headers=AUTH_HEADERS,
    )
    assert imported.status_code == 200
    assert imported.json()["imported"] == 1

    listed = client_with_flashcards_db.get("/api/v1/flashcards", headers=AUTH_HEADERS)
    assert listed.status_code == 200
    card = next(item for item in listed.json()["items"] if item["front"].startswith("Front "))
    front_asset_uuids = extract_flashcard_asset_uuids(card["front"])
    notes_asset_uuids = extract_flashcard_asset_uuids(card["notes"])
    assert len(front_asset_uuids) == 1
    assert len(notes_asset_uuids) == 1

    front_asset = flashcards_db.get_flashcard_asset(front_asset_uuids[0])
    notes_asset = flashcards_db.get_flashcard_asset(notes_asset_uuids[0])
    assert front_asset is not None
    assert notes_asset is not None
    assert front_asset["card_uuid"] == card["uuid"]
    assert notes_asset["card_uuid"] == card["uuid"]


def test_export_apkg_rejects_oversized_total_media(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    uploaded = client_with_flashcards_db.post(
        "/api/v1/flashcards/assets",
        files={"file": ("front.png", PNG_1X1_BYTES, "image/png")},
        headers=AUTH_HEADERS,
    )
    assert uploaded.status_code == 200

    created = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "front": uploaded.json()["markdown_snippet"],
            "back": "Answer",
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200

    monkeypatch.setenv("FLASHCARDS_APKG_MAX_MEDIA_BYTES", "8")
    exported = client_with_flashcards_db.get(
        "/api/v1/flashcards/export",
        params={"format": "apkg"},
        headers=AUTH_HEADERS,
    )
    assert exported.status_code == 400
    assert "APKG media exceeds max total size" in exported.json()["detail"]


def test_import_apkg_rejects_oversized_total_media(
    client_with_flashcards_db: TestClient,
    monkeypatch,
):
    payload_rows = [
        {
            "deck_name": "APKG Managed",
            "model_type": "basic",
            "front": "Front ![Slide](flashcard-asset://asset-front)",
            "back": "Back",
        }
    ]

    def asset_loader(asset_uuid: str):
        return {
            "content": PNG_1X1_BYTES,
            "mime_type": "image/png",
            "original_filename": f"{asset_uuid}.png",
        }

    apkg = export_apkg_from_rows(payload_rows, asset_loader=asset_loader)
    monkeypatch.setenv("FLASHCARDS_APKG_MAX_MEDIA_BYTES", "8")
    imported = client_with_flashcards_db.post(
        "/api/v1/flashcards/import/apkg",
        files={"file": ("managed.apkg", apkg, "application/apkg")},
        headers=AUTH_HEADERS,
    )
    assert imported.status_code == 400
    assert "APKG media exceeds max total size" in imported.json()["detail"]


def test_export_csv_preserves_quotes_and_no_extra_separators(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "QuoteDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "\"Quoted\" Front",
        "back": "Back with \"quotes\"",
        "notes": "Note with 'single' and \"double\" quotes",
        "tags": ["qtag"]
    })
    assert r.status_code == 200

    r = client_with_flashcards_db.get("/api/v1/flashcards/export", headers=AUTH_HEADERS)
    assert r.status_code == 200
    text = r.content.decode('utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    cols = lines[0].split('\t')
    assert len(cols) == 5
    # Quotes should be preserved; no extra separators injected
    assert '"Quoted" Front' in cols[1]
    assert 'Back with "quotes"' in cols[2]
    assert 'Note with' in cols[4]
    # No stray tabs/newlines
    for c in cols:
        assert '\n' not in c and '\r' not in c and '\t' not in c


def test_bulk_create_modeltype_reverse_and_tag_linkage(client_with_flashcards_db: TestClient):
    # Create a deck first
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "BulkDeck"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    deck_id = r.json()["id"]

    payload = [
        {
            "deck_id": deck_id,
            "front": "F1",
            "back": "B1",
            "extra": "E1",
            "model_type": "basic_reverse",
            "tags": ["t1", "t2"],
        },
        {
            "deck_id": deck_id,
            "front": "Cloze {{c1::x}}",
            "back": "",
            "is_cloze": True,
            "tags": ["cz"],
        },
    ]
    r = client_with_flashcards_db.post("/api/v1/flashcards/bulk", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data.get("count") == 2
    items = data.get("items", [])
    assert len(items) == 2
    first = next(i for i in items if i.get("front") == "F1")
    assert first.get("model_type") == "basic_reverse" and first.get("reverse") is True
    assert first.get("extra") == "E1"
    uuid1 = first.get("uuid")
    # Verify keyword linkage for first item
    r = client_with_flashcards_db.get(f"/api/v1/flashcards/{uuid1}/tags", headers=AUTH_HEADERS)
    assert r.status_code == 200
    kw_texts = {kw.get("keyword") for kw in r.json().get("items", [])}
    assert {"t1", "t2"}.issubset(kw_texts)

    second = next(i for i in items if i.get("front", "").startswith("Cloze "))
    assert second.get("model_type") == "cloze"
    assert second.get("is_cloze") is True


def test_create_single_links_tags_to_keywords(client_with_flashcards_db: TestClient):
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "SingleDeck"}, headers=AUTH_HEADERS)
    deck_id = r.json()["id"]
    r = client_with_flashcards_db.post("/api/v1/flashcards", json={
        "deck_id": deck_id,
        "front": "Tagged Single",
        "back": "Ans",
        "tags": ["sx", "sy"],
    }, headers=AUTH_HEADERS)
    assert r.status_code == 200
    card = r.json()
    uuid = card.get("uuid")
    # Verify linkage
    r = client_with_flashcards_db.get(f"/api/v1/flashcards/{uuid}/tags", headers=AUTH_HEADERS)
    assert r.status_code == 200
    items = r.json().get("items", [])
    assert {"sx", "sy"}.issubset({kw.get("keyword") for kw in items})


def test_create_with_invalid_deck_returns_400(client_with_flashcards_db: TestClient):
    # Use a non-existent deck id
    r = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": 999999,
            "front": "Q",
            "back": "A"
        },
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 400
    det = r.json().get("detail") or {}
    assert det.get("error") == "Deck not found"
    assert 999999 in (det.get("invalid_deck_ids") or [])


def test_bulk_create_with_invalid_deck_returns_400(client_with_flashcards_db: TestClient):
    # Create a valid deck to mix
    r = client_with_flashcards_db.post("/api/v1/flashcards/decks", json={"name": "ValidBulk"}, headers=AUTH_HEADERS)
    assert r.status_code == 200
    valid_deck_id = r.json()["id"]
    payload = [
        {"deck_id": valid_deck_id, "front": "Fok", "back": "Bok"},
        {"deck_id": 42424242, "front": "Fbad", "back": "Bbad"},
    ]
    r = client_with_flashcards_db.post("/api/v1/flashcards/bulk", json=payload, headers=AUTH_HEADERS)
    assert r.status_code == 400
    data = r.json()
    # FastAPI wraps detail dict under top-level 'detail'
    det = data.get("detail") or {}
    assert det.get("error")
    assert 42424242 in det.get("invalid_deck_ids", [])


def test_flashcard_tag_suggestions_global_visibility_across_general_and_workspace_decks(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    flashcards_db.upsert_workspace("ws-tags-global", "Workspace Tags Global")

    general_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags General Deck"},
        headers=AUTH_HEADERS,
    )
    assert general_deck.status_code == 200
    general_deck_id = general_deck.json()["id"]

    workspace_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags Workspace Deck", "workspace_id": "ws-tags-global"},
        headers=AUTH_HEADERS,
    )
    assert workspace_deck.status_code == 200
    workspace_deck_id = workspace_deck.json()["id"]

    created_general = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": general_deck_id,
            "front": "General card",
            "back": "Answer",
            "tags": ["general-only", "shared-tag"],
        },
        headers=AUTH_HEADERS,
    )
    assert created_general.status_code == 200

    created_workspace = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={
            "deck_id": workspace_deck_id,
            "front": "Workspace card",
            "back": "Answer",
            "tags": ["workspace-only", "shared-tag"],
        },
        headers=AUTH_HEADERS,
    )
    assert created_workspace.status_code == 200

    suggestions = client_with_flashcards_db.get("/api/v1/flashcards/tags", headers=AUTH_HEADERS)
    assert suggestions.status_code == 200
    payload = suggestions.json()

    counts = {item["tag"]: item["count"] for item in payload["items"]}
    assert counts["shared-tag"] == 2
    assert counts["general-only"] == 1
    assert counts["workspace-only"] == 1


def test_flashcard_tag_suggestions_orders_by_usage_then_alphabetically(
    client_with_flashcards_db: TestClient,
):
    deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags Ordering Deck"},
        headers=AUTH_HEADERS,
    )
    assert deck_response.status_code == 200
    deck_id = deck_response.json()["id"]

    inputs = [
        ("Card 1", ["zeta", "beta"]),
        ("Card 2", ["alpha", "zeta"]),
        ("Card 3", ["beta", "zeta"]),
        ("Card 4", ["alpha"]),
    ]
    for front, tags in inputs:
        response = client_with_flashcards_db.post(
            "/api/v1/flashcards",
            json={"deck_id": deck_id, "front": front, "back": "Answer", "tags": tags},
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200

    suggestions = client_with_flashcards_db.get("/api/v1/flashcards/tags", headers=AUTH_HEADERS)
    assert suggestions.status_code == 200
    items = suggestions.json()["items"]
    top_three = items[:3]

    assert [item["tag"] for item in top_three] == ["zeta", "alpha", "beta"]
    assert [item["count"] for item in top_three] == [3, 2, 2]


def test_flashcard_tag_suggestions_filters_with_trimmed_case_insensitive_q(
    client_with_flashcards_db: TestClient,
):
    deck_response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags Filter Deck"},
        headers=AUTH_HEADERS,
    )
    assert deck_response.status_code == 200
    deck_id = deck_response.json()["id"]

    cards = [
        ("Alpha one", ["AlphaTag"]),
        ("Alpha two", ["my-alpha-mix"]),
        ("Beta one", ["beta"]),
    ]
    for front, tags in cards:
        created = client_with_flashcards_db.post(
            "/api/v1/flashcards",
            json={"deck_id": deck_id, "front": front, "back": "Answer", "tags": tags},
            headers=AUTH_HEADERS,
        )
        assert created.status_code == 200

    suggestions = client_with_flashcards_db.get(
        "/api/v1/flashcards/tags",
        params={"q": "  AlPhA  "},
        headers=AUTH_HEADERS,
    )
    assert suggestions.status_code == 200
    payload = suggestions.json()

    assert payload["count"] == 2
    assert [item["tag"] for item in payload["items"]] == ["AlphaTag", "my-alpha-mix"]
    assert [item["count"] for item in payload["items"]] == [1, 1]


def test_flashcard_tag_suggestions_excludes_deleted_rows_and_keeps_deckless_cards(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    keep_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags Keep Deck"},
        headers=AUTH_HEADERS,
    )
    assert keep_deck.status_code == 200
    keep_deck_id = keep_deck.json()["id"]

    deleted_deck = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Tags Deleted Deck"},
        headers=AUTH_HEADERS,
    )
    assert deleted_deck.status_code == 200
    deleted_deck_id = deleted_deck.json()["id"]

    keep_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": keep_deck_id, "front": "Keep", "back": "Answer", "tags": ["keep-tag"]},
        headers=AUTH_HEADERS,
    )
    assert keep_card.status_code == 200

    deleted_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": keep_deck_id, "front": "Deleted card", "back": "Answer", "tags": ["deleted-card-tag"]},
        headers=AUTH_HEADERS,
    )
    assert deleted_card.status_code == 200
    deleted_card_uuid = deleted_card.json()["uuid"]

    deleted_keyword_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": keep_deck_id, "front": "Deleted keyword", "back": "Answer", "tags": ["deleted-keyword-tag"]},
        headers=AUTH_HEADERS,
    )
    assert deleted_keyword_card.status_code == 200

    deleted_deck_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": deleted_deck_id, "front": "Deleted deck", "back": "Answer", "tags": ["deleted-deck-tag"]},
        headers=AUTH_HEADERS,
    )
    assert deleted_deck_card.status_code == 200

    deckless_card = client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"front": "Deckless card", "back": "Answer", "tags": ["deckless-tag"]},
        headers=AUTH_HEADERS,
    )
    assert deckless_card.status_code == 200

    keyword_table = flashcards_db._map_table_for_backend("keywords")
    with flashcards_db.transaction() as conn:
        conn.execute("UPDATE flashcards SET deleted = 1 WHERE uuid = ?", (deleted_card_uuid,))
        conn.execute(
            f"UPDATE {keyword_table} SET deleted = 1 WHERE keyword = ?",
            ("deleted-keyword-tag",),
        )
        conn.execute("UPDATE decks SET deleted = 1 WHERE id = ?", (deleted_deck_id,))

    suggestions = client_with_flashcards_db.get("/api/v1/flashcards/tags", headers=AUTH_HEADERS)
    assert suggestions.status_code == 200
    tags = {item["tag"]: item["count"] for item in suggestions.json()["items"]}

    assert tags["keep-tag"] == 1
    assert tags["deckless-tag"] == 1
    assert "deleted-card-tag" not in tags
    assert "deleted-keyword-tag" not in tags
    assert "deleted-deck-tag" not in tags


def test_flashcard_tag_suggestions_route_precedence_over_alias_path(
    client_with_flashcards_db: TestClient,
):
    response = client_with_flashcards_db.get("/api/v1/flashcards/tags", headers=AUTH_HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"items", "count"}


def test_flashcard_tag_suggestions_rejects_limit_over_max(
    client_with_flashcards_db: TestClient,
):
    response = client_with_flashcards_db.get(
        "/api/v1/flashcards/tags",
        params={"limit": 101},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422
