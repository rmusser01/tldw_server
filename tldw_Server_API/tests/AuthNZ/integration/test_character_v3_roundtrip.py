"""Character exports round-trip through real cookie-authenticated file import."""

import json

import pytest
import pytest_asyncio

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint
from tldw_Server_API.app.core.Character_Chat.ccv3_parser import validate_v3_card
from tldw_Server_API.app.core.Character_Chat.modules.character_io import extract_json_from_image_file
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
CHARACTERS_PATH = "/api/v1/characters"
GENERATION = {"tldw": {"generation": {"temperature": 0.2, "top_p": 0.9}}}
OPTIONAL_TEXT = (
    "personality", "scenario", "mes_example", "creator_notes", "system_prompt",
    "post_history_instructions", "creator", "character_version",
)


@pytest_asyncio.fixture
async def character_cookie_client(single_user_cookie_client, monkeypatch, tmp_path):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "character-users"))
    if not any(route.path == CHARACTERS_PATH + "/import" for route in iter_served_routes(client.app.routes)):
        client.app.include_router(characters_endpoint.router, prefix=CHARACTERS_PATH)
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    try:
        yield client, api_key
    finally:
        await chacha_deps.shutdown_chacha_resources()


def _headers(client, api_key):
    assert _mint(client, api_key).status_code == 200  # nosec B101
    return {"X-CSRF-Token": client.cookies["csrf_token"]}


@pytest.mark.parametrize("export_format", ["v3", "v2", "png"])
def test_default_character_export_reimports_with_extensions(character_cookie_client, export_format):
    """Default V3 plus established V2/PNG keep their format and extension data."""
    client, api_key = character_cookie_client
    headers = _headers(client, api_key)
    created = client.post(
        CHARACTERS_PATH + "/",
        json={"name": "Roundtrip original", "description": "A test bot", "first_message": "Hi!", "extensions": GENERATION},
        headers=headers,
    )
    assert created.status_code == 201, created.text  # nosec B101
    original_id = created.json()["id"]
    params = {} if export_format == "v3" else {"format": export_format}
    exported = client.get(f"{CHARACTERS_PATH}/{original_id}/export", params=params)
    assert exported.status_code == 200, exported.text  # nosec B101
    if export_format == "png":
        card = json.loads(extract_json_from_image_file(exported.content))
    else:
        card = exported.json()
    expected_spec = "chara_card_v3" if export_format == "v3" else "chara_card_v2"
    assert card["spec"] == expected_spec  # nosec B101
    if export_format == "v3":
        assert validate_v3_card(card) == (True, [])  # nosec B101
        assert all(isinstance(card["data"][field], str) for field in OPTIONAL_TEXT)  # nosec B101
    assert card["data"]["extensions"] == GENERATION  # nosec B101
    card["data"]["name"] = "Roundtrip imported"
    content = json.dumps(card)
    if export_format == "png":
        upload = ("character.png", characters_endpoint._encode_png_with_chara_metadata(exported.content, content), "image/png")
    else:
        upload = ("character.json", content.encode(), "application/json")

    imported = client.post(CHARACTERS_PATH + "/import", files={"character_file": upload}, headers=headers)

    assert imported.status_code == 201, imported.text  # nosec B101
    imported_id = imported.json()["id"]
    assert imported_id != original_id  # nosec B101
    stored = client.get(f"{CHARACTERS_PATH}/{imported_id}").json()
    assert stored["name"] == "Roundtrip imported"  # nosec B101
    assert stored["extensions"] == GENERATION  # nosec B101
    assert stored["first_message"] == "Hi!"  # nosec B101


@pytest.mark.asyncio
async def test_legacy_v3_optional_null_export_import_keeps_csrf_and_owner(character_cookie_client):
    """Previously produced nullable V3 exports use the existing per-user DB path."""
    client, api_key = character_cookie_client
    headers = _headers(client, api_key)
    data = dict.fromkeys(OPTIONAL_TEXT)
    data.update({
        "name": "Legacy V3 imported", "description": "A test bot", "first_mes": "Hi!",
        "alternate_greetings": [], "tags": [], "extensions": GENERATION,
    })
    card = {"spec": "chara_card_v3", "spec_version": "3.0", "data": data}
    upload = ("legacy-character.json", json.dumps(card).encode(), "application/json")
    rejected = client.post(CHARACTERS_PATH + "/import", files={"character_file": upload})
    assert rejected.status_code == 403, rejected.text  # nosec B101

    imported = client.post(CHARACTERS_PATH + "/import", files={"character_file": upload}, headers=headers)

    assert imported.status_code == 201, imported.text  # nosec B101
    owner = client.get("/api/v1/auth/me").json()["id"]
    owned_db = await chacha_deps.get_chacha_db_for_user_id(owner)
    assert str(owned_db.owner_user_id) == str(owner)  # nosec B101
    saved = owned_db.get_character_card_by_id(imported.json()["id"])
    assert saved["extensions"] == GENERATION  # nosec B101
    assert saved["name"] == "Legacy V3 imported"  # nosec B101
    other_db = await chacha_deps.get_chacha_db_for_user_id(owner + 1)
    assert other_db.get_character_card_by_name("Legacy V3 imported") is None  # nosec B101
