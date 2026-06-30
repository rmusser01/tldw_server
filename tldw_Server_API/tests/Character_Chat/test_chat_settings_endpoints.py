import shutil
import tempfile

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.integration


def _valid_attachment(
    *,
    run_id: str = "run_123",
    query: str = "Assess local-vs-web evidence",
    updated_at: str = "2026-03-08T20:00:00Z",
) -> dict:
    return {
        "run_id": run_id,
        "query": query,
        "question": query,
        "outline": [{"title": "Summary"}, {"title": "Risks"}],
        "key_claims": [{"text": "Claim A"}, {"text": "Claim B"}],
        "unresolved_questions": ["What primary source is still missing?"],
        "verification_summary": {"unsupported_claim_count": 1},
        "source_trust_summary": {"high_trust_count": 2},
        "research_url": f"/research?run={run_id}",
        "attached_at": "2026-03-08T19:55:00Z",
        "updatedAt": updated_at,
    }


async def _create_chat(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    characters = await client.get("/api/v1/characters/", headers=headers)
    assert characters.status_code == 200
    character_id = characters.json()[0]["id"]

    created = await client.post(
        "/api/v1/chats/",
        headers=headers,
        json={"character_id": character_id, "title": "Persisted attachment chat"},
    )
    assert created.status_code == 201
    return created.json()["id"]


def _create_research_run(
    *,
    owner_user_id: str = "1",
    status: str = "completed",
    phase: str = "completed",
    query: str = "Assess local-vs-web evidence",
) -> str:
    db = ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))
    session = db.create_session(
        owner_user_id=owner_user_id,
        query=query,
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        status=status,
        phase=phase,
    )
    if status == "completed":
        db.update_status(
            session.id,
            status="completed",
            completed_at="2026-03-08T20:10:00Z",
        )
    return session.id


@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_deep_research_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_deep_research_attachment_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            run_id = _create_research_run()
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-03-08T20:00:00Z",
                    "deepResearchAttachment": _valid_attachment(run_id=run_id),
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text

            get_response = await client.get(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
            )
            assert get_response.status_code == 200, get_response.text
            stored = get_response.json()["settings"]["deepResearchAttachment"]
            assert stored["run_id"] == run_id
            assert stored["verification_summary"]["unsupported_claim_count"] == 1
            assert stored["source_trust_summary"]["high_trust_count"] == 2
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_assistant_overlay(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_assistant_overlay_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-05-22T20:00:00Z",
                    "assistantOverlay": {
                        "kind": "persona",
                        "id": "persona-7",
                        "name": "Research Guide",
                        "avatar_url": "https://example.com/persona-7.png",
                        "system_prompt_snapshot": "Be concise and evidence-driven.",
                        "updatedAt": "2026-05-22T20:00:00Z",
                    },
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text
            stored_put = put_response.json()["settings"]["assistantOverlay"]
            assert stored_put["system_prompt_snapshot"] == "Be concise and evidence-driven."

            get_response = await client.get(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
            )
            assert get_response.status_code == 200, get_response.text
            stored = get_response.json()["settings"]["assistantOverlay"]
            assert stored["kind"] == "persona"
            assert stored["id"] == "persona-7"
            assert stored["name"] == "Research Guide"
            assert stored["avatar_url"] == "https://example.com/persona-7.png"
            assert stored["system_prompt_snapshot"] == "Be concise and evidence-driven."
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_normalizes_assistant_overlay_strings(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_assistant_overlay_trim_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-05-22T20:00:00Z",
                    "assistantOverlay": {
                        "kind": " Persona ",
                        "id": " persona-7 ",
                        "name": " Research Guide ",
                        "avatar_url": " https://example.com/persona-7.png ",
                        "system_prompt_snapshot": " Be concise and evidence-driven. ",
                        "updatedAt": " 2026-05-22T20:00:00Z ",
                    },
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text
            stored = put_response.json()["settings"]["assistantOverlay"]
            assert stored["kind"] == "persona"
            assert stored["id"] == "persona-7"
            assert stored["name"] == "Research Guide"
            assert stored["avatar_url"] == "https://example.com/persona-7.png"
            assert stored["system_prompt_snapshot"] == "Be concise and evidence-driven."
            assert stored["updatedAt"] == "2026-05-22T20:00:00Z"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_invalid_assistant_overlay_kind(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_assistant_overlay_kind_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-05-22T20:00:00Z",
                    "assistantOverlay": {
                        "kind": "regular",
                        "id": "overlay-1",
                        "name": "Overlay Name",
                        "updatedAt": "2026-05-22T20:00:00Z",
                    },
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 422, put_response.text
            assert "assistantOverlay.kind" in put_response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_field", ["id", "name"])
async def test_chat_settings_rejects_oversized_assistant_overlay_identity_fields(
    monkeypatch,
    oversized_field,
):
    tmpdir = tempfile.mkdtemp(prefix="chacha_assistant_overlay_identity_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            assistant_overlay = {
                "kind": "character",
                "id": "overlay-2",
                "name": "Overlay Name",
                "updatedAt": "2026-05-22T20:00:00Z",
            }
            assistant_overlay[oversized_field] = "x" * 20_001
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-05-22T20:00:00Z",
                    "assistantOverlay": assistant_overlay,
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 413, put_response.text
            assert f"assistantOverlay.{oversized_field}" in put_response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assistant_overlay_overrides", "drop_updated_at", "expected_status", "expected_detail"),
    [
        ({"system_prompt_snapshot": {"text": "bad"}}, False, 422, "assistantOverlay.system_prompt_snapshot"),
        ({"system_prompt_snapshot": "x" * 20_001}, False, 413, "assistantOverlay.system_prompt_snapshot"),
        ({}, True, 422, "assistantOverlay.updatedAt"),
    ],
)
async def test_chat_settings_rejects_invalid_assistant_overlay_prompt_snapshot(
    monkeypatch,
    assistant_overlay_overrides,
    drop_updated_at,
    expected_status,
    expected_detail,
):
    tmpdir = tempfile.mkdtemp(prefix="chacha_assistant_overlay_prompt_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            assistant_overlay = {
                "kind": "character",
                "id": "overlay-2",
                "name": "Overlay Name",
                "updatedAt": "2026-05-22T20:00:00Z",
                **assistant_overlay_overrides,
            }
            if drop_updated_at:
                assistant_overlay.pop("updatedAt", None)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-05-22T20:00:00Z",
                    "assistantOverlay": assistant_overlay,
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == expected_status, put_response.text
            assert expected_detail in put_response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_canonicalize_owned_completed_deep_research_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_deep_research_attachment_canonical_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        run_id = _create_research_run()
        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-03-08T20:00:00Z",
                    "deepResearchAttachment": {
                        **_valid_attachment(run_id=run_id),
                        "research_url": "https://example.com/not-research",
                    },
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text

            get_response = await client.get(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
            )
            assert get_response.status_code == 200, get_response.text
            stored = get_response.json()["settings"]["deepResearchAttachment"]
            assert stored["run_id"] == run_id
            assert stored["research_url"] == f"/research?run={run_id}"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_non_completed_deep_research_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_deep_research_attachment_incomplete_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        run_id = _create_research_run(status="running", phase="collecting")
        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-03-08T20:00:00Z",
                    "deepResearchAttachment": _valid_attachment(run_id=run_id),
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 422, put_response.text
            assert "must reference a completed deep research run" in put_response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_deep_research_attachment_history(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_deep_research_history_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            active_run_id = _create_research_run(query="Assess local-vs-web evidence")
            history_run_id_1 = _create_research_run(query="History 1")
            history_run_id_2 = _create_research_run(query="History 2")
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-03-08T20:00:00Z",
                    "deepResearchAttachment": _valid_attachment(run_id=active_run_id),
                    "deepResearchAttachmentHistory": [
                        _valid_attachment(
                            run_id=history_run_id_1,
                            query="History 1",
                            updated_at="2026-03-08T19:50:00Z",
                        ),
                        _valid_attachment(
                            run_id=history_run_id_2,
                            query="History 2",
                            updated_at="2026-03-08T19:40:00Z",
                        ),
                    ],
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text

            get_response = await client.get(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
            )
            assert get_response.status_code == 200, get_response.text
            stored = get_response.json()["settings"]["deepResearchAttachmentHistory"]
            assert [entry["run_id"] for entry in stored] == [history_run_id_1, history_run_id_2]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_roundtrip_persists_deep_research_pinned_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_deep_research_pinned_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            run_id = _create_research_run()
            chat_id = await _create_chat(client, headers)
            payload = {
                "settings": {
                    "schemaVersion": 2,
                    "updatedAt": "2026-03-08T20:00:00Z",
                    "deepResearchPinnedAttachment": _valid_attachment(run_id=run_id),
                }
            }

            put_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json=payload,
            )
            assert put_response.status_code == 200, put_response.text

            get_response = await client.get(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
            )
            assert get_response.status_code == 200, get_response.text
            stored = get_response.json()["settings"]["deepResearchPinnedAttachment"]
            assert stored["run_id"] == run_id
            assert stored["verification_summary"]["unsupported_claim_count"] == 1
            assert stored["source_trust_summary"]["high_trust_count"] == 2
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_unknown_keys_inside_deep_research_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_unknown_key_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            attachment = _valid_attachment()
            attachment["full_bundle"] = {"too": "much"}

            response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:00:00Z",
                        "deepResearchAttachment": attachment,
                    }
                },
            )
            assert response.status_code == 422, response.text
            assert "deepResearchAttachment" in response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_unknown_keys_inside_deep_research_pinned_attachment(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_pinned_unknown_key_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            pinned_attachment = _valid_attachment(run_id="run_pinned")
            pinned_attachment["full_bundle"] = {"too": "much"}

            response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:00:00Z",
                        "deepResearchPinnedAttachment": pinned_attachment,
                    }
                },
            )
            assert response.status_code == 422, response.text
            assert "deepResearchPinnedAttachment" in response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_unknown_keys_inside_deep_research_attachment_history_entry(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_history_unknown_key_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            history_entry = _valid_attachment(run_id="run_hist_bad")
            history_entry["full_bundle"] = {"too": "much"}

            response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:00:00Z",
                        "deepResearchAttachmentHistory": [history_entry],
                    }
                },
            )
            assert response.status_code == 422, response.text
            assert "deepResearchAttachmentHistory[0]" in response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_invalid_attachment_updated_at(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_bad_updated_at_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            attachment = _valid_attachment(updated_at="not-a-timestamp")

            response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:00:00Z",
                        "deepResearchAttachment": attachment,
                    }
                },
            )
            assert response.status_code == 422, response.text
            assert "deepResearchAttachment.updatedAt" in response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_rejects_attachment_history_longer_than_three_entries(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_history_too_long_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat_id = await _create_chat(client, headers)
            history = [
                _valid_attachment(run_id=f"run_hist_{index}", updated_at=f"2026-03-08T20:0{index}:00Z")
                for index in range(4)
            ]

            response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:00:00Z",
                        "deepResearchAttachmentHistory": history,
                    }
                },
            )
            assert response.status_code == 422, response.text
            assert "deepResearchAttachmentHistory" in response.json()["detail"]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_attachment_merge_prefers_newer_attachment_timestamp(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_merge_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            older_run_id = _create_research_run(query="Older attachment")
            newer_run_id = _create_research_run(query="Newer attachment snapshot")
            chat_id = await _create_chat(client, headers)

            baseline_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:10:00Z",
                        "greetingEnabled": True,
                        "deepResearchAttachment": _valid_attachment(
                            run_id=older_run_id,
                            query="Older attachment",
                            updated_at="2026-03-08T20:00:00Z",
                        ),
                    }
                },
            )
            assert baseline_response.status_code == 200, baseline_response.text

            merge_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:05:00Z",
                        "authorNote": "older top-level patch should not win",
                        "deepResearchAttachment": _valid_attachment(
                            run_id=newer_run_id,
                            query="Newer attachment snapshot",
                            updated_at="2026-03-08T20:20:00Z",
                        ),
                    }
                },
            )
            assert merge_response.status_code == 200, merge_response.text
            merged = merge_response.json()["settings"]
            assert merged["greetingEnabled"] is True
            assert merged.get("authorNote") == "older top-level patch should not win"
            assert merged["updatedAt"] == "2026-03-08T20:10:00Z"
            assert merged["deepResearchAttachment"]["run_id"] == newer_run_id
            assert merged["deepResearchAttachment"]["query"] == "Newer attachment snapshot"
            assert merged["deepResearchAttachment"]["updatedAt"] == "2026-03-08T20:20:00Z"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_attachment_history_merge_prefers_newer_entries_and_excludes_active_run(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_history_merge_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            active_old_run_id = _create_research_run(query="Older active attachment")
            active_new_run_id = _create_research_run(query="Newer active attachment")
            history_run_a = _create_research_run(query="History A")
            history_run_shared = _create_research_run(query="History Shared Newer")
            history_run_b = _create_research_run(query="History B")
            chat_id = await _create_chat(client, headers)

            baseline_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:10:00Z",
                        "greetingEnabled": True,
                        "deepResearchAttachment": _valid_attachment(
                            run_id=active_old_run_id,
                            query="Older active attachment",
                            updated_at="2026-03-08T20:10:00Z",
                        ),
                        "deepResearchAttachmentHistory": [
                            _valid_attachment(
                                run_id=history_run_a,
                                query="History A",
                                updated_at="2026-03-08T20:00:00Z",
                            ),
                            _valid_attachment(
                                run_id=history_run_shared,
                                query="History Shared Older",
                                updated_at="2026-03-08T20:05:00Z",
                            ),
                        ],
                    }
                },
            )
            assert baseline_response.status_code == 200, baseline_response.text

            merge_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:20:00Z",
                        "deepResearchAttachment": _valid_attachment(
                            run_id=active_new_run_id,
                            query="Newer active attachment",
                            updated_at="2026-03-08T20:20:00Z",
                        ),
                        "deepResearchAttachmentHistory": [
                            _valid_attachment(
                                run_id=active_new_run_id,
                                query="Should be excluded",
                                updated_at="2026-03-08T20:20:00Z",
                            ),
                            _valid_attachment(
                                run_id=history_run_shared,
                                query="History Shared Newer",
                                updated_at="2026-03-08T20:30:00Z",
                            ),
                            _valid_attachment(
                                run_id=history_run_b,
                                query="History B",
                                updated_at="2026-03-08T20:15:00Z",
                            ),
                        ],
                    }
                },
            )
            assert merge_response.status_code == 200, merge_response.text
            merged = merge_response.json()["settings"]
            assert merged["greetingEnabled"] is True
            assert merged["deepResearchAttachment"]["run_id"] == active_new_run_id
            assert [entry["run_id"] for entry in merged["deepResearchAttachmentHistory"]] == [
                history_run_shared,
                history_run_b,
                history_run_a,
            ]
            assert merged["deepResearchAttachmentHistory"][0]["query"] == "History Shared Newer"
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chat_settings_pinned_attachment_merge_prefers_newer_entry_and_excludes_pinned_run_from_history(monkeypatch):
    tmpdir = tempfile.mkdtemp(prefix="chacha_attachment_pinned_merge_")
    monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
    reset_settings()
    try:
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pinned_run_id = _create_research_run(query="Newer pinned attachment")
            history_old_run_id = _create_research_run(query="History old")
            history_new_run_id = _create_research_run(query="History new")
            chat_id = await _create_chat(client, headers)

            baseline_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:10:00Z",
                        "deepResearchPinnedAttachment": _valid_attachment(
                            run_id=pinned_run_id,
                            query="Older pinned attachment",
                            updated_at="2026-03-08T20:00:00Z",
                        ),
                        "deepResearchAttachmentHistory": [
                            _valid_attachment(
                                run_id=history_old_run_id,
                                query="History old",
                                updated_at="2026-03-08T19:59:00Z",
                            ),
                        ],
                    }
                },
            )
            assert baseline_response.status_code == 200, baseline_response.text

            merge_response = await client.put(
                f"/api/v1/chats/{chat_id}/settings",
                headers=headers,
                json={
                    "settings": {
                        "schemaVersion": 2,
                        "updatedAt": "2026-03-08T20:12:00Z",
                        "deepResearchPinnedAttachment": _valid_attachment(
                            run_id=pinned_run_id,
                            query="Newer pinned attachment",
                            updated_at="2026-03-08T20:11:00Z",
                        ),
                        "deepResearchAttachmentHistory": [
                            _valid_attachment(
                                run_id=pinned_run_id,
                                query="Pinned duplicate in history",
                                updated_at="2026-03-08T20:10:30Z",
                            ),
                            _valid_attachment(
                                run_id=history_new_run_id,
                                query="History new",
                                updated_at="2026-03-08T20:10:00Z",
                            ),
                        ],
                    }
                },
            )
            assert merge_response.status_code == 200, merge_response.text

            merged = merge_response.json()["settings"]
            assert merged["deepResearchPinnedAttachment"]["run_id"] == pinned_run_id
            assert merged["deepResearchPinnedAttachment"]["query"] == "Newer pinned attachment"
            assert [entry["run_id"] for entry in merged["deepResearchAttachmentHistory"]] == [
                history_new_run_id,
                history_old_run_id,
            ]
    finally:
        reset_settings()
        shutil.rmtree(tmpdir, ignore_errors=True)
