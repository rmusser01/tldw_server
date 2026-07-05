"""Live streaming test for complete-v2 against a running mock OpenAI server.

Env-gated (external_api convention): set ``MOCK_OPENAI_BASE_URL`` to run.
Deterministic in-process streaming coverage lives in
``test_complete_v2_streaming_e2e_mock.py`` — this variant exercises the real
HTTP path to ``mock_openai_server`` when one is available.

Rewritten per audits/2026-07-04-test-suite-audit-round2.md (RA3): the old
version accepted ``status_code in (200, 502)`` — passing on server error —
and only checked that one line started with ``data: ``. With the env gate
satisfied, a 502 is a real failure and the stream content is asserted.
"""

import json
import os

import httpx
import pytest

from tldw_Server_API.app.core.AuthNZ.settings import get_settings

pytestmark = pytest.mark.external_api


@pytest.mark.asyncio
async def test_complete_v2_streaming_with_mock_openai(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    base_url = os.getenv("MOCK_OPENAI_BASE_URL")
    if not base_url:
        pytest.skip(
            "MOCK_OPENAI_BASE_URL not set; this is the live-server variant. "
            "Deterministic in-process streaming coverage that always runs lives in "
            "test_complete_v2_streaming_e2e_mock.py, so skipping here masks nothing."
        )

    # monkeypatch (not os.environ) so nothing leaks into later tests
    monkeypatch.setenv("OPENAI_API_BASE_URL", base_url)
    # mock_openai_server validates the key format: it must start with "sk-".
    # (The old test defaulted to "test-mock-key", which 401s at the mock and
    # surfaced as the 502 this test used to tolerate.) Test plugins may set
    # OPENAI_API_KEY to "" , so treat empty/non-sk keys as absent.
    existing_key = os.getenv("OPENAI_API_KEY")
    api_key = existing_key if existing_key and existing_key.startswith("sk-") else "sk-test-mock-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    from tldw_Server_API.app.main import app

    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Character + chat
        r = await client.get("/api/v1/characters/", headers=headers)
        assert r.status_code == 200
        character_id = r.json()[0]["id"]
        r = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
        assert r.status_code == 201
        chat_id = r.json()["id"]

        # Streamed completion — with the mock server up, anything but 200 is a failure
        url = f"/api/v1/chats/{chat_id}/complete-v2"
        data_lines: list[str] = []
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "append_user_message": "hello mock",
                "save_to_db": False,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200, (
                f"streaming completion failed against live mock server: {response.status_code}"
            )
            assert response.headers.get("content-type", "").lower().startswith("text/event-stream")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_lines.append(line[len("data: "):])

    assert data_lines, "no SSE data lines received"
    assert data_lines[-1].strip() == "[DONE]", "stream did not terminate with [DONE]"

    # accumulate deltas from the chunk payloads — the stream must carry content
    content = ""
    for payload in data_lines[:-1]:
        chunk = json.loads(payload)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
    assert content, "accumulated stream deltas were empty"
