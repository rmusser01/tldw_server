"""Official ACE-Step 1.5 HTTP contract and provider boundary regressions."""

import json

import httpx
import pytest

from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest
from tldw_Server_API.app.core.Audio_Studio.providers.ace_step import AceStepHttpAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "failed", "foreign", "redirect", "html", "timeout", "wrong_task", "malformed_url"]
)
async def test_official_v15_generation_contract(monkeypatch, outcome):
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.example.test")
    monkeypatch.setenv(
        "AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test,https://other.example.test"
    )
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_VERSION", "1.5")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_KEY", "private-service-key")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS", "1")
    request = AudioGenerationRequest(
        "music",
        "music",
        "Calm observatory theme",
        None,
        {"lyrics": "", "style": "ambient", "duration": 15},
        "track",
        "music-cue-1",
        "rev_1",
    )
    seen = []
    polls = 0

    async def handler(req):
        nonlocal polls
        seen.append((req.method, req.url.path))
        assert req.url.host == "ace.example.test", "Never contact a foreign output origin"  # nosec B101 - Provider boundary oracle.
        assert req.headers["authorization"] == "Bearer private-service-key"  # nosec B101 - Runtime-only authentication oracle.
        if req.url.path == "/release_task":
            body = json.loads(req.content)
            assert body == {
                "prompt": "Calm observatory theme, ambient",
                "lyrics": "",
                "audio_duration": 15,
                "audio_format": "wav",
                "batch_size": 1,
                "thinking": False,
            }  # nosec B101 - Exact supported official API mapping.
            return httpx.Response(200, json={"code": 200, "data": {"task_id": "task_1"}, "error": None})
        if req.url.path == "/query_result":
            assert json.loads(req.content) == {"task_id_list": ["task_1"]}  # nosec B101 - Poll the admitted task only.
            polls += 1
            status = (
                0 if outcome == "timeout" or (outcome == "success" and polls == 1) else 2 if outcome == "failed" else 1
            )
            file = (
                "https://other.example.test/v1/audio?path=private"
                if outcome == "foreign"
                else "/v1/audio?path=owned.wav"
            )
            if outcome == "malformed_url":
                file = "https://ace.example.test:bad/v1/audio"
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": [
                        {
                            "task_id": "wrong" if outcome == "wrong_task" else "task_1",
                            "status": status,
                            "result": json.dumps([{"file": file}]),
                        }
                    ],
                    "error": None,
                },
            )
        if req.url.path == "/v1/audio":
            if outcome == "redirect":
                return httpx.Response(302, headers={"location": "https://other.example.test/v1/audio"})
            return httpx.Response(
                200,
                headers={"content-type": "text/html" if outcome == "html" else "audio/wav"},
                content=b"RIFF-owned-audio",
            )
        return httpx.Response(404)

    adapter = AceStepHttpAdapter(
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    )
    if outcome == "success":
        result = await adapter.generate(request)
        assert (result.mime_type, result.content_bytes, result.metadata["task_id"]) == (
            "audio/wav",
            b"RIFF-owned-audio",
            "task_1",
        )  # nosec B101 - Real adapter output and provenance.
    else:
        with pytest.raises(Exception) as failure:
            await adapter.generate(request)
        assert getattr(failure.value, "retryable", None) is False, (
            "Unknown or failed accepted task must not automatically resubmit"
        )  # nosec B101 - Duplicate-generation prevention.
    assert seen.count(("POST", "/release_task")) == 1  # nosec B101 - At most one external generation admission.
