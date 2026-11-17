"""Official ACE-Step 1.5 HTTP contract and provider boundary regressions."""

import json

import httpx
import pytest

from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest
from tldw_Server_API.app.core.Audio_Studio.providers import ace_step as ace_step_module
from tldw_Server_API.app.core.Audio_Studio.providers.ace_step import AceStepGenerationError, AceStepHttpAdapter
from tldw_Server_API.app.core.http_client import create_async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "failed", "foreign", "redirect", "html", "timeout", "wrong_task", "malformed_url"]
)
async def test_official_v15_generation_contract(monkeypatch, ace_step_dns, outcome):
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
        # The connection is pinned to the resolved address; Host carries the origin.
        assert (req.headers["host"], req.url.host) == (
            "ace.example.test",
            ace_step_dns["ace.example.test"],
        ), "Never contact a foreign output origin"  # nosec B101 - Provider boundary oracle.
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


# --- Egress (TASK-13950.11): requests go through core.http_client.afetch ---------------------


def _music_request() -> AudioGenerationRequest:
    return AudioGenerationRequest("music", "music", "Calm theme", None, {}, "track", "music-cue-1", "rev_1")


def _v15_server(seen: list[httpx.Request]):
    """MockTransport handler for one successful ACE-Step 1.5 task."""

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/release_task":
            return httpx.Response(200, json={"code": 200, "data": {"task_id": "task_1"}, "error": None})
        if req.url.path == "/query_result":
            rows = [{"task_id": "task_1", "status": 1, "result": json.dumps([{"file": "/v1/audio?path=a.wav"}])}]
            return httpx.Response(200, json={"code": 200, "data": rows, "error": None})
        if req.url.path == "/v1/audio":
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFF-local")
        return httpx.Response(404)

    return handler


def _mock_factory(handler):
    """A client_factory building core.http_client clients over an httpx.MockTransport."""
    return lambda *, timeout, **_kwargs: create_async_client(transport=httpx.MockTransport(handler), timeout=timeout)


def _configure_v15(monkeypatch, base_url: str) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", base_url)
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", base_url)
    monkeypatch.setenv("AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS", "true")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_VERSION", "1.5")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_TIMEOUT_SECONDS", "7")


@pytest.mark.unit
async def test_default_client_reaches_a_configured_local_server_with_the_same_requests(
    monkeypatch, ace_step_dns
):
    _configure_v15(monkeypatch, "http://10.0.0.5:8001")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_KEY", "private-service-key")
    seen: list[httpx.Request] = []

    def factory(**kwargs):
        return create_async_client(transport=httpx.MockTransport(_v15_server(seen)), **kwargs)

    monkeypatch.setattr(ace_step_module, "create_async_client", factory)
    result = await AceStepHttpAdapter().generate(_music_request())

    assert result.content_bytes == b"RIFF-local"
    assert [(req.method, str(req.url)) for req in seen] == [
        ("POST", "http://10.0.0.5:8001/release_task"),
        ("POST", "http://10.0.0.5:8001/query_result"),
        ("GET", "http://10.0.0.5:8001/v1/audio?path=a.wav"),
    ]
    assert all(req.headers["authorization"] == "Bearer private-service-key" for req in seen)
    assert seen[1].content == httpx.Request("POST", "http://x", json={"task_id_list": ["task_1"]}).content
    assert all(req.extensions["timeout"]["read"] == 7 for req in seen)


@pytest.mark.unit
@pytest.mark.parametrize("version", ["legacy", "1.5"])
async def test_allowlisted_host_resolving_to_cloud_metadata_is_refused_before_sending(
    monkeypatch, ace_step_dns, version
):
    _configure_v15(monkeypatch, "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_VERSION", version)
    ace_step_dns["ace.example.test"] = "169.254.169.254"
    seen: list[httpx.Request] = []
    adapter = AceStepHttpAdapter(client_factory=_mock_factory(_v15_server(seen)))

    with pytest.raises(AceStepGenerationError):
        await adapter.generate(_music_request())

    assert seen == []


@pytest.mark.unit
async def test_legacy_redirect_to_an_allowlisted_host_on_a_link_local_address_is_refused(
    monkeypatch, ace_step_dns
):
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.example.test")
    monkeypatch.setenv(
        "AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test,https://other.example.test"
    )
    monkeypatch.delenv("AUDIO_STUDIO_ACE_STEP_API_KEY", raising=False)
    ace_step_dns["other.example.test"] = "169.254.10.10"
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(307, headers={"location": "https://other.example.test/generate"})

    adapter = AceStepHttpAdapter(client_factory=_mock_factory(handler))
    with pytest.raises(AceStepGenerationError):
        await adapter.generate(_music_request())

    assert [req.headers["host"] for req in seen] == ["ace.example.test"]


@pytest.mark.unit
async def test_legacy_redirect_to_a_private_address_off_the_allowlist_is_refused(monkeypatch, ace_step_dns):
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "http://10.0.0.5:8001")
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "http://10.0.0.5:8001")
    monkeypatch.setenv("AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS", "true")
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(307, headers={"location": "http://10.0.0.9:8001/generate"})

    adapter = AceStepHttpAdapter(client_factory=_mock_factory(handler))
    with pytest.raises(ValueError, match="external_audio_redirect_not_allowlisted"):
        await adapter.generate(_music_request())

    assert [str(req.url) for req in seen] == ["http://10.0.0.5:8001/generate"]
