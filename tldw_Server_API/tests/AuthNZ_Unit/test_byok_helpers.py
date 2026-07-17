from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_validate_credential_fields_default_allowlist():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields

    fields = {"org_id": "org-123", "project_id": "proj-456"}
    cleaned = validate_credential_fields("unknown-provider", fields)
    assert cleaned == fields

    with pytest.raises(ValueError):
        validate_credential_fields("unknown-provider", {"api_key": "nope"})


def test_validate_credential_fields_required_policy(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_credential_fields
    from tldw_Server_API.app.core.LLM_Calls import provider_metadata

    monkeypatch.setitem(
        provider_metadata.BYOK_CREDENTIAL_FIELDS,
        "test-provider",
        {"allowed": {"org_id"}, "required": {"org_id"}},
    )

    cleaned = validate_credential_fields("test-provider", {"org_id": "org-789"})
    assert cleaned == {"org_id": "org-789"}

    with pytest.raises(ValueError):
        validate_credential_fields("test-provider", {})


def test_trusted_base_url_request_rejects_legacy_boolean_only_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": [],
                "is_admin": True,
                "is_superuser": True,
            },
        )
        is False
    )


def test_trusted_base_url_request_accepts_permission_claims_from_legacy_user_shape():
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_request

    assert (
        is_trusted_base_url_request(
            None,
            principal=None,
            user={
                "role": "user",
                "roles": ["user"],
                "permissions": ["system.configure"],
                "is_admin": False,
                "is_superuser": False,
            },
        )
        is True
    )


def test_gateway_specs_augment_byok_allowlist_without_discovery(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            BYOK_ALLOWED_PROVIDERS=[
                "openai",
                "gateway:admin-only",
                "gateway:disabled",
                "gateway:orphaned",
            ]
        ),
    )
    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "openrouter": SimpleNamespace(enabled=True, allow_user_api_key=True),
            "gateway:voice-lab": SimpleNamespace(enabled=True, allow_user_api_key=True),
            "gateway:admin-only": SimpleNamespace(enabled=True, allow_user_api_key=False),
            "gateway:disabled": SimpleNamespace(enabled=False, allow_user_api_key=True),
        },
        raising=False,
    )

    assert byok_helpers.resolve_byok_allowlist() == {
        "openai",
        "openrouter",
        "gateway:voice-lab",
    }


def test_gateway_config_failure_preserves_static_allowlist_and_fails_dynamic_closed(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    log_calls: list[tuple[str, tuple[object, ...]]] = []

    class _Logger:
        def warning(self, message: str, *args: object) -> None:
            log_calls.append((message, args))

    def _fail_gateway_specs():
        raise RuntimeError("sensitive gateway config detail")

    monkeypatch.setattr(
        byok_helpers,
        "get_settings",
        lambda: SimpleNamespace(
            BYOK_ALLOWED_PROVIDERS=[
                "openai",
                "anthropic",
                "gateway:voice-lab",
            ]
        ),
    )
    monkeypatch.setattr(byok_helpers, "logger", _Logger(), raising=False)
    monkeypatch.setattr(byok_helpers, "get_byok_gateway_specs", _fail_gateway_specs)

    assert byok_helpers.resolve_byok_allowlist() == {"openai", "anthropic"}
    assert byok_helpers.is_provider_allowlisted("openai") is True
    assert byok_helpers.is_provider_allowlisted("gateway:voice-lab") is False
    assert log_calls
    assert all(call[1] == ("RuntimeError",) for call in log_calls)
    assert "sensitive gateway config detail" not in repr(log_calls)


@pytest.mark.parametrize(
    "credential_fields",
    [
        {"base_url": "https://evil.example/v1"},
        {"url": "https://evil.example/v1"},
        {"headers": {"Authorization": "Bearer attacker"}},
        {"auth_scheme": "basic"},
        {"org_id": "metadata-authority"},
    ],
)
def test_gateway_credential_fields_reject_endpoint_authority_for_every_principal(
    monkeypatch,
    credential_fields,
):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "gateway:voice-lab": SimpleNamespace(
                enabled=True,
                allow_user_api_key=True,
            )
        },
        raising=False,
    )

    with pytest.raises(ValueError, match="Unsupported credential field"):
        byok_helpers.validate_credential_fields(
            "gateway:voice-lab",
            credential_fields,
            allow_base_url=True,
        )

    assert byok_helpers.validate_credential_fields(
        "gateway:voice-lab",
        {},
        allow_base_url=True,
    ) == {}


def test_openrouter_keeps_general_credential_policy_when_used_by_tts(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_helpers

    monkeypatch.setattr(
        byok_helpers,
        "get_byok_gateway_specs",
        lambda: {
            "openrouter": SimpleNamespace(enabled=True, allow_user_api_key=True)
        },
        raising=False,
    )

    assert byok_helpers.validate_credential_fields(
        "openrouter",
        {"org_id": "org-general", "project_id": "project-general"},
    ) == {"org_id": "org-general", "project_id": "project-general"}


def _probe_spec(*, enabled=True, discovery_enabled=True, models_path="models"):
    return SimpleNamespace(
        backend_id="gateway:voice-lab",
        enabled=enabled,
        base_url="https://voice.example/v1/",
        models_path=models_path,
        headers=(("X-Admin", "configured"),),
        discovery_query=(("output_modalities", "speech"),),
        discovery=SimpleNamespace(
            enabled=discovery_enabled,
            timeout_seconds=2.5,
        ),
    )


@pytest.mark.asyncio
async def test_gateway_credential_probe_verified_uses_bounded_configured_discovery(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import byok_testing

    seen = {}

    async def _fake_afetch_json(**kwargs):
        seen.update(kwargs)
        await kwargs["on_response"](200, {"Content-Type": "application/json"})
        return {"data": [{"id": "Vendor/Voice"}]}

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)
    result = await byok_testing.probe_gateway_credentials(
        spec=_probe_spec(),
        api_key="user-key",
    )

    assert result == "verified"
    assert seen["url"] == "https://voice.example/v1/models"
    assert seen["headers"] == {
        "X-Admin": "configured",
        "Authorization": "Bearer user-key",
    }
    assert seen["allow_redirects"] is False
    assert seen["max_bytes"] == 1_048_576


@pytest.mark.asyncio
async def test_gateway_credential_probe_rejects_only_definitive_auth_status(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_testing

    async def _fake_afetch_json(**kwargs):
        await kwargs["on_response"](401, {"Content-Type": "application/json"})
        return {"error": {"message": "secret upstream detail"}}

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)

    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="rejected-key",
        )
        == "rejected"
    )


@pytest.mark.asyncio
async def test_gateway_credential_probe_unavailable_is_stored_unverified(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.exceptions import NetworkError

    async def _fake_afetch_json(**_kwargs):
        raise NetworkError("upstream details must not escape")

    monkeypatch.setattr(byok_testing, "afetch_json", _fake_afetch_json, raising=False)

    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="unverified-key",
        )
        == "stored-unverified"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_length", [None, "2"])
async def test_gateway_credential_probe_stops_oversized_chunked_discovery_early(
    monkeypatch,
    declared_length,
):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.http_client import afetch_json, create_async_client

    chunks = [b'{"data":["', b"x" * 700_000, b"y" * 700_000, b'"]}']

    class _CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0
            self.close_count = 0

        async def __aiter__(self):
            for chunk in chunks:
                self.yielded += 1
                yield chunk

        async def aclose(self) -> None:
            self.close_count += 1

    stream = _CountingStream()
    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers=headers,
            stream=stream,
        )

    client = create_async_client(transport=httpx.MockTransport(handler))

    async def _central_fetch(**kwargs):
        return await afetch_json(client=client, **kwargs)

    monkeypatch.setattr(byok_testing, "afetch_json", _central_fetch)
    try:
        result = await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="user-key",
        )
    finally:
        await client.aclose()

    assert result == "stored-unverified"
    assert stream.yielded < len(chunks)
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_gateway_credential_probe_classifies_rejection_before_reading_body(
    monkeypatch,
):
    import httpx

    from tldw_Server_API.app.core.AuthNZ import byok_testing
    from tldw_Server_API.app.core.http_client import afetch_json, create_async_client

    class _CountingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.yielded = 0
            self.close_count = 0

        async def __aiter__(self):
            self.yielded += 1
            yield b'{"error":"sensitive upstream body"}'

        async def aclose(self) -> None:
            self.close_count += 1

    stream = _CountingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            request=request,
            headers={"Content-Type": "application/json"},
            stream=stream,
        )

    client = create_async_client(transport=httpx.MockTransport(handler))

    async def _central_fetch(**kwargs):
        return await afetch_json(client=client, **kwargs)

    monkeypatch.setattr(byok_testing, "afetch_json", _central_fetch)
    try:
        result = await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(),
            api_key="rejected-key",
        )
    finally:
        await client.aclose()

    assert result == "rejected"
    assert stream.yielded == 0
    assert stream.close_count == 1
    assert (
        await byok_testing.probe_gateway_credentials(
            spec=_probe_spec(discovery_enabled=False),
            api_key="unverified-key",
        )
        == "stored-unverified"
    )
