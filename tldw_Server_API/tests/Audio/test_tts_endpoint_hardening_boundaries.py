"""Security and concurrency regressions for direct TTS endpoints."""

from __future__ import annotations

import asyncio
import json
import traceback
from types import MappingProxyType, SimpleNamespace, TracebackType
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_core
import tldw_Server_API.app.api.v1.endpoints.audio.audio_tts as audio_tts
import tldw_Server_API.app.api.v1.utils.exception_handlers as api_exception_handlers
import tldw_Server_API.app.core.Audio.tts_service as tts_core
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.exceptions import TTSPublicHTTPException
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSInvalidVoiceReferenceError,
    TTSNetworkError,
    TTSValidationError,
)

_SECRET = "tts-provider-secret-sentinel"
_BYOK_BASE_URL = "https://user:secret@tts-job.invalid/v1?token=private"


class _UsageLog:
    def log_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Resolution:
    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.api_key = f"runtime-key-{name}"
        self.auth_source = "api_key"
        self.touch_calls = 0

    async def touch_last_used(self) -> None:
        self.touch_calls += 1


def _make_request(
    path: str = "/api/v1/audio/speech",
    *,
    case_id: str = "default",
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"x-test-case", case_id.encode("ascii"))],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


def _speech_request(
    *,
    text: str = "hello world",
    model: str = "tts-1",
    stream: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> OpenAISpeechRequest:
    return OpenAISpeechRequest(
        input=text,
        model=model,
        voice="alloy",
        stream=stream,
        response_format="mp3",
        extra_params=extra_params,
    )


def _patch_shim(monkeypatch: pytest.MonkeyPatch, resolver: Any) -> None:
    async def _unused_save(**_kwargs: Any) -> dict[str, Any]:
        return {"id": None}

    mapping = {
        "_sanitize_speech_request": tts_core._sanitize_speech_request,
        "_resolve_tts_byok": resolver,
        "save_and_register_tts_audio": _unused_save,
    }
    monkeypatch.setattr(audio_tts, "_audio_shim_attr", mapping.__getitem__)


def _exception_graph_contains(
    exc: BaseException,
    *,
    target: object | None = None,
    text: str | None = None,
) -> bool:
    """Inspect exception state and production traceback locals for a sentinel."""

    pending: list[object] = [exc]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if target is not None and value is target:
            return True
        if text is not None and isinstance(value, str) and text in value:
            return True
        if isinstance(value, BaseException):
            pending.extend(value.args)
            pending.extend(value.__dict__.values())
            if value.__cause__ is not None:
                pending.append(value.__cause__)
            if value.__context__ is not None:
                pending.append(value.__context__)
            if value.__traceback__ is not None:
                pending.append(value.__traceback__)
        elif isinstance(value, TracebackType):
            module = str(value.tb_frame.f_globals.get("__name__", ""))
            if module.startswith("tldw_Server_API.app."):
                pending.extend(value.tb_frame.f_locals.values())
            if value.tb_next is not None:
                pending.append(value.tb_next)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
    return False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_job_persists_only_active_credential_authorization_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        return (
            7,
            {"api_key": _SECRET},
            SimpleNamespace(
                api_key=_SECRET,
                source="user",
                credential_fields={"base_url": _BYOK_BASE_URL},
            ),
        )

    class _JobManager:
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"id": 41, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
            is_admin=True,
            roles=["admin"],
            permissions=["system.configure"],
            org_ids=[17, 18],
            team_ids=[27, 28],
            active_org_id=17,
            active_team_id=27,
        )
    )

    response = await audio_tts.create_speech_job(
        _speech_request(stream=True),
        request,
        current_user=SimpleNamespace(id="7"),
        jm=_JobManager(),
    )

    assert response.status_code == 200
    credential_scope = captured["payload"]["credential_scope"]
    assert set(credential_scope) == {
        "owner_user_id",
        "team_ids",
        "org_ids",
        "credential_source",
        "trusted_base_url_requested",
    }
    assert credential_scope == {
        "owner_user_id": 7,
        "team_ids": [],
        "org_ids": [],
        "credential_source": "user",
        "trusted_base_url_requested": True,
    }
    assert captured["payload"]["speech_request"]["stream"] is False
    serialized = json.dumps(captured, sort_keys=True, default=str)
    assert _SECRET not in serialized
    assert _BYOK_BASE_URL not in serialized
    assert "api_key" not in serialized
    assert "roles" not in serialized
    assert "permissions" not in serialized


@pytest.mark.unit
@pytest.mark.asyncio
async def test_trusted_tts_job_without_custom_byok_base_url_persists_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        return (
            7,
            {"api_key": _SECRET},
            SimpleNamespace(
                api_key=_SECRET,
                source="user",
                credential_fields={},
            ),
        )

    class _JobManager:
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"id": 42, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
            roles=["admin"],
            permissions=["system.configure"],
        )
    )

    response = await audio_tts.create_speech_job(
        _speech_request(),
        request,
        current_user=SimpleNamespace(id="7"),
        jm=_JobManager(),
    )

    assert response.status_code == 200
    assert captured["payload"]["credential_scope"] == {
        "owner_user_id": 7,
        "team_ids": [],
        "org_ids": [],
        "credential_source": "user",
        "trusted_base_url_requested": False,
    }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_source", "expected_team_ids", "expected_org_ids", "expected_trusted_base_url"),
    [
        ("team", [27], [], True),
        ("org", [], [17], True),
        ("server_default", [], [], False),
        ("none", [], [], False),
    ],
)
async def test_tts_job_projects_only_the_resolved_credential_source_scope(
    monkeypatch: pytest.MonkeyPatch,
    credential_source: str,
    expected_team_ids: list[int],
    expected_org_ids: list[int],
    expected_trusted_base_url: bool,
) -> None:
    captured: dict[str, Any] = {}

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        return (
            7,
            {"api_key": _SECRET},
            SimpleNamespace(
                api_key=_SECRET,
                source=credential_source,
                credential_fields={"base_url": _BYOK_BASE_URL},
            ),
        )

    class _JobManager:
        def create_job(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"id": 43, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
            roles=["admin"],
            permissions=["system.configure"],
            org_ids=[17, 18],
            team_ids=[27, 28],
            active_org_id=17,
            active_team_id=27,
        )
    )

    response = await audio_tts.create_speech_job(
        _speech_request(),
        request,
        current_user=SimpleNamespace(id="7"),
        jm=_JobManager(),
    )

    assert response.status_code == 200
    assert captured["payload"]["credential_scope"] == {
        "owner_user_id": 7,
        "team_ids": expected_team_ids,
        "org_ids": expected_org_ids,
        "credential_source": credential_source,
        "trusted_base_url_requested": expected_trusted_base_url,
    }
    serialized = json.dumps(captured, sort_keys=True, default=str)
    assert _SECRET not in serialized
    assert _BYOK_BASE_URL not in serialized


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("credential_source", ["team", "org"])
async def test_tts_job_rejects_shared_source_without_matching_active_scope(
    monkeypatch: pytest.MonkeyPatch,
    credential_source: str,
) -> None:
    queued = False

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        return (
            7,
            {"api_key": _SECRET},
            SimpleNamespace(
                api_key=_SECRET,
                source=credential_source,
                credential_fields={},
            ),
        )

    class _JobManager:
        def create_job(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal queued
            queued = True
            return {"id": 44, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            _speech_request(),
            request,
            current_user=SimpleNamespace(id="7"),
            jm=_JobManager(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"
    assert queued is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_job_rejects_resolver_owner_mismatch_before_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = False

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        return (
            8,
            {"api_key": _SECRET},
            SimpleNamespace(
                api_key=_SECRET,
                source="user",
                credential_fields={},
            ),
        )

    class _JobManager:
        def create_job(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal queued
            queued = True
            return {"id": 45, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            _speech_request(),
            request,
            current_user=SimpleNamespace(id="7"),
            jm=_JobManager(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"
    assert queued is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_job_rejects_mismatched_principal_owner_before_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = False
    queued = False

    async def _resolver(**_kwargs: Any) -> tuple[None, None, None]:
        nonlocal resolved
        resolved = True
        return None, None, None

    class _JobManager:
        def create_job(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal queued
            queued = True
            return {"id": 41, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            _speech_request(),
            request,
            current_user=SimpleNamespace(id="8"),
            jm=_JobManager(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"
    assert resolved is False
    assert queued is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tts_job_rejects_service_principal_before_resolution_or_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = False
    queued = False

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, str], Any]:
        nonlocal resolved
        resolved = True
        return (
            7,
            {"api_key": _SECRET},
            SimpleNamespace(source="user", credential_fields={}),
        )

    class _JobManager:
        def create_job(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal queued
            queued = True
            return {"id": 46, "status": "queued"}

    _patch_shim(monkeypatch, _resolver)
    request = _make_request("/api/v1/audio/speech/jobs")
    request.state.auth = AuthContext(
        principal=AuthPrincipal(
            kind="service",
            user_id=7,
            subject="service:tts-orchestrator",
            permissions=["system.configure"],
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            _speech_request(),
            request,
            current_user=SimpleNamespace(id="7"),
            jm=_JobManager(),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"
    assert resolved is False
    assert queued is False


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["speech", "metadata"])
async def test_direct_tts_rejects_nested_credentials_before_resolution_or_history(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    resolved = False
    history_calls: list[dict[str, Any]] = []

    async def _unexpected_resolver(**_kwargs: Any) -> tuple[None, None, None]:
        nonlocal resolved
        resolved = True
        return None, None, None

    class _UnreachableService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("credential-bearing request reached dispatch")

    _patch_shim(monkeypatch, _unexpected_resolver)
    request_data = _speech_request(
        extra_params={"nested": {"api_key": _SECRET}},
    )

    with pytest.raises(HTTPException) as exc_info:
        if endpoint == "speech":
            await audio_tts.create_speech(
                request_data,
                _make_request(),
                tts_service=_UnreachableService(),
                current_user=SimpleNamespace(id=1),
                media_db=SimpleNamespace(
                    create_tts_history_entry=lambda **kwargs: history_calls.append(kwargs)
                ),
                usage_log=_UsageLog(),
            )
        else:
            await audio_tts.create_speech_metadata(
                request_data,
                _make_request("/api/v1/audio/speech/metadata"),
                tts_service=_UnreachableService(),
                current_user=SimpleNamespace(id=1),
                usage_log=_UsageLog(),
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error_code"] == "credential_fields_not_allowed"
    assert _SECRET not in repr(exc_info.value)
    assert resolved is False
    assert history_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_model_cannot_reach_credentials_or_cached_openai_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = False
    dispatched = False

    async def _unexpected_resolver(**_kwargs: Any) -> tuple[None, None, None]:
        nonlocal resolved
        resolved = True
        return None, None, None

    class _CachedOpenAIService:
        cached_openai_adapter = object()

        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal dispatched
            dispatched = True
            raise AssertionError("unknown model reached cached OpenAI adapter")

    _patch_shim(monkeypatch, _unexpected_resolver)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech(
            _speech_request(model="unknown-admin-disabled-model"),
            _make_request(),
            tts_service=_CachedOpenAIService(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["message"] == "Requested TTS model not found"
    assert resolved is False
    assert dispatched is False


@pytest.mark.unit
def test_provider_errors_remain_bounded_when_debug_details_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBUG_ERROR_DETAILS", "true")

    with pytest.raises(HTTPException) as provider_exc:
        tts_core._raise_for_tts_error(
            TTSNetworkError(f"upstream URL contains {_SECRET}"),
            "req-provider-error",
        )

    assert provider_exc.value.status_code == 502
    assert provider_exc.value.detail == {
        "message": "TTS provider request failed",
        "request_id": "req-provider-error",
    }
    assert _SECRET not in repr(provider_exc.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("validation_error", "expected_status", "expected_message"),
    [
        (
            TTSValidationError(f"invalid local value {_SECRET}"),
            400,
            "TTS validation failed",
        ),
        (
            TTSInvalidVoiceReferenceError(f"invalid voice value {_SECRET}"),
            422,
            "TTS voice reference invalid",
        ),
    ],
)
def test_public_tts_validation_errors_ignore_debug_exception_strings(
    monkeypatch: pytest.MonkeyPatch,
    validation_error: TTSValidationError,
    expected_status: int,
    expected_message: str,
) -> None:
    monkeypatch.setenv("DEBUG_ERROR_DETAILS", "true")

    with pytest.raises(HTTPException) as validation_exc:
        tts_core._raise_for_tts_error(
            validation_error,
            "req-validation-error",
        )

    assert validation_exc.value.status_code == expected_status
    assert validation_exc.value.detail == {
        "message": expected_message,
        "request_id": "req-validation-error",
    }
    assert _SECRET not in repr(validation_exc.value)


@pytest.mark.unit
def test_public_tts_validation_error_never_stringifies_hostile_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBUG_ERROR_DETAILS", "true")

    class _HostileValidationError(TTSValidationError):
        stringify_calls = 0

        def __str__(self) -> str:
            self.stringify_calls += 1
            raise RuntimeError(f"hostile stringification {_SECRET}")

    hostile_error = _HostileValidationError("private validation state")
    with pytest.raises(TTSPublicHTTPException) as exc_info:
        tts_core._raise_for_tts_error(hostile_error, "req-hostile-validation")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "message": "TTS validation failed",
        "request_id": "req-hostile-validation",
    }
    assert hostile_error.stringify_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_speech_validation_failures_are_bounded_and_single_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBUG_ERROR_DETAILS", "true")

    class _Config:
        strict_validation = False

    class _FailingValidator:
        def __init__(self, _config: dict[str, Any]) -> None:
            pass

        def sanitize_text(self, text: str, *, provider: str | None = None) -> str:
            try:
                raise RuntimeError(f"raw validation cause {text} {_SECRET}")
            except RuntimeError as raw_error:
                raise TTSValidationError(
                    f"raw validation message {text} {_SECRET}",
                    provider=provider,
                ) from raw_error

    async def _unexpected_resolver(**_kwargs: Any) -> tuple[None, None, None]:
        raise AssertionError("validation failure reached credential resolution")

    monkeypatch.setattr(tts_core, "get_tts_config", lambda: _Config())
    monkeypatch.setattr(tts_core, "TTSInputValidator", _FailingValidator)
    _patch_shim(monkeypatch, _unexpected_resolver)

    from tldw_Server_API.app import main as app_main

    registered_handler = app_main.app.exception_handlers[TTSPublicHTTPException]
    cases = ("alpha", "beta")
    handler_counts = dict.fromkeys(cases, 0)
    pre_handler_graph = dict.fromkeys(cases, False)
    serialized_errors: dict[str, BaseException] = {}

    async def _observing_handler(request: Request, exc: HTTPException):
        case_id = request.headers["x-test-case"]
        handler_counts[case_id] += 1
        pre_handler_graph[case_id] = _exception_graph_contains(
            exc,
            text=f"raw validation cause {case_id} {_SECRET}",
        )
        response = await registered_handler(request, exc)
        serialized_errors[case_id] = exc
        return response

    app = FastAPI(
        exception_handlers={TTSPublicHTTPException: _observing_handler},
    )

    @app.post("/speech")
    async def _speech_route(request: Request):
        case_id = request.headers["x-test-case"]
        return await audio_tts.create_speech(
            _speech_request(text=case_id),
            request,
            tts_service=SimpleNamespace(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post("/speech", headers={"x-test-case": case_id})
                for case_id in cases
            )
        )

    assert handler_counts == {"alpha": 1, "beta": 1}
    assert pre_handler_graph == {"alpha": False, "beta": False}
    for case_id, response in zip(cases, responses, strict=True):
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["message"] == "TTS validation failed"
        assert "details" not in detail
        assert _SECRET not in response.text
        serialized_error = serialized_errors[case_id]
        assert serialized_error.__traceback__ is None
        assert serialized_error.__cause__ is None
        assert serialized_error.__context__ is None
        assert not _exception_graph_contains(serialized_error, text=_SECRET)


@pytest.mark.unit
def test_public_tts_http_error_detaches_the_provider_exception_graph() -> None:
    """The final HTTP error must not retain raw adapter transport objects."""

    sentinel_url = "https://user:secret@tts-http-boundary.invalid/v1?token=private"
    try:
        raise RuntimeError(sentinel_url)
    except RuntimeError as transport_error:
        try:
            raise TTSNetworkError("TTS request failed") from transport_error
        except TTSNetworkError as provider_error:
            with pytest.raises(HTTPException) as exc_info:
                tts_core._raise_for_tts_error(provider_error, "req-detached")

    assert exc_info.value.status_code == 502
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    rendered = "".join(traceback.format_exception(exc_info.value))
    assert sentinel_url not in rendered
    assert "tts-http-boundary.invalid" not in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_tts_serializer_drops_credential_bearing_endpoint_frames_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI serialization must sever endpoint frames holding BYOK snapshots."""

    public_error_type = TTSPublicHTTPException
    boundary_handler = getattr(
        api_exception_handlers,
        "tts_public_http_exception_handler",
        None,
    )
    assert boundary_handler is not None
    from tldw_Server_API.app import main as app_main

    registered_handler = app_main.app.exception_handlers.get(public_error_type)
    assert registered_handler is not None

    case_names = ("alpha", "beta")
    resolutions = {name: _Resolution(name) for name in case_names}
    endpoints = {
        name: f"https://user:{name}@tts-frame-{name}.invalid/v1?token={name}"
        for name in case_names
    }
    pre_boundary_evidence: dict[str, tuple[bool, bool]] = {}
    serialized_errors: dict[str, BaseException] = {}

    async def _resolver(**kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        case_id = kwargs["request"].headers["x-test-case"]
        resolution = resolutions[case_id]
        return (
            1,
            {
                "api_key": resolution.api_key,
                "credentials_resolved": True,
                "openai_base_url": endpoints[case_id],
            },
            resolution,
        )

    class _FailingService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            raise TTSNetworkError("TTS request failed")

    _patch_shim(monkeypatch, _resolver)

    async def _observing_handler(request: Request, exc: HTTPException):
        case_id = request.headers["x-test-case"]
        pre_boundary_evidence[case_id] = (
            _exception_graph_contains(exc, target=resolutions[case_id]),
            _exception_graph_contains(exc, text=endpoints[case_id]),
        )
        response = await registered_handler(request, exc)
        serialized_errors[case_id] = exc
        return response

    app = FastAPI(exception_handlers={public_error_type: _observing_handler})

    @app.post("/speech")
    async def _speech_route(request: Request):
        case_id = request.headers["x-test-case"]
        return await audio_tts.create_speech(
            _speech_request(text=case_id),
            request,
            tts_service=_FailingService(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post("/speech", headers={"x-test-case": case_id})
                for case_id in case_names
            )
        )

    for case_id, response in zip(case_names, responses, strict=True):
        assert response.status_code == 502
        assert response.json()["detail"]["message"] == "TTS provider request failed"
        assert pre_boundary_evidence[case_id] == (True, True)
        serialized_error = serialized_errors[case_id]
        assert serialized_error.__traceback__ is None
        assert serialized_error.__cause__ is None
        assert serialized_error.__context__ is None
        assert not _exception_graph_contains(
            serialized_error,
            target=resolutions[case_id],
        )
        assert not _exception_graph_contains(
            serialized_error,
            text=endpoints[case_id],
        )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resolution_mode", "expected_detail"),
    [
        (
            "store_error",
            {
                "error_code": "credential_store_unavailable",
                "message": "Provider credential storage is temporarily unavailable.",
            },
        ),
        (
            "missing",
            {
                "error_code": "missing_provider_credentials",
                "message": "TTS provider 'openai' requires an API key.",
            },
        ),
    ],
)
async def test_public_tts_serializer_drops_byok_resolution_snapshot_frames(
    monkeypatch: pytest.MonkeyPatch,
    resolution_mode: str,
    expected_detail: dict[str, str],
) -> None:
    """Credential-policy failures use the same traceback-free TTS boundary."""

    secret_snapshot = {
        "openai_api": {
            "api_key": "server-fallback-key-frame-sentinel",
            "api_base_url": "https://user:secret@tts-policy.invalid/v1",
        }
    }
    captured: dict[str, Any] = {}

    class _OverrideSnapshot:
        def enforce(self, _model: str | None) -> None:
            return None

        def ensure_healthy(self) -> None:
            return None

        def server_fallback(self, base_fallback: Any = None) -> Any:
            return base_fallback

    async def _failing_resolver(*_args: Any, **_kwargs: Any) -> Any:
        if resolution_mode == "store_error":
            raise ByokResolutionError("credential_store_unavailable", "openai")
        return ResolvedByokCredentials(
            provider="openai",
            api_key=None,
            app_config=None,
            credential_fields={},
            source="none",
            allowlisted=True,
            status=ByokResolutionStatus.ABSENT,
        )

    monkeypatch.setattr(
        tts_core,
        "capture_provider_override_call_snapshot",
        lambda _provider: _OverrideSnapshot(),
    )
    monkeypatch.setattr(tts_core, "_capture_tts_provider_config", lambda _provider: {})
    monkeypatch.setattr(
        tts_core,
        "load_server_config_snapshot",
        lambda: secret_snapshot,
    )
    monkeypatch.setattr(
        tts_core,
        "resolve_static_server_fallback_from_snapshot",
        lambda *_args: None,
    )
    monkeypatch.setattr(audio_core, "resolve_byok_credentials", _failing_resolver)
    _patch_shim(monkeypatch, audio_core._resolve_tts_byok)

    from tldw_Server_API.app import main as app_main

    registered_handler = app_main.app.exception_handlers[TTSPublicHTTPException]

    async def _observing_handler(request: Request, exc: HTTPException):
        captured["pre_identity"] = _exception_graph_contains(
            exc,
            target=secret_snapshot,
        )
        response = await registered_handler(request, exc)
        captured["error"] = exc
        return response

    app = FastAPI(
        exception_handlers={TTSPublicHTTPException: _observing_handler},
    )

    @app.post("/speech")
    async def _speech_route(request: Request):
        return await audio_tts.create_speech(
            _speech_request(),
            request,
            tts_service=SimpleNamespace(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/speech")

    assert response.status_code == 503
    assert response.json()["detail"] == expected_detail
    assert captured["pre_identity"] is True
    serialized_error = captured["error"]
    assert serialized_error.__traceback__ is None
    assert not _exception_graph_contains(serialized_error, target=secret_snapshot)
    assert not _exception_graph_contains(
        serialized_error,
        text="server-fallback-key-frame-sentinel",
    )
    assert not _exception_graph_contains(serialized_error, text="tts-policy.invalid")


@pytest.mark.unit
def test_tts_history_error_messages_never_persist_provider_details() -> None:
    assert audio_tts._tts_history_error_message(
        TTSNetworkError(f"upstream URL contains {_SECRET}")
    ) == "TTS provider request failed"
    assert audio_tts._tts_history_error_message(
        HTTPException(
            status_code=502,
            detail={"message": "bad gateway", "details": _SECRET},
        )
    ) == "TTS provider request failed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_streams_touch_their_own_snapshot_on_first_audio_and_redact_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions = {name: _Resolution(name) for name in ("alpha", "beta")}
    provider_states = {
        name: SimpleNamespace(
            endpoint=f"https://user:{name}@tts-stream-{name}.invalid/v1?token={name}",
            api_key=f"stream-key-{name}",
        )
        for name in resolutions
    }
    resolver_barrier = asyncio.Event()
    resolver_count = 0
    release_failures = asyncio.Event()
    history_rows: list[dict[str, Any]] = []

    async def _resolver(**kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        nonlocal resolver_count
        request = kwargs["request"]
        case_id = request.headers["x-test-case"]
        resolver_count += 1
        if resolver_count == 2:
            resolver_barrier.set()
        await resolver_barrier.wait()
        resolution = resolutions[case_id]
        return 1, {"api_key": resolution.api_key, "credentials_resolved": True}, resolution

    class _FailAfterFirstAudioService:
        def generate_speech(self, request_data: OpenAISpeechRequest, *_args: Any, **_kwargs: Any) -> Any:
            async def _generate():
                provider_state = provider_states[request_data.input]
                yield f"audio-{request_data.input}".encode()
                await release_failures.wait()
                raise TTSNetworkError(
                    f"provider failure {provider_state.endpoint} {provider_state.api_key}"
                )

            return _generate()

    _patch_shim(monkeypatch, _resolver)
    monkeypatch.setattr(
        audio_tts,
        "_tts_history_config",
        lambda: {
            "enabled": True,
            "store_text": True,
            "store_failed": True,
            "hash_key": "unit-history-key",
        },
    )
    media_db = SimpleNamespace(
        create_tts_history_entry=lambda **kwargs: history_rows.append(kwargs)
    )

    async def _start(name: str):
        return await audio_tts.create_speech(
            _speech_request(text=name, stream=True),
            _make_request(case_id=name),
            tts_service=_FailAfterFirstAudioService(),
            current_user=SimpleNamespace(id=1),
            media_db=media_db,
            usage_log=_UsageLog(),
        )

    responses = await asyncio.gather(_start("alpha"), _start("beta"))

    assert resolutions["alpha"].touch_calls == 1
    assert resolutions["beta"].touch_calls == 1

    async def _consume(response: Any) -> BaseException | None:
        try:
            async for _chunk in response.body_iterator:
                pass
        except BaseException as exc:  # noqa: BLE001 - cancellation is under test
            return exc
        return None

    consumers = [asyncio.create_task(_consume(response)) for response in responses]
    await asyncio.sleep(0)
    release_failures.set()
    failures = await asyncio.gather(*consumers)

    assert all(isinstance(exc, HTTPException) for exc in failures)
    assert all(getattr(exc, "status_code", None) == 502 for exc in failures)
    assert resolutions["alpha"].touch_calls == 1
    assert resolutions["beta"].touch_calls == 1
    assert len(history_rows) == 2
    assert {row["error_message"] for row in history_rows} == {
        "TTS provider request failed"
    }
    assert _SECRET not in repr(history_rows)
    for failure in failures:
        for provider_state in provider_states.values():
            assert not _exception_graph_contains(failure, target=provider_state)
            assert not _exception_graph_contains(failure, text=provider_state.endpoint)
            assert not _exception_graph_contains(failure, text=provider_state.api_key)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_asgi_stream_failure_after_response_start_does_not_retain_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starlette's response-started wrapper must not restore provider state."""

    resolution = _Resolution("asgi-stream")
    provider_state = SimpleNamespace(
        endpoint="https://user:secret@tts-asgi-stream.invalid/v1?token=private",
        api_key="asgi-stream-provider-key-sentinel",
    )
    handler_called = False

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        return (
            1,
            {
                "api_key": resolution.api_key,
                "credentials_resolved": True,
                "openai_base_url": provider_state.endpoint,
            },
            resolution,
        )

    class _FailAfterFirstAudioService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            async def _generate():
                yield b"first-audio"
                raise TTSNetworkError(
                    f"provider failure {provider_state.endpoint} {provider_state.api_key}"
                )

            return _generate()

    _patch_shim(monkeypatch, _resolver)
    from tldw_Server_API.app import main as app_main

    registered_handler = app_main.app.exception_handlers[TTSPublicHTTPException]

    async def _observing_handler(request: Request, exc: HTTPException):
        nonlocal handler_called
        handler_called = True
        return await registered_handler(request, exc)

    app = FastAPI(
        exception_handlers={TTSPublicHTTPException: _observing_handler},
    )

    @app.post("/speech")
    async def _speech_route(request: Request):
        return await audio_tts.create_speech(
            _speech_request(stream=True),
            request,
            tts_service=_FailAfterFirstAudioService(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        with pytest.raises(RuntimeError, match="response already started") as exc_info:
            await client.post("/speech")

    assert handler_called is False
    assert not _exception_graph_contains(exc_info.value, target=provider_state)
    assert not _exception_graph_contains(exc_info.value, text=provider_state.endpoint)
    assert not _exception_graph_contains(exc_info.value, text=provider_state.api_key)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nonstream_failure_after_first_audio_still_touches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _Resolution()

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        return 1, {"api_key": resolution.api_key, "credentials_resolved": True}, resolution

    class _FailAfterFirstAudioService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            async def _generate():
                yield b"first-audio"
                raise TTSNetworkError(f"provider failure {_SECRET}")

            return _generate()

    _patch_shim(monkeypatch, _resolver)

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech(
            _speech_request(stream=False),
            _make_request(),
            tts_service=_FailAfterFirstAudioService(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=_UsageLog(),
        )

    assert exc_info.value.status_code == 502
    assert resolution.touch_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_cancellation_before_completion_does_not_touch_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _Resolution()
    entered_generation = asyncio.Event()
    never_complete = asyncio.Event()

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        return 1, {"api_key": resolution.api_key, "credentials_resolved": True}, resolution

    class _BlockingMetadataService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            async def _generate():
                entered_generation.set()
                await never_complete.wait()
                if False:  # pragma: no cover - make this an async generator
                    yield b""

            return _generate()

    _patch_shim(monkeypatch, _resolver)
    task = asyncio.create_task(
        audio_tts.create_speech_metadata(
            _speech_request(),
            _make_request("/api/v1/audio/speech/metadata"),
            tts_service=_BlockingMetadataService(),
            current_user=SimpleNamespace(id=1),
            usage_log=_UsageLog(),
        )
    )
    await entered_generation.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert resolution.touch_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_metadata_generation_does_not_touch_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted provider iterator is not evidence of credential use."""
    resolution = _Resolution()

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        return 1, {"api_key": resolution.api_key, "credentials_resolved": True}, resolution

    class _EmptyMetadataService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> Any:
            async def _generate():
                if False:  # pragma: no cover - make this an async generator
                    yield b""

            return _generate()

    _patch_shim(monkeypatch, _resolver)
    response = await audio_tts.create_speech_metadata(
        _speech_request(),
        _make_request("/api/v1/audio/speech/metadata"),
        tts_service=_EmptyMetadataService(),
        current_user=SimpleNamespace(id=1),
        usage_log=_UsageLog(),
    )

    assert response.status_code == 204
    assert resolution.touch_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_only_completion_mapping_touches_credentials_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service-authored metadata is successful completion without an audio yield."""
    resolution = _Resolution()
    alignment = {"words": [{"word": "hello", "start_ms": 0, "end_ms": 120}]}

    async def _resolver(**_kwargs: Any) -> tuple[int, dict[str, Any], _Resolution]:
        return 1, {"api_key": resolution.api_key, "credentials_resolved": True}, resolution

    class _MetadataOnlyService:
        def generate_speech(self, request_data: OpenAISpeechRequest, **_kwargs: Any) -> Any:
            async def _generate():
                object.__setattr__(
                    request_data,
                    "_tts_metadata",
                    MappingProxyType({"alignment": alignment}),
                )
                if False:  # pragma: no cover - make this an async generator
                    yield b""

            return _generate()

    _patch_shim(monkeypatch, _resolver)
    response = await audio_tts.create_speech_metadata(
        _speech_request(),
        _make_request("/api/v1/audio/speech/metadata"),
        tts_service=_MetadataOnlyService(),
        current_user=SimpleNamespace(id=1),
        usage_log=_UsageLog(),
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {"alignment": alignment}
    assert resolution.touch_calls == 1
