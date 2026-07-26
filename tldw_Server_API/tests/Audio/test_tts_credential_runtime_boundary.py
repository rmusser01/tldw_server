"""Credential-boundary regressions for synchronous and queued TTS calls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.audio import audio as audio_endpoint
from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio import tts_service
from tldw_Server_API.app.core.AuthNZ import byok_helpers, llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.byok_helpers import DEFAULT_BYOK_ALLOWED_PROVIDERS
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    ProviderOverridePolicyError,
)
from tldw_Server_API.app.core.Chat import streaming_utils
from tldw_Server_API.app.core.TTS import tts_jobs_worker
from tldw_Server_API.app.core.TTS.adapter_registry import TTSProvider
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
    FishS2CommercialApiBackend,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSNetworkError,
    TTSProviderNotConfiguredError,
    TTSRateLimitError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

_SENTINEL_BYOK_KEY = "tts-byok-sentinel-must-never-be-persisted"


@pytest.fixture(autouse=True)
def _allow_direct_worker_dispatch_marker(monkeypatch) -> None:
    """Keep credential units behind the separately tested durable lease fence."""
    monkeypatch.setattr(
        tts_jobs_worker,
        "_persist_tts_dispatch_marker",
        lambda *_args, **_kwargs: True,
    )


class _NoOverrideSnapshot:
    def enforce(self, _model: str | None) -> None:
        return None

    def ensure_healthy(self) -> None:
        return None

    def server_fallback(self, base_fallback=None):
        return base_fallback


def _request_context() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


async def _resolve_frozen_fallback(provider: str, **kwargs) -> ResolvedByokCredentials:
    fallback = kwargs["fallback_override"]
    key = fallback.api_key
    return ResolvedByokCredentials(
        provider=provider,
        api_key=key,
        app_config=dict(fallback.app_config or {}),
        credential_fields=dict(fallback.credential_fields),
        source="server_default" if key else "none",
        allowlisted=True,
        status=(
            ByokResolutionStatus.RESOLVED
            if key
            else ByokResolutionStatus.ABSENT
        ),
        auth_source=fallback.auth_source,
    )


class _CapturingJobManager:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None

    def create_job(self, **kwargs):
        self.created = kwargs
        return {"id": 41, "status": "queued"}


def _patch_worker_output(monkeypatch, tmp_path, captured: dict[str, object]) -> None:
    class _Service:
        def generate_speech(self, _request, **kwargs):
            captured["provider_overrides"] = kwargs.get("provider_overrides")
            captured["provider"] = kwargs.get("provider")
            captured["user_id"] = kwargs.get("user_id")

            async def _chunks():
                yield b"worker-audio"

            return _chunks()

    class _Collections:
        def resolve_output_storage_path(self, name):
            return name

        def create_output_artifact(self, **kwargs):
            return SimpleNamespace(
                id=73,
                storage_path=kwargs["storage_path"],
                format=kwargs["format_"],
            )

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    async def _get_service():
        return _Service()

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tts_jobs_worker.DatabasePaths,
        "get_user_outputs_dir",
        lambda _user_id: tmp_path,
    )
    monkeypatch.setattr(
        tts_jobs_worker.CollectionsDatabase,
        "for_user",
        lambda user_id: _Collections(),
    )


def _scoped_worker_job(
    *,
    job_id: int,
    owner_user_id: int,
    text: str,
    credential_source: str,
    team_ids: list[int] | None = None,
    org_ids: list[int] | None = None,
    trusted_base_url_requested: bool = False,
) -> dict[str, object]:
    return {
        "id": job_id,
        "job_type": "tts_longform",
        "owner_user_id": str(owner_user_id),
        "payload": {
            "speech_request": {
                "model": "tts-1",
                "input": text,
                "voice": "alloy",
                "response_format": "mp3",
                "stream": False,
            },
            "provider_hint": "openai",
            "credential_scope": {
                "owner_user_id": owner_user_id,
                "team_ids": list(team_ids or []),
                "org_ids": list(org_ids or []),
                "credential_source": credential_source,
                "trusted_base_url_requested": trusted_base_url_requested,
            },
        },
    }


@pytest.mark.asyncio
async def test_create_speech_job_never_persists_resolved_credentials(monkeypatch) -> None:
    """Long-form jobs persist identity/scope, never the resolved secret snapshot."""

    async def _resolve_tts_byok(**_kwargs):
        return (
            7,
            {"api_key": _SENTINEL_BYOK_KEY, "openai_api_key": _SENTINEL_BYOK_KEY},
            ResolvedByokCredentials(
                provider="openai",
                api_key=_SENTINEL_BYOK_KEY,
                app_config={},
                credential_fields={},
                source="user",
                allowlisted=True,
                status=ByokResolutionStatus.RESOLVED,
            ),
        )

    def _shim(name: str):
        if name == "_sanitize_speech_request":
            return lambda _data, *, request_id: "openai"
        if name == "_resolve_tts_byok":
            return _resolve_tts_byok
        raise AssertionError(name)

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim)
    manager = _CapturingJobManager()
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    response = await audio_tts.create_speech_job(
        OpenAISpeechRequest(
            input="queued speech",
            model="tts-1",
            voice="alloy",
            response_format="mp3",
            stream=True,
        ),
        request,
        current_user=SimpleNamespace(id="7"),
        jm=manager,
    )

    assert response.status_code == 200
    assert manager.created is not None
    assert manager.created["max_retries"] == 3
    serialized = json.dumps(manager.created, sort_keys=True, default=str)
    assert _SENTINEL_BYOK_KEY not in serialized
    payload = manager.created["payload"]
    assert isinstance(payload, dict)
    assert "provider_overrides" not in payload
    assert "api_key" not in serialized


@pytest.mark.asyncio
async def test_create_speech_job_rejects_nested_request_credentials(monkeypatch) -> None:
    """Arbitrary provider parameters cannot smuggle secrets into a job record."""

    resolved = False

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, None, None)

    def _shim(name: str):
        if name == "_sanitize_speech_request":
            return lambda _data, *, request_id: "openai"
        if name == "_resolve_tts_byok":
            return _unexpected_resolve
        raise AssertionError(name)

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim)
    manager = _CapturingJobManager()
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    with pytest.raises(HTTPException) as exc_info:
        await audio_tts.create_speech_job(
            OpenAISpeechRequest(
                input="queued speech",
                model="tts-1",
                voice="alloy",
                response_format="mp3",
                extra_params={"nested": {"api_key": _SENTINEL_BYOK_KEY}},
            ),
            request,
            current_user=SimpleNamespace(id="7"),
            jm=manager,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error_code"] == "credential_fields_not_allowed"
    assert _SENTINEL_BYOK_KEY not in repr(exc_info.value)
    assert manager.created is None
    assert resolved is False


def test_tts_job_owner_is_authoritative_over_payload_user() -> None:
    """A persisted payload cannot redirect output or credential ownership."""

    with pytest.raises(tts_jobs_worker.TTSJobError, match="user_id"):
        tts_jobs_worker._resolve_user_id(
            {"owner_user_id": "7"},
            {"user_id": "999"},
        )


@pytest.mark.asyncio
async def test_tts_worker_re_resolves_owner_credentials_and_ignores_payload_secret(
    monkeypatch,
    tmp_path,
) -> None:
    """A queued worker obtains B from owner scope and never dispatches persisted A."""

    persisted_key_a = "persisted-job-key-a-must-be-ignored"
    owner_key_b = "owner-runtime-key-b"
    captured: dict[str, object] = {}
    touched = 0

    class _Resolution:
        async def touch_last_used(self):
            nonlocal touched
            touched += 1

    async def _resolve_tts_byok(**kwargs):
        captured["resolution_kwargs"] = kwargs
        return (
            7,
            {
                "api_key": owner_key_b,
                "openai_api_key": owner_key_b,
                "openai_base_url": "https://owner-b.example/v1/audio/speech",
            },
            _Resolution(),
        )

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "_resolve_tts_byok",
        _resolve_tts_byok,
        raising=False,
    )

    result = await tts_jobs_worker._handle_tts_job(
        {
            "id": 72,
            "job_type": "tts_longform",
            "owner_user_id": "7",
            "payload": {
                "speech_request": {
                    "model": "tts-1",
                    "input": "owner-scoped queued speech",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                "provider_hint": "openai",
                "provider_overrides": {
                    "api_key": persisted_key_a,
                    "openai_api_key": persisted_key_a,
                },
            },
        }
    )

    assert result["output_id"] == 73
    assert captured["provider_overrides"] == {
        "api_key": owner_key_b,
        "openai_api_key": owner_key_b,
        "openai_base_url": "https://owner-b.example/v1/audio/speech",
    }
    assert persisted_key_a not in repr(captured)
    resolution_kwargs = captured["resolution_kwargs"]
    assert isinstance(resolution_kwargs, dict)
    assert resolution_kwargs["provider_hint"] == "openai"
    assert resolution_kwargs["model"] == "tts-1"
    assert resolution_kwargs["current_user"].id == 7
    assert touched == 1


@pytest.mark.asyncio
async def test_tts_worker_revalidates_team_scope_and_uses_rotated_key(
    monkeypatch,
    tmp_path,
) -> None:
    """Execution uses the current key from the exact shared scope validated at enqueue."""

    rotated_key = "rotated-team-key-b"
    captured: dict[str, object] = {}
    resolver_calls: list[dict[str, object]] = []
    membership_calls: list[int] = []

    async def _list_team_memberships(user_id: int):
        membership_calls.append(user_id)
        return [{"team_id": 41, "user_id": user_id}]

    async def _resolve_byok_credentials(provider: str, **kwargs):
        resolver_calls.append(dict(kwargs))
        return ResolvedByokCredentials(
            provider=provider,
            api_key=rotated_key,
            app_config={},
            credential_fields={},
            source="team",
            allowlisted=True,
        )

    async def _resolve_tts_byok(**kwargs):
        resolution = await kwargs["credential_resolver"](
            "openai",
            user_id=7,
            request=kwargs["request"],
            fallback_override=None,
            server_config_snapshot={},
        )
        return 7, {"openai_api_key": resolution.api_key}, resolution

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(
        tts_jobs_worker,
        "list_active_team_memberships_for_user",
        _list_team_memberships,
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "resolve_byok_credentials",
        _resolve_byok_credentials,
        raising=False,
    )

    result = await tts_jobs_worker._handle_tts_job(
        _scoped_worker_job(
            job_id=89,
            owner_user_id=7,
            text="rotated team scope",
            credential_source="team",
            team_ids=[41],
        )
    )

    assert result["output_id"] == 73
    assert membership_calls == [7]
    assert captured["provider_overrides"] == {"openai_api_key": rotated_key}
    assert len(resolver_calls) == 1
    assert resolver_calls[0]["team_ids"] == [41]
    assert resolver_calls[0]["org_ids"] == []
    assert resolver_calls[0]["trusted_base_url_override"] is False


@pytest.mark.asyncio
async def test_tts_worker_revalidates_org_scope_and_uses_rotated_key(
    monkeypatch,
    tmp_path,
) -> None:
    """Execution revalidates and rotates credentials in the exact queued org scope."""

    rotated_key = "rotated-org-key-b"
    captured: dict[str, object] = {}
    resolver_calls: list[dict[str, object]] = []

    async def _list_org_memberships(user_id: int):
        return [{"org_id": 51, "user_id": user_id, "status": "active"}]

    async def _resolve_byok_credentials(provider: str, **kwargs):
        resolver_calls.append(dict(kwargs))
        return ResolvedByokCredentials(
            provider=provider,
            api_key=rotated_key,
            app_config={},
            credential_fields={},
            source="org",
            allowlisted=True,
        )

    async def _resolve_tts_byok(**kwargs):
        resolution = await kwargs["credential_resolver"](
            "openai",
            user_id=7,
            request=kwargs["request"],
            fallback_override=None,
            server_config_snapshot={},
        )
        return 7, {"openai_api_key": resolution.api_key}, resolution

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(
        tts_jobs_worker,
        "list_org_memberships_for_user",
        _list_org_memberships,
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "resolve_byok_credentials",
        _resolve_byok_credentials,
        raising=False,
    )

    result = await tts_jobs_worker._handle_tts_job(
        _scoped_worker_job(
            job_id=95,
            owner_user_id=7,
            text="rotated org scope",
            credential_source="org",
            org_ids=[51],
        )
    )

    assert result["output_id"] == 73
    assert captured["provider_overrides"] == {"openai_api_key": rotated_key}
    assert len(resolver_calls) == 1
    assert resolver_calls[0]["team_ids"] == []
    assert resolver_calls[0]["org_ids"] == [51]
    assert resolver_calls[0]["trusted_base_url_override"] is False


@pytest.mark.asyncio
async def test_concurrent_tts_workers_keep_scoped_credentials_isolated(
    monkeypatch,
    tmp_path,
) -> None:
    """Concurrent execution cannot cross-contaminate owner-bound shared keys."""

    membership_barrier = asyncio.Event()
    membership_arrivals = 0
    dispatched: list[tuple[str, str]] = []

    async def _list_team_memberships(user_id: int):
        nonlocal membership_arrivals
        membership_arrivals += 1
        if membership_arrivals == 2:
            membership_barrier.set()
        await membership_barrier.wait()
        return [{"team_id": 41 if user_id == 7 else 42, "user_id": user_id}]

    async def _resolve_byok_credentials(provider: str, **kwargs):
        team_id = kwargs["team_ids"][0]
        return ResolvedByokCredentials(
            provider=provider,
            api_key=f"team-{team_id}-current-key",
            app_config={},
            credential_fields={},
            source="team",
            allowlisted=True,
        )

    async def _resolve_tts_byok(**kwargs):
        user_id = kwargs["current_user"].id
        resolution = await kwargs["credential_resolver"](
            "openai",
            user_id=user_id,
            request=kwargs["request"],
            fallback_override=None,
            server_config_snapshot={},
        )
        return user_id, {"openai_api_key": resolution.api_key}, resolution

    class _Service:
        def generate_speech(self, request, **kwargs):
            dispatched.append(
                (request.input, kwargs["provider_overrides"]["openai_api_key"])
            )

            async def _chunks():
                yield b"isolated-audio"

            return _chunks()

    captured: dict[str, object] = {}
    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(
        tts_jobs_worker,
        "list_active_team_memberships_for_user",
        _list_team_memberships,
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "resolve_byok_credentials",
        _resolve_byok_credentials,
        raising=False,
    )
    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", lambda: asyncio.sleep(0, result=_Service()))

    results = await asyncio.gather(
        tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=96,
                owner_user_id=7,
                text="owner-seven",
                credential_source="team",
                team_ids=[41],
            )
        ),
        tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=97,
                owner_user_id=8,
                text="owner-eight",
                credential_source="team",
                team_ids=[42],
            )
        ),
    )

    assert [result["output_id"] for result in results] == [73, 73]
    assert sorted(dispatched) == [
        ("owner-eight", "team-42-current-key"),
        ("owner-seven", "team-41-current-key"),
    ]


@pytest.mark.asyncio
async def test_tts_worker_fails_closed_when_persisted_team_scope_is_revoked(
    monkeypatch,
) -> None:
    """A membership removed after enqueue cannot fall through to another key."""

    resolved = False

    async def _list_team_memberships(_user_id: int):
        return []

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, {"openai_api_key": "wrong-scope-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)
    monkeypatch.setattr(
        tts_jobs_worker,
        "list_active_team_memberships_for_user",
        _list_team_memberships,
        raising=False,
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=90,
                owner_user_id=7,
                text="revoked team scope",
                credential_source="team",
                team_ids=[41],
            )
        )

    assert resolved is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_tts_worker_scope_lookup_outage_is_retryable_and_detached(
    monkeypatch,
) -> None:
    """Membership-store outages retry safely without exposing backend detail."""

    sentinel = "membership-store-secret-sentinel"

    async def _fail_team_memberships(_user_id: int):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        tts_jobs_worker,
        "list_active_team_memberships_for_user",
        _fail_team_memberships,
        raising=False,
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=91,
                owner_user_id=7,
                text="membership outage",
                credential_source="team",
                team_ids=[41],
            )
        )

    assert exc_info.value.failure_code == "credential_store_unavailable"
    assert exc_info.value.retryable is True
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_source", ["user", "server_default", "none"])
async def test_tts_worker_fails_closed_when_scoped_owner_is_disabled(
    monkeypatch,
    credential_source: str,
) -> None:
    """Every new queued scope revalidates the owner account before resolution."""

    resolved = False

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 7
            return {"id": user_id, "is_active": False}

    async def _from_pool():
        return _UsersRepo()

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, None, None)

    monkeypatch.setattr(
        tts_jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
    )
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=102,
                owner_user_id=7,
                text="disabled owner",
                credential_source=credential_source,
            )
        )

    assert resolved is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_tts_worker_owner_store_outage_is_retryable_and_detached(
    monkeypatch,
) -> None:
    """Owner revalidation outages never downgrade to stale authorization."""

    sentinel = "owner-store-secret-sentinel"

    async def _fail_from_pool():
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        tts_jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_fail_from_pool),
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=103,
                owner_user_id=7,
                text="owner store unavailable",
                credential_source="user",
            )
        )

    assert exc_info.value.failure_code == "credential_store_unavailable"
    assert exc_info.value.retryable is True
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_tts_worker_reconstructs_superuser_endpoint_authority_sequentially(
    monkeypatch,
) -> None:
    """Current superuser semantics are preserved without concurrent DB-handle access."""

    call_order: list[str] = []

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 7
            return {"id": user_id, "is_active": True, "is_superuser": True}

    async def _from_pool():
        return _UsersRepo()

    class _RbacRepo:
        def __init__(self, *, client_id: str):
            assert client_id == "tts_jobs_worker"

        def get_user_roles(self, user_id: int):
            assert user_id == 7
            call_order.append("roles")
            return []

        def get_effective_permissions(self, user_id: int):
            assert user_id == 7
            call_order.append("permissions")
            return []

    monkeypatch.setattr(
        tts_jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
    )
    monkeypatch.setattr(tts_jobs_worker, "AuthnzRbacRepo", _RbacRepo)

    assert await tts_jobs_worker._current_tts_job_base_url_trust(7) is True
    assert call_order == ["roles", "permissions"]


@pytest.mark.asyncio
async def test_tts_worker_endpoint_authority_bypasses_saturated_default_executor(
    monkeypatch,
) -> None:
    """Current RBAC authorization starts even when unrelated default work is stuck."""

    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    loop = asyncio.get_running_loop()
    default_started = asyncio.Event()
    rbac_started = asyncio.Event()
    default_release = threading.Event()
    call_threads: list[int] = []

    def occupy_default_executor() -> None:
        loop.call_soon_threadsafe(default_started.set)
        default_release.wait(timeout=2.0)

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 7
            return {"id": user_id, "is_active": True, "is_superuser": True}

    async def _from_pool():
        return _UsersRepo()

    class _RbacRepo:
        def __init__(self, *, client_id: str):
            assert client_id == "tts_jobs_worker"

        def get_user_roles(self, user_id: int):
            assert user_id == 7
            call_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(rbac_started.set)
            return []

        def get_effective_permissions(self, user_id: int):
            assert user_id == 7
            call_threads.append(threading.get_ident())
            return []

    monkeypatch.setattr(
        tts_jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
    )
    monkeypatch.setattr(tts_jobs_worker, "AuthnzRbacRepo", _RbacRepo)
    monkeypatch.setattr(
        tts_jobs_worker,
        "TTS_JOB_RBAC_DAEMON_POOL",
        BoundedDaemonPool(1),
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "TTS_JOB_RBAC_TIMEOUT_SECONDS",
        0.2,
        raising=False,
    )

    previous_executor = getattr(loop, "_default_executor", None)
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    default_future = loop.run_in_executor(None, occupy_default_executor)
    operation = asyncio.create_task(
        tts_jobs_worker._current_tts_job_base_url_trust(7)
    )
    try:
        await asyncio.wait_for(default_started.wait(), timeout=1.0)
        assert await asyncio.wait_for(operation, timeout=0.5) is True
        assert rbac_started.is_set()
        assert len(call_threads) == 2
        assert call_threads[0] == call_threads[1]
    finally:
        default_release.set()
        await asyncio.gather(default_future, return_exceptions=True)
        await asyncio.gather(operation, return_exceptions=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_tts_worker_blocked_endpoint_authority_times_out_before_dispatch(
    monkeypatch,
) -> None:
    """A stuck RBAC backend fails closed while its bounded late worker drains."""

    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    loop = asyncio.get_running_loop()
    rbac_started = asyncio.Event()
    rbac_finished = asyncio.Event()
    rbac_release = threading.Event()
    resolved = False

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 7
            return {"id": user_id, "is_active": True, "is_superuser": True}

    async def _from_pool():
        return _UsersRepo()

    class _RbacRepo:
        def __init__(self, *, client_id: str):
            assert client_id == "tts_jobs_worker"

        def get_user_roles(self, user_id: int):
            assert user_id == 7
            loop.call_soon_threadsafe(rbac_started.set)
            rbac_release.wait(timeout=2.0)
            return []

        def get_effective_permissions(self, user_id: int):
            assert user_id == 7
            loop.call_soon_threadsafe(rbac_finished.set)
            return []

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        raise AssertionError("credential resolution must not follow timed-out RBAC")

    monkeypatch.setattr(
        tts_jobs_worker,
        "get_auth_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", SINGLE_USER_FIXED_ID=1),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "AuthnzUsersRepo",
        SimpleNamespace(from_pool=_from_pool),
    )
    monkeypatch.setattr(tts_jobs_worker, "AuthnzRbacRepo", _RbacRepo)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)
    monkeypatch.setattr(
        tts_jobs_worker,
        "TTS_JOB_RBAC_DAEMON_POOL",
        BoundedDaemonPool(1),
        raising=False,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "TTS_JOB_RBAC_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )

    operation = asyncio.create_task(
        tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=104,
                owner_user_id=7,
                text="blocked endpoint authority",
                credential_source="user",
                trusted_base_url_requested=True,
            )
        )
    )
    try:
        await asyncio.wait_for(rbac_started.wait(), timeout=1.0)
        with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
            await asyncio.wait_for(operation, timeout=0.5)
        assert exc_info.value.failure_code == "credential_store_unavailable"
        assert exc_info.value.retryable is True
        assert resolved is False
    finally:
        rbac_release.set()
        await asyncio.gather(operation, return_exceptions=True)

    await asyncio.wait_for(rbac_finished.wait(), timeout=1.0)
    assert resolved is False


@pytest.mark.asyncio
async def test_tts_worker_rejects_resolution_from_a_different_credential_source(
    monkeypatch,
) -> None:
    """A new higher-precedence key cannot silently replace the queued scope."""

    dispatched = False

    async def _list_team_memberships(user_id: int):
        return [{"team_id": 41, "user_id": user_id}]

    async def _resolve_tts_byok(**_kwargs):
        return (
            7,
            {"openai_api_key": "new-user-key"},
            ResolvedByokCredentials(
                provider="openai",
                api_key="new-user-key",
                app_config={},
                credential_fields={},
                source="user",
                allowlisted=True,
            ),
        )

    async def _unexpected_service():
        nonlocal dispatched
        dispatched = True
        raise AssertionError("adapter dispatch must not occur")

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _unexpected_service)
    monkeypatch.setattr(
        tts_jobs_worker,
        "list_active_team_memberships_for_user",
        _list_team_memberships,
        raising=False,
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=92,
                owner_user_id=7,
                text="source changed",
                credential_source="team",
                team_ids=[41],
            )
        )

    assert dispatched is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trusted_base_url_requested", "credential_fields"),
    [
        (True, {}),
        (False, {"base_url": "https://unexpected-endpoint.example/v1"}),
    ],
)
async def test_tts_worker_requires_exact_custom_endpoint_intent_after_rotation(
    monkeypatch,
    trusted_base_url_requested: bool,
    credential_fields: dict[str, str],
) -> None:
    """Endpoint removal or unexpected appearance after enqueue fails closed."""

    dispatched = False

    async def _currently_trusted(_owner_user_id: int) -> bool:
        return True

    async def _resolve_tts_byok(**_kwargs):
        return (
            7,
            {"openai_api_key": "rotated-user-key"},
            ResolvedByokCredentials(
                provider="openai",
                api_key="rotated-user-key",
                app_config={},
                credential_fields=credential_fields,
                source="user",
                allowlisted=True,
            ),
        )

    async def _unexpected_service():
        nonlocal dispatched
        dispatched = True
        raise AssertionError("adapter dispatch must not occur")

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _unexpected_service)
    monkeypatch.setattr(
        tts_jobs_worker,
        "_current_tts_job_base_url_trust",
        _currently_trusted,
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=99,
                owner_user_id=7,
                text="endpoint intent changed",
                credential_source="user",
                trusted_base_url_requested=trusted_base_url_requested,
            )
        )

    assert dispatched is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_tts_worker_normalizes_missing_resolution_to_none_source(
    monkeypatch,
    tmp_path,
) -> None:
    """A keyless authoritative resolution remains compatible with source=none."""

    captured: dict[str, object] = {}

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"credentials_resolved": True}, None)

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)

    result = await tts_jobs_worker._handle_tts_job(
        _scoped_worker_job(
            job_id=100,
            owner_user_id=7,
            text="keyless source",
            credential_source="none",
        )
    )

    assert result["output_id"] == 73
    assert captured["provider_overrides"] == {"credentials_resolved": True}


@pytest.mark.asyncio
async def test_tts_worker_fails_closed_when_custom_endpoint_authority_is_revoked(
    monkeypatch,
) -> None:
    """Persisted endpoint intent never acts as stale authorization."""

    resolved = False

    async def _not_currently_trusted(_owner_user_id: int) -> bool:
        return False

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, {"openai_api_key": "wrong-endpoint-key"}, None)

    monkeypatch.setattr(
        tts_jobs_worker,
        "_current_tts_job_base_url_trust",
        _not_currently_trusted,
        raising=False,
    )
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            _scoped_worker_job(
                job_id=93,
                owner_user_id=7,
                text="revoked custom endpoint",
                credential_source="user",
                trusted_base_url_requested=True,
            )
        )

    assert resolved is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_tts_worker_rejects_non_object_job_payload() -> None:
    """Malformed persisted payloads fail terminally instead of raising AttributeError."""

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 101,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": ["not", "an", "object"],
            }
        )

    assert str(exc_info.value) == "invalid job payload"
    assert exc_info.value.failure_code == "invalid_job_payload"
    assert exc_info.value.retryable is False
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_scope",
    [
        {
            "owner_user_id": 7,
            "team_ids": [],
            "org_ids": [],
            "credential_source": "user",
            "trusted_base_url_requested": False,
            "unexpected": True,
        },
        {
            "owner_user_id": "7",
            "team_ids": [],
            "org_ids": [],
            "credential_source": "user",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [True],
            "org_ids": [],
            "credential_source": "team",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [41, 42],
            "org_ids": [],
            "credential_source": "team",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [],
            "org_ids": [],
            "credential_source": "team",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [],
            "org_ids": [51],
            "credential_source": "user",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [],
            "org_ids": [],
            "credential_source": "unknown",
            "trusted_base_url_requested": False,
        },
        {
            "owner_user_id": 7,
            "team_ids": [],
            "org_ids": [],
            "credential_source": "user",
            "trusted_base_url_requested": "yes",
        },
    ],
)
async def test_tts_worker_rejects_malformed_persisted_credential_scope(
    monkeypatch,
    credential_scope,
) -> None:
    """Persisted authorization context is exact, strictly typed, and source-bound."""

    resolved = False

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, {"openai_api_key": "wrong-key"}, None)

    job = _scoped_worker_job(
        job_id=94,
        owner_user_id=7,
        text="malformed scope",
        credential_source="user",
    )
    job["payload"]["credential_scope"] = credential_scope
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(job)

    assert resolved is False
    assert exc_info.value.failure_code == "credential_scope_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_tts_worker_rejects_legacy_nested_request_credentials(monkeypatch) -> None:
    """A legacy queued payload cannot dispatch or echo embedded credential fields."""

    resolved = False

    async def _unexpected_resolve(**_kwargs):
        nonlocal resolved
        resolved = True
        return (7, None, None)

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _unexpected_resolve)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 73,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "legacy secret payload",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                        "extra_params": {
                            "nested": {"authorization": _SENTINEL_BYOK_KEY}
                        },
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert exc_info.value.retryable is False
    assert _SENTINEL_BYOK_KEY not in str(exc_info.value)
    assert resolved is False


@pytest.mark.asyncio
async def test_tts_worker_rederives_known_provider_from_model(monkeypatch, tmp_path) -> None:
    """A persisted provider hint cannot redirect a known model to another adapter."""

    captured: dict[str, object] = {}

    async def _resolve_tts_byok(**kwargs):
        captured["resolution_kwargs"] = kwargs
        return (7, {"openai_api_key": "owner-key"}, None)

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)

    await tts_jobs_worker._handle_tts_job(
        {
            "id": 74,
            "job_type": "tts_longform",
            "owner_user_id": "7",
            "payload": {
                "speech_request": {
                    "model": "tts-1",
                    "input": "authoritative model routing",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                "provider_hint": "kokoro",
            },
        }
    )

    assert captured["resolution_kwargs"]["provider_hint"] == "openai"
    assert captured["provider"] == "openai"


@pytest.mark.asyncio
async def test_tts_worker_cancellation_during_resolution_propagates(monkeypatch) -> None:
    """Worker cancellation is never converted into a retryable credential failure."""

    async def _cancel_resolution(**_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _cancel_resolution)
    with pytest.raises(asyncio.CancelledError):
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 75,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "cancel me",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )


@pytest.mark.asyncio
async def test_tts_worker_sanitizes_unexpected_credential_resolution_failure(
    monkeypatch,
) -> None:
    """Unexpected resolver failures cannot leak details or enter retry loops."""

    leaked_detail = "credential backend failed with secret=do-not-expose"

    async def _fail_resolution(**_kwargs):
        raise RuntimeError(leaked_detail)

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _fail_resolution)
    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 76,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "sanitize resolver failure",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert exc_info.value.retryable is False
    assert str(exc_info.value) == "provider credentials are unavailable"
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert leaked_detail not in repr(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_error", "expected_retryable", "expected_code"),
    [
        (
            ByokResolutionError("credential_store_unavailable", "openai"),
            True,
            "credential_store_unavailable",
        ),
        (
            ByokResolutionError("credential_scope_revoked", "openai"),
            False,
            "credential_scope_revoked",
        ),
        (
            ProviderOverridePolicyError("provider_disabled", "openai"),
            False,
            "provider_disabled",
        ),
        (
            HTTPException(
                status_code=503,
                detail={
                    "error_code": "missing_provider_credentials",
                    "message": "missing key detail must stay bounded",
                },
            ),
            False,
            "missing_provider_credentials",
        ),
    ],
)
async def test_tts_worker_preserves_bounded_typed_credential_retry_policy(
    monkeypatch,
    credential_error,
    expected_retryable,
    expected_code,
) -> None:
    """Only a transient credential-store outage consumes a configured retry."""

    async def _fail_resolution(**_kwargs):
        raise credential_error

    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _fail_resolution)
    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 77,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "bounded credential retry policy",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert exc_info.value.retryable is expected_retryable
    assert exc_info.value.failure_code == expected_code
    assert str(exc_info.value) in {
        "provider credentials are temporarily unavailable",
        "provider credentials are unavailable",
    }
    assert expected_code not in str(exc_info.value)


@pytest.mark.asyncio
async def test_tts_worker_marks_first_audio_use_before_later_adapter_failure(
    monkeypatch,
) -> None:
    """A partial provider response is usage even when the job later fails."""

    touches = 0

    class _Resolution:
        async def touch_last_used(self):
            nonlocal touches
            touches += 1

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            async def _chunks():
                yield b"provider-audio"
                raise TTSNetworkError("raw adapter detail", provider="openai")

            return _chunks()

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, _Resolution())

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)

    with pytest.raises(tts_jobs_worker.TTSJobError):
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 78,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "partial provider response",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert touches == 1


@pytest.mark.asyncio
async def test_tts_worker_drains_first_byte_usage_touch_before_cancellation(
    monkeypatch,
) -> None:
    """Cancellation cannot orphan first-byte usage accounting or the stream."""

    touch_started = asyncio.Event()
    release_touch = asyncio.Event()
    touch_cancelled = asyncio.Event()
    stream_closed = asyncio.Event()
    touches = 0
    closes = 0

    class _Resolution:
        async def touch_last_used(self):
            nonlocal touches
            touches += 1
            touch_started.set()
            try:
                await release_touch.wait()
            except asyncio.CancelledError:
                touch_cancelled.set()
                raise

    class _Stream:
        def __init__(self):
            self._first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._first:
                self._first = False
                return b"provider-audio"
            await asyncio.Event().wait()

        async def aclose(self):
            nonlocal closes
            closes += 1
            stream_closed.set()

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            return _Stream()

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, _Resolution())

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)

    job_task = asyncio.create_task(
        tts_jobs_worker._handle_tts_job(
            {
                "id": 79,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "cancel after first provider byte",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )
    )
    try:
        await asyncio.wait_for(touch_started.wait(), timeout=1.0)
        job_task.cancel()
        done, _pending = await asyncio.wait({job_task}, timeout=0.5)
        assert job_task in done
        with pytest.raises(asyncio.CancelledError):
            job_task.result()
        await asyncio.wait_for(touch_cancelled.wait(), timeout=1.0)
        await asyncio.wait_for(stream_closed.wait(), timeout=1.0)
    finally:
        release_touch.set()
        if not job_task.done():
            job_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job_task

    assert touches == 1
    assert closes == 1


@pytest.mark.asyncio
async def test_tts_worker_ignores_child_cancelled_usage_touch(monkeypatch, tmp_path) -> None:
    """A best-effort usage child cannot cancel an otherwise successful job."""

    captured: dict[str, object] = {}
    touches = 0

    class _Resolution:
        async def touch_last_used(self):
            nonlocal touches
            touches += 1
            raise asyncio.CancelledError

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, _Resolution())

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)

    result = await tts_jobs_worker._handle_tts_job(
        {
            "id": 82,
            "job_type": "tts_longform",
            "owner_user_id": "7",
            "payload": {
                "speech_request": {
                    "model": "tts-1",
                    "input": "self-cancelled usage child",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                "provider_hint": "openai",
            },
        }
    )

    assert result["output_id"] == 73
    assert touches == 1


@pytest.mark.asyncio
async def test_tts_worker_bounds_stuck_usage_touch(monkeypatch, tmp_path) -> None:
    """A stuck best-effort usage write cannot retain a successful job lease."""

    captured: dict[str, object] = {}
    touch_started = asyncio.Event()
    touch_cancelled = asyncio.Event()
    release_touch = asyncio.Event()

    class _Resolution:
        async def touch_last_used(self):
            touch_started.set()
            try:
                await release_touch.wait()
            except asyncio.CancelledError:
                touch_cancelled.set()
                raise

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, _Resolution())

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)

    job_task = asyncio.create_task(
        tts_jobs_worker._handle_tts_job(
            {
                "id": 85,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "bounded stuck usage touch",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )
    )
    try:
        await asyncio.wait_for(touch_started.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({job_task}, timeout=0.75)
        assert job_task in done
        assert job_task.result()["output_id"] == 73
        await asyncio.wait_for(touch_cancelled.wait(), timeout=1.0)
    finally:
        release_touch.set()
        if not job_task.done():
            job_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job_task


@pytest.mark.asyncio
async def test_tts_worker_child_cancelled_close_preserves_adapter_failure(monkeypatch) -> None:
    """A child-cancelled close cannot mask the bounded provider failure."""

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise TTSNetworkError("raw adapter sentinel", provider="openai")

        async def aclose(self):
            raise asyncio.CancelledError

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            return _Stream()

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 83,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "child-cancelled close",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert str(exc_info.value) == "TTS provider request failed"
    assert exc_info.value.failure_code == "provider_unavailable"


@pytest.mark.asyncio
async def test_tts_worker_bounds_non_cooperative_stream_close(monkeypatch) -> None:
    """A stuck close cannot retain the job lease after a provider failure."""

    close_started = asyncio.Event()
    close_cancelled = asyncio.Event()
    release_close = asyncio.Event()

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise TTSNetworkError("raw adapter sentinel", provider="openai")

        async def aclose(self):
            close_started.set()
            try:
                await release_close.wait()
            except asyncio.CancelledError:
                close_cancelled.set()
                raise

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            return _Stream()

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)

    job_task = asyncio.create_task(
        tts_jobs_worker._handle_tts_job(
            {
                "id": 84,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "bounded stuck close",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )
    )
    try:
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({job_task}, timeout=0.5)
        assert job_task in done
        with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
            job_task.result()
        assert str(exc_info.value) == "TTS provider request failed"
        await asyncio.wait_for(close_cancelled.wait(), timeout=1.0)
    finally:
        release_close.set()
        if not job_task.done():
            job_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job_task


@pytest.mark.asyncio
async def test_tts_worker_usage_touch_cannot_exhaust_stream_cleanup_capacity(
    monkeypatch,
) -> None:
    """A cancellation-resistant usage write cannot prevent iterator cleanup."""

    touch_cancelled = asyncio.Event()
    release_touch = asyncio.Event()
    touch_finished = asyncio.Event()
    stream_closed = asyncio.Event()

    class _Resolution:
        async def touch_last_used(self):
            try:
                await release_touch.wait()
            except asyncio.CancelledError:
                touch_cancelled.set()
                await release_touch.wait()
            finally:
                touch_finished.set()

    class _Stream:
        async def aclose(self):
            stream_closed.set()

    with streaming_utils._STREAM_TASK_CAPACITY_LOCK:
        baseline_cleanup = streaming_utils._STREAM_CLEANUP_TASK_ACTIVE_COUNT
    monkeypatch.setattr(
        streaming_utils,
        "STREAM_CLEANUP_TASK_MAX_ACTIVE",
        baseline_cleanup + 1,
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "TTS_JOB_USAGE_TOUCH_TIMEOUT_SECONDS",
        0.01,
    )

    try:
        await tts_jobs_worker._mark_tts_credentials_used(_Resolution())
        await asyncio.wait_for(touch_cancelled.wait(), timeout=1.0)
        await tts_jobs_worker._close_tts_speech_iter(_Stream())
    finally:
        release_touch.set()
        await asyncio.wait_for(touch_finished.wait(), timeout=1.0)
        await asyncio.sleep(0)

    assert stream_closed.is_set()


@pytest.mark.asyncio
async def test_tts_worker_rejects_invalid_payload_without_retaining_raw_value() -> None:
    """Validation failures expose neither rejected values nor Pydantic causes."""

    sentinel = "invalid-format-secret-sentinel"
    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 86,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "invalid payload",
                        "voice": "alloy",
                        "response_format": sentinel,
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert str(exc_info.value) == "invalid speech_request"
    assert exc_info.value.retryable is False
    assert exc_info.value.failure_code == "invalid_speech_request"
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_tts_worker_unknown_adapter_failure_is_terminal_and_sanitized(
    monkeypatch,
) -> None:
    """Deterministic unknown adapter failures cannot cause a retry storm."""

    sentinel = "deterministic-adapter-secret-sentinel"

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            raise ValueError(sentinel)

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", False, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 87,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "deterministic adapter failure",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert str(exc_info.value) == "TTS generation failed"
    assert exc_info.value.failure_code == "tts_generation_failed"
    assert exc_info.value.retryable is False
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert sentinel not in repr(exc_info.value)


def test_tts_worker_rate_limit_after_dispatch_is_terminal() -> None:
    """Typed rate limits stay sanitized but cannot replay an ambiguous dispatch."""

    sentinel = "rate-limit-secret-sentinel"
    failure = tts_jobs_worker._bounded_tts_job_failure(
        TTSRateLimitError(
            sentinel,
            provider="openai",
            details={"retry_after": 37},
        )
    )

    assert failure.retryable is False
    assert not hasattr(failure, "backoff_seconds")
    assert failure.failure_code == "provider_unavailable"
    assert sentinel not in str(failure)
    assert sentinel not in repr(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["open", "write", "close"])
async def test_tts_worker_history_database_failures_are_noncritical_and_sanitized(
    monkeypatch,
    tmp_path,
    failure_stage: str,
) -> None:
    """SQLite history failures cannot replace primary TTS success or failure."""

    sentinel = f"history-{failure_stage}-secret-sentinel"
    captured: dict[str, object] = {}
    rendered_logs: list[str] = []

    def _capture_debug(message, *args, **_kwargs):
        rendered_logs.append(str(message).format(*args))

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    class _History:
        def create_tts_history_entry(self, **_kwargs):
            if failure_stage == "write":
                raise sqlite3.OperationalError(sentinel)

        def close_connection(self):
            if failure_stage == "close":
                raise sqlite3.OperationalError(sentinel)

    _patch_worker_output(monkeypatch, tmp_path, captured)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", True, raising=False)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "logger",
        SimpleNamespace(
            bind=lambda **_kwargs: SimpleNamespace(debug=_capture_debug),
        ),
    )
    if failure_stage == "open":
        def _fail_history_open(**_kwargs):
            raise sqlite3.OperationalError(sentinel)

        monkeypatch.setattr(
            tts_jobs_worker,
            "create_media_database",
            _fail_history_open,
        )
    else:
        monkeypatch.setattr(
            tts_jobs_worker,
            "_open_media_db_for_history",
            lambda _user_id: _History(),
        )

    result = await tts_jobs_worker._handle_tts_job(
        {
            "id": 88,
            "job_type": "tts_longform",
            "owner_user_id": "7",
            "payload": {
                "speech_request": {
                    "model": "tts-1",
                    "input": "history database boundary",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                "provider_hint": "openai",
            },
        }
    )

    assert result["output_id"] == 73
    assert all(sentinel not in line for line in rendered_logs)


@pytest.mark.asyncio
async def test_tts_worker_history_write_failure_cannot_mask_provider_failure(
    monkeypatch,
) -> None:
    """Optional history failure preserves the typed, sanitized provider outcome."""

    provider_sentinel = "provider-error-secret-sentinel"
    history_sentinel = "history-error-secret-sentinel"
    rendered_logs: list[str] = []

    def _capture_debug(message, *args, **_kwargs):
        rendered_logs.append(str(message).format(*args))

    class _Service:
        def generate_speech(self, _request, **_kwargs):
            raise TTSNetworkError(provider_sentinel, provider="openai")

    class _History:
        def create_tts_history_entry(self, **_kwargs):
            raise sqlite3.OperationalError(history_sentinel)

        def close_connection(self):
            return None

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_ENABLED", True, raising=False)
    monkeypatch.setattr(tts_jobs_worker.settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
    monkeypatch.setattr(
        tts_jobs_worker,
        "_open_media_db_for_history",
        lambda _user_id: _History(),
    )
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tts_jobs_worker,
        "logger",
        SimpleNamespace(
            bind=lambda **_kwargs: SimpleNamespace(debug=_capture_debug),
        ),
    )

    with pytest.raises(tts_jobs_worker.TTSJobError) as exc_info:
        await tts_jobs_worker._handle_tts_job(
            {
                "id": 98,
                "job_type": "tts_longform",
                "owner_user_id": "7",
                "payload": {
                    "speech_request": {
                        "model": "tts-1",
                        "input": "provider failure with history enabled",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": False,
                    },
                    "provider_hint": "openai",
                },
            }
        )

    assert exc_info.value.failure_code == "provider_unavailable"
    assert exc_info.value.retryable is False
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert provider_sentinel not in repr(exc_info.value)
    assert all(history_sentinel not in line for line in rendered_logs)


@pytest.mark.asyncio
async def test_tts_worker_returns_cleanly_when_sdk_stops_before_stop_event(
    monkeypatch,
) -> None:
    """Normal SDK completion consumes cancellation of its private stop watcher."""

    class _SDK:
        def __init__(self, _manager, _config):
            return None

        async def run(self, *, handler):
            assert handler is tts_jobs_worker._handle_tts_job

        def stop(self):
            return None

    monkeypatch.setattr(tts_jobs_worker, "WorkerSDK", _SDK)
    monkeypatch.setattr(tts_jobs_worker, "_jobs_manager", lambda: object())

    await tts_jobs_worker.run_tts_jobs_worker(stop_event=asyncio.Event())


@pytest.mark.asyncio
async def test_tts_worker_sanitizes_concurrent_adapter_failures_before_persistence(
    monkeypatch,
) -> None:
    """Concurrent adapter failures persist one bounded code, never raw details."""

    leaked_details = (
        "upstream body included api_key=adapter-secret-a",
        "request failed at https://user:adapter-secret-b@example.test/audio",
    )
    arrivals = 0
    both_arrived = asyncio.Event()
    history_errors: list[str | None] = []

    class _Service:
        def generate_speech(self, request, **_kwargs):
            async def _chunks():
                nonlocal arrivals
                arrivals += 1
                if arrivals == len(leaked_details):
                    both_arrived.set()
                await both_arrived.wait()
                detail = leaked_details[int(request.input[-1])]
                raise TTSNetworkError(detail, provider="openai")
                yield b"unreachable"

            return _chunks()

    class _History:
        def create_tts_history_entry(self, **kwargs):
            history_errors.append(kwargs.get("error_message"))

        def close_connection(self):
            return None

    async def _get_service():
        return _Service()

    async def _resolve_tts_byok(**_kwargs):
        return (7, {"openai_api_key": "safe-test-key"}, None)

    monkeypatch.setattr(tts_jobs_worker, "get_tts_service_v2", _get_service)
    monkeypatch.setattr(tts_jobs_worker, "_resolve_tts_byok", _resolve_tts_byok)
    monkeypatch.setattr(
        tts_jobs_worker,
        "JobManager",
        lambda: SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: True),
    )
    monkeypatch.setattr(tts_jobs_worker, "emit_job_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tts_jobs_worker,
        "_tts_history_config",
        lambda: {
            "enabled": True,
            "store_failed": True,
            "store_text": False,
            "hash_key": "safe-test-history-hmac-key",
        },
    )
    monkeypatch.setattr(tts_jobs_worker, "_open_media_db_for_history", lambda _user_id: _History())

    def _job(index: int) -> dict[str, object]:
        return {
            "id": 80 + index,
            "job_type": "tts_longform",
            "owner_user_id": "7",
            "payload": {
                "speech_request": {
                    "model": "tts-1",
                    "input": f"concurrent adapter request {index}",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                "provider_hint": "openai",
            },
        }

    failures = await asyncio.gather(
        *[tts_jobs_worker._handle_tts_job(_job(index)) for index in range(2)],
        return_exceptions=True,
    )

    assert all(isinstance(failure, tts_jobs_worker.TTSJobError) for failure in failures)
    assert all(str(failure) == "TTS provider request failed" for failure in failures)
    assert all(getattr(failure, "failure_code", None) == "provider_unavailable" for failure in failures)
    assert all(getattr(failure, "retryable", None) is False for failure in failures)
    assert all(getattr(failure, "__context__", None) is None for failure in failures)
    assert history_errors == ["TTS provider request failed", "TTS provider request failed"]
    persisted = repr((failures, history_errors))
    assert all(detail not in persisted for detail in leaked_details)


@pytest.mark.asyncio
async def test_tts_model_routing_applies_authoritative_overrides_at_adapter_boundary() -> None:
    """Model-based routing cannot silently fall back to the cached adapter config."""

    sentinel_adapter = object()
    captured: dict[str, object] = {}

    class _Registry:
        async def create_adapter_with_overrides(self, provider, overrides):
            captured["provider"] = provider
            captured["overrides"] = overrides
            return sentinel_adapter

    class _Factory:
        registry = _Registry()

        def get_provider_for_model(self, _model):
            return TTSProvider.OPENAI

        async def get_adapter_by_model(self, _model):
            captured["used_cached_adapter"] = True
            return object()

    service = object.__new__(TTSServiceV2)
    service.factory = _Factory()
    service._factory = service.factory
    overrides = {
        "openai_api_key": "model-route-key",
        "openai_base_url": "https://model-route.example/v1/audio/speech",
    }

    adapter = await service._get_adapter("tts-1", overrides=overrides)

    assert adapter is sentinel_adapter
    assert captured == {
        "provider": TTSProvider.OPENAI,
        "overrides": overrides,
    }


@pytest.mark.asyncio
async def test_tts_authoritative_overrides_disable_cross_provider_fallback() -> None:
    """Resolved credentials cannot fall through to another cached provider."""

    captured: dict[str, object] = {}

    class _Service(TTSServiceV2):
        async def _ensure_factory(self):
            return SimpleNamespace()

        def _convert_request(self, _request):
            return SimpleNamespace(extra_params={})

        def _resolve_observability_context(self, _request, *, explicit_request_id=None):
            return explicit_request_id or "request-id", None

        async def _prepare_generate_speech_request(self, **kwargs):
            captured["fallback"] = kwargs["fallback"]
            raise TTSProviderNotConfiguredError("stop after fallback capture")

    service = object.__new__(_Service)
    service._factory = None
    service._stream_errors_as_audio = False
    request = OpenAISpeechRequest(
        input="fail closed",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
    )

    with pytest.raises(TTSProviderNotConfiguredError):
        async for _chunk in service.generate_speech(
            request,
            provider="openai",
            fallback=True,
            provider_overrides={"openai_api_key": "authoritative-key"},
        ):
            pass

    assert captured["fallback"] is False


@pytest.mark.asyncio
async def test_fish_s2_env_key_reaches_backend_through_frozen_snapshot(monkeypatch) -> None:
    """Fish S2 keeps its legacy env aliases without a second live TTS read."""

    monkeypatch.setenv("FISH_AUDIO_API_KEY", "fish-audio-preferred")
    monkeypatch.setenv("FISH_API_KEY", "fish-legacy-lower-priority")
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _resolve_frozen_fallback)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
    )

    assert "fish_s2" in DEFAULT_BYOK_ALLOWED_PROVIDERS
    _user_id, overrides, _resolution = await tts_service._resolve_tts_byok(
        provider_hint="fish_s2",
        model="fish-s2-pro",
        current_user=SimpleNamespace(id=7),
        request=_request_context(),
    )

    assert overrides is not None
    backend = FishS2CommercialApiBackend(overrides)
    assert backend.api_key == "fish-audio-preferred"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credential_error", "expected_status", "expected_code"),
    [
        (
            ProviderOverridePolicyError("provider_disabled", "openai"),
            403,
            "provider_disabled",
        ),
        (
            ByokResolutionError("credential_store_unavailable", "openai"),
            503,
            "credential_store_unavailable",
        ),
    ],
)
async def test_tts_endpoint_bounds_credential_policy_failures(
    monkeypatch,
    credential_error,
    expected_status,
    expected_code,
) -> None:
    """Typed policy/store failures become stable public HTTP responses."""

    async def _raise_credential_error(**_kwargs):
        raise credential_error

    monkeypatch.setattr(tts_service, "_resolve_tts_byok", _raise_credential_error)

    with pytest.raises(HTTPException) as exc_info:
        await audio_endpoint._resolve_tts_byok(
            provider_hint="openai",
            model="tts-1",
            current_user=SimpleNamespace(id=7),
            request=_request_context(),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail["error_code"] == expected_code


@pytest.mark.asyncio
async def test_tts_concurrent_server_rotations_reach_openai_adapter_atomically(monkeypatch) -> None:
    """Concurrent A/B generations cannot splice key and endpoint at the adapter."""

    snapshots = [
        {
            "openai_api": {
                "api_key": "server-key-a",
                "api_base_url": "https://generation-a.example/v1",
            }
        },
        {
            "openai_api": {
                "api_key": "server-key-b",
                "api_base_url": "https://generation-b.example/v1",
            }
        },
    ]
    first_waiting = asyncio.Event()
    release_first = asyncio.Event()

    def _load_snapshot():
        return snapshots.pop(0)

    async def _event_gated_resolver(provider: str, **kwargs):
        fallback = kwargs["fallback_override"]
        endpoint = fallback.app_config["openai_api"]["api_base_url"]
        if "generation-a" in endpoint:
            first_waiting.set()
            await release_first.wait()
        else:
            release_first.set()
        return await _resolve_frozen_fallback(provider, **kwargs)

    monkeypatch.setattr(tts_service, "load_server_config_snapshot", _load_snapshot)
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _event_gated_resolver)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
        raising=False,
    )
    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            providers={"openai": SimpleNamespace(api_key="live-splice-key")}
        ),
    )

    first = asyncio.create_task(
        tts_service._resolve_tts_byok(
            provider_hint="openai",
            current_user=SimpleNamespace(id=7),
            request=_request_context(),
        )
    )
    await first_waiting.wait()
    second = asyncio.create_task(
        tts_service._resolve_tts_byok(
            provider_hint="openai",
            current_user=SimpleNamespace(id=7),
            request=_request_context(),
        )
    )
    first_result, second_result = await asyncio.gather(first, second)

    adapter_a = OpenAIAdapter(config=first_result[1])
    adapter_b = OpenAIAdapter(config=second_result[1])
    assert (adapter_a.api_key, adapter_a.base_url) == (
        "server-key-a",
        "https://generation-a.example/v1/audio/speech",
    )
    assert (adapter_b.api_key, adapter_b.base_url) == (
        "server-key-b",
        "https://generation-b.example/v1/audio/speech",
    )


@pytest.mark.asyncio
async def test_tts_absent_snapshot_stays_absent_before_later_server_key(monkeypatch) -> None:
    """An absent generation cannot recover from a later live TTS config read."""

    snapshots = [
        {"openai_api": {}},
        {"openai_api": {"api_key": "server-key-b"}},
    ]
    monkeypatch.setattr(tts_service, "load_server_config_snapshot", lambda: snapshots.pop(0))
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _resolve_frozen_fallback)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        lambda _provider: _NoOverrideSnapshot(),
        raising=False,
    )
    monkeypatch.setattr(
        tts_service,
        "get_tts_config",
        lambda: SimpleNamespace(
            providers={"openai": SimpleNamespace(api_key="later-live-key")}
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await tts_service._resolve_tts_byok(
            provider_hint="openai",
            current_user=SimpleNamespace(id=7),
            request=_request_context(),
        )
    assert exc_info.value.detail["error_code"] == "missing_provider_credentials"

    _user_id, overrides, _resolution = await tts_service._resolve_tts_byok(
        provider_hint="openai",
        current_user=SimpleNamespace(id=7),
        request=_request_context(),
    )
    assert OpenAIAdapter(config=overrides).api_key == "server-key-b"


@pytest.mark.asyncio
async def test_tts_rejects_admin_disabled_provider_before_resolution(monkeypatch) -> None:
    """TTS uses the same provider-disable policy as Chat/RAG."""

    original = llm_provider_overrides.get_llm_provider_overrides_snapshot()
    called = False

    async def _unexpected_resolver(*_args, **_kwargs):
        nonlocal called
        called = True
        return await _resolve_frozen_fallback(*_args, **_kwargs)

    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", is_enabled=False)}
    )
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _unexpected_resolver)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        llm_provider_overrides.capture_provider_override_call_snapshot,
        raising=False,
    )
    try:
        with pytest.raises(ProviderOverridePolicyError) as exc_info:
            await tts_service._resolve_tts_byok(
                provider_hint="openai",
                model="tts-1",
                current_user=SimpleNamespace(id=7),
                request=_request_context(),
            )
        assert exc_info.value.policy_code == "provider_disabled"
        assert called is False
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(original)


@pytest.mark.asyncio
async def test_tts_rejects_admin_disallowed_model_before_resolution(monkeypatch) -> None:
    """TTS enforces the admin model allowlist before credential resolution."""

    original = llm_provider_overrides.get_llm_provider_overrides_snapshot()
    called = False

    async def _unexpected_resolver(*_args, **_kwargs):
        nonlocal called
        called = True
        return await _resolve_frozen_fallback(*_args, **_kwargs)

    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["tts-1-hd"],
            )
        }
    )
    monkeypatch.setattr(tts_service, "resolve_byok_credentials", _unexpected_resolver)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        llm_provider_overrides.capture_provider_override_call_snapshot,
        raising=False,
    )
    try:
        with pytest.raises(ProviderOverridePolicyError) as exc_info:
            await tts_service._resolve_tts_byok(
                provider_hint="openai",
                model="tts-1",
                current_user=SimpleNamespace(id=7),
                request=_request_context(),
            )
        assert exc_info.value.policy_code == "model_not_allowed"
        assert called is False
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(original)


@pytest.mark.asyncio
async def test_tts_fails_closed_when_override_store_is_unhealthy(monkeypatch) -> None:
    """TTS cannot bypass an unhealthy authoritative override store."""

    original = llm_provider_overrides.get_llm_provider_overrides_snapshot()
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({}, healthy=False)
    monkeypatch.setattr(
        tts_service,
        "capture_provider_override_call_snapshot",
        llm_provider_overrides.capture_provider_override_call_snapshot,
        raising=False,
    )
    try:
        with pytest.raises(ByokResolutionError) as exc_info:
            await tts_service._resolve_tts_byok(
                provider_hint="openai",
                model="tts-1",
                current_user=SimpleNamespace(id=7),
                request=_request_context(),
            )
        assert exc_info.value.code == "credential_store_unavailable"
    finally:
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(original)
