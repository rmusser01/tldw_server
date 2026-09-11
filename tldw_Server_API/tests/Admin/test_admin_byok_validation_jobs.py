from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ import byok_testing
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    PROVIDER_STREAM_ERROR_MESSAGES,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    bind_provider_call_credentials,
)
from tldw_Server_API.app.services import (
    admin_byok_validation_jobs_worker as validation_worker,
)


@dataclass
class _FakeValidationRunsRepo:
    run: dict[str, object]
    running_calls: list[tuple[str, str | None]] = field(default_factory=list)
    complete_calls: list[dict[str, int]] = field(default_factory=list)
    failed_calls: list[tuple[str, str]] = field(default_factory=list)

    async def get_run(self, run_id: str):
        if self.run.get("id") != run_id:
            return None
        return dict(self.run)

    async def mark_running(self, run_id: str, *, job_id: str | None):
        self.running_calls.append((run_id, job_id))
        updated = dict(self.run)
        updated["status"] = "running"
        updated["job_id"] = job_id
        return updated

    async def mark_complete(
        self,
        run_id: str,
        *,
        keys_checked: int,
        valid_count: int,
        invalid_count: int,
        error_count: int,
    ):
        self.complete_calls.append(
            {
                "keys_checked": keys_checked,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "error_count": error_count,
            }
        )
        updated = dict(self.run)
        updated["status"] = "complete"
        updated["keys_checked"] = keys_checked
        updated["valid_count"] = valid_count
        updated["invalid_count"] = invalid_count
        updated["error_count"] = error_count
        return updated

    async def mark_failed(self, run_id: str, *, error_message: str):
        self.failed_calls.append((run_id, error_message))
        updated = dict(self.run)
        updated["status"] = "failed"
        updated["error_message"] = error_message
        return updated


class _SnapshotRecordingAdapter:
    """Record exact validation requests at the real sync-adapter boundary."""

    def __init__(self, *, expected_calls: int = 1, gated: bool = False) -> None:
        self.expected_calls = expected_calls
        self.gated = gated
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.all_entered = threading.Event()
        self.release = threading.Event()

    def chat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        bound, credentials = bind_provider_call_credentials(
            "custom-openai-api-2",
            request,
            consume=True,
        )
        assert is_runtime_issued_provider_call_credentials(
            credentials,
            provider="custom-openai-api-2",
        )
        with self._lock:
            self.calls.append(copy.deepcopy(bound))
            if len(self.calls) >= self.expected_calls:
                self.all_entered.set()
        if self.gated and not self.release.wait(timeout=5.0):
            raise AssertionError("Timed out waiting to release validation adapter")
        return {"choices": [{"message": {"content": "pong"}}]}


def _install_real_job_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: _SnapshotRecordingAdapter,
    snapshot_loader: Any,
    capacity: int,
) -> BoundedDaemonPool:
    """Install real job-to-credential-test dispatch with a controlled snapshot."""
    pool = BoundedDaemonPool(capacity)
    monkeypatch.setattr(
        validation_worker,
        "load_server_config_snapshot",
        snapshot_loader,
        raising=False,
    )
    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(
        byok_testing,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda *_args, **_kwargs: "live-model-must-not-be-used",
    )
    monkeypatch.setattr(byok_testing, "SYNC_ADAPTER_CALL_POOL", pool)
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(capacity),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )
    monkeypatch.setenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        str(capacity),
    )
    monkeypatch.setenv(
        "ADMIN_BYOK_VALIDATION_PER_PROVIDER_CONCURRENCY",
        str(capacity),
    )
    return pool


@pytest.mark.asyncio
async def test_handle_byok_validation_job_marks_running_and_complete() -> None:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        CandidateLoadResult,
        handle_byok_validation_job,
    )

    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-1",
            "status": "queued",
            "org_id": 42,
            "provider": None,
        }
    )

    async def _load_candidates(run: dict[str, object]) -> CandidateLoadResult:
        assert run["id"] == "run-1"
        return CandidateLoadResult(
            candidates=[
                {"provider": "openai", "api_key": "valid-openai-1", "credential_fields": None},
                {"provider": "openai", "api_key": "invalid-openai-2", "credential_fields": None},
                {"provider": "anthropic", "api_key": "valid-anthropic-1", "credential_fields": None},
            ],
            error_count=1,
        )

    async def _validate(*, provider: str, api_key: str, credential_fields=None, model=None):
        if api_key.startswith("invalid-"):
            raise ChatAuthenticationError(message="rejected", provider=provider)
        return "ok"

    result = await handle_byok_validation_job(
        {"id": "job-1", "payload": {"run_id": "run-1"}},
        repo=repo,
        candidate_loader=_load_candidates,
        test_provider_credentials_fn=_validate,
    )

    assert repo.running_calls == [("run-1", "job-1")]
    assert repo.complete_calls == [
        {
            "keys_checked": 3,
            "valid_count": 2,
            "invalid_count": 1,
            "error_count": 1,
        }
    ]
    assert repo.failed_calls == []
    assert result["status"] == "complete"
    assert result["keys_checked"] == 3
    assert result["valid_count"] == 2
    assert result["error_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_kind", "expected_public_code", "candidate_is_invalid"),
    [
        ("authentication", "provider_authentication_failed", True),
        ("bad_request", "provider_configuration_invalid", True),
        ("unavailable", "provider_unavailable", False),
        ("missing", "missing_provider_credentials", False),
    ],
)
async def test_validation_job_classifies_real_sanitized_adapter_outcome(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
    expected_public_code: str,
    candidate_is_invalid: bool,
) -> None:
    """Only tested-candidate auth/config rejection is a job-level invalid key."""
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        CandidateLoadResult,
        handle_byok_validation_job,
    )

    sentinel = f"sk-private-{expected_public_code}-/private/provider-response.json"

    class _RejectingAdapter:
        def __init__(self) -> None:
            self.call_count = 0

        def chat(
            self,
            _request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            self.call_count += 1
            if error_kind == "authentication":
                raise ChatAuthenticationError(
                    message=f"provider rejected {sentinel}",
                    provider="openai",
                )
            if error_kind == "bad_request":
                raise ChatBadRequestError(
                    message=f"provider rejected {sentinel}",
                    provider="openai",
                )
            if error_kind == "missing":
                raise ChatConfigurationError(
                    message=f"provider rejected {sentinel}",
                    provider="openai",
                    error_code="missing_provider_credentials",
                )
            raise ChatProviderError(
                message=f"provider rejected {sentinel}",
                provider="openai",
                status_code=502,
            )

    adapter = _RejectingAdapter()
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-real-sanitized",
            "status": "queued",
            "org_id": None,
            "provider": "openai",
        }
    )

    async def _load_candidates(_run: dict[str, object]) -> CandidateLoadResult:
        return CandidateLoadResult(
            candidates=[
                {
                    "provider": "openai",
                    "api_key": "sk-candidate-under-test",
                    "credential_fields": None,
                }
            ]
        )

    monkeypatch.setattr(byok_testing, "_is_test_mode", lambda: False)
    monkeypatch.setattr(
        byok_testing,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: adapter),
    )
    monkeypatch.setattr(
        byok_testing,
        "resolve_default_model_for_provider",
        lambda *_args, **_kwargs: "validation-model",
    )
    monkeypatch.setattr(
        byok_testing,
        "SYNC_ADAPTER_CALL_POOL",
        BoundedDaemonPool(2),
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSION",
        threading.BoundedSemaphore(1),
        raising=False,
    )
    monkeypatch.setattr(
        byok_testing,
        "_PROVIDER_HEALTH_ADMISSIONS_BY_PROVIDER",
        {},
        raising=False,
    )

    async def run_job() -> dict[str, Any]:
        return await handle_byok_validation_job(
            {
                "id": "job-real-sanitized",
                "payload": {"run_id": "run-real-sanitized"},
            },
            repo=repo,
            candidate_loader=_load_candidates,
            test_provider_credentials_fn=byok_testing.test_provider_credentials,
        )

    if candidate_is_invalid:
        result = await run_job()
        assert adapter.call_count == 1
        assert repo.failed_calls == []
        assert repo.complete_calls == [
            {
                "keys_checked": 1,
                "valid_count": 0,
                "invalid_count": 1,
                "error_count": 0,
            }
        ]
        assert result["status"] == "complete"
        assert result["invalid_count"] == 1
        assert sentinel not in repr(repo.complete_calls)
        return

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await run_job()
    assert adapter.call_count == 1
    assert exc_info.value.code == expected_public_code
    assert repo.complete_calls == []
    assert repo.failed_calls == [
        ("run-real-sanitized", "provider_validation_failed")
    ]
    assert sentinel not in repr(exc_info.value)
    assert sentinel not in repr(repo.failed_calls)


@pytest.mark.asyncio
async def test_handle_byok_validation_job_marks_failed_with_redacted_summary() -> None:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        handle_byok_validation_job,
    )

    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-1",
            "status": "queued",
            "org_id": None,
            "provider": "openai",
        }
    )

    async def _load_candidates(run: dict[str, object]) -> list[dict[str, object]]:
        return [{"provider": "openai", "api_key": "valid-openai-1", "credential_fields": None}]

    sentinel = (
        "sk-validator-finalizer-secret "
        "https://provider.example/private/validator.json?token=secret"
    )

    async def _validate(*, provider: str, api_key: str, credential_fields=None, model=None):
        raise RuntimeError(sentinel)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await handle_byok_validation_job(
            {"id": "job-9", "payload": {"run_id": "run-1"}},
            repo=repo,
            candidate_loader=_load_candidates,
            test_provider_credentials_fn=_validate,
        )

    assert repo.complete_calls == []
    assert repo.failed_calls == [("run-1", "provider_validation_failed")]
    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.message == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)
    assert sentinel not in repr(repo.failed_calls)


@pytest.mark.asyncio
async def test_run_validation_scan_uses_bounded_per_provider_concurrency() -> None:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import _run_validation_scan

    current_by_provider: dict[str, int] = {}
    max_by_provider: dict[str, int] = {}

    async def _validate(*, provider: str, api_key: str, credential_fields=None, model=None):
        current_by_provider[provider] = current_by_provider.get(provider, 0) + 1
        max_by_provider[provider] = max(max_by_provider.get(provider, 0), current_by_provider[provider])
        await asyncio.sleep(0.01)
        current_by_provider[provider] -= 1
        return "ok"

    candidates = [
        {"provider": "openai", "api_key": "valid-1", "credential_fields": None},
        {"provider": "openai", "api_key": "valid-2", "credential_fields": None},
        {"provider": "openai", "api_key": "valid-3", "credential_fields": None},
        {"provider": "anthropic", "api_key": "valid-4", "credential_fields": None},
    ]

    summary = await _run_validation_scan(
        candidates,
        test_provider_credentials_fn=_validate,
        per_provider_limit=2,
    )

    assert summary["keys_checked"] == 4
    assert summary["valid_count"] == 4
    assert summary["invalid_count"] == 0
    assert summary["error_count"] == 0
    assert max_by_provider["openai"] <= 2


@pytest.mark.asyncio
async def test_load_default_validation_candidates_skips_unreadable_secrets(monkeypatch) -> None:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        load_default_validation_candidates,
    )

    class _SharedRepo:
        async def list_secrets(self, **kwargs):
            return [
                {"scope_type": "org", "scope_id": 42, "provider": "openai"},
            ]

        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            return {"encrypted_blob": "broken"}

    class _UserRepo:
        async def list_secrets_for_user(self, user_id: int):
            return []

        async def fetch_secret_for_user(self, user_id: int, provider: str):
            return None

    class _UsersRepo:
        async def list_users(self, *, offset: int, limit: int, org_ids=None):
            return [], 0

    async def _get_shared_repo():
        return _SharedRepo()

    async def _get_user_repo():
        return _UserRepo()

    async def _from_pool():
        return _UsersRepo()

    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.admin_byok_service.get_shared_byok_repo",
        _get_shared_repo,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.admin_byok_service.get_user_byok_repo",
        _get_user_repo,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.AuthnzUsersRepo.from_pool",
        _from_pool,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.loads_envelope",
        lambda encrypted_blob: encrypted_blob,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_byok_validation_jobs_worker.decrypt_byok_payload",
        lambda envelope: (_ for _ in ()).throw(ValueError("broken secret")),
    )

    result = await load_default_validation_candidates({"org_id": None, "provider": None})

    assert result.candidates == []
    assert result.error_count == 1


@pytest.mark.asyncio
async def test_load_default_validation_candidates_supports_v2_and_skips_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All owner scopes use runtime auth-source precedence and skip bad payloads."""
    sentinels = {
        "org": "sk-org-malformed-secret-must-not-log",
        "team": "sk-team-malformed-secret-must-not-log",
        "user": "sk-user-malformed-secret-must-not-log",
    }
    payloads: dict[str, dict[str, Any]] = {
        "bad-org": {"private": sentinels["org"]},
        "legacy-org": {
            "api_key": "legacy-org-key",
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {"access_token": "oauth-must-not-be-selected"},
            },
        },
        "bad-team": {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {"refresh_token": sentinels["team"]},
            },
        },
        "v2-team": {
            "credential_version": 2,
            "active_auth_source": "api_key",
            "credentials": {
                "api_key": {"api_key": "v2-team-api-key"},
                "oauth": {"access_token": "team-oauth-must-not-be-selected"},
            },
            "credential_fields": {"project_id": "team-project"},
        },
        "bad-user": {
            "api_key": sentinels["user"],
            "credential_fields": ["not", "an", "object"],
        },
        "v2-user": {
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "api_key": {"api_key": "user-api-key-must-not-be-selected"},
                "oauth": {"access_token": "v2-user-oauth-token"},
            },
        },
    }

    class _SharedRepo:
        async def list_secrets(self, **kwargs):
            scope_type = kwargs.get("scope_type")
            if scope_type == "org":
                return [
                    {"scope_type": "org", "scope_id": 42, "provider": "broken-org"},
                    {"scope_type": "org", "scope_id": 42, "provider": "anthropic"},
                ]
            if scope_type == "team":
                return [
                    {"scope_type": "team", "scope_id": 7, "provider": "broken-team"},
                    {"scope_type": "team", "scope_id": 7, "provider": "openai"},
                ]
            return []

        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            labels = {
                ("org", 42, "broken-org"): "bad-org",
                ("org", 42, "anthropic"): "legacy-org",
                ("team", 7, "broken-team"): "bad-team",
                ("team", 7, "openai"): "v2-team",
            }
            return {"encrypted_blob": labels[(scope_type, scope_id, provider)]}

    class _UserRepo:
        async def list_secrets_for_user(self, user_id: int):
            assert user_id == 9
            return [
                {"provider": "broken-user"},
                {"provider": "openai"},
            ]

        async def fetch_secret_for_user(self, user_id: int, provider: str):
            assert user_id == 9
            labels = {
                "broken-user": "bad-user",
                "openai": "v2-user",
            }
            return {"encrypted_blob": labels[provider]}

    class _UsersRepo:
        async def list_users(self, *, offset: int, limit: int, org_ids=None):
            assert limit == 200
            assert org_ids == [42]
            return ([{"id": 9}], 1) if offset == 0 else ([], 1)

    async def _get_shared_repo():
        return _SharedRepo()

    async def _get_user_repo():
        return _UserRepo()

    async def _from_pool():
        return _UsersRepo()

    async def _list_teams(org_id: int, *, limit: int, offset: int):
        assert org_id == 42
        assert limit == 200
        return [{"id": 7}] if offset == 0 else []

    warnings: list[str] = []
    test_logger = SimpleNamespace(
        warning=lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(validation_worker, "logger", test_logger)
    monkeypatch.setattr(validation_worker.admin_byok_service, "get_shared_byok_repo", _get_shared_repo)
    monkeypatch.setattr(validation_worker.admin_byok_service, "get_user_byok_repo", _get_user_repo)
    monkeypatch.setattr(validation_worker.AuthnzUsersRepo, "from_pool", _from_pool)
    monkeypatch.setattr(validation_worker.admin_orgs_service, "list_teams_by_org", _list_teams)
    monkeypatch.setattr(validation_worker, "loads_envelope", lambda encrypted_blob: encrypted_blob)
    monkeypatch.setattr(validation_worker, "decrypt_byok_payload", lambda envelope: payloads[envelope])

    result = await validation_worker.load_default_validation_candidates(
        {"org_id": 42, "provider": None}
    )

    assert result.error_count == 3
    assert result.candidates == [
        {
            "provider": "anthropic",
            "api_key": "legacy-org-key",
            "credential_fields": None,
            "auth_source": "api_key",
            "source": "shared",
            "scope_type": "org",
            "scope_id": 42,
        },
        {
            "provider": "openai",
            "api_key": "v2-team-api-key",
            "credential_fields": {"project_id": "team-project"},
            "auth_source": "api_key",
            "source": "shared",
            "scope_type": "team",
            "scope_id": 7,
        },
        {
            "provider": "openai",
            "api_key": "v2-user-oauth-token",
            "credential_fields": None,
            "auth_source": "oauth",
            "source": "user",
            "user_id": 9,
        },
    ]
    rendered_warnings = repr(warnings)
    assert len(warnings) == 3
    assert all(sentinel not in rendered_warnings for sentinel in sentinels.values())
    assert "must-not-be-selected" not in repr(result.candidates)


@pytest.mark.asyncio
async def test_handle_byok_validation_job_raises_for_missing_run() -> None:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        handle_byok_validation_job,
    )

    repo = _FakeValidationRunsRepo(run={"id": "other-run"})

    with pytest.raises(ValueError, match="missing_run"):
        await handle_byok_validation_job(
            {"id": "job-7", "payload": {"run_id": "run-1"}},
            repo=repo,
        )


@pytest.mark.asyncio
async def test_validation_job_overlays_candidate_on_one_frozen_server_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored credentials use Chat's endpoint/model generation without its key."""
    snapshot_calls = 0
    server_key = "sk-server-snapshot-must-not-dispatch"
    snapshot = {
        "custom_openai_api_2": {
            "api_key": server_key,
            "api_ip": "https://snapshot-a.example/v1",
            "model": "snapshot-model-a",
            "org_id": "snapshot-org-a",
            "project_id": "server-project-must-be-overlaid",
        }
    }

    def load_snapshot() -> dict[str, Any]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return copy.deepcopy(snapshot)

    adapter = _SnapshotRecordingAdapter()
    pool = _install_real_job_validation_boundary(
        monkeypatch,
        adapter=adapter,
        snapshot_loader=load_snapshot,
        capacity=1,
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-frozen-snapshot",
            "status": "queued",
            "org_id": None,
            "provider": "custom-openai-api-2",
        }
    )

    async def load_candidates(_run: dict[str, object]) -> validation_worker.CandidateLoadResult:
        return validation_worker.CandidateLoadResult(
            candidates=[
                {
                    "provider": "custom-openai-api-2",
                    "api_key": "sk-candidate-a",
                    "credential_fields": {"project_id": "candidate-project-a"},
                }
            ]
        )

    result = await validation_worker.handle_byok_validation_job(
        {"id": "job-frozen-snapshot", "payload": {"run_id": "run-frozen-snapshot"}},
        repo=repo,
        candidate_loader=load_candidates,
        test_provider_credentials_fn=byok_testing.test_provider_credentials,
    )

    assert result["status"] == "complete"
    assert result["valid_count"] == 1
    assert snapshot_calls == 1
    assert len(adapter.calls) == 1
    dispatched = adapter.calls[0]
    assert dispatched["api_key"] == "sk-candidate-a"
    assert dispatched["model"] == "snapshot-model-a"
    assert dispatched["credentials_resolved"] is True
    assert dispatched["app_config"]["custom_openai_api_2"] == {
        "api_ip": "https://snapshot-a.example/v1",
        "model": "snapshot-model-a",
        "org_id": "snapshot-org-a",
        "project_id": "candidate-project-a",
    }
    assert server_key not in repr(adapter.calls)
    assert "live-model-must-not-be-used" not in repr(adapter.calls)
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_concurrent_validation_job_candidates_share_one_frozen_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotating server config cannot mix endpoint/model/key generations in one scan."""
    snapshots = iter(
        [
            {
                "custom_openai_api_2": {
                    "api_key": "sk-server-generation-a",
                    "api_ip": "https://generation-a.example/v1",
                    "model": "model-a",
                    "org_id": "org-a",
                }
            },
            {
                "custom_openai_api_2": {
                    "api_key": "sk-server-generation-b",
                    "api_ip": "https://generation-b.example/v1",
                    "model": "model-b",
                    "org_id": "org-b",
                }
            },
        ]
    )
    snapshot_calls = 0

    def load_snapshot() -> dict[str, Any]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return copy.deepcopy(next(snapshots))

    adapter = _SnapshotRecordingAdapter(expected_calls=2, gated=True)
    pool = _install_real_job_validation_boundary(
        monkeypatch,
        adapter=adapter,
        snapshot_loader=load_snapshot,
        capacity=2,
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-concurrent-snapshot",
            "status": "queued",
            "org_id": None,
            "provider": "custom-openai-api-2",
        }
    )

    async def load_candidates(_run: dict[str, object]) -> validation_worker.CandidateLoadResult:
        return validation_worker.CandidateLoadResult(
            candidates=[
                {
                    "provider": "custom-openai-api-2",
                    "api_key": "sk-candidate-a",
                    "credential_fields": {"project_id": "candidate-project-a"},
                },
                {
                    "provider": "custom-openai-api-2",
                    "api_key": "sk-candidate-b",
                    "credential_fields": {"project_id": "candidate-project-b"},
                },
            ]
        )

    task = asyncio.create_task(
        validation_worker.handle_byok_validation_job(
            {
                "id": "job-concurrent-snapshot",
                "payload": {"run_id": "run-concurrent-snapshot"},
            },
            repo=repo,
            candidate_loader=load_candidates,
            test_provider_credentials_fn=byok_testing.test_provider_credentials,
        )
    )
    try:
        assert await asyncio.to_thread(adapter.all_entered.wait, 1.0)
    finally:
        adapter.release.set()
        outcomes = await asyncio.gather(task, return_exceptions=True)

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], dict)
    assert outcomes[0]["valid_count"] == 2
    assert snapshot_calls == 1
    assert len(adapter.calls) == 2
    by_key = {call["api_key"]: call for call in adapter.calls}
    assert set(by_key) == {"sk-candidate-a", "sk-candidate-b"}
    for candidate_key, project_id in (
        ("sk-candidate-a", "candidate-project-a"),
        ("sk-candidate-b", "candidate-project-b"),
    ):
        dispatched = by_key[candidate_key]
        assert dispatched["model"] == "model-a"
        assert dispatched["app_config"]["custom_openai_api_2"] == {
            "api_ip": "https://generation-a.example/v1",
            "model": "model-a",
            "org_id": "org-a",
            "project_id": project_id,
        }
    observed = repr(adapter.calls)
    assert "sk-server-generation-a" not in observed
    assert "sk-server-generation-b" not in observed
    assert "generation-b.example" not in observed
    assert "model-b" not in observed
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_phase",
    ["candidate_loader", "snapshot_loader", "candidate_overlay"],
)
async def test_validation_job_exposes_only_bounded_failure_to_jobs_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    """Internal validation failures cannot escape into Jobs error metadata."""
    sentinel = (
        f"sk-{failure_phase}-finalizer-secret "
        f"https://provider.example/private/{failure_phase}.json?token=secret"
    )
    adapter = _SnapshotRecordingAdapter()
    _install_real_job_validation_boundary(
        monkeypatch,
        adapter=adapter,
        snapshot_loader=lambda: {
            "custom_openai_api_2": {
                "api_ip": "https://snapshot.example/v1",
                "model": "snapshot-model",
            }
        },
        capacity=1,
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-finalizer-failure",
            "status": "queued",
            "org_id": None,
            "provider": "custom-openai-api-2",
        }
    )

    async def load_candidates(
        _run: dict[str, object],
    ) -> validation_worker.CandidateLoadResult:
        if failure_phase == "candidate_loader":
            raise RuntimeError(sentinel)
        return validation_worker.CandidateLoadResult(
            candidates=[
                {
                    "provider": "custom-openai-api-2",
                    "api_key": "sk-candidate",
                    "credential_fields": None,
                }
            ]
        )

    def fail_snapshot() -> dict[str, Any]:
        raise RuntimeError(sentinel)

    if failure_phase == "snapshot_loader":
        monkeypatch.setattr(
            validation_worker,
            "load_server_config_snapshot",
            fail_snapshot,
        )
    elif failure_phase == "candidate_overlay":
        monkeypatch.setattr(
            validation_worker,
            "merge_server_fallback_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(sentinel)),
        )

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await validation_worker.handle_byok_validation_job(
            {
                "id": "job-finalizer-failure",
                "payload": {"run_id": "run-finalizer-failure"},
            },
            repo=repo,
            candidate_loader=load_candidates,
            test_provider_credentials_fn=byok_testing.test_provider_credentials,
        )

    assert repo.complete_calls == []
    assert repo.failed_calls == [
        ("run-finalizer-failure", "provider_validation_failed")
    ]
    assert adapter.calls == []
    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.message == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)
    assert sentinel not in repr(repo.failed_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_method",
    ["get_run", "mark_running", "mark_complete"],
)
async def test_validation_job_sanitizes_repository_lifecycle_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure_method: str,
) -> None:
    """Repository failures cannot escape into Jobs error metadata."""
    sentinel = (
        f"sk-{failure_method}-finalizer-secret "
        f"postgresql://admin:secret@db.internal/{failure_method}"
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-repository-failure",
            "status": "queued",
            "org_id": None,
            "provider": "openai",
        }
    )

    async def fail_repository_call(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(sentinel)

    async def load_no_candidates(
        _run: dict[str, object],
    ) -> list[validation_worker.ByokValidationCandidate]:
        return []

    async def unused_validator(**_kwargs: Any) -> str:
        raise AssertionError("No candidates should reach the validator")

    monkeypatch.setattr(repo, failure_method, fail_repository_call)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await validation_worker.handle_byok_validation_job(
            {
                "id": "job-repository-failure",
                "payload": {"run_id": "run-repository-failure"},
            },
            repo=repo,
            candidate_loader=load_no_candidates,
            test_provider_credentials_fn=unused_validator,
        )

    assert repo.failed_calls == [
        ("run-repository-failure", "provider_validation_failed")
    ]
    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.message == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in repr(exc_info.value)
    assert sentinel not in repr(repo.failed_calls)


@pytest.mark.asyncio
async def test_validation_job_sanitizes_failure_when_mark_failed_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed best-effort status update cannot replace the bounded failure."""
    loader_sentinel = (
        "sk-loader-finalizer-secret "
        "https://provider.example/private/loader?token=secret"
    )
    repository_sentinel = (
        "sk-mark-failed-finalizer-secret "
        "postgresql://admin:secret@db.internal/mark-failed"
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-mark-failed-failure",
            "status": "queued",
            "org_id": None,
            "provider": "openai",
        }
    )

    async def fail_candidate_load(
        _run: dict[str, object],
    ) -> list[validation_worker.ByokValidationCandidate]:
        raise RuntimeError(loader_sentinel)

    async def fail_mark_failed(run_id: str, *, error_message: str) -> None:
        repo.failed_calls.append((run_id, error_message))
        raise RuntimeError(repository_sentinel)

    async def unused_validator(**_kwargs: Any) -> str:
        raise AssertionError("Candidate loading must fail before validation")

    monkeypatch.setattr(repo, "mark_failed", fail_mark_failed)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await validation_worker.handle_byok_validation_job(
            {
                "id": "job-mark-failed-failure",
                "payload": {"run_id": "run-mark-failed-failure"},
            },
            repo=repo,
            candidate_loader=fail_candidate_load,
            test_provider_credentials_fn=unused_validator,
        )

    assert repo.failed_calls == [
        ("run-mark-failed-failure", "provider_validation_failed")
    ]
    assert exc_info.value.code == "provider_unavailable"
    assert exc_info.value.message == PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"]
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    rendered_failure = repr(exc_info.value)
    assert loader_sentinel not in rendered_failure
    assert repository_sentinel not in rendered_failure
    assert loader_sentinel not in repr(repo.failed_calls)
    assert repository_sentinel not in repr(repo.failed_calls)


@pytest.mark.asyncio
async def test_validation_job_preserves_cancellation_without_marking_failed() -> None:
    """Task cancellation remains control flow rather than a Jobs failure."""
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-cancelled",
            "status": "queued",
            "org_id": None,
            "provider": "openai",
        }
    )
    loader_started = asyncio.Event()
    keep_loader_running = asyncio.Event()

    async def blocked_candidate_load(
        _run: dict[str, object],
    ) -> list[validation_worker.ByokValidationCandidate]:
        loader_started.set()
        await keep_loader_running.wait()
        return []

    async def unused_validator(**_kwargs: Any) -> str:
        raise AssertionError("Cancelled loading must not reach validation")

    task = asyncio.create_task(
        validation_worker.handle_byok_validation_job(
            {
                "id": "job-cancelled",
                "payload": {"run_id": "run-cancelled"},
            },
            repo=repo,
            candidate_loader=blocked_candidate_load,
            test_provider_credentials_fn=unused_validator,
        )
    )
    await asyncio.wait_for(loader_started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert repo.complete_calls == []
    assert repo.failed_calls == []


@pytest.mark.asyncio
async def test_validation_job_rejects_malformed_candidate_fields_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt stored credential fields fail closed without reaching an adapter."""
    adapter = _SnapshotRecordingAdapter()
    _install_real_job_validation_boundary(
        monkeypatch,
        adapter=adapter,
        snapshot_loader=lambda: {
            "custom_openai_api_2": {
                "api_ip": "https://snapshot.example/v1",
                "model": "snapshot-model",
            }
        },
        capacity=1,
    )
    repo = _FakeValidationRunsRepo(
        run={
            "id": "run-malformed-fields",
            "status": "queued",
            "org_id": None,
            "provider": "custom-openai-api-2",
        }
    )

    async def load_candidates(
        _run: dict[str, object],
    ) -> validation_worker.CandidateLoadResult:
        return validation_worker.CandidateLoadResult(
            candidates=[
                {
                    "provider": "custom-openai-api-2",
                    "api_key": "sk-candidate",
                    "credential_fields": [],  # type: ignore[typeddict-item]
                }
            ]
        )

    with pytest.raises(ByokResolutionError):
        await validation_worker.handle_byok_validation_job(
            {
                "id": "job-malformed-fields",
                "payload": {"run_id": "run-malformed-fields"},
            },
            repo=repo,
            candidate_loader=load_candidates,
            test_provider_credentials_fn=byok_testing.test_provider_credentials,
        )

    assert adapter.calls == []
    assert repo.complete_calls == []
    assert repo.failed_calls == [
        ("run-malformed-fields", "provider_validation_failed")
    ]


@pytest.mark.parametrize(
    ("neutral", "legacy", "expected"),
    [
        ("3", "7", 3),
        (None, "5", 5),
        (None, None, 2),
        ("not-an-integer", None, 2),
        ("0", "5", 5),
        ("999", "4", 8),
        (None, "999", 8),
    ],
    ids=[
        "neutral-precedence",
        "legacy-fallback",
        "default",
        "invalid-neutral-default",
        "zero-neutral-legacy",
        "neutral-clamped",
        "legacy-clamped",
    ],
)
def test_validation_worker_uses_shared_bounded_per_provider_capacity(
    monkeypatch: pytest.MonkeyPatch,
    neutral: str | None,
    legacy: str | None,
    expected: int,
) -> None:
    for name, value in (
        ("PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY", neutral),
        ("ADMIN_BYOK_VALIDATION_PER_PROVIDER_CONCURRENCY", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert validation_worker._per_provider_limit() == expected


@pytest.mark.asyncio
async def test_oversized_legacy_capacity_cannot_create_unbounded_scan_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "PROVIDER_CREDENTIAL_VALIDATION_PER_PROVIDER_CONCURRENCY",
        raising=False,
    )
    monkeypatch.setenv(
        "ADMIN_BYOK_VALIDATION_PER_PROVIDER_CONCURRENCY",
        "999999",
    )
    original_create_task = validation_worker.asyncio.create_task
    created_tasks: list[asyncio.Task[Any]] = []

    def tracked_create_task(coro: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(validation_worker.asyncio, "create_task", tracked_create_task)

    async def validate_candidate(**_kwargs: Any) -> str:
        await asyncio.sleep(0)
        return "ok"

    candidates: list[validation_worker.ByokValidationCandidate] = [
        {
            "provider": "openai",
            "api_key": f"sk-candidate-{index}",
            "credential_fields": None,
        }
        for index in range(40)
    ]

    summary = await validation_worker._run_validation_scan(
        candidates,
        test_provider_credentials_fn=validate_candidate,
    )

    assert summary["valid_count"] == 40
    assert 1 <= len(created_tasks) <= 8
    assert all(task.done() for task in created_tasks)
