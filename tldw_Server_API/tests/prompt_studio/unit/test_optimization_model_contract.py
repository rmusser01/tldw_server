"""Durable, secret-free model configuration contract for Prompt Studio optimization."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as optimization_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationConfig,
    OptimizationCreate,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
    OptimizationSimpleCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    ACCEPTED_OPTIMIZATION_STRATEGIES,
    optimization_execution_strategy,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter import (
    PromptStudioJobsAdapter,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    OptimizationEngine,
)

pytestmark = pytest.mark.unit

_SECRET_SENTINEL = "TASK12963_OPTIMIZATION_SECRET_SENTINEL"
_RUNTIME_SENTINEL = "TASK12963_OPTIMIZATION_RUNTIME_SENTINEL"

_CONVENTIONAL_SECRET_TOKENS = frozenset(
    {
        "authorization",
        "auth",
        "password",
        "secret",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "token",
        "cookie",
        "jwt",
    }
)

_BEDROCK_MODEL_CONFIG = {
    "provider": "bedrock",
    "model": "anthropic.claude-3-haiku",
    "parameters": {
        "temperature": 0.23,
        "max_tokens": 128,
        "timeout_seconds": 17,
    },
}


class _RuntimeHandle:
    def __repr__(self) -> str:
        return _RUNTIME_SENTINEL


class _CapturingDb:
    client_id = "optimization-contract"

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[tuple[int, dict[str, Any]]] = []

    def get_prompt_with_project(self, prompt_id: int, include_deleted: bool = False) -> dict[str, Any]:
        return {
            "id": prompt_id,
            "project_id": 71,
            "project_user_id": "owner-a",
        }

    @staticmethod
    def get_test_case(test_case_id: int) -> dict[str, int]:
        return {"id": test_case_id, "project_id": 71}

    @staticmethod
    def get_test_cases_by_ids(test_case_ids: list[int]) -> list[dict[str, int]]:
        return [
            {"id": test_case_id, "project_id": 71}
            for test_case_id in test_case_ids
        ]

    def create_optimization(self, **kwargs: Any) -> dict[str, Any]:
        captured = copy.deepcopy(kwargs)
        self.created.append(captured)
        optimization_id = 900 + len(self.created)
        return {
            "id": optimization_id,
            "uuid": f"optimization-{optimization_id}",
            **captured,
        }

    def update_optimization(self, optimization_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        captured = copy.deepcopy(updates)
        self.updated.append((optimization_id, captured))
        return {
            "id": optimization_id,
            "uuid": f"optimization-{optimization_id}",
            **captured,
        }


class _LegacyOptimizationDb:
    client_id = "legacy-optimization-contract"

    def __init__(self, optimization_config: dict[str, Any]) -> None:
        self.row: dict[str, Any] = {
            "id": 44,
            "project_id": 7,
            "initial_prompt_id": 12,
            "optimizer_type": "mipro",
            "optimization_config": copy.deepcopy(optimization_config),
            "test_case_ids": [3],
            "max_iterations": 1,
            "status": "pending",
        }
        self.config_updates: list[dict[str, Any]] = []

    def get_optimization(self, optimization_id: int, **_kwargs: Any) -> dict[str, Any] | None:
        if optimization_id != self.row["id"]:
            return None
        return copy.deepcopy(self.row)

    def update_optimization(self, optimization_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        assert optimization_id == self.row["id"]
        captured = copy.deepcopy(updates)
        if "optimization_config" in captured:
            self.config_updates.append(copy.deepcopy(captured["optimization_config"]))
        self.row.update(captured)
        return copy.deepcopy(self.row)

    def set_optimization_status(
        self,
        optimization_id: int,
        status: str,
        **_kwargs: Any,
    ) -> None:
        assert optimization_id == self.row["id"]
        self.row["status"] = status

    def complete_optimization(self, optimization_id: int, **updates: Any) -> None:
        assert optimization_id == self.row["id"]
        self.row.update(updates)
        self.row["status"] = "completed"


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": []})


def _legacy_model_aliases() -> dict[str, Any]:
    return {
        "provider": "AWS_BEDROCK",
        "model_name": "anthropic.claude-3-haiku",
        "temperature": 0.23,
        "max_tokens": 128,
        "timeout_seconds": 17,
    }


def _assert_secret_free(value: Any) -> None:
    serialized = json.dumps(value, default=repr, sort_keys=True)
    assert _SECRET_SENTINEL not in serialized
    assert _RUNTIME_SENTINEL not in serialized

    def _walk(current: Any) -> None:
        if isinstance(current, dict):
            for key, nested in current.items():
                token = re.sub(r"[^a-z0-9]", "", str(key).lower())
                assert not (
                    token == "apikey"
                    or token.endswith("apikey")
                    or token == "appconfig"
                    or "credential" in token
                    or ("runtime" in token and "handle" in token)
                ), f"sensitive durable key survived: {key}"
                _walk(nested)
        elif isinstance(current, (list, tuple)):
            for nested in current:
                _walk(nested)

    _walk(value)


def _assert_no_conventional_secret_aliases(value: Any) -> None:
    """Assert an explicit secret-key denylist independent of production logic."""
    serialized = json.dumps(value, default=repr, sort_keys=True)
    assert _SECRET_SENTINEL not in serialized

    def _walk(current: Any) -> None:
        if isinstance(current, dict):
            for key, nested in current.items():
                token = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                assert token not in _CONVENTIONAL_SECRET_TOKENS, (
                    f"conventional secret alias survived: {key}"
                )
                _walk(nested)
        elif isinstance(current, (list, tuple)):
            for nested in current:
                _walk(nested)

    _walk(value)


def _conventional_secret_alias_payload() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical in (
        "authorization",
        "auth",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "token",
        "cookie",
        "jwt",
    ):
        parts = canonical.split("_")
        variants = {
            canonical,
            canonical.upper(),
            "-".join(parts),
            "".join(part.title() for part in parts),
        }
        for variant in variants:
            aliases[variant] = f"{_SECRET_SENTINEL}:{variant}"
    return aliases


def _capture_jobs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def _create_job(self: Any, **kwargs: Any) -> dict[str, Any]:
        captured.append(copy.deepcopy(kwargs))
        return {"id": 800 + len(captured), "status": "queued"}

    monkeypatch.setattr(
        optimization_endpoint.PromptStudioJobsAdapter,
        "create_job",
        _create_job,
        raising=True,
    )
    return captured


@pytest.mark.asyncio
async def test_simple_submission_uses_prompt_project_not_compatibility_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)
    require_write = AsyncMock(return_value=True)
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        require_write,
        raising=True,
    )
    payload = OptimizationSimpleCreateRequest.model_validate(
        {
            "prompt_id": 12,
            "project_id": 999,
            "name": "WebUI compatibility run",
            "description": "Accepted for compatibility with the checked-in wizard",
            "model_config": _legacy_model_aliases(),
            "test_case_ids": [3],
            "config": {
                "strategy": "mipro",
                "max_iterations": 1,
            },
        }
    )

    await optimization_endpoint.create_optimization_simple(
        payload=payload,
        request=_request("/api/v1/prompt-studio/optimizations"),
        db=db,
        user_context={"user_id": "owner-a"},
    )

    assert db.created[0]["project_id"] == 71
    assert db.created[0]["name"] == "WebUI compatibility run"
    assert db.created[0]["optimizer_type"] == "mipro"
    assert db.created[0]["optimization_config"]["model_config"] == (
        _BEDROCK_MODEL_CONFIG
    )
    assert db.updated == [(901, {"test_case_ids": [3]})]
    assert jobs[0]["payload"]["project_id"] == 71
    assert jobs[0]["payload"]["test_case_ids"] == [3]
    assert jobs[0]["payload"]["optimizer_type"] == "mipro"
    require_write.assert_awaited_once()
    assert require_write.await_args.args[0] == 71


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["simple", "compare"])
async def test_admin_submission_tenants_core_job_to_project_owner_and_keeps_actor_audit(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        AsyncMock(return_value=True),
        raising=True,
    )
    admin_context = {"user_id": "admin-b", "is_admin": True}

    if entrypoint == "simple":
        await optimization_endpoint.create_optimization_simple(
            payload=OptimizationSimpleCreateRequest.model_validate(
                {
                    "prompt_id": 12,
                    "test_case_ids": [3],
                    "config": {
                        "optimizer_type": "mipro",
                        "max_iterations": 1,
                    },
                }
            ),
            request=_request("/api/v1/prompt-studio/optimizations"),
            db=db,
            user_context=admin_context,
        )
    else:
        await optimization_endpoint.compare_strategies(
            request=CompareStrategiesRequest.model_validate(
                {
                    "prompt_id": 12,
                    "test_case_ids": [3],
                    "strategies": ["mipro", "bootstrap"],
                }
            ),
            http_request=_request(
                "/api/v1/prompt-studio/optimizations/compare-strategies"
            ),
            _=True,
            db=db,
            user_context=admin_context,
        )

    assert {job["user_id"] for job in jobs} == {"owner-a"}
    assert {job["payload"]["created_by"] for job in jobs} == {"admin-b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["simple", "full", "compare"])
async def test_optimization_submission_rejects_cross_project_test_cases(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CrossProjectDb(_CapturingDb):
        @staticmethod
        def get_test_case(test_case_id: int) -> dict[str, int]:
            return {"id": test_case_id, "project_id": 72}

        @staticmethod
        def get_test_cases_by_ids(test_case_ids: list[int]) -> list[dict[str, int]]:
            return [
                {"id": test_case_id, "project_id": 72}
                for test_case_id in test_case_ids
            ]

    db = _CrossProjectDb()
    jobs = _capture_jobs(monkeypatch)
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        AsyncMock(return_value=True),
        raising=True,
    )

    with pytest.raises(HTTPException):
        if entrypoint == "simple":
            await optimization_endpoint.create_optimization_simple(
                payload=OptimizationSimpleCreateRequest.model_validate(
                    {
                        "prompt_id": 12,
                        "test_case_ids": [3],
                        "config": {"strategy": "mipro"},
                    }
                ),
                request=_request("/api/v1/prompt-studio/optimizations"),
                db=db,
                user_context={"user_id": "owner-a"},
            )
        elif entrypoint == "full":
            request_model = OptimizationCreate.model_validate(
                {
                    "project_id": 71,
                    "initial_prompt_id": 12,
                    "test_case_ids": [3],
                    "optimization_config": {
                        "optimizer_type": "mipro",
                        "max_iterations": 1,
                        "target_metric": "accuracy",
                        "model_config": _BEDROCK_MODEL_CONFIG,
                    },
                }
            )
            await optimization_endpoint.create_optimization(
                optimization_data=request_model,
                request=_request("/api/v1/prompt-studio/optimizations/create"),
                _=True,
                db=db,
                security_config={},
                user_context={"user_id": "owner-a"},
                idempotency_key=None,
            )
        else:
            request_model = CompareStrategiesRequest.model_validate(
                {
                    "prompt_id": 12,
                    "test_case_ids": [3],
                    "strategies": ["mipro", "bootstrap"],
                    "model_configuration": _BEDROCK_MODEL_CONFIG,
                }
            )
            await optimization_endpoint.compare_strategies(
                request=request_model,
                http_request=_request(
                    "/api/v1/prompt-studio/optimizations/compare"
                ),
                _=True,
                db=db,
                user_context={"user_id": "owner-a"},
            )

    assert db.created == []
    assert jobs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["simple", "full", "compare"])
async def test_optimization_submission_rejects_empty_test_cases_before_side_effects(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)
    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        AsyncMock(return_value=True),
        raising=True,
    )

    with pytest.raises(ValidationError) as exc_info:
        if entrypoint == "simple":
            await optimization_endpoint.create_optimization_simple(
                payload=OptimizationSimpleCreateRequest.model_validate(
                    {"prompt_id": 12, "test_case_ids": []}
                ),
                request=_request("/api/v1/prompt-studio/optimizations"),
                db=db,
                user_context={"user_id": "owner-a"},
            )
        elif entrypoint == "full":
            await optimization_endpoint.create_optimization(
                optimization_data=OptimizationCreate.model_validate(
                    {
                        "project_id": 71,
                        "initial_prompt_id": 12,
                        "test_case_ids": [],
                        "optimization_config": {
                            "optimizer_type": "mipro",
                            "max_iterations": 1,
                            "target_metric": "accuracy",
                        },
                    }
                ),
                request=_request("/api/v1/prompt-studio/optimizations/create"),
                _=True,
                db=db,
                security_config={},
                user_context={"user_id": "owner-a"},
                idempotency_key=None,
            )
        else:
            await optimization_endpoint.compare_strategies(
                request=CompareStrategiesRequest.model_validate(
                    {
                        "prompt_id": 12,
                        "test_case_ids": [],
                        "strategies": ["mipro", "bootstrap"],
                    }
                ),
                http_request=_request(
                    "/api/v1/prompt-studio/optimizations/compare"
                ),
                _=True,
                db=db,
                user_context={"user_id": "owner-a"},
            )

    assert any(
        tuple(error["loc"]) == ("test_case_ids",)
        and error["type"] == "too_short"
        for error in exc_info.value.errors()
    )
    assert db.created == []
    assert jobs == []


@pytest.mark.asyncio
async def test_advertised_strategies_have_explicit_durable_execution_contract() -> None:
    response = await optimization_endpoint.get_optimization_strategies()
    advertised = {
        str(strategy["name"]): str(
            strategy.get("execution_strategy") or strategy["name"]
        )
        for strategy in response.data
    }

    assert advertised == {
        "mipro": "mipro",
        "bootstrap": "bootstrap",
        "iterative": "iterative",
        "mcts": "mcts",
    }
    assert set(advertised) <= ACCEPTED_OPTIMIZATION_STRATEGIES
    assert {
        strategy: optimization_execution_strategy(strategy)
        for strategy in advertised
    } == advertised


def test_strategy_discovery_route_precedes_dynamic_job_route_and_matches_webui_contract():
    app = FastAPI()
    app.include_router(optimization_endpoint.router)

    response = TestClient(app).get(
        "/api/v1/prompt-studio/optimizations/strategies"
    )

    assert response.status_code == 200, response.text
    strategies = response.json()["data"]
    assert strategies
    for strategy in strategies:
        assert isinstance(strategy["supported_params"], list)
        assert isinstance(strategy["default_params"], dict)
        assert isinstance(strategy["requires_test_cases"], bool)
        assert isinstance(strategy["supports_early_stopping"], bool)


def test_compare_request_normalizes_legacy_aliases_to_canonical_model_config() -> None:
    request = CompareStrategiesRequest.model_validate(
        {
            "prompt_id": 12,
            "test_case_ids": [3],
            "strategies": ["mipro", "bootstrap"],
            "model_configuration": _legacy_model_aliases(),
        }
    )

    assert request.model_configuration == _BEDROCK_MODEL_CONFIG


def test_compare_request_rejects_ambiguous_model_name_without_provider() -> None:
    with pytest.raises(ValidationError, match="provider"):
        CompareStrategiesRequest.model_validate(
            {
                "prompt_id": 12,
                "test_case_ids": [3],
                "strategies": ["bootstrap", "mipro"],
                "model_configuration": {
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.4,
                },
            }
        )


def test_canonical_request_preserves_nested_nonsecret_json_parameters() -> None:
    model_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "parameters": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            }
        },
    }

    request = CompareStrategiesRequest.model_validate(
        {
            "prompt_id": 12,
            "test_case_ids": [3],
            "strategies": ["mipro", "bootstrap"],
            "model_configuration": model_config,
        }
    )

    assert request.model_configuration == model_config


def test_simple_request_normalizes_model_aliases_without_losing_strategy_fields() -> None:
    request = OptimizationSimpleCreateRequest.model_validate(
        {
            "prompt_id": 12,
            "test_case_ids": [3],
            "config": {
                "optimizer_type": "mipro",
                "max_iterations": 4,
                "target_metric": "accuracy",
                "model_configuration": _legacy_model_aliases(),
            },
        }
    )

    assert request.config == {
        "optimizer_type": "mipro",
        "max_iterations": 4,
        "target_metric": "accuracy",
        "model_config": _BEDROCK_MODEL_CONFIG,
    }


def test_full_create_config_normalizes_legacy_model_configuration() -> None:
    config = OptimizationConfig.model_validate(
        {
            "optimizer_type": "mipro",
            "max_iterations": 4,
            "target_metric": "accuracy",
            "model_configuration": _legacy_model_aliases(),
        }
    )

    dumped = config.model_dump(exclude_none=True)
    assert dumped["model_config"] == _BEDROCK_MODEL_CONFIG
    assert "model_configuration" not in dumped


_SENSITIVE_BOUNDARY_CASES = [
    pytest.param("full", "top", "API_KEY", id="full-top-api-key-case"),
    pytest.param("full", "strategy", "App_Config", id="full-strategy-app-config"),
    pytest.param("full", "strategy_nested", "credentialsResolved", id="full-strategy-credentials"),
    pytest.param("full", "canonical_model", "CredentialFields", id="full-model-credential-fields"),
    pytest.param("full", "legacy_model", "runtimeHandle", id="full-legacy-runtime-handle"),
    pytest.param("full", "durable_nested", "OpenAIApiKey", id="full-checkpoint-provider-key"),
    pytest.param("simple", "top", "api-key", id="simple-top-api-key-alias"),
    pytest.param("simple", "strategy", "CREDENTIALS", id="simple-strategy-credentials"),
    pytest.param("simple", "canonical_model", "appConfig", id="simple-model-app-config"),
    pytest.param("simple", "legacy_model", "credentials_resolved", id="simple-legacy-resolved"),
    pytest.param("simple", "durable_nested", "providerCredentialRuntime", id="simple-resume-runtime"),
    pytest.param("simple", "strategy_nested", "Anthropic_API_Key", id="simple-nested-provider-key"),
    pytest.param("compare", "request_top", "APP_CONFIG", id="compare-top-app-config"),
    pytest.param("compare", "canonical_model", "apiKey", id="compare-model-api-key-alias"),
    pytest.param("compare", "canonical_model", "Credential_Fields", id="compare-model-credential-fields"),
    pytest.param("compare", "parameter_nested", "Runtime-Handle", id="compare-parameter-runtime"),
    pytest.param("compare", "parameter_nested", "CredentialsResolved", id="compare-parameter-resolved"),
    pytest.param("compare", "canonical_model", "custom_openai_api_key", id="compare-provider-api-key"),
]


def _secret_value(field_name: str) -> Any:
    normalized = re.sub(r"[^a-z0-9]", "", field_name.lower())
    if "runtime" in normalized or "handle" in normalized:
        return _RuntimeHandle()
    return {"sentinel": _SECRET_SENTINEL}


def _hostile_submission_payload(entrypoint: str, location: str, field_name: str) -> dict[str, Any]:
    value = _secret_value(field_name)
    safe_model = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "parameters": {"temperature": 0.2},
    }

    if entrypoint == "compare":
        payload: dict[str, Any] = {
            "prompt_id": 12,
            "test_case_ids": [3],
            "strategies": ["mipro", "bootstrap"],
            "model_configuration": copy.deepcopy(safe_model),
        }
        if location == "request_top":
            payload[field_name] = value
        elif location == "parameter_nested":
            payload["model_configuration"]["parameters"]["response_format"] = {
                "nested": [{field_name: value}]
            }
        else:
            payload["model_configuration"][field_name] = value
        return payload

    config: dict[str, Any] = {
        "optimizer_type": "mipro",
        "max_iterations": 1,
        "target_metric": "accuracy",
        "model_config": copy.deepcopy(safe_model),
    }
    if location == "top":
        config[field_name] = value
    elif location == "strategy":
        config["strategy_params"] = {field_name: value}
    elif location == "strategy_nested":
        config["strategy_params"] = {"candidate": [{"nested": {field_name: value}}]}
    elif location == "canonical_model":
        config["model_config"][field_name] = value
    elif location == "legacy_model":
        config.pop("model_config")
        config["model_configuration"] = copy.deepcopy(safe_model)
        config["model_configuration"][field_name] = value
    elif location == "durable_nested":
        durable_key = "strategy_params" if entrypoint == "full" else "resume_state"
        config[durable_key] = {"checkpoint": [{"nested": {field_name: value}}]}
    else:  # pragma: no cover - guarded by the parameter table
        raise AssertionError(f"unsupported secret location: {location}")

    if entrypoint == "full":
        return {
            "project_id": 71,
            "initial_prompt_id": 12,
            "optimization_config": config,
        }
    return {"prompt_id": 12, "config": config}


@pytest.mark.parametrize(("entrypoint", "location", "field_name"), _SENSITIVE_BOUNDARY_CASES)
@pytest.mark.asyncio
async def test_sensitive_fields_never_cross_db_or_jobs_durable_boundaries(
    entrypoint: str,
    location: str,
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)

    async def _allow_write(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write,
        raising=True,
    )
    payload = _hostile_submission_payload(entrypoint, location, field_name)
    rejected = False

    try:
        if entrypoint == "full":
            request_model = OptimizationCreate.model_validate(payload)
            await optimization_endpoint.create_optimization(
                optimization_data=request_model,
                request=_request("/api/v1/prompt-studio/optimizations/create"),
                _=True,
                db=db,
                security_config={},
                user_context={"user_id": "owner-a"},
                idempotency_key=None,
            )
        elif entrypoint == "simple":
            request_model = OptimizationSimpleCreateRequest.model_validate(payload)
            await optimization_endpoint.create_optimization_simple(
                payload=request_model,
                request=_request("/api/v1/prompt-studio/optimizations"),
                db=db,
                user_context={"user_id": "owner-a"},
            )
        else:
            request_model = CompareStrategiesRequest.model_validate(payload)
            await optimization_endpoint.compare_strategies(
                request=request_model,
                http_request=_request("/api/v1/prompt-studio/optimizations/compare"),
                _=True,
                db=db,
                user_context={"user_id": "owner-a"},
            )
    except ValidationError:
        rejected = True
    except HTTPException as exc:
        assert 400 <= exc.status_code < 500
        rejected = True

    assert rejected or db.created or jobs
    for durable_value in (*db.created, *jobs):
        _assert_secret_free(durable_value)


@pytest.mark.parametrize("entrypoint", ["full", "simple", "compare"])
@pytest.mark.asyncio
async def test_conventional_secret_aliases_never_reach_actual_prompt_or_jobs_databases(
    entrypoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise recursive aliases through real endpoint persistence boundaries."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / f"jobs-{entrypoint}.sqlite"))
    monkeypatch.setenv("JOBS_SECRET_REDACT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "0")

    async def _allow_write(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write,
        raising=True,
    )
    db = PromptStudioDatabase(
        str(tmp_path / f"prompt-studio-{entrypoint}.sqlite"),
        client_id=f"secret-alias-{entrypoint}",
    )
    project = db.create_project(name=f"Secret alias {entrypoint}", description="")
    prompt = db.create_prompt(
        project_id=int(project["id"]),
        name="Secret alias prompt",
        system_prompt="Answer safely.",
        user_prompt="Question: {question}",
    )
    test_case = db.create_test_case(
        project_id=int(project["id"]),
        name="Secret alias case",
        inputs={"question": "hello"},
        expected_outputs={"response": "hello"},
    )
    hostile = {
        "level_one": [
            {
                "level_two": _conventional_secret_alias_payload(),
            }
        ]
    }
    safe_model = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "parameters": {"temperature": 0.2},
    }
    rejected = False
    optimization_ids: list[int] = []

    try:
        if entrypoint == "full":
            request_model = OptimizationCreate.model_validate(
                {
                    "project_id": project["id"],
                    "initial_prompt_id": prompt["id"],
                    "test_case_ids": [test_case["id"]],
                    "optimization_config": {
                        "optimizer_type": "mipro",
                        "max_iterations": 1,
                        "target_metric": "accuracy",
                        "model_config": safe_model,
                        "strategy_params": {"checkpoint": hostile},
                    },
                }
            )
            response = await optimization_endpoint.create_optimization(
                optimization_data=request_model,
                request=_request("/api/v1/prompt-studio/optimizations/create"),
                _=True,
                db=db,
                security_config={},
                user_context={"user_id": "7"},
                idempotency_key=None,
            )
            response_data = response.model_dump().get("data") or {}
            optimization_ids.append(
                int((response_data.get("optimization") or {})["id"])
            )
        elif entrypoint == "simple":
            request_model = OptimizationSimpleCreateRequest.model_validate(
                {
                    "prompt_id": prompt["id"],
                    "config": {
                        "optimizer_type": "mipro",
                        "max_iterations": 1,
                        "target_metric": "accuracy",
                        "model_config": safe_model,
                        "resume_state": hostile,
                    },
                }
            )
            await optimization_endpoint.create_optimization_simple(
                payload=request_model,
                request=_request("/api/v1/prompt-studio/optimizations"),
                db=db,
                user_context={"user_id": "7"},
            )
        else:
            compare_model = copy.deepcopy(safe_model)
            compare_model["parameters"]["response_format"] = hostile
            request_model = CompareStrategiesRequest.model_validate(
                {
                    "prompt_id": prompt["id"],
                    "test_case_ids": [test_case["id"]],
                    "strategies": ["mipro", "bootstrap"],
                    "model_configuration": compare_model,
                }
            )
            response = await optimization_endpoint.compare_strategies(
                request=request_model,
                http_request=_request(
                    "/api/v1/prompt-studio/optimizations/compare"
                ),
                _=True,
                db=db,
                user_context={"user_id": "7"},
            )
            response_data = response.model_dump().get("data") or {}
            optimization_ids.extend(
                int(value)
                for value in response_data.get("optimization_ids") or []
            )
    except ValidationError:
        rejected = True
    except HTTPException as exc:
        assert 400 <= exc.status_code < 500
        rejected = True

    try:
        jobs = PromptStudioJobsAdapter().list_jobs(
            db=db,
            user_id="7",
            job_type="optimization",
            limit=100,
        )
        if rejected:
            assert jobs == []
            return

        assert jobs
        if entrypoint == "simple":
            optimization_ids.extend(
                int(job["entity_id"])
                for job in jobs
                if job.get("entity_id") is not None
            )
        rows = [
            db.get_optimization(optimization_id, include_deleted=True)
            for optimization_id in optimization_ids
        ]
        assert rows
        assert all(row is not None for row in rows)
        for row in rows:
            assert row is not None
            _assert_no_conventional_secret_aliases(row["optimization_config"])
        for job in jobs:
            payload = job.get("payload") or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            _assert_no_conventional_secret_aliases(payload)
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_simple_endpoint_persists_the_same_canonical_config_to_db_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)

    async def _allow_write(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write,
        raising=True,
    )
    request_model = OptimizationSimpleCreateRequest.model_validate(
        {
            "prompt_id": 12,
            "test_case_ids": [3],
            "config": {
                "optimizer_type": "mipro",
                "max_iterations": 4,
                "target_metric": "accuracy",
                "model_configuration": _legacy_model_aliases(),
            },
        }
    )

    await optimization_endpoint.create_optimization_simple(
        payload=request_model,
        request=_request("/api/v1/prompt-studio/optimizations"),
        db=db,
        user_context={"user_id": "owner-a"},
    )

    db_config = db.created[0]["optimization_config"]
    job_config = jobs[0]["payload"]["optimization_config"]
    assert db_config == job_config == request_model.config
    assert db_config["model_config"] == _BEDROCK_MODEL_CONFIG
    assert "model_configuration" not in db_config
    _assert_secret_free(db_config)
    _assert_secret_free(jobs[0]["payload"])


@pytest.mark.asyncio
async def test_full_create_endpoint_persists_the_same_canonical_config_to_db_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)

    async def _allow_write(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write,
        raising=True,
    )
    request_model = OptimizationCreate.model_validate(
        {
            "project_id": 71,
            "initial_prompt_id": 12,
            "test_case_ids": [3],
            "optimization_config": {
                "optimizer_type": "mipro",
                "max_iterations": 4,
                "target_metric": "accuracy",
                "model_configuration": _legacy_model_aliases(),
            },
        }
    )

    await optimization_endpoint.create_optimization(
        optimization_data=request_model,
        request=_request("/api/v1/prompt-studio/optimizations/create"),
        _=True,
        db=db,
        security_config={},
        user_context={"user_id": "owner-a"},
        idempotency_key=None,
    )

    db_config = db.created[0]["optimization_config"]
    job_config = jobs[0]["payload"]["optimization_config"]
    assert db_config == job_config
    assert db_config["model_config"] == _BEDROCK_MODEL_CONFIG
    assert "model_configuration" not in db_config
    _assert_secret_free(db_config)
    _assert_secret_free(jobs[0]["payload"])


@pytest.mark.asyncio
async def test_compare_endpoint_persists_the_same_canonical_config_to_db_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _CapturingDb()
    jobs = _capture_jobs(monkeypatch)

    async def _allow_write(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        optimization_endpoint,
        "require_project_write_access",
        _allow_write,
        raising=True,
    )
    request_model = CompareStrategiesRequest.model_validate(
        {
            "prompt_id": 12,
            "test_case_ids": [3],
            "strategies": ["mipro", "bootstrap"],
            "model_configuration": _legacy_model_aliases(),
        }
    )

    await optimization_endpoint.compare_strategies(
        request=request_model,
        http_request=_request("/api/v1/prompt-studio/optimizations/compare"),
        _=True,
        db=db,
        user_context={"user_id": "owner-a"},
    )

    db_config = db.created[0]["optimization_config"]
    job_config = jobs[0]["payload"]["optimization_config"]
    assert db_config == job_config
    assert db_config["model_config"] == _BEDROCK_MODEL_CONFIG
    assert "model_configuration" not in db_config
    _assert_secret_free(db_config)
    _assert_secret_free(jobs[0]["payload"])


@pytest.mark.asyncio
async def test_legacy_optimization_row_is_scrubbed_and_normalized_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config = {
        "optimizer_type": "mipro",
        "target_metric": "accuracy",
        "strategy_params": {"beam_width": 2},
        "credentials_resolved": True,
        "metadata": {"credential_fields": {"token": _SECRET_SENTINEL}},
        "model_configuration": {
            "provider": "aws-bedrock",
            "model_name": "anthropic.claude-3-haiku",
            "temperature": 0.2,
            "max_tokens": 64,
            "timeout_seconds": 11,
            "parameters": {
                "top_p": 0.8,
                "nested": {"api_key": _SECRET_SENTINEL},
            },
            "app_config": {"token": _SECRET_SENTINEL},
            "credential_runtime": _RUNTIME_SENTINEL,
        },
    }
    db = _LegacyOptimizationDb(legacy_config)
    engine = OptimizationEngine(db)  # type: ignore[arg-type]
    seen: dict[str, Any] = {}

    async def _capture_mipro(**kwargs: Any) -> dict[str, Any]:
        seen.update(copy.deepcopy(kwargs))
        return {
            "optimized_prompt_id": 12,
            "initial_score": 0.5,
            "final_score": 0.6,
            "improvement": 0.1,
            "iterations": 1,
        }

    monkeypatch.setattr(engine.mipro, "optimize", _capture_mipro, raising=True)

    await engine.optimize(44)

    expected_model_config = {
        "provider": "bedrock",
        "model": "anthropic.claude-3-haiku",
        "parameters": {
            "temperature": 0.2,
            "max_tokens": 64,
            "timeout_seconds": 11,
            "top_p": 0.8,
        },
    }
    assert seen["model_config"] == expected_model_config
    assert db.config_updates
    persisted = db.row["optimization_config"]
    assert persisted["model_config"] == expected_model_config
    assert "model_configuration" not in persisted
    assert persisted["strategy_params"] == {"beam_width": 2}
    _assert_secret_free(persisted)


@pytest.mark.asyncio
async def test_secret_only_legacy_row_fails_before_default_provider_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _LegacyOptimizationDb(
        {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "model_configuration": {
                "API_KEY": _SECRET_SENTINEL,
                "credentialsResolved": True,
                "runtimeHandle": _RuntimeHandle(),
            },
        }
    )
    engine = OptimizationEngine(db)  # type: ignore[arg-type]
    optimize = AsyncMock(
        return_value={
            "optimized_prompt_id": 12,
            "initial_score": 0.0,
            "final_score": 0.0,
            "improvement": 0.0,
            "iterations": 0,
        }
    )
    monkeypatch.setattr(engine.mipro, "optimize", optimize, raising=True)

    with pytest.raises(ValueError, match="provider|model|configuration"):
        await engine.optimize(44)

    optimize.assert_not_awaited()
    assert db.config_updates
    _assert_secret_free(db.row["optimization_config"])
