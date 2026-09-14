"""Fail-closed public input contracts for Prompt Studio optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_optimization as optimization_endpoint,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import SecurityConfig
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    OptimizationConfig,
    OptimizationCreate,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
    OptimizationSimpleCreateRequest,
)

pytestmark = pytest.mark.unit

_CANONICAL_MODEL_CONFIG = {
    "provider": "bedrock",
    "model": "anthropic.claude-3-haiku",
    "parameters": {"temperature": 0.2},
}


def _full_config(**updates: Any) -> OptimizationConfig:
    payload = {
        "optimizer_type": "mipro",
        "target_metric": "accuracy",
        "model_config": _CANONICAL_MODEL_CONFIG,
    }
    payload.update(updates)
    return OptimizationConfig.model_validate(payload)


def _public_request_factories() -> tuple[Callable[[dict[str, Any]], BaseModel], ...]:
    return (
        lambda model: _full_config(model_config=model),
        lambda model: OptimizationSimpleCreateRequest.model_validate(
            {
                "prompt_id": 1,
                "config": {"model_config": model},
            }
        ),
        lambda model: CompareStrategiesRequest.model_validate(
            {
                "prompt_id": 1,
                "test_case_ids": [1],
                "strategies": ["mipro", "bootstrap"],
                "model_configuration": model,
            }
        ),
    )


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key_override",
        "API-Key-Override",
        "apiKeyOverride",
        "azure_client_secret",
        "AZURE-CLIENT-SECRET",
        "AzureClientSecret",
        "github_access_token",
        "GITHUB-ACCESS-TOKEN",
        "githubAccessToken",
    ],
)
def test_recursive_secret_key_variants_are_rejected_by_every_public_schema(
    secret_key: str,
) -> None:
    hostile_model = {
        **_CANONICAL_MODEL_CONFIG,
        "parameters": {
            "response_format": {
                "nested": [{secret_key: "must-never-persist"}],
            }
        },
    }

    for factory in _public_request_factories():
        with pytest.raises(ValidationError, match="credential field"):
            factory(hostile_model)


def test_model_only_input_is_ambiguous_and_rejected_by_every_public_schema() -> None:
    for factory in _public_request_factories():
        with pytest.raises(ValidationError, match="provider"):
            factory({"model": "gpt-4o-mini"})


def test_llm_model_config_alias_is_canonical_across_public_schemas() -> None:
    legacy = {
        "provider": "AWS_BEDROCK",
        "model_name": "anthropic.claude-3-haiku",
        "temperature": 0.2,
    }

    full = OptimizationConfig.model_validate(
        {
            "optimizer_type": "mipro",
            "target_metric": "accuracy",
            "llm_model_config": legacy,
        }
    )
    simple = OptimizationSimpleCreateRequest.model_validate(
        {"prompt_id": 1, "llm_model_config": legacy}
    )
    compare = CompareStrategiesRequest.model_validate(
        {
            "prompt_id": 1,
            "test_case_ids": [1],
            "strategies": ["mipro", "bootstrap"],
            "llm_model_config": legacy,
        }
    )

    assert full.llm_model_config == _CANONICAL_MODEL_CONFIG
    assert simple.llm_model_config == _CANONICAL_MODEL_CONFIG
    assert simple.config["model_config"] == _CANONICAL_MODEL_CONFIG
    assert compare.model_configuration == _CANONICAL_MODEL_CONFIG


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "provider": "openai",
            "api_name": "anthropic",
            "model": "gpt-4o-mini",
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "model_name": "gpt-4.1-mini",
        },
    ],
)
def test_conflicting_provider_or_model_aliases_are_rejected(model_config: dict[str, Any]) -> None:
    for factory in _public_request_factories():
        with pytest.raises(ValidationError, match="conflict"):
            factory(model_config)


def test_conflicting_model_config_container_aliases_are_rejected() -> None:
    openai = {"provider": "openai", "model": "gpt-4o-mini"}
    bedrock = {"provider": "bedrock", "model": "anthropic.claude-3-haiku"}

    with pytest.raises(ValidationError, match="conflict"):
        OptimizationConfig.model_validate(
            {
                "optimizer_type": "mipro",
                "target_metric": "accuracy",
                "model_config": openai,
                "model_configuration": bedrock,
            }
        )
    with pytest.raises(ValidationError, match="conflict"):
        OptimizationSimpleCreateRequest.model_validate(
            {
                "prompt_id": 1,
                "llm_model_config": openai,
                "config": {"model_config": bedrock},
            }
        )
    with pytest.raises(ValidationError, match="conflict"):
        CompareStrategiesRequest.model_validate(
            {
                "prompt_id": 1,
                "test_case_ids": [1],
                "strategies": ["mipro", "bootstrap"],
                "config": openai,
                "model_configuration": bedrock,
            }
        )


@pytest.mark.parametrize(
    "historical_strategy",
    [
        "hill_climb",
        "hill_climbing",
        "random_search",
        "grid_search",
        "bayesian",
        "beam_search",
        "greedy",
        "anneal",
        "simulated_annealing",
        "genetic",
        "hyperparameter",
        "hparam",
    ],
)
def test_unsupported_historical_strategies_are_rejected_at_submission(
    historical_strategy: str,
) -> None:
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        _full_config(optimizer_type=historical_strategy)
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationSimpleCreateRequest.model_validate(
            {"prompt_id": 1, "strategy": historical_strategy}
        )
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        CompareStrategiesRequest.model_validate(
            {
                "prompt_id": 1,
                "test_case_ids": [1],
                "strategies": [historical_strategy, "mipro"],
            }
        )


@pytest.mark.asyncio
async def test_optimization_rate_limit_uses_the_five_request_optimize_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def _capture(
        operation: str,
        *,
        user_context: dict[str, Any],
        security_config: SecurityConfig,
    ) -> bool:
        seen.append(operation)
        return True

    monkeypatch.setattr(optimization_endpoint, "check_rate_limit", _capture)

    assert await optimization_endpoint._rl_optimizations(
        user_context={"user_id": "7"},
        security_config=SecurityConfig(),
    )
    assert seen == ["optimize"]


def test_simple_post_has_the_optimization_rate_limit_dependency() -> None:
    route = next(
        route
        for route in optimization_endpoint.router.routes
        if isinstance(route, APIRoute)
        and route.path == "/api/v1/prompt-studio/optimizations"
        and "POST" in route.methods
    )

    assert optimization_endpoint._rl_optimizations in {
        dependency.call for dependency in route.dependant.dependencies
    }


@pytest.mark.parametrize("request_kind", ["full", "simple", "compare"])
def test_test_case_ids_are_bounded_and_unique(request_kind: str) -> None:
    too_many = list(range(1, 1002))
    duplicates = [1, 1]

    def _validate(test_case_ids: list[int]) -> BaseModel:
        if request_kind == "full":
            return OptimizationCreate.model_validate(
                {
                    "project_id": 1,
                    "initial_prompt_id": 1,
                    "optimization_config": _full_config().model_dump(by_alias=True),
                    "test_case_ids": test_case_ids,
                }
            )
        if request_kind == "simple":
            return OptimizationSimpleCreateRequest.model_validate(
                {"prompt_id": 1, "test_case_ids": test_case_ids}
            )
        return CompareStrategiesRequest.model_validate(
            {
                "prompt_id": 1,
                "test_case_ids": test_case_ids,
                "strategies": ["mipro", "bootstrap"],
            }
        )

    with pytest.raises(ValidationError):
        _validate(too_many)
    with pytest.raises(ValidationError, match="unique"):
        _validate(duplicates)


def test_compare_strategy_fanout_is_nonempty_bounded_and_unique() -> None:
    base = {"prompt_id": 1, "test_case_ids": [1]}

    with pytest.raises(ValidationError):
        CompareStrategiesRequest.model_validate({**base, "strategies": []})
    with pytest.raises(ValidationError, match="unique"):
        CompareStrategiesRequest.model_validate(
            {**base, "strategies": ["mipro", "mipro"]}
        )
    with pytest.raises(ValidationError):
        CompareStrategiesRequest.model_validate(
            {
                **base,
                "strategies": [
                    "mipro",
                    "bootstrap",
                    "iterative",
                    "mcts",
                    "mipro",
                ],
            }
        )


def test_simple_max_iterations_uses_the_existing_full_schema_bound() -> None:
    with pytest.raises(ValidationError, match="500"):
        OptimizationSimpleCreateRequest.model_validate(
            {
                "prompt_id": 1,
                "config": {"max_iterations": 501},
            }
        )


@pytest.mark.parametrize("request_kind", ["full", "simple", "compare"])
def test_durable_config_size_is_bounded(request_kind: str) -> None:
    oversized = "x" * (32 * 1024)
    model = {
        **_CANONICAL_MODEL_CONFIG,
        "parameters": {"response_format": {"description": oversized}},
    }

    with pytest.raises(ValidationError, match="too large"):
        if request_kind == "full":
            _full_config(model_config=model)
        elif request_kind == "simple":
            OptimizationSimpleCreateRequest.model_validate(
                {"prompt_id": 1, "config": {"model_config": model}}
            )
        else:
            CompareStrategiesRequest.model_validate(
                {
                    "prompt_id": 1,
                    "test_case_ids": [1],
                    "strategies": ["mipro", "bootstrap"],
                    "model_configuration": model,
                }
            )
