import uuid

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization import (
    DURABLE_OPTIMIZATION_STRATEGIES,
    OptimizationConfig,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_optimization_requests import (
    CompareStrategiesRequest,
    OptimizationSimpleCreateRequest,
)
from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    LEGACY_OPTIMIZATION_STRATEGIES,
    optimization_execution_strategy,
    reconcile_optimization_strategy,
)

pytestmark = pytest.mark.unit

_HISTORICAL_ALIASES = ("anneal", "hill_climb", "hparam", "hyperparam")


def _mk_project(client, backend_label: str) -> int:
    name = f"ValProj-{uuid.uuid4().hex[:6]} ({backend_label})"
    response = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": name, "status": "active"},
    )
    assert response.status_code in (200, 201), response.text
    return (response.json().get("data") or response.json()).get("id")


def _mk_prompt(client, project_id: int, backend_label: str) -> int:
    response = client.post(
        "/api/v1/prompt-studio/prompts/create",
        json={
            "project_id": project_id,
            "name": f"Base-{uuid.uuid4().hex[:6]} ({backend_label})",
            "system_prompt": "S",
            "user_prompt": "{{q}}",
        },
    )
    assert response.status_code in (200, 201), response.text
    return (response.json().get("data") or {}).get("id") or response.json().get(
        "id"
    )


def _mk_test_case(db, project_id: int, backend_label: str) -> int:
    test_case = db.create_test_case(
        project_id=project_id,
        name=f"Case-{uuid.uuid4().hex[:6]} ({backend_label})",
        inputs={"q": "hello"},
        expected_outputs={"answer": "hello"},
    )
    return int(test_case["id"])


@pytest.mark.parametrize("strategy", sorted(DURABLE_OPTIMIZATION_STRATEGIES))
def test_supported_durable_strategies_are_accepted_by_request_schemas(strategy):
    optimization = OptimizationConfig(
        optimizer_type=strategy,
        target_metric="accuracy",
    )
    simple = OptimizationSimpleCreateRequest(prompt_id=1, strategy=strategy)

    assert optimization.optimizer_type == strategy
    assert simple.strategy == strategy
    assert simple.config["optimizer_type"] == strategy


def test_compare_schema_accepts_the_complete_durable_strategy_set():
    request = CompareStrategiesRequest(
        prompt_id=1,
        test_case_ids=[2],
        strategies=sorted(DURABLE_OPTIMIZATION_STRATEGIES),
    )

    assert set(request.strategies) == DURABLE_OPTIMIZATION_STRATEGIES


@pytest.mark.parametrize("strategy", sorted(LEGACY_OPTIMIZATION_STRATEGIES))
def test_every_legacy_strategy_is_rejected_instead_of_remapped(strategy):
    with pytest.raises(ValueError, match="Unsupported optimization strategy"):
        optimization_execution_strategy(strategy)


def test_conflicting_strategy_sources_fail_closed():
    with pytest.raises(ValueError, match="Optimization strategy mismatch"):
        reconcile_optimization_strategy("mipro", "bootstrap")


@pytest.mark.parametrize("alias", _HISTORICAL_ALIASES)
def test_historical_strategy_aliases_are_rejected_across_request_schemas(alias):
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationConfig(optimizer_type=alias, target_metric="accuracy")
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationSimpleCreateRequest(prompt_id=1, strategy=alias)
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        CompareStrategiesRequest(
            prompt_id=1,
            test_case_ids=[2],
            strategies=[alias, "mipro"],
        )


def test_conflicting_native_strategy_sources_fail_closed_in_simple_request():
    with pytest.raises(ValidationError, match="Optimization strategy mismatch"):
        OptimizationSimpleCreateRequest(
            prompt_id=1,
            strategy="mipro",
            config={"optimizer_type": "bootstrap"},
        )


def test_simple_config_historical_strategy_field_is_rejected():
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationSimpleCreateRequest(
            prompt_id=1,
            config={"strategy": "hparam"},
        )


def test_unknown_strategy_fails_at_full_create_schema():
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationConfig(
            optimizer_type="quantum_search",
            target_metric="accuracy",
        )


def test_unknown_strategy_fails_at_compat_schema():
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        OptimizationSimpleCreateRequest(prompt_id=1, strategy="quantum_search")


def test_unknown_strategy_fails_at_compare_schema():
    with pytest.raises(ValidationError, match="Unsupported optimization strategy"):
        CompareStrategiesRequest(
            prompt_id=1,
            test_case_ids=[2],
            strategies=["quantum_search", "mipro"],
        )


def test_iterative_passes_and_hill_climb_is_rejected(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    body = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "iterative",
            "max_iterations": 3,
            "target_metric": "accuracy",
            "early_stopping": True,
        },
        "test_case_ids": [test_case_id],
        "name": "iter",
    }

    iterative = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=body,
    )
    assert iterative.status_code in (200, 201), iterative.text

    body["optimization_config"]["optimizer_type"] = "hill_climb"
    body["name"] = "hill"
    hill_climb = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=body,
    )
    assert hill_climb.status_code == 422, hill_climb.text


def test_grid_search_is_rejected_even_when_models_are_supplied(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "grid_search",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "models_to_test": [],
        },
        "test_case_ids": [test_case_id],
        "name": "grid-empty",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "grid-ok",
        "optimization_config": {
            **bad["optimization_config"],
            "models_to_test": ["gpt-4o-mini"],
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text
