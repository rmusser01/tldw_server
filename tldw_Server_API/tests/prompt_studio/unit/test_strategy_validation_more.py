import uuid

import pytest

pytestmark = pytest.mark.unit


def _mk_project(client, backend_label: str) -> int:
    name = f"Val3Proj-{uuid.uuid4().hex[:6]} ({backend_label})"
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


def test_hyperparameter_is_rejected_regardless_of_legacy_knobs(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "hyperparameter",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {
                "search_method": "invalid",
                "params_to_optimize": [],
            },
        },
        "test_case_ids": [test_case_id],
        "name": "hyper-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "hyperparameter",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {
                "search_method": "bayesian",
                "params_to_optimize": ["temperature", "max_tokens"],
                "max_trials": 5,
            },
        },
        "test_case_ids": [test_case_id],
        "name": "hyper-ok",
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text


def test_random_search_is_rejected_regardless_of_legacy_knobs(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "random_search",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {"max_trials": 0},
        },
        "test_case_ids": [test_case_id],
        "name": "rand-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "rand-ok",
        "optimization_config": {
            **bad["optimization_config"],
            "strategy_params": {"max_trials": 5},
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text


def test_beam_search_is_rejected_regardless_of_diversity_rate(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "beam_search",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {"beam_width": 3, "diversity_rate": 1.5},
        },
        "test_case_ids": [test_case_id],
        "name": "beam-div-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "beam-div-ok",
        "optimization_config": {
            **bad["optimization_config"],
            "strategy_params": {"beam_width": 3, "diversity_rate": 0.3},
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text
