import uuid

import pytest

pytestmark = pytest.mark.unit


def _mk_project(client, backend_label: str) -> int:
    name = f"Val2Proj-{uuid.uuid4().hex[:6]} ({backend_label})"
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


def test_beam_search_is_rejected_regardless_of_legacy_knobs(prompt_studio_dual_backend_client):
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
            "strategy_params": {"beam_width": 1},
        },
        "test_case_ids": [test_case_id],
        "name": "beam-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "beam-good",
        "optimization_config": {
            **bad["optimization_config"],
            "strategy_params": {"beam_width": 3},
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text


def test_anneal_is_rejected_regardless_of_legacy_knobs(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "anneal",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {"cooling_rate": 1.5},
        },
        "test_case_ids": [test_case_id],
        "name": "anneal-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "anneal-good",
        "optimization_config": {
            **bad["optimization_config"],
            "strategy_params": {"cooling_rate": 0.2, "initial_temp": 1.0},
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text


def test_genetic_is_rejected_regardless_of_legacy_knobs(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client
    project_id = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, project_id, backend_label)
    test_case_id = _mk_test_case(db, project_id, backend_label)
    bad = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "genetic",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {"mutation_rate": -0.1, "population_size": 1},
        },
        "test_case_ids": [test_case_id],
        "name": "genetic-bad",
    }

    rejected = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=bad,
    )
    assert rejected.status_code == 422

    good = {
        **bad,
        "name": "genetic-good",
        "optimization_config": {
            **bad["optimization_config"],
            "strategy_params": {"mutation_rate": 0.2, "population_size": 5},
        },
    }
    accepted = client.post(
        "/api/v1/prompt-studio/optimizations/create",
        json=good,
    )
    assert accepted.status_code == 422, accepted.text
