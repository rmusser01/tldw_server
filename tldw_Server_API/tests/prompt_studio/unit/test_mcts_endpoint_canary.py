import uuid

import pytest

pytestmark = pytest.mark.unit


def _mk_project(client, backend_label: str) -> int:
    name = f"MCTSProj-{uuid.uuid4().hex[:6]} ({backend_label})"
    r = client.post("/api/v1/prompt-studio/projects/", json={"name": name, "status": "active"})
    assert r.status_code in (200, 201), r.text
    return (r.json().get("data") or r.json()).get("id")


def _mk_prompt(client, project_id: int, backend_label: str) -> int:
    r = client.post(
        "/api/v1/prompt-studio/prompts/create",
        json={
            "project_id": project_id,
            "name": f"MCTSPrompt-{uuid.uuid4().hex[:6]} ({backend_label})",
            "system_prompt": "Be precise.",
            "user_prompt": "Echo: {{text}}",
        },
    )
    assert r.status_code in (200, 201), r.text
    return (r.json().get("data") or {}).get("id") or r.json().get("id")


def test_create_mcts_optimization_canary(prompt_studio_dual_backend_client, monkeypatch):
    backend_label, client, db = prompt_studio_dual_backend_client

    # Enable MCTS (gated by default)
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    pid = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, pid, backend_label)
    test_case = db.create_test_case(
        project_id=pid,
        name=f"MCTSCase-{uuid.uuid4().hex[:6]} ({backend_label})",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )

    body = {
        "project_id": pid,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "mcts",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": 2,
                "mcts_max_depth": 2,
                "mcts_exploration_c": 1.4,
                "prompt_candidates_per_node": 1
            }
        },
        "test_case_ids": [int(test_case["id"])],
        "name": "mcts-canary-endpoint"
    }

    r = client.post("/api/v1/prompt-studio/optimizations/create", json=body)
    assert r.status_code in (200, 201), r.text
    payload = r.json()
    data = payload.get("data") or {}
    # Minimal assertions: optimization created and job queued
    assert data
    assert (data.get("optimization") or {}).get("id") or data.get("id")
    assert data.get("job_id") is not None
