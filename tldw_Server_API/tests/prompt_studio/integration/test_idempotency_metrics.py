import json

import pytest


class _StubMM:
    def __init__(self):
        self.increments = []  # (name, value, labels)

    def increment(self, name, value=1, labels=None):
        self.increments.append((name, value, labels or {}))


class _StubPSMetrics:
    def __init__(self):
        self.metrics_manager = _StubMM()


pytestmark = pytest.mark.integration


def _create_project_and_prompt(db):
    proj = db.create_project("met-test-proj")
    pr = db.create_prompt(project_id=proj["id"], name="met-test-prompt")
    test_case = db.create_test_case(
        project_id=proj["id"],
        name="met-test-case",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )
    return proj, pr, test_case


def test_optimization_idempotency_metrics(prompt_studio_dual_backend_client, monkeypatch):
    label, client, db = prompt_studio_dual_backend_client

    # Prepare entities
    proj, pr, test_case = _create_project_and_prompt(db)

    # Stub metrics in the endpoint module
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_optimization as mod
    stub = _StubPSMetrics()
    monkeypatch.setattr(mod, "prompt_studio_metrics", stub, raising=True)

    # First call => miss_total
    body = {
        "project_id": proj["id"],
        "initial_prompt_id": pr["id"],
        "optimization_config": {
            "optimizer_type": "iterative",
            "max_iterations": 1,
            "target_metric": "accuracy"
        },
        "test_case_ids": [test_case["id"]]
    }
    headers = {"Content-Type": "application/json", "Idempotency-Key": "idemp-test-1"}
    r1 = client.post("/api/v1/prompt-studio/optimizations/create", data=json.dumps(body), headers=headers)
    assert r1.status_code in (200, 201)

    # Second call => hit_total
    r2 = client.post("/api/v1/prompt-studio/optimizations/create", data=json.dumps(body), headers=headers)
    assert r2.status_code in (200, 201)

    names = [n for (n, _, _) in stub.metrics_manager.increments]
    assert "prompt_studio.idempotency.miss_total" in names, "idempotency miss counter not incremented"
    assert "prompt_studio.idempotency.hit_total" in names, "idempotency hit counter not incremented"
