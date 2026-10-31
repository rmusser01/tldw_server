import json
import threading

import pytest


class _StubMM:
    def __init__(self):
        self.increments = []

    def increment(self, name, value=1, labels=None):
        self.increments.append((name, value, labels or {}))


class _StubPSMetrics:
    def __init__(self):
        self.metrics_manager = _StubMM()


pytestmark = pytest.mark.integration


def _create_project_and_prompt(db):
    proj = db.create_project("idem-conc-proj")
    pr = db.create_prompt(project_id=proj["id"], name="idem-conc-prompt")
    test_case = db.create_test_case(
        project_id=proj["id"],
        name="idem-conc-case",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )
    return proj, pr, test_case


def test_idempotency_concurrency_hits(prompt_studio_dual_backend_client, monkeypatch):
    label, client, db = prompt_studio_dual_backend_client

    proj, pr, test_case = _create_project_and_prompt(db)

    # Stub metrics in endpoint module
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_optimization as mod
    stub = _StubPSMetrics()
    monkeypatch.setattr(mod, "prompt_studio_metrics", stub, raising=True)

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
    headers = {"Content-Type": "application/json", "Idempotency-Key": "idemp-conc-1"}

    results = []
    results_lock = threading.Lock()

    def worker():
        r = client.post("/api/v1/prompt-studio/optimizations/create", data=json.dumps(body), headers=headers)
        assert r.status_code in (200, 201)
        data = r.json().get("data", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
        opt = data.get("optimization") or data.get("optimization_id") or data
        # Accept either structure; normalize to id
        opt_id = opt.get("id") if isinstance(opt, dict) else None
        with results_lock:
            results.append(opt_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # All responses should reference the same optimization id
    uniq = set(results)
    assert len(uniq) == 1 and None not in uniq, f"expected single canonical id, got {uniq}"

    # Metrics: 1 miss and N-1 hits
    names = [n for (n, _, _) in stub.metrics_manager.increments]
    assert "prompt_studio.idempotency.miss_total" in names
    assert "prompt_studio.idempotency.hit_total" in names
    miss = sum(1 for (n, _, _) in stub.metrics_manager.increments if n == "prompt_studio.idempotency.miss_total")
    hit = sum(1 for (n, _, _) in stub.metrics_manager.increments if n == "prompt_studio.idempotency.hit_total")
    assert miss >= 1
    assert hit >= (len(threads) - 1)
