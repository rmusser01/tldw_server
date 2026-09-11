import uuid

import pytest

pytestmark = pytest.mark.integration


def _mk_project(client, backend_label: str) -> int:
    name = f"StatusProj-{uuid.uuid4().hex[:6]} ({backend_label})"
    r = client.post("/api/v1/prompt-studio/projects/", json={"name": name, "status": "active"})
    assert r.status_code in (200, 201), r.text
    return (r.json().get("data") or r.json()).get("id")


def _mk_prompt(client, project_id: int, backend_label: str) -> int:
    r = client.post(
        "/api/v1/prompt-studio/prompts/create",
        json={
            "project_id": project_id,
            "name": f"Base-{uuid.uuid4().hex[:6]} ({backend_label})",
            "system_prompt": "S",
            "user_prompt": "{{q}}",
        },
    )
    assert r.status_code in (200, 201), r.text
    return (r.json().get("data") or {}).get("id") or r.json().get("id")


def test_status_reports_queue_and_leases(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client

    # Baseline status call
    r0 = client.get("/api/v1/prompt-studio/status")
    assert r0.status_code == 200
    d0 = r0.json().get("data") or {}
    assert {"queue_depth", "processing", "leases", "by_status", "by_type"} <= set(d0.keys())

    # Create a queued optimization to bump queue depth
    pid = _mk_project(client, backend_label)
    prompt_id = _mk_prompt(client, pid, backend_label)
    test_case = db.create_test_case(
        project_id=pid,
        name=f"StatusCase-{uuid.uuid4().hex[:6]} ({backend_label})",
        inputs={"q": "hello"},
        expected_outputs={"answer": "hello"},
    )

    body = {
        "project_id": pid,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "iterative",
            "max_iterations": 1,
            "target_metric": "accuracy",
            "early_stopping": True,
        },
        "test_case_ids": [test_case["id"]],
        "name": "qstatus",
    }
    r = client.post("/api/v1/prompt-studio/optimizations/create", json=body)
    assert r.status_code in (200, 201), r.text

    r1 = client.get("/api/v1/prompt-studio/status", params={"warn_seconds": 60})
    assert r1.status_code == 200
    d1 = r1.json().get("data") or {}
    assert isinstance(d1.get("queue_depth", 0), int)
    assert d1["queue_depth"] >= 1
    leases = d1.get("leases") or {}
    # Leases keys present; values are ints
    assert {"active", "expiring_soon", "stale_processing"} <= set(leases.keys())
    for v in leases.values():
        assert isinstance(v, int)
