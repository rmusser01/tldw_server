"""Integration tests for Idempotency-Key on Prompt Studio create endpoints.

Covers:
- Projects: POST /api/v1/prompt-studio/projects/
- Prompts:  POST /api/v1/prompt-studio/prompts/create
- Optimizations: POST /api/v1/prompt-studio/optimizations/create

Runs against both backends via the dual-backend fixture (Postgres skipped if unreachable).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _headers_with_idem(key: str) -> dict[str, str]:
    return {"Idempotency-Key": key, "Content-Type": "application/json"}


def test_project_create_idempotency_dual_backend(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client

    project_name = f"IdemProj-{uuid.uuid4().hex[:8]} ({backend_label})"
    idem_key = f"proj-{uuid.uuid4().hex}"

    body = {"name": project_name, "description": "idem test", "status": "active"}

    r1 = client.post("/api/v1/prompt-studio/projects/", json=body, headers=_headers_with_idem(idem_key))
    assert r1.status_code in (200, 201), r1.text
    id1 = (r1.json().get("data") or {}).get("id") or r1.json().get("id")
    assert isinstance(id1, int)

    r2 = client.post("/api/v1/prompt-studio/projects/", json=body, headers=_headers_with_idem(idem_key))
    assert r2.status_code in (200, 201), r2.text
    id2 = (r2.json().get("data") or {}).get("id") or r2.json().get("id")
    assert id2 == id1

    # Ensure no duplicate projects exist for this user with same name
    projects = db.list_projects(user_id="test-user-123", search=project_name)["projects"]
    assert sum(1 for p in projects if p.get("name") == project_name) == 1


def test_prompt_create_idempotency_dual_backend(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client

    # Create a project first
    pname = f"IdemProjP-{uuid.uuid4().hex[:6]} ({backend_label})"
    pr = client.post("/api/v1/prompt-studio/projects/", json={"name": pname, "status": "active"})
    assert pr.status_code in (200, 201), pr.text
    project_id = (pr.json().get("data") or pr.json()).get("id")
    assert isinstance(project_id, int)

    idem_key = f"prompt-{uuid.uuid4().hex}"
    pbody = {
        "project_id": project_id,
        "name": f"Summ-{uuid.uuid4().hex[:6]}",
        "system_prompt": "Summarize clearly",
        "user_prompt": "{{text}}",
    }

    r1 = client.post("/api/v1/prompt-studio/prompts/create", json=pbody, headers=_headers_with_idem(idem_key))
    assert r1.status_code in (200, 201), r1.text
    id1 = (r1.json().get("data") or {}).get("id") or r1.json().get("id")
    assert isinstance(id1, int)

    r2 = client.post("/api/v1/prompt-studio/prompts/create", json=pbody, headers=_headers_with_idem(idem_key))
    assert r2.status_code in (200, 201), r2.text
    id2 = (r2.json().get("data") or {}).get("id") or r2.json().get("id")
    assert id2 == id1

    # Verify only one prompt with that name in the project
    plist = db.list_prompts(project_id, page=1, per_page=50)["prompts"]
    assert sum(1 for p in plist if p.get("id") == id1) == 1


def test_optimization_create_idempotency_dual_backend(prompt_studio_dual_backend_client):
    backend_label, client, db = prompt_studio_dual_backend_client

    # Create a project + prompt first
    pname = f"IdemProjO-{uuid.uuid4().hex[:6]} ({backend_label})"
    pr = client.post("/api/v1/prompt-studio/projects/", json={"name": pname, "status": "active"})
    assert pr.status_code in (200, 201), pr.text
    project_id = (pr.json().get("data") or pr.json()).get("id")

    pbody = {
        "project_id": project_id,
        "name": f"Base-{uuid.uuid4().hex[:6]}",
        "system_prompt": "S",
        "user_prompt": "{{q}}",
    }
    prp = client.post("/api/v1/prompt-studio/prompts/create", json=pbody)
    assert prp.status_code in (200, 201), prp.text
    prompt_id = (prp.json().get("data") or {}).get("id") or prp.json().get("id")
    test_case = db.create_test_case(
        project_id=project_id,
        name=f"Case-{uuid.uuid4().hex[:6]}",
        inputs={"q": "hello"},
        expected_outputs={"answer": "hello"},
    )

    idem_key = f"opt-{uuid.uuid4().hex}"
    obody: dict[str, Any] = {
        "project_id": project_id,
        "initial_prompt_id": prompt_id,
        "optimization_config": {
            "optimizer_type": "iterative",
            "max_iterations": 2,
            "target_metric": "accuracy",
            "early_stopping": True,
        },
        "test_case_ids": [test_case["id"]],
        "name": f"Opt-{uuid.uuid4().hex[:6]}",
    }

    r1 = client.post("/api/v1/prompt-studio/optimizations/create", json=obody, headers=_headers_with_idem(idem_key))
    assert r1.status_code in (200, 201), r1.text
    data1 = r1.json().get("data", {})
    oid1 = (data1.get("optimization") or {}).get("id")
    job1 = data1.get("job_id")
    assert isinstance(oid1, int)
    assert isinstance(job1, (int, str))

    r2 = client.post("/api/v1/prompt-studio/optimizations/create", json=obody, headers=_headers_with_idem(idem_key))
    assert r2.status_code in (200, 201), r2.text
    data2 = r2.json().get("data", {})
    oid2 = (data2.get("optimization") or {}).get("id")
    job2 = data2.get("job_id")
    assert oid2 == oid1
    # On idempotent path, job_id may be None (we don’t enqueue again)
    assert job2 in (None, job1)

    # Ensure the core Job for this optimization exists
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.jobs_adapter import PromptStudioJobsAdapter
    adapter = PromptStudioJobsAdapter()
    if job1 is not None:
        job = adapter.get_job(
            str(job1),
            db=db,
            user_id="test-user-123",
            job_type="optimization",
        )
        assert job is not None
        assert job.get("entity_id") == oid1
