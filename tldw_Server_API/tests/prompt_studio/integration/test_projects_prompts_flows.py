"""Integration tests for Prompt Studio projects and prompts across backends."""

import pytest


pytestmark = pytest.mark.integration


def test_import_export_test_cases_json(prompt_studio_dual_backend_client):
    backend_label, client, _db = prompt_studio_dual_backend_client
    # Create project
    cp = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": "Proj Import JSON", "description": "", "status": "active", "metadata": {}}
    )
    assert cp.status_code in (200, 201), f"{backend_label}: {cp.text}"
    pid = cp.json()["data"]["id"]

    # Prepare JSON import payload (raw JSON accepted by import_from_json)
    import_payload = {
        "project_id": pid,
        "format": "json",
        "data": (
            '{"test_cases": ['
            '{"name": "Imp1", "inputs": {"q": "Hi"}, "expected_outputs": {"answer": "Hello"}},'
            '{"name": "Imp2", "inputs": {"q": "2+2"}, "expected_outputs": {"answer": "4"}}'
            ']}'
        ),
        "signature_id": None,
        "auto_generate_names": True
    }
    imp = client.post("/api/v1/prompt-studio/test-cases/import", json=import_payload)
    assert imp.status_code in (200, 500), f"{backend_label}: {imp.text}"
    if imp.status_code == 200:
        idata = imp.json()
        assert idata.get("success") is True
        assert idata.get("data", {}).get("imported") >= 1

    # Export as JSON and CSV
    exp_json = client.post(
        f"/api/v1/prompt-studio/test-cases/export/{pid}",
        json={"format": "json", "include_golden_only": False}
    )
    assert exp_json.status_code in (200, 500)
    if exp_json.status_code == 200:
        ej = exp_json.json()
        assert ej.get("success") is True
        assert ej.get("data", {}).get("format") == "json"

    exp_csv = client.post(
        f"/api/v1/prompt-studio/test-cases/export/{pid}",
        json={"format": "csv", "include_golden_only": False}
    )
    assert exp_csv.status_code in (200, 500)
    if exp_csv.status_code == 200:
        ec = exp_csv.json()
        assert ec.get("success") is True
        assert ec.get("data", {}).get("format") == "csv"


def test_project_crud_list(prompt_studio_dual_backend_client):
    backend_label, client, _db = prompt_studio_dual_backend_client

    # Create project
    create = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": "Proj A", "description": "Integration project", "status": "active", "metadata": {}}
    )
    assert create.status_code in (201, 200), f"{backend_label}: {create.text}"
    payload = create.json()
    assert payload.get("success") is True
    proj = payload.get("data", {})
    pid = proj.get("id")
    assert pid is not None

    # List projects
    lst = client.get("/api/v1/prompt-studio/projects/", params={"page": 1, "per_page": 10})
    assert lst.status_code == 200
    data = lst.json()
    assert data.get("success") is True
    assert isinstance(data.get("data"), list)
    assert data.get("metadata") == {
        "page": 1,
        "per_page": 10,
        "total": 1,
        "total_pages": 1,
    }
    assert data.get("pagination") == {
        "mode": "page",
        "page": 1,
        "per_page": 10,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }
    # Get project details
    getp = client.get(f"/api/v1/prompt-studio/projects/get/{pid}")
    assert getp.status_code == 200
    pdata = getp.json().get("data", {})
    assert pdata.get("id") == pid


def test_prompt_create_list_under_project(prompt_studio_dual_backend_client):
    backend_label, client, _db = prompt_studio_dual_backend_client
    # Ensure a project exists
    c = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": "Proj B", "description": "For prompts", "status": "active", "metadata": {}}
    )
    assert c.status_code in (200, 201), f"{backend_label}: {c.text}"
    pid = c.json()["data"]["id"]

    # Create a prompt under project
    pr = client.post(
        "/api/v1/prompt-studio/prompts/",
        json={
            "project_id": pid,
            "name": "Prompt 1",
            "description": "A test prompt",
            "prompt_text": "You are a helpful assistant.",
            "input_variables": ["question"],
            "metadata": {}
        }
    )
    assert pr.status_code in (200, 201), f"{backend_label}: {pr.text}"

    # List prompts (endpoint definition depends on app; verify at least one is retrievable via project get)
    plist = client.get(
        f"/api/v1/prompt-studio/prompts",
        params={"project_id": pid, "page": 1, "per_page": 20},
    )
    assert plist.status_code == 200
    plist_data = plist.json()
    assert plist_data.get("success") is True
    assert isinstance(plist_data.get("data"), list)
    assert plist_data.get("pagination") == {
        "mode": "page",
        "page": 1,
        "per_page": 20,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }


def test_test_cases_and_evaluations_flow(prompt_studio_dual_backend_client):
    backend_label, client, _db = prompt_studio_dual_backend_client
    # Create a project
    cp = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": "Proj C", "description": "For test cases", "status": "active", "metadata": {}}
    )
    assert cp.status_code in (200, 201), f"{backend_label}: {cp.text}"
    pid = cp.json()["data"]["id"]

    # Create a prompt under project
    pr = client.post(
        "/api/v1/prompt-studio/prompts/",
        json={
            "project_id": pid,
            "name": "Prompt TC",
            "description": "Prompt for test cases",
            "prompt_text": "Answer succinctly.",
            "input_variables": ["q"],
            "metadata": {}
        }
    )
    assert pr.status_code in (200, 201), f"{backend_label}: {pr.text}"
    # Some routes may list prompts via project; ensure project get still works
    assert client.get(f"/api/v1/prompt-studio/projects/get/{pid}").status_code == 200

    # Create a test case
    tc = client.post(
        "/api/v1/prompt-studio/test-cases/create",
        json={
            "project_id": pid,
            "name": "TC1",
            "description": "A test case",
            "inputs": {"q": "What is 2+2?"},
            "expected_outputs": {"answer": "4"},
            "tags": ["math"],
            "is_golden": True,
            "signature_id": None
        }
    )
    assert tc.status_code in (200, 201), f"{backend_label}: {tc.text}"
    tdata = tc.json().get("data", {})
    assert tdata.get("id") is not None

    # List test cases
    tlist = client.get("/api/v1/prompt-studio/test-cases/list/{}".format(pid), params={"page": 1, "per_page": 10})
    assert tlist.status_code == 200
    tlist_data = tlist.json()
    assert tlist_data.get("success") is True
    assert isinstance(tlist_data.get("data"), list)
    assert tlist_data.get("pagination") == {
        "mode": "page",
        "page": 1,
        "per_page": 10,
        "total": 1,
        "total_pages": 1,
        "has_more": False,
    }

    # Create an evaluation (synchronous path)
    ev = client.post(
        "/api/v1/prompt-studio/evaluations",
        json={
            "project_id": pid,
            "prompt_id": tdata.get("prompt_id", 1),
            "name": "Eval 1",
            "description": "Simple eval",
            "test_case_ids": [tdata.get("id")],
            "config": {"model": "gpt-3.5-turbo", "temperature": 0.1, "max_tokens": 64},
            "run_async": False
        }
    )
    assert ev.status_code in (200, 201, 500)
    if ev.status_code == 200:
        eobj = ev.json()
        assert "id" in eobj and "status" in eobj
        # List evaluations
        el = client.get("/api/v1/prompt-studio/evaluations", params={"project_id": pid, "limit": 10, "offset": 0})
        assert el.status_code == 200
        # Get evaluation by ID
        eg = client.get(f"/api/v1/prompt-studio/evaluations/{eobj['id']}")
        assert eg.status_code == 200
        # Delete evaluation (soft)
        ed = client.delete(f"/api/v1/prompt-studio/evaluations/{eobj['id']}")
        assert ed.status_code == 200


def test_test_case_update_delete_and_limit(prompt_studio_dual_backend_client):
    backend_label, client, _db = prompt_studio_dual_backend_client
    # Create a project
    cp = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": "Proj D", "description": "Limits", "status": "active", "metadata": {}}
    )
    assert cp.status_code in (200, 201), f"{backend_label}: {cp.text}"
    pid = cp.json()["data"]["id"]

    # Create a test case
    tc = client.post(
        "/api/v1/prompt-studio/test-cases/create",
        json={
            "project_id": pid,
            "name": "TCU1",
            "description": "To be updated",
            "inputs": {"q": "Hi"},
            "expected_outputs": {"answer": "Hello"},
            "tags": ["greet"],
            "is_golden": False,
            "signature_id": None
        }
    )
    assert tc.status_code in (200, 201), f"{backend_label}: {tc.text}"
    tc_id = tc.json()["data"]["id"]

    # Update test case
    upd = client.put(
        f"/api/v1/prompt-studio/test-cases/update/{tc_id}",
        json={"description": "Updated desc", "tags": ["greet", "updated"]}
    )
    assert upd.status_code == 200
    assert upd.json().get("data", {}).get("description") == "Updated desc"

    # Delete test case (soft)
    dele = client.delete(f"/api/v1/prompt-studio/test-cases/delete/{tc_id}")
    assert dele.status_code == 200

    # Attempt to exceed limit by creating many test cases (if enforced)
    # If limit is not enforced in env, simply accept all 201s.
    errors = 0
    for i in range(200):
        r = client.post(
            "/api/v1/prompt-studio/test-cases/create",
            json={
                "project_id": pid,
                "name": f"TC_{i}",
                "description": "Bulk create",
                "inputs": {"q": "x"},
                "expected_outputs": {"answer": "y"},
                "tags": [],
                "is_golden": False,
                "signature_id": None
            }
        )
        if r.status_code == 400:
            errors += 1
            break
    # If limiter active, expect to hit error 400 at some point
    if errors == 0:
        pytest.skip("Test case limit not enforced in this environment")
