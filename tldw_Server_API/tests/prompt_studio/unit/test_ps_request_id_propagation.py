import uuid


def test_ps_optimization_simple_includes_request_id_in_payload(
    monkeypatch,
    prompt_studio_dual_backend_client,
):
    captured = {}

    # Monkeypatch the Prompt Studio Jobs adapter to capture payloads
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import jobs_adapter as ps_jobs

    def fake_create_job(  # noqa: D401
        self,
        *,
        user_id=None,
        job_type=None,
        entity_id=None,
        payload=None,
        project_id=None,
        priority=5,
        max_retries=3,
        request_id=None,
        trace_id=None,
    ):
        captured["payload"] = payload
        return {"id": 777, "status": "queued"}

    monkeypatch.setattr(ps_jobs.PromptStudioJobsAdapter, "create_job", fake_create_job, raising=True)

    backend_label, client, db = prompt_studio_dual_backend_client
    project_name = f"ReqID Project {uuid.uuid4().hex[:6]} ({backend_label})"
    prompt_name = f"ReqID Prompt {uuid.uuid4().hex[:6]} ({backend_label})"
    project_resp = client.post(
        "/api/v1/prompt-studio/projects/",
        json={"name": project_name, "status": "active"},
        headers={
            "X-API-KEY": "test-api-key-12345",
        },
    )
    assert project_resp.status_code in (200, 201), project_resp.text
    project_id = (project_resp.json().get("data") or {}).get("id") or project_resp.json().get("id")

    prompt_resp = client.post(
        "/api/v1/prompt-studio/prompts/create",
        json={
            "project_id": project_id,
            "name": prompt_name,
            "system_prompt": "System",
            "user_prompt": "{{text}}",
        },
        headers={
            "X-API-KEY": "test-api-key-12345",
        },
    )
    assert prompt_resp.status_code in (200, 201), prompt_resp.text
    prompt_id = (prompt_resp.json().get("data") or {}).get("id") or prompt_resp.json().get("id")
    test_case = db.create_test_case(
        project_id=project_id,
        name=f"ReqID Case {uuid.uuid4().hex[:6]} ({backend_label})",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )

    r = client.post(
        "/api/v1/prompt-studio/optimizations",
        json={
            "project_id": project_id,
            "prompt_id": prompt_id,
            "config": {"optimizer_type": "iterative"},
            "test_case_ids": [int(test_case["id"])],
        },
        headers={
            "X-API-KEY": "test-api-key-12345",
            "X-Request-ID": "req-ps-001",
        },
    )
    assert r.status_code == 200, r.text
    assert captured.get("payload", {}).get("request_id") == "req-ps-001"
    optimization_uuid = captured.get("payload", {}).get("optimization_uuid")
    assert isinstance(optimization_uuid, str)
    assert optimization_uuid
