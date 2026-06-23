from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _force_docker_preflight_available(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.models import RuntimeType
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "false")
    # Keep deterministic route wiring for minimal app profile.
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    _force_docker_preflight_available(monkeypatch)
    from tldw_Server_API.app.main import app

    return TestClient(app)


def test_queued_post_run_status_has_no_started_or_finished_markers(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        body: dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["echo", "queued"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        if r.status_code != 200:
            pytest.fail(f"Expected 200 from POST /sandbox/runs, got {r.status_code}: {r.text}")
        data = r.json()
        if data["phase"] != "queued":
            pytest.fail(f"Expected queued phase in non-execution mode, got {data['phase']!r}")
        if data["started_at"] is not None:
            pytest.fail(f"Queued run should not include started_at marker, got {data['started_at']!r}")
        if data["finished_at"] is not None:
            pytest.fail(f"Queued run should not include finished_at marker, got {data['finished_at']!r}")
        if data["exit_code"] is not None:
            pytest.fail(f"Queued run should not include exit_code, got {data['exit_code']!r}")
        if data.get("status_reason_code") != "queued":
            pytest.fail(f"Queued run should expose status_reason_code='queued', got {data.get('status_reason_code')!r}")
        details = data.get("status_reason_details")
        if details != {
            "code": "queued",
            "category": "lifecycle",
            "severity": "info",
            "terminal": False,
            "retryable": False,
            "operator_action": "none",
            "user_message_key": "sandbox.status.queued",
        }:
            pytest.fail(f"Queued run should expose queued status_reason_details, got {details!r}")


def test_post_run_response_fields_are_persisted_consistently_for_get(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        body: dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["echo", "durable"],
            "timeout_sec": 5,
        }
        create_resp = client.post("/api/v1/sandbox/runs", json=body)
        if create_resp.status_code != 200:
            pytest.fail(
                f"Expected 200 from POST /sandbox/runs, got {create_resp.status_code}: {create_resp.text}"
            )
        created = create_resp.json()
        run_id = created["id"]

        get_resp = client.get(f"/api/v1/sandbox/runs/{run_id}")
        if get_resp.status_code != 200:
            pytest.fail(f"Expected 200 from GET /sandbox/runs/{run_id}, got {get_resp.status_code}: {get_resp.text}")
        persisted = get_resp.json()

        # Ensure POST and GET remain consistent for key client-visible status fields.
        keys = [
            "id",
            "spec_version",
            "runtime",
            "phase",
            "status_reason_code",
            "status_reason_details",
            "exit_code",
            "started_at",
            "finished_at",
            "message",
            "policy_hash",
            "session_id",
            "persona_id",
            "workspace_id",
            "workspace_group_id",
            "scope_snapshot_id",
        ]
        for key in keys:
            if persisted.get(key) != created.get(key):
                pytest.fail(
                    f"POST/GET mismatch for key={key!r}: created={created.get(key)!r}, persisted={persisted.get(key)!r}"
                )
