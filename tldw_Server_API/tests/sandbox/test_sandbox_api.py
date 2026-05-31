from __future__ import annotations

import os
from typing import Any, Dict
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType, Session, TrustLevel
from tldw_Server_API.app.core.Sandbox.orchestrator import SessionActiveRunsConflict


def _client(monkeypatch) -> TestClient:


     # Enable test-mode behaviors in auth to avoid API key requirements
    monkeypatch.setenv("TEST_MODE", "1")
    return TestClient(app)


def test_runtimes_discovery_shape(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        r = client.get("/api/v1/sandbox/runtimes")
        assert r.status_code == 200
        data = r.json()
        assert "runtimes" in data and isinstance(data["runtimes"], list)
        assert len(data["runtimes"]) >= 1
        required_keys = [
            "name",
            "available",
            "default_images",
            "max_cpu",
            "max_mem_mb",
            "max_upload_mb",
            "max_log_bytes",
            "max_artifact_file_bytes",
            "max_artifact_total_bytes",
            "workspace_cap_mb",
            "artifact_ttl_hours",
            "supported_spec_versions",
            "isolation_warnings",
        ]
        for runtime in data["runtimes"]:
            for key in required_keys:
                assert key in runtime
            assert isinstance(runtime["isolation_warnings"], list)


def test_create_session_scaffold(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "timeout_sec": 60,
        }
        r = client.post("/api/v1/sandbox/sessions", json=body, headers={"Idempotency-Key": "abc-123"})
        assert r.status_code == 200
        j = r.json()
        assert "id" in j and j["runtime"] in {"docker", "firecracker"}
        # Replay with same key/body returns same id
        r2 = client.post("/api/v1/sandbox/sessions", json=body, headers={"Idempotency-Key": "abc-123"})
        assert r2.status_code == 200
        assert r2.json()["id"] == j["id"]
        # Change body with same key triggers 409
        body2 = {**body, "timeout_sec": 61}
        r3 = client.post("/api/v1/sandbox/sessions", json=body2, headers={"Idempotency-Key": "abc-123"})
        assert r3.status_code == 409


def test_create_session_returns_execution_defaults(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.12-slim",
            "cpu_limit": 1.5,
            "memory_mb": 768,
            "timeout_sec": 77,
            "network_policy": "deny_all",
            "env": {"SESSION_TOKEN": "present"},
            "labels": {"team": "sandbox"},
            "trust_level": "trusted",
        }
        r = client.post("/api/v1/sandbox/sessions", json=body)
        assert r.status_code == 200
        j = r.json()
        assert j["base_image"] == "python:3.12-slim"
        assert j["cpu_limit"] == 1.5
        assert j["memory_mb"] == 768
        assert j["timeout_sec"] == 77
        assert j["network_policy"] == "deny_all"
        assert j["env"] == {"SESSION_TOKEN": "present"}
        assert j["labels"] == {"team": "sandbox"}
        assert j["trust_level"] == "trusted"


def test_start_run_scaffold_returns_completed_with_metadata(monkeypatch) -> None:


    with _client(monkeypatch) as client:
        body: Dict[str, Any] = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["python", "-c", "print('hello')"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body, headers={"Idempotency-Key": "idem-run-1"})
        assert r.status_code == 200
        j = r.json()
        assert j["phase"] == "completed"
        # Spec and metadata fields present
        assert j.get("spec_version") == "1.0"
        assert j.get("runtime") in {"docker", "firecracker"}
        # policy_hash may be present; if provided, must be non-empty
        if "policy_hash" in j and j["policy_hash"] is not None:
            assert isinstance(j["policy_hash"], str) and len(j["policy_hash"]) > 0
        # Replay with same key/body returns same run id
        r2 = client.post("/api/v1/sandbox/runs", json=body, headers={"Idempotency-Key": "idem-run-1"})
        assert r2.status_code == 200
        assert r2.json()["id"] == j["id"]
        # Change body with same key triggers 409
        body2 = {**body, "timeout_sec": 6}
        r3 = client.post("/api/v1/sandbox/runs", json=body2, headers={"Idempotency-Key": "idem-run-1"})
        assert r3.status_code == 409


def test_start_run_rejects_missing_session_and_base_image(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "command": ["python", "-c", "print('hello')"],
            },
        )
        assert r.status_code == 422


def test_start_run_rejects_both_session_and_base_image(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        session_resp = client.post(
            "/api/v1/sandbox/sessions",
            json={
                "spec_version": "1.0",
                "runtime": "docker",
                "base_image": "python:3.11-slim",
            },
        )
        assert session_resp.status_code == 200
        session_id = str(session_resp.json()["id"])

        r = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "session_id": session_id,
                "base_image": "python:3.11-slim",
                "command": ["python", "-c", "print('hello')"],
            },
        )
        assert r.status_code == 422


def test_session_backed_run_inherits_session_defaults(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    captured: dict[str, Any] = {}

    def _fake_start_run_scaffold(*, user_id, spec, spec_version, idem_key, raw_body, explicit_fields=None):
        session = sb._service.get_session(spec.session_id) if spec.session_id else None
        if session is not None and explicit_fields is not None:
            if "runtime" not in explicit_fields and session.runtime is not None:
                spec.runtime = session.runtime
            if "base_image" not in explicit_fields and session.base_image:
                spec.base_image = session.base_image
            if "resources.cpu" not in explicit_fields and session.cpu_limit is not None:
                spec.cpu = session.cpu_limit
            if "resources.memory_mb" not in explicit_fields and session.memory_mb is not None:
                spec.memory_mb = session.memory_mb
            if "timeout_sec" not in explicit_fields and session.timeout_sec is not None:
                spec.timeout_sec = int(session.timeout_sec)
            if "network_policy" not in explicit_fields and session.network_policy is not None:
                spec.network_policy = session.network_policy
            if "env" not in explicit_fields:
                spec.env = dict(session.env or {})
            if "trust_level" not in explicit_fields and session.trust_level is not None:
                spec.trust_level = session.trust_level
        captured["user_id"] = user_id
        captured["spec"] = spec
        return RunStatus(
            id="run-session-defaults",
            phase=RunPhase.queued,
            spec_version=spec_version,
            runtime=spec.runtime or RuntimeType.docker,
            base_image=spec.base_image,
            session_id=spec.session_id,
            started_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(sb._service, "start_run_scaffold", _fake_start_run_scaffold)

    with _client(monkeypatch) as client:
        session_resp = client.post(
            "/api/v1/sandbox/sessions",
            json={
                "spec_version": "1.0",
                "runtime": "docker",
                "base_image": "python:3.12-slim",
                "cpu_limit": 1.5,
                "memory_mb": 768,
                "timeout_sec": 77,
                "network_policy": "deny_all",
                "env": {"SESSION_TOKEN": "present"},
                "trust_level": "trusted",
            },
        )
        assert session_resp.status_code == 200
        session_id = str(session_resp.json()["id"])

        run_resp = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "session_id": session_id,
                "command": ["python", "-c", "print('hello')"],
            },
        )
        assert run_resp.status_code == 200
        assert run_resp.json()["base_image"] == "python:3.12-slim"

    spec = captured["spec"]
    assert spec.session_id == session_id
    assert spec.runtime == RuntimeType.docker
    assert spec.base_image == "python:3.12-slim"
    assert spec.cpu == 1.5
    assert spec.memory_mb == 768
    assert spec.timeout_sec == 77
    assert spec.network_policy == "deny_all"
    assert spec.env == {"SESSION_TOKEN": "present"}
    assert spec.trust_level == TrustLevel.trusted


def test_session_backed_run_refreshes_session_defaults_after_prelock_work(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    old_session = Session(
        id="sess-refresh",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        expires_at=None,
        timeout_sec=30,
        env={"SESSION_TOKEN": "old"},
        trust_level=TrustLevel.standard,
    )
    new_session = Session(
        id="sess-refresh",
        runtime=RuntimeType.docker,
        base_image="python:3.12-slim",
        expires_at=None,
        timeout_sec=77,
        env={"SESSION_TOKEN": "new"},
        trust_level=TrustLevel.trusted,
    )
    state = {"session": old_session}
    captured: dict[str, Any] = {}

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(sb._service, "get_session", lambda session_id: state["session"])
    monkeypatch.setattr(sb._service._orch, "get_session", lambda session_id, **kwargs: state["session"])
    monkeypatch.setattr(sb._service._orch, "get_session_workspace_path", lambda session_id, **kwargs: str(tmp_path))

    def _switch_session_and_parse(files):
        state["session"] = new_session
        return []

    def _capture_enqueue(*, user_id, spec, spec_version, idem_key, body):
        captured["spec"] = spec
        return RunStatus(
            id="run-refresh",
            phase=RunPhase.queued,
            spec_version=spec_version,
            runtime=spec.runtime or RuntimeType.docker,
            base_image=spec.base_image,
            session_id=spec.session_id,
            started_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr(sb._service, "parse_inline_files", _switch_session_and_parse)
    monkeypatch.setattr(sb._service._orch, "enqueue_run", _capture_enqueue)

    with _client(monkeypatch) as client:
        run_resp = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "session_id": "sess-refresh",
                "command": ["python", "-c", "print('hello')"],
            },
        )
        assert run_resp.status_code == 200

    spec = captured["spec"]
    assert spec.base_image == "python:3.12-slim"
    assert spec.timeout_sec == 77
    assert spec.env == {"SESSION_TOKEN": "new"}
    assert spec.trust_level == TrustLevel.trusted


def test_session_backed_run_returns_not_found_if_session_disappears_during_start(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")
    monkeypatch.setattr(
        sb._service,
        "get_session",
        lambda session_id: SimpleNamespace(
            runtime=RuntimeType.docker,
            base_image="python:3.11-slim",
            env={},
            timeout_sec=None,
            cpu_limit=None,
            memory_mb=None,
            network_policy=None,
            trust_level=None,
            persona_id=None,
            workspace_id=None,
            workspace_group_id=None,
            scope_snapshot_id=None,
        ),
    )

    def _raise_session_not_found(*, user_id, spec, spec_version, idem_key, raw_body, explicit_fields=None):
        raise ValueError("session_not_found")

    monkeypatch.setattr(sb._service, "start_run_scaffold", _raise_session_not_found)

    with _client(monkeypatch) as client:
        response = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "session_id": "sess-gone",
                "command": ["python", "-c", "print('hello')"],
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "session_not_found"}


def test_delete_session_cancels_and_drains_active_runs(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    with _client(monkeypatch) as client:
        session_resp = client.post(
            "/api/v1/sandbox/sessions",
            json={
                "spec_version": "1.0",
                "runtime": "docker",
                "base_image": "python:3.11-slim",
            },
        )
        assert session_resp.status_code == 200
        session_id = str(session_resp.json()["id"])

        run_resp = client.post(
            "/api/v1/sandbox/runs",
            json={
                "spec_version": "1.0",
                "runtime": "docker",
                "session_id": session_id,
                "command": ["python", "-c", "print('queued')"],
                "timeout_sec": 15,
            },
        )
        assert run_resp.status_code == 200
        run_id = str(run_resp.json()["id"])
        assert run_resp.json()["phase"] == "queued"

        delete_resp = client.delete(f"/api/v1/sandbox/sessions/{session_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json().get("ok") is True

        run_after = client.get(f"/api/v1/sandbox/runs/{run_id}")
        assert run_after.status_code == 200
        assert run_after.json().get("phase") == "killed"


def test_restore_snapshot_returns_conflict_when_active_runs_exist(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_active_runs(session_id: str, snapshot_id: str) -> bool:
        raise SessionActiveRunsConflict(
            session_id=session_id,
            active_runs=1,
        )

    monkeypatch.setattr(sb._service, "restore_snapshot", _raise_active_runs)

    with _client(monkeypatch) as client:
        response = client.post(
            "/api/v1/sandbox/sessions/sess-restore-busy/restore",
            json={"snapshot_id": "snap-123"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "error": "session_has_active_runs",
            "active_runs": 1,
            "session_id": "sess-restore-busy",
        }
    }


def test_create_snapshot_returns_conflict_when_active_runs_exist(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_active_runs(session_id: str) -> dict[str, Any]:
        raise SessionActiveRunsConflict(
            session_id=session_id,
            active_runs=1,
        )

    monkeypatch.setattr(sb._service, "create_snapshot", _raise_active_runs)

    with _client(monkeypatch) as client:
        response = client.post("/api/v1/sandbox/sessions/sess-snapshot-busy/snapshot")

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "error": "session_has_active_runs",
            "active_runs": 1,
            "session_id": "sess-snapshot-busy",
        }
    }


def test_clone_session_returns_conflict_when_active_runs_exist(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_active_runs(session_id: str, new_name: str | None = None):
        raise SessionActiveRunsConflict(
            session_id=session_id,
            active_runs=1,
        )

    monkeypatch.setattr(sb._service, "clone_session", _raise_active_runs)

    with _client(monkeypatch) as client:
        response = client.post(
            "/api/v1/sandbox/sessions/sess-clone-busy/clone",
            json={"new_session_name": "copy"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "error": "session_has_active_runs",
            "active_runs": 1,
            "session_id": "sess-clone-busy",
        }
    }


def test_create_snapshot_sanitizes_oserror(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_oserror(session_id: str) -> dict[str, Any]:
        _ = session_id
        raise OSError("snapshot path exploded")

    monkeypatch.setattr(sb._service, "create_snapshot", _raise_oserror)

    with _client(monkeypatch) as client:
        response = client.post("/api/v1/sandbox/sessions/sess-snapshot-error/snapshot")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to create snapshot"


def test_restore_snapshot_sanitizes_oserror(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_oserror(session_id: str, snapshot_id: str) -> bool:
        _ = (session_id, snapshot_id)
        raise OSError("snapshot restore exploded")

    monkeypatch.setattr(sb._service, "restore_snapshot", _raise_oserror)

    with _client(monkeypatch) as client:
        response = client.post(
            "/api/v1/sandbox/sessions/sess-restore-error/restore",
            json={"snapshot_id": "snap-123"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to restore snapshot"


def test_clone_session_sanitizes_oserror(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb

    monkeypatch.setattr(sb, "_require_session_owner", lambda session_id, current_user: "1")

    def _raise_oserror(session_id: str, new_name: str | None = None):
        _ = (session_id, new_name)
        raise OSError("session clone exploded")

    monkeypatch.setattr(sb._service, "clone_session", _raise_oserror)

    with _client(monkeypatch) as client:
        response = client.post(
            "/api/v1/sandbox/sessions/sess-clone-error/clone",
            json={"new_session_name": "copy"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to clone session"
