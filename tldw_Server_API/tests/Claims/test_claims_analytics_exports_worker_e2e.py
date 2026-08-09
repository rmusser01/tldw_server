from __future__ import annotations

import asyncio
import errno
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import claims as claims_endpoint
from tldw_Server_API.app.api.v1.endpoints.claims import router as claims_router
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers, claims_service
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import (
    MediaDatabase,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

pytestmark = pytest.mark.integration


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="claims-export-e2e",
        roles=["admin"],
        permissions=[CLAIMS_ADMIN],
        is_admin=True,
    )


def _app(db: MediaDatabase) -> FastAPI:
    app = FastAPI()
    app.include_router(claims_router, prefix="/api/v1")
    app.dependency_overrides[claims_endpoint.get_auth_principal] = _principal
    app.dependency_overrides[claims_endpoint.get_request_user] = lambda: SimpleNamespace(
        id=1,
        username="claims-export-e2e",
    )
    app.dependency_overrides[get_media_db_for_user] = lambda: db
    return app


async def _run_worker_until_completed(manager: JobManager) -> None:
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="claims",
            queue="default",
            worker_id="claims-export-e2e",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )

    async def on_completed(_job: dict[str, Any], _result: dict[str, Any]) -> None:
        sdk.stop()

    await asyncio.wait_for(
        sdk.run(handler=claims_job_handlers.process_claims_job, on_completed=on_completed),
        timeout=2,
    )


@pytest.mark.asyncio
async def test_claims_analytics_export_api_worker_and_download_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_path = tmp_path / "jobs.sqlite"
    media_path = tmp_path / "owner-media.sqlite"
    manager = JobManager(jobs_path)
    db = MediaDatabase(db_path=str(media_path), client_id="claims-export-e2e")
    app = _app(db)
    try:
        monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "1")
        monkeypatch.setenv("CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED", "1")
        monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: manager)
        monkeypatch.setattr(
            claims_job_handlers,
            "get_user_media_db_path",
            lambda _owner_user_id: media_path,
        )
        event = db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="high",
            payload_json='{"ratio":0.75}',
        )

        with TestClient(app) as client:
            created = client.post(
                "/api/v1/claims/analytics/export",
                json={
                    "format": "json",
                    "filters": {"event_type": "unsupported_ratio"},
                    "pagination": {"limit": 10, "offset": 0},
                },
            )
            assert created.status_code == 202, created.text
            created_body = created.json()
            export_id = created_body["export_id"]
            job_id = int(created_body["job_id"])
            assert created_body["status"] == "queued"
            assert created_body["job_status"] == "queued"

            await _run_worker_until_completed(manager)

            assert manager.get_job(job_id)["status"] == "completed"
            artifact = db.get_claims_analytics_export(export_id, user_id="1")
            assert artifact["status"] == "ready"
            listed = client.get("/api/v1/claims/analytics/exports")
            assert listed.status_code == 200, listed.text
            listed_export = listed.json()["exports"]
            assert len(listed_export) == 1
            assert listed_export[0]["export_id"] == export_id
            assert listed_export[0]["status"] == "ready"
            assert listed_export[0]["job_id"] == job_id
            assert listed_export[0]["job_status"] == "completed"
            assert listed_export[0]["filters"] == {
                "workspace_id": None,
                "event_type": "unsupported_ratio",
                "severity": None,
                "provider": None,
                "model": None,
                "start_time": None,
                "end_time": artifact["snapshot_at"],
            }
            assert listed_export[0]["pagination"] == {
                "limit": 10,
                "offset": 0,
                "total": None,
            }
            assert listed_export[0]["download_url"] == (
                f"/api/v1/claims/analytics/export/{export_id}"
            )
            downloaded = client.get(f"/api/v1/claims/analytics/export/{export_id}")
            assert downloaded.status_code == 200, downloaded.text
            assert downloaded.json() == {
                "events": [
                    {
                        "id": int(event["id"]),
                        "user_id": "1",
                        "event_type": "unsupported_ratio",
                        "severity": "high",
                        "created_at": event["created_at"],
                        "delivered_at": None,
                        "payload": {"ratio": 0.75},
                    }
                ],
                "filters": {
                    "event_type": "unsupported_ratio",
                    "end_time": artifact["snapshot_at"],
                },
                "pagination": {"limit": 10, "offset": 0, "total": 1},
            }
    finally:
        app.dependency_overrides.clear()
        db.close_connection()


@pytest.mark.asyncio
async def test_claims_analytics_export_retry_recovers_and_late_failure_cannot_overwrite_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_path = tmp_path / "retry-jobs.sqlite"
    media_path = tmp_path / "retry-owner-media.sqlite"
    manager = JobManager(jobs_path)
    db = MediaDatabase(db_path=str(media_path), client_id="claims-export-retry")
    app = _app(db)
    real_ready = MediaDatabase.mark_claims_analytics_export_ready
    real_transition = MediaDatabase.transition_claims_analytics_export_status
    ready_attempts = 0
    transition_calls: list[tuple[tuple[str, ...], str, bool]] = []
    try:
        monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "1")
        monkeypatch.setenv("CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED", "1")
        monkeypatch.setenv("CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT", "1")
        monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: manager)
        monkeypatch.setattr(
            claims_job_handlers,
            "get_user_media_db_path",
            lambda _owner_user_id: media_path,
        )

        def fail_once_after_processing(media_db: MediaDatabase, **kwargs: Any) -> bool:
            nonlocal ready_attempts
            ready_attempts += 1
            if ready_attempts == 1:
                raise OSError(errno.EBUSY, "temporary storage busy")
            return real_ready(media_db, **kwargs)

        monkeypatch.setattr(
            MediaDatabase,
            "mark_claims_analytics_export_ready",
            fail_once_after_processing,
        )

        def record_transition(media_db: MediaDatabase, **kwargs: Any) -> bool:
            transitioned = real_transition(media_db, **kwargs)
            transition_calls.append(
                (
                    tuple(kwargs["from_statuses"]),
                    str(kwargs["to_status"]),
                    transitioned,
                )
            )
            return transitioned

        monkeypatch.setattr(
            MediaDatabase,
            "transition_claims_analytics_export_status",
            record_transition,
        )

        with TestClient(app) as client:
            created = client.post(
                "/api/v1/claims/analytics/export",
                json={"format": "json", "pagination": {"limit": 1, "offset": 0}},
            )
            assert created.status_code == 202, created.text
            export_id = created.json()["export_id"]
            job_id = int(created.json()["job_id"])

            first_sdk = WorkerSDK(
                manager,
                WorkerConfig(
                    domain="claims",
                    queue="default",
                    worker_id="claims-export-retry-first",
                    lease_seconds=5,
                    renew_threshold_seconds=1,
                    renew_jitter_seconds=0,
                    retry_backoff_seconds=0,
                ),
            )

            async def stop_after_failure(_job: dict[str, Any]) -> dict[str, Any]:
                try:
                    return await claims_job_handlers.process_claims_job(_job)
                finally:
                    first_sdk.stop()

            await asyncio.wait_for(first_sdk.run(handler=stop_after_failure), timeout=2)
            first_job = manager.get_job(job_id)
            assert first_job["status"] == "queued"
            assert first_job["retry_count"] == 1
            assert db.get_claims_analytics_export(export_id, user_id="1")["status"] == "failed"

            await _run_worker_until_completed(manager)

            assert manager.get_job(job_id)["status"] == "completed"
            ready = db.get_claims_analytics_export(export_id, user_id="1")
            assert ready["status"] == "ready"
            assert ready_attempts == 2
            assert (("failed",), "processing", True) in transition_calls
            assert db.transition_claims_analytics_export_status(
                export_id=export_id,
                user_id="1",
                from_statuses=("processing",),
                to_status="failed",
                error_code="late_failure",
                error_message="must not replace ready",
            ) is False
            late_result = await claims_job_handlers.process_claims_job(manager.get_job(job_id))
            assert late_result == {
                "outcome": "skipped",
                "reason": "already_ready",
                "export_id": export_id,
            }
            assert db.get_claims_analytics_export(export_id, user_id="1")["status"] == "ready"
    finally:
        app.dependency_overrides.clear()
        db.close_connection()
