import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _route_exists(path: str, method: str) -> bool:
    wanted_method = method.upper()
    for route in app.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == path and wanted_method in route_methods:
            return True
    return False


@pytest.mark.unit
def test_reembed_schedule_sanitizes_enqueue_failure(monkeypatch):
    if not _route_exists("/api/v1/embeddings/reembed/schedule", "POST"):
        pytest.skip("re-embed schedule route is disabled in this test app configuration")

    from tldw_Server_API.app.core.Embeddings import redis_pipeline
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(
        self,
        *,
        domain,
        queue,
        job_type,
        payload,
        owner_user_id,
        project_id=None,
        priority=5,
        max_retries=3,
        available_at=None,
        idempotency_key=None,
        request_id=None,
        trace_id=None,
        **kwargs,
    ):
        return {
            "id": 99,
            "uuid": "root-job-uuid",
            "domain": domain,
            "queue": queue,
            "job_type": job_type,
            "status": "queued",
        }

    def fake_fail_job(self, *args, **kwargs):
        return None

    def fake_enqueue_chunking_job(*, payload, root_job_uuid, force_regenerate=False, require_redis=True):
        raise RuntimeError("redis password leaked")

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)
    monkeypatch.setattr(jobs_manager.JobManager, "fail_job", fake_fail_job, raising=True)
    monkeypatch.setattr(redis_pipeline, "enqueue_chunking_job", fake_enqueue_chunking_job, raising=True)

    client = TestClient(app)
    response = client.post(
        "/api/v1/embeddings/reembed/schedule",
        json={"media_id": 1},
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to schedule re-embed"
