import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.media_add_deps import get_add_media_form
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.integration


def _load_ingest_jobs_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "app"
        / "api"
        / "v1"
        / "endpoints"
        / "media"
        / "ingest_jobs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tldw_test_ingest_jobs_events_stream",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load ingest_jobs module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for model_name in (
        "MediaIngestJobItem",
        "SubmitMediaIngestJobsResponse",
        "MediaIngestJobStatus",
        "CancelMediaIngestJobResponse",
        "MediaIngestJobListResponse",
    ):
        model_cls = getattr(module, model_name, None)
        if model_cls is not None and hasattr(model_cls, "model_rebuild"):
            model_cls.model_rebuild(_types_namespace=module.__dict__)
    return module


@pytest.fixture(scope="module")
def ingest_jobs_module():
    return _load_ingest_jobs_module()


@pytest.fixture()
def test_client(ingest_jobs_module):
    app = FastAPI()
    app.include_router(ingest_jobs_module.router, prefix="/api/v1/media")

    async def _override_add_media_form(request: Request):
        form = await request.form()
        media_type = str(form.get("media_type") or "").strip()
        urls_raw = [str(item).strip() for item in form.getlist("urls")]
        urls = [item for item in urls_raw if item]
        return AddMediaForm(
            media_type=media_type,
            urls=urls or None,
        )

    async def _override_user():
        return User(
            id=1,
            username="owner",
            email="owner@example.com",
            role="user",
            is_active=True,
            is_superuser=False,
            is_admin=False,
        )

    async def _override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="user:1",
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=["media.create"],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[get_add_media_form] = _override_add_media_form
    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _set_jobs_db(monkeypatch, tmp_path, ingest_jobs_module):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    ingest_jobs_module._job_manager_cache.clear()


def _collect_sse_events(iter_lines, *, timeout_s: float, stop_predicate=None):
    deadline = time.time() + timeout_s
    events: list[dict] = []
    current_event = None

    for raw_line in iter_lines:
        if time.time() > deadline:
            break
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload = line.split(":", 1)[1].strip()
        if payload == "[DONE]":
            events.append({"event": current_event or "done", "data": payload})
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = payload
        item = {"event": current_event, "data": data}
        events.append(item)
        if stop_predicate is not None and stop_predicate(item):
            break

    return events


def test_media_ingest_events_stream_user_scoped(test_client, ingest_jobs_module, monkeypatch, tmp_path):
    _set_jobs_db(monkeypatch, tmp_path, ingest_jobs_module)
    monkeypatch.setenv("JOBS_SSE_TEST_MAX_SECONDS", "2")

    submit_resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={"media_type": "audio", "urls": "https://example.com/stream-owned-audio.mp3"},
    )
    assert submit_resp.status_code == 200, submit_resp.text
    submit_body = submit_resp.json()
    batch_id = submit_body.get("batch_id")
    assert batch_id
    job_id = int(submit_body["jobs"][0]["id"])

    def _cancel_job_later():
        time.sleep(0.2)
        test_client.delete(f"/api/v1/media/ingest/jobs/{job_id}")

    cancel_thread = threading.Thread(target=_cancel_job_later, daemon=True)
    cancel_thread.start()

    with test_client.stream(
        "GET",
        f"/api/v1/media/ingest/jobs/events/stream?batch_id={batch_id}",
    ) as stream_resp:
        assert stream_resp.status_code == 200, stream_resp.text
        assert "text/event-stream" in stream_resp.headers.get("content-type", "").lower()

        events = _collect_sse_events(
            stream_resp.iter_lines(),
            timeout_s=4,
            stop_predicate=lambda item: (
                item.get("event") == "job"
                and isinstance(item.get("data"), dict)
                and item["data"].get("event_type") == "job.cancelled"
            ),
        )

    cancel_thread.join(timeout=1)

    snapshot_event = next((ev for ev in events if ev.get("event") == "snapshot"), None)
    assert snapshot_event is not None, f"Missing snapshot event. Seen events: {events!r}"
    snapshot_jobs = (snapshot_event.get("data") or {}).get("jobs") or []
    assert any(int(job.get("id")) == job_id for job in snapshot_jobs), snapshot_jobs

    assert any(
        ev.get("event") == "job"
        and isinstance(ev.get("data"), dict)
        and int(ev["data"].get("job_id") or -1) == job_id
        and isinstance(ev["data"].get("attrs"), dict)
        for ev in events
    ), f"Missing job event for job_id={job_id}. Seen events: {events!r}"


def test_media_ingest_events_stream_non_owner_forbidden(
    test_client,
    ingest_jobs_module,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path, ingest_jobs_module)

    submit_resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={"media_type": "audio", "urls": "https://example.com/not-owned-audio.mp3"},
    )
    assert submit_resp.status_code == 200, submit_resp.text
    batch_id = submit_resp.json().get("batch_id")
    assert batch_id

    async def _override_other_user():
        return User(
            id=999,
            username="other-user",
            email="other@example.com",
            role="user",
            is_active=True,
            is_superuser=False,
            is_admin=False,
        )

    async def _override_other_principal():
        return AuthPrincipal(
            kind="user",
            user_id=999,
            api_key_id=None,
            subject="user:999",
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=["media.create"],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )

    test_client.app.dependency_overrides[get_request_user] = _override_other_user
    test_client.app.dependency_overrides[get_auth_principal] = _override_other_principal
    try:
        denied_resp = test_client.get(
            f"/api/v1/media/ingest/jobs/events/stream?batch_id={batch_id}",
        )
    finally:
        test_client.app.dependency_overrides.pop(get_request_user, None)
        test_client.app.dependency_overrides.pop(get_auth_principal, None)

    assert denied_resp.status_code == 403, denied_resp.text
