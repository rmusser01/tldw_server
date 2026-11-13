from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import research_runs
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Research.service import ResearchService


class DummyJobs:
    def __init__(self):
        self.next_id = 10

    def create_job(self, **kwargs):
        self.next_id += 1
        return {"id": self.next_id, "uuid": f"job-{self.next_id}", "status": "queued", **kwargs}

    def cancel_job(self, job_id: int, *, reason: str | None = None):
        return True


def _research_client(tmp_path):
    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service
    return TestClient(app), service


def test_research_runs_list_supports_status_offset_and_session_filter(tmp_path):
    client, service = _research_client(tmp_path)
    with client:
        first = client.post("/api/v1/research/runs", json={"query": "First research"}).json()
        second = client.post("/api/v1/research/runs", json={"query": "Second research"}).json()
        service._db_for_user("1").update_status_with_event(
            first["id"],
            status="cancelled",
            owner_user_id="1",
            event_type="status",
            event_payload={"id": first["id"], "status": "cancelled"},
            phase="drafting_plan",
            control_state="cancelled",
            active_job_id=None,
        )

        cancelled = client.get("/api/v1/research/runs", params={"status": "cancelled", "limit": 10})
        assert cancelled.status_code == 200
        assert [item["id"] for item in cancelled.json()] == [first["id"]]

        offset = client.get("/api/v1/research/runs", params={"offset": 1, "limit": 1})
        assert offset.status_code == 200
        assert [item["id"] for item in offset.json()] == [first["id"]]

        filtered = client.get("/api/v1/research/runs", params={"session_id": second["id"], "limit": 10})
        assert filtered.status_code == 200
        assert [item["id"] for item in filtered.json()] == [second["id"]]


def test_delete_research_run_removes_it_from_server_listing(tmp_path):
    client, _service = _research_client(tmp_path)
    with client:
        created = client.post("/api/v1/research/runs", json={"query": "Disposable research"}).json()

        delete_response = client.delete(f"/api/v1/research/runs/{created['id']}")

        assert delete_response.status_code == 200
        assert delete_response.json() == {"deleted": True}
        list_response = client.get("/api/v1/research/runs", params={"limit": 10})
        assert list_response.status_code == 200
        assert all(item["id"] != created["id"] for item in list_response.json())


def test_research_stream_disables_intermediary_compression(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("RESEARCH_RUNS_SSE_TEST_MAX_SECONDS", "0.02")
    client, _service = _research_client(tmp_path)
    with client:
        run = client.post("/api/v1/research/runs", json={"query": "Rowan Observatory"}).json()
        response = client.get(f"/api/v1/research/runs/{run['id']}/events/stream")
        assert response.status_code == 200  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert "no-cache" in response.headers["cache-control"]  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert "no-transform" in response.headers["cache-control"]  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert response.headers["x-accel-buffering"] == "no"  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.


@pytest.mark.parametrize("live", [False, True])
def test_research_stream_current_checkpoint_keeps_reviewable_proposal(tmp_path, monkeypatch, live):
    import json

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("RESEARCH_RUNS_SSE_TEST_MAX_SECONDS", "0.2")
    monkeypatch.setenv("RESEARCH_RUNS_SSE_POLL_INTERVAL", "0.01")
    client, service = _research_client(tmp_path)
    proposal = {"query": "Rowan Observatory", "focus_areas": ["telescope"], "stop_criteria": {"min_cited_sections": 1}}
    with client:
        run = client.post("/api/v1/research/runs", json={"query": proposal["query"]}).json()
        db = service._db_for_user("1")
        checkpoints = []

        def publish():
            checkpoint = db.create_checkpoint(
                session_id=run["id"], checkpoint_type="plan_review", proposed_payload=proposal
            )
            db.update_phase(run["id"], phase="awaiting_plan_review", status="waiting_human", active_job_id=None)
            db.record_run_event(
                owner_user_id="1",
                session_id=run["id"],
                event_type="checkpoint",
                event_payload=service._checkpoint_event_payload(checkpoint=checkpoint, phase="awaiting_plan_review"),
                phase="awaiting_plan_review",
            )
            checkpoints.append(checkpoint)

        if live:
            list_events = service.list_run_events_after
            reads = 0

            def events_after(**kwargs):
                nonlocal reads
                reads += 1
                if reads == 2:
                    publish()
                return list_events(**kwargs)

            monkeypatch.setattr(service, "list_run_events_after", events_after)
        else:
            publish()
        response = client.get(f"/api/v1/research/runs/{run['id']}/events/stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for frame in response.text.split("\n\n")
            if "event: checkpoint" in frame
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]
        assert len(events) == len(checkpoints) == 1  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert events[0]["checkpoint_id"] == checkpoints[0].id  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert events[0].get("proposed_payload") == proposal  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
        assert db.get_checkpoint(checkpoints[0].id).status == "pending"  # nosec B101 # Pytest regression checks SSE delivery and checkpoint integrity.
