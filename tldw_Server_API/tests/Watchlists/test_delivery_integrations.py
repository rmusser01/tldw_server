import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=945, username="wluser", email="wl@example.com", is_active=True)

    base_dir = tmp_path / "test_user_dbs_delivery_integrations"
    base_dir.mkdir(parents=True, exist_ok=True)
    template_dir = tmp_path / "watchlist_templates_delivery"
    template_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("WATCHLIST_TEMPLATE_DIR", str(template_dir))
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("EMAIL_PROVIDER", "mock")

    from fastapi import FastAPI
    from tldw_Server_API.app.core.config import API_V1_PREFIX
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_watchlist_email_attachment_filename_sanitizes_separators_and_length():
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    title = "Daily / Digest \\ Briefing " + ("x" * 300)
    filename = watchlists._build_watchlist_email_attachment_filename(title, 55, "md")

    assert filename.endswith(".md")
    assert "/" not in filename
    assert "\\" not in filename
    assert "\r" not in filename
    assert "\n" not in filename
    assert len(filename) <= 255

    fallback_filename = watchlists._build_watchlist_email_attachment_filename("/\\", 55, "md")
    assert fallback_filename == "watchlist-output-55.md"


def test_output_deliveries_email_and_chatbook(client_with_user: TestClient):
    c = client_with_user

    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Delivery Feed",
            "url": "https://example.com/feed-delivery.xml",
            "source_type": "rss",
            "tags": ["deliveries"],
        },
    )
    assert source.status_code == 200, source.text

    job = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Delivery Digest",
            "scope": {"tags": ["deliveries"]},
            "output_prefs": {
                "deliveries": {
                    "email": {"enabled": True, "recipients": ["default@example.com"], "attach_file": False},
                    "chatbook": {"enabled": True, "metadata": {"category": "digest"}},
                }
            },
        },
    )
    assert job.status_code == 200, job.text
    job_id = job.json()["id"]

    run = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert run.status_code == 200, run.text
    run_id = run.json()["id"]

    output = c.post(
        "/api/v1/watchlists/outputs",
        json={
            "run_id": run_id,
            "title": "Digest",
            "deliveries": {
                "email": {"subject": "Daily Digest", "recipients": ["override@example.com"], "attach_file": True},
                "chatbook": {"title": "Digest Document", "description": "Auto", "metadata": {"origin": "test"}},
            },
        },
    )
    assert output.status_code == 200, output.text
    payload = output.json()

    deliveries = payload.get("metadata", {}).get("deliveries", [])
    assert len(deliveries) == 2
    channels = {entry["channel"] for entry in deliveries}
    assert channels == {"email", "chatbook"}

    email_result = next(entry for entry in deliveries if entry["channel"] == "email")
    assert email_result["status"] in {"sent", "partial"}

    chatbook_result = next(entry for entry in deliveries if entry["channel"] == "chatbook")
    assert chatbook_result["status"] in {"stored", "failed"}

    chatbook_id = payload.get("metadata", {}).get("chatbook_document_id")
    if chatbook_result["status"] == "stored":
        assert isinstance(chatbook_id, int)
        assert payload.get("chatbook_path") == f"generated_document:{chatbook_id}"
        db_path = DatabasePaths.get_chacha_db_path(945)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT metadata FROM generated_documents WHERE id = ?", (chatbook_id,)).fetchone()
        assert row is not None
        stored_meta = json.loads(row[0])
        assert stored_meta.get("job_id") == job_id
        assert stored_meta.get("run_id") == run_id
        assert stored_meta.get("origin") == "test"


def test_output_delivery_uses_job_default_email_subject_when_not_overridden(
    client_with_user: TestClient,
    monkeypatch,
):
    c = client_with_user

    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Delivery Subject Feed",
            "url": "https://example.com/feed-delivery-subject.xml",
            "source_type": "rss",
            "tags": ["delivery-subject"],
        },
    )
    assert source.status_code == 200, source.text

    job = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Delivery Subject Digest",
            "scope": {"tags": ["delivery-subject"]},
            "output_prefs": {
                "deliveries": {
                    "email": {
                        "enabled": True,
                        "recipients": ["subject-default@example.com"],
                        "subject": "Watchlists Subject Default",
                        "attach_file": False,
                    }
                }
            },
        },
    )
    assert job.status_code == 200, job.text
    job_id = job.json()["id"]

    run = c.post(f"/api/v1/watchlists/jobs/{job_id}/run")
    assert run.status_code == 200, run.text
    run_id = run.json()["id"]

    captured: dict[str, str] = {}

    class _FakeNotificationsService:
        def __init__(self, *, user_id: int, user_email: str | None = None) -> None:
            self.user_id = user_id
            self.user_email = user_email

        async def deliver_email(
            self,
            *,
            subject: str,
            html_body: str,  # noqa: ARG002
            text_body: str | None = None,  # noqa: ARG002
            recipients: list[str] | None = None,  # noqa: ARG002
            attachments: list[dict[str, object]] | None = None,  # noqa: ARG002
            fallback_to_user_email: bool = True,  # noqa: ARG002
        ):
            captured["subject"] = subject
            return type(
                "_Result",
                (),
                {"channel": "email", "status": "sent", "details": {"provider": "fake"}},
            )()

        def deliver_chatbook(self, *args, **kwargs):  # noqa: ANN001, D401
            return type(
                "_Result",
                (),
                {"channel": "chatbook", "status": "skipped", "details": {"reason": "disabled"}},
            )()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.NotificationsService",
        _FakeNotificationsService,
    )

    output = c.post(
        "/api/v1/watchlists/outputs",
        json={
            "run_id": run_id,
            "title": "Digest Without Override Subject",
        },
    )
    assert output.status_code == 200, output.text
    payload = output.json()

    assert captured.get("subject") == "Watchlists Subject Default"
    deliveries = payload.get("metadata", {}).get("deliveries", [])
    assert len(deliveries) == 1
    assert deliveries[0].get("channel") == "email"
    assert deliveries[0].get("status") == "sent"
