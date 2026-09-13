"""Synthetic upload-to-search validation; no personal mailbox or model access.

Auth/quota/billing and application lifespan are outside this focused harness.
The production endpoint functions, form parsing, file validation, email parser,
persistence and SQLite search/detail operations execute without substitutes.
"""

import mailbox
import socket
import zipfile
from email.message import EmailMessage
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_usage_event_logger
from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.api.v1.endpoints.media.add import add_media
from tldw_Server_API.app.api.v1.endpoints.media.process_emails import process_emails_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

pytestmark = pytest.mark.integration

OFFLINE_OPTIONS = {
    "media_type": "email",
    "perform_analysis": "false",
    "perform_claims_extraction": "false",
    "perform_chunking": "false",
    "auto_chunking_use_llm": "false",
    "generate_embeddings": "false",
    "keep_original_file": "false",
}


@pytest.fixture()
def offline_client(tmp_path, monkeypatch):
    """Use real temporary storage and fail even on swallowed forbidden calls."""
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import ChatAutoChunkBoundaryAssistant
    from tldw_Server_API.app.core.Claims_Extraction import claims_utils
    from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib

    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("model, background job, or outbound request")
        raise AssertionError("Offline email validation forbids model work and outbound requests")

    monkeypatch.setattr(Summarization_General_Lib, "analyze", forbidden)
    monkeypatch.setattr(claims_utils, "extract_claims_for_chunks", forbidden)
    monkeypatch.setattr(ChatAutoChunkBoundaryAssistant, "refine", forbidden)
    monkeypatch.setattr(EmbeddingsJobsAdapter, "create_job", forbidden)
    monkeypatch.setattr(BackgroundTasks, "add_task", forbidden)
    for name in (
        "fetch",
        "afetch",
        "apost",
        "fetch_json",
        "afetch_json",
        "download",
        "adownload",
        "astream_bytes",
        "astream_sse",
        "stream_response",
    ):
        monkeypatch.setattr(http_client, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setitem(persistence.settings, "EMAIL_NATIVE_PERSIST_ENABLED", True)
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    monkeypatch.setitem(email_endpoint.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", False)
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="offline-email-test")
    app = FastAPI()
    # Register production functions without route-level auth/billing dependencies.
    app.add_api_route("/api/v1/media/add", add_media, methods=["POST"])
    app.add_api_route("/api/v1/media/process-emails", process_emails_endpoint, methods=["POST"])
    app.add_api_route("/api/v1/email/search", email_endpoint.search_email_messages, methods=["GET"])
    app.add_api_route(
        "/api/v1/email/messages/{email_message_id}", email_endpoint.get_email_message_detail, methods=["GET"]
    )
    app.dependency_overrides[get_media_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: User(
        id=7, username="synthetic", email="synthetic@example.test", is_active=True, is_admin=True
    )
    app.dependency_overrides[get_usage_event_logger] = lambda: SimpleNamespace(log_event=lambda **_kwargs: None)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        db.close_connection()
        assert calls == [], f"Forbidden calls occurred, including any caught by application code: {calls}"


def synthetic_message(number: int, *, html_only: bool = False, same_body: bool = False) -> EmailMessage:
    """Build synthetic MIME mail, optionally reproducing TASK-13250's collision."""
    message = EmailMessage()
    message["From"] = "Synthetic Sender <sender@example.test>"
    message["To"] = "Synthetic Reader <reader@example.test>"
    message["Cc"] = "Copy <copy@example.test>"
    message["Bcc"] = "Hidden <hidden@example.test>"
    message["Subject"] = f"Synthetic café report {number}"
    message["Message-ID"] = f"<synthetic-{number}@example.test>"
    message["Date"] = "Tue, 10 Feb 2026 09:30:00 -0500"
    body = "Uniquequartz synthetic body." if same_body else f"Uniquequartz synthetic body {number}."
    message.set_content(f"<p>{body}</p>" if html_only else body, subtype="html" if html_only else "plain")
    message.add_attachment(
        b"attachment-only-token",
        maintype="application",
        subtype="octet-stream",
        filename="synthetic.bin",
        cid="<synthetic-attachment>",
    )
    return message


def upload(client, filename: str, data: bytes, **options):
    """Submit the real multipart form and reject partial processing errors."""
    response = client.post(
        "/api/v1/media/add",
        files={"files": (filename, data, "application/octet-stream")},
        data={**OFFLINE_OPTIONS, **options},
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["status"] == "Success", result
    assert not result.get("analysis"), result
    assert not result.get("embeddings_scheduled"), result
    return result


def search(client, query: str):
    """Query the production operator planner and SQLite FTS through HTTP."""
    response = client.get("/api/v1/email/search", params={"q": query})
    assert response.status_code == 200, response.text
    return response.json()["items"]


@pytest.mark.parametrize("container", ["eml", "zip", "mbox"])
def test_synthetic_upload_search_detail_and_reimport(offline_client, tmp_path, container):
    messages = [synthetic_message(1), synthetic_message(2)]
    if container == "eml":
        uploads = [(f"synthetic-{index}.eml", msg.as_bytes()) for index, msg in enumerate(messages)]
        options = {}
    elif container == "zip":
        output = BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for index, msg in enumerate(messages):
                archive.writestr(f"synthetic-{index}.eml", msg.as_bytes())
        uploads = [("synthetic.zip", output.getvalue())]
        options = {"accept_archives": "true"}
    else:
        path = tmp_path / "synthetic.mbox"
        archive = mailbox.mbox(str(path))
        try:
            for msg in messages:
                archive.add(msg)
            archive.flush()
        finally:
            archive.close()
        uploads = [("synthetic.mbox", path.read_bytes())]
        options = {"accept_mbox": "true"}

    for filename, data in uploads:
        upload(offline_client, filename, data, **options)
    rows = search(offline_client, "from:sender@example.test has:attachment uniquequartz")
    assert len(rows) == 2, rows
    ids = {row["email_message_id"] for row in rows}
    assert len(ids) == 2
    for row in rows:
        response = offline_client.get(f"/api/v1/email/messages/{row['email_message_id']}")
        assert response.status_code == 200, response.text
        detail = response.json()
        assert detail["body_text"] in {"Uniquequartz synthetic body 1.", "Uniquequartz synthetic body 2."}
        assert detail["message_id"] in {"<synthetic-1@example.test>", "<synthetic-2@example.test>"}
        assert detail["subject"].startswith("Synthetic café report")
        for role, address in [("from", "sender"), ("to", "reader"), ("cc", "copy"), ("bcc", "hidden")]:
            assert detail["participants"][role][0]["email"] == f"{address}@example.test"
        attachment = detail["attachments"][0]
        assert attachment["filename"] == "synthetic.bin"
        assert attachment["size_bytes"] == len(b"attachment-only-token")
        assert attachment["content_type"] == "application/octet-stream"
        assert attachment["content_id"] == "<synthetic-attachment>"
        assert attachment["extracted_text_available"] is False
    assert search(offline_client, "attachment-only-token") == []
    assert len(search(offline_client, "to:reader@example.test cc:copy@example.test bcc:hidden@example.test")) == 2
    assert len(search(offline_client, "after:2026-02-09 before:2026-02-11")) == 2
    for filename, data in uploads:
        upload(offline_client, filename, data, **options)
    assert {row["email_message_id"] for row in search(offline_client, "uniquequartz")} == ids


def test_html_only_synthetic_email_is_searchable(offline_client):
    upload(offline_client, "synthetic-html.eml", synthetic_message(3, html_only=True).as_bytes())
    rows = search(offline_client, "uniquequartz")
    assert len(rows) == 1
    detail = offline_client.get(f"/api/v1/email/messages/{rows[0]['email_message_id']}").json()
    assert detail["body_text"] == "Uniquequartz synthetic body 3."


def test_same_body_upload_collision_characterization(offline_client):
    """Record a known limitation, not acceptance of FR-INGEST-001 (TASK-13251).

    The initial acceptance probe expected two rows and failed for EML/ZIP/MBOX.
    This diagnostic preserves the current reproducible evidence until identity
    dedupe is fixed; then replace it with a two-message regression assertion.
    """
    first = upload(offline_client, "first.eml", synthetic_message(1, same_body=True).as_bytes())
    second = upload(offline_client, "second.eml", synthetic_message(2, same_body=True).as_bytes())
    assert first["db_id"] == second["db_id"]
    rows = search(offline_client, "uniquequartz")
    assert len(rows) == 1
    detail = offline_client.get(f"/api/v1/email/messages/{rows[0]['email_message_id']}").json()
    assert detail["message_id"] == "<synthetic-2@example.test>"


def test_process_only_endpoint_does_not_persist(offline_client):
    """Processing results must be ingested via /media/add to become searchable."""
    response = offline_client.post(
        "/api/v1/media/process-emails",
        files={"files": ("process-only.eml", synthetic_message(4).as_bytes(), "message/rfc822")},
        data=OFFLINE_OPTIONS,
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["content"] == "Uniquequartz synthetic body 4."
    assert result["db_id"] is None
    assert result["analysis"] is None
    assert search(offline_client, "uniquequartz") == []
