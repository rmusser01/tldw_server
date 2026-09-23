from __future__ import annotations

import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import slack as slack_endpoint


def _sign(secret: str, timestamp: int, body: bytes) -> str:
    base = f"v0:{timestamp}:".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


@pytest.fixture()
def slack_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-signing-secret")
    monkeypatch.setenv("SLACK_REPLAY_WINDOW_SECONDS", "300")
    monkeypatch.setenv("SLACK_INGRESS_RATE_LIMIT_PER_MINUTE", "1000")
    slack_endpoint._reset_slack_state_for_tests()

    app = FastAPI()
    app.include_router(slack_endpoint.router, prefix="/api/v1")
    return TestClient(app)


def test_slack_commands_parse_rag_route(slack_client: TestClient) -> None:
    form = urlencode(
        {
            "team_id": "T1",
            "user_id": "U1",
            "command": "/tldw",
            "text": "rag latest release notes",
            "trigger_id": "1337.42",
        }
    )
    body = form.encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-slack-request-timestamp": str(ts),
        "x-slack-signature": _sign("test-signing-secret", ts, body),
        "content-type": "application/x-www-form-urlencoded",
    }

    response = slack_client.post("/api/v1/slack/commands", data=body, headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["parsed"]["action"] == "rag"
    assert payload["parsed"]["route"] == "rag.search"
    assert payload["parsed"]["input"] == "latest release notes"
    job_id = payload["job_id"]
    assert isinstance(job_id, int)
    assert payload["response_mode"] == "ephemeral"

    # The REST job lookup is an admin-only ops route: it reads the global jobs
    # table with no tenant scope, so an unauthenticated caller used to be able
    # to enumerate every tenant's jobs. Users get status from the signed
    # in-band "status" command instead, which scopes by actor.
    job_status = slack_client.get(f"/api/v1/slack/jobs/{job_id}")
    assert job_status.status_code in (401, 403)


def test_slack_commands_status_queries_job(slack_client: TestClient) -> None:
    queue_form = urlencode(
        {
            "team_id": "T1",
            "user_id": "U1",
            "command": "/tldw",
            "text": "ask hello",
            "trigger_id": "1337.50",
        }
    )
    queue_body = queue_form.encode("utf-8")
    queue_ts = int(time.time())
    queue_headers = {
        "x-slack-request-timestamp": str(queue_ts),
        "x-slack-signature": _sign("test-signing-secret", queue_ts, queue_body),
        "content-type": "application/x-www-form-urlencoded",
    }
    queued = slack_client.post("/api/v1/slack/commands", data=queue_body, headers=queue_headers)
    assert queued.status_code == 200
    queued_payload = queued.json()
    job_id = queued_payload["job_id"]
    assert isinstance(job_id, int)

    status_form = urlencode(
        {
            "team_id": "T1",
            "user_id": "U1",
            "command": "/tldw",
            "text": f"status {job_id}",
            "trigger_id": "1337.51",
        }
    )
    status_body = status_form.encode("utf-8")
    status_ts = int(time.time())
    status_headers = {
        "x-slack-request-timestamp": str(status_ts),
        "x-slack-signature": _sign("test-signing-secret", status_ts, status_body),
        "content-type": "application/x-www-form-urlencoded",
    }
    status_response = slack_client.post("/api/v1/slack/commands", data=status_body, headers=status_headers)
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "accepted"
    assert status_payload["job"]["id"] == job_id


def test_slack_commands_unknown_returns_usage(slack_client: TestClient) -> None:
    form = urlencode(
        {
            "team_id": "T1",
            "user_id": "U1",
            "command": "/tldw",
            "text": "foobar do thing",
            "trigger_id": "1337.42",
        }
    )
    body = form.encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-slack-request-timestamp": str(ts),
        "x-slack-signature": _sign("test-signing-secret", ts, body),
        "content-type": "application/x-www-form-urlencoded",
    }

    response = slack_client.post("/api/v1/slack/commands", data=body, headers=headers)
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "unknown_command"
    assert "Supported commands" in payload["usage"]


def test_slack_events_app_mention_defaults_to_ask(slack_client: TestClient) -> None:
    event_payload = {
        "type": "event_callback",
        "event_id": "EvMention1",
        "event": {
            "type": "app_mention",
            "text": "<@U123> what is the uptime?",
        },
    }
    body = json.dumps(event_payload).encode("utf-8")
    ts = int(time.time())
    headers = {
        "x-slack-request-timestamp": str(ts),
        "x-slack-signature": _sign("test-signing-secret", ts, body),
        "content-type": "application/json",
    }

    response = slack_client.post("/api/v1/slack/events", data=body, headers=headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["parsed"]["action"] == "ask"
    assert payload["parsed"]["route"] == "chat.ask"
    assert payload["parsed"]["input"] == "what is the uptime?"
