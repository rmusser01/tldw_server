"""Media-ingest Jobs routes must bind claims to PostgreSQL's real FORCE policies."""

import json
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def owned_jobs(pg_restricted_backend, monkeypatch):
    """Reuse the official non-bypass role and unchanged Jobs schema/policies."""
    config = pg_restricted_backend.config
    dsn = "postgresql://{}:{}@{}:{}/{}".format(
        quote(config.pg_user, safe=""),
        quote(config.pg_password, safe=""),
        config.pg_host,
        config.pg_port,
        config.pg_database,
    )
    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "1")
    monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    monkeypatch.delenv("JOBS_PG_SKIP_SCHEMA_INIT", raising=False)
    manager = JobManager(backend="postgres", db_url=dsn)
    rows = {}
    try:
        for owner in (11, 22):
            JobManager.set_rls_context(is_admin=False, domain_allowlist="media_ingest", owner_user_id=str(owner))
            rows[owner] = manager.create_job(
                domain="media_ingest",
                queue="default",
                job_type="media_ingest_item",
                owner_user_id=str(owner),
                payload={"source": f"private-owner-{owner}.txt", "source_kind": "file", "media_type": "document"},
            )
        JobManager.clear_rls_context()
        yield manager, rows
    finally:
        JobManager.clear_rls_context()


def component_client(manager, principal):
    """Exercise the real route dependency graph with fixed authenticated test claims."""
    app = FastAPI()
    app.include_router(ingest_jobs.router, prefix="/api/v1/media")

    async def user():
        return ingest_jobs.User(id=principal.user_id, username="scope-user", email=None, is_active=True)

    async def claims():
        return principal

    async def rate_limit():
        return None

    app.dependency_overrides[ingest_jobs.get_request_user] = user
    app.dependency_overrides[ingest_jobs.get_auth_principal] = claims
    app.dependency_overrides[ingest_jobs.get_job_manager] = lambda: manager
    app.dependency_overrides[ingest_jobs.check_rate_limit] = rate_limit
    return TestClient(app)


@pytest.mark.parametrize(
    ("actor", "roles", "admin_hint", "target", "expected"),
    [
        (11, ["user"], False, 11, 200),
        (22, ["user"], False, 22, 200),
        (11, ["user"], False, 22, 404),
        (22, ["user"], False, 11, 404),
        (33, ["admin"], False, 11, 200),
        (11, ["user"], True, 22, 404),
    ],
    ids=["alice-own", "bob-own", "alice-foreign", "bob-foreign", "claimed-admin", "untrusted-admin-hint"],
)
def test_media_ingest_status_binds_real_postgres_scope(owned_jobs, actor, roles, admin_hint, target, expected):
    manager, rows = owned_jobs
    principal = AuthPrincipal(kind="user", user_id=actor, roles=roles, permissions=["media.read"], is_admin=admin_hint)
    with component_client(manager, principal) as client:
        response = client.get(f"/api/v1/media/ingest/jobs/{rows[target]['id']}")
    assert response.status_code == expected  # nosec B101 # Real owner/admin positive and reciprocal denials.
    if expected == 200:
        body = response.json()
        assert (
            body["owner_user_id"] == str(target) and body["source"] == f"private-owner-{target}.txt"
        )  # nosec B101 # Actual returned owner/content.
    else:
        assert (
            f"private-owner-{target}.txt" not in response.text
        )  # nosec B101 # Denial must not disclose source metadata.


@pytest.mark.parametrize("actor", [11, 22])
def test_media_ingest_stream_keeps_postgres_actor_in_producer(owned_jobs, monkeypatch, actor):
    """The response task must inherit scope until its final SSE body is drained."""
    manager, rows = owned_jobs
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("JOBS_SSE_TEST_MAX_SECONDS", "0.2")
    monkeypatch.setenv("JOBS_EVENTS_POLL_INTERVAL", "0.02")
    principal = AuthPrincipal(kind="user", user_id=actor, roles=["user"], permissions=["media.read"])
    with component_client(manager, principal) as client:
        response = client.get("/api/v1/media/ingest/jobs/events/stream")
    assert response.status_code == 200  # nosec B101 # Real scoped streaming response.
    data = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    snapshot = next(item for item in data if "jobs" in item)
    assert [job["id"] for job in snapshot["jobs"]] == [rows[actor]["id"]]  # nosec B101 # Owned snapshot is usable.
    events = [item for item in data if "job_id" in item]
    assert events and all(
        item["job_id"] == rows[actor]["id"] for item in events
    )  # nosec B101 # Producer sees only its actor.
    assert f"private-owner-{33-actor}.txt" not in response.text  # nosec B101 # Foreign metadata never enters SSE.
