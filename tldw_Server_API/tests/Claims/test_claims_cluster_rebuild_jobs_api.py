"""Producer routing must not provision storage or invent Jobs lifecycle state."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.endpoints import claims, jobs_admin, jobs_owned
from tldw_Server_API.app.core.Claims_Extraction import claims_jobs, claims_service
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionRejectionReason, AdmissionResult
from tldw_Server_API.app.core.Jobs.resource_contracts import ResourceAdmission
from tldw_Server_API.tests.Jobs.conftest import jobs_pg_dsn as jobs_pg_dsn
from tldw_Server_API.tests.Jobs.test_jobs_owned_status_api import context_values, principal
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import snapshot, sql

pytestmark = pytest.mark.integration
PATH = "/api/v1/claims/clusters/rebuild"


def forbidden(*args, **kwargs):
    pytest.fail("async routing invoked creating Media DB or synchronous rebuild")


def build_app(user=None, *, owned=True, admin=False):
    app = FastAPI()
    app.include_router(claims.router, prefix="/api/v1")
    if owned:
        app.include_router(jobs_owned.router, prefix="/api/v1")
    if admin:
        app.include_router(jobs_admin.router, prefix="/api/v1")
    set_user(app, user or principal())
    return app


def set_user(app, user):
    app.dependency_overrides[get_auth_principal] = lambda: user
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user.user_id)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    # The real app materializes global configuration during startup, before
    # ingress. Its base-directory bootstrap is not an owner Media DB open.
    claims_service.settings.get("CLAIMS_JOBS_QUEUE")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "false")
    monkeypatch.setenv("JOBS_PG_RLS_ENABLE", "false")
    JobManager.set_acquire_gate(False)
    return claims.get_job_manager()


def test_async_routing_does_not_open_creating_media_db(manager, tmp_path, monkeypatch):
    missing = tmp_path / "absent-owner"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(missing))
    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", forbidden)
    monkeypatch.setattr(claims_service, "rebuild_claim_clusters", forbidden)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
        assert response.status_code == 202, response.text
        body = response.json()
        assert set(body) == {"job_id", "job_uuid", "status_url", "reused"}
        assert response.headers["location"] == body["status_url"] == f"/api/v1/jobs/owned/{body['job_uuid']}"
        assert body["reused"] is False
        status = client.get(body["status_url"])
        assert status.status_code == 200, status.text
        assert status.json()["uuid"] == body["job_uuid"]
    assert not missing.exists()


def test_equal_settings_reuse_and_both_conflicts_have_bounded_identity(manager, monkeypatch):
    with TestClient(build_app()) as client:
        first = client.post(
            PATH, params={"method": "exact", "similarity_threshold": 0.1}, headers={"Idempotency-Key": "one"}
        )
        assert first.status_code == 202, first.text
        monkeypatch.setattr(manager, "get_job", forbidden)
        reused = client.post(PATH, params={"method": "exact", "similarity_threshold": 0.9})
        replay = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "one"})
        assert reused.status_code == replay.status_code == 202
        assert reused.json() == replay.json() == {**first.json(), "reused": True}
        for headers, code in [
            ({}, "claims_cluster_active_conflict"),
            ({"Idempotency-Key": "one"}, "claims_cluster_request_conflict"),
        ]:
            conflict = client.post(PATH, params={"method": "exact", "min_size": 3}, headers=headers)
            assert conflict.status_code == 409, conflict.text
            assert conflict.json() == {
                "code": code,
                "job_id": first.json()["job_id"],
                "job_uuid": first.json()["job_uuid"],
                "status_url": first.json()["status_url"],
            }


def test_terminal_key_replay_keeps_uuid_without_local_execution(manager):
    with TestClient(build_app()) as client:
        first = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "retained"})
        assert first.status_code == 202, first.text
        manager.cancel_job(first.json()["job_id"])
        replay = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "retained"})
        assert replay.status_code == 202, replay.text
        assert replay.json() == {**first.json(), "reused": True}
        assert client.get(replay.json()["status_url"]).json()["status"] == "cancelled"


@pytest.mark.parametrize("key", ["", " key", "key ", "x" * 129, "non\tprintable", "non\x7fprintable", "caf\xe9"])
def test_async_keys_are_validated_without_echo_or_log(manager, key):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        # Bytes permit non-ASCII transport so the application's validator sees it.
        with TestClient(build_app()) as client:
            response = client.post(
                PATH, params={"method": "exact"}, headers={b"Idempotency-Key": key.encode("latin-1")}
            )
        assert response.status_code == 422, response.text
        assert "job_uuid" not in response.text
        if key:
            assert key not in response.text
            assert key not in "".join(messages)
    finally:
        logger.remove(sink)
    assert manager.list_jobs(domain="claims") == []


@pytest.mark.parametrize("key", ["a", "x" * 128, "internal space !~"])
def test_valid_keys_are_digest_only(manager, key):
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    row = manager.get_job(response.json()["job_id"])
    assert row["idempotency_key"] is None
    assert set(row["payload"]) == {
        "version",
        "owner_user_id",
        "method",
        "min_size",
        "similarity_threshold",
        "algorithm_version",
    }
    if len(key) > 1:
        assert key not in json.dumps(row)


@pytest.mark.parametrize(
    "params",
    [
        {"min_size": "true"},
        {"min_size": 0},
        {"min_size": 1001},
        {"similarity_threshold": "NaN"},
        {"similarity_threshold": "Infinity"},
        {"similarity_threshold": -0.1},
        {"similarity_threshold": 1.1},
    ],
)
def test_query_bounds_remain_even_when_exact_threshold_is_ignored(manager, params):
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact", **params})
    assert response.status_code == 422
    assert manager.list_jobs(domain="claims") == []


def test_invalid_method_retains_400(manager):
    with TestClient(build_app()) as client:
        assert client.post(PATH, params={"method": "bogus"}).status_code == 400
    assert manager.list_jobs(domain="claims") == []


@pytest.mark.parametrize("owner", [None, 0, -1, True, "01", " 1", "../1"])
def test_strict_owner_before_manager_resolution(manager, monkeypatch, owner):
    app = build_app()
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=owner)
    monkeypatch.setattr(claims, "get_job_manager", forbidden, raising=False)
    with TestClient(app) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 400, response.text
    assert "job_uuid" not in response.text


@pytest.mark.parametrize(
    "user,params",
    [(principal(permissions=["claims.review"]), {}), (principal(), {"user_id": 2}), (principal(), {"user_id": 1})],
)
def test_claims_auth_precedes_manager_and_reference_lookup(manager, monkeypatch, user, params):
    monkeypatch.setattr(claims, "get_job_manager", forbidden, raising=False)
    with TestClient(build_app(user)) as client:
        response = client.post(PATH, params={"method": "exact", **params})
    assert response.status_code == 403
    assert "job_uuid" not in response.text


def test_platform_explicit_self_uses_owned_link_without_jobs_admin_guard(manager, monkeypatch):
    monkeypatch.setattr(claims, "authorize_jobs_admin_domain_read", forbidden, raising=False)
    with TestClient(build_app(principal(permissions=["system.configure"]))) as client:
        response = client.post(PATH, params={"method": "exact", "user_id": 1})
    assert response.status_code == 202, response.text
    assert response.json()["status_url"].startswith("/api/v1/jobs/owned/")


def test_foreign_owner_requires_actual_jobs_role_and_domain_guard(manager, monkeypatch):
    monkeypatch.setattr(claims, "get_job_manager", forbidden, raising=False)
    with TestClient(build_app(principal(permissions=["system.configure"]), admin=True)) as client:
        response = client.post(PATH, params={"method": "exact", "user_id": 2})
    assert response.status_code == 403, response.text
    assert "job_uuid" not in response.text


def test_foreign_domain_denial_precedes_admission(manager, monkeypatch):
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "true")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "notes")
    monkeypatch.setattr(claims, "get_job_manager", forbidden, raising=False)
    with TestClient(build_app(principal(roles=["admin"], permissions=["*"]), admin=True)) as client:
        response = client.post(PATH, params={"method": "exact", "user_id": 2})
    assert response.status_code == 403, response.text
    assert response.json() == {"detail": "Not allowed for domain claims"}
    assert manager.list_jobs(domain="claims") == []


def test_foreign_authorized_reference_pins_uuid(manager):
    with TestClient(build_app(principal(roles=["admin"], permissions=["*"]), admin=True)) as client:
        response = client.post(PATH, params={"method": "exact", "user_id": 2})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status_url"] == f"/api/v1/jobs/{body['job_id']}?domain=claims&job_uuid={body['job_uuid']}"
    assert response.headers["location"] == body["status_url"]
    assert manager.get_job(body["job_id"])["owner_user_id"] == "2"


@pytest.mark.parametrize("foreign", [False, True])
def test_required_status_route_missing_fails_before_manager(manager, monkeypatch, foreign):
    monkeypatch.setattr(claims, "get_job_manager", forbidden, raising=False)
    app = build_app(principal(roles=["admin"], permissions=["*"]), owned=foreign, admin=not foreign)
    with TestClient(app) as client:
        response = client.post(PATH, params={"method": "exact", **({"user_id": 2} if foreign else {})})
    assert response.status_code == 503, response.text
    assert "job_uuid" not in response.text


def test_async_every_attempt_including_replay_is_ingress_limited(manager, monkeypatch):
    calls = []

    async def ingress(request):
        calls.append(threading.get_ident())
        if len(calls) == 3:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "7"})

    monkeypatch.setattr(claims, "check_rate_limit", ingress)
    with TestClient(build_app()) as client:
        for _ in range(2):
            assert client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "one"}).status_code == 202
        limited = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "one"})
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "7"
    assert len(calls) == 3


@pytest.mark.parametrize("user,params", [(principal(permissions=["claims.review"]), {}), (principal(), {"user_id": 2})])
def test_async_denied_attempt_still_runs_ingress_guard(manager, monkeypatch, user, params):
    calls = []

    async def ingress(request):
        calls.append(request.url.path)

    monkeypatch.setattr(claims, "check_rate_limit", ingress)
    monkeypatch.setattr(claims, "get_job_manager", forbidden)
    with TestClient(build_app(user)) as client:
        response = client.post(PATH, params={"method": "exact", **params})
    assert response.status_code == 403
    assert calls == [PATH]


@pytest.mark.parametrize(
    "reason,status",
    [
        (AdmissionRejectionReason.QUOTA_EXCEEDED, 429),
        (AdmissionRejectionReason.QUEUE_PAUSED, 503),
        (AdmissionRejectionReason.QUEUE_DRAINING, 503),
    ],
)
def test_jobs_rejections_are_not_false_conflicts(manager, monkeypatch, reason, status):
    monkeypatch.setattr(
        manager,
        "admit_resource_job",
        lambda *a, **k: ResourceAdmission(
            "rejected", None, AdmissionResult.rejected(reason, message="private backend data")
        ),
    )
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == status, response.text
    assert "private backend data" not in response.text
    assert "job_uuid" not in response.text


@pytest.mark.parametrize("lost_ack", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_unavailable_or_lost_ack_is_sanitized_503_never_fallback(manager, monkeypatch, lost_ack, error_type):
    real = manager.admit_resource_job

    def unavailable(*args, **kwargs):
        if lost_ack:
            real(*args, **kwargs)
        raise error_type("secret key or database path")

    monkeypatch.setattr(manager, "admit_resource_job", unavailable)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 503, response.text
    assert "secret" not in response.text
    assert len(manager.list_jobs(domain="claims")) == int(lost_ack)


def test_jobs_manager_unavailable_is_sanitized(manager, monkeypatch):
    def unavailable():
        raise OSError("secret Jobs location")

    monkeypatch.setattr(claims, "get_job_manager", unavailable)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 503
    assert "secret" not in response.text


def test_unknown_admission_with_job_is_not_acceptance(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "admit_resource_job",
        lambda *a, **k: SimpleNamespace(kind="unknown", job={"id": 1, "uuid": "not-an-identity"}),
    )
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 503
    assert "not-an-identity" not in response.text


@pytest.mark.parametrize(
    "key,value",
    [
        ("min_size", True),
        ("similarity_threshold", True),
        ("similarity_threshold", float("nan")),
        ("user_id", True),
        ("request_key", " invalid "),
    ],
)
def test_public_admission_validates_strict_contract_before_jobs(manager, key, value):
    args = {
        "min_size": 2,
        "method": "embeddings",
        "similarity_threshold": 0.8,
        "user_id": None,
        "request_key": None,
        "principal": principal(roles=["admin"], permissions=["*"]),
        "current_user": SimpleNamespace(id=1),
        "job_manager": manager,
    }
    args[key] = value
    with pytest.raises(HTTPException) as error:
        claims_service.admit_claim_cluster_rebuild(**args)
    assert error.value.status_code == (400 if key == "user_id" else 422)
    assert manager.list_jobs(domain="claims") == []


@pytest.mark.parametrize("quota", ["JOBS_QUOTA_MAX_QUEUED", "JOBS_QUOTA_SUBMITS_PER_MIN"])
def test_real_jobs_quota_mapping(manager, monkeypatch, quota):
    monkeypatch.setenv(quota, "1")
    manager.create_job(domain="claims", queue="default", job_type="other", payload={}, owner_user_id="1")
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 429, response.text
    assert "job_uuid" not in response.text


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def fair_share_managers(request, manager, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    if request.param == "sqlite":
        return manager, manager

    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    dsn = request.getfixturevalue("jobs_pg_dsn")
    system = JobManager(backend="postgres", db_url=dsn)
    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(dsn)
    # Keep the official system observer separate from the restricted owner login.
    monkeypatch.delenv("JOBS_PG_RLS_ROLE")
    owner = JobManager(backend="postgres", db_url=restricted)
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
        role = sql(owner, "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user", rows=True)
        assert role == [{"rolsuper": False, "rolbypassrls": False}]
    monkeypatch.setattr(claims, "get_job_manager", lambda: owner)
    return owner, system


def test_real_fair_share_http_refusal_is_quota_without_mutation(fair_share_managers, monkeypatch):
    _, system = fair_share_managers
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    system.create_job(domain="other-domain", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    before = snapshot(system)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 429, response.text
    assert response.json() == {"detail": "Jobs admission quota exceeded"}
    assert snapshot(system) == before


def test_real_fair_share_http_replay_and_reuse_precede_limit(fair_share_managers, monkeypatch):
    _, system = fair_share_managers
    monkeypatch.setenv("JOBS_MAX_PER_USER", "2")
    system.create_job(domain="other-domain", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    with TestClient(build_app()) as client:
        first = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "fair-share"})
        assert first.status_code == 202, first.text
        assert client.get(first.json()["status_url"]).status_code == 200
        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        before = snapshot(system)
        for headers in ({}, {"Idempotency-Key": "fair-share"}):
            reused = client.post(PATH, params={"method": "exact"}, headers=headers)
            assert reused.status_code == 202, reused.text
            assert reused.json() == {**first.json(), "reused": True}
        assert snapshot(system) == before
        assert system.cancel_job(first.json()["job_id"])
        terminal = snapshot(system)
        replay = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "fair-share"})
        assert replay.status_code == 202, replay.text
        assert replay.json() == {**first.json(), "reused": True}
        assert snapshot(system) == terminal


def test_type_registration_unavailable_fails_before_create(manager, monkeypatch):
    monkeypatch.setattr(claims_service, "resource_requirement_for", lambda *a: None, raising=False)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 503, response.text
    assert manager.list_jobs(domain="claims") == []


@pytest.mark.parametrize("minimal", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_status_gate_observes_actual_operator_router_registration(manager, monkeypatch, minimal, foreign, disabled):
    from configparser import ConfigParser

    from tldw_Server_API.app.api.v1.router_groups.admin import iter_admin_router_specs
    from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs
    from tldw_Server_API.app.api.v1.router_registry import register_router_specs
    from tldw_Server_API.app.core import config

    monkeypatch.setattr(config, "is_test_mode", lambda: False)
    monkeypatch.setattr(config, "is_explicit_pytest_runtime", lambda: False)
    route_key = "jobs" if foreign else "jobs-owned"
    policy = ConfigParser()
    policy.read_dict(
        {
            "API-Routes": {
                "stable_only": "true",
                "enable": "" if disabled else route_key,
                "disable": route_key if disabled else "",
            }
        }
    )
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: policy)
    config._route_toggle_policy.cache_clear()
    try:
        app = build_app(principal(roles=["admin"], permissions=["*"]), owned=False)
        specs = iter_minimal_optional_router_specs() if minimal else iter_admin_router_specs()
        register_router_specs(app, [spec for spec in specs if spec.name == ("jobs_admin" if foreign else "jobs_owned")])
        # Minimal's existing admin spec is unconditional.
        unavailable = disabled and not (minimal and foreign)
        with TestClient(app) as client:
            response = client.post(PATH, params={"method": "exact", **({"user_id": 2} if foreign else {})})
        assert response.status_code == (503 if unavailable else 202), response.text
        if unavailable:
            assert response.json() == {"detail": "Claims cluster rebuild status unavailable"}
            assert manager.list_jobs(domain="claims") == []
        else:
            body = response.json()
            stored = manager.get_job(body["job_id"])
            assert stored["uuid"] == body["job_uuid"]
            assert stored["owner_user_id"] == ("2" if foreign else "1")
            assert response.headers["location"] == body["status_url"]
    finally:
        config._route_toggle_policy.cache_clear()


def test_async_flag_snapshot_is_not_rechecked_by_admission(manager, monkeypatch):
    def captured():
        monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "false")
        return True

    monkeypatch.setattr(DB_Deps, "claims_cluster_rebuild_jobs_enabled", captured, raising=False)
    with TestClient(build_app()) as client:
        response = client.post(PATH, params={"method": "exact"})
    assert response.status_code == 202, response.text
    stored = manager.get_job(response.json()["job_id"])
    assert stored["uuid"] == response.json()["job_uuid"]
    assert stored["status"] == "queued"
    assert stored["payload"]["method"] == "exact"


async def test_route_context_is_frozen_and_async_dependency_is_never_opened(manager, monkeypatch):
    from dataclasses import FrozenInstanceError

    from starlette.requests import Request

    monkeypatch.setattr(DB_Deps, "get_media_db_for_user", forbidden)
    generator = DB_Deps.get_claim_cluster_rebuild_context(Request({"type": "http"}), SimpleNamespace(id=None))
    context = await generator.__anext__()
    try:
        assert context.async_enabled is True and context.db is None
        with pytest.raises(FrozenInstanceError):
            context.async_enabled = False
    finally:
        await generator.aclose()


@pytest.fixture
def legacy_cluster_session(tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.session import MediaDbSession
    from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed

    db = MediaDatabase(db_path=tmp_path / "media.db", client_id="1")
    try:
        seed(db, owner=1, text="One fact")
        seed(db, owner=1, text="one FACT")
        yield db, MediaDbSession(db_path=str(tmp_path / "media.db"), client_id="1", database=db)
    finally:
        db.close_connection()


@pytest.mark.parametrize("service_error", [False, True])
def test_disabled_sync_snapshot_preserves_cleanup_and_ignores_key(
    manager, legacy_cluster_session, monkeypatch, service_error
):
    db, session = legacy_cluster_session
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "false")

    def resolve(user):
        monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
        # Give the real session an owned connection so teardown has an
        # observable resource to return on success and service rejection.
        db._set_persistent_conn(db.backend.get_pool().get_connection())
        return session

    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", resolve)
    with TestClient(build_app(owned=False)) as client:
        response = client.post(
            PATH,
            params={"method": "bogus" if service_error else "exact", "min_size": 1},
            headers={"Idempotency-Key": " invalid "},
        )
    assert response.status_code == (400 if service_error else 200), response.text
    assert db._get_persistent_conn() is None
    assert manager.list_jobs(domain="claims") == []
    if service_error:
        assert response.json() == {"detail": "Invalid clustering method"}
        assert db.execute_query("SELECT * FROM claim_clusters").fetchall() == []
    else:
        assert response.json()["clusters_created"] == 1
        assert response.json()["claims_assigned"] == 2
        assert response.json()["watchlist_notifications"]["status"] == "skipped"
        assert len(db.execute_query("SELECT * FROM claim_cluster_membership").fetchall()) == 2
    assert "location" not in response.headers


@pytest.mark.parametrize("service_error", [False, True])
def test_disabled_sync_service_runs_off_request_thread(manager, legacy_cluster_session, monkeypatch, service_error):
    db, session = legacy_cluster_session
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "false")
    resolver_threads, rebuild_threads = [], []
    real_rebuild = claims_service.rebuild_claim_clusters

    def resolve(user):
        resolver_threads.append(threading.get_ident())
        return session

    def observe_rebuild(**kwargs):
        rebuild_threads.append(threading.get_ident())
        return real_rebuild(**kwargs)

    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", resolve)
    monkeypatch.setattr(claims_service, "rebuild_claim_clusters", observe_rebuild)
    with TestClient(build_app(owned=False)) as client:
        response = client.post(PATH, params={"method": "bogus" if service_error else "exact", "min_size": 1})
    assert response.status_code == (400 if service_error else 200), response.text
    assert len(resolver_threads) == len(rebuild_threads) == 1
    assert resolver_threads[0] != rebuild_threads[0]
    if service_error:
        assert response.json() == {"detail": "Invalid clustering method"}
        assert db.execute_query("SELECT * FROM claim_clusters").fetchall() == []
    else:
        assert response.json()["claims_assigned"] == 2
        assert len(db.execute_query("SELECT * FROM claim_cluster_membership").fetchall()) == 2


@pytest.mark.parametrize("async_guard", ["get_job_manager", "check_rate_limit"])
def test_disabled_sync_bypasses_async_ingress_controls(manager, legacy_cluster_session, monkeypatch, async_guard):
    db, session = legacy_cluster_session
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "false")

    def reject_async_control(*args, **kwargs):
        raise AssertionError(f"legacy route invoked async-only {async_guard}")

    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", lambda user: session)
    monkeypatch.setattr(claims, async_guard, reject_async_control)
    with TestClient(build_app(owned=False)) as client:
        response = client.post(
            PATH, params={"method": "exact", "min_size": 1}, headers={"Idempotency-Key": " invalid "}
        )
    assert response.status_code == 200, response.text
    assert response.json()["clusters_created"] == 1
    assert response.json()["claims_assigned"] == 2
    assert len(db.execute_query("SELECT * FROM claim_cluster_membership").fetchall()) == 2
    assert manager.list_jobs(domain="claims") == []


async def test_sync_dependency_delegates_and_finalizes_on_cancellation(manager, monkeypatch):
    from starlette.requests import Request

    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "false")
    events = []
    session = object()

    async def delegated(request, current_user):
        events.append("opened")
        try:
            yield session
        finally:
            events.append("closed")

    monkeypatch.setattr(DB_Deps, "get_media_db_for_user", delegated)
    generator = DB_Deps.get_claim_cluster_rebuild_context(Request({"type": "http"}), SimpleNamespace(id=1))
    context = await generator.__anext__()
    assert context.async_enabled is False and context.db is session
    with pytest.raises(asyncio.CancelledError):
        await generator.athrow(asyncio.CancelledError())
    assert events == ["opened", "closed"]


def test_openapi_documents_legacy_and_typed_async_responses(manager):
    operation = build_app().openapi()["paths"][PATH]["post"]
    assert {"200", "202", "403", "409", "503"} <= set(operation["responses"])
    assert operation["responses"]["202"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ClaimsClusterRebuildAccepted"
    )
    assert operation["responses"]["409"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ClaimsClusterRebuildConflict"
    )
    assert operation["responses"]["202"]["headers"]["Location"]["schema"]["type"] == "string"
    header = next(p for p in operation["parameters"] if p["name"] == "Idempotency-Key")
    assert header["in"] == "header" and header["required"] is False
    assert header["schema"] == {"type": "string"}


@pytest.mark.pg_jobs
def test_async_admission_restores_jobs_rls_context_across_threadpool(jobs_pg_dsn, monkeypatch):
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
    manager = JobManager(backend="postgres", db_url=restricted)
    with manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
        assert sql(
            manager,
            "SELECT current_user AS role,rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
            rows=True,
        ) == [{"role": "jobs_rls", "rolsuper": False, "rolbypassrls": False}]
    real_enqueue = claims_jobs.enqueue_claims_cluster_rebuild
    observed, event_threads = [], []

    def observed_enqueue(**kwargs):
        before = context_values()
        try:
            return real_enqueue(**kwargs)
        finally:
            observed.append((threading.get_ident(), before, context_values()))

    monkeypatch.setattr(claims_jobs, "enqueue_claims_cluster_rebuild", observed_enqueue)
    app = build_app()

    @app.middleware("http")
    async def unset_context(request, call_next):
        event_threads.append(threading.get_ident())
        assert context_values() == (False, None, None)
        response = await call_next(request)
        assert context_values() == (False, None, None)
        return response

    with TestClient(app) as client:
        first = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "a"})
        assert first.status_code == 202, first.text
        assert client.get(first.json()["status_url"]).status_code == 200
        replay = client.post(PATH, params={"method": "exact"}, headers={"Idempotency-Key": "a"})
        assert replay.status_code == 202 and replay.json()["job_uuid"] == first.json()["job_uuid"]
        set_user(app, principal(2))
        assert client.get(first.json()["status_url"]).status_code == 404
        own_b = client.post(PATH, params={"method": "exact"})
        assert own_b.status_code == 202, own_b.text
        status = client.get(own_b.json()["status_url"])
        assert status.status_code == 200, status.text
        assert status.json()["uuid"] == own_b.json()["job_uuid"]
        assert status.json()["status"] == "queued"
    with manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
        assert manager.get_job(first.json()["job_id"])["owner_user_id"] == "1"
        assert manager.get_job(own_b.json()["job_id"]) is None
    with manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="2"):
        assert manager.get_job(own_b.json()["job_id"])["owner_user_id"] == "2"
        assert manager.get_job(first.json()["job_id"]) is None
    assert len(observed) == 3
    assert all(
        worker not in event_threads and before == after == (False, None, None) for worker, before, after in observed
    )

    # A genuine Jobs-admin authorization authorizes the link, not an ambient
    # database privilege: actual admission still scopes the target owner B.
    foreign_app = build_app(principal(roles=["admin"], permissions=["*"]), admin=True)
    with TestClient(foreign_app) as client:
        foreign = client.post(PATH, params={"method": "exact", "user_id": 2})
        assert foreign.status_code == 202, foreign.text
        assert foreign.json()["job_uuid"] == own_b.json()["job_uuid"]
    assert observed[-1][1:] == ((False, None, None), (False, None, None))
