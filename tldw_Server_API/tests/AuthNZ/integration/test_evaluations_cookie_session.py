"""Evaluations accepts canonical cookie owners without weakening its guards."""

import os
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth, evaluations_datasets, evaluations_recipes
from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_READ
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import _mint
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
RECIPES = "/api/v1/evaluations/recipes"
DATASETS = "/api/v1/evaluations/datasets"
DATASET_BODY = {"name": "Cookie-owned dataset", "samples": [{"input": "hello", "expected": "hi"}]}


@pytest_asyncio.fixture
async def evaluation_cookie_client(single_user_cookie_client, monkeypatch, tmp_path):
    """Keep real sessions, auth, CSRF, handlers and per-user content SQLite."""
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "evaluations-users"))
    monkeypatch.delenv("EVALUATIONS_TEST_DB_PATH", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "shutdown-jobs.db"))
    for router, path in [(evaluations_recipes.recipes_router, RECIPES),
                         (evaluations_datasets.datasets_router, DATASETS)]:
        if not any(route.path == path for route in iter_served_routes(client.app.routes)):
            client.app.include_router(router, prefix="/api/v1/evaluations")
    flags = {name: os.environ.get(name) for name in ("TEST_MODE", "TLDW_TEST_MODE")}
    for name in flags:
        monkeypatch.setenv(name, "false")
    monkeypatch.setenv("TESTING", "false")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    services = OrderedDict()
    monkeypatch.setattr(unified_evaluation_service, "_service_instances_by_user", services)
    try:
        yield client, api_key
    finally:
        for service in services.values():
            client.portal.call(service.initialize)
            client.portal.call(service.shutdown)
        client.portal.call(reset_api_key_manager)
        # Match the enclosing minimal-app fixture's startup mode at shutdown.
        for name, value in flags.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)


def _authenticate(client, api_key):
    assert _mint(client, api_key).status_code == 200  # nosec B101 - pytest real mint assertion.
    return {"X-CSRF-Token": client.cookies["csrf_token"]}


@pytest.mark.parametrize("path", [RECIPES, DATASETS])
@pytest.mark.parametrize("suffix", ["", "/"])
def test_evaluation_cookie_session_lists_real_catalogs(evaluation_cookie_client, path, suffix):
    client, api_key = evaluation_cookie_client
    _authenticate(client, api_key)

    response = client.get(path + suffix)

    assert response.status_code == 200, response.text  # nosec B101 - pytest authenticated route assertion.
    if path == RECIPES:
        assert "summarization_quality" in {item["recipe_id"] for item in response.json()}  # nosec B101 - pytest real registry assertion.
    else:
        assert response.json()["data"] == []  # nosec B101 - pytest isolated dataset assertion.


@pytest.mark.parametrize("path", [RECIPES, DATASETS])
@pytest.mark.parametrize("state", ["absent", "tampered", "expired", "revoked", "wrong-device"])
def test_evaluation_rejects_unusable_cookie(evaluation_cookie_client, monkeypatch, path, state):
    client, api_key = evaluation_cookie_client
    if state in {"expired", "wrong-device"}:
        original_create = SessionManager.create_session

        async def create_unusable(self, *args, **kwargs):
            if state == "expired":
                kwargs["expires_at_override"] = datetime.now(timezone.utc) - timedelta(days=1)
            else:
                kwargs["device_id"] = "ordinary-device"
            return await original_create(self, *args, **kwargs)

        monkeypatch.setattr(SessionManager, "create_session", create_unusable)
    if state in {"expired", "revoked", "wrong-device"}:
        headers = _authenticate(client, api_key)
    cookie_name = get_settings().SINGLE_USER_SESSION_COOKIE_NAME
    if state == "revoked":
        token = client.cookies[cookie_name]
        assert client.delete("/api/v1/auth/single-user/session", headers=headers).status_code == 200  # nosec B101 - pytest real revocation assertion.
        client.cookies.set(cookie_name, token, path="/api")
    elif state == "tampered":
        client.cookies.set(cookie_name, "invalid-cookie", path="/api")

    response = client.get(path)

    assert response.status_code == 401, response.text  # nosec B101 - pytest invalid session assertion.


@pytest.mark.parametrize("path", [RECIPES, DATASETS])
@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer invalid-key"}, {"X-API-KEY": "invalid-key"},
    {"Authorization": ""}, {"X-API-KEY": ""}, {"Authorization": "Basic invalid"},
])
def test_evaluation_cookie_never_overrides_explicit_headers(evaluation_cookie_client, path, headers):
    client, api_key = evaluation_cookie_client
    _authenticate(client, api_key)

    response = client.get(path, headers=headers)

    assert response.status_code == 401, response.text  # nosec B101 - pytest header precedence assertion.


@pytest.mark.parametrize("path", [RECIPES, DATASETS])
def test_evaluation_cookie_retains_ip_allowlist(evaluation_cookie_client, monkeypatch, path):
    client, api_key = evaluation_cookie_client
    _authenticate(client, api_key)
    monkeypatch.setenv("SINGLE_USER_ALLOWED_IPS", "203.0.113.10")
    reset_settings()

    response = client.get(path)

    assert response.status_code == 401, response.text  # nosec B101 - pytest denied IP assertion.
    assert response.json()["detail"]["error"]["code"] == "ip_not_allowed"  # nosec B101 - pytest retained IP guard assertion.


@pytest.mark.parametrize("header", ["X-API-KEY", "Authorization"])
def test_evaluation_retains_primary_key_and_bearer_without_cookie(evaluation_cookie_client, header):
    client, api_key = evaluation_cookie_client
    value = f"Bearer {api_key}" if header == "Authorization" else api_key

    for path in (RECIPES, DATASETS):
        response = client.get(path, headers={header: value})
        assert response.status_code == 200, response.text  # nosec B101 - pytest explicit authentication assertion.


def test_evaluation_cookie_dataset_write_retains_csrf_and_owner(evaluation_cookie_client):
    client, api_key = evaluation_cookie_client
    headers = _authenticate(client, api_key)
    owner = str(get_settings().SINGLE_USER_FIXED_ID)

    denied = client.post(DATASETS, json=DATASET_BODY)
    assert denied.status_code == 403, denied.text  # nosec B101 - pytest CSRF denial assertion.
    created = client.post(DATASETS, json=DATASET_BODY, headers=headers)
    assert created.status_code == 201, created.text  # nosec B101 - pytest authorized dataset write assertion.
    owned_id = created.json()["id"]
    assert created.json()["created_by"] == owner  # nosec B101 - pytest canonical owner assertion.
    service = unified_evaluation_service.get_unified_evaluation_service_for_user(owner)
    foreign_id = service.db.create_dataset(name="Foreign", samples=[{"input": "other"}], created_by="foreign-owner")
    response = client.get(DATASETS)
    assert response.status_code == 200, response.text  # nosec B101 - pytest owner list assertion.
    assert [item["id"] for item in response.json()["data"]] == [owned_id]  # nosec B101 - pytest foreign-row exclusion assertion.
    assert client.get(f"{DATASETS}/{foreign_id}").status_code == 404  # nosec B101 - pytest foreign-detail denial assertion.


@pytest.mark.parametrize("path", [RECIPES, DATASETS])
@pytest.mark.parametrize("can_read", [False, True])
def test_evaluation_cookie_retains_read_permission_gate(evaluation_cookie_client, monkeypatch, path, can_read):
    client, api_key = evaluation_cookie_client
    _authenticate(client, api_key)
    user = User_DB_Handling.User(id=get_settings().SINGLE_USER_FIXED_ID, username="scoped-owner",
                                roles=["user"], permissions=[EVALS_READ] if can_read else [], is_admin=True)
    # Keep real session validation; vary verified User claims to exercise RBAC.
    monkeypatch.setattr(User_DB_Handling, "get_single_user_instance", lambda: user)

    response = client.get(path)

    assert response.status_code == (200 if can_read else 403), response.text  # nosec B101 - pytest claim-based permission assertion.


def test_evaluation_cookie_read_permission_does_not_allow_manage(evaluation_cookie_client, monkeypatch):
    client, api_key = evaluation_cookie_client
    headers = _authenticate(client, api_key)
    user = User_DB_Handling.User(id=get_settings().SINGLE_USER_FIXED_ID, username="reader",
                                roles=["user"], permissions=[EVALS_READ], is_admin=True)
    monkeypatch.setattr(User_DB_Handling, "get_single_user_instance", lambda: user)

    response = client.post(DATASETS, json=DATASET_BODY, headers=headers)

    assert response.status_code == 403, response.text  # nosec B101 - pytest distinct manage permission assertion.


def test_evaluation_other_cookie_name_is_not_a_candidate(evaluation_cookie_client):
    client, _api_key = evaluation_cookie_client
    client.cookies.set("unrelated_cookie", "opaque", path="/api")

    assert client.get(RECIPES).status_code == 401  # nosec B101 - pytest configured cookie-name assertion.


def test_evaluation_cookie_is_not_admitted_in_multi_user_mode(evaluation_cookie_client, monkeypatch):
    client, api_key = evaluation_cookie_client
    _authenticate(client, api_key)
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()

    assert client.get(RECIPES).status_code == 401  # nosec B101 - pytest auth-mode boundary assertion.


@pytest.mark.asyncio
async def test_evaluation_cookie_canonical_failure_stays_closed(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Canonical authentication unavailable")

    monkeypatch.setattr(evaluations_auth, "get_request_user", unavailable)
    monkeypatch.setenv("TESTING", "false")
    settings = get_settings()
    request = Request({"type": "http", "client": ("127.0.0.1", 1234),
                       "headers": [(b"cookie", f"{settings.SINGLE_USER_SESSION_COOKIE_NAME}=opaque".encode())]})

    with pytest.raises(HTTPException) as rejected:
        await evaluations_auth.verify_api_key(credentials=None, x_api_key=None, request=request)
    assert rejected.value.status_code == 503  # nosec B101 - pytest canonical failure propagation assertion.
