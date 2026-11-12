"""Workflow editor HTTP requests retain the real authenticated captured owner."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_CONTROL, WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.repos import rbac_admin_repo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import create_user
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Utils.app_services import app_services
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
ROOT = "/api/v1/workflows"
DEFINITION = {"name": "Owner-bound canvas", "version": 1, "steps": [
    {"id": "clean", "type": "text_clean", "config": {"text": "{{ inputs.text }}"}},
]}


@pytest.fixture
def private_workflow_paths(tmp_path, monkeypatch):
    """Bind the official app fixture's auxiliary stores to this test directory."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "alerts.db"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


@pytest.fixture
def workflow_accounts(private_workflow_paths, single_user_cookie_client, tmp_path, monkeypatch):
    """Reuse official SQLite AuthNZ bootstrap, real JWT auth and private content DB."""
    client, _api_key = single_user_cookie_client
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()
    app_services().jwt_service = None

    async def seed_accounts():
        accounts = {}
        for name in ("alice", "bob"):
            accounts[name] = await create_user(name, f"{name}@workflow.test", "unused-test-hash", is_verified=True)
        pool = await get_db_pool()
        async with pool.transaction() as conn:
            for permission in (WORKFLOWS_RUNS_READ, WORKFLOWS_RUNS_CONTROL):
                row = await rbac_admin_repo.ensure_permission(conn, is_postgres=False, name=permission,
                                                              description="Workflow test permission", category="workflows")
                for account in accounts.values():
                    await rbac_admin_repo.upsert_user_override(conn, is_postgres=False, user_id=account["id"],
                        permission_id=row["id"], granted=True, expires_at=None)
        return accounts

    accounts = client.portal.call(seed_accounts)
    for name, account in accounts.items():
        token = get_jwt_service().create_access_token(account["id"], name, "user")
        account["headers"] = {"Authorization": f"Bearer {token}"}
        actual = client.get(ROOT + "/auth/check", headers=account["headers"])
        assert actual.status_code == 200, actual.text  # nosec B101 - pytest contract
        assert actual.json()["user_id"] == str(account["id"])  # nosec B101
        assert actual.json()["is_admin"] is False  # nosec B101

    db = WorkflowsDatabase(str(tmp_path / "workflows.db"))
    ids = {}
    for name, account in accounts.items():
        ids[name] = db.create_definition(tenant_id="default", name=name, version=1,
            owner_id=str(account["id"]), visibility="private", description=None, tags=[],
            definition={**DEFINITION, "name": name})
    ids["other_tenant"] = db.create_definition(tenant_id="other-tenant", name="tenant-private", version=1,
        owner_id=str(accounts["bob"]["id"]), visibility="private", description=None, tags=[],
        definition={**DEFINITION, "name": "tenant-private"})
    runs = {}
    for name in ("alice", "bob", "other_tenant"):
        owner = accounts["alice"] if name == "alice" else accounts["bob"]
        runs[name] = str(uuid4())
        db.create_run(run_id=runs[name], tenant_id="other-tenant" if name == "other_tenant" else "default",
            user_id=str(owner["id"]), inputs={"text": "Private input"}, workflow_id=ids[name],
            definition_version=1, definition_snapshot=DEFINITION)
    db.create_step_run(step_run_id=f"{runs['bob']}:clean:1", tenant_id="default", run_id=runs["bob"],
        step_id="clean", name="Clean", step_type="text_clean", status="waiting_human",
        assigned_to=str(accounts["bob"]["id"]))
    reads = []

    def content_db():
        reads.append("resolve")
        return db

    previous = client.app.dependency_overrides.get(workflows._get_db)
    client.app.dependency_overrides[workflows._get_db] = content_db
    try:
        yield SimpleNamespace(client=client, db=db, accounts=accounts, ids=ids, runs=runs, reads=reads)
    finally:
        if previous is None:
            client.app.dependency_overrides.pop(workflows._get_db, None)
        else:
            client.app.dependency_overrides[workflows._get_db] = previous
        db.close()


def operation(context, name):
    """Map the editor's fifteen service operations to thirteen real route definitions."""
    wid, run_id = context.ids["bob"], context.runs["bob"]
    paths = {
        "list": ("GET", ROOT, None),
        "create": ("POST", ROOT, DEFINITION),
        "definition": ("GET", f"{ROOT}/{wid}", None),
        "version": ("POST", f"{ROOT}/{wid}/versions", {**DEFINITION, "version": 2}),
        "preflight": ("POST", ROOT + "/preflight", {"definition": DEFINITION, "validation_mode": "block"}),
        "run": ("POST", f"{ROOT}/{wid}/run?mode=async", {"inputs": {"text": "Private input"}, "idempotency_key": "scoped-attempt"}),
        "status": ("GET", f"{ROOT}/runs/{run_id}", None),
        "steps": ("GET", f"{ROOT}/runs/{run_id}/steps", None),
        "investigation": ("GET", f"{ROOT}/runs/{run_id}/investigation", None),
        "attempts": ("GET", f"{ROOT}/runs/{run_id}/steps/clean/attempts", None),
        **{action: ("POST", f"{ROOT}/runs/{run_id}/{action}", None) for action in ("pause", "resume", "cancel")},
        **{action: ("POST", f"{ROOT}/runs/{run_id}/steps/clean/{action}", {}) for action in ("approve", "reject")},
    }
    return paths[name]


@pytest.mark.parametrize("name", ["list", "create", "definition", "version", "preflight", "run", "status",
                                  "steps", "investigation", "attempts", "pause", "resume", "cancel", "approve", "reject"])
def test_changed_owner_refuses_before_workflow_content_access(workflow_accounts, name):
    context = workflow_accounts
    method, path, body = operation(context, name)
    headers = {**context.accounts["bob"]["headers"],
               "X-TLDW-Expected-User-ID": str(context.accounts["alice"]["id"])}
    response = context.client.request(method, path, headers=headers, **({"json": body} if body is not None else {}))
    assert response.status_code == 412, response.text  # nosec B101
    assert response.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101
    assert response.headers["cache-control"] == "no-store"  # nosec B101
    assert context.reads == []  # nosec B101 - no content dependency/handler was entered
    assert len(context.db.list_definitions(tenant_id="default", owner_id=str(context.accounts["bob"]["id"]))) == 1  # nosec B101
    assert context.db.get_run(context.runs["bob"]).status == "queued"  # nosec B101


@pytest.mark.parametrize("scoped", [False, True], ids=["absent-header", "matching-header"])
def test_matching_and_unscoped_owner_can_save_reopen_list_and_read_run(workflow_accounts, scoped):
    context = workflow_accounts
    headers = dict(context.accounts["bob"]["headers"])
    if scoped:
        headers["X-TLDW-Expected-User-ID"] = str(context.accounts["bob"]["id"])
    created = context.client.post(ROOT, headers=headers, json=DEFINITION)
    assert created.status_code == 201, created.text  # nosec B101
    reopened = context.client.get(f"{ROOT}/{created.json()['id']}", headers=headers)
    assert reopened.status_code == 200, reopened.text  # nosec B101
    actual_step = reopened.json()["definition"]["steps"][0]
    assert {key: actual_step[key] for key in DEFINITION["steps"][0]} == DEFINITION["steps"][0]  # nosec B101
    listed = context.client.get(ROOT, headers=headers)
    assert listed.status_code == 200, listed.text  # nosec B101
    assert {row["id"] for row in listed.json()} == {context.ids["bob"], created.json()["id"]}  # nosec B101
    status = context.client.get(f"{ROOT}/runs/{context.runs['bob']}", headers=headers)
    assert status.status_code == 200, status.text  # nosec B101
    assert status.json()["user_id"] == str(context.accounts["bob"]["id"])  # nosec B101


@pytest.mark.parametrize("foreign", ["alice", "other_tenant"], ids=["foreign-owner", "foreign-tenant"])
@pytest.mark.parametrize("operation_name", ["definition", "status", "cancel"])
def test_matching_expected_owner_retains_resource_owner_and_tenant_refusal(workflow_accounts, foreign, operation_name):
    context = workflow_accounts
    headers = {**context.accounts["bob"]["headers"], "X-TLDW-Expected-User-ID": str(context.accounts["bob"]["id"])}
    if operation_name == "definition":
        method, path = "GET", f"{ROOT}/{context.ids[foreign]}"
    else:
        method = "POST" if operation_name == "cancel" else "GET"
        path = f"{ROOT}/runs/{context.runs[foreign]}" + ("/cancel" if operation_name == "cancel" else "")
    response = context.client.request(method, path, headers=headers)
    assert response.status_code == 404, response.text  # nosec B101
    assert context.db.get_run(context.runs[foreign]).status == "queued"  # nosec B101


def test_workflow_socket_authenticates_by_token_and_ignores_the_http_expected_user_header(workflow_accounts):
    """The WebSocket resolves bob from his AuthNZ token; the HTTP-only expected-user
    assertion (here naming alice) is not applied to it."""
    context = workflow_accounts
    headers = {**context.accounts["bob"]["headers"], "X-TLDW-Expected-User-ID": str(context.accounts["alice"]["id"])}
    with context.client.websocket_connect(f"{ROOT}/ws?run_id={context.runs['bob']}", headers=headers) as socket:
        event = socket.receive_json()
        assert event["type"] == "snapshot", event  # nosec B101


@pytest.mark.parametrize("restriction", ["scope", "endpoint"])
def test_real_virtual_jwt_retains_workflow_run_restrictions(workflow_accounts, restriction):
    """Valid, unrevoked real-owner tokens still obey scope and endpoint fences."""
    context = workflow_accounts
    account = context.accounts["bob"]
    token = get_jwt_service().create_virtual_access_token(
        user_id=account["id"], username="bob", role="user",
        scope="not-workflows" if restriction == "scope" else "workflows", ttl_minutes=5,
        additional_claims={"allowed_endpoints": ["workflows.run_saved"]} if restriction == "endpoint" else None,
    )
    payload = get_jwt_service().decode_access_token(token)
    assert payload["sub"] == str(account["id"])  # nosec B101
    manager = context.client.portal.call(get_session_manager)
    assert context.client.portal.call(manager.is_token_blacklisted, token, payload["jti"]) is False  # nosec B101
    headers = {"Authorization": f"Bearer {token}", "X-TLDW-Expected-User-ID": str(account["id"])}
    requests = [(ROOT + "/run", {"definition": DEFINITION, "inputs": {}})]
    if restriction == "scope":
        requests.insert(0, (f"{ROOT}/{context.ids['bob']}/run", {"inputs": {}}))
    expected_detail = "Forbidden: invalid token scope" if restriction == "scope" else "Forbidden: endpoint not permitted for token"
    before = [(row.run_id, row.status) for row in context.db.list_runs(tenant_id="default")]
    for path, body in requests:
        response = context.client.post(path, headers=headers, json=body)
        assert response.status_code == 403, response.json()  # nosec B101
        assert response.json()["detail"] == expected_detail  # nosec B101
    assert [(row.run_id, row.status) for row in context.db.list_runs(tenant_id="default")] == before  # nosec B101
    assert len(context.db.list_definitions(tenant_id="default", owner_id=str(account["id"]))) == 1  # nosec B101
