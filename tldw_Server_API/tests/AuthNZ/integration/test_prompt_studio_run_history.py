"""Read durable Prompt Studio snapshots with real authentication and transport."""

from types import SimpleNamespace

import httpx
import pytest

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service, reset_jwt_service
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import create_user
from tldw_Server_API.app.core.LLM_Calls.providers.local_adapters import OllamaAdapter
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
ROOT = "/api/v1/prompt-studio"


@pytest.fixture
def private_history_paths(tmp_path, monkeypatch):
    """Keep official app stores private through their complete lifespan teardown."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(tmp_path / "alerts.db"))
    monkeypatch.setenv("SYSTEM_LOG_FILE_PATH", str(tmp_path / "system.log"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))


@pytest.fixture
def history_accounts(private_history_paths, single_user_cookie_client, monkeypatch):
    """Official AuthNZ bootstrap, genuine non-admin users, and production routers."""
    client, _api_key = single_user_cookie_client
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.setenv("TLDW_TEST_MODE", "false")
    monkeypatch.setenv("BYOK_ENABLED", "false")
    reset_settings()
    reset_jwt_service()

    async def accounts():
        return {name: await create_user(name, f"{name}@history.test", "unused-test-hash", is_verified=True)
                for name in ("alice", "bob")}

    owners = client.portal.call(accounts)
    for name, owner in owners.items():
        owner["headers"] = {"Authorization": "Bearer " + get_jwt_service().create_access_token(owner["id"], name, "user")}

    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
        prompt_studio_evaluations,
        prompt_studio_projects,
        prompt_studio_prompts,
        prompt_studio_test_cases,
    )
    served = {route.path for route in iter_served_routes(client.app.routes)}
    for path, router in ((ROOT + "/projects/", prompt_studio_projects.router),
                         (ROOT + "/prompts/create", prompt_studio_prompts.router),
                         (ROOT + "/test-cases/create", prompt_studio_test_cases.router),
                         (ROOT + "/evaluations/{evaluation_id}", prompt_studio_evaluations.router)):
        if path not in served:
            client.app.include_router(router)

    owner = owners["alice"]
    context = {"user_id": str(owner["id"]), "client_id": "history-test", "is_admin": False}
    with prompt_studio_deps.managed_prompt_studio_db(context) as db:
        project = db.create_project(name="Rowan", user_id=str(owner["id"]))
        prompt = db.create_prompt(project["id"], "Rowan", system_prompt="Give the full name.", user_prompt="{{name}}")
        case = db.create_test_case(project["id"], "Name", inputs={"name": "Mira"}, expected_outputs={"response": "Mira Vale"})
        foreign = db.create_project(name="Bob private", user_id=str(owners["bob"]["id"]))
        foreign_prompt = db.create_prompt(foreign["id"], "Secret", user_prompt="Private")
        foreign_case = db.create_test_case(foreign["id"], "Private", inputs={})
        foreign_run = db.create_test_run(project_id=foreign["id"], prompt_id=foreign_prompt["id"],
                                         test_case_id=foreign_case["id"], model_name="private", inputs={}, outputs={})
        yield SimpleNamespace(client=client, owners=owners, headers=owner["headers"], db=db,
                              project=project, prompt=prompt, case=case, foreign=foreign,
                              foreign_run=foreign_run)
    client.portal.call(prompt_studio_deps.shutdown_prompt_studio_deps)


def saved_run(context, **changes):
    """Use the existing create-only store, including honest older failed records."""
    fields = {"project_id": context.project["id"], "prompt_id": context.prompt["id"],
              "test_case_id": context.case["id"], "model_name": "qwen2.5:0.5b",
              "model_params": {"provider": "ollama", "temperature": 0.7, "max_tokens": 1000},
              "inputs": {"name": "Mira"}, "outputs": {"response": "Mira Vale"},
              "expected_outputs": {"response": "Mira Vale"}}
    return context.db.create_test_run(**{**fields, **changes})


def test_real_run_then_revision_reads_original_snapshots_without_provider_replay(history_accounts, monkeypatch):
    context = history_accounts
    calls = []

    def transport(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "choices": [{"message": {"content": "Mira Vale"}}], "usage": {"total_tokens": 7}})

    monkeypatch.setattr(OllamaAdapter, "http_fetcher", staticmethod(transport))
    result = context.client.post(ROOT + "/test-cases/run", headers=context.headers,
        json={"prompt_id": context.prompt["id"], "test_case_ids": [context.case["id"]],
              "provider": "ollama", "model": "qwen2.5:0.5b"})
    assert result.status_code == 200, result.text  # nosec B101 - pytest contract
    run_id = result.json()["results"][0]["id"]
    assert len(calls) == 1  # nosec B101 - pytest contract
    revision = context.db.create_prompt_version(context.prompt["id"], change_description="Full sentence",
                                               user_prompt="Full sentence for {{name}}")
    context.db.update_test_case(context.case["id"], {"inputs": {"name": "Changed"}, "expected_outputs": {"response": "Changed"}})
    detail = context.client.get(f"{ROOT}/test-cases/runs/{run_id}", headers=context.headers)
    assert detail.status_code == 200, detail.text  # nosec B101 - pytest contract
    row = detail.json()["data"]
    assert row["prompt_id"] == context.prompt["id"] != revision["id"]  # nosec B101 - pytest contract
    assert row["prompt_version_number"] == 1  # nosec B101 - pytest contract
    assert row["prompt_system_prompt"] == "Give the full name."  # nosec B101 - pytest contract
    assert row["prompt_user_prompt"] == "{{name}}"  # nosec B101 - pytest contract
    assert row["inputs"] == {"name": "Mira"}  # nosec B101 - pytest contract
    assert row["outputs"] == {"response": "Mira Vale"}  # nosec B101 - pytest contract
    assert row["expected_outputs"] == {"response": "Mira Vale"}  # nosec B101 - pytest contract
    assert row["model_params"] == {"provider": "ollama", "temperature": 0.7, "max_tokens": 1000}  # nosec B101 - pytest contract
    page = context.client.get(ROOT + "/test-cases/runs", params={"project_id": context.project["id"]}, headers=context.headers)
    assert page.status_code == 200, page.text  # nosec B101 - pytest contract
    assert page.json()["data"][0]["id"] == run_id  # nosec B101 - pytest contract
    assert len(calls) == 1   # nosec B101 - pytest contract; Open/list must never generate again.


def test_history_pages_have_stable_id_tiebreaker_and_failed_nullable_fields(history_accounts):
    context = history_accounts
    rows = [saved_run(context, model_params=None, expected_outputs=None, outputs={}, error_message="Provider unavailable")
            for _ in range(3)]
    actual = []
    for page in (1, 2, 3):
        response = context.client.get(ROOT + "/test-cases/runs", headers=context.headers,
            params={"project_id": context.project["id"], "page": page, "per_page": 1})
        assert response.status_code == 200, response.text  # nosec B101 - pytest contract
        data = response.json()
        actual.append(data["data"][0]["id"])
        assert data["data"][0]["error_message"] == "Provider unavailable"  # nosec B101 - pytest contract
        assert data["data"][0]["model_params"] is None  # nosec B101 - pytest contract
        assert data["data"][0]["expected_outputs"] is None  # nosec B101 - pytest contract
        assert data["pagination"]["total"] == 3  # nosec B101 - pytest contract
    assert actual == [row["id"] for row in reversed(rows)]  # nosec B101 - pytest contract


@pytest.mark.parametrize("per_page", [0, 101])
def test_history_rejects_out_of_bounds_page_sizes(history_accounts, per_page):
    context = history_accounts
    response = context.client.get(ROOT + "/test-cases/runs", headers=context.headers,
        params={"project_id": context.project["id"], "per_page": per_page})
    assert response.status_code == 422, response.text  # nosec B101 - pytest contract


@pytest.mark.parametrize("operation", ["list", "detail", "evaluation"])
def test_history_authorizes_stored_project_not_caller_supplied_identity(history_accounts, operation):
    context = history_accounts
    evaluation = context.db.create_evaluation(project_id=context.foreign["id"],
                                              prompt_id=context.foreign_run["prompt_id"])
    path = {"list": f"{ROOT}/test-cases/runs?project_id={context.foreign['id']}",
            "detail": f"{ROOT}/test-cases/runs/{context.foreign_run['id']}?project_id={context.project['id']}",
            "evaluation": f"{ROOT}/evaluations/{evaluation['id']}"}[operation]
    response = context.client.get(path, headers=context.headers)
    assert response.status_code == 403, response.text  # nosec B101 - pytest contract
    assert "Private" not in response.text  # nosec B101 - pytest contract


@pytest.mark.parametrize("operation", ["list", "detail", "evaluation"])
def test_captured_principal_change_refuses_history_reads(history_accounts, operation):
    context = history_accounts
    row = saved_run(context)
    evaluation = context.db.create_evaluation(project_id=context.project["id"], prompt_id=context.prompt["id"])
    path = {"list": f"{ROOT}/test-cases/runs?project_id={context.project['id']}",
            "detail": f"{ROOT}/test-cases/runs/{row['id']}",
            "evaluation": f"{ROOT}/evaluations/{evaluation['id']}"}[operation]
    response = context.client.get(path, headers={**context.headers,
        "X-TLDW-Expected-User-ID": str(context.owners["bob"]["id"])})
    assert response.status_code == 412, response.text  # nosec B101 - pytest contract
    assert response.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101 - pytest contract
    assert response.headers["cache-control"] == "no-store"  # nosec B101 - pytest contract


@pytest.mark.parametrize("shape", ["dict", "list", "missing"])
def test_evaluation_reads_stored_ids_error_and_safe_historical_configuration(history_accounts, shape):
    context = history_accounts
    row = saved_run(context)
    model = {"provider": "ollama", "model": "qwen2.5:0.5b", "temperature": 0.2,
             "max_tokens": 32, "api_key": "not-public", "unknown_secret": "not-public"}  # nosec B105 - synthetic scrub marker, not a credential
    configs = model if shape == "dict" else [model] if shape == "list" else None
    evaluation = context.db.create_evaluation(project_id=context.project["id"], prompt_id=context.prompt["id"],
        model_configs=configs, test_case_ids=[context.case["id"]])
    context.db.update_evaluation(evaluation["id"], {"test_run_ids": [row["id"]], "status": "failed",
        "aggregate_metrics": {"accuracy": 0.25}, "error_message": "Evaluation failed"})
    response = context.client.get(f"{ROOT}/evaluations/{evaluation['id']}", headers=context.headers)
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    data = response.json()
    assert data["test_case_ids"] == [context.case["id"]]  # nosec B101 - pytest contract
    assert data["test_run_ids"] == [row["id"]]  # nosec B101 - pytest contract
    assert data["aggregate_metrics"] == data["metrics"] == {"accuracy": 0.25}  # nosec B101 - pytest contract
    assert data["error_message"] == "Evaluation failed"  # nosec B101 - pytest contract
    assert data["model_configs"] == (None if shape == "missing" else [{k: v for k, v in model.items() if k not in {"api_key", "unknown_secret"}}])  # nosec B101 - pytest contract
    assert "not-public" not in response.text  # nosec B101 - pytest contract
    assert data["config"] == ({} if shape == "missing" else data["model_configs"][0])  # nosec B101 - pytest contract


def test_missing_run_is_not_fabricated_and_unsafe_run_params_are_not_projected(history_accounts):
    context = history_accounts
    missing = context.client.get(ROOT + "/test-cases/runs/999999", headers=context.headers)
    assert missing.status_code == 404  # nosec B101 - pytest contract
    row = saved_run(context, model_params={"provider": "ollama", "temperature": 0.3,
        "max_tokens": 12, "api_key": "not-public", "arbitrary": {"secret": "not-public"}})  # nosec B105 - synthetic scrub marker, not a credential
    response = context.client.get(f"{ROOT}/test-cases/runs/{row['id']}", headers=context.headers)
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    assert response.json()["data"]["model_params"] == {"provider": "ollama", "temperature": 0.3, "max_tokens": 12}  # nosec B101 - pytest contract
    assert "not-public" not in response.text  # nosec B101 - pytest contract


@pytest.mark.parametrize("scoped", [False, True], ids=["absent-header", "matching-header"])
def test_saved_history_read_compatible_with_optional_matching_owner_header(history_accounts, scoped):
    context = history_accounts
    row = saved_run(context)
    headers = dict(context.headers)
    if scoped:
        headers["X-TLDW-Expected-User-ID"] = str(context.owners["alice"]["id"])
    response = context.client.get(f"{ROOT}/test-cases/runs/{row['id']}", headers=headers)
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    assert response.json()["data"]["id"] == row["id"]  # nosec B101 - pytest contract


@pytest.mark.parametrize("path", ["/test-cases/runs?project_id=1", "/test-cases/runs/1", "/evaluations/1"])
def test_history_requires_real_authentication(history_accounts, path):
    response = history_accounts.client.get(ROOT + path)
    assert response.status_code == 401, response.text  # nosec B101 - pytest contract


@pytest.mark.parametrize("configs", ["malformed-json", ["not-a-model"], 12])
def test_malformed_historical_model_configuration_does_not_fabricate_defaults(history_accounts, configs):
    context = history_accounts
    evaluation = context.db.create_evaluation(project_id=context.project["id"], prompt_id=context.prompt["id"],
        model_configs=configs)
    response = context.client.get(f"{ROOT}/evaluations/{evaluation['id']}", headers=context.headers)
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    assert response.json()["model_configs"] is None  # nosec B101 - pytest contract
    assert response.json()["config"] == {}  # nosec B101 - pytest contract


def test_changed_authenticated_owner_with_colliding_run_ids_cannot_open_original_template(history_accounts):
    context = history_accounts
    original = saved_run(context)
    bob = context.owners["bob"]
    with prompt_studio_deps.managed_prompt_studio_db({"user_id": str(bob["id"]), "client_id": "history-test"}) as db:
        project = db.create_project(name="Bob", user_id=str(bob["id"]))
        prompt = db.create_prompt(project["id"], "Other template", system_prompt="Private Bob instructions")
        case = db.create_test_case(project["id"], "Other", inputs={})
        collision = db.create_test_run(project_id=project["id"], prompt_id=prompt["id"], test_case_id=case["id"],
                                        model_name="private", inputs={}, outputs={})
        # Alice's first row is the private-project control; match the selected numeric run ID naturally.
        while collision["id"] < original["id"]:
            collision = db.create_test_run(project_id=project["id"], prompt_id=prompt["id"], test_case_id=case["id"],
                                            model_name="private", inputs={}, outputs={})
    assert collision["id"] == original["id"]  # nosec B101 - pytest contract
    response = context.client.get(f"{ROOT}/test-cases/runs/{original['id']}", headers={**bob["headers"],
        "X-TLDW-Expected-User-ID": str(context.owners["alice"]["id"])})
    assert response.status_code == 412, response.text  # nosec B101 - pytest contract
    assert "Private Bob instructions" not in response.text  # nosec B101 - pytest contract
