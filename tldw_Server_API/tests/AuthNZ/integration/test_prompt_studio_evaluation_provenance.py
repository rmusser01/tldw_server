"""Saved evaluation reads preserve provenance without replaying execution."""

import copy

import pytest

from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories.evaluations import EvaluationsRepository
from tldw_Server_API.tests.AuthNZ.integration.test_prompt_studio_run_history import (
    ROOT,
)
from tldw_Server_API.tests.AuthNZ.integration.test_prompt_studio_run_history import (
    history_accounts as history_accounts,
)
from tldw_Server_API.tests.AuthNZ.integration.test_prompt_studio_run_history import (
    private_history_paths as private_history_paths,
)
from tldw_Server_API.tests.AuthNZ.integration.test_prompt_studio_run_history import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration


def evaluation(context, **fields):
    """Seed through the real existing writer, with no provider or scorer call."""
    return context.db.create_evaluation(project_id=context.project["id"],
        prompt_id=context.prompt["id"], name="Saved Rowan revision", description="Exact recorded comparison",
        **fields)


def read_pair(context, row):
    response = context.client.get(ROOT + "/evaluations", headers=context.headers,
        params={"project_id": context.project["id"]})
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    detail = context.client.get(f"{ROOT}/evaluations/{row['id']}", headers=context.headers)
    assert detail.status_code == 200, detail.text  # nosec B101 - pytest contract
    listed = next(item for item in response.json()["evaluations"] if item["id"] == row["id"])
    return listed, detail.json()


def test_list_and_detail_preserve_saved_public_provenance_without_rewriting_records(history_accounts):
    context = history_accounts
    configs = [{"provider": "ollama", "model": "qwen2.5:0.5b", "temperature": 0, "max_tokens": 32,
                "api_key": "synthetic-not-public", "unknown_secret": "synthetic-not-public"},  # nosec B105 - synthetic scrub marker
               {"api_name": "openai", "model_name": "recorded-second-model", "top_p": 0.8}]
    row = evaluation(context, model_configs=configs, test_case_ids=[context.case["id"]])
    context.db.update_evaluation(row["id"], {"test_run_ids": [23, 24], "status": "failed",
        "aggregate_metrics": {"accuracy": 0, "pass_rate": 0.5}, "error_message": "Recorded refusal"})  # nosec B105 - metric, not a credential
    before = copy.deepcopy(context.db.get_evaluation(row["id"]))
    listed, detail = read_pair(context, row)
    for data in (listed, detail):
        assert data["name"] == "Saved Rowan revision"  # nosec B101 - pytest contract
        assert data["description"] == "Exact recorded comparison"  # nosec B101 - pytest contract
        assert data["project_id"] == context.project["id"]  # nosec B101 - pytest contract
        assert data["prompt_id"] == context.prompt["id"]  # nosec B101 - pytest contract
        assert data["test_case_ids"] == [context.case["id"]]  # nosec B101 - pytest contract
        assert data["test_run_ids"] == [23, 24]  # nosec B101 - pytest contract
        assert data["model_configs"] == [{k: v for k, v in configs[0].items() if k not in {"api_key", "unknown_secret"}}, configs[1]]  # nosec B101 - pytest contract
        assert data["config"] == data["model_configs"][0]  # nosec B101 - pytest contract
        assert data["aggregate_metrics"] == data["metrics"] == {"accuracy": 0, "pass_rate": 0.5}  # nosec B101, B105 - pytest metric, not a credential
        assert data["error_message"] == "Recorded refusal"  # nosec B101 - pytest contract
        assert "synthetic-not-public" not in str(data)  # nosec B101 - pytest contract
    assert context.db.get_evaluation(row["id"]) == before  # nosec B101 - pytest contract


@pytest.mark.parametrize("shape", ["missing", "empty", "legacy-dict"])
def test_both_reads_distinguish_missing_empty_and_legacy_recorded_metadata(history_accounts, shape):
    context = history_accounts
    configs = None if shape == "missing" else [] if shape == "empty" else {"provider": "ollama", "model": "recorded"}
    # The canonical case-ID column is NOT NULL; the existing writer records [].
    cases = []
    row = evaluation(context, model_configs=configs, test_case_ids=cases)
    if shape != "missing":
        context.db.update_evaluation(row["id"], {"aggregate_metrics": {"accuracy": 0}})
    for data in read_pair(context, row):
        assert data["test_case_ids"] == cases  # nosec B101 - pytest contract
        assert data["model_configs"] == (None if shape == "missing" else [] if shape == "empty" else [configs])  # nosec B101 - pytest contract
        assert data["aggregate_metrics"] == (None if shape == "missing" else {"accuracy": 0})  # nosec B101 - pytest contract


def test_list_rejects_foreign_stored_project_before_reading_rows(history_accounts, monkeypatch):
    context = history_accounts
    calls = []
    monkeypatch.setattr(EvaluationsRepository, "list", lambda *args, **kwargs: calls.append(kwargs))
    response = context.client.get(ROOT + "/evaluations", headers=context.headers,
        params={"project_id": context.foreign["id"]})
    assert response.status_code == 403, response.text  # nosec B101 - pytest contract
    assert calls == []  # nosec B101 - pytest contract


def test_list_refuses_changed_expected_owner_before_reading_rows(history_accounts, monkeypatch):
    context = history_accounts
    calls = []
    monkeypatch.setattr(EvaluationsRepository, "list", lambda *args, **kwargs: calls.append(kwargs))
    response = context.client.get(ROOT + "/evaluations", headers={**context.headers,
        "X-TLDW-Expected-User-ID": str(context.owners["bob"]["id"])},
        params={"project_id": context.project["id"]})
    assert response.status_code == 412, response.text  # nosec B101 - pytest contract
    assert response.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101 - pytest contract
    assert response.headers["cache-control"] == "no-store"  # nosec B101 - pytest contract
    assert calls == []  # nosec B101 - pytest contract


@pytest.mark.parametrize("matching", [False, True])
def test_list_keeps_optional_matching_expected_owner_compatibility(history_accounts, matching):
    context = history_accounts
    row = evaluation(context)
    headers = dict(context.headers)
    if matching:
        headers["X-TLDW-Expected-User-ID"] = str(context.owners["alice"]["id"])
    response = context.client.get(ROOT + "/evaluations", headers=headers,
        params={"project_id": context.project["id"]})
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    assert response.json()["evaluations"][0]["id"] == row["id"]  # nosec B101 - pytest contract


def test_list_requires_real_authentication(history_accounts):
    response = history_accounts.client.get(ROOT + "/evaluations", params={"project_id": history_accounts.project["id"]})
    assert response.status_code == 401, response.text  # nosec B101 - pytest contract

    assert set(response.json()) == {"detail"}  # nosec B101 - Unauthenticated requests expose no evaluation provenance.
    assert response.json()["detail"]  # nosec B101 - Preserve recovery information without private data.


def test_projection_preserves_existing_exact_offset_pagination(history_accounts):
    context = history_accounts
    rows = [evaluation(context, model_configs={"model": f"saved-{index}"}) for index in range(3)]
    response = context.client.get(ROOT + "/evaluations", headers=context.headers,
        params={"project_id": context.project["id"], "limit": 1, "offset": 1})
    assert response.status_code == 200, response.text  # nosec B101 - pytest contract
    assert [item["id"] for item in response.json()["evaluations"]] == [rows[1]["id"]]  # nosec B101 - pytest contract
    assert response.json()["total"] == 3  # nosec B101 - pytest contract
    assert response.json()["evaluations"][0]["model_configs"] == [{"model": "saved-1"}]  # nosec B101 - pytest contract
