"""A Prompt Studio evaluation stops at an LLM quota refusal and records it (TASK-13638)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager

pytestmark = pytest.mark.unit


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
    )


class _EvaluationDb:
    client_id = "quota-refusal"

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return {"id": prompt_id, "project_id": 71, "deleted": False, "system_prompt": "Answer.", "user_prompt": "Q: {question}"}

    def get_test_cases_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        return [
            {"id": i, "project_id": 71, "inputs": {"question": "q"}, "expected_outputs": {"response": "a"}} for i in ids
        ]

    def get_test_case(self, test_case_id: int) -> dict[str, Any]:
        return self.get_test_cases_by_ids([test_case_id])[0]

    def get_evaluation(self, evaluation_id: int) -> dict[str, Any]:
        return {
            "id": evaluation_id,
            "uuid": "eval-uuid",
            "prompt_id": 1,
            "test_case_ids": [2, 3],
            "model_configs": {"model": "m"},
        }

    def update_evaluation(self, evaluation_id: int, updates: dict[str, Any]) -> None:
        self.updates.append(dict(updates))

    def create_test_run(self, **kwargs: Any) -> dict[str, Any]:
        self.runs.append(kwargs)
        return {"id": len(self.runs)}


@pytest.mark.asyncio
async def test_evaluation_stops_at_the_refusal_and_marks_the_record_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _EvaluationDb()
    calls: list[dict[str, Any]] = []

    def _refused(self: EvaluationManager, **kwargs: Any) -> Any:
        calls.append(kwargs)
        raise _refusal()

    monkeypatch.setattr(EvaluationManager, "_call_adapter_text", _refused)

    with pytest.raises(LLMQuotaExceededError):
        await EvaluationManager(db).run_evaluation_with_existing_record(
            evaluation_id=5, prompt_id=1, test_case_ids=[2, 3], model="m"
        )

    assert len(calls) == 1
    assert db.runs == []
    assert db.updates[-1]["status"] == "failed"
    assert db.updates[-1]["error_message"] == "llm_quota_exceeded: Monthly LLM token limit reached"


@pytest.mark.asyncio
async def test_background_evaluation_does_not_raise_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_evaluations as eval_endpoint

    class _Manager:
        def __init__(self, _db: Any) -> None:
            return None

        async def run_evaluation_with_existing_record(self, **_kwargs: Any) -> dict[str, Any]:
            raise _refusal()

    class _Runtime:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(eval_endpoint, "EvaluationManager", _Manager)
    monkeypatch.setattr(eval_endpoint, "provider_requires_api_key", lambda _provider: False)
    handle = SimpleNamespace(api_key=None, app_config={}, credentials_resolved=True)
    db = _EvaluationDb()

    await eval_endpoint.run_evaluation_async(
        5,
        db,
        provider="openai",
        model="m",
        credential_runtime=_Runtime(),
        provider_credentials=handle,
    )

    # Only the start is written here: the generic handler does not overwrite the manager's llm_quota_exceeded record.
    assert [update["status"] for update in db.updates] == ["running"]


@pytest.mark.asyncio
async def test_a_refused_evaluation_keeps_the_links_to_the_cases_that_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failed record still points to the test runs stored before the refusal (TASK-13864)."""
    db = _EvaluationDb()
    calls: list[dict[str, Any]] = []

    def _second_refused(self: EvaluationManager, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) > 1:
            raise _refusal()
        return "a"

    monkeypatch.setattr(EvaluationManager, "_call_adapter_text", _second_refused)

    with pytest.raises(LLMQuotaExceededError):
        await EvaluationManager(db).run_evaluation_with_existing_record(
            evaluation_id=5, prompt_id=1, test_case_ids=[2, 3], model="m"
        )

    assert len(db.runs) == 1
    assert db.updates[-1]["status"] == "failed"
    assert db.updates[-1]["test_run_ids"] == [1]
