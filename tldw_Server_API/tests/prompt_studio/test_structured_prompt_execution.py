import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner


def _make_prompt_definition_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "input",
                "label": "Input",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are a careful evaluator.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Evaluate {{input}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ],
        "assembly_config": {
            "legacy_system_roles": ["system", "developer"],
            "legacy_user_roles": ["user"],
            "block_separator": "\n\n",
        },
    }


def _seed_structured_prompt_and_case(isolated_db):
    project = isolated_db.create_project(name="Structured Execution Project", user_id="test-user")
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        name="Structured Execution Prompt",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=_make_prompt_definition_payload(),
        few_shot_examples=[
            {
                "inputs": {"input": "Indexes"},
                "outputs": {"answer": "Use the covering index."},
            }
        ],
        modules_config=[
            {"type": "style_rules", "enabled": True, "config": {"tone": "concise"}}
        ],
    )
    test_case = isolated_db.create_test_case(
        project_id=project["id"],
        name="Structured Execution Test Case",
        inputs={"input": "SQLite FTS"},
        expected_outputs={"response": "ok"},
        is_golden=True,
    )
    return prompt, test_case


@pytest.mark.asyncio
async def test_test_runner_uses_structured_assembly_for_execution(isolated_db):
    prompt, test_case = _seed_structured_prompt_and_case(isolated_db)
    captured: dict[str, object] = {}

    def _fake_call_adapter(
        *,
        provider: str,
        model: str,
        messages_payload,
        system_message,
        temperature: float,
        max_tokens: int,
        app_config=None,
        api_key_override=None,
        credentials_resolved: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        captured["messages_payload"] = messages_payload
        captured["system_message"] = system_message
        captured["credentials_resolved"] = credentials_resolved
        captured["timeout_seconds"] = timeout_seconds
        return "ok"

    runner = TestRunner(isolated_db)
    runner._call_adapter = _fake_call_adapter  # type: ignore[method-assign]

    result = await runner.run_single_test(
        prompt_id=prompt["id"],
        test_case_id=test_case["id"],
        model_config={"provider": "openai", "model": "gpt-4", "parameters": {}},
    )

    assert [message["role"] for message in captured["messages_payload"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert captured["system_message"] is None
    assert captured["credentials_resolved"] is False
    assert captured["timeout_seconds"] is None
    assert result["actual"]["response"] == "ok"
    assert result["scores"]["aggregate_score"] == 1.0


def test_evaluation_manager_uses_structured_assembly_for_evaluations(isolated_db, monkeypatch):
    prompt, test_case = _seed_structured_prompt_and_case(isolated_db)
    captured: dict[str, object] = {}

    def _fake_call_adapter_text(
        *,
        provider: str,
        messages_payload,
        temperature: float,
        max_tokens: int,
        api_key,
        model,
        app_config=None,
        timeout=None,
    ) -> str:
        captured["messages_payload"] = messages_payload
        return "ok"

    monkeypatch.setattr(
        EvaluationManager,
        "_call_adapter_text",
        staticmethod(_fake_call_adapter_text),
    )

    manager = EvaluationManager(isolated_db)
    result = manager.run_evaluation(
        prompt_id=prompt["id"],
        test_case_ids=[test_case["id"]],
        model="gpt-4",
        provider="openai",
    )

    assert [message["role"] for message in captured["messages_payload"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert result["status"] == "completed"
    assert result["metrics"]["average_score"] == 1.0
