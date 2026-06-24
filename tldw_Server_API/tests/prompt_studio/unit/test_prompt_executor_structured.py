from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor


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


def test_legacy_build_prompt_replaces_exact_placeholder_names():
    executor = PromptExecutor(SimpleNamespace(client_id="unit-test"))

    rendered = executor._build_prompt(
        {"user_prompt": "$id $idea {id} {{id}} <id> {i-d}"},
        signature=None,
        inputs={"id": "42", "i-d": "bad"},
    )

    assert rendered == "42 $idea 42 42 42 {i-d}"


@pytest.mark.asyncio
async def test_execute_prompt_uses_structured_assembled_messages(isolated_db, monkeypatch):
    project = isolated_db.create_project(name="Executor Structured Project", user_id="test-user")
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        name="Structured Executor Prompt",
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

    captured: dict[str, object] = {}

    async def _fake_call_llm(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["prompt"] = kwargs.get("prompt")
        return {"content": '{"answer": "ok"}', "tokens": 11}

    monkeypatch.setattr(PromptExecutor, "_call_llm", staticmethod(_fake_call_llm))

    executor = PromptExecutor(isolated_db)
    result = await executor.execute_prompt(
        prompt["id"],
        {"input": "SQLite FTS"},
        {"provider": "openai", "model": "gpt-4", "parameters": {}},
    )

    assert [message["role"] for message in captured["messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert captured["messages"][1]["content"] == "Module style_rules: tone=concise"
    assert captured["messages"][2]["content"] == 'Example input: {"input": "Indexes"}'
    assert captured["messages"][3]["content"] == 'Example output: {"answer": "Use the covering index."}'
    assert captured["messages"][4]["content"] == "Evaluate SQLite FTS"
    assert captured["system_prompt"] is None
    assert result["success"] is True
    assert [message["role"] for message in result["metadata"]["assembled_messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert result["parsed_output"]["raw"] == '{"answer": "ok"}'


@pytest.mark.asyncio
async def test_execute_prompt_keeps_legacy_prompt_string_path(isolated_db, monkeypatch):
    project = isolated_db.create_project(name="Executor Legacy Project", user_id="test-user")
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        name="Legacy Executor Prompt",
        system_prompt="Stay concise.",
        user_prompt="Evaluate {input}",
    )

    captured: dict[str, object] = {}

    async def _fake_call_llm(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["prompt"] = kwargs.get("prompt")
        return {"content": "ok", "tokens": 3}

    monkeypatch.setattr(PromptExecutor, "_call_llm", staticmethod(_fake_call_llm))

    executor = PromptExecutor(isolated_db)
    result = await executor.execute_prompt(
        prompt["id"],
        {"input": "SQLite FTS"},
        {"provider": "openai", "model": "gpt-4", "parameters": {}},
    )

    assert captured["messages"] is None
    assert captured["system_prompt"] == "Stay concise."
    assert captured["prompt"] == "Evaluate SQLite FTS"
    assert result["success"] is True
    assert result["metadata"]["assembled_messages"] == [
        {"role": "system", "content": "Stay concise."},
        {"role": "user", "content": "Evaluate SQLite FTS"},
    ]
    assert result["parsed_output"]["raw"] == "ok"
