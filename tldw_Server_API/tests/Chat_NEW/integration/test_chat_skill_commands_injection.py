import pytest


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return ""


def _last_message(messages: list[dict], role: str) -> dict | None:
    for message in reversed(messages):
        if message.get("role") == role:
            return message
    return None


@pytest.mark.integration
def test_skill_command_system_mode_injects_output_and_preserves_args(
    monkeypatch,
    credentialed_test_client,
    auth_headers,
):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    async def fake_execute(ctx, skill_name, skill_args):
        assert skill_name == "summarize"
        assert skill_args == "release notes"
        return {
            "success": True,
            "execution_mode": "inline",
            "rendered_prompt": "Skill inline output",
            "fork_output": None,
        }

    monkeypatch.setattr(command_router, "_execute_skill", fake_execute)

    captured = {"messages": None, "kwargs": None}

    def fake_call(**kwargs):
        captured["kwargs"] = kwargs
        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "/skill summarize release notes"}],
        "stream": False,
    }
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 200

    messages = captured["messages"] or []
    system_msg = _last_message(messages, "system")
    user_msg = _last_message(messages, "user")
    assert user_msg is not None
    system_text = _message_text(system_msg) if system_msg is not None else str((captured["kwargs"] or {}).get("system_message") or "")
    assert "[/skill]" in system_text
    assert "Skill inline output" in system_text
    assert _message_text(user_msg).strip() == "summarize release notes"


@pytest.mark.integration
def test_skill_command_preface_mode_prefixes_user_message(
    monkeypatch,
    credentialed_test_client,
    auth_headers,
):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "preface")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    async def fake_execute(ctx, skill_name, skill_args):
        return {
            "success": True,
            "execution_mode": "inline",
            "rendered_prompt": "Skill preface output",
            "fork_output": None,
        }

    monkeypatch.setattr(command_router, "_execute_skill", fake_execute)

    captured = {"messages": None, "kwargs": None}

    def fake_call(**kwargs):
        captured["kwargs"] = kwargs
        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "/skill summarize release notes"}],
        "stream": False,
    }
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 200

    messages = captured["messages"] or []
    user_msg = _last_message(messages, "user")
    assert user_msg is not None
    user_text = _message_text(user_msg)
    assert user_text.startswith("[/skill]")
    assert "Skill preface output" in user_text
    assert "summarize release notes" in user_text


@pytest.mark.integration
def test_skill_command_replace_mode_uses_fork_output(
    monkeypatch,
    credentialed_test_client,
    auth_headers,
):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "replace")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    async def fake_execute(ctx, skill_name, skill_args):
        return {
            "success": True,
            "execution_mode": "fork",
            "rendered_prompt": "ignored",
            "fork_output": "Fork execution output",
        }

    monkeypatch.setattr(command_router, "_execute_skill", fake_execute)

    captured = {"messages": None, "kwargs": None}

    def fake_call(**kwargs):
        captured["kwargs"] = kwargs
        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "/skill summarize release notes"}],
        "stream": False,
    }
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 200

    messages = captured["messages"] or []
    user_msg = _last_message(messages, "user")
    assert user_msg is not None
    user_text = _message_text(user_msg)
    assert user_text.startswith("[/skill]")
    assert "Fork execution output" in user_text


@pytest.mark.integration
def test_skills_command_system_mode_lists_invocable_skills(
    monkeypatch,
    credentialed_test_client,
    auth_headers,
):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.setenv("CHAT_COMMAND_INJECTION_MODE", "system")

    from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
    from tldw_Server_API.app.core.Chat import command_router

    async def fake_list(ctx, filter_text=None):
        skills = [
            {"name": "summarize", "description": "Summarize docs", "argument_hint": "<topic>"},
            {"name": "hidden-skill", "description": "Hidden", "argument_hint": None},
        ]
        return [skill for skill in skills if skill["name"] != "hidden-skill"]

    monkeypatch.setattr(command_router, "_list_invocable_skills", fake_list)

    captured = {"messages": None, "kwargs": None}

    def fake_call(**kwargs):
        captured["kwargs"] = kwargs
        captured["messages"] = kwargs.get("messages_payload")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", fake_call)

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "/skills"}],
        "stream": False,
    }
    response = credentialed_test_client.post(
        "/api/v1/chat/completions",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 200

    messages = captured["messages"] or []
    system_msg = _last_message(messages, "system")
    system_text = _message_text(system_msg) if system_msg is not None else str((captured["kwargs"] or {}).get("system_message") or "")
    assert "[/skills]" in system_text
    assert "summarize" in system_text
    assert "hidden-skill" not in system_text
