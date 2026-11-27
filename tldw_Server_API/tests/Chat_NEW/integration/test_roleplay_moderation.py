"""Roleplay resume generation is moderated at the gateway (TASK-13950.27 PR E).

Lives next to the resume fixtures; the stub registry and moderation fixtures come from ``tests/Moderation``.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.tests.Chat_NEW.integration.test_roleplay_resume_completion import (  # noqa: F401 - fixtures
    input_request,
    post_input,
    resume_api,
)
from tldw_Server_API.tests.Moderation.conftest import (  # noqa: F401 - fixtures
    _clean_hook_state,
    make_service,
    moderation_hooks,
    stub_llm_registry,
)

pytestmark = pytest.mark.real_chat_service


def test_roleplay_resume_input_block(resume_api, moderation_hooks, stub_llm_registry, monkeypatch):
    client, db, cid, headers = resume_api
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    accepted = post_input(client, cid, headers, {**input_request(db, cid), "content": "forbidden"})
    assert accepted.status_code == 201, accepted.text
    body = {"tldw_roleplay_resume_v1": {
        "version": 1, "input_message_id": "resume-chosen", "input_message_version": 1,
        "assistant_message_id": "resume-assistant", "expected": accepted.json()["fence"]}}
    response = client.post(f"/api/v1/chats/{cid}/complete-v2", headers=headers, json=body)
    assert response.status_code == 400, response.text
    assert "Input violates moderation policy" in response.text
    assert stub_llm_registry.requests == []
