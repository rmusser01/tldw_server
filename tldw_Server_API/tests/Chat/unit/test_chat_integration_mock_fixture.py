from __future__ import annotations

import os
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas import chat_request_schemas
from tldw_Server_API.tests.Chat.integration import conftest as integration_conftest

pytestmark = pytest.mark.unit

_OPENAI_ENDPOINT_ALIASES = (
    "OPENAI_API_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "MOCK_OPENAI_BASE_URL",
)


class _MockServerRequest:
    def getfixturevalue(self, name: str) -> str:
        assert name == "mock_openai_server"
        return "http://127.0.0.1:18081"


def test_forced_openai_mock_replaces_and_restores_preseeded_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_key = "preseeded-real-key"
    original_schema_key = "preseeded-schema-key"
    original_endpoints = {
        name: f"https://preseeded-{index}.example/v1"
        for index, name in enumerate(_OPENAI_ENDPOINT_ALIASES)
    }
    monkeypatch.setenv("USE_OPENAI_MOCK_SERVER", "true")
    monkeypatch.setenv("OPENAI_API_KEY", original_key)
    monkeypatch.setitem(chat_request_schemas.API_KEYS, "openai", original_schema_key)
    for name, value in original_endpoints.items():
        monkeypatch.setenv(name, value)

    fixture_patch = pytest.MonkeyPatch()
    fixture_impl: Any = integration_conftest._auto_configure_openai_mock.__wrapped__
    fixture = fixture_impl(_MockServerRequest(), fixture_patch)
    next(fixture)

    expected_endpoint = "http://127.0.0.1:18081/v1"
    assert os.environ["OPENAI_API_KEY"] == "sk-mock-key-12345"
    assert chat_request_schemas.API_KEYS["openai"] == "sk-mock-key-12345"
    assert {
        name: os.environ.get(name) for name in _OPENAI_ENDPOINT_ALIASES
    } == dict.fromkeys(_OPENAI_ENDPOINT_ALIASES, expected_endpoint)

    with pytest.raises(StopIteration):
        next(fixture)
    fixture_patch.undo()

    assert os.environ["OPENAI_API_KEY"] == original_key
    assert chat_request_schemas.API_KEYS["openai"] == original_schema_key
    assert {
        name: os.environ.get(name) for name in _OPENAI_ENDPOINT_ALIASES
    } == original_endpoints
