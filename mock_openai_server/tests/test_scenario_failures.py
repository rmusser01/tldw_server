"""Unit coverage for mock OpenAI scenario failure injection."""

import json

from fastapi.testclient import TestClient

from mock_openai.config import MockConfig
from mock_openai.server import (
    _scenario_failure_counts,
    app,
    get_config_instance,
    scenario_failure_response,
)


def test_scenario_failure_applies_configured_number_of_times():
    _scenario_failure_counts.clear()
    config = MockConfig.from_dict(
        {
            "scenario_failures": {
                "chat_completions": [
                    {
                        "match": {"model": "gpt-4.1-mini"},
                        "status_code": 503,
                        "message": "UAT transient chat failure",
                        "type": "server_error",
                        "code": "uat_fail_once",
                        "times": 1,
                    }
                ]
            }
        }
    )
    request_data = {"model": "gpt-4.1-mini", "messages": []}

    first = scenario_failure_response("chat_completions", request_data, config)
    second = scenario_failure_response("chat_completions", request_data, config)

    assert first is not None
    assert first.status_code == 503
    assert json.loads(first.body)["error"] == {
        "message": "UAT transient chat failure",
        "type": "server_error",
        "code": "uat_fail_once",
    }
    assert second is None


def test_scenario_failure_respects_match_rules():
    _scenario_failure_counts.clear()
    config = MockConfig.from_dict(
        {
            "scenario_failures": {
                "chat_completions": [
                    {
                        "match": {"content_regex": "retry me"},
                        "status_code": 503,
                        "times": 1,
                    }
                ]
            }
        }
    )

    assert (
        scenario_failure_response(
            "chat_completions",
            {"model": "gpt", "messages": [{"role": "user", "content": "hello"}]},
            config,
        )
        is None
    )
    assert (
        scenario_failure_response(
            "chat_completions",
            {"model": "gpt", "messages": [{"role": "user", "content": "retry me"}]},
            config,
        )
        is not None
    )


def test_scenario_failures_apply_to_non_chat_endpoint_handlers():
    _scenario_failure_counts.clear()
    config = MockConfig.from_dict(
        {
            "server": {"require_auth": False},
            "scenario_failures": {
                "embeddings": [
                    {
                        "match": {"model": "text-embedding-3-small"},
                        "status_code": 503,
                        "message": "Embedding scenario failure",
                        "code": "embedding_fail_once",
                        "times": 1,
                    }
                ],
                "completions": [
                    {
                        "match": {"model": "gpt-3.5-turbo-instruct"},
                        "status_code": 504,
                        "message": "Completion scenario failure",
                        "code": "completion_fail_once",
                        "times": 1,
                    }
                ],
                "models": [
                    {
                        "match": {},
                        "status_code": 503,
                        "message": "Model list scenario failure",
                        "code": "models_fail_once",
                        "times": 1,
                    }
                ],
            },
        }
    )
    app.dependency_overrides[get_config_instance] = lambda: config

    try:
        with TestClient(app) as client:
            embeddings = client.post(
                "/v1/embeddings",
                json={
                    "model": "text-embedding-3-small",
                    "input": "hello",
                },
            )
            completions = client.post(
                "/v1/completions",
                json={
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": "Once",
                },
            )
            models = client.get("/v1/models")

        assert embeddings.status_code == 503
        assert embeddings.json()["error"]["code"] == "embedding_fail_once"
        assert completions.status_code == 504
        assert completions.json()["error"]["code"] == "completion_fail_once"
        assert models.status_code == 503
        assert models.json()["error"]["code"] == "models_fail_once"
    finally:
        app.dependency_overrides.clear()
        _scenario_failure_counts.clear()
