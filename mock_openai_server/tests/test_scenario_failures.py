"""Unit coverage for mock OpenAI scenario failure injection."""

import json

from mock_openai.config import MockConfig
from mock_openai.server import _scenario_failure_counts, scenario_failure_response


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
