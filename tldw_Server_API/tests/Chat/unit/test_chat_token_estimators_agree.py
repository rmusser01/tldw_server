"""The chat path's prompt-size estimators agree (TASK-13843, review finding chat-7).

Rate limiting, the billing reservation, the usage-quota check, queue admission, the
prompt-cost guardrail, the post-call stream estimates and every recorded provider call's
estimate row (usage_accounting) size a prompt with one estimate:
``prompt_cost_envelope.estimate_segment_tokens`` (data:image payloads redacted by one
pattern, four characters per token rounded up).
"""

import json

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import estimate_segment_tokens
from tldw_Server_API.app.core.LLM_Calls import usage_accounting

_IMAGE = "data:image/png;base64," + "A" * 4000
_MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": _IMAGE}},
        ],
    },
]


@pytest.mark.unit
@pytest.mark.parametrize("extra", range(4), ids=lambda n: f"len_mod_4_plus_{n}")
def test_chat_prompt_estimators_agree_on_the_same_request(extra):
    messages = [*_MESSAGES, {"role": "user", "content": "x" * extra}]
    body = json.dumps(messages, default=str)
    expected = estimate_segment_tokens(body)

    assert chat_service.estimate_tokens_from_json(body) == expected
    assert chat_service._estimate_tokens_from_messages(messages) == expected
    assert usage_accounting._prompt_estimate({"messages": messages}) == expected


@pytest.mark.unit
def test_the_estimate_rounds_up():
    assert chat_service.estimate_tokens_from_json("abcde") == 2


@pytest.mark.unit
def test_an_empty_request_still_costs_one_token():
    assert chat_service.estimate_tokens_from_json("") == 1


@pytest.mark.unit
def test_the_endpoint_has_no_redactor_or_estimator_of_its_own():
    assert chat_endpoint.estimate_tokens_from_json is chat_service.estimate_tokens_from_json
    for name in ("_sanitize_json_for_rate_limit", "_estimate_tokens_for_queue", "DATA_URI_REDACT_RE"):
        assert not hasattr(chat_endpoint, name), name
    for name in ("_sanitize_data_uris", "_sanitize_messages_for_token_estimate", "DATA_URI_REDACT_RE"):
        assert not hasattr(chat_service, name), name
