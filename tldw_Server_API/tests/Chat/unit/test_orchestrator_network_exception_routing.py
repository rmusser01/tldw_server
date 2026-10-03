"""The exception tuple must catch everything _is_network_exception claims to classify.

chat_api_call's handler classifies a provider failure: status-bearing errors become
ChatAuthenticationError / ChatRateLimitError / ChatBadRequestError / ChatProviderError,
and status-less network failures take the _is_network_exception branch and become
ChatProviderError(504).

_is_network_exception explicitly names NetworkError and RetryExhaustedError, but neither
was in _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS, so neither could ever reach the handler --
they escaped chat_api_call unmapped and the 504 branch was unreachable. Fixing the
double-escaped status regex alone does NOT fix this path.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.chat_orchestrator import (
    _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS,
    _is_network_exception,
)
from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(NetworkError("connection reset"), id="NetworkError"),
        pytest.param(RetryExhaustedError("gave up"), id="RetryExhaustedError"),
    ],
)
def test_network_exceptions_are_catchable_by_the_handler(exc: Exception) -> None:
    assert _is_network_exception(exc), "precondition: the classifier claims this type"
    assert isinstance(exc, _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS), (
        f"{type(exc).__name__} is classified as a network exception but is not in "
        "_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS, so it escapes chat_api_call unmapped "
        "and the ChatProviderError(504) branch is unreachable for it"
    )


def test_status_bearing_network_error_is_still_status_classified() -> None:
    """A NetworkError carrying a status in its text must classify by status, not as 504."""
    from tldw_Server_API.app.core.Chat.chat_orchestrator import (
        _get_http_status_from_exception,
    )

    assert _get_http_status_from_exception(NetworkError("HTTP 429")) == 429


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        pytest.param(NetworkError("Provider returned HTTP 429 Too Many Requests"), 429, id="429"),
        pytest.param(NetworkError("connection reset by peer"), 504, id="status-less"),
    ],
)
def test_chat_api_call_maps_a_network_error_through_the_real_handler(
    monkeypatch: pytest.MonkeyPatch, raised: Exception, expected_status: int
) -> None:
    """End to end through chat_api_call's own except clause, not just classification.

    TASK-13381: a message-only NetworkError('HTTP 429') must surface as ChatRateLimitError
    (429), and a status-less one as ChatProviderError(504) -- never as a raw NetworkError
    or a generic 500.
    """
    from tldw_Server_API.app.core.Chat import chat_orchestrator
    from tldw_Server_API.app.core.exceptions import ChatProviderError, ChatRateLimitError

    def _failing_dispatch(**_kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(chat_orchestrator, "perform_chat_api_call", _failing_dispatch)

    expected_type = ChatRateLimitError if expected_status == 429 else ChatProviderError
    with pytest.raises(expected_type) as excinfo:
        chat_orchestrator.chat_api_call("openai", [{"role": "user", "content": "hi"}])
    assert excinfo.value.status_code == expected_status
