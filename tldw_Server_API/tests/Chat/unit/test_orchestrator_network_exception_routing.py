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
