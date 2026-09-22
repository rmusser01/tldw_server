"""All copies of get_http_status_from_exception must agree, and must recover a status
carried only in an exception message.

Two of the three copies used r"HTTP\\s+(\\d{3})" -- double-escaped inside a raw string,
so the pattern searched for a literal backslash and could never match. Extraction then
returned None, build_sanitized_chat_error took its status_code-is-None branch, and an
upstream 429 reached the client as ChatProviderError's default 502: no Retry-After, and
no rate-limit classification for any caller keyed on 429.

core/http_client.py raises NetworkError(f"HTTP {status}") with no status_code kwarg when
httpx is unavailable, and core/Embeddings/connection_pool.py does the same, so the message
text is the only carrier on those paths.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.chat_orchestrator import (
    _get_http_status_from_exception as chat_extract,
)
from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    get_http_status_from_exception as llm_extract,
)
from tldw_Server_API.app.core.Local_LLM.http_utils import (
    get_http_status_from_exception as local_extract,
)

EXTRACTORS = pytest.mark.parametrize(
    "extract",
    [
        pytest.param(llm_extract, id="LLM_Calls.error_utils"),
        pytest.param(chat_extract, id="Chat.chat_orchestrator"),
        pytest.param(local_extract, id="Local_LLM.http_utils"),
    ],
)


@EXTRACTORS
@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_status_recovered_from_network_error_message(extract, status: int) -> None:
    assert extract(NetworkError(f"HTTP {status}")) == status


@EXTRACTORS
def test_status_recovered_from_embedded_message(extract) -> None:
    assert extract(NetworkError("Provider returned HTTP 429 Too Many Requests")) == 429


@EXTRACTORS
def test_no_status_present_returns_none(extract) -> None:
    assert extract(NetworkError("connection reset by peer")) is None


@EXTRACTORS
def test_attribute_branch_still_wins(extract) -> None:
    class _Resp:
        status_code = 503

    exc = NetworkError("HTTP 429")
    exc.response = _Resp()  # type: ignore[attr-defined]
    assert extract(exc) == 503
