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


# --- after consolidation into core/Utils/http_status_extraction.py ---------------


def test_every_entry_point_is_the_same_object() -> None:
    """All former copies now resolve to one implementation, not four look-alikes."""
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.LLM_Calls import error_utils as llm_err
    from tldw_Server_API.app.core.Local_LLM import http_utils as local_http
    from tldw_Server_API.app.core.Utils.http_status_extraction import (
        get_http_status_from_exception as canonical,
    )

    assert llm_err.get_http_status_from_exception is canonical
    assert local_http.get_http_status_from_exception is canonical
    assert Embeddings_Create._get_http_status_from_exception is canonical


def test_is_http_status_error_recognises_requests_everywhere() -> None:
    """The three TTS copies were httpx-only; the shared one also knows requests."""
    from tldw_Server_API.app.core.TTS.adapters import (
        elevenlabs_adapter,
        openai_adapter,
    )
    from tldw_Server_API.app.core.Utils.http_status_extraction import (
        is_http_status_error as canonical,
    )

    assert openai_adapter._is_http_status_error is canonical
    assert elevenlabs_adapter._is_http_status_error is canonical

    class _RequestsHTTPError(Exception):
        pass

    _RequestsHTTPError.__module__ = "requests.exceptions"
    _RequestsHTTPError.__name__ = "HTTPError"
    assert canonical(_RequestsHTTPError()) is True


def test_qwen3_method_delegates_to_the_shared_classifier() -> None:
    from tldw_Server_API.app.core.TTS.adapters.qwen3_runtime_remote import (
        RemoteQwenRuntime,
    )

    class _HttpxStatusError(Exception):
        pass

    _HttpxStatusError.__module__ = "httpx"
    _HttpxStatusError.__name__ = "HTTPStatusError"

    runtime = RemoteQwenRuntime.__new__(RemoteQwenRuntime)
    assert runtime._is_http_status_error(_HttpxStatusError()) is True
    assert runtime._is_http_status_error(ValueError("nope")) is False


def test_error_text_and_network_classifier_have_one_implementation() -> None:
    """get_http_error_text and is_network_error were also copied into Local_LLM."""
    from tldw_Server_API.app.core.LLM_Calls import error_utils as llm_err
    from tldw_Server_API.app.core.Local_LLM import http_utils as local_http
    from tldw_Server_API.app.core.Utils import http_status_extraction as canonical

    assert llm_err.get_http_error_text is canonical.get_http_error_text
    assert local_http.get_http_error_text is canonical.get_http_error_text
    assert llm_err.is_network_error is canonical.is_network_error
    assert local_http.is_network_error is canonical.is_network_error


def test_error_text_reads_an_unread_streaming_body() -> None:
    """httpx raises ResponseNotRead (a RuntimeError) on .text of an unread stream.

    The helper's read-then-retry branch only runs if that exception is caught;
    dropping RuntimeError from the noncritical tuple made it dead code and let
    ResponseNotRead escape from inside callers' error handlers.
    """
    import httpx

    from tldw_Server_API.app.core.Utils.http_status_extraction import get_http_error_text

    request = httpx.Request("POST", "http://upstream.invalid/v1/chat")
    response = httpx.Response(429, request=request, stream=httpx.ByteStream(b"slow down"))
    exc = httpx.HTTPStatusError("429", request=request, response=response)
    assert get_http_error_text(exc) == "slow down"


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        ("httpx", "ConnectTimeout", True),
        ("httpx", "RequestError", True),
        ("httpx", "HTTPStatusError", False),
        ("requests.exceptions", "ConnectionError", True),
        ("requests.exceptions", "HTTPError", False),
        ("builtins", "ValueError", False),
    ],
)
def test_is_network_error_classification(module: str, name: str, expected: bool) -> None:
    from tldw_Server_API.app.core.Utils.http_status_extraction import is_network_error

    cls = type(name, (Exception,), {})
    cls.__module__ = module
    assert is_network_error(cls()) is expected
    assert is_network_error(NetworkError("down")) is True
