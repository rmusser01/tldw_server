"""Regression guard for TASK-13287.

`get_http_status_from_exception` exists in four independent copies. Two of them match
the status out of an exception's message with

    re.search(r"HTTP\\\\s+(\\\\d{3})", str(exc))

Inside a raw string `\\\\s` is a literal backslash followed by `s`, so the pattern looks
for `HTTP\\s 429` and can never match `HTTP 429`. The third copy,
`Local_LLM/http_utils.py`, has the single-escape form and works.

Consequence: `core/http_client.py:_AiohttpResponse.raise_for_status` raises
`NetworkError(f"HTTP {status}")` with no `.status_code` attribute when httpx is
unavailable, and `Embeddings/connection_pool.py` does the same. Extraction returns
None, `build_sanitized_chat_error` produces a `ChatProviderError` with no status, and
`core/exceptions.py` defaults that to **502**. An upstream 429 therefore reaches the
client as 502, with no Retry-After and no rate-limit classification for any caller
keyed on 429.

The parity test at the end is the durable part: four copies of one extraction rule is
how this drifted in the first place, and a test that asserts they agree will fail the
next time one is edited alone.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import NetworkError

pytestmark = pytest.mark.unit


def _extractors():
    """Every live entry point for this rule, by import path.

    All four now resolve to the single implementation in `LLM_Calls/error_utils`; the
    other three are kept as re-exports so their importers are unaffected. They stay
    listed here because the point of the parity assertions is that a future edit to any
    one of them is caught -- including an edit that re-introduces a local copy.
    """
    from tldw_Server_API.app.core.Chat import chat_orchestrator
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.LLM_Calls import error_utils
    from tldw_Server_API.app.core.Local_LLM import http_utils

    return {
        "LLM_Calls/error_utils": error_utils.get_http_status_from_exception,
        "Chat/chat_orchestrator": chat_orchestrator._get_http_status_from_exception,
        "Local_LLM/http_utils": http_utils.get_http_status_from_exception,
        "Embeddings/Embeddings_Create": Embeddings_Create._get_http_status_from_exception,
    }


@pytest.mark.parametrize("name", list(_extractors()))
def test_response_status_wins_over_an_exception_attribute(name: str) -> None:
    """Pin the precedence, because the four copies did not agree on it.

    `Embeddings_Create` checked `exc.status_code` before `exc.response.status_code`;
    the other three checked the response first. They only diverge when an exception
    carries both and they disagree, which is rare but silent when it happens -- so the
    order is now a recorded decision rather than an accident of which copy you hit.

    The response wins: it is the status actually returned on the wire, whereas an
    attribute on the exception may have been set by a wrapper further up.

    One case per copy: a single diverging copy must not stop the other three being
    reported.
    """

    class _Response:
        """Minimal response double carrying only the wire status."""

        status_code = 429

    class _Both(Exception):
        """Exception carrying a status on itself AND on its response, disagreeing.

        The only shape where the precedence is observable.
        """

        status_code = 500
        response = _Response()

    assert _extractors()[name](_Both("upstream")) == 429, (
        f"{name} preferred the exception attribute (500) over the response status "
        "(429); the four copies disagreed on this and it is now pinned"
    )


@pytest.mark.parametrize("name", list(_extractors()))
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 429", 429),
        ("HTTP 429 rate limited", 429),
        ("HTTP 503 from local backend", 503),
        ("HTTP  404", 404),  # multiple spaces: \s+ must still match
    ],
)
def test_status_is_extracted_from_a_network_error_message(
    name: str, message: str, expected: int
) -> None:
    extractor = _extractors()[name]

    got = extractor(NetworkError(message))

    assert got == expected, (
        f"{name} returned {got!r} for {message!r}. A double-escaped regex cannot match "
        "a real 'HTTP <code>' message, so the status is lost and the caller defaults "
        "to 502 -- an upstream 429 reaches the client as 502 with no Retry-After."
    )


def test_all_copies_agree() -> None:
    """Four copies of one rule drifted; pin that they answer identically."""
    cases = [
        NetworkError("HTTP 429 rate limited"),
        NetworkError("HTTP 500 upstream exploded"),
        NetworkError("connection reset"),  # no status present
    ]
    for exc in cases:
        answers = {name: fn(exc) for name, fn in _extractors().items()}
        assert len(set(answers.values())) == 1, (
            f"copies disagree for {str(exc)!r}: {answers}"
        )


@pytest.mark.parametrize("name", list(_extractors()))
def test_explicit_status_attributes_still_win(name: str) -> None:
    """Control: message parsing is the fallback, not the primary path."""

    class _WithStatus(Exception):
        """Exception whose own status disagrees with the HTTP code in its message."""

        status_code = 418

    assert _extractors()[name](_WithStatus("HTTP 429 in the text")) == 418, (
        f"{name} preferred the message over an explicit status_code attribute"
    )


@pytest.mark.parametrize("name", list(_extractors()))
def test_no_status_returns_none(name: str) -> None:
    """Control: absence must stay distinguishable from a parsed value."""
    assert _extractors()[name](NetworkError("connection reset by peer")) is None, name


def test_chat_orchestrator_cannot_yet_reach_this_path() -> None:
    """Documents a separate, still-open defect rather than silently fixing it.

    Correcting the regex makes the extraction work when called, but the Chat path
    still cannot exercise it: `NetworkError` is absent from
    `_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS`, so the handler that calls the extractor
    never runs for a NetworkError, and the ChatProviderError(504) branch downstream is
    unreachable. Widening what the chat error handler catches is a behaviour change
    with its own blast radius and is left to its own task.

    If this assertion starts failing, NetworkError has been added to the tuple and
    this note (and the task) should be retired.
    """
    from tldw_Server_API.app.core.Chat import chat_orchestrator

    assert not issubclass(
        NetworkError, chat_orchestrator._CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS
    ), "NetworkError is now caught by the chat provider tuple -- update TASK-13287"
