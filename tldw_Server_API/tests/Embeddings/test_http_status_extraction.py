"""Feature-scoped guard for the Embeddings HTTP-status extractor (TASK-13287).

The cross-module parity assertions live in
`tests/LLM_Calls/test_http_status_extraction_parity.py`, because their subject is that
all four copies of the rule agree -- that comparison cannot be split across four
feature subtrees without destroying it.

This module exists so a feature-scoped run of `tests/Embeddings/` still covers the
Embeddings extractor on its own. It asserts the behaviour that matters here rather than
re-deriving the parity comparison: `Embeddings/connection_pool.py` raises
`NetworkError(f"HTTP {status}")` with no status attribute, so if extraction cannot parse
that message the status is lost and `core/exceptions.py` defaults the response to 502 --
an upstream 429 reaches the client as 502, with no Retry-After.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import NetworkError

pytestmark = pytest.mark.unit


def _extractor():
    """The Embeddings entry point for the status-extraction rule."""
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    return Embeddings_Create._get_http_status_from_exception


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 429", 429),
        ("HTTP 429 rate limited", 429),
        ("HTTP 503 from embedding backend", 503),
        ("HTTP  404", 404),  # multiple spaces: \s+ must still match
    ],
)
def test_status_is_extracted_from_a_connection_pool_network_error(
    message: str, expected: int
) -> None:
    """The shape `Embeddings/connection_pool.py` actually raises must parse.

    A double-escaped regex matches a literal backslash and so can never match a real
    "HTTP <code>" message. This is the case that regressed.
    """
    got = _extractor()(NetworkError(message))

    assert got == expected, (
        f"Embeddings extraction returned {got!r} for {message!r}; the status is lost and "
        "the caller defaults to 502, so an upstream 429 reaches the client as 502 with "
        "no Retry-After"
    )


def test_response_status_wins_over_an_exception_attribute() -> None:
    """Embeddings used to check the exception attribute first; pin the corrected order.

    The response status is the one actually returned on the wire, whereas an attribute
    on the exception may have been set by a wrapper further up.
    """

    class _Response:
        """Minimal response double carrying only the wire status."""

        status_code = 429

    class _Both(Exception):
        """Exception carrying a status on itself AND on its response, disagreeing."""

        status_code = 500
        response = _Response()

    assert _extractor()(_Both("upstream")) == 429, (
        "Embeddings preferred the exception attribute (500) over the response status (429)"
    )


def test_no_status_returns_none() -> None:
    """Control: absence must stay distinguishable from a parsed value.

    `None` means "unknown" to `build_sanitized_chat_error`; a sentinel status here is
    what produced the wrong 502 in the first place.
    """
    assert _extractor()(NetworkError("connection reset by peer")) is None
