"""Regression: a malformed share-token signature must be a 400, not an unhandled 500.

`_decode_knowledge_qa_share_token` decoded the signature segment outside its try block,
so a segment whose length mod 4 == 1 raised binascii.Error (a ValueError) past the
handler and surfaced as an unauthenticated HTTP 500 on the public share route
GET /api/v1/chat/shared/conversations/{share_token}.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.chat import _decode_knowledge_qa_share_token


@pytest.mark.parametrize(
    "token",
    [
        "AAAA.A",        # signature length % 4 == 1 -> binascii.Error
        "AAAA.AAAAA",    # length 5 -> same class of failure
        "AAAA.!!!!",     # non-alphabet characters
        "AAAA.=",        # stray padding only
    ],
)
def test_malformed_signature_segment_raises_400_not_500(token: str) -> None:
    # The contract is a HANDLED client error. 400 for input we cannot decode,
    # 403 for input that decodes but fails the signature compare. Never an
    # unhandled exception, which the route surfaces as an unauthenticated 500.
    with pytest.raises(HTTPException) as excinfo:
        _decode_knowledge_qa_share_token(token)
    assert 400 <= excinfo.value.status_code < 500, (
        f"expected a 4xx for {token!r}, got {excinfo.value.status_code}"
    )


def test_wrong_arity_still_400() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _decode_knowledge_qa_share_token("not-a-valid-token")
    assert excinfo.value.status_code == 400
