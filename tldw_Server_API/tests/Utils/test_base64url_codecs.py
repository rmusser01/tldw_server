"""Two base64url decoders, split by trust class rather than merged.

The 23 production copies of this idiom had already diverged three ways on strictness.
These tests pin what the shared helpers do, and demonstrate the defect the 16 lax copies
carry: base64.urlsafe_b64decode without validate=True silently DISCARDS out-of-alphabet
characters, so a tampered segment decodes to a different-but-valid byte string instead
of raising.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

import pytest

from tldw_Server_API.app.core.Utils.base64url import (
    Base64SegmentError,
    SignatureMismatchError,
    decode_canonical_segment,
    decode_opaque_cursor_segment,
    encode_segment,
    verify_signed_token,
)


@pytest.mark.parametrize(
    "raw",
    [b"", b"a", b"ab", b"abc", b"abcd", b"\x00\xff\xfe", bytes(range(256))],
)
def test_roundtrip_every_padding_class(raw: bytes) -> None:
    assert decode_opaque_cursor_segment(encode_segment(raw)) == raw
    assert decode_canonical_segment(encode_segment(raw)) == raw


def test_unpadded_output_and_urlsafe_alphabet() -> None:
    token = encode_segment(bytes(range(256)))
    assert "=" not in token
    assert "+" not in token and "/" not in token


def test_the_defect_the_lax_copies_carry() -> None:
    """Demonstrates why validate=True matters, using the stdlib call the copies made."""
    tampered = encode_segment(b"hello world") + "!!"
    silently_accepted = base64.urlsafe_b64decode(tampered + "=" * (-len(tampered) % 4))
    assert silently_accepted == b"hello world", "stdlib discarded the junk rather than raising"

    with pytest.raises(Base64SegmentError):
        decode_opaque_cursor_segment(tampered)


@pytest.mark.parametrize("bad", ["abc!def", "a b c", "ab\ncd", "€€€€"])
def test_out_of_alphabet_is_rejected(bad: str) -> None:
    with pytest.raises(Base64SegmentError):
        decode_opaque_cursor_segment(bad)


def test_oversized_is_refused_before_decoding() -> None:
    huge = encode_segment(b"x" * 10_000)
    with pytest.raises(Base64SegmentError, match="exceeds"):
        decode_opaque_cursor_segment(huge, max_encoded_len=128)


def test_signed_requires_canonical_form_and_opaque_does_not() -> None:
    """Two strings, same bytes: fine for a cursor, not for something signed."""
    token = encode_segment(b"payload")
    non_canonical = token + "="          # re-padded, decodes identically
    assert decode_opaque_cursor_segment(non_canonical) == b"payload"
    with pytest.raises(Base64SegmentError, match="canonical"):
        decode_canonical_segment(non_canonical)


def test_non_string_input_is_rejected() -> None:
    for bad in (None, 123, b"bytes"):
        with pytest.raises(Base64SegmentError):
            decode_opaque_cursor_segment(bad)  # type: ignore[arg-type]


def test_error_is_a_valueerror_so_existing_handlers_still_catch_it() -> None:
    """Call sites currently catch ValueError/binascii.Error; migration must not slip past."""
    assert issubclass(Base64SegmentError, ValueError)


_KEY = b"k" * 32


def _sign(payload: bytes, key: bytes = _KEY) -> str:
    signature = hmac.new(key, payload, hashlib.sha256).digest()
    return f"{encode_segment(payload)}.{encode_segment(signature)}"


def test_signed_token_valid_returns_payload() -> None:
    assert verify_signed_token(_sign(b'{"a":1}'), _KEY) == b'{"a":1}'


def test_signed_token_wrong_key_is_rejected() -> None:
    with pytest.raises(SignatureMismatchError):
        verify_signed_token(_sign(b"payload", key=b"other" * 8), _KEY)


def test_signed_token_tampered_payload_is_rejected() -> None:
    _, signature = _sign(b'{"user":1}').split(".")
    forged = encode_segment(b'{"user":2}') + "." + signature
    with pytest.raises(SignatureMismatchError):
        verify_signed_token(forged, _KEY)


def test_signed_token_mismatch_is_a_segment_error() -> None:
    """Callers that map every bad token to one status keep catching ValueError."""
    assert issubclass(SignatureMismatchError, Base64SegmentError)


@pytest.mark.parametrize("key", [b"", None, "text-key"])
def test_signed_token_cannot_be_used_without_a_key(key) -> None:
    with pytest.raises(TypeError):
        verify_signed_token(_sign(b"payload"), key)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "token",
    ["", "onlyonesegment", "a.b.c", "AAAA.!!!!", "AAAA.A", None],
)
def test_signed_token_malformed_is_rejected(token) -> None:
    with pytest.raises(Base64SegmentError):
        verify_signed_token(token, _KEY)  # type: ignore[arg-type]


def test_signed_token_non_canonical_segment_is_rejected() -> None:
    payload, signature = _sign(b"payload").split(".")
    with pytest.raises(Base64SegmentError, match="canonical"):
        verify_signed_token(f"{payload}=.{signature}", _KEY)


def test_signed_token_oversized_is_refused_before_decoding() -> None:
    with pytest.raises(Base64SegmentError, match="exceeds"):
        verify_signed_token(_sign(b"x" * 1000), _KEY, max_encoded_len=128)


def test_signed_token_over_encoded_payload_segment() -> None:
    """Some tokens MAC the encoded payload segment rather than the raw bytes."""
    encoded = encode_segment(b"payload")
    signature = hmac.new(_KEY, encoded.encode("ascii"), hashlib.sha256).digest()
    token = f"{encoded}.{encode_segment(signature)}"
    assert verify_signed_token(token, _KEY, sign_encoded_payload=True) == b"payload"
    with pytest.raises(SignatureMismatchError):
        verify_signed_token(token, _KEY)
