"""Two base64url decoders, split by trust class rather than merged.

The 23 production copies of this idiom had already diverged three ways on strictness.
These tests pin what the shared helpers do, and demonstrate the defect the 16 lax copies
carry: base64.urlsafe_b64decode without validate=True silently DISCARDS out-of-alphabet
characters, so a tampered segment decodes to a different-but-valid byte string instead
of raising.
"""

from __future__ import annotations

import base64

import pytest

from tldw_Server_API.app.core.Utils.base64url import (
    Base64SegmentError,
    decode_opaque_cursor_segment,
    decode_signed_token_segment,
    encode_segment,
)


@pytest.mark.parametrize(
    "raw",
    [b"", b"a", b"ab", b"abc", b"abcd", b"\x00\xff\xfe", bytes(range(256))],
)
def test_roundtrip_every_padding_class(raw: bytes) -> None:
    assert decode_opaque_cursor_segment(encode_segment(raw)) == raw
    assert decode_signed_token_segment(encode_segment(raw)) == raw


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
        decode_signed_token_segment(non_canonical)


def test_non_string_input_is_rejected() -> None:
    for bad in (None, 123, b"bytes"):
        with pytest.raises(Base64SegmentError):
            decode_opaque_cursor_segment(bad)  # type: ignore[arg-type]


def test_error_is_a_valueerror_so_existing_handlers_still_catch_it() -> None:
    """Call sites currently catch ValueError/binascii.Error; migration must not slip past."""
    assert issubclass(Base64SegmentError, ValueError)
