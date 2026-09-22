from __future__ import annotations

"""Encode and decode unpadded URL-safe base64 segments.

The review found this idiom written independently at 23 production sites across 22
files, in two notational spellings (``"=" * (-len(x) % 4)`` and
``"=" * ((4 - len(x) % 4) % 4)``) -- so a grep-based fix would have missed sites -- and
with strictness already diverged three ways:

* 5 sites decode with ``validate=True`` and explicit ``altchars``, rejecting
  out-of-alphabet input;
* 2 (both in ``endpoints/notes.py``) decode leniently and then re-encode and compare,
  rejecting non-canonical encodings by a different mechanism;
* 16 decode bare. ``urlsafe_b64decode`` without ``validate=True`` silently DISCARDS
  characters outside the alphabet rather than raising, so a tampered or truncated
  segment decodes to a different-but-valid byte string.

**Two entry points, deliberately.** The sites split into two trust classes, and a single
flattened helper -- especially one growing a ``verify=False`` default -- would be worse
than the duplication, because it would let a caller reach for the unsigned path where a
signed one was required:

``decode_opaque_cursor_segment``
    An opaque pagination cursor the server minted and the client echoes back. Bounded
    and alphabet-validated. Non-canonical encodings are accepted: nothing keys off the
    cursor string itself.

``decode_signed_token_segment``
    One segment of a token whose integrity is established by a signature the caller
    verifies separately. Additionally requires canonical form, so two distinct strings
    cannot decode to the same signed bytes.

Neither function verifies a signature. That stays with the caller that owns the key.
"""

import base64

__all__ = [
    "Base64SegmentError",
    "decode_opaque_cursor_segment",
    "decode_signed_token_segment",
    "encode_segment",
]

# Generous by default: every observed caller's payload is far below this. A caller with
# a tighter contract passes its own bound.
DEFAULT_MAX_ENCODED_LEN = 4096

_ALTCHARS = b"-_"


class Base64SegmentError(ValueError):
    """A base64url segment was malformed, oversized, or non-canonical."""


def encode_segment(raw: bytes) -> str:
    """Encode ``raw`` as unpadded URL-safe base64."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode(
    segment: str,
    *,
    require_canonical: bool,
    max_encoded_len: int | None,
) -> bytes:
    if not isinstance(segment, str):
        raise Base64SegmentError("segment must be a string")
    limit = DEFAULT_MAX_ENCODED_LEN if max_encoded_len is None else max_encoded_len
    if limit is not None and len(segment) > limit:
        # Bound BEFORE decoding: refusing early is what keeps an oversized segment from
        # being expanded into memory at all.
        raise Base64SegmentError(f"segment exceeds {limit} encoded bytes")

    padded = segment + ("=" * (-len(segment) % 4))
    try:
        # validate=True is the whole point: without it, out-of-alphabet characters are
        # silently discarded and a tampered segment decodes to different valid bytes.
        raw = base64.b64decode(padded.encode("ascii"), altchars=_ALTCHARS, validate=True)
    except Exception as exc:  # binascii.Error, UnicodeEncodeError, ValueError
        raise Base64SegmentError("segment is not valid base64url") from exc

    if require_canonical and encode_segment(raw) != segment:
        raise Base64SegmentError("segment is not in canonical base64url form")
    return raw


def decode_opaque_cursor_segment(
    segment: str,
    *,
    max_encoded_len: int | None = None,
) -> bytes:
    """Decode an opaque, unauthenticated pagination cursor segment."""
    return _decode(segment, require_canonical=False, max_encoded_len=max_encoded_len)


def decode_signed_token_segment(
    segment: str,
    *,
    max_encoded_len: int | None = None,
) -> bytes:
    """Decode one segment of a signed token.

    Enforces canonical form so two distinct strings cannot decode to the same signed
    bytes. Does NOT verify the signature: that belongs with whoever holds the key.
    """
    return _decode(segment, require_canonical=True, max_encoded_len=max_encoded_len)
