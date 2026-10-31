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

**Two trust classes, two entry points.** The sites split into opaque pagination
cursors and HMAC-signed tokens. A single flattened helper -- especially one growing a
``verify=False`` default -- would be worse than the duplication, because it would let a
caller reach for the unsigned path where a signed one was required:

``decode_opaque_cursor_segment``
    An opaque pagination cursor the server minted and the client echoes back. Bounded
    and alphabet-validated. Non-canonical encodings are accepted: nothing keys off the
    cursor string itself.

``verify_signed_token``
    A ``<payload>.<signature>`` HMAC-SHA256 token. REQUIRES the key and checks the
    signature with ``hmac.compare_digest`` before returning the payload, so it cannot be
    used to read a token without verifying it. Both segments must be canonical, so two
    distinct strings cannot carry the same signed bytes.

``decode_canonical_segment`` is the strict single-segment decoder for stored values
that are not tokens (e.g. the salt and digest of an API-key KDF hash). It verifies
nothing; never use it to read a client-supplied token.
"""

import base64
import hmac

__all__ = [
    "Base64SegmentError",
    "SignatureMismatchError",
    "decode_canonical_segment",
    "decode_opaque_cursor_segment",
    "encode_segment",
    "verify_signed_token",
]

# Generous by default: every observed caller's payload is far below this. A caller with
# a tighter contract passes its own bound.
DEFAULT_MAX_ENCODED_LEN = 4096

_ALTCHARS = b"-_"


class Base64SegmentError(ValueError):
    """A base64url segment was malformed, oversized, or non-canonical."""


class SignatureMismatchError(Base64SegmentError):
    """A well-formed signed token whose signature does not verify under the key."""


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


def decode_canonical_segment(
    segment: str,
    *,
    max_encoded_len: int | None = None,
) -> bytes:
    """Decode one stored segment, rejecting non-canonical encodings.

    Verifies nothing. For a client-supplied token use ``verify_signed_token``.
    """
    return _decode(segment, require_canonical=True, max_encoded_len=max_encoded_len)


def verify_signed_token(
    token: str,
    key: bytes,
    *,
    max_encoded_len: int | None = None,
    sign_encoded_payload: bool = False,
) -> bytes:
    """Verify a ``<payload>.<signature>`` HMAC-SHA256 token and return the payload bytes.

    ``sign_encoded_payload`` selects what was MACed: the raw payload bytes (default) or
    the ASCII payload segment as it appears in the token.

    Raises ``TypeError`` for a missing key (a configuration error, not a bad token),
    ``SignatureMismatchError`` for a well-formed token that fails verification, and
    ``Base64SegmentError`` for anything malformed or larger than ``max_encoded_len``.
    """
    if not isinstance(key, bytes) or not key:
        raise TypeError("verify_signed_token requires a non-empty bytes key")
    if not isinstance(token, str):
        raise Base64SegmentError("token must be a string")
    limit = DEFAULT_MAX_ENCODED_LEN if max_encoded_len is None else max_encoded_len
    if len(token) > limit:
        raise Base64SegmentError(f"token exceeds {limit} encoded bytes")
    parts = token.split(".")
    if len(parts) != 2:
        raise Base64SegmentError("token must have exactly two segments")
    payload_segment, signature_segment = parts
    payload = _decode(payload_segment, require_canonical=True, max_encoded_len=limit)
    signature = _decode(signature_segment, require_canonical=True, max_encoded_len=limit)
    signed = payload_segment.encode("ascii") if sign_encoded_payload else payload
    if not hmac.compare_digest(signature, hmac.digest(key, signed, "sha256")):
        raise SignatureMismatchError("token signature does not verify")
    return payload
