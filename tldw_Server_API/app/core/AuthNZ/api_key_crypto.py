from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from tldw_Server_API.app.core.Utils.base64url import decode_signed_token_segment

API_KEY_PREFIX = "tldw_"
API_KEY_SEPARATOR = "."
API_KEY_ID_BYTES = 6  # 12 hex chars
API_KEY_SECRET_BYTES = 32
API_KEY_KDF_SCHEME = "pbkdf2_sha256"
API_KEY_KDF_ITERATIONS = 210_000
API_KEY_KDF_SALT_BYTES = 16


def generate_api_key_id() -> str:
    """Generate a short key identifier embedded in new API keys."""
    return secrets.token_hex(API_KEY_ID_BYTES)


def generate_api_key_secret() -> str:
    """Generate a high-entropy secret for API keys."""
    return secrets.token_urlsafe(API_KEY_SECRET_BYTES)


def format_api_key(key_id: str, secret: str) -> str:
    """Format a full API key with prefix and delimiter."""
    return f"{API_KEY_PREFIX}{key_id}{API_KEY_SEPARATOR}{secret}"


def parse_api_key(api_key: str) -> tuple[str, str] | None:
    """Extract (key_id, secret) from the new API key format."""
    if not api_key or not api_key.startswith(API_KEY_PREFIX):
        return None

    remainder = api_key[len(API_KEY_PREFIX):]
    if API_KEY_SEPARATOR not in remainder:
        return None

    key_id, secret = remainder.split(API_KEY_SEPARATOR, 1)
    if not key_id or not secret:
        return None

    normalized = key_id.lower()
    if not (8 <= len(normalized) <= 32):
        return None
    if not all(ch in "0123456789abcdef" for ch in normalized):
        return None

    return normalized, secret


def kdf_hash_api_key(
    api_key: str,
    *,
    salt: bytes | None = None,
    iterations: int = API_KEY_KDF_ITERATIONS,
) -> str:
    """Return a PBKDF2-HMAC-SHA256 encoded hash string for an API key."""
    if salt is None:
        salt = secrets.token_bytes(API_KEY_KDF_SALT_BYTES)
    if not api_key:
        raise ValueError("api_key is required for KDF hashing")

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        api_key.encode("utf-8"),
        salt,
        iterations,
        dklen=32,
    )
    salt_b64 = _b64encode(salt)
    derived_b64 = _b64encode(derived)
    return f"{API_KEY_KDF_SCHEME}${iterations}${salt_b64}${derived_b64}"


# Bounds for a stored hash. Generous relative to what kdf_hash_api_key produces
# (a 32-byte digest and a 16-byte salt), tight enough to refuse a corrupted row.
_MAX_KDF_ITERATIONS = 10_000_000
_MAX_HASH_SEGMENT_LEN = 512


def verify_kdf_hash(api_key: str, encoded: str) -> bool:
    """Verify an API key against a PBKDF2-HMAC-SHA256 encoded hash."""
    try:
        scheme, iterations_raw, salt_b64, derived_b64 = encoded.split("$", 3)
    except ValueError:
        return False

    if scheme != API_KEY_KDF_SCHEME:
        return False

    try:
        iterations = int(iterations_raw)
    except ValueError:
        return False

    # A stored iteration count is attacker-influenced only via a corrupted row, but an
    # unbounded one is a hang rather than a False, and a non-positive one is nonsense.
    if not 1 <= iterations <= _MAX_KDF_ITERATIONS:
        return False

    try:
        salt = _b64decode(salt_b64)
        expected = _b64decode(derived_b64)
    except Exception:
        return False

    # dklen=0 raises. Guarding here rather than relying on the try below keeps the
    # intent explicit: a stored hash with an empty digest is malformed, not an error.
    if not salt or not expected:
        return False

    try:
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            api_key.encode("utf-8"),
            salt,
            iterations,
            dklen=len(expected),
        )
    except (ValueError, OverflowError, MemoryError):
        # Previously OUTSIDE the try: a malformed stored hash escaped verify_kdf_hash as
        # an exception, so the auth path returned 500 instead of 401 for every request
        # presenting that key.
        return False
    return hmac.compare_digest(derived, expected)


def is_kdf_hash(encoded: str) -> bool:
    """Return True when the hash string uses the KDF scheme."""
    return isinstance(encoded, str) and encoded.startswith(f"{API_KEY_KDF_SCHEME}$")


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(encoded: str) -> bytes:
    """Decode one segment of a stored API-key hash.

    Uses the shared signed-token decoder: alphabet-validated, length-bounded, and
    canonical-form enforcing. The previous bare urlsafe_b64decode silently DISCARDED
    out-of-alphabet characters, so a corrupted segment decoded to different valid bytes
    instead of raising.
    """
    return decode_signed_token_segment(encoded, max_encoded_len=_MAX_HASH_SEGMENT_LEN)


__all__ = [
    "API_KEY_PREFIX",
    "API_KEY_SEPARATOR",
    "API_KEY_KDF_SCHEME",
    "API_KEY_KDF_ITERATIONS",
    "generate_api_key_id",
    "generate_api_key_secret",
    "format_api_key",
    "parse_api_key",
    "kdf_hash_api_key",
    "verify_kdf_hash",
    "is_kdf_hash",
]
