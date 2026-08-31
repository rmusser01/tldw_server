"""
crypto_utils.py

Shared cryptographic helpers for AuthNZ components.

Currently exposes a uniform HMAC key derivation routine to avoid drift
between JWTService, APIKeyManager, CSRF, and SessionManager.
"""

from __future__ import annotations

import functools
import hashlib
import os

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

_HMAC_KDF_SALT_LEGACY = b"tldw_authnz_hmac_kdf_v1"
_HMAC_KDF_SALT_PREFIX = b"tldw_authnz_hmac_kdf_v2:"
_HMAC_KDF_ITERATIONS = 100_000
_HMAC_KDF_DKLEN = 32


def _ensure_secret_bytes(secret: str | None) -> bytes | None:
    if secret is None:
        return None
    if isinstance(secret, bytes):
        return secret
    return str(secret).encode("utf-8")


def _derive_hmac_kdf_salt(source: bytes) -> bytes:
    # Derive a per-secret salt from a domain-separated fingerprint of the source.
    return hashlib.sha256(_HMAC_KDF_SALT_PREFIX + hashlib.sha256(source).digest()).digest()


# Derived keys are memoized on the secret material they come from.
#
# The KDF is deliberately expensive -- 100,000 PBKDF2 rounds -- but its input
# here is server configuration (SINGLE_USER_API_KEY, API_KEY_PEPPER, JWT
# secrets), not anything a caller supplies, and API key validation re-derived it
# several times per request:
#
#   POST /api/v1/chat/completions   8 derivations, 110 ms of a 167 ms request
#   POST /api/v1/embeddings         6 derivations,  82 ms of a 107 ms request
#
# Caching changes no security property. The stretching exists to make the server
# secret expensive to recover from a leaked fingerprint, and an attacker gains
# no additional attempts from a cache the server keeps of its own configuration.
# The per-request secret -- the presented API key -- is the HMAC *message*, not
# the KDF input, and is unaffected.
#
# Keying on the source bytes means a rotated or reconfigured secret is a
# different key and misses the cache, so there is nothing to invalidate. The
# bound keeps rotation or per-test settings churn from growing it without limit.
_HMAC_KEY_CACHE_MAXSIZE = 64


@functools.lru_cache(maxsize=_HMAC_KEY_CACHE_MAXSIZE)
def _derive_hmac_key_cached(source: bytes, legacy: bool) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        source,
        _HMAC_KDF_SALT_LEGACY if legacy else _derive_hmac_kdf_salt(source),
        _HMAC_KDF_ITERATIONS,
        dklen=_HMAC_KDF_DKLEN,
    )


def _derive_hmac_key_from_source(source: bytes, *, legacy: bool = False) -> bytes:
    return _derive_hmac_key_cached(source, legacy)


def reset_hmac_key_cache() -> None:
    """Drop memoized derived keys. For tests that rotate secrets in place."""
    _derive_hmac_key_cached.cache_clear()


def derive_hmac_key_from_source(raw: str | bytes, *, legacy: bool = False) -> bytes:
    """Derive a 32-byte HMAC key from raw secret material using the configured KDF.

    Args:
        raw: Secret material to derive from.
        legacy: When True, use the fixed legacy salt for backward compatibility.
    """
    source = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
    return _derive_hmac_key_from_source(source, legacy=legacy)


def derive_hmac_key(settings: Settings | None = None) -> bytes:
    """Derive a 32-byte HMAC key from configured secrets.

    Order of preference:
    - single_user: derive from SHA256(SINGLE_USER_API_KEY)
    - otherwise: API_KEY_PEPPER if set
    - otherwise: JWT secrets/keys (HS or RS/ES)
    - fallback: only in explicit test contexts

    The returned key is derived using PBKDF2-HMAC-SHA256 to produce
    a uniform 32-byte key suitable for HMAC-SHA256.
    """
    keys = derive_hmac_key_candidates(settings)
    if not keys:
        raise ValueError("derive_hmac_key_candidates returned no usable keys")
    return keys[0]


def derive_hmac_key_candidates(settings: Settings | None = None) -> list[bytes]:
    """Return ordered HMAC key candidates derived from configured secrets.

    The first item represents the *current* secret material. Subsequent entries
    capture legacy/secondary secrets that should remain valid during rotations.

    Important: Public keys are intentionally excluded from HMAC/encryption key
    derivation to avoid using non-secret material as cryptographic input.
    Legacy fixed-salt derivations are retained as secondary candidates to
    preserve verification of stored hashes created before the per-secret salt.
    """
    s = settings or get_settings()
    auth_mode = getattr(s, "AUTH_MODE", "single_user")

    # Detect pytest context and known deterministic JWT secret used only for testing
    test_mode_env = is_test_mode()
    allow_test_fallback = test_mode_env or env_flag_enabled("TLDW_ALLOW_TEST_FALLBACK_KEYS")
    pytest_active = os.getenv("PYTEST_CURRENT_TEST") is not None
    in_test_context = test_mode_env or pytest_active
    test_secret_env = os.getenv("JWT_SECRET_TEST_KEY", "test-secret-jwt-key-please-change-1234567890")

    digest_sources: list[bytes] = []
    seen: set[bytes] = set()

    def add_source(raw: str | None, *, prehash: bool = False) -> None:
        if not raw:
            return
        data = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
        if prehash:
            data = hashlib.sha256(data).digest()
        if data in seen:
            return
        seen.add(data)
        digest_sources.append(data)

    # Single-user mode prefers the configured API key (double hashed for parity with legacy logic)
    if auth_mode == "single_user" and getattr(s, "SINGLE_USER_API_KEY", None):
        add_source(s.SINGLE_USER_API_KEY, prehash=True)
        # Allow optional pepper override afterwards
        add_source(getattr(s, "API_KEY_PEPPER", None))
    else:
        # Multi-user (or single-user without explicit key): enforce real secret material
        add_source(getattr(s, "API_KEY_PEPPER", None))
        # If only a public key is configured and the JWT secret was auto-filled by
        # the test fallback, ignore that secret and use the deterministic fallback
        # later to keep test behavior stable.
        jwt_secret_candidate = getattr(s, "JWT_SECRET_KEY", None)
        only_public_key = bool(getattr(s, "JWT_PUBLIC_KEY", None)) and not getattr(s, "JWT_PRIVATE_KEY", None)
        auto_test_secret = in_test_context and jwt_secret_candidate == test_secret_env
        placeholder_test_secret = jwt_secret_candidate == "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS"
        if not (only_public_key and (auto_test_secret or (in_test_context and placeholder_test_secret))):
            add_source(jwt_secret_candidate)
        add_source(getattr(s, "JWT_PRIVATE_KEY", None))

    # Secondary / legacy material to support key rotations
    add_source(getattr(s, "JWT_SECONDARY_SECRET", None))
    add_source(getattr(s, "JWT_SECONDARY_PRIVATE_KEY", None))
    # Note: secondary public keys are also excluded by design

    if not digest_sources:
        # Allow fallback only with explicit test-mode intent to prevent production misuse.
        if not allow_test_fallback:
            raise ValueError(
                "derive_hmac_key could not locate a configured secret. "
                "Set API_KEY_PEPPER (recommended) or provide JWT_SECRET_KEY / JWT_PRIVATE_KEY. "
                "For explicit test-only fallback, set TEST_MODE=1."
            )
        # SECURITY: Additional production guard - never use fallback in production environment
        environment = os.getenv("ENVIRONMENT", "").strip().lower()
        prod_flag = env_flag_enabled("tldw_production")
        if environment in {"production", "prod"} or prod_flag:
            raise ValueError(
                "CRITICAL: Test fallback secret cannot be used in production environment. "
                "Configure API_KEY_PEPPER or JWT_SECRET_KEY for production use."
            )
        # Log warning about using deterministic fallback key
        logger.warning(
            "Using deterministic test fallback HMAC key. "
            "This is only safe in test environments. "
            "Configure API_KEY_PEPPER or JWT_SECRET_KEY for production."
        )
        digest_sources.append(b"tldw_default_api_key_hmac")

    # Derive uniform 32-byte HMAC keys from each material using PBKDF2-HMAC-SHA256.
    # This is intentionally computationally expensive to harden low-entropy secrets
    # (for example, human-chosen API keys) against brute-force attacks.
    keys: list[bytes] = []
    for source in digest_sources:
        hashed = _derive_hmac_key_from_source(source)
        if hashed not in keys:
            keys.append(hashed)
        legacy = _derive_hmac_key_from_source(source, legacy=True)
        if legacy not in keys:
            keys.append(legacy)
    return keys


__all__ = [
    "derive_hmac_key",
    "derive_hmac_key_candidates",
    "derive_hmac_key_from_source",
    "reset_hmac_key_cache",
]
