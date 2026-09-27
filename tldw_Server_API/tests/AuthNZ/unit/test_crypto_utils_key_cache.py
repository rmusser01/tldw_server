"""Contracts for memoized HMAC key derivation.

The KDF is deliberately expensive -- 100,000 PBKDF2 rounds -- and API key
validation re-derived it several times per request. Memoizing it is only safe
while it stays indistinguishable from deriving every time, so these pin the
properties that make it so.
"""

from __future__ import annotations

import hashlib

import pytest

from tldw_Server_API.app.core.AuthNZ import crypto_utils
from tldw_Server_API.app.core.AuthNZ.crypto_utils import (
    derive_hmac_key_from_source,
    reset_hmac_key_cache,
)


def _uncached(source: bytes, *, legacy: bool = False) -> bytes:
    """Derive without touching the cache, the way the code used to."""
    salt = (
        crypto_utils._HMAC_KDF_SALT_LEGACY
        if legacy
        else crypto_utils._derive_hmac_kdf_salt(source)
    )
    return hashlib.pbkdf2_hmac(
        "sha256",
        source,
        salt,
        crypto_utils._HMAC_KDF_ITERATIONS,
        dklen=crypto_utils._HMAC_KDF_DKLEN,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_hmac_key_cache()
    yield
    reset_hmac_key_cache()


@pytest.mark.unit
@pytest.mark.parametrize("legacy", [False, True])
def test_cached_key_matches_deriving_every_time(legacy: bool) -> None:
    """The whole point: caching must not change the derived key."""
    secret = b"a-server-secret"

    first = derive_hmac_key_from_source(secret, legacy=legacy)
    second = derive_hmac_key_from_source(secret, legacy=legacy)

    assert first == _uncached(secret, legacy=legacy)
    assert second == first


@pytest.mark.unit
def test_different_secrets_derive_different_keys() -> None:
    """A cache that collided across secrets would be a key-confusion bug."""
    assert derive_hmac_key_from_source(b"one") != derive_hmac_key_from_source(b"two")


@pytest.mark.unit
def test_legacy_salt_is_a_separate_entry() -> None:
    """Same secret, different salt mode, different key."""
    secret = b"a-server-secret"

    assert derive_hmac_key_from_source(secret) != derive_hmac_key_from_source(
        secret, legacy=True
    )


@pytest.mark.unit
def test_cache_is_keyed_by_fingerprint_not_by_the_secret() -> None:
    """The cache must not be another long-lived copy of the configured secret."""
    secret = b"a-server-secret"
    derive_hmac_key_from_source(secret)

    keys = list(crypto_utils._HMAC_KEY_CACHE)
    assert (hashlib.sha256(secret).digest(), False) in keys
    assert all(secret != key[0] for key in keys), "the raw secret is a cache key"


@pytest.mark.unit
def test_cache_is_bounded() -> None:
    """Secret rotation and per-test settings churn must not grow it forever."""
    for index in range(crypto_utils._HMAC_KEY_CACHE_MAXSIZE * 4):
        derive_hmac_key_from_source(f"secret-{index}".encode())

    assert len(crypto_utils._HMAC_KEY_CACHE) == crypto_utils._HMAC_KEY_CACHE_MAXSIZE


@pytest.mark.unit
def test_reset_forces_the_next_derivation_to_run_again() -> None:
    """Tests that rotate a secret in place depend on this."""
    secret = b"a-server-secret"
    derive_hmac_key_from_source(secret)
    assert crypto_utils._HMAC_KEY_CACHE

    reset_hmac_key_cache()

    assert not crypto_utils._HMAC_KEY_CACHE
    assert derive_hmac_key_from_source(secret) == _uncached(secret)
