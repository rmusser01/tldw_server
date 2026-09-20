from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EnvelopeAuthenticationError,
    EnvelopeCipher,
)


def test_every_encryption_uses_fresh_dek_and_96_bit_nonces() -> None:
    cipher = EnvelopeCipher(b"p" * 32, key_version=3)

    first = cipher.encrypt(b"profile content", b"object-a")
    second = cipher.encrypt(b"profile content", b"object-a")

    assert len(first.nonce) == len(first.wrapped_dek_nonce) == 12
    assert first.key_version == 3
    assert first.nonce != second.nonce
    assert first.wrapped_dek_nonce != second.wrapped_dek_nonce
    assert first.wrapped_dek != second.wrapped_dek
    assert cipher.decrypt(first, b"object-a") == b"profile content"


def test_envelope_is_bound_to_associated_data_without_leaking_plaintext() -> None:
    canary = b"PRIVATE-PROFILE-CANARY-CRYPTO"
    cipher = EnvelopeCipher(b"p" * 32)
    envelope = cipher.encrypt(canary, b"profile-a:record-a:v1")

    with pytest.raises(EnvelopeAuthenticationError) as caught:
        cipher.decrypt(envelope, b"profile-b:record-a:v1")

    assert canary not in str(caught.value).encode("utf-8")
    assert "authentication" in str(caught.value).lower()


@pytest.mark.parametrize("size", [0, 16, 31, 33])
def test_profile_encryption_key_must_be_exactly_32_bytes(size: int) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        EnvelopeCipher(b"x" * size)
