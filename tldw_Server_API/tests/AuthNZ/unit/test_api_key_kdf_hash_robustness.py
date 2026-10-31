"""verify_kdf_hash must return a bool for any stored value, never raise.

It is called on the authentication path (key_resolution.py:93, api_key_manager.py:526)
and both callers expect a bool. Two inputs made it raise instead:

* an empty final segment -- "scheme$iters$salt$" splits cleanly into four parts and
  decodes to b"", so pbkdf2_hmac(..., dklen=0) raised ValueError from OUTSIDE the try;
* an unbounded iteration count, which is a hang rather than a False.

Both mean a 500 on the auth path rather than a 401, for every request presenting that key.
"""

from __future__ import annotations

import base64

import pytest

from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
    API_KEY_KDF_SCHEME,
    kdf_hash_api_key,
    verify_kdf_hash,
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_happy_path_still_verifies() -> None:
    encoded = kdf_hash_api_key("correct-horse")
    assert verify_kdf_hash("correct-horse", encoded) is True
    assert verify_kdf_hash("wrong", encoded) is False


@pytest.mark.parametrize(
    "encoded",
    [
        f"{API_KEY_KDF_SCHEME}$210000${_b64(b'salt')}$",      # empty derived -> dklen=0
        f"{API_KEY_KDF_SCHEME}$210000$${_b64(b'x')}",          # empty salt
        f"{API_KEY_KDF_SCHEME}$210000$$",                      # both empty
    ],
)
def test_empty_segments_return_false_rather_than_raising(encoded: str) -> None:
    assert verify_kdf_hash("any-key", encoded) is False


@pytest.mark.parametrize(
    "encoded",
    [
        f"{API_KEY_KDF_SCHEME}$210000$!!!!${_b64(b'x')}",      # out-of-alphabet salt
        f"{API_KEY_KDF_SCHEME}$210000${_b64(b'salt')}$!!!!",   # out-of-alphabet derived
    ],
)
def test_malformed_base64_returns_false(encoded: str) -> None:
    assert verify_kdf_hash("any-key", encoded) is False


@pytest.mark.parametrize("iterations", ["0", "-1", "99999999999", "1e9"])
def test_out_of_range_iterations_return_false_without_hanging(iterations: str) -> None:
    encoded = f"{API_KEY_KDF_SCHEME}${iterations}${_b64(b'salt')}${_b64(b'x' * 32)}"
    assert verify_kdf_hash("any-key", encoded) is False


def test_garbage_shapes_return_false() -> None:
    for encoded in ("", "not-a-hash", f"{API_KEY_KDF_SCHEME}$only-two", "$$$"):
        assert verify_kdf_hash("any-key", encoded) is False
