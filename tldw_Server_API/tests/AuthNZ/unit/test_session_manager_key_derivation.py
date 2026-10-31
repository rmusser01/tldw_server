"""Session key derivation goes through the shared AuthNZ KDF (TASK-13325).

session_manager used to re-implement crypto_utils.derive_hmac_key_candidates with a
different salt, iteration count and source handling. It now uses the canonical
helper, and keeps the old derivation only as trailing rotation candidates.
"""

import base64
import hashlib
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key_candidates
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidSessionError
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager

_SETTINGS = {
    "single_user": SimpleNamespace(
        AUTH_MODE="single_user",
        SINGLE_USER_API_KEY="single-user-session-derivation-key",
        API_KEY_PEPPER="pepper-for-session-derivation-tests",
    ),
    "multi_user": SimpleNamespace(
        AUTH_MODE="multi_user",
        JWT_SECRET_KEY="multi-user-session-derivation-secret-abcdefghij",
        JWT_SECONDARY_SECRET="previous-session-derivation-secret-abcdefghij",
    ),
}


def _manager(settings: SimpleNamespace) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.settings = settings
    return manager


def _legacy_key(secret: str) -> bytes:
    """The pre-TASK-13325 derivation, restated independently of the module."""
    derived = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), b"session_encryption_salt_v1", 600_000, dklen=32)
    return base64.urlsafe_b64encode(derived)


@pytest.mark.parametrize("mode", sorted(_SETTINGS))
def test_session_keys_lead_with_the_canonical_candidates(mode: str) -> None:
    settings = _SETTINGS[mode]
    canonical = [base64.urlsafe_b64encode(key) for key in derive_hmac_key_candidates(settings)]

    derived = _manager(settings)._derive_secret_key_candidates()

    assert derived[: len(canonical)] == canonical


def _ready_manager(settings: SimpleNamespace) -> SessionManager:
    manager = _manager(settings)
    manager._fernet_candidates = [Fernet(key) for key in manager._derive_secret_key_candidates()]
    manager.cipher_suite = manager._fernet_candidates[0]
    return manager


def test_session_encrypted_under_the_old_derivation_still_decrypts() -> None:
    """The old key is no longer primary, so this only passes if decrypt_token walks on.

    It used to stop at the first candidate: Fernet raises InvalidToken, which the
    loop did not catch, so no trailing rotation candidate was ever reached.
    """
    settings = _SETTINGS["single_user"]
    manager = _ready_manager(settings)
    old_key = _legacy_key(settings.SINGLE_USER_API_KEY)
    stored = base64.urlsafe_b64encode(Fernet(old_key).encrypt(b"old-session-token")).decode("utf-8")

    assert manager._derive_secret_key_candidates()[0] != old_key
    assert manager.decrypt_token(stored) == "old-session-token"


def test_token_under_no_candidate_is_an_invalid_session() -> None:
    manager = _ready_manager(_SETTINGS["multi_user"])
    stored = base64.urlsafe_b64encode(Fernet(Fernet.generate_key()).encrypt(b"x")).decode("utf-8")

    with pytest.raises(InvalidSessionError):
        manager.decrypt_token(stored)
