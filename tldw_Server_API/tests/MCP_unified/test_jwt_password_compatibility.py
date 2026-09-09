"""Preserve MCP's Passlib 1.7.4 password format across the libpass migration."""

# The fresh-process check runs fixed code in the active test interpreter.
import base64
import hashlib
import subprocess  # nosec B404
import sys

import pytest
from passlib.exc import PasswordSizeError

from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import JWTManager

pytestmark = pytest.mark.unit

# Synthetic fixtures generated with the original Passlib 1.7.4 dependency.
LEGACY_ASCII_HASH = "$pbkdf2-sha256$200000$ca5V6l2rFWJMiXEOodR6jw$sZ8cLw.i48gjGodV1fj23cODyh17rYb2rA0FMiwTmyw"
LEGACY_HASHES = [
    pytest.param(
        "synthetic-password-123!",
        LEGACY_ASCII_HASH,
        id="current-rounds",
    ),
    pytest.param(
        "café-密碼-🦦",
        "$pbkdf2-sha256$200000$VGqNsfa.t9YawzhnDKE0Bg$zVcoqib2k5Lf1TrEQ7s2.EPHmVUHhOojePZB.N4m9G0",
        id="unicode",
    ),
    pytest.param(
        "synthetic-historical-hash",
        "$pbkdf2-sha256$1$$vNufW7s4SrI9fY0l6oMX4xeknz4Os9f3bh5HsOdU9SU",
        id="legacy-empty-salt",
    ),
    pytest.param(
        "synthetic-historical-hash",
        "$pbkdf2-sha256$29000$AA$uNfd1Kp3O1sQhT1CtA0u2NZtvzdUukgqTg7Ot/Ecjmc",
        id="legacy-rounds",
    ),
    pytest.param(
        "synthetic-historical-hash",
        "$pbkdf2-sha256$300000$AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8$jy6xBDleQZEgIzF5ikVXD9w/p4DWaMHZVqA5gxlwVd4",
        id="higher-rounds",
    ),
]


def test_password_dependency_does_not_import_deprecated_crypt() -> None:
    """Check in a fresh interpreter so pytest's prior imports cannot hide it."""
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "from passlib.context import CryptContext; import sys; assert 'crypt' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("password", "encoded"), LEGACY_HASHES)
@pytest.mark.parametrize("correct_password", [True, False])
def test_verify_original_passlib_hash(password: str, encoded: str, correct_password: bool) -> None:
    """Stored rounds and encodings remain usable, with mismatches rejected."""
    manager = JWTManager.__new__(JWTManager)
    attempted = password if correct_password else password + "-wrong"
    assert manager.verify_password(attempted, encoded) is correct_password


@pytest.mark.parametrize(
    "password",
    ["", "synthetic-password", "café-密碼-🦦", "cafe\u0301", "null\0byte", "a" * 4096, "é" * 4096, b"\xff\0bytes"],
    ids=["empty", "ascii", "unicode", "combining", "nul", "max-ascii", "max-unicode", "bytes"],
)
def test_new_hash_retains_pbkdf2_contract(password: str | bytes) -> None:
    """New hashes retain their cost, salt size and independently verified digest."""
    manager = JWTManager.__new__(JWTManager)
    encoded = manager.hash_password(password)
    _, scheme, rounds, salt_text, checksum = encoded.split("$")
    salt = base64.b64decode(salt_text.replace(".", "+") + "=" * (-len(salt_text) % 4))
    secret = password.encode("utf-8") if isinstance(password, str) else password
    expected = (
        base64.b64encode(hashlib.pbkdf2_hmac("sha256", secret, salt, 200000)).decode().rstrip("=").replace("+", ".")
    )
    assert (scheme, rounds, len(salt), checksum) == ("pbkdf2-sha256", "200000", 16, expected)
    assert manager.verify_password(password, encoded) is True


def test_password_hashes_use_fresh_salts() -> None:
    """Repeated hashing of the same synthetic password must not reuse a salt."""
    manager = JWTManager.__new__(JWTManager)
    assert (
        manager.hash_password("synthetic-repeat").split("$")[3]
        != manager.hash_password("synthetic-repeat").split("$")[3]
    )


@pytest.mark.parametrize("password", ["a" * 4097, "é" * 4097], ids=["ascii", "unicode"])
def test_password_length_limit_is_preserved(password: str) -> None:
    """Keep the existing character limit for new hashes."""
    with pytest.raises(PasswordSizeError):
        JWTManager.__new__(JWTManager).hash_password(password)


@pytest.mark.parametrize(
    "encoded",
    [
        None,
        1,
        "",
        "not-a-hash",
        LEGACY_ASCII_HASH.replace("pbkdf2-sha256", "argon2id"),
        LEGACY_ASCII_HASH.replace("200000", "0"),
        LEGACY_ASCII_HASH.replace("200000", "-1"),
        LEGACY_ASCII_HASH.replace("200000", "0200000"),
        LEGACY_ASCII_HASH.replace("200000", "not-a-number"),
        LEGACY_ASCII_HASH.replace("ca5V6l2rFWJMiXEOodR6jw", "!"),
        LEGACY_ASCII_HASH.rsplit("$", 1)[0] + "$bad!",
        LEGACY_ASCII_HASH[:-1],
        LEGACY_ASCII_HASH + "$extra",
    ],
)
def test_malformed_password_hash_fails_closed(encoded: str | int | None) -> None:
    """Invalid stored data returns False through the public MCP helper."""
    assert JWTManager.__new__(JWTManager).verify_password("synthetic-password-123!", encoded) is False
