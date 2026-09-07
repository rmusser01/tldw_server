"""Keep the existing token contract without the vulnerable JOSE dependency."""

import importlib
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

pytestmark = pytest.mark.unit
KEY = "migration-test-key-at-least-32-bytes"
# Produced with python-jose 3.5.0 before migration; the key is public test data.
LEGACY_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiI0MiIsImlhdCI6NDEwMjQ0NDgwMCwiZXhwIjo0MTAyNDQ4NDAwLCJqdGkiOiJsZWdhY3kiLCJ0eXBlIjoiYWNjZXNzIn0."
    "wxm8P42memv-GfRwg4KZO0v6V5HLqM9uBChPMxP-EpI"
)


def _compat():
    return importlib.import_module("tldw_Server_API.app.core.Utils.jwt_compat")


def test_pre_migration_token_with_future_iat_remains_valid():
    assert _compat().decode(LEGACY_TOKEN, KEY, algorithms=["HS256"])["sub"] == "42"


@pytest.mark.parametrize(
    "claims",
    [
        {"iat": "bad"},
        {"iat": None},
        {"exp": "bad"},
        {"nbf": "bad"},
        {"exp": None},
        {"nbf": None},
        {"iat": "bad", "exp": 0},
        {"at_hash": "unverifiable"},
        {"aud": "wrong"},
        {"aud": [17]},
        {"iss": "wrong"},
        {"sub": 42},
        {"jti": 42},
        {"nbf": 4102444800},
    ],
)
def test_invalid_claims_keep_claim_error_mapping(claims):
    compat = _compat()
    token = pyjwt.encode(claims, KEY, algorithm="HS256")
    with pytest.raises(compat.JWTClaimsError):
        compat.decode(
            token, KEY, algorithms=["HS256"], audience="expected", issuer="expected" if "iss" in claims else None
        )


def test_absent_optional_audience_keeps_legacy_acceptance():
    token = pyjwt.encode({"sub": "42"}, KEY, algorithm="HS256")
    assert _compat().decode(token, KEY, algorithms=["HS256"], audience="expected")["sub"] == "42"


def test_expired_signature_keeps_specific_error():
    compat = _compat()
    token = pyjwt.encode({"exp": int(time.time()) - 60}, KEY, algorithm="HS256")
    with pytest.raises(compat.ExpiredSignatureError):
        compat.decode(token, KEY, algorithms=["HS256"])


def test_signature_and_algorithm_allowlist_remain_enforced():
    compat = _compat()
    with pytest.raises(compat.JWTError):
        compat.decode(LEGACY_TOKEN, KEY, algorithms=["RS256"])
    with pytest.raises(compat.JWTError):
        compat.decode(LEGACY_TOKEN, "different-key-with-at-least-32-bytes", algorithms=["HS256"])


def test_bad_signature_is_not_reclassified_from_untrusted_claims():
    token = pyjwt.encode({"exp": "bad"}, KEY, algorithm="HS256")
    with pytest.raises(pyjwt.InvalidSignatureError):
        _compat().decode(token, "different-key-with-at-least-32-bytes", algorithms=["HS256"])


def test_malformed_token_keeps_decode_error():
    with pytest.raises(pyjwt.DecodeError):
        _compat().decode("not-a-token", KEY, algorithms=["HS256"])


@pytest.mark.parametrize("algorithm", ["RS256", "ES256"])
def test_oidc_dictionary_jwk_is_verified_with_allowed_algorithm(algorithm):
    key = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if algorithm == "RS256"
        else ec.generate_private_key(ec.SECP256R1())
    )
    jwk = pyjwt.get_algorithm_by_name(algorithm).to_jwk(key.public_key(), as_dict=True)
    token = pyjwt.encode({"sub": "42", "aud": "client", "iss": "issuer"}, key, algorithm=algorithm)
    assert _compat().decode(token, jwk, algorithms=[algorithm], audience="client", issuer="issuer")["sub"] == "42"


def test_unverified_metadata_does_not_validate_expiry_or_claim_types():
    token = pyjwt.encode({"exp": 0, "jti": 17, "iat": "bad"}, KEY, algorithm="HS256")
    assert _compat().get_unverified_claims(token) == {"exp": 0, "jti": 17, "iat": "bad"}


def test_locked_graph_contains_no_vulnerable_jose_fallback():
    from pathlib import Path

    import tomllib

    packages = tomllib.loads(Path("uv.lock").read_text())["package"]
    assert not {"ecdsa", "python-jose"}.intersection(item["name"] for item in packages)
