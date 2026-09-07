"""
RS256 unit tests for JWTService: roundtrip and dual-key fallback.
"""

import pytest
from datetime import datetime

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings


def _gen_rsa_keypair_pem():


    """Generate an RSA keypair and return (private_pem, public_pem) as UTF-8 strings."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


class TestJWTServiceRS256:
    def _rs_settings(self, private_pem: str, public_pem: str, **overrides):
        return Settings(
            AUTH_MODE="multi_user",
            JWT_ALGORITHM="RS256",
            JWT_PRIVATE_KEY=private_pem,
            JWT_PUBLIC_KEY=public_pem,
            # Token lifetimes
            ACCESS_TOKEN_EXPIRE_MINUTES=30,
            REFRESH_TOKEN_EXPIRE_DAYS=7,
            # Minimal other settings
            PASSWORD_MIN_LENGTH=8,
            PASSWORD_REQUIRE_UPPERCASE=True,
            PASSWORD_REQUIRE_LOWERCASE=True,
            PASSWORD_REQUIRE_DIGIT=True,
            PASSWORD_REQUIRE_SPECIAL=False,
            REGISTRATION_ENABLED=True,
            REGISTRATION_REQUIRE_CODE=False,
            REGISTRATION_CODES=[],
            DEFAULT_USER_ROLE="user",
            DEFAULT_STORAGE_QUOTA_MB=1000,
            EMAIL_VERIFICATION_REQUIRED=False,
            CORS_ORIGINS=["*"],
            API_PREFIX="/api/v1",
            **overrides,
        )

    def test_rs256_access_token_roundtrip(self):

        priv, pub = _gen_rsa_keypair_pem()
        svc = JWTService(settings=self._rs_settings(priv, pub))

        token = svc.create_access_token(user_id=42, username="alice", role="user")
        payload = svc.decode_access_token(token)

        assert payload["sub"] == "42"
        assert payload["username"] == "alice"
        assert payload["role"] == "user"
        assert payload["type"] == "access"
        assert "exp" in payload and "iat" in payload and "jti" in payload

    def test_rs256_dual_key_fallback_decode(self):

             # Old keypair used to sign an existing token
        old_priv, old_pub = _gen_rsa_keypair_pem()
        svc_old = JWTService(settings=self._rs_settings(old_priv, old_pub))
        old_token = svc_old.create_access_token(user_id=7, username="bob", role="user")

        # New keypair configured as primary, but accept old public via secondary for rotation period
        new_priv, new_pub = _gen_rsa_keypair_pem()
        svc_new = JWTService(
            settings=self._rs_settings(new_priv, new_pub, JWT_SECONDARY_PUBLIC_KEY=old_pub)
        )

        payload = svc_new.decode_access_token(old_token)
        assert payload["sub"] == "7"
        assert payload["username"] == "bob"
        assert payload["type"] == "access"

    def test_rs256_private_key_only_configuration_verifies_tokens(self):
        private_pem, _ = _gen_rsa_keypair_pem()
        svc = JWTService(settings=self._rs_settings(private_pem, ""))
        token = svc.create_access_token(user_id=42, username="alice", role="user")

        assert svc.decode_access_token(token)["sub"] == "42"

    def test_rs256_issuer_audience_enforced(self):

        priv, pub = _gen_rsa_keypair_pem()
        svc = JWTService(settings=self._rs_settings(priv, pub, JWT_ISSUER="tldw.rs", JWT_AUDIENCE="tldw.clients"))
        token = svc.create_access_token(user_id=5, username="eve", role="user")
        assert svc.decode_access_token(token)["sub"] == "5"

        # Create mismatched audience token using a separate service
        svc_bad = JWTService(settings=self._rs_settings(priv, pub, JWT_ISSUER="tldw.rs", JWT_AUDIENCE="wrong.aud"))
        bad = svc_bad.create_access_token(user_id=5, username="eve", role="user")
        import pytest
        with pytest.raises(Exception):
            svc.decode_access_token(bad)
