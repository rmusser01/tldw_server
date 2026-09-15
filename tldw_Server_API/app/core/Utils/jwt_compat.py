"""PyJWT-backed validation preserving the server's existing JWT claim contract.

Only the operations used by the former python-jose consumers are exposed.
Signature, expiry, issuer, subject and JTI validation stay in PyJWT. Legacy
issued-at validation checks numeric form, not whether issuance is in the future;
an optional audience is validated only when the token contains one. No existing
consumer supplies an access token, so an at_hash claim must still be rejected.
"""

from collections.abc import Sequence
from typing import Any

import jwt as _jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt import ExpiredSignatureError, encode, get_unverified_header
from jwt import PyJWTError as JWTError
from jwt.exceptions import (
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidIssuedAtError,
    InvalidIssuerError,
    InvalidJTIError,
    InvalidSubjectError,
    MissingRequiredClaimError,
)

__all__ = [
    "JWTError",
    "JWTClaimsError",
    "ExpiredSignatureError",
    "encode",
    "decode",
    "get_unverified_header",
    "get_unverified_claims",
]


class JWTClaimsError(_jwt.InvalidTokenError):
    """A validly signed token has claims rejected by the application contract."""


def _validate_numeric_dates(claims: dict[str, Any]) -> None:
    """Keep malformed temporal claims in the application's claims-error category."""
    for name in ("iat", "nbf", "exp"):
        if name in claims:
            try:
                int(claims[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise JWTClaimsError(f"{name} claim must be an integer.") from exc


def decode(
    token: str,
    key: Any,
    *,
    algorithms: Sequence[str],
    audience: str | None = None,
    issuer: str | None = None,
) -> dict[str, Any]:
    """Verify using the caller's allowlist and preserve legacy claim semantics."""
    algorithm = get_unverified_header(token).get("alg")
    if algorithm not in algorithms:
        raise _jwt.InvalidAlgorithmError("The specified alg value is not allowed")
    if isinstance(key, dict):
        key = _jwt.PyJWK.from_dict(key, algorithm=algorithm).key
    if algorithm.startswith("RS"):
        key = _jwt.get_algorithm_by_name(algorithm).prepare_key(key)
        if isinstance(key, RSAPrivateKey):
            key = key.public_key()
    try:
        claims = _jwt.decode(
            token,
            key,
            algorithms=algorithms,
            issuer=issuer,
            options={"verify_iat": False, "verify_aud": False},
        )
    except _jwt.InvalidSignatureError:
        raise
    except (ExpiredSignatureError, _jwt.DecodeError, TypeError, OverflowError) as exc:
        # Rejection only: inspect metadata to preserve legacy error categories.
        # This path never returns claims or authorizes a token.
        try:
            rejected_claims = get_unverified_claims(token)
        except JWTError:
            raise exc from None
        _validate_numeric_dates(rejected_claims)
        raise
    except (
        InvalidAudienceError,
        InvalidIssuedAtError,
        InvalidIssuerError,
        InvalidJTIError,
        InvalidSubjectError,
        ImmatureSignatureError,
        MissingRequiredClaimError,
    ) as exc:
        raise JWTClaimsError(str(exc)) from exc
    _validate_numeric_dates(claims)
    if "aud" in claims:
        audiences = claims["aud"]
        if isinstance(audiences, str):
            audiences = [audiences]
        if not isinstance(audiences, list) or not all(isinstance(item, str) for item in audiences):
            raise JWTClaimsError("Invalid claim format in token")
        if audience not in audiences:
            raise JWTClaimsError("Invalid audience")
    if "at_hash" in claims:
        raise JWTClaimsError("No access_token provided to compare against at_hash claim.")
    return claims


def get_unverified_claims(token: str) -> dict[str, Any]:
    """Read routing/revocation metadata only; never use this for authorization."""
    return _jwt.decode(token, options={"verify_signature": False})
