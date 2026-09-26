"""Regression guards for TASK-13289 and TASK-13297 (knowledge-QA share tokens).

TASK-13289 — `_get_knowledge_qa_share_signing_key` ended with

    fallback = (os.getenv("JWT_SECRET_KEY") or "knowledge_qa_share_link_default")

so when `KNOWLEDGE_QA_SHARE_LINK_SECRET` was unset, `derive_hmac_key()` raised one of
the broad `_CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS`, and `JWT_SECRET_KEY` was unset,
every share token was signed with a constant published in this open-source
repository. Anyone could mint a token the verifier accepts. The `@lru_cache(maxsize=1)`
meant a single transient derivation failure at first use pinned that weak key for the
rest of the process lifetime.

TASK-13297 — `_decode_knowledge_qa_share_token` decoded the *signature* segment
outside the `try` that guards the payload decode, so a malformed signature raised an
unhandled `binascii.Error` and the public, unauthenticated share route returned
HTTP 500 where 400 is correct.
"""

import base64
import hashlib
import hmac
import json

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat as chat_ep

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit

_LITERAL = b"knowledge_qa_share_link_default"


@pytest.fixture(autouse=True)
def _clear_key_cache(monkeypatch: pytest.MonkeyPatch):
    """The signing key is lru_cached; start each test cold."""
    chat_ep._get_knowledge_qa_share_signing_key.cache_clear()
    monkeypatch.delenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    yield
    chat_ep._get_knowledge_qa_share_signing_key.cache_clear()


def _break_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom():
        raise RuntimeError("HSM unavailable")

    monkeypatch.setattr(chat_ep, "derive_hmac_key", _boom)


# --------------------------------------------------------------------------
# TASK-13289
# --------------------------------------------------------------------------

def test_signing_key_never_falls_back_to_a_published_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_derivation(monkeypatch)

    with pytest.raises(RuntimeError, match="Cannot sign knowledge-QA share links"):
        key = chat_ep._get_knowledge_qa_share_signing_key()
        assert key != _LITERAL, (
            "share tokens are being signed with a constant published in this "
            "repository -- anyone can forge a valid share link"
        )


def test_explicit_secret_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", "explicit-secret")
    assert chat_ep._get_knowledge_qa_share_signing_key() == b"explicit-secret"


def test_jwt_secret_is_an_acceptable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """JWT_SECRET_KEY is a real secret, unlike the literal -- it stays valid."""
    _break_derivation(monkeypatch)
    monkeypatch.setenv("JWT_SECRET_KEY", "a-real-deployment-secret")
    assert chat_ep._get_knowledge_qa_share_signing_key() == b"a-real-deployment-secret"


def test_transient_derivation_failure_is_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single failure must not lock in a degraded key for the process lifetime."""
    state = {"fail": True}

    def _flaky():
        if state["fail"]:
            raise RuntimeError("transient")
        return b"real-derived-key"

    monkeypatch.setattr(chat_ep, "derive_hmac_key", _flaky)

    with pytest.raises(RuntimeError):
        chat_ep._get_knowledge_qa_share_signing_key()

    state["fail"] = False
    assert chat_ep._get_knowledge_qa_share_signing_key() == b"real-derived-key", (
        "a transient derivation failure was memoized, so the process never "
        "recovers the real signing key"
    )


# --------------------------------------------------------------------------
# TASK-13297
# --------------------------------------------------------------------------

def _valid_token(monkeypatch: pytest.MonkeyPatch, payload: dict) -> str:
    monkeypatch.setenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", "test-secret")
    chat_ep._get_knowledge_qa_share_signing_key.cache_clear()
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8")
    ).decode("utf-8").rstrip("=")
    sig = hmac.new(b"test-secret", encoded.encode("utf-8"), hashlib.sha256).digest()
    return f"{encoded}.{base64.urlsafe_b64encode(sig).decode('utf-8').rstrip('=')}"


@pytest.mark.parametrize(
    "bad_signature",
    ["A", "!!!!", "====", "a b c"],
    ids=["short", "illegal_chars", "only_padding", "spaces"],
)
def test_malformed_signature_returns_400_not_500(
    monkeypatch: pytest.MonkeyPatch, bad_signature: str
) -> None:
    monkeypatch.setenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", "test-secret")

    with pytest.raises(HTTPException) as exc_info:
        chat_ep._decode_knowledge_qa_share_token(f"AAAA.{bad_signature}")

    assert exc_info.value.status_code in (400, 403), (
        f"a malformed signature segment produced {exc_info.value.status_code}; an "
        "unhandled binascii.Error on a public unauthenticated route becomes HTTP 500"
    )


def test_malformed_payload_still_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the payload guard must keep working."""
    monkeypatch.setenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", "test-secret")
    chat_ep._get_knowledge_qa_share_signing_key.cache_clear()
    encoded = base64.urlsafe_b64encode(b"not json").decode("utf-8").rstrip("=")
    sig = hmac.new(b"test-secret", encoded.encode("utf-8"), hashlib.sha256).digest()
    token = f"{encoded}.{base64.urlsafe_b64encode(sig).decode('utf-8').rstrip('=')}"

    with pytest.raises(HTTPException) as exc_info:
        chat_ep._decode_knowledge_qa_share_token(token)
    assert exc_info.value.status_code == 400


def test_wrong_signature_still_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: a well-formed but incorrect signature is still a 403."""
    token = _valid_token(monkeypatch, {"v": 1})
    encoded_payload, _ = token.split(".")
    forged = base64.urlsafe_b64encode(b"x" * 32).decode("utf-8").rstrip("=")

    with pytest.raises(HTTPException) as exc_info:
        chat_ep._decode_knowledge_qa_share_token(f"{encoded_payload}.{forged}")
    assert exc_info.value.status_code == 403


def test_valid_token_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: legitimate tokens must still verify."""
    token = _valid_token(monkeypatch, {"v": 1, "cid": "abc"})
    assert chat_ep._decode_knowledge_qa_share_token(token) == {"v": 1, "cid": "abc"}
