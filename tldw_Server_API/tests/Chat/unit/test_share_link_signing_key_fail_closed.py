"""Regression (TASK-13289): share-link signing must fail closed when key derivation fails.

`_get_knowledge_qa_share_signing_key` used to fall back to ``JWT_SECRET_KEY`` or the
literal ``"knowledge_qa_share_link_default"`` (published in this repo) when
``derive_hmac_key()`` raised, and memoized that weak key for the process lifetime.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat as chat_mod

pytestmark = pytest.mark.unit


def _boom(*_args, **_kwargs):
    raise ValueError("no usable HMAC key material")


@pytest.fixture
def broken_key_derivation(monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_QA_SHARE_LINK_SECRET", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.setattr(chat_mod, "derive_hmac_key", _boom)
    chat_mod._get_knowledge_qa_share_signing_key.cache_clear()
    yield
    chat_mod._get_knowledge_qa_share_signing_key.cache_clear()


def _forge_with_public_literal(payload: dict) -> str:
    encoded = chat_mod._urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig = hmac.new(b"knowledge_qa_share_link_default", encoded.encode("utf-8"), hashlib.sha256).digest()
    return f"{encoded}.{chat_mod._urlsafe_b64encode(sig)}"


def test_token_forged_with_public_literal_is_not_accepted(broken_key_derivation) -> None:
    forged = _forge_with_public_literal({"v": 1, "share_id": "attacker"})
    with pytest.raises(HTTPException) as excinfo:
        chat_mod._decode_knowledge_qa_share_token(forged)
    assert excinfo.value.status_code == 503


def test_minting_fails_instead_of_issuing_weak_token(broken_key_derivation) -> None:
    with pytest.raises(HTTPException) as excinfo:
        chat_mod._build_knowledge_qa_share_token({"v": 1, "share_id": "x"})
    assert excinfo.value.status_code == 503


def test_transient_failure_is_not_pinned(broken_key_derivation, monkeypatch) -> None:
    with pytest.raises(HTTPException):
        chat_mod._get_knowledge_qa_share_signing_key()
    good = b"k" * 32
    monkeypatch.setattr(chat_mod, "derive_hmac_key", lambda *a, **k: good)
    assert chat_mod._get_knowledge_qa_share_signing_key() == good
    token = chat_mod._build_knowledge_qa_share_token({"v": 1, "share_id": "ok"})
    assert chat_mod._decode_knowledge_qa_share_token(token) == {"v": 1, "share_id": "ok"}
