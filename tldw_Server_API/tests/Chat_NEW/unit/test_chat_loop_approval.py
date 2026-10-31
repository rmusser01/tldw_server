"""Unit tests for chat-loop approval token mint/verify."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.chat_loop_approval import (
    InMemoryApprovalNonceStore,
    mint_approval_token,
    verify_approval_token,
)


@pytest.mark.unit
def test_approval_token_rejects_mismatched_args_hash() -> None:
    token = mint_approval_token(
        run_id="run_1",
        seq=7,
        tool_call_id="tc_1",
        args_hash="hash_a",
        secret="test-secret",
    )

    ok, error = verify_approval_token(
        token=token,
        run_id="run_1",
        seq=7,
        tool_call_id="tc_1",
        args_hash="hash_b",
        secret="test-secret",
    )

    assert ok is False
    assert error is not None
    assert "args_hash" in error


@pytest.mark.unit
def test_approval_token_is_single_use_when_nonce_store_present() -> None:
    store = InMemoryApprovalNonceStore()
    token = mint_approval_token(
        run_id="run_1",
        seq=8,
        tool_call_id="tc_2",
        args_hash="hash_x",
        secret="test-secret",
    )

    first_ok, _ = verify_approval_token(
        token=token,
        run_id="run_1",
        seq=8,
        tool_call_id="tc_2",
        args_hash="hash_x",
        secret="test-secret",
        nonce_store=store,
    )
    second_ok, second_error = verify_approval_token(
        token=token,
        run_id="run_1",
        seq=8,
        tool_call_id="tc_2",
        args_hash="hash_x",
        secret="test-secret",
        nonce_store=store,
    )

    assert first_ok is True
    assert second_ok is False
    assert second_error is not None
    assert "already used" in second_error


_BINDING = {"run_id": "run_1", "seq": 7, "tool_call_id": "tc_1", "args_hash": "hash_a"}


@pytest.mark.unit
def test_approval_token_signed_with_another_secret_is_rejected() -> None:
    token = mint_approval_token(**_BINDING, secret="other-secret")
    ok, error = verify_approval_token(token=token, **_BINDING, secret="test-secret")
    assert (ok, error) == (False, "signature mismatch")


@pytest.mark.unit
def test_approval_token_with_junk_appended_is_rejected() -> None:
    """A lax decoder dropped out-of-alphabet junk and accepted the original token."""
    token = mint_approval_token(**_BINDING, secret="test-secret")
    ok, error = verify_approval_token(token=token + "!!!!", **_BINDING, secret="test-secret")
    assert (ok, error) == (False, "token decode failed")


@pytest.mark.unit
def test_approval_token_with_non_object_payload_is_rejected_not_raised() -> None:
    import hashlib
    import hmac

    from tldw_Server_API.app.core.Utils.base64url import encode_segment

    payload = b"[1, 2]"
    signature = hmac.new(b"test-secret", payload, hashlib.sha256).digest()
    token = f"{encode_segment(payload)}.{encode_segment(signature)}"
    ok, error = verify_approval_token(token=token, **_BINDING, secret="test-secret")
    assert (ok, error) == (False, "token decode failed")
