"""Approval token mint/verify utilities for risky chat-loop tool calls."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from threading import RLock
from typing import Any

from tldw_Server_API.app.core.Utils.base64url import SignatureMismatchError, verify_signed_token


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


_FALLBACK_PROCESS_SECRET = secrets.token_urlsafe(32)


def _approval_secret(secret: str | None) -> str:
    if isinstance(secret, str) and secret.strip():
        return secret
    from_env = os.getenv("CHAT_LOOP_APPROVAL_SECRET")
    if isinstance(from_env, str) and from_env.strip():
        return from_env
    return _FALLBACK_PROCESS_SECRET


class InMemoryApprovalNonceStore:
    """Thread-safe nonce ledger for one-time approval token use."""

    def __init__(self) -> None:
        self._used: dict[str, int] = {}
        self._lock = RLock()

    def consume(self, nonce: str, exp: int) -> bool:
        """Mark nonce as used. Returns False when nonce was already consumed."""
        now = int(time.time())
        with self._lock:
            # Opportunistic pruning keeps the ledger bounded in long-lived workers.
            expired = [key for key, expiry in self._used.items() if expiry <= now]
            for key in expired:
                self._used.pop(key, None)

            if nonce in self._used:
                return False
            self._used[nonce] = exp
            return True


def mint_approval_token(
    *,
    run_id: str,
    seq: int,
    tool_call_id: str,
    args_hash: str,
    ttl_seconds: int = 300,
    secret: str | None = None,
) -> str:
    exp = int(time.time()) + max(1, int(ttl_seconds))
    payload = {
        "run_id": run_id,
        "seq": int(seq),
        "tool_call_id": tool_call_id,
        "args_hash": args_hash,
        "exp": exp,
        "nonce": secrets.token_urlsafe(12),
    }
    payload_raw = _canonical_json(payload).encode("utf-8")
    signature = hmac.new(_approval_secret(secret).encode("utf-8"), payload_raw, hashlib.sha256).digest()
    return f"{_b64url_encode(payload_raw)}.{_b64url_encode(signature)}"


def verify_approval_token(
    *,
    token: str,
    run_id: str,
    seq: int,
    tool_call_id: str,
    args_hash: str,
    secret: str | None = None,
    nonce_store: InMemoryApprovalNonceStore | None = None,
) -> tuple[bool, str | None]:
    try:
        payload_raw = verify_signed_token(token, _approval_secret(secret).encode("utf-8"))
    except SignatureMismatchError:
        return False, "signature mismatch"
    except ValueError:
        return False, "token decode failed"
    try:
        payload = json.loads(payload_raw.decode("utf-8"))
    except ValueError:
        return False, "token decode failed"
    if not isinstance(payload, dict):
        return False, "token decode failed"

    expected_fields: dict[str, Any] = {
        "run_id": run_id,
        "seq": int(seq),
        "tool_call_id": tool_call_id,
        "args_hash": args_hash,
    }
    for field_name, expected_value in expected_fields.items():
        if payload.get(field_name) != expected_value:
            return False, f"{field_name} mismatch"

    exp = int(payload.get("exp", 0))
    if exp <= int(time.time()):
        return False, "token expired"

    nonce = str(payload.get("nonce") or "").strip()
    if not nonce:
        return False, "nonce missing"

    if nonce_store is not None and not nonce_store.consume(nonce, exp):
        return False, "token already used"

    return True, None
