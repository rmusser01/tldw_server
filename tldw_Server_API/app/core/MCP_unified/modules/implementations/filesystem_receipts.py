"""Stateless read receipts for MCP filesystem tools."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


class ReadReceiptError(ValueError):
    """Raised when a read receipt cannot be validated."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class ReadReceiptPayload:
    """Decoded read receipt payload."""

    path: str
    sha256: str
    size: int
    expires_at: int
    workspace_id: str | None = None
    session_id: str | None = None


class ReadReceiptManager:
    """Issue and validate HMAC-signed read receipts."""

    def __init__(self, *, secret: str | bytes | None = None, ttl_seconds: int = 1_800) -> None:
        if isinstance(secret, bytes):
            secret_bytes = secret
        elif isinstance(secret, str) and secret.strip():
            secret_bytes = secret.strip().encode("utf-8")
        else:
            secret_bytes = b""
        self._secret = secret_bytes
        self._enabled = bool(secret_bytes)
        self._ttl_seconds = max(1, int(ttl_seconds))

    @property
    def enabled(self) -> bool:
        """Return whether a stable receipt-signing secret is configured."""

        return self._enabled

    def issue(
        self,
        *,
        path: str,
        sha256: str,
        size: int,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Return a signed receipt for a complete file preimage."""

        if not self._enabled:
            raise ReadReceiptError("read_receipt_secret_unconfigured")
        now = int(time.time())
        payload: dict[str, Any] = {
            "v": 1,
            "path": path,
            "sha256": sha256,
            "size": int(size),
            "issued_at": now,
            "expires_at": now + self._ttl_seconds,
        }
        if workspace_id:
            payload["workspace_id"] = str(workspace_id)
        if session_id:
            payload["session_id"] = str(session_id)
        payload_bytes = _canonical_json(payload)
        signature = hmac.new(self._secret, payload_bytes, hashlib.sha256).hexdigest()
        envelope = {"payload": payload, "signature": signature}
        return base64.urlsafe_b64encode(_canonical_json(envelope)).decode("ascii")

    def validate(self, receipt: str) -> ReadReceiptPayload:
        """Validate and decode a signed receipt."""

        if not self._enabled:
            raise ReadReceiptError("read_receipt_secret_unconfigured")
        try:
            envelope_bytes = base64.urlsafe_b64decode(receipt.encode("ascii"))
            envelope = json.loads(envelope_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ReadReceiptError("read_receipt_invalid") from exc
        if not isinstance(envelope, dict):
            raise ReadReceiptError("read_receipt_invalid")
        payload = envelope.get("payload")
        signature = envelope.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise ReadReceiptError("read_receipt_invalid")
        expected = hmac.new(self._secret, _canonical_json(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ReadReceiptError("read_receipt_invalid")
        expires_at = int(payload.get("expires_at") or 0)
        if expires_at < int(time.time()):
            raise ReadReceiptError("read_receipt_expired")
        return ReadReceiptPayload(
            path=str(payload.get("path") or ""),
            sha256=str(payload.get("sha256") or ""),
            size=int(payload.get("size") or 0),
            expires_at=expires_at,
            workspace_id=str(payload.get("workspace_id") or "") or None,
            session_id=str(payload.get("session_id") or "") or None,
        )


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize JSON bytes with stable key ordering for HMAC signing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
