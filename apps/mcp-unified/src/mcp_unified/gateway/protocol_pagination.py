"""Deterministic MCP catalog pagination with authenticated opaque cursors."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any

from .protocol_errors import GatewayInvalidApplicationResult
from .protocol_limits import GatewayLimits
from .protocol_profiles import GatewayProtocolProfile
from .protocol_projection import _name, _normalize_uri
from .protocol_validation import _JSONStructureError, _validate_json_structure
from .runtime import GatewayJSONValue

_CATALOG_IDENTITY_FIELDS = {
    "tools/list": "name",
    "prompts/list": "name",
    "resources/list": "uri",
    "resources/templates/list": "uriTemplate",
}
_CURSOR_MAX_BYTES = 2_048


@dataclass(frozen=True, slots=True)
class GatewayCatalogPage:
    """One deterministic catalog slice and its optional continuation cursor."""

    items: list[dict[str, GatewayJSONValue]]
    next_cursor: str | None


def _json_bytes(value: object, *, max_depth: int = GatewayLimits().max_json_depth) -> bytes:
    try:
        _validate_json_structure(value, max_depth=max_depth)
    except _JSONStructureError as exc:
        raise ValueError("catalog items must be finite JSON objects") from exc
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("catalog items must be finite JSON objects") from exc


def _clone_item(value: object, *, max_depth: int) -> dict[str, GatewayJSONValue]:
    decoded = json.loads(_json_bytes(value, max_depth=max_depth))
    if not isinstance(decoded, dict):
        raise ValueError("catalog items must be JSON objects")
    return decoded


def _identity(method: str, item: dict[str, GatewayJSONValue]) -> str:
    field = _CATALOG_IDENTITY_FIELDS.get(method)
    if field is None:
        raise ValueError("unsupported catalog method")
    value = item.get(field)
    if method in {"resources/list", "resources/templates/list"}:
        try:
            return _normalize_uri(
                value,
                template=method == "resources/templates/list",
            )
        except GatewayInvalidApplicationResult as exc:
            raise ValueError("invalid catalog identity") from exc
    try:
        return _name(value)
    except GatewayInvalidApplicationResult as exc:
        raise ValueError("invalid catalog identity") from exc


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("invalid cursor") from exc
    if _b64encode(decoded) != value:
        raise ValueError("invalid cursor")
    return decoded


class GatewayCatalogPaginator:
    """Sort, fingerprint, and page catalogs with a connection-local secret."""

    def __init__(
        self,
        limits: GatewayLimits = GatewayLimits(),
        *,
        secret: bytes | None = None,
    ) -> None:
        if secret is not None and (not isinstance(secret, bytes) or len(secret) != 32):
            raise ValueError("cursor secret must contain exactly 32 bytes")
        self._limits = limits
        self._secret = secret if secret is not None else secrets.token_bytes(32)

    def page(
        self,
        *,
        method: str,
        profile: GatewayProtocolProfile,
        items: list[dict[str, GatewayJSONValue]],
        cursor: str | None,
    ) -> GatewayCatalogPage:
        """Return one stable page after authenticating all cursor bindings."""

        if method not in _CATALOG_IDENTITY_FIELDS:
            raise ValueError("unsupported catalog method")
        if not isinstance(items, list):
            raise ValueError("catalog items must be a list")
        if len(items) > self._limits.max_catalog_items:
            raise ValueError("catalog item limit exceeded")

        normalized: list[tuple[str, dict[str, GatewayJSONValue]]] = []
        identities: set[str] = set()
        for raw_item in items:
            item = _clone_item(raw_item, max_depth=self._limits.max_json_depth)
            identity = _identity(method, item)
            if identity in identities:
                raise ValueError("duplicate catalog identity")
            identities.add(identity)
            identity_field = _CATALOG_IDENTITY_FIELDS[method]
            item[identity_field] = identity
            normalized.append((identity, item))
        normalized.sort(key=lambda pair: pair[0])
        sorted_items = [item for _, item in normalized]
        fingerprint = hashlib.sha256(_json_bytes(sorted_items, max_depth=self._limits.max_json_depth)).hexdigest()

        page_size = self._limits.default_catalog_page_size
        offset = 0
        if cursor is not None:
            payload = self._decode_cursor(cursor)
            if payload != {
                "method": method,
                "version": profile.version,
                "offset": payload.get("offset"),
                "pageSize": page_size,
                "fingerprint": fingerprint,
            }:
                raise ValueError("invalid cursor")
            offset_value = payload.get("offset")
            if (
                isinstance(offset_value, bool)
                or not isinstance(offset_value, int)
                or offset_value < page_size
                or offset_value % page_size != 0
                or offset_value >= len(sorted_items)
            ):
                raise ValueError("invalid cursor")
            offset = offset_value

        page_items = sorted_items[offset : offset + page_size]
        next_offset = offset + len(page_items)
        next_cursor = None
        if next_offset < len(sorted_items):
            next_cursor = self._encode_cursor(
                {
                    "method": method,
                    "version": profile.version,
                    "offset": next_offset,
                    "pageSize": page_size,
                    "fingerprint": fingerprint,
                }
            )
        return GatewayCatalogPage(items=page_items, next_cursor=next_cursor)

    def _encode_cursor(self, payload: dict[str, Any]) -> str:
        payload_part = _b64encode(_json_bytes(payload))
        signature = hmac.digest(self._secret, payload_part.encode("ascii"), "sha256")
        cursor = f"{payload_part}.{_b64encode(signature)}"
        if len(cursor.encode("ascii")) > _CURSOR_MAX_BYTES:
            raise ValueError("invalid cursor")
        return cursor

    def _decode_cursor(self, cursor: object) -> dict[str, Any]:
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("invalid cursor")
        try:
            cursor_bytes = cursor.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("invalid cursor") from exc
        if len(cursor_bytes) > _CURSOR_MAX_BYTES or cursor.count(".") != 1:
            raise ValueError("invalid cursor")
        payload_part, signature_part = cursor.split(".")
        signature = _b64decode(signature_part)
        expected = hmac.digest(self._secret, payload_part.encode("ascii"), "sha256")
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid cursor")
        try:
            payload = json.loads(_b64decode(payload_part))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid cursor") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "method",
            "version",
            "offset",
            "pageSize",
            "fingerprint",
        }:
            raise ValueError("invalid cursor")
        return payload


__all__ = ["GatewayCatalogPage", "GatewayCatalogPaginator"]
