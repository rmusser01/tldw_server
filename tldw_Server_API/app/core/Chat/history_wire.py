"""Browser H1 canonical JSON parity and selected durable raw-body hashing."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import rfc8785


def _object_key_order(key: str) -> tuple[int, int, bytes]:
    """JSON.stringify enumerates array-index keys before sorted ordinary keys."""
    if key.isascii() and key.isdecimal() and len(key) <= 10:
        index = int(key)
        if index < 4294967295 and str(index) == key:
            return (0, index, b"")
    return (1, 0, key.encode("utf-16be"))


def canonical_history_json(value: Any) -> str:
    """Serialize JSON-only input with the frontend's key order and ECMAScript scalars.

    Undefined object members are already absent in a dispatched JSON body.
    Reject Python-only values, lossy integer tokens, surrogates and nonfinite numbers.
    """

    def encode(item: Any) -> bytes:
        if type(item) is int and abs(item) > 9007199254740991:
            # Node emits some finite Numbers as integer tokens beyond RFC 8785's int domain.
            try:
                scalar = rfc8785.dumps(float(item))
            except OverflowError as exc:
                raise ValueError("history wire integer exceeds finite Number domain") from exc
            if scalar != str(item).encode("ascii"):
                raise ValueError("history wire integer is not an ECMAScript integer-shaped Number token")
            return scalar
        if item is None or type(item) in (str, bool, int, float):
            return rfc8785.dumps(item)
        if type(item) is list:
            return b"[" + b",".join(encode(member) for member in item) + b"]"
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("history wire object keys must be strings")
            return (
                b"{"
                + b",".join(
                    rfc8785.dumps(key) + b":" + encode(item[key]) for key in sorted(item, key=_object_key_order)
                )
                + b"}"
            )
        raise ValueError("unsupported history wire input")

    try:
        return encode(value).decode("utf-8")
    except RecursionError as exc:
        raise ValueError("cyclic or excessively nested history wire input") from exc


def history_wire_digest(value: Any) -> str:
    """Return unprefixed lowercase SHA-256 of canonical UTF-8 wire bytes."""
    return hashlib.sha256(canonical_history_json(value).encode("utf-8")).hexdigest()


def history_request_projection(raw_body: dict[str, Any]) -> dict[str, Any]:
    """Detach final dispatched body, excluding only nested history_v1 provenance.

    Never pass a Pydantic dump: omitted, explicit null and defaults are distinct.
    Transport/query scope remains a separate owner fence, not part of this body.
    """
    if type(raw_body) is not dict or type(raw_body.get("tldw_turn")) is not dict:
        raise ValueError("selected durable digest requires a raw JSON body and tldw_turn")
    projection = {**raw_body, "tldw_turn": dict(raw_body["tldw_turn"])}
    projection["tldw_turn"].pop("history_v1", None)
    canonical_history_json(projection)
    return deepcopy(projection)


def selected_durable_request_digest(raw_body: dict[str, Any]) -> str:
    """Hash the exact final body projection, without server-injected defaults."""
    return history_wire_digest(history_request_projection(raw_body))
