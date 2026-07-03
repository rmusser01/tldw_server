"""Validation helpers for Visual Identity asset source context metadata."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

MAX_SOURCE_CONTEXT_BYTES = 8 * 1024
MAX_SOURCE_CONTEXT_DEPTH = 4
MAX_SOURCE_CONTEXT_KEYS = 50
MAX_SOURCE_CONTEXT_KEY_LENGTH = 64
MAX_SOURCE_CONTEXT_STRING_LENGTH = 512
PROMPT_TEXT_KEYS = {"prompt", "negative_prompt", "system_prompt", "user_prompt", "prompt_text"}
PROMPT_REFERENCE_KEYS = {"prompt_id", "prompt_ref", "prompt_label"}

_BASE64_LIKE_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def canonicalize_source_context(value: object) -> dict[str, Any]:
    """Return a bounded, deterministic source context object."""
    key_count = 0

    def validate_node(node: object, depth: int) -> Any:
        nonlocal key_count
        if depth > MAX_SOURCE_CONTEXT_DEPTH:
            raise ValueError("invalid_source_context")

        if isinstance(node, Mapping):
            canonical: dict[str, Any] = {}
            keys = list(node)
            for key in keys:
                if not isinstance(key, str):
                    raise ValueError("invalid_source_context")
                if not 1 <= len(key) <= MAX_SOURCE_CONTEXT_KEY_LENGTH:
                    raise ValueError("invalid_source_context")

            for key in sorted(keys):
                key_lower = key.lower()
                if key_lower in PROMPT_TEXT_KEYS and key_lower not in PROMPT_REFERENCE_KEYS:
                    raise ValueError("invalid_source_context")

                key_count += 1
                if key_count > MAX_SOURCE_CONTEXT_KEYS:
                    raise ValueError("invalid_source_context")
                canonical[key] = validate_node(node[key], depth + 1)
            return canonical

        if isinstance(node, list):
            return [validate_node(item, depth + 1) for item in node]

        if node is None or isinstance(node, (bool, int)):
            return node

        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError("invalid_source_context")
            return node

        if isinstance(node, str):
            if len(node) > MAX_SOURCE_CONTEXT_STRING_LENGTH:
                raise ValueError("invalid_source_context")
            if node.lower().startswith("data:"):
                raise ValueError("invalid_source_context")
            if _is_base64_like_payload(node):
                raise ValueError("invalid_source_context")
            return node

        raise ValueError("invalid_source_context")

    if not isinstance(value, Mapping):
        raise ValueError("invalid_source_context")

    context = validate_node(value, 0)
    payload = _source_context_json(context)
    if len(payload.encode("utf-8")) > MAX_SOURCE_CONTEXT_BYTES:
        raise ValueError("invalid_source_context")
    return context


def source_context_payload_hash(value: Mapping[str, Any]) -> str:
    """Hash canonical source context JSON using sorted keys."""
    context = canonicalize_source_context(value)
    payload = _source_context_json(context)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_context_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_base64_like_payload(value: str) -> bool:
    if len(value) < 48 or len(value) % 4:
        return False
    if not _BASE64_LIKE_RE.fullmatch(value):
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(decoded) >= 32
