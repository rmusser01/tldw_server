"""Bounded, inert artwork credits shared by native packs and Buddy snapshots."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

ARTWORK_MANIFEST_KEY = "tldw/artwork"
MAX_ARTWORK_BYTES = 512 * 1024
_TEXT_LIMITS = {"creator": 512, "license": 4096, "source_url": 2048, "notices": 64 * 1024}


def validate_artwork_record(value: object) -> dict[str, Any]:
    """Validate a version-1 credit record without interpreting or fetching it."""
    if not isinstance(value, dict) or set(value) != {"version", *_TEXT_LIMITS}:
        raise ValueError("artwork_attribution_invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("artwork_attribution_invalid")
    for field, limit in _TEXT_LIMITS.items():
        text = value[field]
        if text is None and field != "notices":
            continue
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError("artwork_attribution_invalid")
        try:
            valid = len(text.encode("utf-8")) <= limit and not any(
                (ord(char) < 32 and char not in "\t\r\n") or 127 <= ord(char) <= 159 for char in text
            )
        except UnicodeError:
            valid = False
        if not valid:
            raise ValueError("artwork_attribution_invalid")
    url = value["source_url"]
    if url is not None:
        _validate_source_url(url)
    return dict(value)


def _validate_source_url(url: str) -> None:
    # A display-only public reference, compatible with native Chatbook carriers.
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        labels = host.split(".")
        valid = (
            parsed.scheme == "https"
            and parsed.netloc.lower() == host
            and len(labels) > 1
            and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in labels)
            and re.fullmatch(r"[a-z][a-z0-9-]*", labels[-1]) is not None
            and not host.endswith((".local", ".localhost", ".internal", ".lan", ".home"))
            and not any(char.isspace() or char in "\\?#" for char in url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("artwork_attribution_invalid")


def encode_native_artwork(value: object) -> str:
    """Return the canonical native source_context.artwork string."""
    record = validate_artwork_record(value)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_ARTWORK_BYTES:
        raise ValueError("artwork_attribution_invalid")
    return encoded


def artwork_manifest_for_import(pack: dict[str, Any]) -> dict[str, Any]:
    """Merge agreeing native credit carriers, excluding unrelated source context."""
    manifest = pack.get("visual_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("malformed_metadata: metadata/pack.json")
    manifest = dict(manifest)
    if ARTWORK_MANIFEST_KEY in manifest:
        manifest[ARTWORK_MANIFEST_KEY] = validate_artwork_record(manifest[ARTWORK_MANIFEST_KEY])
    context = pack.get("source_context", {})
    if not isinstance(context, dict):
        return manifest
    if "artwork" not in context:
        return manifest
    carrier = context["artwork"]
    try:
        if not isinstance(carrier, str) or len(carrier.encode("utf-8")) > MAX_ARTWORK_BYTES:
            raise ValueError("artwork_attribution_invalid")
        record = validate_artwork_record(json.loads(carrier))
        if encode_native_artwork(record) != carrier:
            raise ValueError("artwork_attribution_invalid")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("artwork_attribution_invalid") from exc
    if ARTWORK_MANIFEST_KEY in manifest and manifest[ARTWORK_MANIFEST_KEY] != record:
        raise ValueError("artwork_attribution_conflict")
    manifest[ARTWORK_MANIFEST_KEY] = record
    return manifest
