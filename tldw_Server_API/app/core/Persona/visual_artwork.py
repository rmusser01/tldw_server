"""Bounded, inert artwork credits shared by native packs and Buddy snapshots."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from tldw_Server_API.app.core.exception_types import PersonaArtworkValidationError

ARTWORK_MANIFEST_KEY = "tldw/artwork"
MAX_ARTWORK_BYTES = 512 * 1024
_TEXT_LIMITS = {"creator": 512, "license": 4096, "source_url": 2048, "notices": 64 * 1024}


def validate_artwork_record(value: object) -> dict[str, Any]:
    """Validate a version-1 credit record without interpreting or fetching it.

    Args:
        value: External object containing exactly the five native credit fields.

    Returns:
        A new dictionary retaining the original validated strings and nulls.

    Raises:
        PersonaArtworkValidationError: Fields, version, text bounds or URL syntax
            violate the supported artwork contract.
    """
    if not isinstance(value, dict) or set(value) != {"version", *_TEXT_LIMITS}:
        raise PersonaArtworkValidationError("artwork_attribution_invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise PersonaArtworkValidationError("artwork_attribution_invalid")
    for field, limit in _TEXT_LIMITS.items():
        text = value[field]
        if text is None and field != "notices":
            continue
        if not isinstance(text, str) or len(text) > limit:
            raise PersonaArtworkValidationError("artwork_attribution_invalid")
        try:
            valid = len(text.encode("utf-8")) <= limit and not any(
                (ord(char) < 32 and char not in "\t\r\n") or 127 <= ord(char) <= 159 for char in text
            )
        except UnicodeError:
            valid = False
        if not valid:
            raise PersonaArtworkValidationError("artwork_attribution_invalid")
    url = value["source_url"]
    if url is not None:
        _validate_source_url(url)
    return dict(value)


def _validate_source_url(url: str) -> None:
    """Validate a display-only HTTPS reference without DNS or network access.

    Args:
        url: A bounded source URL from an artwork record.

    Raises:
        PersonaArtworkValidationError: The URL has credentials, a port, query,
            fragment, invalid public hostname syntax or a local-domain suffix.
    """
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
        raise PersonaArtworkValidationError("artwork_attribution_invalid")


def encode_native_artwork(value: object) -> str:
    """Return the canonical native source_context.artwork string.

    Args:
        value: External version-1 artwork record to validate and serialize.

    Returns:
        Sorted, compact Unicode JSON suitable for the native archive carrier.

    Raises:
        PersonaArtworkValidationError: The record or encoded byte size is invalid.
    """
    record = validate_artwork_record(value)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_ARTWORK_BYTES:
        raise PersonaArtworkValidationError("artwork_attribution_invalid")
    return encoded


def artwork_manifest_for_import(pack: dict[str, Any]) -> dict[str, Any]:
    """Merge agreeing native credit carriers, excluding unrelated source context.

    Args:
        pack: Native metadata/pack.json pack object with a visual manifest and
            optional canonical source_context.artwork string.

    Returns:
        A shallow manifest copy with validated credits under tldw/artwork when
        present. Unrelated source context is ignored and the input is unchanged.

    Raises:
        PersonaArtworkValidationError: The manifest is not an object, a credit
            carrier is malformed or oversized, or supported carriers disagree.
    """
    manifest = pack.get("visual_manifest")
    if not isinstance(manifest, dict):
        raise PersonaArtworkValidationError("malformed_metadata: metadata/pack.json")
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
            raise PersonaArtworkValidationError("artwork_attribution_invalid")
        record = validate_artwork_record(json.loads(carrier))
        if encode_native_artwork(record) != carrier:
            raise PersonaArtworkValidationError("artwork_attribution_invalid")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PersonaArtworkValidationError("artwork_attribution_invalid") from exc
    if ARTWORK_MANIFEST_KEY in manifest and manifest[ARTWORK_MANIFEST_KEY] != record:
        raise PersonaArtworkValidationError("artwork_attribution_conflict")
    manifest[ARTWORK_MANIFEST_KEY] = record
    return manifest
