"""Deterministic canonical event-body preparation and validation."""

from __future__ import annotations

import hmac
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .crypto import (
    EVENT_BODY_MAX_BYTES,
    WebhookKeyError,
    WebhookKeyErrorCode,
    WebhookKeyRing,
)
from .domain import (
    EventSourceKind,
    PendingIncidentWebhookMarker,
    WebhookError,
    WebhookErrorCode,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        EventCaptureResult,
        EventInsert,
        StoredWebhookEvent,
    )

_MAX_JSON_DEPTH = 64


def decrypt_pending_incident_marker_body(
    ring: WebhookKeyRing,
    marker: PendingIncidentWebhookMarker,
) -> tuple[bytes, dict[str, str | int]]:
    """Decrypt current marker AAD, with a bounded legacy-AAD fallback."""

    current_identity = dict(marker.envelope_identity)
    try:
        return (
            ring.decrypt_bytes(
                purpose=marker.envelope_purpose,
                identity=current_identity,
                protected=marker.body,
            ),
            current_identity,
        )
    except WebhookKeyError as exc:
        legacy_identity = dict(marker.legacy_envelope_identity)
        if (
            exc.code is not WebhookKeyErrorCode.CONTEXT_MISMATCH
            or not marker.uses_legacy_aad
            or legacy_identity == current_identity
        ):
            raise
        return (
            ring.decrypt_bytes(
                purpose=marker.envelope_purpose,
                identity=legacy_identity,
                protected=marker.body,
            ),
            legacy_identity,
        )


@dataclass(frozen=True)
class PreparedEventInsert:
    """Protected event write plus the detached data needed for replay proof."""

    event: EventInsert = field(repr=False)
    data: dict[str, object] = field(repr=False)


def _canonical_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("event data nesting is invalid")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event data number is invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("event data key is invalid")
        for item in value.values():
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("event data value is invalid")


def snapshot_json_object(data: dict[str, object]) -> dict[str, object]:
    """Return a detached canonical JSON snapshot or reject unsafe input."""

    _validate_json_value(data)
    try:
        encoded = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        snapshot = json.loads(encoded)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("event data is invalid") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("event data must be a JSON object")
    return snapshot


def canonical_event_body(
    *,
    event_id: str,
    event_type: str,
    api_version: str,
    created_at: datetime,
    data: dict[str, object],
) -> bytes:
    """Encode the immutable public event envelope with a strict size bound."""

    try:
        encoded = json.dumps(
            {
                "id": event_id,
                "type": event_type,
                "api_version": api_version,
                "created_at": _canonical_timestamp(created_at),
                "data": data,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("event body is invalid") from exc
    if len(encoded) > EVENT_BODY_MAX_BYTES:
        raise ValueError("event body is too large")
    return encoded


def parse_canonical_event_body(
    plaintext: bytes,
    *,
    event_id: str,
    event_type: str,
    api_version: str,
    created_at: datetime,
) -> dict[str, object]:
    """Validate exact canonical event bytes and return detached public data."""

    if (
        not isinstance(plaintext, bytes)
        or not plaintext
        or len(plaintext) > EVENT_BODY_MAX_BYTES
    ):
        raise ValueError("event body is invalid")
    try:
        decoded = json.loads(plaintext)
        if not isinstance(decoded, dict) or set(decoded) != {
            "id",
            "type",
            "api_version",
            "created_at",
            "data",
        }:
            raise ValueError("event body is invalid")
        if (
            decoded["id"] != event_id
            or decoded["type"] != event_type
            or decoded["api_version"] != api_version
            or decoded["created_at"] != _canonical_timestamp(created_at)
            or not isinstance(decoded["data"], dict)
        ):
            raise ValueError("event body is invalid")
        data = snapshot_json_object(decoded["data"])
        expected = canonical_event_body(
            event_id=event_id,
            event_type=event_type,
            api_version=api_version,
            created_at=created_at,
            data=data,
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("event body is invalid") from exc
    if not hmac.compare_digest(plaintext, expected):
        raise ValueError("event body is invalid")
    return data


def prepare_event_insert(
    *,
    ring: WebhookKeyRing,
    event_id: str,
    event_type: str,
    api_version: str,
    source_kind: EventSourceKind,
    aggregate_type: str | None,
    aggregate_id: str | None,
    aggregate_version: str | None,
    source_command_id: str | None,
    source_component: str,
    source_request_id: str | None,
    created_at: datetime,
    data: dict[str, object],
) -> PreparedEventInsert:
    """Build one encrypted repository insert from a closed event source."""

    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        EventInsert,
    )

    snapshot = snapshot_json_object(data)
    body = canonical_event_body(
        event_id=event_id,
        event_type=event_type,
        api_version=api_version,
        created_at=created_at,
        data=snapshot,
    )
    event = EventInsert(
        id=event_id,
        event_type=event_type,
        api_version=api_version,
        source_kind=source_kind,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        aggregate_version=aggregate_version,
        source_command_id=source_command_id,
        source_component=source_component,
        source_request_id=source_request_id,
        body=ring.encrypt_event_body(
            event_id=event_id,
            api_version=api_version,
            body=body,
        ),
        body_size_bytes=len(body),
        created_at=created_at,
    )
    return PreparedEventInsert(event=event, data=snapshot)


def verify_event_replay(
    *,
    ring: WebhookKeyRing,
    result: EventCaptureResult,
    prepared: PreparedEventInsert,
) -> None:
    """Require an existing source event to match the requested body exactly."""

    stored = result.event
    expected_source = prepared.event
    if not (
        stored.event.event_type == expected_source.event_type
        and stored.event.source_kind is expected_source.source_kind
        and stored.aggregate_type == expected_source.aggregate_type
        and stored.aggregate_id == expected_source.aggregate_id
        and stored.aggregate_version == expected_source.aggregate_version
        and stored.source_command_id == expected_source.source_command_id
        and stored.source_component == expected_source.source_component
        and stored.source_request_id == expected_source.source_request_id
    ):
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
    expected_body = canonical_event_body(
        event_id=stored.id,
        event_type=stored.event.event_type,
        api_version=stored.event.api_version,
        created_at=stored.event.created_at,
        data=prepared.data,
    )
    try:
        plaintext = ring.decrypt_event_body(
            event_id=stored.id,
            api_version=stored.event.api_version,
            protected=stored.body,
        )
    except WebhookKeyError:
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT) from None
    if (
        stored.body_size_bytes != len(expected_body)
        or stored.body_size_bytes != len(plaintext)
        or not hmac.compare_digest(plaintext, expected_body)
    ):
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)


def validate_stored_event_body(
    event: StoredWebhookEvent,
    plaintext: bytes,
) -> None:
    """Require exact canonical bytes matching immutable event metadata."""

    if not isinstance(plaintext, bytes) or len(plaintext) != event.body_size_bytes:
        raise ValueError("persisted event body is invalid")
    try:
        parse_canonical_event_body(
            plaintext,
            event_id=event.event.id,
            event_type=event.event.event_type,
            api_version=event.event.api_version,
            created_at=event.event.created_at,
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("persisted event body is invalid") from exc


__all__ = [
    "PreparedEventInsert",
    "canonical_event_body",
    "parse_canonical_event_body",
    "prepare_event_insert",
    "snapshot_json_object",
    "validate_stored_event_body",
    "verify_event_replay",
]
