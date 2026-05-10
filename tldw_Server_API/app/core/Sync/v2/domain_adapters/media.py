from __future__ import annotations

"""Sync v2 compatibility adapter for legacy media sync rows."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tldw_Server_API.app.core.Sync.sync_contract import SyncEntity, SyncOperation as LegacySyncOperation

from ..adapters import AdapterAccepted, AdapterRejected, SyncAdapterContext, SyncAdapterOutcome
from ..models import SyncDataset, SyncDomain, SyncEnvelopeCreate

_LEGACY_TO_V2_OPERATION = {
    LegacySyncOperation.CREATE.value: "upsert",
    LegacySyncOperation.UPDATE.value: "upsert",
    LegacySyncOperation.DELETE.value: "delete",
    LegacySyncOperation.LINK.value: "link",
    LegacySyncOperation.UNLINK.value: "unlink",
}

_VERSIONED_MEDIA_ENTITIES = {
    SyncEntity.MEDIA.value,
    SyncEntity.KEYWORDS.value,
}

_SENDABLE_MEDIA_ENTITIES = _VERSIONED_MEDIA_ENTITIES | {SyncEntity.MEDIA_KEYWORDS.value}
_MEDIA_KEYWORD_OPERATIONS = {LegacySyncOperation.LINK.value, LegacySyncOperation.UNLINK.value}
_VERSIONED_MEDIA_OPERATIONS = {
    LegacySyncOperation.CREATE.value,
    LegacySyncOperation.UPDATE.value,
    LegacySyncOperation.DELETE.value,
}


@dataclass(slots=True)
class MediaCompatibilityAdapter:
    """Validate legacy media sync semantics before Sync v2 persists envelopes."""

    domain: SyncDomain = "media"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Accept or reject media envelopes using the legacy sync contract."""

        del dataset
        del context
        legacy_entity = _legacy_entity(envelope)
        legacy_operation = _legacy_operation(envelope)

        if legacy_entity not in _SENDABLE_MEDIA_ENTITIES:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="unsupported_legacy_media_entity",
                message="Legacy media sync entity is not supported by Sync v2 media adapter.",
            )
        if legacy_operation not in _LEGACY_TO_V2_OPERATION:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="invalid_legacy_media_operation",
                message="Legacy media sync operation is not supported by Sync v2 media adapter.",
            )
        if _LEGACY_TO_V2_OPERATION[legacy_operation] != envelope.operation:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="invalid_legacy_media_operation",
                message="Legacy media sync operation does not match the Sync v2 operation.",
            )
        if legacy_entity == SyncEntity.MEDIA_KEYWORDS.value:
            return self._evaluate_media_keyword_link(envelope, legacy_operation)
        if legacy_operation not in _VERSIONED_MEDIA_OPERATIONS:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="invalid_legacy_media_operation",
                message="Only MediaKeywords supports legacy link and unlink operations.",
            )

        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

    def _evaluate_media_keyword_link(
        self,
        envelope: SyncEnvelopeCreate,
        legacy_operation: str,
    ) -> SyncAdapterOutcome:
        if legacy_operation not in _MEDIA_KEYWORD_OPERATIONS:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="invalid_legacy_media_operation",
                message="MediaKeywords only supports legacy link and unlink operations.",
            )
        if not _media_keyword_link_metadata(envelope):
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="missing_media_keyword_link_metadata",
                message="MediaKeywords link and unlink envelopes require media_uuid and keyword_uuid.",
            )
        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def legacy_media_sync_log_to_envelope(
    entry: Any,
    *,
    dataset_id: str,
    payload_ciphertext: str | None,
    adapter_version: int = 1,
) -> SyncEnvelopeCreate:
    """Convert a legacy media sync-log entry to a Sync v2 envelope."""

    data = _entry_mapping(entry)
    legacy_entity = _enum_value(data["entity"])
    legacy_operation = _enum_value(data["operation"])
    payload, payload_text = _payload_mapping_and_text(data.get("payload"))

    if legacy_entity not in _SENDABLE_MEDIA_ENTITIES:
        raise ValueError(f"Unsupported legacy media sync entity: {legacy_entity}")
    if legacy_operation not in _LEGACY_TO_V2_OPERATION:
        raise ValueError(f"Unsupported legacy media sync operation: {legacy_operation}")

    routing_metadata = {
        "legacy_entity": legacy_entity,
        "legacy_operation": legacy_operation,
    }
    entity_uuid = str(data["entity_uuid"])
    entity_id = entity_uuid
    stable_key = f"{_stable_key_prefix(legacy_entity)}:{entity_uuid}"
    payload_clear: dict[str, Any] = {"entity_kind": legacy_entity}

    if legacy_entity == SyncEntity.MEDIA_KEYWORDS.value:
        media_uuid = str(payload.get("media_uuid") or "")
        keyword_uuid = str(payload.get("keyword_uuid") or "")
        if not media_uuid or not keyword_uuid:
            raise ValueError("MediaKeywords sync entries require media_uuid and keyword_uuid")
        entity_id = f"{media_uuid}:{keyword_uuid}"
        stable_key = f"media_keywords:{media_uuid}:{keyword_uuid}"
        routing_metadata.update({"media_uuid": media_uuid, "keyword_uuid": keyword_uuid})
        payload_clear["link_type"] = "media_keyword"
    elif "deleted" in payload:
        payload_clear["deleted"] = bool(payload["deleted"])

    return SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id=f"legacy-media:{data['change_id']}",
        domain="media",
        entity_id=entity_id,
        operation=_LEGACY_TO_V2_OPERATION[legacy_operation],
        adapter_version=adapter_version,
        device_id=str(data.get("client_id") or "") or None,
        stable_key=stable_key,
        client_timestamp=str(data.get("timestamp") or "") or None,
        entity_version=data.get("version"),
        routing_metadata=routing_metadata,
        payload_ciphertext=payload_ciphertext,
        payload_clear=payload_clear,
        payload_hash=_payload_hash(payload_text),
        payload_size_bytes=len(payload_text.encode("utf-8")),
    )


def _legacy_entity(envelope: SyncEnvelopeCreate) -> str:
    return str(
        envelope.routing_metadata.get("legacy_entity")
        or envelope.payload_clear.get("entity_kind")
        or envelope.payload_clear.get("entity_type")
        or ""
    )


def _legacy_operation(envelope: SyncEnvelopeCreate) -> str:
    value = envelope.routing_metadata.get("legacy_operation")
    if value is not None:
        return str(value)
    if envelope.operation == "upsert":
        return LegacySyncOperation.UPDATE.value
    return str(envelope.operation)


def _media_keyword_link_metadata(envelope: SyncEnvelopeCreate) -> bool:
    media_uuid = envelope.routing_metadata.get("media_uuid") or envelope.payload_clear.get("media_id")
    keyword_uuid = envelope.routing_metadata.get("keyword_uuid") or envelope.payload_clear.get("target_entity_id")
    return bool(media_uuid and keyword_uuid)


def _entry_mapping(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return dict(entry)
    if hasattr(entry, "model_dump"):
        return dict(entry.model_dump())
    return {
        key: getattr(entry, key)
        for key in (
            "change_id",
            "entity",
            "entity_uuid",
            "operation",
            "timestamp",
            "client_id",
            "version",
            "payload",
        )
    }


def _payload_mapping_and_text(payload_value: Any) -> tuple[dict[str, Any], str]:
    if payload_value is None or payload_value == "":
        return {}, ""
    if isinstance(payload_value, Mapping):
        payload = dict(payload_value)
        return payload, json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if not isinstance(payload_value, str):
        raise ValueError("Legacy media sync payload must be a JSON object")

    payload = json.loads(payload_value)
    if not isinstance(payload, dict):
        raise ValueError("Legacy media sync payload must be a JSON object")
    return payload, payload_value


def _payload_hash(payload_text: str) -> str:
    return f"sha256:{hashlib.sha256(payload_text.encode('utf-8')).hexdigest()}"


def _stable_key_prefix(legacy_entity: str) -> str:
    if legacy_entity == SyncEntity.MEDIA.value:
        return "media"
    if legacy_entity == SyncEntity.KEYWORDS.value:
        return "keyword"
    return "media_keywords"


def _enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


__all__ = [
    "MediaCompatibilityAdapter",
    "legacy_media_sync_log_to_envelope",
]
