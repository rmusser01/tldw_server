from __future__ import annotations

"""Canonical persisted Sync mutation-group validation."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from .errors import SyncIdempotencyConflictError
from .models import SyncEnvelope, SyncEnvelopeCreate, normalize_sync_timestamp

SYNC_MUTATION_GROUP_MAX_SIZE = 1_000


class StoredMutationGroupValidationError(SyncIdempotencyConflictError):
    """Safe persisted-group integrity failure."""

    def __init__(self, error_code: str, failing_step: int) -> None:
        super().__init__("Sync stored mutation group validation failed")
        self.error_code = error_code
        self.failing_step = failing_step


def _mutation_group_plan_hash(
    envelopes: Sequence[SyncEnvelope | SyncEnvelopeCreate],
    *,
    timestamp_format: str,
) -> str:
    encoded = json.dumps(
        [
            {
                "dataset_id": envelope.dataset_id,
                "client_envelope_id": envelope.client_envelope_id,
                "domain": envelope.domain,
                "operation": envelope.operation,
                "object_id": envelope.object_id,
                "device_id": envelope.device_id,
                "client_profile_id": envelope.client_profile_id,
                "client_sequence": envelope.client_sequence,
                "base_server_cursor": envelope.base_server_cursor,
                "base_object_revision": envelope.base_object_revision,
                "base_object_hash": envelope.base_object_hash,
                "object_revision": envelope.object_revision,
                "parent_id": envelope.parent_id,
                "schema_version": envelope.schema_version,
                "payload": envelope.payload,
                "payload_clear": envelope.payload_clear,
                "payload_ciphertext": envelope.payload_ciphertext,
                "payload_hash": envelope.payload_hash,
                "payload_size_bytes": envelope.payload_size_bytes,
                "created_at_client": (
                    envelope.created_at_client
                    if timestamp_format == "stored"
                    else _formatted_timestamp(
                        envelope.created_at_client,
                        utc_z=timestamp_format == "utc_z",
                    )
                ),
                "deleted": envelope.deleted,
                "encryption_metadata": envelope.encryption_metadata,
                "status": envelope.status,
                "stable_key": envelope.stable_key,
                "dependencies": envelope.dependencies,
                "routing_metadata": envelope.routing_metadata,
                "adapter_version": envelope.adapter_version,
                "base_version": envelope.base_version,
                "entity_version": envelope.entity_version,
                "mutation_group_id": envelope.mutation_group_id,
                "mutation_step": envelope.mutation_step,
                "mutation_step_count": envelope.mutation_step_count,
            }
            for envelope in envelopes
        ],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _formatted_timestamp(value: object | None, *, utc_z: bool) -> str | None:
    normalized = normalize_sync_timestamp(value)
    if utc_z and normalized is not None and normalized.endswith("+00:00"):
        return f"{normalized[:-6]}Z"
    return normalized


def mutation_group_plan_hash(
    envelopes: Sequence[SyncEnvelope | SyncEnvelopeCreate],
) -> str:
    """Return the canonical full-envelope fingerprint for a mutation plan."""

    return _mutation_group_plan_hash(envelopes, timestamp_format="canonical")


def materialization_group_view(
    envelopes: Sequence[SyncEnvelope],
) -> list[SyncEnvelope]:
    """Resolve immutable in-plan cursor markers for product materialization."""

    resolved: list[SyncEnvelope] = []
    prior_by_object: dict[tuple[str, str], SyncEnvelope] = {}
    for envelope in envelopes:
        current = envelope
        key = (envelope.domain, envelope.object_id)
        if envelope.base_server_cursor == 0:
            prior = prior_by_object.get(key)
            if (
                prior is None
                or prior.server_cursor is None
                or envelope.base_object_revision != prior.object_revision
                or envelope.base_object_hash != prior.payload_hash
            ):
                raise StoredMutationGroupValidationError(
                    "mutation_group_virtual_base_invalid",
                    envelope.mutation_step or 0,
                )
            current = replace(envelope, base_server_cursor=prior.server_cursor)
        resolved.append(current)
        prior_by_object[key] = current
    return resolved


def validate_stored_mutation_group(
    envelopes: Sequence[SyncEnvelope],
    *,
    dataset_id: str,
    mutation_group_id: str,
) -> None:
    """Validate persisted group shape and its canonical content fingerprint."""

    if not envelopes:
        raise StoredMutationGroupValidationError("mutation_group_step_missing", 0)
    if len(envelopes) > SYNC_MUTATION_GROUP_MAX_SIZE:
        raise StoredMutationGroupValidationError(
            "mutation_group_limit_exceeded",
            SYNC_MUTATION_GROUP_MAX_SIZE,
        )
    first = envelopes[0]
    expected_count = first.mutation_step_count
    if not isinstance(expected_count, int) or expected_count < 1:
        raise StoredMutationGroupValidationError("mutation_group_shape_invalid", 0)

    seen: set[int] = set()
    for index, envelope in enumerate(envelopes):
        step = envelope.mutation_step
        if not isinstance(step, int) or step < 0 or step >= expected_count:
            raise StoredMutationGroupValidationError("mutation_group_shape_invalid", index)
        if step in seen:
            raise StoredMutationGroupValidationError("mutation_group_step_duplicate", step)
        seen.add(step)
        if (
            envelope.dataset_id != dataset_id
            or envelope.mutation_group_id != mutation_group_id
            or envelope.mutation_step_count != expected_count
        ):
            raise StoredMutationGroupValidationError("mutation_group_shape_invalid", step)

    missing = sorted(set(range(expected_count)) - seen)
    if missing:
        raise StoredMutationGroupValidationError("mutation_group_step_missing", missing[0])
    if len(envelopes) != expected_count:
        raise StoredMutationGroupValidationError("mutation_group_shape_invalid", expected_count)

    plan_hash = first.mutation_plan_hash
    if not plan_hash:
        raise StoredMutationGroupValidationError("mutation_group_plan_hash_invalid", 0)
    for envelope in envelopes:
        if envelope.mutation_plan_hash != plan_hash:
            raise StoredMutationGroupValidationError(
                "mutation_group_plan_hash_invalid",
                envelope.mutation_step or 0,
            )
    valid_hashes = {
        mutation_group_plan_hash(envelopes),
        _mutation_group_plan_hash(envelopes, timestamp_format="stored"),
        _mutation_group_plan_hash(envelopes, timestamp_format="utc_z"),
    }
    if plan_hash not in valid_hashes:
        raise StoredMutationGroupValidationError("mutation_group_fingerprint_invalid", 0)


__all__ = [
    "SYNC_MUTATION_GROUP_MAX_SIZE",
    "StoredMutationGroupValidationError",
    "materialization_group_view",
    "mutation_group_plan_hash",
    "validate_stored_mutation_group",
]
