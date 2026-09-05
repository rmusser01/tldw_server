"""Versioned, content-free Personal Context ongoing-sync wire primitives."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

_OPAQUE_TOKEN = r"^[A-Za-z0-9._~-]{16,256}$"  # nosec B105
_SHA256_DIGEST = r"^[0-9a-f]{64}$"

PERSONAL_CONTEXT_ONGOING_ENDPOINTS = {
    "capabilities": ("GET", "/api/v1/sync/capabilities"),
    "activation_prepare": ("POST", "/api/v1/sync/personal-context/bootstrap"),
    "activation_acknowledge": (
        "POST",
        "/api/v1/sync/personal-context/activation/acknowledge",
    ),
    "push": ("POST", "/api/v1/sync/push"),
    "pull": ("GET", "/api/v1/sync/pull"),
    "conflict_list": ("GET", "/api/v1/sync/conflicts"),
    "conflict_resolve": ("POST", "/api/v1/sync/conflicts/resolve"),
    "purge": ("POST", "/api/v1/sync/personal-context/purge"),
}


class PersonalContextExchangeProof(BaseModel):
    """Opaque activation continuity proof required for ongoing sync version one."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ongoing_sync_version: Literal[1]
    activation_epoch: StrictStr = Field(min_length=16, max_length=256, pattern=_OPAQUE_TOKEN)
    continuity_token: StrictStr = Field(min_length=16, max_length=256, pattern=_OPAQUE_TOKEN)


class PersonalContextAuthorityMetadata(BaseModel):
    """Content-free authority identity associated with a Personal Context envelope."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True,
        json_schema_extra={
            "if": {"properties": {"role": {"const": "home_authority"}}},
            "then": {
                "required": [
                    "publication_batch_id", "profile_publication_sequence",
                    "batch_ordinal", "batch_size",
                ],
                "properties": {
                    field: {"not": {"type": "null"}}
                    for field in (
                        "publication_batch_id", "profile_publication_sequence",
                        "batch_ordinal", "batch_size",
                    )
                },
            },
            "else": {
                "properties": {
                    field: {"type": "null"}
                    for field in (
                        "publication_batch_id", "profile_publication_sequence",
                        "batch_ordinal", "batch_size",
                    )
                },
            },
        },
    )

    role: Literal["client_ingress", "home_authority"]
    publication_batch_id: StrictStr | None = Field(None, min_length=16, max_length=128)
    profile_publication_sequence: StrictInt | None = Field(None, ge=1)
    batch_ordinal: StrictInt | None = Field(None, ge=0)
    batch_size: StrictInt | None = Field(None, ge=1, le=100)

    @model_validator(mode="after")
    def validate_publication_fields(self) -> PersonalContextAuthorityMetadata:
        """Require publication identity only on server-origin authority envelopes."""

        publication_fields = (
            self.publication_batch_id,
            self.profile_publication_sequence,
            self.batch_ordinal,
            self.batch_size,
        )
        if self.role == "home_authority":
            if any(value is None for value in publication_fields):
                raise ValueError("home authority requires publication fields")
        elif any(value is not None for value in publication_fields):
            raise ValueError("client ingress forbids publication fields")
        return self


class PersonalContextRelayContinuation(BaseModel):
    """Content-free continuation state emitted with a Personal Context pull."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    state: Literal["complete", "personal_context_relay_pending", "relay_poisoned"]
    scan_watermark: StrictStr | None = Field(None, min_length=16, max_length=512)


class PersonalContextActivationReceipt(BaseModel):
    """Bounded receipt for activation preparation and acknowledgement."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    activation_id: StrictStr = Field(min_length=16, max_length=128)
    baseline_digest: StrictStr = Field(
        min_length=64,
        max_length=64,
        pattern=_SHA256_DIGEST,
    )
    purge_generation: StrictInt = Field(ge=0)
    publication_watermark: StrictInt = Field(ge=0)
    home_server_cursor: StrictInt = Field(ge=0)
    home_manifest_revision: StrictInt = Field(ge=0)
    home_manifest_version_id: StrictStr = Field(min_length=16, max_length=128)
    state: Literal["prepared", "installed", "acknowledged", "active"]


class PersonalContextCleanupAck(BaseModel):
    """Content-free acknowledgement that derived privacy cleanup completed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    object_id: StrictStr = Field(min_length=16, max_length=128)
    version_id: StrictStr = Field(min_length=16, max_length=128)
    purge_generation: StrictInt = Field(ge=0)
    server_cleanup_complete: StrictBool


class PersonalContextPurgeReceipt(BaseModel):
    """Bounded result of a signed device-originated global purge request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: StrictStr = Field(min_length=16, max_length=128)
    profile_id: StrictStr = Field(min_length=16, max_length=128)
    purge_generation: StrictInt = Field(ge=1)
    barrier_envelope_id: StrictStr = Field(min_length=16, max_length=128)
    state: Literal["accepted", "barrier_pending", "acknowledged"]


def validate_client_personal_context_metadata(
    metadata: PersonalContextAuthorityMetadata | None,
) -> None:
    """Reject server-only authority metadata supplied through public client ingress."""

    if metadata is not None and metadata.role == "home_authority":
        raise ValueError("client envelope cannot claim home authority")


def export_personal_context_ongoing_contract() -> dict[str, object]:
    """Export the version-one API contract without importing API models at startup."""

    from pydantic.json_schema import models_json_schema

    from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
        PersonalContextSyncCapabilitiesResponse,
        SyncConflictListResponse,
        SyncConflictResolveRequest,
        SyncConflictResolveResponse,
        SyncPersonalContextActivationAcknowledgeRequest,
        SyncPersonalContextActivationAcknowledgeResponse,
        SyncPersonalContextBootstrapRequest,
        SyncPersonalContextBootstrapResponse,
        SyncPersonalContextPurgeRequest,
        SyncPersonalContextPurgeResponse,
        SyncPullResponse,
        SyncPushRequest,
        SyncPushResponse,
    )

    model_classes = (
        PersonalContextCleanupAck,
        PersonalContextSyncCapabilitiesResponse,
        SyncPersonalContextBootstrapRequest,
        SyncPersonalContextBootstrapResponse,
        SyncPersonalContextActivationAcknowledgeRequest,
        SyncPersonalContextActivationAcknowledgeResponse,
        SyncPushRequest,
        SyncPushResponse,
        SyncPullResponse,
        SyncConflictListResponse,
        SyncConflictResolveRequest,
        SyncConflictResolveResponse,
        SyncPersonalContextPurgeRequest,
        SyncPersonalContextPurgeResponse,
    )
    _, schema = models_json_schema(
        [(model, "validation") for model in model_classes],
        title="tldw Personal Context ongoing sync v1",
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:tldw:personal-context:ongoing-sync:v1",
        "x-tldw-contract-version": 1,
        "x-tldw-endpoints": PERSONAL_CONTEXT_ONGOING_ENDPOINTS,
        **schema,
    }


__all__ = [
    "PERSONAL_CONTEXT_ONGOING_ENDPOINTS",
    "PersonalContextActivationReceipt",
    "PersonalContextAuthorityMetadata",
    "PersonalContextCleanupAck",
    "PersonalContextExchangeProof",
    "PersonalContextPurgeReceipt",
    "PersonalContextRelayContinuation",
    "export_personal_context_ongoing_contract",
    "validate_client_personal_context_metadata",
]
