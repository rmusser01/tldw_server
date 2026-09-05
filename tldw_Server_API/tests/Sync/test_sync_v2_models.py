import pytest
from pydantic import BaseModel, ValidationError

from tldw_Server_API.app.api.v1.endpoints.sync import _core_envelope_from_api
from tldw_Server_API.app.api.v1.schemas import sync_v2_models as api_sync_models
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    SYNC_V2_MAX_PUSH_ENVELOPES,
    SyncAttachmentUploadRequest,
    SyncAttachmentUploadResponse,
    SyncBackgroundDomainStatusResponse,
    SyncBackgroundLeaseRequest,
    SyncBackgroundLeaseResponse,
    SyncBackgroundPolicyPatchRequest,
    SyncBackgroundPolicyResponse,
    SyncBackgroundStatusResponse,
    SyncBlobChunkUploadResponse,
    SyncBlobDownloadManifestResponse,
    SyncBlobUploadCompleteResponse,
    SyncBlobUploadCreateRequest,
    SyncBlobUploadSessionResponse,
    SyncCapabilitiesResponse,
    SyncConflictListResponse,
    SyncConflictResolveRequest,
    SyncDatasetEnrollRequest,
    SyncKeyRecoveryBundleRecord,
    SyncKeyRecoveryBundleRequest,
    SyncPersonalContextActivationAcknowledgeRequest,
    SyncPersonalContextPurgeRequest,
    SyncPushRequest,
    SyncPushResponse,
    SyncRestoreCompletenessResponse,
    SyncV2Envelope,
)

ATTACHMENT_INTENT_ID = "2c4cb609-c4db-44f9-8e35-f078bd36d6b2"
ATTACHMENT_INTENT_NOTE_ID = "a1677eb1-1f41-4c86-a8dd-1eaa14b014e2"
from tldw_Server_API.app.core.Sync.v2 import models as core_sync_models
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope as CoreSyncEnvelope

M1_DOMAINS = ["notes.note", "chat.conversation", "chat.message", "attachment.ref"]
WORKSPACE_DOMAINS = ["workspaces.workspace", "workspaces.source_ref"]
SOURCE_CACHE_DOMAINS = ["source_cache.entry"]
MEDIA_DOMAINS = ["media.item", "media.keyword", "media.keyword_link"]
NOTES_ORGANIZATION_DOMAINS = (
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
)
NOTES_LINK_DOMAINS = ["notes.link"]
PERSONAL_CONTEXT_DOMAINS = [
    "personal_context.manifest",
    "personal_context.scope",
    "personal_context.record",
    "personal_context.proposal",
    "personal_context.purge",
]
SUPPORTED_DOMAINS = (
    M1_DOMAINS
    + WORKSPACE_DOMAINS
    + SOURCE_CACHE_DOMAINS
    + MEDIA_DOMAINS
    + list(NOTES_ORGANIZATION_DOMAINS)
    + NOTES_LINK_DOMAINS
    + PERSONAL_CONTEXT_DOMAINS
)


def _encryption_policy_model_classes():
    assert hasattr(api_sync_models, "SyncEncryptionPolicyMetadata")
    assert hasattr(core_sync_models, "SyncEncryptionPolicyMetadata")
    return (
        api_sync_models.SyncEncryptionPolicyMetadata,
        core_sync_models.SyncEncryptionPolicyMetadata,
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"object_revision": 7}, 7),
        ({"object_revision": True}, None),
        ({"object_revision": 0}, None),
        ({"object_revision": 2**63}, None),
        ({}, 1),
        (
            {
                "base_server_cursor": 8,
                "base_object_revision": 4,
                "base_object_hash": "sha256:base",
                "base_version": "record-v4",
            },
            5,
        ),
        ({"base_server_cursor": 8}, None),
        (
            {
                "base_server_cursor": True,
                "base_object_revision": 4,
                "base_object_hash": "sha256:base",
            },
            None,
        ),
        (
            {
                "base_server_cursor": 8,
                "base_object_revision": 0,
                "base_object_hash": "sha256:base",
            },
            None,
        ),
        (
            {
                "base_server_cursor": 8,
                "base_object_revision": 2**63 - 1,
                "base_object_hash": "sha256:base",
            },
            None,
        ),
        (
            {
                "base_server_cursor": 8,
                "base_object_revision": 4,
                "base_object_hash": "",
            },
            None,
        ),
        (
            {
                "base_server_cursor": 8,
                "base_object_revision": 4,
                "base_object_hash": "sha256:base",
                "base_version": 3,
            },
            None,
        ),
        ({"base_version": "unexpected-genesis-base"}, None),
    ],
)
def test_personal_context_ingress_result_revision_is_strict(
    values: dict[str, object],
    expected: int | None,
) -> None:
    fields: dict[str, object] = {
        "object_revision": None,
        "base_server_cursor": None,
        "base_object_revision": None,
        "base_object_hash": None,
        "base_version": None,
    }
    fields.update(values)

    assert core_sync_models.resolve_personal_context_ingress_result_revision(**fields) == expected


def _m1_envelope_payload(**overrides):
    payload = {
        "client_envelope_id": "env-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "client_profile_id": "chatbook-profile-1",
        "client_sequence": 17,
        "base_server_cursor": None,
        "base_object_revision": None,
        "base_object_hash": None,
        "server_cursor": None,
        "domain": "notes.note",
        "object_id": "note-1",
        "operation": "upsert",
        "parent_id": None,
        "schema_version": 1,
        "payload": {
            "title": "Research note",
            "content": "Canonical Markdown",
            "conversation_id": None,
            "message_id": None,
        },
        "payload_hash": "sha256:test",
        "object_revision": None,
        "created_at_client": "2026-05-23T18:12:44Z",
        "received_at_server": None,
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def test_capabilities_advertise_personal_and_workspace_domains_with_server_trusted_encryption():
    capabilities = SyncCapabilitiesResponse()

    assert capabilities.protocol_version == "sync-v2-m1"
    assert capabilities.domains == SUPPORTED_DOMAINS
    assert capabilities.operations == {
        "notes.note": ["upsert", "tombstone"],
        "chat.conversation": ["upsert", "tombstone"],
        "chat.message": ["append", "tombstone"],
        "attachment.ref": ["upsert", "tombstone"],
        "workspaces.workspace": ["upsert", "tombstone"],
        "workspaces.source_ref": ["upsert", "tombstone"],
        "source_cache.entry": ["upsert", "tombstone"],
        "media.item": ["upsert", "tombstone"],
        "media.keyword": ["upsert", "tombstone"],
        "media.keyword_link": ["upsert", "tombstone"],
        "notes.keyword": ["upsert", "tombstone"],
        "notes.keyword_link": ["upsert", "tombstone"],
        "notes.keyword_collection": ["upsert", "tombstone"],
        "notes.keyword_collection_link": ["upsert", "tombstone"],
        "notes.folder": ["upsert", "tombstone"],
        "notes.folder_link": ["upsert", "tombstone"],
        "notes.link": ["upsert", "tombstone"],
        "personal_context.manifest": ["upsert"],
        "personal_context.scope": ["upsert"],
        "personal_context.record": ["upsert", "tombstone"],
        "personal_context.proposal": ["upsert"],
        "personal_context.purge": ["tombstone"],
    }
    assert capabilities.encryption["policy"] == "server_trusted_v1"
    assert capabilities.encryption["ready"] is True
    assert capabilities.encryption_policies == ["server_trusted_v1"]
    assert capabilities.blob_transfer == {"supported": False}
    assert capabilities.domain_schemas["notes.note"] == {
        "schema_version": 1,
        "encryption_policy": "server_trusted_v1",
        "upsert": {
            "required": ["title", "content"],
            "properties": {
                "title": {"type": "string", "max_length": 255},
                "content": {"type": "string", "max_length": 5_000_000},
                "conversation_id": {"type": ["string", "null"]},
                "message_id": {"type": ["string", "null"]},
            },
            "additional_properties": False,
        },
        "tombstone": {"operation": "tombstone"},
        "restore": {
            "operation": "upsert",
            "routing_metadata": {"restore_intent": True},
            "requires_current_base": True,
        },
    }
    assert "client_private_v1" not in capabilities.model_dump_json()


def test_personal_context_capability_contract_is_typed_and_bounded() -> None:
    capabilities = SyncCapabilitiesResponse()

    assert set(PERSONAL_CONTEXT_DOMAINS).issubset(capabilities.domains)
    assert capabilities.personal_context.model_dump() == {
        "available": False,
        "blockers": ["personal_context_profile_key_unavailable"],
        "ongoing_sync_version": 0,
        "ongoing_sync_blockers": [],
        "activation_epoch": None,
        "continuity_token": None,
        "authorization_policy": "server_trusted_v1",
        "min_schema_version": 1,
        "max_schema_version": 1,
        "integrity_algorithm": "hmac-sha256-v1",
        "integrity_key_distribution": "wrapped-bootstrap-v1",
        "privacy_cleanup_ack": "personal-context-cleanup-v1",
        "purge_generation": "personal-context-purge-v1",
        "max_record_bytes": 16_384,
        "max_search_results": 20,
        "max_proposals_per_turn": 5,
        "max_proposals_per_session": 25,
        "max_unresolved_proposals": 200,
    }
    assert all(
        capabilities.supported_adapter_versions[domain] == []
        for domain in PERSONAL_CONTEXT_DOMAINS
    )


def test_personal_context_domains_match_core_and_api_literals() -> None:
    expected = tuple(PERSONAL_CONTEXT_DOMAINS)

    assert expected == core_sync_models.PERSONAL_CONTEXT_SYNC_DOMAINS
    assert expected == api_sync_models.PERSONAL_CONTEXT_SYNC_DOMAINS
    assert set(expected).issubset(core_sync_models.SyncDomain.__args__)
    assert set(expected).issubset(api_sync_models.SyncDomain.__args__)
    assert core_sync_models.PERSONAL_CONTEXT_SYNC_OPERATIONS == {
        "personal_context.manifest": ["upsert"],
        "personal_context.scope": ["upsert"],
        "personal_context.record": ["upsert", "tombstone"],
        "personal_context.proposal": ["upsert"],
        "personal_context.purge": ["tombstone"],
    }
    assert api_sync_models.PERSONAL_CONTEXT_SYNC_OPERATIONS == (
        core_sync_models.PERSONAL_CONTEXT_SYNC_OPERATIONS
    )


@pytest.mark.parametrize(
    ("domain", "operations"),
    [
        ("personal_context.manifest", {"upsert"}),
        ("personal_context.scope", {"upsert"}),
        ("personal_context.record", {"upsert", "tombstone"}),
        ("personal_context.proposal", {"upsert"}),
        ("personal_context.purge", {"tombstone"}),
    ],
)
def test_personal_context_schema_is_discoverable(
    domain: str,
    operations: set[str],
) -> None:
    schema = core_sync_models.sync_v2_domain_schemas()[domain]

    assert schema["schema_version"] == 1
    assert schema["encryption_policy"] == "server_trusted_v1"
    assert operations.issubset(schema)
    assert SyncCapabilitiesResponse().domain_schemas[domain] == schema


@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_notes_organization_schema_is_server_trusted_v1(domain: str) -> None:
    assert domain in core_sync_models.SyncDomain.__args__
    schema = core_sync_models.sync_v2_domain_schemas()[domain]
    assert schema["schema_version"] == 1
    assert schema["encryption_policy"] == "server_trusted_v1"
    assert {"upsert", "tombstone"}.issubset(schema)

    capability_schema = SyncCapabilitiesResponse().domain_schemas[domain]
    assert capability_schema == schema


def test_notes_link_schema_is_server_trusted_v1_and_separate_from_organization() -> None:
    assert "notes.link" in core_sync_models.SyncDomain.__args__
    assert "notes.link" not in core_sync_models.NOTES_ORGANIZATION_DOMAINS
    assert core_sync_models.NOTES_LINK_DOMAINS == ("notes.link",)
    assert core_sync_models.NOTES_LINK_SYNC_OPERATIONS == {
        "notes.link": ["upsert", "tombstone"]
    }

    schema = core_sync_models.sync_v2_domain_schemas()["notes.link"]
    assert schema["schema_version"] == 1
    assert schema["encryption_policy"] == "server_trusted_v1"
    assert set(schema) >= {"upsert", "tombstone"}
    upsert = schema["upsert"]
    assert upsert["properties"]["source_note_id"] == {
        "type": "string",
        "format": "uuid4",
        "canonical_lowercase": True,
    }
    assert upsert["properties"]["target_note_id"] == {
        "type": "string",
        "format": "uuid4",
        "canonical_lowercase": True,
    }
    assert upsert["constraints"] == {
        "distinct_endpoints": True,
        "undirected_endpoint_order": "source_note_id <= target_note_id",
    }
    assert schema["tombstone"]["constraints"] == upsert["constraints"]
    assert schema["tombstone"]["properties"]["reason"]["max_length"] == 256
    assert SyncCapabilitiesResponse().domain_schemas["notes.link"] == schema


def test_notes_task_domains_are_known_internally_but_not_supported_or_public() -> None:
    dormant = {"notes.task", "notes.task_activity"}

    assert dormant.issubset(set(core_sync_models.SyncDomain.__args__))
    assert dormant.isdisjoint(core_sync_models.SYNC_V2_SUPPORTED_DOMAINS)
    assert dormant.isdisjoint(core_sync_models.SYNC_V2_SUPPORTED_OPERATIONS)
    assert dormant.isdisjoint(core_sync_models.sync_v2_domain_schemas())
    assert dormant.isdisjoint(core_sync_models.sync_v2_server_supported_adapter_versions())
    assert dormant.isdisjoint(core_sync_models.sync_v2_dataset_writable_adapter_versions())

    private_schemas = core_sync_models._sync_v2_internal_domain_schemas()
    assert set(private_schemas) >= dormant
    assert private_schemas["notes.task"]["schema_version"] == 1
    assert private_schemas["notes.task"]["operations"] == ["upsert", "tombstone"]
    assert private_schemas["notes.task_activity"]["schema_version"] == 1
    assert private_schemas["notes.task_activity"]["operations"] == [
        "upsert",
        "tombstone",
    ]


def test_notes_moodboard_studio_domains_are_known_but_strictly_dormant() -> None:
    dormant = {
        "notes.moodboard",
        "notes.moodboard_note",
        "notes.studio_document",
    }

    assert core_sync_models.NOTES_MOODBOARD_STUDIO_DOMAINS == (
        "notes.moodboard",
        "notes.moodboard_note",
        "notes.studio_document",
    )
    assert dormant.issubset(set(core_sync_models.SyncDomain.__args__))
    assert dormant.issubset(core_sync_models.SYNC_V2_KNOWN_DOMAINS)
    assert dormant.issubset(core_sync_models.SYNC_V2_INTERNAL_OPERATIONS)
    assert dormant.isdisjoint(core_sync_models.SYNC_V2_SUPPORTED_DOMAINS)
    assert dormant.isdisjoint(core_sync_models.SYNC_V2_SUPPORTED_OPERATIONS)
    assert dormant.isdisjoint(core_sync_models.sync_v2_domain_schemas())
    assert dormant.isdisjoint(core_sync_models.sync_v2_server_supported_adapter_versions())
    assert dormant.isdisjoint(core_sync_models.sync_v2_dataset_writable_adapter_versions())

    private_schemas = core_sync_models._sync_v2_internal_domain_schemas()
    assert dormant.issubset(private_schemas)
    for domain in dormant:
        assert private_schemas[domain]["schema_version"] == 1
        assert private_schemas[domain]["operations"] == ["upsert", "tombstone"]

    assert dormant.isdisjoint(SyncCapabilitiesResponse().domains)
    assert dormant.isdisjoint(SyncCapabilitiesResponse().operations)
    assert dormant.isdisjoint(SyncCapabilitiesResponse().domain_schemas)

    with pytest.raises(ValidationError):
        api_sync_models.SyncDeviceRegisterRequest.model_validate(
            {
                "display_name": "Premature client",
                "supported_domains": ["notes.moodboard"],
            }
        )
    with pytest.raises(ValidationError):
        api_sync_models.SyncProfileBootstrapRequest.model_validate(
            {
                "mode": "offline_sync",
                "requested_domains": ["notes.studio_document"],
            }
        )


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": "11111111-1111-4111-8111-111111111111",
                "keyword_sync_id": "22222222-2222-4222-8222-222222222222",
            },
        ),
        (
            "notes.keyword_collection_link",
            {
                "collection_sync_id": "33333333-3333-4333-8333-333333333333",
                "keyword_sync_id": "22222222-2222-4222-8222-222222222222",
            },
        ),
        (
            "notes.folder_link",
            {
                "note_id": "11111111-1111-4111-8111-111111111111",
                "folder_sync_id": "44444444-4444-4444-8444-444444444444",
            },
        ),
    ],
)
def test_link_tombstone_capability_payload_parses_strictly(
    domain: str,
    payload: dict[str, object],
) -> None:
    from tldw_Server_API.app.core.Sync.v2.notes_organization import (
        parse_notes_organization_payload,
    )

    descriptor = SyncCapabilitiesResponse().domain_schemas[domain]["tombstone"]
    derived_payload = {field: payload[field] for field in descriptor["required"]}

    assert parse_notes_organization_payload(
        domain, "tombstone", derived_payload
    ) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {
            "title": 42,
            "content": "Markdown",
            "conversation_id": None,
            "message_id": None,
        },
        {
            "title": "Research note",
            "content": ["not", "text"],
            "conversation_id": None,
            "message_id": None,
        },
        {
            "title": "x" * 256,
            "content": "Markdown",
            "conversation_id": None,
            "message_id": None,
        },
        {
            "title": "Research note",
            "content": "x" * 5_000_001,
            "conversation_id": None,
            "message_id": None,
        },
        {
            "title": "Research note",
            "content": "Markdown",
            "conversation_id": 17,
            "message_id": None,
        },
        {
            "title": "Research note",
            "content": "Markdown",
            "conversation_id": None,
            "message_id": {"id": "message-1"},
        },
    ],
)
def test_notes_note_upsert_schema_rejects_wrong_types_and_oversized_payloads(payload):
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(_m1_envelope_payload(payload=payload))


def test_capabilities_normalize_legacy_supported_domains_to_supported_defaults():
    capabilities = SyncCapabilitiesResponse.model_validate(
        {"supported_domains": ["notes", "chat", "source_cache", "media"]}
    )

    assert capabilities.domains == SUPPORTED_DOMAINS


def test_dataset_enroll_request_defaults_to_m1_personal_server_trusted_dataset():
    request = SyncDatasetEnrollRequest.model_validate({})

    assert request.scope_type == "personal"
    assert request.domains == M1_DOMAINS
    assert request.encryption_policy == "server_trusted_v1"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "policy": "server_trusted_v1",
            "key_epoch": 1,
            "attestation": {
                "scope": "user_database_directory",
                "covers": ["Sync_v2.db", "ChaChaNotes.db"],
                "configured": True,
            },
        },
        {
            "policy": "passphrase_wrapped_v1",
            "key_epoch": 2,
            "kdf_metadata": {
                "algorithm": "argon2id",
                "params_hash": "sha256:kdf-params",
            },
            "recovery_key_record_id": "key-passphrase-1",
        },
        {
            "policy": "device_wrapped_v1",
            "key_epoch": 3,
            "device_key_record_ids": ["key-device-1"],
        },
        {
            "policy": "client_private_v1",
            "key_epoch": 4,
            "server_materialization": "metadata_only",
        },
    ],
)
def test_encryption_policy_metadata_accepts_m3_policy_modes(payload):
    api_policy_cls, core_policy_cls = _encryption_policy_model_classes()

    api_policy = api_policy_cls.model_validate(payload)
    core_policy = core_policy_cls(**payload)

    assert api_policy.policy == payload["policy"]
    assert core_policy.policy == payload["policy"]
    assert api_policy.key_epoch == payload["key_epoch"]
    assert core_policy.key_epoch == payload["key_epoch"]
    assert "wrapped_key_blob" not in api_policy.model_dump()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "policy": "server_trusted_v1",
            "attestation": {"configured": False},
        },
        {
            "policy": "passphrase_wrapped_v1",
            "kdf_metadata": {"algorithm": "argon2id"},
            "recovery_key_record_id": "key-passphrase-1",
        },
        {
            "policy": "passphrase_wrapped_v1",
            "kdf_metadata": {
                "algorithm": "argon2id",
                "params_hash": "sha256:kdf-params",
            },
        },
        {
            "policy": "device_wrapped_v1",
            "device_key_record_ids": [],
        },
        {
            "policy": "client_private_v1",
            "server_materialization": "allowed",
        },
        {
            "policy": "client_private_v1",
            "key_epoch": 0,
            "server_materialization": "metadata_only",
        },
    ],
)
def test_encryption_policy_metadata_rejects_incomplete_or_unsafe_modes(payload):
    api_policy_cls, core_policy_cls = _encryption_policy_model_classes()

    with pytest.raises(ValidationError):
        api_policy_cls.model_validate(payload)
    with pytest.raises(ValueError):
        core_policy_cls(**payload)


def test_encryption_policy_metadata_rejects_secret_key_material_fields():
    api_policy_cls, core_policy_cls = _encryption_policy_model_classes()
    payload = {
        "policy": "passphrase_wrapped_v1",
        "key_epoch": 1,
        "kdf_metadata": {
            "algorithm": "argon2id",
            "params_hash": "sha256:kdf-params",
        },
        "recovery_key_record_id": "key-passphrase-1",
        "wrapped_key_blob": "wrapped:secret-key-material",
    }

    with pytest.raises(ValidationError) as api_error:
        api_policy_cls.model_validate(payload)
    with pytest.raises(TypeError) as core_error:
        core_policy_cls(**payload)

    assert "wrapped:secret-key-material" not in str(api_error.value)
    assert "wrapped:secret-key-material" not in str(core_error.value)


def test_key_recovery_bundle_models_include_epoch_rotation_metadata():
    request = SyncKeyRecoveryBundleRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "wrapped_key_blob": "wrapped:opaque",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "opaque-salt"},
            "encryption_policy": "passphrase_wrapped_v1",
            "key_epoch": 4,
            "active_from_server_sequence": 17,
            "superseded_at": "2026-05-10T12:30:00+00:00",
            "wrapped_for": "passphrase",
            "rewrap_status": "pending",
        }
    )
    record = SyncKeyRecoveryBundleRecord.model_validate(
        {
            "key_record_id": "key-1",
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:opaque",
            "encryption_policy": "device_wrapped_v1",
            "key_epoch": 2,
            "active_from_server_sequence": 19,
            "wrapped_for": "device",
            "rewrap_status": "complete",
        }
    )

    assert request.encryption_policy == "passphrase_wrapped_v1"
    assert request.key_epoch == 4
    assert request.active_from_server_sequence == 17
    assert request.superseded_at == "2026-05-10T12:30:00+00:00"
    assert request.wrapped_for == "passphrase"
    assert request.rewrap_status == "pending"
    assert record.encryption_policy == "device_wrapped_v1"
    assert record.key_epoch == 2
    assert record.active_from_server_sequence == 19
    assert record.wrapped_for == "device"
    assert record.rewrap_status == "complete"


@pytest.mark.parametrize(
    "payload",
    [
        {"key_epoch": 0},
        {"active_from_server_sequence": -1},
        {"wrapped_for": "plaintext"},
        {"rewrap_status": "leaked"},
        {"encryption_policy": "legacy"},
    ],
)
def test_key_recovery_bundle_models_reject_invalid_epoch_rotation_metadata(payload):
    request_payload = {
        "dataset_id": "dataset-1",
        "wrapped_key_blob": "wrapped:opaque",
    }
    request_payload.update(payload)

    with pytest.raises(ValidationError):
        SyncKeyRecoveryBundleRequest.model_validate(request_payload)


def test_key_rotation_models_validate_redacted_preview_and_commit_shapes():
    assert hasattr(api_sync_models, "SyncKeyRotationPreviewRequest")
    assert hasattr(api_sync_models, "SyncKeyRotationCommitRequest")
    assert hasattr(api_sync_models, "SyncKeyRotationResponse")

    preview_request = api_sync_models.SyncKeyRotationPreviewRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "source_key_record_ids": ["key-1"],
        }
    )
    commit_request = api_sync_models.SyncKeyRotationCommitRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "rotation_id": "rotation-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "wrapped_key_blob": "wrapped:super-secret-key-material",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "secret-salt"},
            "source_key_record_ids": ["key-1"],
            "wrapped_for": "passphrase",
        }
    )
    response = api_sync_models.SyncKeyRotationResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "next_key_epoch": 2,
            "active_from_server_sequence": 8,
            "can_commit": True,
            "committed": False,
            "affected_key_records": [
                {
                    "key_record_id": "key-1",
                    "key_epoch": 1,
                    "encryption_policy": "server_trusted_v1",
                    "wrapped_for": "recovery",
                    "rewrap_status": "not_required",
                }
            ],
            "new_key_record": {
                "key_record_id": "key-rotation-1",
                "key_epoch": 2,
                "encryption_policy": "passphrase_wrapped_v1",
                "wrapped_for": "passphrase",
                "rewrap_status": "complete",
            },
            "retained_envelope_range": {
                "from_server_sequence": 1,
                "through_server_sequence": 7,
            },
        }
    )

    assert preview_request.source_key_record_ids == ["key-1"]
    assert commit_request.rotation_id == "rotation-1"
    assert commit_request.rewrap_status == "complete"
    assert response.next_key_epoch == 2
    assert response.affected_key_records[0].key_record_id == "key-1"
    assert "wrapped_key_blob" not in response.model_dump_json()
    assert "kdf_metadata" not in response.model_dump_json()
    assert "super-secret-key-material" not in response.model_dump_json()
    assert "secret-salt" not in response.model_dump_json()


@pytest.mark.parametrize(
    "payload",
    [
        {"rotation_id": ""},
        {"target_encryption_policy": "legacy"},
        {"key_epoch": 0},
        {"wrapped_for": "plaintext"},
        {"rewrap_status": "leaked"},
    ],
)
def test_key_rotation_commit_request_rejects_invalid_metadata(payload):
    assert hasattr(api_sync_models, "SyncKeyRotationCommitRequest")
    request_payload = {
        "dataset_id": "dataset-1",
        "rotation_id": "rotation-1",
        "wrapped_key_blob": "wrapped:opaque",
        "kdf_metadata": {"algorithm": "scrypt", "salt": "opaque-salt"},
    }
    request_payload.update(payload)

    with pytest.raises(ValidationError):
        api_sync_models.SyncKeyRotationCommitRequest.model_validate(request_payload)


def test_dataset_enroll_request_accepts_explicit_workspace_metadata_domains():
    request = SyncDatasetEnrollRequest.model_validate(
        {
            "scope_type": "workspace",
            "workspace_id": "workspace-1",
            "domains": WORKSPACE_DOMAINS,
        }
    )

    assert request.scope_type == "workspace"
    assert request.workspace_id == "workspace-1"
    assert request.domains == WORKSPACE_DOMAINS


def test_dataset_enroll_request_accepts_explicit_source_cache_domain():
    request = SyncDatasetEnrollRequest.model_validate(
        {"domains": SOURCE_CACHE_DOMAINS}
    )

    assert request.scope_type == "personal"
    assert request.domains == SOURCE_CACHE_DOMAINS


def test_dataset_enroll_request_accepts_explicit_media_metadata_domains():
    request = SyncDatasetEnrollRequest.model_validate(
        {"domains": MEDIA_DOMAINS}
    )

    assert request.scope_type == "personal"
    assert request.domains == MEDIA_DOMAINS


def test_background_policy_lease_and_status_models_validate_m3_shapes():
    policy = SyncBackgroundPolicyResponse.model_validate(
        {"dataset_id": "dataset-1", "device_id": "device-1"}
    )
    patch = SyncBackgroundPolicyPatchRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "enabled": False,
            "paused_reason": "user_paused",
            "pending_local_changes": True,
        }
    )
    lease_request = SyncBackgroundLeaseRequest.model_validate(
        {"dataset_id": "dataset-1", "device_id": "device-1", "ttl_seconds": 120}
    )
    lease = SyncBackgroundLeaseResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "lease_id": "lease-1",
            "status": "acquired",
            "acquired": True,
            "expires_at": "2026-05-23T18:02:00+00:00",
            "updated_at": "2026-05-23T18:00:00+00:00",
        }
    )
    status = SyncBackgroundStatusResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "policy": policy.model_dump(),
            "lease": lease.model_dump(),
            "quota_pressure": {
                "reserved_blob_bytes": 8,
                "used_blob_bytes": 16,
                "limit_bytes": 100,
                "pressure_ratio": 0.24,
            },
            "restore_completeness": "content_complete",
            "domains": [
                SyncBackgroundDomainStatusResponse(
                    domain="notes.note",
                    last_server_sequence=3,
                    last_pulled_sequence=1,
                    cursor_lag_count=2,
                    unresolved_conflicts=1,
                    replayable_failures=1,
                    last_successful_push_at="2026-05-23T18:01:00+00:00",
                    last_successful_pull_at="2026-05-23T18:02:00+00:00",
                ).model_dump()
            ],
        }
    )

    assert policy.enabled is True
    assert policy.minimum_interval_seconds == 300
    assert policy.backoff_floor_seconds == 60
    assert policy.respect_metered_networks is True
    assert patch.enabled is False
    assert patch.pending_local_changes is True
    assert lease_request.lease_id is None
    assert lease.acquired is True
    assert status.domains[0].cursor_lag_count == 2

    with pytest.raises(ValidationError):
        SyncBackgroundLeaseRequest.model_validate(
            {"dataset_id": "dataset-1", "device_id": "device-1", "ttl_seconds": 0}
        )


def test_sync_envelope_accepts_m1_fields_and_legacy_transition_aliases():
    envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            entity_id="note-from-old-client",
            object_id=None,
            server_sequence=101,
            payload=None,
            payload_clear={
                "title": "Legacy payload alias",
                "content": "Canonical note body",
            },
        )
    )

    assert envelope.object_id == "note-from-old-client"
    assert envelope.entity_id == "note-from-old-client"
    assert envelope.server_cursor == 101
    assert envelope.server_sequence == 101
    assert envelope.payload == {
        "title": "Legacy payload alias",
        "content": "Canonical note body",
    }
    assert envelope.payload_clear == envelope.payload
    assert envelope.client_sequence == 17
    assert envelope.encryption_metadata == {"policy": "server_trusted_v1"}


@pytest.mark.parametrize(("value", "expected_type"), [("version-1", str), (1, int)])
def test_sync_envelope_entity_version_preserves_wire_type(
    value: object,
    expected_type: type,
) -> None:
    api_envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(entity_version=value)
    )

    assert type(api_envelope.entity_version) is expected_type
    assert type(_core_envelope_from_api(api_envelope).entity_version) is expected_type


@pytest.mark.parametrize("value", [True, 1.0])
def test_sync_envelope_entity_version_rejects_coercible_non_wire_types(
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(_m1_envelope_payload(entity_version=value))


def test_core_sync_mutation_group_metadata_round_trips() -> None:
    expected_sha256 = "a" * 64
    create = core_sync_models.SyncEnvelopeCreate(
        dataset_id="dataset-1",
        client_envelope_id="env-group-0",
        domain="notes.note",
        operation="upsert",
        object_id="note-1",
        mutation_group_id="mutation-group-1",
        mutation_step=0,
        mutation_step_count=3,
        mutation_plan_hash=expected_sha256,
    )
    envelope = CoreSyncEnvelope(
        dataset_id=create.dataset_id,
        client_envelope_id=create.client_envelope_id,
        domain=create.domain,
        operation=create.operation,
        object_id=create.object_id,
        server_cursor=101,
        mutation_group_id=create.mutation_group_id,
        mutation_step=create.mutation_step,
        mutation_step_count=create.mutation_step_count,
        mutation_plan_hash=create.mutation_plan_hash,
    )

    assert envelope.mutation_group_id == "mutation-group-1"
    assert envelope.mutation_step == 0
    assert envelope.mutation_step_count == 3
    assert envelope.mutation_plan_hash == expected_sha256


def test_core_sync_mutation_group_metadata_allows_legacy_absence() -> None:
    create = core_sync_models.SyncEnvelopeCreate(
        dataset_id="dataset-1",
        client_envelope_id="env-legacy",
        domain="notes.note",
        operation="upsert",
        object_id="note-1",
    )
    envelope = CoreSyncEnvelope(
        dataset_id=create.dataset_id,
        client_envelope_id=create.client_envelope_id,
        domain=create.domain,
        operation=create.operation,
        object_id=create.object_id,
        server_cursor=101,
    )

    assert create.mutation_group_id is None
    assert create.mutation_step is None
    assert create.mutation_step_count is None
    assert create.mutation_plan_hash is None
    assert envelope.mutation_group_id is None
    assert envelope.mutation_step is None
    assert envelope.mutation_step_count is None
    assert envelope.mutation_plan_hash is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"mutation_group_id": "group-1"},
        {
            "mutation_group_id": "   ",
            "mutation_step": 0,
            "mutation_step_count": 1,
            "mutation_plan_hash": "a" * 64,
        },
        {
            "mutation_group_id": "group-1",
            "mutation_step": -1,
            "mutation_step_count": 1,
            "mutation_plan_hash": "a" * 64,
        },
        {
            "mutation_group_id": "group-1",
            "mutation_step": 1,
            "mutation_step_count": 1,
            "mutation_plan_hash": "a" * 64,
        },
        {
            "mutation_group_id": "group-1",
            "mutation_step": 0,
            "mutation_step_count": 0,
            "mutation_plan_hash": "a" * 64,
        },
        {
            "mutation_group_id": "group-1",
            "mutation_step": 0,
            "mutation_step_count": 1,
            "mutation_plan_hash": "A" * 64,
        },
        {
            "mutation_group_id": "group-1",
            "mutation_step": 0,
            "mutation_step_count": 1,
            "mutation_plan_hash": "a" * 63,
        },
    ],
)
@pytest.mark.parametrize("stored", [False, True])
def test_core_sync_mutation_group_metadata_rejects_partial_or_invalid_values(
    overrides: dict[str, object],
    stored: bool,
) -> None:
    values = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-group-0",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        **overrides,
    }
    model = CoreSyncEnvelope if stored else core_sync_models.SyncEnvelopeCreate
    if stored:
        values["server_cursor"] = 101

    with pytest.raises(ValueError, match="mutation group"):
        model(**values)


def test_sync_envelope_accepts_source_cache_entry_domain():
    envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="source_cache.entry",
            object_id="source-1:sha256-source",
            stable_key="source_cache.entry:source-1:sha256-source",
            payload={
                "source_id": "source-1",
                "content_hash": "sha256:source",
                "provenance": {"kind": "url", "uri": "https://example.test/source"},
            },
            payload_hash="sha256:source-cache-entry",
        )
    )

    assert envelope.domain == "source_cache.entry"
    assert envelope.operation == "upsert"


def test_sync_envelope_accepts_media_metadata_domains():
    item = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="media.item",
            object_id="media-1",
            stable_key="media.item:media-1",
            payload={"media_id": "media-1", "media_type": "video", "title": "Lecture"},
            payload_hash="sha256:media-item",
        )
    )
    keyword = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="media.keyword",
            object_id="keyword-1",
            stable_key="media.keyword:keyword-1",
            payload={"keyword_id": "keyword-1", "name": "research"},
            payload_hash="sha256:media-keyword",
        )
    )
    link = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="media.keyword_link",
            object_id="media-1:keyword-1",
            stable_key="media.keyword_link:media-1:keyword-1",
            payload={"media_id": "media-1", "keyword_id": "keyword-1"},
            payload_hash="sha256:media-keyword-link",
        )
    )

    assert [item.domain, keyword.domain, link.domain] == MEDIA_DOMAINS


def test_sync_envelope_rejects_legacy_source_cache_domain():
    with pytest.raises(ValidationError) as exc_info:
        SyncV2Envelope.model_validate(_m1_envelope_payload(domain="source_cache"))

    assert "source_cache" in str(exc_info.value)


def test_sync_envelope_rejects_legacy_media_domain():
    with pytest.raises(ValidationError) as exc_info:
        SyncV2Envelope.model_validate(_m1_envelope_payload(domain="media"))

    assert "media" in str(exc_info.value)


def test_sync_envelope_rejects_client_private_as_m1_default():
    with pytest.raises(ValidationError) as exc_info:
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(encryption_metadata={"policy": "client_private_v1"})
        )

    assert "client_private_v1" in str(exc_info.value)


def test_sync_envelope_rejects_non_object_payload_as_validation_error():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(_m1_envelope_payload(payload=[]))


def test_push_request_accepts_locked_contract_cursor_and_options():
    request = SyncPushRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "client_profile_id": "profile-1",
            "base_server_cursor": 128,
            "envelopes": [],
            "options": {"stop_on_conflict": True},
        }
    )

    assert request.client_profile_id == "profile-1"
    assert request.base_server_cursor == 128
    assert request.options.stop_on_conflict is True


def test_sync_v2_models_exports_push_options_for_star_imports():
    assert "SyncPushOptions" in api_sync_models.__all__


def test_whole_object_tombstone_requires_base_metadata():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                operation="tombstone",
                deleted=True,
                base_server_cursor=None,
                base_object_revision=None,
                base_object_hash=None,
            )
        )


def test_whole_object_update_requires_complete_base_metadata():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(base_server_cursor=98, base_object_revision=4)
        )


def test_chat_message_append_requires_stable_message_id_and_payload_hash():
    valid = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="chat.message",
            operation="append",
            object_id="msg-1",
            parent_id="conversation-1",
            payload_hash="sha256:message",
        )
    )

    assert valid.object_id == "msg-1"
    assert valid.payload_hash == "sha256:message"

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                domain="chat.message",
                operation="append",
                object_id="",
                payload_hash="sha256:message",
            )
        )

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                domain="chat.message",
                operation="append",
                object_id="msg-1",
                payload_hash="",
            )
        )


def test_core_sync_envelope_accepts_server_sequence_transition_alias():
    envelope = CoreSyncEnvelope(
        server_sequence=42,
        dataset_id="dataset-1",
        client_envelope_id="env-1",
        domain="notes.note",
        operation="upsert",
        object_id="note-1",
        payload={"title": "Alias"},
        payload_hash="sha256:alias",
    )

    assert envelope.server_cursor == 42
    assert envelope.server_sequence == 42


def test_core_envelope_mapping_strips_api_only_server_fields_before_persistence():
    api_envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            envelope_id="srv_env_000000000001",
            server_cursor=1,
            received_at_server="2026-05-23T18:12:46Z",
            status="accepted",
            apply_status="applied",
        )
    )

    core_envelope = _core_envelope_from_api(api_envelope)

    assert core_envelope.client_envelope_id == "env-1"
    assert core_envelope.object_id == "note-1"
    assert core_envelope.server_cursor is None
    assert core_envelope.received_at_server is None
    assert core_envelope.status == "accepted"
    assert core_envelope.apply_status == "pending"


def test_api_sync_responses_accept_terminal_superseded_apply_status():
    envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            envelope_id="srv_env_000000000001",
            server_cursor=1,
            status="accepted",
            apply_status="superseded",
        )
    )
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "accepted": [
                {
                    "client_envelope_id": "env-1",
                    "server_cursor": 1,
                    "apply_status": "superseded",
                }
            ],
        }
    )

    assert envelope.apply_status == "superseded"
    assert response.accepted[0].apply_status == "superseded"


def test_conflict_resolution_request_uses_locked_m1_batch_shape():
    request = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "resolutions": [
                {
                    "conflict_id": "conflict-note-1",
                    "action": "overwrite",
                    "resolution_envelope": None,
                },
                {
                    "conflict_id": "conflict-note-2",
                    "action": "duplicate_rename",
                    "resolution_envelope": _m1_envelope_payload(
                        client_envelope_id="env-resolution",
                        object_id="note-copy",
                        payload_hash="sha256:copy",
                    ),
                },
                {
                    "conflict_id": "conflict-note-3",
                    "action": "skip",
                },
            ],
        }
    )

    assert request.dataset_id == "dataset-1"
    assert request.device_id == "device-1"
    assert [resolution.action for resolution in request.resolutions] == [
        "overwrite",
        "duplicate_rename",
        "skip",
    ]
    assert request.resolutions[1].resolution_envelope.object_id == "note-copy"


@pytest.mark.parametrize(
    "legacy_action",
    ["accept_local", "accept_remote", "merge", "dismiss"],
)
def test_conflict_resolution_request_rejects_pre_m1_actions(legacy_action: str):
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": legacy_action,
                    }
                ],
            }
        )


def test_conflict_resolution_request_rejects_legacy_single_conflict_shape():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "conflict_id": "conflict-1",
                "action": "accept_local",
                "resolved_by_device_id": "device-1",
            }
        )


def test_conflict_resolution_request_rejects_duplicate_rename_without_envelope():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": "duplicate_rename",
                    }
                ],
            }
        )


def test_conflict_resolution_request_rejects_skip_with_resolution_envelope():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": "skip",
                        "resolution_envelope": _m1_envelope_payload(
                            client_envelope_id="env-skip-resolution",
                            object_id="note-1",
                            payload_hash="sha256:skip-resolution",
                        ),
                    }
                ],
            }
        )


def test_conflict_resolution_rejects_client_home_authority_claim() -> None:
    with pytest.raises(ValidationError, match="home authority"):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": "duplicate_rename",
                        "resolution_envelope": _m1_envelope_payload(
                            client_envelope_id="env-resolution-home-authority",
                            domain="personal_context.record",
                            object_id="note-copy",
                            authority={
                                "role": "home_authority",
                                "publication_batch_id": "batch_0123456789abcdef",
                                "profile_publication_sequence": 1,
                                "batch_ordinal": 0,
                                "batch_size": 1,
                            },
                        ),
                    }
                ],
            }
        )


def test_conflict_batch_endpoint_resolves_locked_m1_request_shape():
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
    from tldw_Server_API.app.api.v1.endpoints.sync import resolve_sync_v2_conflicts
    from tldw_Server_API.app.core.Sync.v2.models import SyncConflict

    class FakeSyncService:
        def __init__(self):
            self.calls = []

        def resolve_conflicts_batch(self, **kwargs):
            results = []
            for index, resolution in enumerate(kwargs["resolutions"]):
                conflict_id, public_action, resolution_envelope, *_fields = resolution
                call = {
                    "conflict_id": conflict_id,
                    "dataset_id": kwargs["dataset_id"],
                    "action": public_action,
                    "resolution_envelope": resolution_envelope,
                }
                self.calls.append(call)
                server_cursor = 123 if public_action == "duplicate_rename" else 12
                envelope_id = (
                    "srv_env_000000000123"
                    if public_action == "duplicate_rename"
                    else None
                )
                results.append(
                    (
                        index,
                        SyncConflict(
                            conflict_id=conflict_id,
                            dataset_id="dataset-1",
                            domain="notes.note",
                            object_id="note-1",
                            conflict_type="version_divergence",
                            status=(
                                "dismissed" if public_action == "skip" else "resolved"
                            ),
                            base_envelope_id=None,
                            local_envelope_id=None,
                            remote_envelope_id=None,
                            server_cursor=server_cursor,
                            metadata={},
                            created_at="2026-05-23T18:12:44Z",
                            resolved_at="2026-05-23T18:13:44Z",
                            resolved_by_device_id=kwargs["device_id"],
                            resolved_by_envelope_id=envelope_id,
                            resolution_action=public_action,
                        ),
                    )
                )
            return results, [], None

    service = FakeSyncService()
    request = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "resolutions": [
                {"conflict_id": "conflict-1", "action": "overwrite"},
                {
                    "conflict_id": "conflict-2",
                    "action": "duplicate_rename",
                    "resolution_envelope": _m1_envelope_payload(
                        client_envelope_id="env-resolution",
                        object_id="note-copy",
                        payload_hash="sha256:copy",
                    ),
                },
                {"conflict_id": "conflict-3", "action": "skip"},
            ],
        }
    )

    response = resolve_sync_v2_conflicts(
        request,
        user=User(id="user-1", username="user-1"),
        service=service,
    )

    assert response.dataset_id == "dataset-1"
    assert [item.action for item in response.resolved] == [
        "overwrite",
        "duplicate_rename",
        "skip",
    ]
    assert response.rejected == []
    assert [call["conflict_id"] for call in service.calls] == [
        "conflict-1",
        "conflict-2",
        "conflict-3",
    ]
    assert [call["dataset_id"] for call in service.calls] == [
        "dataset-1",
        "dataset-1",
        "dataset-1",
    ]
    assert service.calls[1]["resolution_envelope"].client_envelope_id == "env-resolution"
    assert service.calls[1].get("resolved_by_envelope_id") is None
    assert service.calls[2]["action"] == "skip"
    response_payload = response.model_dump(exclude_none=True)
    assert response_payload["resolved"][1]["envelope_id"] == "srv_env_000000000123"
    assert response_payload["resolved"][1]["server_cursor"] == 123
    assert response_payload["server_cursor"] == 123
    assert "resolved_by_envelope_id" not in response_payload["resolved"][1]


def test_push_response_reports_per_envelope_outcomes():
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "env-1", "server_cursor": 1}],
            "rejected": [
                {
                    "client_envelope_id": "env-2",
                    "error_code": "unsupported_adapter_version",
                    "message": "adapter version is not supported",
                }
            ],
            "conflicts": [
                {
                    "conflict_id": "conflict-1",
                    "client_envelope_id": "env-3",
                    "object_id": "note-1",
                    "domain": "notes.note",
                    "server_cursor": 2,
                }
            ],
            "next_cursor": "2",
        }
    )

    assert response.accepted[0].server_cursor == 1
    assert response.rejected[0].error_code == "unsupported_adapter_version"
    assert response.conflicts[0].conflict_id == "conflict-1"


def test_push_request_allows_dataset_mismatch_for_per_envelope_outcomes():
    request = SyncPushRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                _m1_envelope_payload(
                    client_envelope_id="env-2",
                    dataset_id="dataset-2",
                )
            ],
        }
    )

    assert request.envelopes[0].dataset_id == "dataset-2"


def test_push_request_cannot_set_server_assigned_mutation_group_metadata() -> None:
    request = SyncPushRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                _m1_envelope_payload(
                    mutation_group_id="client-group",
                    mutation_step=0,
                    mutation_step_count=1,
                    mutation_plan_hash="a" * 64,
                )
            ],
        }
    )

    core = _core_envelope_from_api(request.envelopes[0])

    assert core.mutation_group_id is None
    assert core.mutation_step is None
    assert core.mutation_step_count is None
    assert core.mutation_plan_hash is None


def test_push_request_rejects_oversized_envelope_batches():
    envelopes = [
        _m1_envelope_payload(client_envelope_id=f"env-{index}")
        for index in range(SYNC_V2_MAX_PUSH_ENVELOPES + 1)
    ]

    with pytest.raises(ValidationError):
        SyncPushRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "envelopes": envelopes,
            }
        )


def test_push_request_requires_top_level_device_id():
    with pytest.raises(ValidationError):
        SyncPushRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "envelopes": [_m1_envelope_payload()],
            }
        )


def _ongoing_exchange_proof() -> dict[str, object]:
    return {
        "ongoing_sync_version": 1,
        "activation_epoch": "epoch_0123456789abcdef",
        "continuity_token": "continuity_0123456789abcdef",
    }


def test_ongoing_sync_models_preserve_version_zero_until_readiness() -> None:
    capabilities = api_sync_models.PersonalContextSyncCapabilitiesResponse()

    assert capabilities.ongoing_sync_version == 0
    assert capabilities.ongoing_sync_blockers == []
    assert capabilities.activation_epoch is None
    assert capabilities.continuity_token is None

    with pytest.raises(ValidationError):
        api_sync_models.PersonalContextSyncCapabilitiesResponse.model_validate(
            {
                "ongoing_sync_version": 1,
                "activation_epoch": "epoch_0123456789abcdef",
                "continuity_token": "continuity_0123456789abcdef",
                "ongoing_sync_blockers": ["personal_context_transport_unavailable"],
            }
        )


def test_ongoing_exchange_shapes_are_available_on_sync_boundaries() -> None:
    proof = _ongoing_exchange_proof()
    push = SyncPushRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "personal_context_exchange": proof,
        }
    )
    pull = api_sync_models.SyncPullResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "personal_context_relay": {
                "state": "personal_context_relay_pending",
                "scan_watermark": "cursor_0123456789abcdef",
            },
            "personal_context_exchange": proof,
        }
    )
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "personal_context_exchange": proof,
        }
    )

    assert push.personal_context_exchange is not None
    assert pull.personal_context_relay is not None
    assert response.personal_context_exchange == push.personal_context_exchange


def test_personal_context_conflict_list_response_requires_a_proof() -> None:
    proof = _ongoing_exchange_proof()
    response = SyncConflictListResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "conflicts": [],
            "personal_context_exchange": proof,
        }
    )

    assert response.personal_context_exchange.ongoing_sync_version == 1
    with pytest.raises(ValidationError):
        SyncConflictListResponse.model_validate(
            {"dataset_id": "dataset-1", "conflicts": []}
        )


def test_conflict_request_defers_personal_context_shape_to_loaded_conflict() -> None:
    proof = _ongoing_exchange_proof()
    request = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "personal_context_exchange": proof,
            "resolutions": [
                {
                    "conflict_id": "conflict_0123456789abcdef",
                    "action": "skip",
                }
            ],
        }
    )
    unproven = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "resolutions": [
                {
                    "conflict_id": "conflict_0123456789abcdef",
                    "action": "skip",
                    "expected_local_envelope_id": "local_0123456789abcdef",
                    "expected_remote_envelope_id": "remote_0123456789abcdef",
                    "idempotency_key": "resolve_0123456789abcdef",
                }
            ],
        }
    )

    assert request.resolutions[0].idempotency_key is None
    assert unproven.resolutions[0].idempotency_key == "resolve_0123456789abcdef"


def test_ongoing_activation_and_purge_requests_are_strict() -> None:
    proof = _ongoing_exchange_proof()
    acknowledgment = SyncPersonalContextActivationAcknowledgeRequest.model_validate(
        {
            "dataset_id": "dataset_0123456789abcdef",
            "device_id": "device_0123456789abcdef",
            "activation_id": "activation_0123456789abcdef",
            "baseline_digest": "a" * 64,
            "local_receipt_id": "receipt_0123456789abcdef",
            "personal_context_exchange": proof,
        }
    )
    purge = SyncPersonalContextPurgeRequest.model_validate(
        {
            "dataset_id": "dataset_0123456789abcdef",
            "device_id": "device_0123456789abcdef",
            "request_id": "request_0123456789abcdef",
            "expected_purge_generation": 0,
            "idempotency_key": "purge_0123456789abcdef",
            "signature": "s" * 32,
        }
    )

    assert acknowledgment.personal_context_exchange.ongoing_sync_version == 1
    assert purge.expected_purge_generation == 0
    with pytest.raises(ValidationError):
        SyncPersonalContextActivationAcknowledgeRequest.model_validate(
            {
                **acknowledgment.model_dump(),
                "baseline_digest": "!" + "a" * 64,
            }
        )
    with pytest.raises(ValidationError):
        SyncPersonalContextPurgeRequest.model_validate(
            {
                **purge.model_dump(),
                "unexpected": "field",
            }
        )


def test_attachment_upload_request_response_models():
    request = SyncAttachmentUploadRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "domain": "attachment.ref",
            "object_id": "attachment-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 128,
            "payload_ciphertext": "encrypted-bytes",
            "payload_hash": "sha256:attachment",
            "metadata": {
                "parent_domain": "notes.note",
                "parent_object_id": "note-1",
                "availability": "metadata_only",
            },
        }
    )
    response = SyncAttachmentUploadResponse.model_validate(
        {
            "attachment_id": "attachment-1",
            "dataset_id": "dataset-1",
            "stored": True,
            "size_bytes": 128,
            "payload_hash": "sha256:attachment",
            "download_url": "/api/v1/sync/attachments/attachment-1",
        }
    )

    assert request.domain == "attachment.ref"
    assert request.object_id == "attachment-1"
    assert request.size_bytes == response.size_bytes
    assert response.stored is True


def test_m2_blob_protocol_models_validate_session_manifest_and_restore_completeness():
    payload_hash = "sha256:" + "a" * 64
    chunk_hash = "sha256:" + "b" * 64

    create_request = SyncBlobUploadCreateRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 4096,
            "payload_hash": payload_hash,
            "chunk_size": 1024,
            "chunk_count": 4,
            "idempotency_key": "upload-once",
        }
    )
    session_response = SyncBlobUploadSessionResponse.model_validate(
        {
            "upload_id": "upload-1",
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "status": "uploading",
            "chunk_size": 1024,
            "chunk_count": 4,
            "uploaded_chunks": [0, 1],
            "missing_chunks": [2, 3],
            "quota": {"reserved_blob_bytes": 4096},
        }
    )
    chunk_response = SyncBlobChunkUploadResponse.model_validate(
        {
            "upload_id": "upload-1",
            "chunk_index": 2,
            "accepted": True,
            "size_bytes": 1024,
            "chunk_hash": chunk_hash,
            "missing_chunks": [3],
        }
    )
    complete_response = SyncBlobUploadCompleteResponse.model_validate(
        {
            "upload_id": "upload-1",
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "blob_id": "blob-1",
            "status": "available",
            "stored": True,
            "deduplicated": False,
            "size_bytes": 4096,
            "payload_hash": payload_hash,
        }
    )
    manifest_response = SyncBlobDownloadManifestResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "blob_id": "blob-1",
            "availability": "available",
            "content_type": "application/octet-stream",
            "size_bytes": 4096,
            "payload_hash": payload_hash,
            "chunks": [
                {
                    "chunk_index": 0,
                    "offset_bytes": 0,
                    "size_bytes": 1024,
                    "chunk_hash": chunk_hash,
                    "download_url": "/api/v1/sync/attachments/attachment-1/chunks/0",
                }
            ],
        }
    )
    completeness = SyncRestoreCompletenessResponse.model_validate(
        {
            "restore_status": "content_complete",
            "domain_details": [
                {
                    "domain": "attachment.ref",
                    "status": "content_complete",
                    "selected_count": 1,
                    "required_blob_count": 1,
                    "available_blob_count": 1,
                    "missing_blob_count": 0,
                    "verified_blob_count": 0,
                }
            ],
            "blob_details": [
                {
                    "attachment_id": "attachment-1",
                    "payload_hash": payload_hash,
                    "size_bytes": 4096,
                    "content_type": "application/octet-stream",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "server_availability": "available",
                    "required_for_restore": True,
                }
            ],
        }
    )

    assert create_request.encryption_policy == "server_trusted_v1"
    assert create_request.entity_id == "note-1"
    assert session_response.missing_chunks == [2, 3]
    assert chunk_response.chunk_index == 2
    assert complete_response.status == "available"
    assert manifest_response.chunks[0].chunk_hash == chunk_hash
    assert completeness.restore_status == "content_complete"


def test_m2_blob_upload_rejects_non_sha256_hashes():
    with pytest.raises(ValidationError):
        SyncBlobUploadCreateRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "domain": "attachment.ref",
                "object_id": "attachment-1",
                "attachment_id": "attachment-1",
                "content_type": "application/octet-stream",
                "size_bytes": 4096,
                "payload_hash": "md5:not-accepted",
                "chunk_size": 1024,
                "chunk_count": 4,
            }
        )


def test_attachment_intent_schema_accepts_strict_create_and_replace_models():
    from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
        SyncNotesAttachmentCreateIntent,
        SyncNotesAttachmentReplaceIntent,
    )

    create = SyncNotesAttachmentCreateIntent.model_validate(
        {
            "intent": "create",
            "note_id": ATTACHMENT_INTENT_NOTE_ID,
            "attachment_id": ATTACHMENT_INTENT_ID,
            "file_name": " Report.PDF ",
        }
    )
    replace = SyncNotesAttachmentReplaceIntent.model_validate(
        {
            "intent": "replace",
            "note_id": ATTACHMENT_INTENT_NOTE_ID,
            "attachment_id": ATTACHMENT_INTENT_ID,
            "base_server_cursor": 12,
            "base_object_revision": 3,
            "base_object_hash": "sha256:" + "a" * 64,
        }
    )

    assert create.file_name == "Report.pdf"
    assert replace.base_object_hash == "sha256:" + "a" * 64


@pytest.mark.parametrize(
    "payload",
    [
        {
            "intent": "create",
            "note_id": ATTACHMENT_INTENT_NOTE_ID,
            "attachment_id": ATTACHMENT_INTENT_ID,
            "file_name": "report.pdf",
            "base_server_cursor": 1,
        },
        {
            "intent": "replace",
            "note_id": ATTACHMENT_INTENT_NOTE_ID,
            "attachment_id": ATTACHMENT_INTENT_ID,
            "base_server_cursor": 1,
            "base_object_revision": 1,
            "base_object_hash": "sha256:" + "a" * 64,
            "file_name": "report.pdf",
        },
        {
            "intent": "replace",
            "note_id": ATTACHMENT_INTENT_NOTE_ID,
            "attachment_id": ATTACHMENT_INTENT_ID,
            "base_server_cursor": 0,
            "base_object_revision": 1,
            "base_object_hash": "sha256:" + "a" * 64,
        },
    ],
)
def test_attachment_intent_schema_rejects_unknown_fields_and_invalid_base(payload):
    from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
        SyncNotesAttachmentCreateIntent,
        SyncNotesAttachmentReplaceIntent,
    )

    model = (
        SyncNotesAttachmentCreateIntent
        if payload["intent"] == "create"
        else SyncNotesAttachmentReplaceIntent
    )
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_capabilities_accept_m2_blob_transfer_and_quota_details():
    capabilities = SyncCapabilitiesResponse.model_validate(
        {
            "protocol_version": "sync-v2-m2",
            "min_supported_protocol_version": "sync-v2-m1",
            "supports_attachments": True,
            "blob_transfer": {
                "supported": True,
                "resumable_upload": True,
                "resumable_download": True,
                "chunk_checksums": True,
                "full_checksum": "sha256",
                "storage_backend": "local_fs",
            },
            "quota": {
                "max_blob_bytes": 104857600,
                "max_chunk_bytes": 4194304,
                "max_active_uploads": 8,
                "user_blob_quota_bytes": 10737418240,
            },
        }
    )

    assert capabilities.protocol_version == "sync-v2-m2"
    assert capabilities.blob_transfer["supported"] is True
    assert capabilities.blob_transfer["full_checksum"] == "sha256"
    assert capabilities.quota["max_chunk_bytes"] == 4194304


def test_capabilities_advertise_supported_and_writable_adapter_versions_separately() -> None:
    capabilities = SyncCapabilitiesResponse()

    assert capabilities.supported_adapter_versions["attachment.ref"] == [1, 2]
    assert capabilities.writable_adapter_versions["attachment.ref"] == []
    assert capabilities.supported_adapter_versions["notes.note"] == [1]
    assert capabilities.writable_adapter_versions["notes.note"] == []


def test_device_adapter_version_omission_means_version_one() -> None:
    request = api_sync_models.SyncDeviceRegisterRequest.model_validate(
        {
            "display_name": "Legacy device",
            "supported_domains": ["notes.note", "attachment.ref"],
        }
    )

    assert request.supported_adapter_versions == {
        "notes.note": [1],
        "attachment.ref": [1],
    }
    assert request.capabilities["supported_adapter_versions"] == {
        "notes.note": [1],
        "attachment.ref": [1],
    }
    assert request.capabilities["requested_domains"] == [
        "notes.note",
        "attachment.ref",
    ]


@pytest.mark.parametrize(
    ("requested_domains", "message"),
    [
        (["unknown.domain"], "unknown Sync domain"),
        (["notes.note"] * 101, "at most 100 domains"),
    ],
)
def test_adapter_version_omission_still_validates_requested_domains(
    requested_domains: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        core_sync_models.normalize_supported_adapter_versions(
            None,
            requested_domains=requested_domains,
        )


def test_profile_bootstrap_adapter_version_omission_means_version_one() -> None:
    request = api_sync_models.SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "requested_domains": ["notes.note", "attachment.ref"],
        }
    )

    assert request.supported_adapter_versions == {
        "notes.note": [1],
        "attachment.ref": [1],
    }
    assert request.client_instance["supported_adapter_versions"] == {
        "notes.note": [1],
        "attachment.ref": [1],
    }


@pytest.mark.parametrize(
    ("request_model", "version_container"),
    [
        (api_sync_models.SyncDeviceRegisterRequest, "capabilities"),
        (api_sync_models.SyncProfileBootstrapRequest, "client_instance"),
    ],
)
def test_partial_adapter_version_map_defaults_omitted_requested_domains_to_v1(
    request_model: type[BaseModel],
    version_container: str,
) -> None:
    payload: dict[str, object] = {
        "supported_adapter_versions": {"attachment.ref": [2]},
    }
    if request_model is api_sync_models.SyncDeviceRegisterRequest:
        payload.update(
            display_name="Versioned device",
            supported_domains=["notes.note", "attachment.ref"],
        )
    else:
        payload.update(
            mode="offline_sync",
            requested_domains=["notes.note", "attachment.ref"],
        )

    request = request_model.model_validate(payload)

    expected = {"notes.note": [1], "attachment.ref": [2]}
    assert request.supported_adapter_versions == expected
    assert getattr(request, version_container)["supported_adapter_versions"] == expected


@pytest.mark.parametrize(
    ("version_map", "message"),
    [
        (
            {f"unknown-{index}": [1] for index in range(101)},
            "at most 100 domains",
        ),
        ({"attachment.ref": list(range(1, 10))}, "at most 8 versions"),
        ({"attachment.ref": [1, 1]}, "duplicate adapter versions"),
        ({"attachment.ref": []}, "non-empty"),
        ({"attachment.ref": [0]}, "positive integers"),
        ({"attachment.ref": [-1]}, "positive integers"),
        ({"attachment.ref": [True]}, "valid integer"),
        ({"attachment.ref": ["2"]}, "valid integer"),
        ({"unknown.domain": [1]}, "unknown Sync domain"),
    ],
)
def test_device_adapter_version_map_is_bounded_and_strict(
    version_map: dict[str, list[object]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        api_sync_models.SyncDeviceRegisterRequest.model_validate(
            {
                "display_name": "Versioned device",
                "supported_domains": ["attachment.ref"],
                "supported_adapter_versions": version_map,
            }
        )


def test_device_adapter_version_map_accepts_eight_sorted_versions() -> None:
    request = api_sync_models.SyncDeviceRegisterRequest.model_validate(
        {
            "display_name": "Versioned device",
            "supported_domains": ["attachment.ref"],
            "supported_adapter_versions": {
                "attachment.ref": [8, 3, 1, 2, 4, 5, 6, 7],
            },
        }
    )

    assert request.supported_adapter_versions == {
        "attachment.ref": [1, 2, 3, 4, 5, 6, 7, 8]
    }


def test_attachment_ref_v2_api_envelope_carries_adapter_version() -> None:
    from tldw_Server_API.app.core.Sync.v2 import attachment_refs_v2

    payload = {
        "attachment_id": "a1111111-1111-4111-8111-111111111111",
        "parent_domain": "notes.note",
        "parent_object_id": "b2222222-2222-4222-8222-222222222222",
        "file_name": "diagram.png",
        "original_file_name": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 512,
        "blob_hash": "sha256:" + "a" * 64,
        "created_at": "2026-08-11T20:30:00+00:00",
        "last_modified": "2026-08-11T20:30:00+00:00",
        "created_by": "device-1",
    }
    request = SyncV2Envelope.model_validate(
        {
            "dataset_id": "dataset-1",
            "client_envelope_id": "attachment-v2-create",
            "device_id": "device-1",
            "domain": "attachment.ref",
            "operation": "upsert",
            "object_id": payload["attachment_id"],
            "schema_version": 2,
            "adapter_version": 2,
            "object_revision": 1,
            "payload": payload,
            "payload_hash": attachment_refs_v2.attachment_ref_v2_object_hash(
                "upsert",
                payload,
                object_revision=1,
            ),
            "created_at_client": "2026-08-11T20:30:00Z",
        }
    )

    assert request.adapter_version == 2
    assert _core_envelope_from_api(request).adapter_version == 2


def test_attachment_ref_v2_capability_schema_advertises_strict_tombstones() -> None:
    from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
        AttachmentRefV2Payload,
        AttachmentRefV2TombstonePayload,
    )

    schema = SyncCapabilitiesResponse().domain_schemas["attachment.ref"]

    assert schema["tombstone"]["required"] == [
        *schema["upsert"]["required"],
        "deleted_at",
    ]
    assert schema["tombstone"]["properties"]["reason"] == {
        "type": ["string", "null"],
        "max_length": 256,
    }
    assert schema["tombstone"]["additional_properties"] is False
    for operation, model in (
        ("upsert", AttachmentRefV2Payload),
        ("tombstone", AttachmentRefV2TombstonePayload),
    ):
        generated = model.model_json_schema()
        advertised = schema[operation]
        assert advertised["required"] == generated["required"]
        assert advertised["additional_properties"] is generated["additionalProperties"]
        for field_name, field_schema in generated["properties"].items():
            if "minLength" in field_schema:
                assert advertised["properties"][field_name]["min_length"] == field_schema[
                    "minLength"
                ]


def test_version_ack_api_defaults_adapter_version_to_one() -> None:
    request = api_sync_models.SyncDeviceDomainAckRequest.model_validate(
        {
            "domain": "notes.note",
            "through_server_sequence": 7,
            "applied_at": "2026-08-11T20:30:00Z",
        }
    )
    response = api_sync_models.SyncDeviceDomainAckResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain": "notes.note",
            "through_server_sequence": 7,
            "applied_at": "2026-08-11T20:30:00Z",
            "updated_at": "2026-08-11T20:30:00Z",
        }
    )

    assert request.adapter_version == 1
    assert response.adapter_version == 1


def test_blob_id_ack_api_is_separate_from_legacy_attachment_ack() -> None:
    digest = "sha256:" + "a" * 64
    v2 = api_sync_models.SyncDeviceBlobIdAckRequest.model_validate(
        {
            "blob_id": "blob-immutable-1",
            "payload_hash": digest,
            "verified_at": "2026-08-11T20:30:00Z",
        }
    )

    with pytest.raises(ValidationError):
        api_sync_models.SyncDeviceBlobAckRequest.model_validate(
            {
                "blob_id": "blob-immutable-1",
                "payload_hash": digest,
                "verified_at": "2026-08-11T20:30:00Z",
            }
        )

    assert v2.blob_id == "blob-immutable-1"


def test_blob_id_ack_batch_is_bounded_and_omitted_by_default() -> None:
    digest = "sha256:" + "a" * 64
    base = {
        "dataset_id": "dataset-1",
        "device_id": "device-1",
    }

    omitted = api_sync_models.SyncDeviceAcknowledgmentsRequest.model_validate(base)
    assert omitted.blob_id_acks == []

    with pytest.raises(ValidationError):
        api_sync_models.SyncDeviceAcknowledgmentsRequest.model_validate(
            {
                **base,
                "blob_id_acks": [
                    {
                        "blob_id": f"blob-{index}",
                        "payload_hash": digest,
                        "verified_at": "2026-08-11T20:30:00Z",
                    }
                    for index in range(801)
                ],
            }
        )


def test_blob_id_ack_endpoint_keeps_legacy_and_v2_inputs_distinct() -> None:
    from pathlib import Path

    endpoint_source = Path(
        "tldw_Server_API/app/api/v1/endpoints/sync.py"
    ).read_text(encoding="utf-8")
    acknowledgment_block = endpoint_source.split(
        "def acknowledge_sync_v2_device_state", 1
    )[1].split("@router.", 1)[0]

    assert "SyncDeviceBlobAckCreate" in acknowledgment_block
    assert "attachment_id=ack.attachment_id" in acknowledgment_block
    assert "SyncDeviceBlobIdAckCreate" in acknowledgment_block
    assert "blob_id=ack.blob_id" in acknowledgment_block
    assert "blob_id=ack.attachment_id" not in acknowledgment_block
