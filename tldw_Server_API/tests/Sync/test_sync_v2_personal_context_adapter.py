from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, replace

import pytest
from tldw_profile_core import ProposalState
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterContext,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDataset,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
    global_scope,
    manifest,
    preference_record,
    proposal,
)

pytestmark = pytest.mark.unit

INTEGRITY_KEY = b"i" * 32
ENCRYPTION_KEY = b"e" * 32
INTEGRITY_KEY_ID = "pc-integrity-key-1"


def _dataset() -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-a",
        owner_user_id="user-a",
        scope_type="personal",
        encryption_policy="server_trusted_v1",
        domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
        workspace_id=None,
        metadata={
            "personal_context": {
                "profile_id": "profile-a",
                "integrity_key_id": INTEGRITY_KEY_ID,
                "purge_generation": 0,
            }
        },
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:00:00Z",
    )


def _payloads() -> dict[str, tuple[dict[str, object], str, str | None]]:
    values = {
        "personal_context.manifest": manifest(),
        "personal_context.scope": global_scope(),
        "personal_context.record": preference_record(),
        "personal_context.proposal": proposal(),
    }
    payloads = {
        domain: (
            value.model_dump(mode="json"),
            (
                value.profile_id
                if domain == "personal_context.manifest"
                else value.scope_id
                if domain == "personal_context.scope"
                else value.record_id
                if domain == "personal_context.record"
                else value.proposal_id
            ),
            (
                None
                if domain == "personal_context.manifest"
                else value.profile_id
                if domain == "personal_context.scope"
                else value.scope_id
            ),
        )
        for domain, value in values.items()
    }
    payloads["personal_context.purge"] = (
        {"schema_version": 1, "profile_id": "profile-a", "purge_generation": 1},
        "profile-a",
        None,
    )
    return payloads


def _tag(payload: dict[str, object]) -> str:
    digest = hmac.new(INTEGRITY_KEY, canonical_json_bytes(payload), hashlib.sha256)
    return f"hmac-sha256-v1:{digest.hexdigest()}"


def _envelope(
    domain: str,
    *,
    payload: dict[str, object] | None = None,
    object_id: str | None = None,
    parent_id: str | None = None,
    payload_hash: str | None = None,
    base_object_hash: str | None = None,
    object_revision: int = 1,
) -> SyncEnvelopeCreate:
    default_payload, default_object_id, default_parent_id = _payloads()[domain]
    value = default_payload if payload is None else payload
    return SyncEnvelopeCreate(
        dataset_id="dataset-a",
        client_envelope_id=f"device-a:{domain}:{object_revision}",
        device_id="device-a",
        domain=domain,
        operation="tombstone" if domain == "personal_context.purge" else "upsert",
        object_id=default_object_id if object_id is None else object_id,
        parent_id=default_parent_id if parent_id is None else parent_id,
        adapter_version=1,
        schema_version=1,
        payload=value,
        payload_hash=_tag(value) if payload_hash is None else payload_hash,
        payload_size_bytes=len(canonical_json_bytes(value)),
        base_object_hash=base_object_hash,
        object_revision=object_revision,
        encryption_metadata={"policy": "server_trusted_v1"},
        routing_metadata={"integrity_key_id": INTEGRITY_KEY_ID},
    )


def _adapter(domain: str) -> PersonalContextDomainAdapter:
    return PersonalContextDomainAdapter(
        domain=domain,
        integrity_key_resolver=lambda _dataset, _key_id: INTEGRITY_KEY,
    )


@pytest.mark.parametrize("domain", PERSONAL_CONTEXT_SYNC_DOMAINS)
def test_adapter_accepts_exact_canonical_whole_objects(domain: str) -> None:
    result = _adapter(domain).evaluate_envelope(_envelope(domain), dataset=_dataset())

    assert isinstance(result, AdapterAccepted)


def test_adapter_rejects_invalid_integrity_without_echoing_body() -> None:
    result = _adapter("personal_context.record").evaluate_envelope(
        _envelope("personal_context.record", payload_hash="hmac-sha256-v1:" + "0" * 64),
        dataset=_dataset(),
    )

    assert isinstance(result, AdapterRejected)
    assert result.error_code == "personal_context_integrity_invalid"
    assert "concise" not in result.message


def test_adapter_rejects_malformed_dataset_state_without_raising() -> None:
    dataset = replace(_dataset(), metadata={})

    result = _adapter("personal_context.record").evaluate_envelope(
        _envelope("personal_context.record"), dataset=dataset
    )

    assert isinstance(result, AdapterRejected)
    assert result.error_code == "personal_context_dataset_invalid"


def test_adapter_rejects_missing_canonical_profile_key_without_raising() -> None:
    def missing_profile(_dataset: SyncDataset, _key_id: str) -> bytes:
        raise KeyError("profile")

    adapter = PersonalContextDomainAdapter(
        domain="personal_context.record",
        integrity_key_resolver=missing_profile,
    )

    result = adapter.evaluate_envelope(
        _envelope("personal_context.record"),
        dataset=_dataset(),
    )

    assert isinstance(result, AdapterRejected)
    assert result.error_code == "personal_context_integrity_unavailable"


def test_adapter_rejects_device_only_record() -> None:
    value = preference_record().model_dump(mode="json")
    value["controls"]["sync_mode"] = "device_only"

    result = _adapter("personal_context.record").evaluate_envelope(
        _envelope("personal_context.record", payload=value),
        dataset=_dataset(),
    )

    assert isinstance(result, AdapterRejected)
    assert result.error_code == "personal_context_device_only_forbidden"


def test_adapter_rejects_pending_proposal_with_device_only_record() -> None:
    value = proposal().model_dump(mode="json")
    value["proposed_record"]["controls"]["sync_mode"] = "device_only"

    result = _adapter("personal_context.proposal").evaluate_envelope(
        _envelope("personal_context.proposal", payload=value),
        dataset=_dataset(),
    )

    assert isinstance(result, AdapterRejected)
    assert result.error_code == "personal_context_device_only_forbidden"


def test_adapter_returns_conflict_for_wrong_object_identity() -> None:
    result = _adapter("personal_context.record").evaluate_envelope(
        _envelope("personal_context.record", object_id="other-record"),
        dataset=_dataset(),
    )

    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "personal_context_identity_conflict"


def test_adapter_requires_exact_base_lineage_for_updates() -> None:
    first = _envelope("personal_context.record")
    head = SyncEnvelope(
        **{
            **asdict(first),
            "server_cursor": 1,
            "envelope_id": "envelope-1",
        }
    )
    updated = preference_record(
        version_id="record-v2",
        parent_version_id="record-v1",
        value="structured",
    ).model_dump(mode="json")
    incoming = _envelope(
        "personal_context.record",
        payload=updated,
        base_object_hash="hmac-sha256-v1:" + "f" * 64,
        object_revision=2,
    )

    result = _adapter("personal_context.record").evaluate_envelope(
        incoming,
        dataset=_dataset(),
        context=SyncAdapterContext(prior_envelopes=(head,)),
    )

    assert isinstance(result, AdapterConflict)
    assert result.conflict_type == "personal_context_base_conflict"


def test_adapter_accepts_record_parent_lineage_without_prior_server_hash() -> None:
    first = _envelope("personal_context.record")
    head = SyncEnvelope(
        **{
            **asdict(first),
            "server_cursor": 1,
            "envelope_id": "envelope-1",
        }
    )
    updated = preference_record(
        version_id="record-v2",
        parent_version_id="record-v1",
        value="structured",
    ).model_dump(mode="json")

    result = _adapter("personal_context.record").evaluate_envelope(
        _envelope(
            "personal_context.record",
            payload=updated,
            object_revision=2,
        ),
        dataset=_dataset(),
        context=SyncAdapterContext(prior_envelopes=(head,)),
    )

    assert isinstance(result, AdapterAccepted)


def test_adapter_accepts_exact_pending_to_terminal_proposal_lineage() -> None:
    first = _envelope("personal_context.proposal")
    head = SyncEnvelope(
        **{
            **asdict(first),
            "server_cursor": 1,
            "envelope_id": "envelope-1",
        }
    )
    pending = proposal()
    terminal = pending.model_copy(
        update={
            "state": ProposalState.REJECTED,
            "proposed_record": None,
            "confidence": None,
        }
    ).model_dump(mode="json")

    result = _adapter("personal_context.proposal").evaluate_envelope(
        _envelope(
            "personal_context.proposal",
            payload=terminal,
            object_revision=2,
        ),
        dataset=_dataset(),
        context=SyncAdapterContext(prior_envelopes=(head,)),
    )

    assert isinstance(result, AdapterAccepted)


def test_storage_round_trip_preserves_payload_clear_only_envelope() -> None:
    payload = preference_record().model_dump(mode="json")
    envelope = replace(
        _envelope("personal_context.record", payload=payload),
        payload={},
        payload_clear=payload,
    )
    adapter = PersonalContextDomainAdapter(
        domain="personal_context.record",
        integrity_key_resolver=lambda _dataset, _key_id: INTEGRITY_KEY,
        encryption_key_resolver=lambda _dataset: (ENCRYPTION_KEY, 1),
    )

    protected = adapter.protect_for_storage(envelope, dataset=_dataset())
    restored = adapter.restore_from_storage(protected, dataset=_dataset())

    assert protected.payload == {}
    assert protected.payload_clear == {}
    assert restored.payload == payload
    assert restored.payload_clear == payload


def test_storage_round_trip_survives_profile_encryption_key_rotation(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key(b"a"))
    repository = PersonalContextRepository(
        PersonalizationDB.for_path(tmp_path / "Personalization.db")
    )
    repository.create_profile(manifest(), global_scope())
    adapter = PersonalContextDomainAdapter(
        domain="personal_context.record",
        integrity_key_resolver=lambda _dataset, _key_id: INTEGRITY_KEY,
        encryption_key_resolver=lambda _dataset: repository.sync_encryption_key(
            "profile-a"
        ),
    )
    envelope = _envelope("personal_context.record")

    protected = adapter.protect_for_storage(envelope, dataset=_dataset())
    repository.rotate_encryption_key("profile-a")
    restored = adapter.restore_from_storage(protected, dataset=_dataset())

    assert restored.payload == envelope.payload
