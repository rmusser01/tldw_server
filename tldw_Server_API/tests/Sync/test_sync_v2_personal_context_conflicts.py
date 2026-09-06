"""Real canonical/Sync conflict choices, immutable candidates and crash recovery."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, replace
from datetime import datetime, tzinfo
from typing import Any

import pytest
from tldw_profile_core import ProfileRecord, canonical_bytes

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
    ProfileConflictError,
    RecordMutation,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import PersonalContextExchangeProof
from tldw_Server_API.app.core.Sync.v2.service import SyncPushConflict, SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_activation import activation_store as activation_store
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import (
    _DEVICE_ID,
    _USER_ID,
    _record_body,
    _register_device,
    _seed_exchange,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import (
    production_factories as production_factories,
)

pytestmark = pytest.mark.unit

LinkedRuntime = tuple[PersonalContextService, SyncV2Service, str, PersonalContextExchangeProof, ProfileRecord]


@pytest.fixture()
def linked(
    production_factories: tuple[PersonalContextService, SyncV2Service], monkeypatch: pytest.MonkeyPatch
) -> LinkedRuntime:
    """Exercise linked."""
    canonical, sync = production_factories
    relay_type = type(sync.personal_context_relay)
    original_relay = relay_type.relay_profile

    def deterministic_relay(relay: Any, **kwargs: Any) -> Any:
        """Exercise deterministic relay."""
        relay.clock_ns = lambda: 0
        return original_relay(relay, **kwargs)

    monkeypatch.setattr(relay_type, "relay_profile", deterministic_relay)
    return _link(canonical, sync)


def _link(canonical: PersonalContextService, sync: SyncV2Service) -> LinkedRuntime:
    """Exercise link."""
    canonical.create_profile()
    _register_device(sync)
    initial = sync.bootstrap_personal_context(user_id=_USER_ID, device_id=_DEVICE_ID)
    sync.complete_personal_context_link(
        user_id=_USER_ID,
        device_id=_DEVICE_ID,
        dataset_id=initial.dataset_id,
        bootstrap_cursor=initial.cursor,
    )
    exchange = PersonalContextExchangeProof.model_validate(_seed_exchange(sync, initial.dataset_id))
    record = canonical.create_manual_record(
        **_record_body(canonical.list_scopes()[0].scope_id, "TASK13163-SHARED-CANARY")
    )
    sync.pull(
        user_id=_USER_ID,
        dataset_id=initial.dataset_id,
        device_id=_DEVICE_ID,
        domains=["personal_context.record"],
        personal_context_exchange=exchange,
    )
    return canonical, sync, initial.dataset_id, exchange, record


def _reopen(linked: LinkedRuntime) -> LinkedRuntime:
    """Exercise reopen."""
    from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import personal_context_service_for_user
    from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends
    from tldw_Server_API.app.core.Sync.v2 import factory
    from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import _clear_factory_caches

    canonical, sync, dataset_id, exchange, record = linked
    backend = sync.store.db.backend
    reset_managed_sqlite_backends(backends=[backend])
    _clear_factory_caches()
    reopened = personal_context_service_for_user(_USER_ID)
    restarted = factory.sync_v2_service_for_user(_USER_ID)
    assert reopened._repository.database is not canonical._repository.database
    assert restarted.store.db.backend is not backend
    return reopened, restarted, dataset_id, exchange, record


@pytest.mark.parametrize("boundary", ["capture", "receipt"])
def test_reopened_owners_resume_exact_conflict_after_interruption(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """Verify reopened owners resume exact conflict after interruption."""
    from tldw_Server_API.app.core.Sync.v2.personal_context_conflicts import PersonalContextConflictService

    canonical, sync, dataset_id, _exchange, record = linked
    if boundary == "capture":
        incoming = _envelope(
            linked, record.model_copy(update={"version_id": "interrupted-local-version"}), "interrupted-local-envelope"
        )
        with monkeypatch.context() as fault:

            def fail_attachment(*args: Any, **kwargs: Any) -> None:
                """Exercise fail attachment."""
                raise RuntimeError("injected capture-before-attachment interruption")

            fault.setattr(PersonalContextConflictService, "_attach_candidate", fail_attachment)
            assert _push(linked, incoming).rejected[0].retryable
        conflict_id = PersonalContextConflictService.conflict_id(dataset_id, _DEVICE_ID, incoming.client_envelope_id)
        journal = canonical.get_sync_conflict(conflict_id)
        runtime = _reopen(linked)
        replay = _push(runtime, incoming)
        assert len(replay.conflicts) == 1
        assert replay.conflicts[0].expected_remote_envelope_id == journal["remote_envelope_id"]
        assert replay.conflicts[0].authority_candidate.payload == journal["candidate"]
        assert asdict(_push(runtime, incoming).conflicts[0]) == asdict(replay.conflicts[0])
    else:
        conflict, _source = _conflict(linked)
        replacement = record.model_copy(
            update={"version_id": "reopened-reviewed-version", "parent_version_id": record.version_id}
        )
        command = _envelope(linked, replacement, "reopened-reviewed-envelope", base=record)
        with monkeypatch.context() as fault:

            def fail_finalize(*args: Any, **kwargs: Any) -> None:
                """Exercise fail finalize."""
                raise RuntimeError("injected receipt-before-finalization interruption")

            fault.setattr(type(sync.store), "resolve_conflict", fail_finalize)
            assert _resolve(linked, conflict, "overwrite", command)[1] == [0]
        manifest = canonical.get_manifest()
        with canonical._repository.database.transaction() as connection:
            publications = connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
        runtime = _reopen(linked)
        assert _resolve(runtime, conflict, "overwrite", command)[1] == []
        assert runtime[0].get_manifest() == manifest
        assert runtime[0].get_record(record.record_id) == replacement
        with runtime[0]._repository.database.transaction() as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
                == publications
            )


def test_old_resolution_rejected_after_purge_and_recreation_attempt(linked: LinkedRuntime) -> None:
    """Verify old resolution rejected after purge and recreation attempt."""
    from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError

    conflict, _source = _conflict(linked)
    canonical = linked[0]
    canonical.purge_profile(mode="everywhere", confirmation="DELETE EVERYWHERE", expected_purge_generation=0)
    runtime = _reopen(linked)
    with pytest.raises(ProfileConflictError):
        runtime[0].create_profile()
    with pytest.raises(SyncStoreError):
        _resolve(runtime, conflict)
    assert not runtime[0].list_records()
    with runtime[0]._repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type IN ('sync_conflict', 'sync_conflict_receipt')"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("domain", ["personal_context.scope", "personal_context.proposal"])
def test_non_record_candidate_replay_and_explicit_choice(
    linked: LinkedRuntime, domain: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify non record candidate replay and explicit choice."""
    from datetime import UTC, datetime, timedelta

    from tldw_Server_API.app.core.DB_Management import Personal_Context_Repository as repository_module

    class FixedDateTime(datetime):
        """Hold expiry-sensitive canonical proposal checks at one instant."""

        @classmethod
        def now(cls: type[datetime], tz: tzinfo | None = None) -> datetime:
            """Exercise now."""
            return datetime(2090, 9, 5, tzinfo=UTC).astimezone(tz)

    from tldw_Server_API.tests.Personalization.test_personal_context_service import _pending_proposal_for_scope

    canonical, sync, dataset_id, exchange, record = linked
    monkeypatch.setattr(repository_module, "datetime", FixedDateTime)
    monkeypatch.setattr(canonical, "_clock", lambda: FixedDateTime.now(UTC))
    scope = canonical.list_scopes()[0]
    if domain == "personal_context.scope":
        original = scope
        local = original.model_copy(update={"version_id": "conflicted-scope-version"})
        replacement = original.model_copy(update={"version_id": "reviewed-scope-version"})
        object_id, parent = scope.scope_id, scope.profile_id
        version, replacement_version = local.version_id, replacement.version_id
    else:
        proposal = _pending_proposal_for_scope(
            canonical, profile_id=record.profile_id, scope_id=scope.scope_id, proposal_id="conflicted-proposal-13163"
        )
        now = FixedDateTime.now(UTC)
        original = canonical.create_proposal(
            type(proposal).model_validate(
                {**proposal.model_dump(mode="python"), "created_at": now, "expires_at": now + timedelta(days=90)}
            )
        )
        local = type(original).model_validate({**original.model_dump(mode="python"), "confidence": 0.9})
        replacement = type(original).model_validate(
            {**original.model_dump(mode="python"), "state": "rejected", "proposed_record": None, "confidence": None}
        )
        object_id, parent = original.proposal_id, original.scope_id
        version = "sync-proposal-sha256:" + hashlib.sha256(canonical_bytes(local)).hexdigest()
        replacement_version = "sync-proposal-sha256:" + hashlib.sha256(canonical_bytes(replacement)).hexdigest()
    sync.pull(
        user_id=_USER_ID,
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        domains=[domain],
        personal_context_exchange=exchange,
    )
    incoming = _whole_object_envelope(linked, local, domain, object_id, version, parent=parent)
    result = _push(linked, incoming)
    assert result.conflicts, (
        [(item.apply_status, item.apply_error_code) for item in result.accepted],
        [(item.error_code, item.retryable) for item in result.rejected],
    )
    conflict = result.conflicts[0]
    assert conflict.authority_candidate.payload == original.model_dump(mode="json")
    assert asdict(_push(linked, incoming).conflicts[0]) == asdict(conflict)
    command = _whole_object_envelope(
        linked, replacement, domain, object_id, replacement_version, parent=parent, base=original
    )
    assert _resolve(linked, conflict, "overwrite", command)[1] == []
    assert canonical.get_sync_conflict(conflict.conflict_id)["receipt"]["resulting_object_id"] == object_id


def test_invalid_purge_envelope_is_rejected_without_review(linked: LinkedRuntime) -> None:
    """Verify invalid purge envelope is rejected without review."""
    envelope = replace(
        _envelope(linked, linked[4], "invalid-purge-envelope"), domain="personal_context.purge", operation="tombstone"
    )
    result = _push(linked, envelope)
    assert (
        not result.conflicts
        and not result.accepted
        and result.rejected[0].error_code == "personal_context_payload_invalid"
    )


def test_signed_invalid_purge_generation_rejected_without_candidate_or_mutation(linked: LinkedRuntime) -> None:
    """Verify signed invalid purge generation rejected without candidate or mutation."""
    from tldw_profile_core.canonical import canonical_json_bytes

    canonical, _sync, dataset_id, _exchange, record = linked
    before = canonical.get_manifest()
    key_id, key = canonical.sync_integrity_key(record.profile_id)
    payload = {"schema_version": 1, "profile_id": record.profile_id, "purge_generation": 2}
    encoded = canonical_json_bytes(payload)
    envelope = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        client_envelope_id="invalid-generation-purge",
        domain="personal_context.purge",
        operation="tombstone",
        object_id=record.profile_id,
        parent_id=None,
        entity_version=2,
        payload=payload,
        payload_size_bytes=len(encoded),
        payload_hash="hmac-sha256-v1:" + hmac.new(key, encoded, hashlib.sha256).hexdigest(),
        routing_metadata={"integrity_key_id": key_id, "profile_id": record.profile_id, "purge_generation": 0},
    )
    with canonical._repository.database.transaction() as connection:
        publications = connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
    result = _push(linked, envelope)
    assert not result.accepted and not result.conflicts and len(result.rejected) == 1
    rejection = result.rejected[0]
    assert rejection.error_code == "personal_context_purge_generation_invalid"
    assert not rejection.retryable
    assert record.profile_id not in rejection.message
    assert canonical.get_manifest() == before
    with canonical._repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type = 'sync_conflict'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
            == publications
        )


def test_conflict_canaries_absent_from_sync_storage_and_diagnostics(linked: LinkedRuntime) -> None:
    """Verify conflict canaries absent from sync storage and diagnostics."""
    from pathlib import Path

    from loguru import logger

    diagnostics = []
    sink = logger.add(lambda message: diagnostics.append(str(message)))
    try:
        conflict, source = _conflict(linked, collision=True)
        assert _push(linked, source).conflicts
        assert _resolve(linked, conflict, expected_remote="stale-remote-canary")[1] == [0]
    finally:
        logger.remove(sink)
    for canary in (b"TASK13163-SHARED-CANARY", b"TASK13163-LOCAL-CANARY"):
        assert canary.decode() not in "".join(diagnostics)
        for path in (Path(linked[0]._repository.database.db_path), Path(linked[1].store.db.backend.config.sqlite_path)):
            for artifact in (path, Path(str(path) + "-wal")):
                if artifact.exists():
                    assert canary not in artifact.read_bytes()


def _envelope(
    linked: LinkedRuntime, record: ProfileRecord, envelope_id: str, *, base: Any = None
) -> SyncEnvelopeCreate:
    """Exercise envelope."""
    canonical, sync, dataset_id, _exchange, _record = linked
    key_id, key = canonical.sync_integrity_key(record.profile_id)
    payload = canonical_bytes(record)
    return SyncEnvelopeCreate(
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        client_envelope_id=envelope_id,
        domain="personal_context.record",
        operation="upsert",
        object_id=record.record_id,
        parent_id=record.scope_id,
        payload=record.model_dump(mode="json"),
        payload_hash="hmac-sha256-v1:" + hmac.new(key, payload, hashlib.sha256).hexdigest(),
        payload_size_bytes=len(payload),
        entity_version=record.version_id,
        base_object_hash=(None if base is None else "sha256:" + hashlib.sha256(canonical_bytes(base)).hexdigest()),
        routing_metadata={"integrity_key_id": key_id, "profile_id": record.profile_id, "purge_generation": 0},
    )


def _push(linked: LinkedRuntime, envelope: SyncEnvelopeCreate | None) -> Any:
    """Exercise push."""
    _canonical, sync, dataset_id, exchange, _record = linked
    return sync.push(
        user_id=_USER_ID,
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        envelopes=[envelope],
        personal_context_exchange=exchange,
    )


def _whole_object_envelope(
    linked: LinkedRuntime,
    value: Any,
    domain: str,
    object_id: str,
    version: str,
    *,
    parent: str | None = None,
    base: Any = None,
) -> SyncEnvelopeCreate:
    """Exercise whole object envelope."""
    canonical, _sync, dataset_id, _exchange, _record = linked
    key_id, key = canonical.sync_integrity_key(value.profile_id)
    payload = canonical_bytes(value)
    return SyncEnvelopeCreate(
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        client_envelope_id=f"review-{domain}-{version}",
        domain=domain,
        operation="upsert",
        object_id=object_id,
        parent_id=parent,
        payload=value.model_dump(mode="json"),
        entity_version=version,
        payload_hash="hmac-sha256-v1:" + hmac.new(key, payload, hashlib.sha256).hexdigest(),
        payload_size_bytes=len(payload),
        base_object_hash=None if base is None else "sha256:" + hashlib.sha256(canonical_bytes(base)).hexdigest(),
        routing_metadata={"integrity_key_id": key_id, "profile_id": value.profile_id, "purge_generation": 0},
    )


@pytest.mark.parametrize("stale", [False, True])
def test_linked_client_manifest_rejected_without_conflict_or_publication(linked: LinkedRuntime, stale: bool) -> None:
    """Verify linked client manifest rejected without conflict or publication."""
    canonical, _sync, _dataset, _exchange, _record = linked
    before = canonical.get_manifest()
    envelope = _whole_object_envelope(
        linked,
        before,
        "personal_context.manifest",
        before.profile_id,
        before.current_version_id,
        base=None if stale else before,
    )
    with canonical._repository.database.transaction() as connection:
        publications = connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
    result = _push(linked, envelope)
    assert not result.accepted and not result.conflicts and len(result.rejected) == 1
    assert result.rejected[0].error_code == "personal_context_manifest_client_forbidden"
    assert canonical.get_manifest() == before
    with canonical._repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type = 'sync_conflict'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM personal_context_publication_batches").fetchone()[0]
            == publications
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity_version", json.dumps("tampered-version-13163")),
        ("operation", "tombstone"),
        ("parent_id", "tampered-parent-13163"),
        ("created_at_client", "2026-01-01T00:00:00+00:00"),
        ("received_at_server", "2026-01-01T00:00:00+00:00"),
        ("payload_hash", "hmac-sha256-v1:" + "0" * 64),
        ("payload_size_bytes", 1),
        ("routing_metadata_json", None),
    ],
)
def test_tampered_remote_candidate_replay_and_resolution_fail_closed(
    linked: LinkedRuntime, field: str, value: Any
) -> None:
    """Verify tampered remote candidate replay and resolution fail closed."""
    conflict, incoming = _conflict(linked)
    canonical, sync, _dataset, _exchange, _record = linked
    if field == "routing_metadata_json":
        metadata = dict(conflict.authority_candidate.routing_metadata)
        metadata["personal_context_authority"] = {"role": "client_ingress"}
        value = json.dumps(metadata)
    # Field names are a closed test parameter list; production never receives them.
    sync.store.db.execute(
        f"UPDATE sync_envelopes SET {field} = ? WHERE client_envelope_id = ?",
        (value, conflict.expected_remote_envelope_id),
    )
    before = canonical.get_manifest()
    replay = _push(linked, incoming)
    assert not replay.conflicts and replay.rejected[0].retryable
    assert _resolve(linked, conflict)[1] == [0]
    assert canonical.get_manifest() == before
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"
    assert canonical.get_sync_conflict(conflict.conflict_id)["state"] == "unresolved"


@pytest.mark.parametrize("authenticated_body", [False, True])
def test_tampered_local_ciphertext_cannot_release_review(linked: LinkedRuntime, authenticated_body: bool) -> None:
    """Verify tampered local ciphertext cannot release review."""
    conflict, _incoming = _conflict(linked)
    canonical, sync, _dataset, _exchange, _record = linked
    if authenticated_body:
        dataset = sync.store.get_dataset(linked[2])
        source = sync.store.get_envelope_by_client_id(linked[2], conflict.expected_local_envelope_id)
        clear = sync._restore_personal_context_from_storage(dataset, source)
        altered = {**clear.payload, "payload": {**clear.payload["payload"], "value": "tampered-local-body"}}
        protected = sync._protect_personal_context_for_storage(
            dataset, replace(clear, payload=altered, payload_clear=altered)
        )
        sync.store.db.execute(
            "UPDATE sync_envelopes SET payload_ciphertext = ?, encryption_metadata_json = ? WHERE client_envelope_id = ?",
            (
                protected.payload_ciphertext,
                json.dumps(protected.encryption_metadata),
                conflict.expected_local_envelope_id,
            ),
        )
    else:
        sync.store.db.execute(
            "UPDATE sync_envelopes SET payload_ciphertext = ? WHERE client_envelope_id = ?",
            ("invalid-ciphertext", conflict.expected_local_envelope_id),
        )
    before = canonical.get_manifest()
    assert _resolve(linked, conflict)[1] == [0]
    assert canonical.get_manifest() == before
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"


@pytest.mark.parametrize("committed", [False, True])
def test_new_activation_between_batch_check_and_canonical_decision_rejects(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch, committed: bool
) -> None:
    """Verify new activation between batch check and canonical decision rejects."""
    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService

    conflict, _incoming = _conflict(linked)
    canonical, sync, _dataset, _exchange, record = linked
    if committed:
        with monkeypatch.context() as fault:

            def fail_finalize(*args: Any, **kwargs: Any) -> None:
                """Exercise fail finalize."""
                raise RuntimeError("injected Sync finalization failure")

            fault.setattr(type(sync.store), "resolve_conflict", fail_finalize)
            assert _resolve(linked, conflict)[1] == [0]
    before = canonical.get_manifest()
    journal = canonical.get_sync_conflict(conflict.conflict_id)
    original = type(canonical).resolve_sync_conflict

    def transition_then_resolve(owner: Any, **command: Any) -> Any:
        """Exercise transition then resolve."""
        PersonalContextActivationService(owner._repository).prepare(record.profile_id, device_id=_DEVICE_ID, fresh=True)
        return original(owner, **command)

    monkeypatch.setattr(type(canonical), "resolve_sync_conflict", transition_then_resolve)
    assert _resolve(linked, conflict)[1] == [0]
    assert canonical.get_manifest() == before
    assert canonical.get_sync_conflict(conflict.conflict_id) == journal
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"


@pytest.mark.parametrize(
    "boundary,replay,collision",
    [
        ("capture", False, False),
        ("staging", False, True),
        ("capture", True, False),
        ("staging", True, True),
    ],
)
def test_candidate_activation_transition_rejects_and_new_activation_recovers(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch, boundary: str, replay: bool, collision: bool
) -> None:
    """Verify candidate activation transition rejects and new activation recovers."""
    from tldw_Server_API.app.core.Personalization.personal_context_activation import PersonalContextActivationService
    from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
    from tldw_Server_API.app.core.Sync.v2.personal_context_conflicts import PersonalContextConflictService

    canonical, sync, dataset_id, old_exchange, record = linked
    if replay:
        previous, incoming = _conflict(linked, collision=collision)
    else:
        previous = None
        incoming = _envelope(
            linked,
            record.model_copy(
                update={
                    "record_id": "activation-collision-record" if collision else record.record_id,
                    "version_id": "activation-conflict-version",
                    "parent_version_id": None,
                }
            ),
            "activation-conflict-envelope",
        )
    conflict_id = PersonalContextConflictService.conflict_id(dataset_id, _DEVICE_ID, incoming.client_envelope_id)
    before = canonical.get_manifest()
    original_capture = type(canonical).capture_sync_conflict
    original_journal = canonical.get_sync_conflict(conflict_id) if replay else None
    prepared = []

    def transition(owner: Any) -> Any:
        """Exercise transition."""
        prepared.append(
            PersonalContextActivationService(owner._repository).prepare(
                record.profile_id, device_id=_DEVICE_ID, fresh=True
            )
        )

    def capture_across_transition(owner: Any, **identity: Any) -> Any:
        """Exercise capture across transition."""
        if boundary == "capture":
            transition(owner)
        journal = original_capture(owner, **identity)
        if boundary == "staging":
            transition(owner)
        return journal

    with monkeypatch.context() as fault:
        fault.setattr(type(canonical), "capture_sync_conflict", capture_across_transition)
        with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
            _push(linked, incoming)
    assert canonical.get_manifest() == before
    if replay or boundary == "staging":
        journal = canonical.get_sync_conflict(conflict_id)
        if replay:
            assert journal == original_journal
        else:
            assert sync.store.get_conflict(conflict_id) is None
    else:
        with canonical._repository.database.transaction() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type = 'sync_conflict'"
                ).fetchone()[0]
                == 0
            )
    with pytest.raises(SyncStoreError, match="personal_context_activation_required"):
        sync.require_active_exchange(
            dataset=sync.store.get_dataset(dataset_id), user_id=_USER_ID, device_id=_DEVICE_ID, exchange=old_exchange
        )
    newer = PersonalContextExchangeProof.model_validate(_seed_exchange(sync, dataset_id))
    assert canonical._repository.load_activation(prepared[0].activation_id).state == "active"
    if boundary == "staging" and not replay:
        changed = _envelope(
            linked,
            ProfileRecord.model_validate({**incoming.payload, "version_id": "changed-activation-retry"}),
            incoming.client_envelope_id,
        )
        denied = _push((canonical, sync, dataset_id, newer, record), changed)
        assert not denied.conflicts and not denied.accepted
        assert denied.rejected[0].error_code == "idempotency_conflict"
        assert canonical.get_sync_conflict(conflict_id) == journal
    resumed = _push((canonical, sync, dataset_id, newer, record), incoming)
    assert len(resumed.conflicts) == 1, [(item.error_code, item.retryable) for item in resumed.rejected]
    if previous is not None:
        assert asdict(resumed.conflicts[0]) == asdict(previous)
    elif boundary == "staging":
        assert resumed.conflicts[0].expected_remote_envelope_id == journal["remote_envelope_id"]
    assert resumed.personal_context_exchange == newer


def _conflict(linked: LinkedRuntime, *, collision: bool = False) -> tuple[SyncPushConflict, SyncEnvelopeCreate]:
    """Exercise conflict."""
    canonical, sync, dataset_id, exchange, original = linked
    local = ProfileRecord.model_validate(
        {
            **original.model_dump(mode="python"),
            "record_id": "incoming-collision-record" if collision else original.record_id,
            "version_id": "local-candidate-version-13163",
            "parent_version_id": None,
            "payload": {**original.payload.model_dump(), "value": "TASK13163-LOCAL-CANARY"},
        }
    )
    envelope = _envelope(linked, local, "incoming-conflict-envelope-13163")
    result = _push(linked, envelope)
    assert len(result.conflicts) == 1, [(item.apply_status, item.apply_error_code) for item in result.accepted]
    return result.conflicts[0], envelope


def _resolve(
    linked: LinkedRuntime,
    conflict: SyncPushConflict,
    action: str = "skip",
    envelope: SyncEnvelopeCreate | None = None,
    **overrides: Any,
) -> Any:
    """Exercise resolve."""
    _canonical, sync, dataset_id, exchange, _record = linked
    values = {
        "expected_local": conflict.expected_local_envelope_id,
        "expected_remote": conflict.expected_remote_envelope_id,
        "idempotency_key": "resolution-command-key-13163",
        **overrides,
    }
    return sync.resolve_conflicts_batch(
        user_id=_USER_ID,
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        personal_context_exchange=exchange,
        resolutions=[
            (
                conflict.conflict_id,
                action,
                envelope,
                values["expected_local"],
                values["expected_remote"],
                values["idempotency_key"],
            )
        ],
    )


@pytest.mark.parametrize("collision", [False, True])
def test_push_candidate_is_durable_and_exact_on_replay(linked: LinkedRuntime, collision: bool) -> None:
    """Verify push candidate is durable and exact on replay."""
    conflict, envelope = _conflict(linked, collision=collision)
    assert conflict.expected_local_envelope_id == envelope.client_envelope_id
    assert conflict.expected_remote_envelope_id
    assert conflict.authority_candidate.authority.role == "home_authority"
    assert conflict.authority_candidate.apply_status == "applied"
    replay = _push(linked, envelope).conflicts[0]
    assert asdict(replay) == asdict(conflict)
    stored = linked[1].store.get_envelope_by_server_cursor(conflict.authority_candidate.server_cursor)
    assert stored.payload == {} and stored.payload_ciphertext


def test_stale_review_does_not_release_freeze_or_resolve(linked: LinkedRuntime) -> None:
    """Verify stale review does not release freeze or resolve."""
    conflict, _envelope = _conflict(linked)
    canonical, sync, _dataset, _exchange, record = linked
    before = canonical.get_manifest()
    resolved, rejected, _proof = _resolve(linked, conflict, expected_remote="stale-candidate-envelope-13163")
    assert not resolved and rejected == [0]
    assert canonical.get_manifest() == before
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"
    with pytest.raises(ProfileConflictError):
        canonical.update_record(
            record.record_id,
            RecordMutation(payload={**record.payload.model_dump(), "value": "blocked"}),
            expected_version_id=record.version_id,
        )


@pytest.mark.parametrize("collision", [False, True])
def test_explicit_overwrite_replays_after_sync_finalization_failure(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch, collision: bool
) -> None:
    """Verify explicit overwrite replays after sync finalization failure."""
    conflict, _source = _conflict(linked, collision=collision)
    canonical, sync, _dataset, _exchange, record = linked
    replacement = ProfileRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "parent_version_id": record.version_id,
            "version_id": "reviewed-replacement-version-13163",
            "payload": {**record.payload.model_dump(), "value": "reviewed local values"},
        }
    )
    command = _envelope(linked, replacement, "reviewed-replacement-envelope-13163", base=record)
    original = type(sync.store).resolve_conflict

    def fail_finalize(*args: Any, **kwargs: Any) -> None:
        """Exercise fail finalize."""
        raise RuntimeError("injected Sync finalization failure")

    monkeypatch.setattr(type(sync.store), "resolve_conflict", fail_finalize)
    assert _resolve(linked, conflict, "overwrite", command)[1] == [0]
    assert canonical.get_record(record.record_id) == replacement
    manifest = canonical.get_manifest()
    monkeypatch.setattr(type(sync.store), "resolve_conflict", original)
    assert _resolve(linked, conflict, "overwrite", command)[1] == []
    assert canonical.get_manifest() == manifest
    assert len(canonical.list_records()) == 1
    changed = replace(command, client_envelope_id="changed-resolution-envelope-13163")
    assert _resolve(linked, conflict, "overwrite", changed)[1] == [0]


def test_both_candidates_appear_pinned_in_retention_preview_and_apply(linked: LinkedRuntime) -> None:
    """Verify both candidates appear pinned in retention preview and apply."""
    conflict, _source = _conflict(linked, collision=True)
    _canonical, sync, dataset_id, _exchange, _record = linked
    preview = sync.retention_dry_run(user_id=_USER_ID, dataset_id=dataset_id, audit_mode=False)
    pinned = {item.server_sequence for item in preview.candidates if "retention_conflict_pinned" in item.blockers}
    assert {conflict.server_sequence, conflict.authority_candidate.server_cursor} <= pinned
    sync.retention_compact(user_id=_USER_ID, dataset_id=dataset_id, confirm=True)
    assert sync.store.get_envelope_by_server_cursor(conflict.server_sequence).payload_ciphertext
    assert sync.store.get_envelope_by_server_cursor(conflict.authority_candidate.server_cursor).payload_ciphertext


def test_candidate_attachment_failure_is_retryable_and_reuses_canonical_snapshot(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify candidate attachment failure is retryable and reuses canonical snapshot."""
    canonical, sync, _dataset, _exchange, record = linked
    local = ProfileRecord.model_validate(
        {**record.model_dump(mode="python"), "version_id": "retry-local-version-13163"}
    )
    incoming = _envelope(linked, local, "retry-local-envelope-13163")
    original = type(sync.store).insert_conflict

    def fail_attachment(*args: Any, **kwargs: Any) -> None:
        """Exercise fail attachment."""
        raise RuntimeError("injected candidate attachment failure")

    monkeypatch.setattr(type(sync.store), "insert_conflict", fail_attachment)
    first = _push(linked, incoming)
    assert first.rejected[0].retryable and not first.conflicts
    from tldw_Server_API.app.core.Sync.v2.personal_context_conflicts import PersonalContextConflictService

    conflict_id = PersonalContextConflictService.conflict_id(linked[2], _DEVICE_ID, incoming.client_envelope_id)
    before = canonical.get_sync_conflict(conflict_id)
    monkeypatch.setattr(type(sync.store), "insert_conflict", original)
    replay = _push(linked, incoming).conflicts[0]
    assert replay.expected_remote_envelope_id == before["remote_envelope_id"]
    assert replay.authority_candidate.payload == before["candidate"]


@pytest.mark.parametrize("action", ["skip", "duplicate_rename"])
def test_collision_is_resolved_only_by_explicit_reviewed_choice(linked: LinkedRuntime, action: str) -> None:
    """Verify collision is resolved only by explicit reviewed choice."""
    conflict, incoming = _conflict(linked, collision=True)
    canonical, sync, _dataset, _exchange, record = linked
    assert canonical.get_record(record.record_id) == record
    assert len(canonical.list_records()) == 1
    command = None
    if action == "duplicate_rename":
        distinct = ProfileRecord.model_validate(
            {
                **incoming.payload,
                "record_id": "reviewed-distinct-record-13163",
                "version_id": "reviewed-distinct-version-13163",
                "semantic_key": {"namespace": "preference", "subject": "reviewed.distinct"},
            }
        )
        command = _envelope(linked, distinct, "reviewed-distinct-envelope-13163")
    manifest = canonical.get_manifest()
    assert _resolve(linked, conflict, action, command)[1] == []
    assert canonical.get_record(record.record_id) == record
    assert len(canonical.list_records()) == (2 if command else 1)
    assert canonical.get_manifest().revision == manifest.revision + (1 if command else 0)
    assert _push(linked, incoming).rejected[0].error_code == "personal_context_conflict_resolved"


def test_collision_wrong_target_cannot_silently_replace_identity(linked: LinkedRuntime) -> None:
    """Verify collision wrong target cannot silently replace identity."""
    conflict, incoming = _conflict(linked, collision=True)
    assert _resolve(linked, conflict, "overwrite", incoming)[1] == [0]
    assert linked[0].get_record(linked[4].record_id) == linked[4]
    assert linked[1].store.get_conflict(conflict.conflict_id).status == "unresolved"


@pytest.mark.parametrize("collision", [False, True])
def test_duplicate_requires_concrete_active_semantic_key(linked: LinkedRuntime, collision: bool) -> None:
    """Keep-both cannot consume a candidate without creating a distinct keyed fact."""
    from tldw_Server_API.app.core.exceptions import PersonalContextConflictInputError

    conflict, incoming = _conflict(linked, collision=collision)
    canonical, sync, _dataset, _exchange, _record = linked
    before = canonical.get_manifest()
    distinct = ProfileRecord.model_validate(
        {**incoming.payload, "record_id": "missing-key-duplicate", "semantic_key": None}
    )
    command = _envelope(linked, distinct, "missing-key-resolution")
    with pytest.raises(PersonalContextConflictInputError):
        canonical.resolve_sync_conflict(
            conflict_id=conflict.conflict_id,
            dataset_id=linked[2],
            device_id=_DEVICE_ID,
            expected_local_envelope_id=conflict.expected_local_envelope_id,
            expected_remote_envelope_id=conflict.expected_remote_envelope_id,
            idempotency_key="invalid-direct-choice",
            action="duplicate_rename",
            command=asdict(command),
            purge_generation=0,
            exchange=linked[3],
        )
    assert _resolve(linked, conflict, "duplicate_rename", command)[1] == [0]
    assert canonical.get_manifest() == before
    assert len(canonical.list_records()) == 1
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"
    assert canonical.get_sync_conflict(conflict.conflict_id)["state"] == "unresolved"


@pytest.mark.parametrize("action", ["overwrite", "duplicate_rename"])
@pytest.mark.parametrize("relay_fails", [False, True])
def test_resolution_relays_after_canonical_commit(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch, action: str, relay_fails: bool
) -> None:
    """A reviewed mutation relays after both commits and tolerates egress failure."""
    from contextlib import contextmanager

    conflict, incoming = _conflict(linked, collision=True)
    canonical, sync, dataset_id, _exchange, record = linked
    original_guard = sync.store.conflict_resolution_guard
    locked = False

    @contextmanager
    def tracked_guard(*args: Any, **kwargs: Any) -> Any:
        """Exercise tracked guard."""
        nonlocal locked
        with original_guard(*args, **kwargs) as guarded:
            locked = True
            try:
                yield guarded
            finally:
                locked = False

    monkeypatch.setattr(sync.store, "conflict_resolution_guard", tracked_guard)
    payload = {**incoming.payload, "version_id": "immediate-resolution-version"}
    if action == "overwrite":
        payload.update(record_id=record.record_id, parent_version_id=record.version_id)
    else:
        payload.update(
            record_id="immediate-distinct-record", semantic_key={"namespace": "preference", "subject": "distinct"}
        )
    replacement = ProfileRecord.model_validate(payload)
    command = _envelope(
        linked, replacement, "immediate-resolution-envelope", base=record if action == "overwrite" else None
    )
    relay_type = type(sync.personal_context_relay)
    original = relay_type.relay_profile
    relayed = []

    def relay_after_receipt(owner: Any, **kwargs: Any) -> Any:
        """Observe the committed receipt only after releasing the outer Sync fence."""
        assert not locked
        assert canonical.get_sync_conflict(conflict.conflict_id)["state"] == "resolved"
        relayed.append(kwargs["profile_id"])
        if relay_fails:
            raise RuntimeError("injected relay failure")
        return original(owner, **kwargs)

    monkeypatch.setattr(relay_type, "relay_profile", relay_after_receipt)
    assert _resolve(linked, conflict, action, command)[1] == []
    assert relayed == [record.profile_id]
    if not relay_fails:
        head = sync.store.get_current_head(dataset_id, "personal_context.record", replacement.record_id)
        assert head.entity_version == replacement.version_id
    assert canonical.get_record(replacement.record_id) == replacement


def test_direct_resolution_replay_schedules_committed_publication(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct service callers schedule replayed publication debt after receipt commit."""
    conflict, _incoming = _conflict(linked)
    canonical, _sync, dataset_id, exchange, record = linked
    replacement = record.model_copy(
        update={"version_id": "direct-reviewed-version", "parent_version_id": record.version_id}
    )
    command = _envelope(linked, replacement, "direct-reviewed-envelope", base=record)
    relayed = []

    def committed(profile_id: str) -> None:
        """Observe the durable receipt when direct service relay is requested."""
        assert canonical.get_sync_conflict(conflict.conflict_id)["state"] == "resolved"
        relayed.append(profile_id)

    canonical.set_after_commit_relay(committed)
    values = {
        "conflict_id": conflict.conflict_id,
        "dataset_id": dataset_id,
        "device_id": _DEVICE_ID,
        "expected_local_envelope_id": conflict.expected_local_envelope_id,
        "expected_remote_envelope_id": conflict.expected_remote_envelope_id,
        "idempotency_key": "direct-reviewed-choice",
        "action": "overwrite",
        "command": asdict(command),
        "purge_generation": 0,
        "exchange": exchange,
    }
    first = canonical.resolve_sync_conflict(**values)
    assert canonical.resolve_sync_conflict(**values) == first
    assert relayed == [record.profile_id, record.profile_id]


def test_collision_freezes_both_ids_but_unrelated_ingress_continues(linked: LinkedRuntime) -> None:
    """Verify collision freezes both ids but unrelated ingress continues."""
    conflict, incoming = _conflict(linked, collision=True)
    canonical, sync, _dataset, _exchange, record = linked
    escaped = ProfileRecord.model_validate(
        {**incoming.payload, "semantic_key": {"namespace": "preference", "subject": "uncontested"}}
    )
    with pytest.raises(ProfileConflictError):
        canonical.create_record(escaped)
    with pytest.raises(ProfileConflictError):
        canonical.archive_record(record.record_id, expected_version_id=record.version_id)
    unrelated = ProfileRecord.model_validate(
        {**escaped.model_dump(mode="python"), "record_id": "unrelated-record-13163"}
    )
    result = _push(linked, _envelope(linked, unrelated, "unrelated-envelope-13163"))
    assert len(result.accepted) == 1 and result.accepted[0].apply_status == "applied"
    assert canonical.get_record(unrelated.record_id) == unrelated


@pytest.mark.parametrize("resolved", [False, True])
def test_exact_journal_survives_reopen_rotation_and_contains_no_plaintext(
    linked: LinkedRuntime, resolved: bool
) -> None:
    """Verify exact journal survives reopen rotation and contains no plaintext."""
    conflict, _source = _conflict(linked, collision=True)
    if resolved:
        assert _resolve(linked, conflict)[1] == []
    canonical, sync, _dataset, _exchange, record = linked
    from tldw_Server_API.app.core.DB_Management.Personal_Context_Repository import PersonalContextRepository
    from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB

    path = canonical._repository.database.db_path
    reopened = PersonalContextRepository(PersonalizationDB.for_path(path))
    before = reopened.get_sync_conflict(record.profile_id, conflict.conflict_id)
    reopened.rotate_encryption_key(record.profile_id)
    assert reopened.get_sync_conflict(record.profile_id, conflict.conflict_id) == before
    with reopened.database.transaction() as connection:
        rows = connection.execute(
            "SELECT * FROM personal_context_object_versions WHERE object_type IN ('sync_conflict', 'sync_conflict_receipt')"
        ).fetchall()
        assert rows and all(row["key_version"] == 2 for row in rows)
        assert any(row["object_type"] == "sync_conflict_receipt" for row in rows) == resolved
    from pathlib import Path

    assert b"TASK13163-" not in Path(path).read_bytes()
    if not resolved:
        assert asdict(_push(linked, _source).conflicts[0]) == asdict(conflict)
    assert _resolve(linked, conflict)[1] == []


def test_batch_partial_failure_preserves_stale_item_and_commits_valid_skip(linked: LinkedRuntime) -> None:
    """Verify batch partial failure preserves stale item and commits valid skip."""
    conflict, _source = _conflict(linked)
    _canonical, sync, dataset_id, exchange, _record = linked
    valid = (
        conflict.conflict_id,
        "skip",
        None,
        conflict.expected_local_envelope_id,
        conflict.expected_remote_envelope_id,
        "valid-skip-command-13163",
    )
    stale = (*valid[:4], "stale-remote-candidate-13163", valid[5])
    resolved, rejected, _proof = sync.resolve_conflicts_batch(
        user_id=_USER_ID,
        dataset_id=dataset_id,
        device_id=_DEVICE_ID,
        resolutions=[stale, valid],
        personal_context_exchange=exchange,
    )
    assert rejected == [0] and [index for index, _ in resolved] == [1]


@pytest.mark.parametrize("defect", ["owner", "device", "proof"])
def test_wrong_authority_cannot_resolve_or_release(linked: LinkedRuntime, defect: str) -> None:
    """Verify wrong authority cannot resolve or release."""
    conflict, _source = _conflict(linked)
    _canonical, sync, dataset_id, exchange, _record = linked
    from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError

    with pytest.raises(SyncStoreError):
        sync.resolve_conflicts_batch(
            user_id="wrong-owner" if defect == "owner" else _USER_ID,
            dataset_id=dataset_id,
            device_id="wrong-device" if defect == "device" else _DEVICE_ID,
            resolutions=[
                (
                    conflict.conflict_id,
                    "skip",
                    None,
                    conflict.expected_local_envelope_id,
                    conflict.expected_remote_envelope_id,
                    "unauthorized-command-13163",
                )
            ],
            personal_context_exchange=None if defect == "proof" else exchange,
        )
    assert sync.store.get_conflict(conflict.conflict_id).status == "unresolved"


def test_capacity_exhaustion_preserves_existing_candidate_and_retries_new_push(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify capacity exhaustion preserves existing candidate and retries new push."""
    from tldw_Server_API.app.core.DB_Management import Personal_Context_Repository as repository_module

    conflict, _source = _conflict(linked)
    monkeypatch.setattr(repository_module, "_MAX_CONFLICT_HEADS", 1)
    record = linked[4]
    local = ProfileRecord.model_validate(
        {**record.model_dump(mode="python"), "version_id": "second-conflicted-version-13163"}
    )
    result = _push(linked, _envelope(linked, local, "second-conflicted-envelope-13163"))
    assert not result.conflicts and result.rejected[0].retryable
    assert linked[1].store.get_envelope_by_server_cursor(conflict.authority_candidate.server_cursor).payload_ciphertext


def test_resolved_receipt_retains_exact_replay_without_occupying_active_capacity(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify resolved receipt retains exact replay without occupying active capacity."""
    from tldw_Server_API.app.core.DB_Management import Personal_Context_Repository as repository_module

    conflict, _source = _conflict(linked)
    monkeypatch.setattr(repository_module, "_MAX_CONFLICT_HEADS", 1)
    assert _resolve(linked, conflict)[1] == []
    record = linked[4]
    local = ProfileRecord.model_validate({**record.model_dump(mode="python"), "version_id": "new-review-version-13163"})
    result = _push(linked, _envelope(linked, local, "new-review-envelope-13163"))
    assert len(result.conflicts) == 1
    assert _resolve(linked, conflict)[1] == []
    with linked[0]._repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type = 'sync_conflict'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type = 'sync_conflict_receipt'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("resolved", [False, True])
def test_purge_removes_encrypted_conflict_owner_and_cannot_recreate_receipt(
    linked: LinkedRuntime, resolved: bool
) -> None:
    """Verify purge removes encrypted conflict owner and cannot recreate receipt."""
    conflict, _source = _conflict(linked, collision=True)
    if resolved:
        assert _resolve(linked, conflict)[1] == []
    canonical, sync, _dataset, _exchange, record = linked
    canonical.purge_profile(mode="everywhere", confirmation="DELETE EVERYWHERE", expected_purge_generation=0)
    with canonical._repository.database.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_versions WHERE object_type IN ('sync_conflict', 'sync_conflict_receipt')"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_object_heads WHERE object_type IN ('sync_conflict', 'sync_conflict_receipt')"
            ).fetchone()[0]
            == 0
        )
    from tldw_Server_API.app.core.Personalization.personal_context_repository_models import ConcurrentProfileUpdateError

    with pytest.raises(ConcurrentProfileUpdateError):
        canonical.get_sync_conflict(conflict.conflict_id)
    assert len(canonical.list_records()) == 0


def test_http_push_serializes_protected_candidate(linked: LinkedRuntime) -> None:
    """Verify http push serializes protected candidate."""
    from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_certification import _production_client

    canonical, sync, dataset_id, exchange, record = linked
    local = ProfileRecord.model_validate({**record.model_dump(mode="python"), "version_id": "http-local-version-13163"})
    envelope = _envelope(linked, local, "http-local-envelope-13163")
    with _production_client(sync._certification_production_app) as client:
        response = client.post(
            "/api/v1/sync/push",
            json={
                "dataset_id": dataset_id,
                "device_id": _DEVICE_ID,
                "personal_context_exchange": exchange.model_dump(mode="json"),
                "envelopes": [asdict(envelope)],
            },
        )
    assert response.status_code == 200
    body = response.json()["conflicts"][0]
    assert body["authority_candidate"]["authority"]["role"] == "home_authority"
    assert body["expected_remote_envelope_id"] == body["authority_candidate"]["client_envelope_id"]


@pytest.mark.parametrize("action", ["skip", "overwrite", "duplicate_rename"])
def test_postgres_candidate_attachment_replay_and_retention(
    production_factories: tuple[PersonalContextService, SyncV2Service],
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    """Verify postgres candidate attachment replay and retention."""
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
    from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

    canonical, sync = production_factories
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    sync.store = SyncV2Store(SyncDatabase(backend=backend))
    relay_type = type(sync.personal_context_relay)
    original_relay = relay_type.relay_profile

    def deterministic_relay(relay: Any, **kwargs: Any) -> Any:
        """Exercise deterministic relay."""
        relay.clock_ns = lambda: 0
        return original_relay(relay, **kwargs)

    monkeypatch.setattr(relay_type, "relay_profile", deterministic_relay)
    try:
        runtime = _link(canonical, sync)
        conflict, incoming = _conflict(runtime, collision=True)
        replay = _push(runtime, incoming)
        assert replay.conflicts, [(item.error_code, item.retryable) for item in replay.rejected]
        assert replay.conflicts[0].expected_remote_envelope_id == conflict.expected_remote_envelope_id
        assert sync.store.envelope_is_conflict_pinned(runtime[2], conflict.expected_remote_envelope_id)
        from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError

        with pytest.raises(SyncStoreError, match="retention_conflict_pinned"):
            with sync.store.retention_domain_guard(
                runtime[2], "personal_context.record", [runtime[4].record_id]
            ) as guarded:
                guarded.record_domain_compaction(
                    runtime[2],
                    "personal_context.record",
                    through_server_sequence=conflict.authority_candidate.server_cursor,
                    state={},
                )
        record = runtime[4]
        command = None
        if action != "skip":
            replacement = ProfileRecord.model_validate(
                {
                    **record.model_dump(mode="python"),
                    "version_id": "postgres-reviewed-version",
                    "record_id": record.record_id if action == "overwrite" else "postgres-distinct-record",
                    "parent_version_id": record.version_id if action == "overwrite" else None,
                    "semantic_key": record.semantic_key
                    if action == "overwrite"
                    else {"namespace": "preference", "subject": "postgres.distinct"},
                }
            )
            command = _envelope(
                runtime, replacement, "postgres-reviewed-envelope", base=record if action == "overwrite" else None
            )
        assert _resolve(runtime, conflict, action, command)[1] == []
        assert not sync.store.envelope_is_conflict_pinned(runtime[2], conflict.expected_remote_envelope_id)
        if command is not None:
            assert (
                sync.store.get_current_head(runtime[2], "personal_context.record", command.object_id).entity_version
                == replacement.version_id
            )
    finally:
        backend.get_pool().close_all()


def test_guarded_authority_delete_respects_unresolved_references(activation_store: SyncV2Store) -> None:
    """Exercise the actual destructive query on SQLite and PostgreSQL."""
    from tldw_Server_API.app.core.Sync.v2.models import SyncConflictCreate

    metadata = {
        "role": "home_authority",
        "publication_batch_id": "protected-publication-13163",
        "profile_publication_sequence": 1,
        "batch_ordinal": 0,
        "batch_size": 1,
    }
    candidate = activation_store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="activation-dataset",
            device_id="server-origin",
            client_envelope_id="protected-candidate-13163",
            domain="personal_context.record",
            operation="upsert",
            object_id="protected-object-13163",
            payload_ciphertext="opaque-encrypted-storage-fixture",
            payload_hash="sha256:" + "a" * 64,
            routing_metadata={
                "profile_id": "profile-0123456789",
                "purge_generation": 0,
                "personal_context_authority": metadata,
            },
        )
    )
    activation_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="protected-conflict-13163",
            dataset_id=candidate.dataset_id,
            domain=candidate.domain,
            object_id=candidate.object_id,
            conflict_type="fixture_conflict",
            local_envelope_id=candidate.client_envelope_id,
            remote_envelope_id=candidate.client_envelope_id,
            server_cursor=candidate.server_cursor,
        )
    )
    values = dict(
        server_cursor=candidate.server_cursor,
        dataset_id=candidate.dataset_id,
        client_envelope_id=candidate.client_envelope_id,
        profile_id="profile-0123456789",
        purge_generation=0,
        **{key: value for key, value in metadata.items() if key != "role"},
    )
    assert activation_store.db.discard_pending_personal_context_authority(**values) == "mismatch"
    assert activation_store.get_envelope_by_server_cursor(candidate.server_cursor) is not None
    activation_store.resolve_conflict("protected-conflict-13163")
    assert activation_store.db.discard_pending_personal_context_authority(**values) == "removed"


def test_informational_conflict_pins_allow_unrelated_domain_compaction(activation_store: SyncV2Store) -> None:
    """Synthetic review rows retain ciphertext without blocking another domain."""
    from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
    from tldw_Server_API.app.core.Sync.v2.models import SyncConflictCreate
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service

    store = activation_store
    first = store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="activation-dataset",
            device_id="server-origin",
            client_envelope_id="scope-v1",
            domain="personal_context.scope",
            operation="upsert",
            object_id="unrelated-scope",
            entity_version="scope-version-1",
            payload_ciphertext="opaque-scope-1",
            payload_hash="sha256:scope-1",
            apply_status="applied",
            object_revision=1,
        )
    )
    store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="activation-dataset",
            device_id="server-origin",
            client_envelope_id="scope-v2",
            domain="personal_context.scope",
            operation="upsert",
            object_id="unrelated-scope",
            entity_version="scope-version-2",
            payload_ciphertext="opaque-scope-2",
            payload_hash="sha256:scope-2",
            apply_status="applied",
            object_revision=2,
            base_server_cursor=first.server_cursor,
            base_object_revision=first.object_revision,
            base_object_hash=first.payload_hash,
        )
    )
    candidates = [
        store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id="activation-dataset",
                device_id="server-origin",
                client_envelope_id=identity,
                domain="personal_context.record",
                operation="upsert",
                object_id="contested-record",
                status="conflict",
                payload_ciphertext="opaque-review-candidate",
                payload_hash="sha256:review",
                apply_status="applied",
            )
        )
        for identity in ("local-candidate", "remote-candidate")
    ]
    store.insert_conflict(
        SyncConflictCreate(
            conflict_id="unresolved-review",
            dataset_id="activation-dataset",
            domain="personal_context.record",
            object_id="contested-record",
            conflict_type="personal_context_base_conflict",
            local_envelope_id=candidates[0].client_envelope_id,
            remote_envelope_id=candidates[1].client_envelope_id,
            server_cursor=candidates[0].server_cursor,
        )
    )
    service = SyncV2Service(store=store, adapters=SyncAdapterRegistry([]))
    result = service.retention_compact(user_id="activation-user", dataset_id="activation-dataset", confirm=True)
    assert result.mutation_performed
    assert store.get_domain_compaction_sequence("activation-dataset", "personal_context.scope") == first.server_cursor
    assert all(store.get_envelope_by_server_cursor(row.server_cursor).payload_ciphertext for row in candidates)


def test_candidate_staging_requires_surviving_canonical_authority(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify candidate staging requires surviving canonical authority."""
    canonical, sync, _dataset, _exchange, record = linked
    original = type(canonical).capture_sync_conflict

    def revoke_after_capture(owner: Any, **identity: Any) -> Any:
        """Exercise revoke after capture."""
        journal = original(owner, **identity)
        with owner._repository.database.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM personal_context_object_heads WHERE object_type = 'sync_conflict' AND object_id = ?",
                (journal["conflict_id"],),
            )
        return journal

    monkeypatch.setattr(type(canonical), "capture_sync_conflict", revoke_after_capture)
    incoming = _envelope(
        linked,
        ProfileRecord.model_validate(
            {**record.model_dump(mode="python"), "version_id": "revoked-candidate-version-13163"}
        ),
        "revoked-candidate-envelope-13163",
    )
    result = _push(linked, incoming)
    assert not result.conflicts and result.rejected[0].retryable


def test_candidate_replay_acquires_sync_fence_before_canonical_lock(
    linked: LinkedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify candidate replay acquires sync fence before canonical lock."""
    from contextlib import contextmanager

    conflict, incoming = _conflict(linked)
    canonical, sync, dataset_id, _exchange, _record = linked
    original_guard = sync.store.conflict_resolution_guard
    original_capture = type(canonical).capture_sync_conflict
    locked = False

    @contextmanager
    def tracked_guard(*args: Any, **kwargs: Any) -> Any:
        """Exercise tracked guard."""
        nonlocal locked
        with original_guard(*args, **kwargs) as guarded:
            locked = True
            try:
                yield guarded
            finally:
                locked = False

    def capture_under_sync_fence(owner: Any, **identity: Any) -> Any:
        """Exercise capture under sync fence."""
        assert locked, "canonical capture must follow the Sync fence"
        return original_capture(owner, **identity)

    monkeypatch.setattr(sync.store, "conflict_resolution_guard", tracked_guard)
    monkeypatch.setattr(type(canonical), "capture_sync_conflict", capture_under_sync_fence)
    source = sync.store.get_envelope_by_client_id(dataset_id, incoming.client_envelope_id)
    replay = sync._ensure_personal_context_conflict_candidate(
        sync.store.get_dataset(dataset_id), source, sync.store, exchange=_exchange
    )
    assert replay.expected_remote_envelope_id == conflict.expected_remote_envelope_id
